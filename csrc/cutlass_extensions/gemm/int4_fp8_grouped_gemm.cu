#include <vector>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

using namespace cute;

// Type definitions
using MmaType = cutlass::float_e4m3_t;      // FP8 e4m3 type
using QuantType = cutlass::int4b_t;         // 4-bit integer type
using ElementAccumulator = float;           // Accumulator type
using ElementScale = float;                 // Scale type
using ElementC = cutlass::half_t;           // Default output type (FP16)
using ElementD = ElementC;                  // Default output type (FP16)
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// Architecture-specific configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;
using TileShape = Shape<_128, _16, cute::Int<TileShapeK>>;
using ClusterShape = Shape<_1, _1, _1>;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;
using LayoutScale = cutlass::layout::RowMajor;

// Alignments
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Element packing for scales
using ElementScalePacked = cutlass::Array<ElementScale, 1>;

// Kernel schedule and epilogue definitions
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<QuantType, ElementScalePacked>, LayoutB_Transpose *, AlignmentB,
    MmaType, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// Define the final kernel and GEMM operation types
using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

// Stride definitions
using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;

struct Int4Fp8GemmParams {
    // Problem size parameters
    int num_groups;
    typename ProblemShape::UnderlyingProblemShape* problem_sizes; // Sizes of GEMM problems

    // Tensors
    const MmaType **a_ptrs;        // Array of pointers to A matrices
    const QuantType **b_ptrs;      // Array of pointers to B matrices
    const ElementScalePacked **scale_ptrs; // Array of pointers to scale factors
    const ElementC **c_ptrs;       // Array of pointers to C matrices (input)
    ElementD **d_ptrs;             // Array of pointers to D matrices (output)

    // Strides for each tensor
    StrideA *stride_a;        // Strides for A matrices
    StrideB *stride_b;        // Strides for B matrices
    StrideC *stride_c;        // Strides for C matrices
    StrideD *stride_d;        // Strides for D matrices
    const StrideS *stride_s;  // Strides for scales

    // Alpha and beta scaling factors
    ElementAccumulator *alpha;     // Array of alpha values for each problem
    ElementAccumulator *beta;      // Array of beta values for each problem

    // Scale chunk size
    int chunk_size;                // Size of each chunk for scales (typically K/chunks)

    // Workspace memory
    void *workspace;               // Workspace memory
    size_t workspace_size;         // Size of the workspace
};

cudaError_t runInt4Fp8GroupedGemm(Int4Fp8GemmParams& params) {
    // Prepare device pointers for alpha and beta if they're not nullptr
    ElementAccumulator** d_alpha_ptr_array = nullptr;
    ElementAccumulator** d_beta_ptr_array = nullptr;

    if (params.alpha && params.beta) {
        cudaMalloc(&d_alpha_ptr_array, params.num_groups * sizeof(ElementAccumulator*));
        cudaMalloc(&d_beta_ptr_array, params.num_groups * sizeof(ElementAccumulator*));

        // Copy alpha and beta arrays to device
        cudaMemcpy(d_alpha_ptr_array, params.alpha, params.num_groups * sizeof(ElementAccumulator*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_ptr_array, params.beta, params.num_groups * sizeof(ElementAccumulator*), cudaMemcpyHostToDevice);
    }

    // Set up GemmUniversalMode and fusion arguments
    using Args = typename GemmScaleOnly::Arguments;
    Args arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    if (params.alpha && params.beta) {
        // Use per-group alpha/beta values
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;
        fusion_args.alpha_ptr_array = d_alpha_ptr_array;
        fusion_args.beta_ptr_array = d_beta_ptr_array;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
        fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    } else {
        // Use default alpha=1, beta=0
        fusion_args.alpha = 1.0f;
        fusion_args.beta = 0.0f;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;
        fusion_args.alpha_ptr_array = nullptr;
        fusion_args.beta_ptr_array = nullptr;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
        fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }

    // Create hardware info for the current device
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Create gemm arguments
    arguments = Args {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {params.num_groups, params.problem_sizes, nullptr},
        {params.b_ptrs, params.stride_b, params.a_ptrs, params.stride_a, params.scale_ptrs, params.stride_s, params.chunk_size},
        {fusion_args, params.c_ptrs, params.stride_c, params.d_ptrs, params.stride_d},
        hw_info
    };

    // Instantiate GEMM
    GemmScaleOnly gemm;

    // Get workspace size
    params.workspace_size = GemmScaleOnly::get_workspace_size(arguments);

    // Allocate workspace if not provided
    void* workspace_ptr = params.workspace;
    bool allocated_workspace = false;

    if (!workspace_ptr && params.workspace_size > 0) {
        cudaMalloc(&workspace_ptr, params.workspace_size);
        allocated_workspace = true;
    }

    // Check if the problem is supported
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (allocated_workspace) {
            cudaFree(workspace_ptr);
        }
        if (d_alpha_ptr_array) cudaFree(d_alpha_ptr_array);
        if (d_beta_ptr_array) cudaFree(d_beta_ptr_array);
        return cudaErrorInvalidValue;
    }

    // Initialize the GEMM with arguments and workspace
    status = gemm.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        if (allocated_workspace) {
            cudaFree(workspace_ptr);
        }
        if (d_alpha_ptr_array) cudaFree(d_alpha_ptr_array);
        if (d_beta_ptr_array) cudaFree(d_beta_ptr_array);
        return cudaErrorInvalidValue;
    }

    // Run the GEMM
    status = gemm.run();

    // Free allocated resources
    if (allocated_workspace) {
        cudaFree(workspace_ptr);
    }
    if (d_alpha_ptr_array) cudaFree(d_alpha_ptr_array);
    if (d_beta_ptr_array) cudaFree(d_beta_ptr_array);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Return status
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <typename StrideType>
std::vector<StrideType> createStrides(const std::vector<torch::Tensor>& tensors, bool is_transposed = false) {
    std::vector<StrideType> strides;
    strides.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        int64_t M = tensor.size(0);
        int64_t N = tensor.size(1);
        if (is_transposed) {
            // For transposed layout
            strides.push_back(cutlass::make_cute_packed_stride(StrideType{}, {static_cast<int>(N), static_cast<int>(M), 1}));
        } else {
            // For standard layout
            strides.push_back(cutlass::make_cute_packed_stride(StrideType{}, {static_cast<int>(M), static_cast<int>(N), 1}));
        }
    }

    return strides;
}

template <typename T>
std::vector<const T*> getDevicePtrs(const std::vector<torch::Tensor>& tensors) {
    std::vector<const T*> ptrs;
    ptrs.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        ptrs.push_back(reinterpret_cast<const T*>(tensor.data_ptr()));
    }

    return ptrs;
}

template <typename T>
std::vector<T*> getMutableDevicePtrs(const std::vector<torch::Tensor>& tensors) {
    std::vector<T*> ptrs;
    ptrs.reserve(tensors.size());

    for (auto& tensor : tensors) {
        // Remove const for output tensors
        ptrs.push_back(reinterpret_cast<T*>(tensor.data_ptr()));
    }

    return ptrs;
}

std::vector<typename ProblemShape::UnderlyingProblemShape> createProblemSizes(
    const std::vector<int>& m_values,
    const std::vector<int>& n_values,
    const std::vector<int>& k_values)
{
    int num_groups = m_values.size();
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes(num_groups);

    for (int i = 0; i < num_groups; i++) {
        // Note: We swap M and N because of the transpose
        problem_sizes[i] = make_tuple(n_values[i], m_values[i], k_values[i]);
    }

    return problem_sizes;
}

// Helper function to preprocess int4 tensors
std::vector<torch::Tensor> preprocessInt4Tensors(const std::vector<torch::Tensor>& raw_tensors) {
    std::vector<torch::Tensor> processed_tensors;
    processed_tensors.reserve(raw_tensors.size());

    for (const auto& tensor : raw_tensors) {
        // Create a new tensor with the same size to hold the processed values
        auto processed = torch::empty_like(tensor);

        // Call the CUTLASS encoding function
        cutlass::unified_encode_int4b(
            reinterpret_cast<const cutlass::int4b_t*>(tensor.data_ptr()),
            reinterpret_cast<cutlass::int4b_t*>(processed.data_ptr()),
            tensor.numel()
        );

        processed_tensors.push_back(processed);
    }

    return processed_tensors;
}

// Helper function to preprocess scale tensors
std::vector<torch::Tensor> preprocessScaleTensors(const std::vector<torch::Tensor>& scale_tensors) {
    std::vector<torch::Tensor> packed_tensors;
    packed_tensors.reserve(scale_tensors.size());

    for (const auto& tensor : scale_tensors) {
        // Create a tensor for packed scales
        auto packed = torch::empty_like(tensor);

        // Pack the scales using our template function
        cutlass::pack_scale_fp32<ElementScale, ElementScalePacked>(
            reinterpret_cast<const float*>(tensor.data_ptr()),
            reinterpret_cast<ElementScalePacked*>(packed.data_ptr()),
            tensor.numel(),
            ElementScalePacked::kElements
        );

        packed_tensors.push_back(packed);
    }

    return packed_tensors;
}

/**
 * @brief Main function to run int4 * fp8 grouped GEMM from PyTorch
 *
 * This function performs multiple GEMM operations in parallel where each operation multiplies
 * an FP8 matrix (A) with a quantized INT4 matrix (B), applying per-channel scaling factors.
 * It's designed for efficient execution on NVIDIA Hopper GPUs, leveraging Tensor Cores for
 * optimal performance with mixed precision arithmetic.
 *
 * Rather than taking vectors of tensors, this function uses expert_offsets to manage
 * multiple GEMM operations within the unified tensors.
 *
 * @param a_tensor Tensor containing all A matrices (fp8_e4m3) with shape [total_m, K]
 * @param b_tensor Tensor containing all B matrices (int4 packed as int8) with shape [total_n, K/2]
 * @param scale_tensor Tensor containing all scale factors with shape [total_n, K/chunk_size]
 * @param c_tensor Tensor containing all C matrices (input) with shape [total_m, total_n]
 * @param expert_offsets Tensor containing expert offsets for determining group boundaries
 * @param a_strides Optional custom strides for A matrices (nullptr for default)
 * @param b_strides Optional custom strides for B matrices (nullptr for default)
 * @param c_strides Optional custom strides for C matrices (nullptr for default)
 * @param problem_sizes Optional problem sizes (nullptr for auto-detection)
 * @param chunk_size Size of each chunk for scales (K / number of scale chunks); if 0, will be auto-detected
 * @param alpha Optional scalar multiplier for the product of A and B matrices
 * @param beta Optional scalar multiplier for matrix C
 * @return torch::Tensor Output tensor D with shape [total_m, total_n]
 */
torch::Tensor int4Fp8GroupedGemm(
    const torch::Tensor& a_tensor,
    const torch::Tensor& b_tensor,
    const torch::Tensor& scale_tensor,
    const torch::Tensor& c_tensor,
    const torch::Tensor& expert_offsets,
    const StrideA* a_strides = nullptr,
    const StrideB* b_strides = nullptr,
    const StrideC* c_strides = nullptr,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes = nullptr,
    int chunk_size = 0,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    // Check inputs
    TORCH_CHECK(a_tensor.dim() == 2, "A tensor must be 2D");
    TORCH_CHECK(b_tensor.dim() == 2, "B tensor must be 2D");
    TORCH_CHECK(scale_tensor.dim() == 2, "Scale tensor must be 2D");
    TORCH_CHECK(c_tensor.dim() == 2, "C tensor must be 2D");
    TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");

    // Get number of groups from expert_offsets
    int num_groups = static_cast<int>(expert_offsets.size(0));

    // Check tensor types
    TORCH_CHECK(a_tensor.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
    TORCH_CHECK(b_tensor.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");

    // Set CUDA device based on the input tensor
    const at::cuda::CUDAGuard device_guard(a_tensor.device());

    // Create output tensor
    auto d_tensor = torch::empty_like(c_tensor);

    // Split tensors into groups based on expert_offsets
    std::vector<torch::Tensor> a_tensors, b_tensors, scale_tensors, c_tensors, d_tensors;

    auto offsets = expert_offsets.cpu().data_ptr<int64_t>();
    int64_t K = a_tensor.size(1);

    for (int i = 0; i < num_groups; i++) {
        // Extract the appropriate slices for each group
        int64_t M_offset = offsets[i];
        int64_t M = (i < num_groups - 1) ? (offsets[i+1] - offsets[i]) : (a_tensor.size(0) - offsets[i]);

        a_tensors.push_back(a_tensor.slice(0, M_offset, M_offset + M));
        b_tensors.push_back(b_tensor);  // Each group uses the whole B tensor
        scale_tensors.push_back(scale_tensor);  // Each group uses the whole scale tensor
        c_tensors.push_back(c_tensor.slice(0, M_offset, M_offset + M));
        d_tensors.push_back(d_tensor.slice(0, M_offset, M_offset + M));
    }

    // Preprocess int4 tensors
    std::vector<torch::Tensor> processed_b_tensors = preprocessInt4Tensors(b_tensors);

    // Preprocess scale tensors
    std::vector<torch::Tensor> packed_scale_tensors = preprocessScaleTensors(scale_tensors);

    // Track if we allocate problem_sizes ourselves
    bool allocated_problem_sizes = false;
    typename ProblemShape::UnderlyingProblemShape* device_problem_sizes = problem_sizes;

    // Collect problem dimensions if not provided
    if (!problem_sizes) {
        std::vector<int> m_values(num_groups), n_values(num_groups), k_values(num_groups);
        std::vector<typename ProblemShape::UnderlyingProblemShape> host_problem_sizes;

        for (int i = 0; i < num_groups; ++i) {
            const auto& a = a_tensors[i];
            const auto& b = b_tensors[i];

            int64_t M = a.size(0);
            int64_t K = a.size(1);
            int64_t N = b.size(0);

            // Verify that K is consistent with packed int4 format in B
            TORCH_CHECK(b.size(1) * 2 == K,
                "B tensor has inconsistent dimensions for int4 format. "
                "B.size(1) should be A.size(1)/2 because each byte stores two int4 values.");

            m_values[i] = static_cast<int>(M);
            n_values[i] = static_cast<int>(N);
            k_values[i] = static_cast<int>(K);
        }

        // Create problem sizes
        host_problem_sizes = createProblemSizes(m_values, n_values, k_values);

        // Allocate device memory for problem sizes and copy
        cudaMalloc(&device_problem_sizes, num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape));
        cudaMemcpy(
            device_problem_sizes,
            host_problem_sizes.data(),
            num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape),
            cudaMemcpyHostToDevice
        );
        allocated_problem_sizes = true;
    } else {
        // We're using provided problem_sizes
        device_problem_sizes = problem_sizes;
    }

    // Create strides for tensors if not provided
    std::vector<StrideA> local_stride_a;
    std::vector<StrideB> local_stride_b;
    std::vector<StrideC> local_stride_c;
    std::vector<StrideD> local_stride_d;
    std::vector<StrideS> local_stride_s;

    StrideA* stride_a_ptr = (StrideA*) a_strides;
    StrideB* stride_b_ptr = (StrideB*) b_strides;
    StrideC* stride_c_ptr = (StrideC*) c_strides;

    if (!a_strides) {
        local_stride_a = createStrides<StrideA>(a_tensors);
        stride_a_ptr = local_stride_a.data();
    }

    if (!b_strides) {
        local_stride_b = createStrides<StrideB>(processed_b_tensors);
        stride_b_ptr = local_stride_b.data();
    }

    if (!c_strides) {
        local_stride_c = createStrides<StrideC>(c_tensors);
        stride_c_ptr = local_stride_c.data();
    }

    // Always create D strides
    local_stride_d = createStrides<StrideD>(d_tensors);

    // Create scale strides
    local_stride_s = createStrides<StrideS>(packed_scale_tensors);

    // Determine chunk_size if not provided
    if (chunk_size <= 0) {
        // Default to K / number of scales in the scale tensor
        int64_t K = a_tensor.size(1);
        int64_t scale_cols = scale_tensor.size(1);
        chunk_size = static_cast<int>(K / scale_cols);
        TORCH_CHECK(chunk_size > 0, "Cannot determine chunk size. Please provide it explicitly.");
    }

    // Get device pointers
    auto a_ptrs = getDevicePtrs<MmaType>(a_tensors);
    auto b_ptrs = getDevicePtrs<QuantType>(processed_b_tensors);
    auto scale_ptrs = getDevicePtrs<ElementScalePacked>(packed_scale_tensors);
    auto c_ptrs = getDevicePtrs<ElementC>(c_tensors);
    auto d_ptrs = getMutableDevicePtrs<ElementD>(d_tensors);

    // Create parameters structure
    Int4Fp8GemmParams params;
    params.num_groups = num_groups;
    params.problem_sizes = device_problem_sizes;
    params.a_ptrs = a_ptrs.data();
    params.b_ptrs = b_ptrs.data();
    params.scale_ptrs = scale_ptrs.data();
    params.c_ptrs = c_ptrs.data();
    params.d_ptrs = d_ptrs.data();
    params.stride_a = stride_a_ptr;
    params.stride_b = stride_b_ptr;
    params.stride_c = stride_c_ptr;
    params.stride_d = local_stride_d.data();
    params.stride_s = local_stride_s.data();

    // Allocate and set alpha and beta values
    ElementAccumulator* alpha_ptr = nullptr;
    ElementAccumulator* beta_ptr = nullptr;
    cudaMalloc(&alpha_ptr, sizeof(ElementAccumulator));
    cudaMalloc(&beta_ptr, sizeof(ElementAccumulator));
    cudaMemcpy(alpha_ptr, &alpha, sizeof(ElementAccumulator), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_ptr, &beta, sizeof(ElementAccumulator), cudaMemcpyHostToDevice);

    params.alpha = alpha_ptr;
    params.beta = beta_ptr;
    params.chunk_size = chunk_size;
    params.workspace = nullptr;  // Allocated inside runInt4Fp8GroupedGemm

    // Run the GEMM
    cudaError_t status = runInt4Fp8GroupedGemm(params);

    // Free allocated resources
    if (alpha_ptr) cudaFree(alpha_ptr);
    if (beta_ptr) cudaFree(beta_ptr);

    // Free problem sizes if we allocated them
    if (allocated_problem_sizes && device_problem_sizes) {
        cudaFree(device_problem_sizes);
    }

    if (status != cudaSuccess) {
        TORCH_CHECK(false, "int4_fp8_grouped_gemm failed with error: ", cudaGetErrorString(status));
    }

    return d_tensor;
}