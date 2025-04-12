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

    // Scale chunk size
    int chunk_size;                // Size of each chunk for scales (typically K/chunks)

    // Workspace memory
    size_t workspace_size;         // Size of the workspace
};

cudaError_t runInt4Fp8GroupedGemm(Int4Fp8GemmParams& params) {
    // Set up GemmUniversalMode and fusion arguments
    using Args = typename GemmScaleOnly::Arguments;
    Args arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    // Use default alpha=1, beta=0
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

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
    cutlass::device_memory::allocation<uint8_t> workspace(params.workspace_size);
    // Check if the problem is supported
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    // Initialize the GEMM with arguments and workspace
    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    // Run the GEMM
    status = gemm.run();

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
 * The function uses expert_offsets to manage multiple GEMM operations within the unified tensors.
 * Each group has its own M, N, K dimensions specified in problem_sizes.
 *
 * @param a_tensor Tensor containing all A matrices (fp8_e4m3) with shape [total_m, K]
 * @param b_tensor Tensor containing all B matrices (int4 packed as int8) with shape [total_n, K/2]
 * @param scale_tensor Tensor containing all scale factors with shape [total_n, K/chunk_size]
 * @param expert_offsets Tensor containing expert offsets for determining group boundaries
 * @param problem_sizes Tensor containing problem sizes with shape [num_groups, 3] (M, N, K for each group)
 * @param chunk_size Size of each chunk for scales (K / number of scale chunks)
 * @return torch::Tensor Output tensor D with shape [total_m, total_n]
 */
torch::Tensor int4_fp8_grouped_gemm(
    torch::Tensor const& a_tensor,
    torch::Tensor const& b_tensor,
    torch::Tensor const& scale_tensor,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    int64_t chunk_size)
{
    // Check inputs
    TORCH_CHECK(a_tensor.dim() == 2, "A tensor must be 2D");
    TORCH_CHECK(b_tensor.dim() == 2, "B tensor must be 2D");
    TORCH_CHECK(scale_tensor.dim() == 2, "Scale tensor must be 2D");
    TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
    TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

    // Get number of groups from expert_offsets
    int num_groups = static_cast<int>(expert_offsets.size(0));

    // Check tensor shapes
    TORCH_CHECK(problem_sizes.size(0) == num_groups, "problem_sizes must have num_groups rows");
    TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (M, N, K)");

    // Check tensor types
    TORCH_CHECK(a_tensor.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
    TORCH_CHECK(b_tensor.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");

    // Set CUDA device based on the input tensor
    const at::cuda::CUDAGuard device_guard(a_tensor.device());

    // Create output tensor with appropriate shape
    auto d_tensor = torch::empty({a_tensor.size(0), b_tensor.size(0)}, torch::dtype(torch::kFloat16).device(a_tensor.device()));

    // Split tensors into groups based on expert_offsets
    std::vector<torch::Tensor> a_tensors, b_tensors, scale_tensors, d_tensors;

    auto offsets = expert_offsets.cpu().data_ptr<int64_t>();
    int64_t K = a_tensor.size(1);

    for (int i = 0; i < num_groups; i++) {
        // Extract the appropriate slices for each group
        int64_t M_offset = offsets[i];
        int64_t M = (i < num_groups - 1) ? (offsets[i+1] - offsets[i]) : (a_tensor.size(0) - offsets[i]);

        a_tensors.push_back(a_tensor.slice(0, M_offset, M_offset + M));
        b_tensors.push_back(b_tensor);  // Each group uses the whole B tensor
        scale_tensors.push_back(scale_tensor);  // Each group uses the whole scale tensor
        d_tensors.push_back(d_tensor.slice(0, M_offset, M_offset + M));
    }

    // Preprocess int4 tensors
    std::vector<torch::Tensor> processed_b_tensors = preprocessInt4Tensors(b_tensors);

    // Preprocess scale tensors
    std::vector<torch::Tensor> packed_scale_tensors = preprocessScaleTensors(scale_tensors);

    // Create problem sizes and strides based on input tensor
    std::vector<typename ProblemShape::UnderlyingProblemShape> local_problem_sizes(num_groups);
    std::vector<StrideA> local_strides_a(num_groups);
    std::vector<StrideB> local_strides_b(num_groups);

    auto problem_sizes_cpu = problem_sizes.cpu();
    auto problem_sizes_ptr = problem_sizes_cpu.data_ptr<int>();

    for (int i = 0; i < num_groups; ++i) {
        int M = problem_sizes_ptr[i * 3 + 0];
        int N = problem_sizes_ptr[i * 3 + 1];
        int K = problem_sizes_ptr[i * 3 + 2];

        local_problem_sizes[i] = make_tuple(M, N, K);
        local_strides_a[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        local_strides_b[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    }

    // Create D strides
    std::vector<StrideD> local_stride_d = createStrides<StrideD>(d_tensors);

    // Create scale strides
    std::vector<StrideS> local_stride_s = createStrides<StrideS>(packed_scale_tensors);

    // Get device pointers
    auto a_ptrs = getDevicePtrs<MmaType>(a_tensors);
    auto b_ptrs = getDevicePtrs<QuantType>(processed_b_tensors);
    auto scale_ptrs = getDevicePtrs<ElementScalePacked>(packed_scale_tensors);
    auto d_ptrs = getMutableDevicePtrs<ElementD>(d_tensors);

    // Create parameters structure
    Int4Fp8GemmParams params;
    params.num_groups = num_groups;
    params.problem_sizes = local_problem_sizes.data();
    params.a_ptrs = a_ptrs.data();
    params.b_ptrs = b_ptrs.data();
    params.scale_ptrs = scale_ptrs.data();
    params.c_ptrs = nullptr;  // No C tensor
    params.d_ptrs = d_ptrs.data();
    params.stride_a = local_strides_a.data();
    params.stride_b = local_strides_b.data();
    params.stride_c = nullptr;  // No C tensor
    params.stride_d = local_stride_d.data();
    params.stride_s = local_stride_s.data();
    params.chunk_size = chunk_size;

    // Run the GEMM
    cudaError_t status = runInt4Fp8GroupedGemm(params);

    if (status != cudaSuccess) {
        TORCH_CHECK(false, "int4_fp8_grouped_gemm failed with error: ", cudaGetErrorString(status));
    }

    return d_tensor;
}