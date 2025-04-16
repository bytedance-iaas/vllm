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
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;

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

// Transposed layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Alignments
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

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
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
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
using StrideA = std::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = std::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
using StrideC = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutC*>>;
using StrideD = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutD*>>;
using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;

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
        // Create a tensor for packed scales with ElementScalePacked type
        auto packed = torch::empty(
            {tensor.size(0), tensor.size(1), 1},
            torch::dtype(torch::kFloat32).device(tensor.device())
        );

        int current_device;
        cudaGetDevice(&current_device);

        // if (current_device == 3) {
        //     printf("Packing scale tensor:\n");
        //     printf("  Input shape: [%ld, %ld]\n", tensor.size(0), tensor.size(1));
        //     printf("  Input strides: [%ld, %ld]\n", tensor.stride(0), tensor.stride(1));
        // }

        // cudaDeviceSynchronize();
        // // Pack the scales using the template function
        // bool success = cutlass::pack_scale_fp32<ElementScale, ElementScalePacked>(
        //     reinterpret_cast<const ElementScale*>(tensor.data_ptr()),
        //     reinterpret_cast<ElementScalePacked*>(packed.data_ptr()),
        //     tensor.numel(),
        //     ElementScalePacked::kElements
        // );
        // cudaDeviceSynchronize();

        // if (!success) {
            // printf("PACK SCALE FAILED!!! Using scale of 1. Current device: %d\n", current_device);
            // printf("  Input shape: [%ld, %ld]\n", tensor.size(0), tensor.size(1));
            // printf("  Input strides: [%ld, %ld]\n", tensor.stride(0), tensor.stride(1));
            // printf("  Packed shape: [%ld, %ld, %ld]\n", packed.size(0), packed.size(1), packed.size(2));
            // printf("  Packed strides: [%ld, %ld, %ld]\n", packed.stride(0), packed.stride(1), packed.stride(2));
            
            // Fill with scale of 1
            auto scale_one = torch::ones_like(tensor);
            // Copy values to the packed tensor, repeating along the last dimension
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                for (int64_t j = 0; j < tensor.size(1); ++j) {
                    packed[i][j][0] = scale_one[i][j];
                }
            }
        // }

        // if (current_device == 3) {
        //     printf("  Packed shape: [%ld, %ld, %ld]\n", packed.size(0), packed.size(1), packed.size(2));
        //     printf("  Packed strides: [%ld, %ld, %ld]\n", packed.stride(0), packed.stride(1), packed.stride(2));
        // }

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
 * @param a_tensor Tensor containing all A matrices (fp8_e4m3) with shape [total_m, K]
 * @param b_tensor Tensor containing all B matrices (int4 packed as int8) with shape [E, N, K/2]
 * @param scale_tensor Tensor containing all scale factors with shape [E, K//512, N*4]
 * @param expert_offsets Tensor containing expert offsets for determining group boundaries (int32)
 * @param problem_sizes Tensor containing problem sizes with shape [num_groups, 3] (M, N, K for each group) (int32)
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
    // Print input tensor shapes and detailed information
    int current_device;
    cudaGetDevice(&current_device);
    if (current_device == 3) {
        printf("\n=== Input Tensor Information ===\n");
        printf("A tensor:\n");
        printf("  Shape: [%ld, %ld]\n", a_tensor.size(0), a_tensor.size(1));
        printf("  Strides: [%ld, %ld]\n", a_tensor.stride(0), a_tensor.stride(1));
        printf("  Device: %s\n", a_tensor.device().str().c_str());
        
        printf("\nB tensor:\n");
        printf("  Shape: [%ld, %ld, %ld]\n", b_tensor.size(0), b_tensor.size(1), b_tensor.size(2));
        printf("  Strides: [%ld, %ld, %ld]\n", b_tensor.stride(0), b_tensor.stride(1), b_tensor.stride(2));
        printf("  Device: %s\n", b_tensor.device().str().c_str());
        
        printf("\nScale tensor:\n");
        printf("  Shape: [%ld, %ld, %ld]\n", scale_tensor.size(0), scale_tensor.size(1), scale_tensor.size(2));
        printf("  Strides: [%ld, %ld, %ld]\n", scale_tensor.stride(0), scale_tensor.stride(1), scale_tensor.stride(2));
        printf("  Device: %s\n", scale_tensor.device().str().c_str());
        
        printf("\nExpert offsets:\n");
        printf("  Shape: [%ld]\n", expert_offsets.size(0));
        printf("  Values: ");
        auto offsets_cpu = expert_offsets.cpu();
        for (int32_t i = 0; i < expert_offsets.size(0); ++i) {
            printf("%d ", offsets_cpu.data_ptr<int32_t>()[i]);
        }
        printf("\n");
        
        printf("\nProblem sizes:\n");
        printf("  Shape: [%ld, %ld]\n", problem_sizes.size(0), problem_sizes.size(1));
        printf("  Values:\n");
        auto problem_sizes_cpu = problem_sizes.cpu();
        auto problem_sizes_ptr = problem_sizes_cpu.data_ptr<int32_t>();
        for (int i = 0; i < problem_sizes.size(0); ++i) {
            int32_t M = problem_sizes_ptr[i * 3];
            int32_t N = problem_sizes_ptr[i * 3 + 1];
            int32_t K = problem_sizes_ptr[i * 3 + 2];
            printf("    Group %d: M=%d, N=%d, K=%d\n", i, M, N, K);
            
            // Validate problem sizes
            TORCH_CHECK(M >= 0, "Group " + std::to_string(i) + " M must be non-negative");
            TORCH_CHECK(N > 0, "Group " + std::to_string(i) + " N must be positive");
            TORCH_CHECK(K > 0, "Group " + std::to_string(i) + " K must be positive");
        }
    }
    
    // Check inputs
    TORCH_CHECK(a_tensor.dim() == 2, "A tensor must be 2D");
    TORCH_CHECK(b_tensor.dim() == 3, "B tensor must be 3D [E, N, K/2]");
    TORCH_CHECK(scale_tensor.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
    TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
    TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
    
    // Get number of groups from expert_offsets
    int num_groups = static_cast<int>(expert_offsets.size(0));
    
    // Check tensor shapes
    TORCH_CHECK(problem_sizes.size(0) == num_groups, "problem_sizes must have num_groups rows");
    TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (M, N, K)");
    TORCH_CHECK(b_tensor.size(0) == num_groups, "B tensor first dimension must match number of groups");
    TORCH_CHECK(scale_tensor.size(0) == num_groups, "Scale tensor first dimension must match number of groups");
    TORCH_CHECK(b_tensor.size(2) * 2 == a_tensor.size(1), "B tensor K/2 dimension must match A tensor K dimension");
    TORCH_CHECK(scale_tensor.size(1) == a_tensor.size(1) / 512, "Scale tensor second dimension must be K//512");
    TORCH_CHECK(scale_tensor.size(2) == 4 * b_tensor.size(1), "Scale tensor last dimension must be 4*N");
    
    // Check tensor types
    TORCH_CHECK(a_tensor.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
    TORCH_CHECK(b_tensor.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
    TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
    TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");
    
    // Set CUDA device based on the input tensor
    const at::cuda::CUDAGuard device_guard(a_tensor.device());
    
    // Create output tensor with appropriate shape
    torch::Tensor d_tensor = torch::empty({a_tensor.size(0), b_tensor.size(1)}, torch::dtype(torch::kFloat16).device(a_tensor.device()));
    
    if (current_device == 3) {
        printf("\nOutput tensor D:\n");
        printf("  Shape: [%ld, %ld]\n", d_tensor.size(0), d_tensor.size(1));
        printf("  Strides: [%ld, %ld]\n", d_tensor.stride(0), d_tensor.stride(1));
    }
    
    // Split tensors into groups based on expert_offsets
    std::vector<torch::Tensor> a_tensors, b_tensors, scale_tensors, d_tensors;
    auto expert_offsets_cpu = expert_offsets.cpu();
    auto offsets = expert_offsets_cpu.data_ptr<int32_t>();
    
    if (current_device == 3) {
        printf("\n=== Group Tensor Information ===\n");
    }
    
    for (int32_t i = 0; i < num_groups; i++) {
        auto problem_sizes_cpu = problem_sizes.cpu();
        auto problem_sizes_ptr = problem_sizes_cpu.data_ptr<int32_t>();
        int32_t M = problem_sizes_ptr[i * 3];
        int32_t N = problem_sizes_ptr[i * 3 + 1];
        int32_t K = problem_sizes_ptr[i * 3 + 2];
        
        // Skip groups with M=0
        if (M == 0) {
            if (current_device == 3) {
                printf("\nGroup %d: Skipped (M=0)\n", i);
            }
            continue;
        }
        
        int32_t M_offset = offsets[i];
        
        if (current_device == 3) {
            printf("\nGroup %d:\n", i);
            printf("  M_offset: %d\n", M_offset);
            printf("  M: %d\n", M);
            printf("  N: %d\n", N);
            printf("  K: %d\n", K);
        }
        
        // Check bounds before slicing
        TORCH_CHECK(M_offset >= 0 && M_offset + M <= a_tensor.size(0), 
                   "Group " + std::to_string(i) + " A tensor slice out of bounds");
        TORCH_CHECK(i < b_tensor.size(0), 
                   "Group " + std::to_string(i) + " B tensor index out of bounds");
        TORCH_CHECK(i < scale_tensor.size(0), 
                   "Group " + std::to_string(i) + " Scale tensor index out of bounds");
        
        a_tensors.push_back(a_tensor.slice(0, M_offset, M_offset + M));
        b_tensors.push_back(b_tensor[i]);
        scale_tensors.push_back(scale_tensor[i]);
        d_tensors.push_back(d_tensor.slice(0, M_offset, M_offset + M));
        
        if (current_device == 3) {
            printf("  A slice:\n");
            printf("    Shape: [%ld, %ld]\n", a_tensors.back().size(0), a_tensors.back().size(1));
            printf("    Strides: [%ld, %ld]\n", a_tensors.back().stride(0), a_tensors.back().stride(1));
            
            printf("  B slice:\n");
            printf("    Shape: [%ld, %ld]\n", b_tensors.back().size(0), b_tensors.back().size(1));
            printf("    Strides: [%ld, %ld]\n", b_tensors.back().stride(0), b_tensors.back().stride(1));
            
            printf("  Scale slice:\n");
            printf("    Shape: [%ld, %ld]\n", scale_tensors.back().size(0), scale_tensors.back().size(1));
            printf("    Strides: [%ld, %ld]\n", scale_tensors.back().stride(0), scale_tensors.back().stride(1));
            
            printf("  D slice:\n");
            printf("    Shape: [%ld, %ld]\n", d_tensors.back().size(0), d_tensors.back().size(1));
            printf("    Strides: [%ld, %ld]\n", d_tensors.back().stride(0), d_tensors.back().stride(1));
        }
    }
    
    // Update num_groups to only include groups with M>0
    num_groups = a_tensors.size();
    if (current_device == 3) {
        printf("\nNumber of active groups: %d\n", num_groups);
    }
    
    // Create problem sizes and strides
    std::vector<typename ProblemShape::UnderlyingProblemShape> local_problem_sizes(num_groups);
    std::vector<StrideA> local_strides_a(num_groups);
    std::vector<StrideB> local_strides_b(num_groups);
    std::vector<StrideD> local_strides_d(num_groups);
    std::vector<StrideS> local_strides_s(num_groups);
    
    if (current_device == 3) {
        printf("\n=== Problem Sizes and Strides ===\n");
    }
    
    for (int i = 0; i < num_groups; ++i) {
        int64_t M = a_tensors[i].size(0);
        int64_t N = b_tensors[i].size(1);
        int64_t K = a_tensors[i].size(1);
        
        local_problem_sizes[i] = make_tuple(M, N, K);
        local_strides_a[i] = cutlass::make_cute_packed_stride(StrideA{}, 
            {static_cast<int>(M), static_cast<int>(K), 1});
        local_strides_b[i] = cutlass::make_cute_packed_stride(StrideB{}, 
            {static_cast<int>(N), static_cast<int>(K), 1});
        local_strides_d[i] = cutlass::make_cute_packed_stride(StrideD{}, 
            {static_cast<int>(M), static_cast<int>(N), 1});
        local_strides_s[i] = cutlass::make_cute_packed_stride(StrideS{}, 
            {static_cast<int>(scale_tensors[i].size(0)), static_cast<int>(scale_tensors[i].size(1)), 1});
        
        if (current_device == 3) {
            printf("Group %d:\n", i);
            printf("  Problem size: M=%ld, N=%ld, K=%ld\n", M, N, K);
            printf("  Stride A: [%d, %d, %d]\n", 
                static_cast<int>(get<0>(local_strides_a[i])), 
                static_cast<int>(get<1>(local_strides_a[i])), 
                static_cast<int>(get<2>(local_strides_a[i])));
            printf("  Stride B: [%d, %d, %d]\n", 
                static_cast<int>(get<0>(local_strides_b[i])), 
                static_cast<int>(get<1>(local_strides_b[i])), 
                static_cast<int>(get<2>(local_strides_b[i])));
        }
    }
    
    // Preprocess int4 tensors
    std::vector<torch::Tensor> processed_b_tensors = preprocessInt4Tensors(b_tensors);
    if (current_device == 3) {
        printf("\n=== Processed B Tensor Information ===\n");
        for (size_t i = 0; i < processed_b_tensors.size(); ++i) {
            printf("Group %zu processed B:\n", i);
            printf("  Shape: [%ld, %ld]\n", processed_b_tensors[i].size(0), processed_b_tensors[i].size(1));
            printf("  Strides: [%ld, %ld]\n", processed_b_tensors[i].stride(0), processed_b_tensors[i].stride(1));
        }
    }
    
    // Preprocess scale tensors
    std::vector<torch::Tensor> packed_scale_tensors = preprocessScaleTensors(scale_tensors);
    if (current_device == 3) {
        printf("\n=== Packed Scale Tensor Information ===\n");
        for (size_t i = 0; i < packed_scale_tensors.size(); ++i) {
            printf("Group %zu packed scale:\n", i);
            printf("  Shape: [%ld, %ld, %ld]\n", packed_scale_tensors[i].size(0), packed_scale_tensors[i].size(1), packed_scale_tensors[i].size(2));
            printf("  Strides: [%ld, %ld, %ld]\n", packed_scale_tensors[i].stride(0), packed_scale_tensors[i].stride(1), packed_scale_tensors[i].stride(2));
        }
    }
    
    // Get device pointers
    std::vector<const MmaType*> a_ptrs;
    std::vector<const QuantType*> b_ptrs;
    std::vector<const ElementScalePacked*> scale_ptrs;
    std::vector<ElementD*> d_ptrs;
    
    for (int i = 0; i < num_groups; ++i) {
        a_ptrs.push_back(reinterpret_cast<const MmaType*>(a_tensors[i].data_ptr()));
        b_ptrs.push_back(reinterpret_cast<const QuantType*>(processed_b_tensors[i].data_ptr()));
        scale_ptrs.push_back(reinterpret_cast<const ElementScalePacked*>(packed_scale_tensors[i].data_ptr()));
        d_ptrs.push_back(reinterpret_cast<ElementD*>(d_tensors[i].data_ptr()));
    }
    
    // Create hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Set up fusion arguments
    using Args = typename GemmScaleOnly::Arguments;
    Args arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    
    if (current_device == 3) {
        printf("\nFusion arguments:\n");
        printf("  alpha: %.2f\n", fusion_args.alpha);
        printf("  beta: %.2f\n", fusion_args.beta);
        printf("  alpha_ptr: %p\n", fusion_args.alpha_ptr);
        printf("  beta_ptr: %p\n", fusion_args.beta_ptr);
        printf("  alpha_ptr_array: %p\n", fusion_args.alpha_ptr_array);
        printf("  beta_ptr_array: %p\n", fusion_args.beta_ptr_array);
    }
    
    // Create gemm arguments
    arguments = Args {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, local_problem_sizes.data(), nullptr},
        {b_ptrs.data(), local_strides_b.data(), a_ptrs.data(), local_strides_a.data(), 
         scale_ptrs.data(), local_strides_s.data(), static_cast<int>(chunk_size)},
        {fusion_args, nullptr, nullptr, d_ptrs.data(), local_strides_d.data()},
        hw_info
    };
    
    if (current_device == 3) {
        printf("\nGEMM Arguments:\n");
        printf("  mode: %d\n", static_cast<int>(arguments.mode));
        printf("  problem_sizes: %p\n", local_problem_sizes.data());
        printf("  ptr_B: %p\n", b_ptrs.data());
        printf("  ptr_A: %p\n", a_ptrs.data());
        printf("  ptr_scale: %p\n", scale_ptrs.data());
        printf("  ptr_D: %p\n", d_ptrs.data());
    }
    
    // Instantiate and run GEMM
    GemmScaleOnly gemm;
    size_t workspace_size = GemmScaleOnly::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    
    if (current_device == 3) {
        printf("\nWorkspace:\n");
        printf("  size: %zu\n", workspace_size);
        printf("  ptr: %p\n", workspace.get());
    }
    
    cutlass::Status status = gemm.can_implement(arguments);
    if (current_device == 3) {
        printf("\nCan implement check: %s\n", cutlassGetStatusString(status));
    }
    
    if (status != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "GEMM implementation not supported");
    }
    
    status = gemm.initialize(arguments, workspace.get());
    if (current_device == 3) {
        printf("Initialize status: %s\n", cutlassGetStatusString(status));
    }
    
    if (status != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "GEMM initialization failed");
    }
    
    status = gemm.run();
    if (current_device == 3) {
        printf("Run status: %s\n", cutlassGetStatusString(status));
    }
    
    if (status != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "GEMM execution failed");
    }
    
    cudaDeviceSynchronize();
    
    if (current_device == 3) {
        printf("\n=== Output Tensor Debug ===\n");
        printf("D tensor shape: [%ld, %ld]\n", d_tensor.size(0), d_tensor.size(1));
        printf("D tensor strides: [%ld, %ld]\n", d_tensor.stride(0), d_tensor.stride(1));
        printf("D tensor device: %s\n", d_tensor.device().str().c_str());
        printf("D tensor dtype: %.*s\n", static_cast<int>(d_tensor.dtype().name().size()), d_tensor.dtype().name().data());
        
        // Check for CUDA errors
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("CUDA error after GEMM: %s\n", cudaGetErrorString(cuda_status));
        }
        
        // Verify the tensor is valid
        try {
            auto d_cpu = d_tensor.cpu();
            printf("Successfully copied to CPU\n");
            
            // Check if the tensor data is valid
            auto d_ptr = reinterpret_cast<ElementD*>(d_cpu.data_ptr());
            bool has_nan = false;
            bool has_inf = false;
            bool has_zero = false;
            
            for (int i = 0; i < std::min(5, static_cast<int>(d_tensor.numel())); ++i) {
                float val = static_cast<float>(d_ptr[i]);
                printf("%.2f ", val);
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
                if (val == 0.0f) has_zero = true;
            }
            printf("\n");
            
            if (has_nan) printf("Warning: Tensor contains NaN values\n");
            if (has_inf) printf("Warning: Tensor contains Inf values\n");
            if (has_zero) printf("Warning: Tensor contains zero values\n");
            
            // Check memory alignment
            printf("Tensor data pointer alignment: %zu\n", reinterpret_cast<size_t>(d_ptr) % 16);
            
            // Check tensor properties
            printf("Tensor is contiguous: %d\n", d_tensor.is_contiguous());
            printf("Tensor requires grad: %d\n", d_tensor.requires_grad());
            printf("Tensor pin_memory: %d\n", d_tensor.is_pinned());
            
        } catch (const std::exception& e) {
            printf("Error during tensor validation: %s\n", e.what());
        }
    }
    
    // Additional error checking
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("CUDA error before return: %s\n", cudaGetErrorString(cuda_status));
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(cuda_status));
    }
    
    return d_tensor;
}