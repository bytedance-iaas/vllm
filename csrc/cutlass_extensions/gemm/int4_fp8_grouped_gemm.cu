/**
 * @file int4_fp8_grouped_gemm.cu
 * @brief Implementation of grouped GEMM operation with int4 and fp8 mixed precision
 *
 * This file implements a grouped GEMM operation that multiplies FP8 matrices (A) with
 * quantized INT4 matrices (B), applying per-channel scaling factors. The implementation
 * is optimized for NVIDIA Hopper GPUs, leveraging Tensor Cores for mixed precision arithmetic.
 *
 * Key features:
 * - Supports grouped GEMM operations with multiple experts
 * - Uses FP8 (e4m3) for matrix A
 * - Uses INT4 quantization for matrix B with per-channel scaling
 * - Implements preprocessing for INT4 encoding and scale packing
 * - Optimized for Hopper architecture with Tensor Core operations
 */

 #include <vector>
 #include <torch/all.h>
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
 #include <ATen/cuda/CUDAContext.h>
 #include "int4_fp8_get_group_starts.cuh"

 using namespace cute;

 // Type definitions
 using MmaType = cutlass::float_e4m3_t;      // FP8 e4m3 type
 using QuantType = cutlass::int4b_t;         // 4-bit integer type
 using ElementAccumulator = float;           // Accumulator type
 using ElementScale = cutlass::half_t;       // Scale type
 using ElementScalePacked = cutlass::Array<ElementScale, 4>;
 using ElementC = cutlass::half_t;           // Default output type (FP16)
 using ElementD = ElementC;                  // Default output type (FP16)
 using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

 // Architecture-specific configurations
 using ArchTag = cutlass::arch::Sm90;
 using OperatorClass = cutlass::arch::OpClassTensorOp;
 constexpr int TileShapeK = 512;
 using TileShape = Shape<_128, _16, cute::Int<TileShapeK>>;
 using ClusterShape = Shape<_2, _1, _1>;

 // Layout configurations
 using LayoutA = cutlass::layout::RowMajor;
 using LayoutB = cutlass::layout::ColumnMajor;
 using LayoutC = cutlass::layout::RowMajor;
 using LayoutD = LayoutC;

 // Transposed layouts
 using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
 using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
 using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
 using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

 // Alignments
 constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
 constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
 constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
 constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

 // Kernel schedule and epilogue definitions
 using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
 using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

 using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
     ArchTag, OperatorClass,
     TileShape, ClusterShape,
     cutlass::epilogue::collective::EpilogueTileAuto,
     ElementAccumulator, ElementAccumulator,
     ElementC, LayoutC_Transpose *, AlignmentC,
     ElementD, LayoutD_Transpose *, AlignmentD,
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

 using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
 using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
 using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
 using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
 using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;

 /**
  * @brief Main function to run int4 * fp8 grouped GEMM from PyTorch
  *
  * This function performs multiple GEMM operations in parallel where each operation multiplies
  * an FP8 matrix (A) with a quantized INT4 matrix (B), applying per-channel scaling factors.
  * It's designed for efficient execution on NVIDIA Hopper GPUs, leveraging Tensor Cores for
  * optimal performance with mixed precision arithmetic.
  *
  * The function includes preprocessing steps for both INT4 tensors and scale factors to ensure
  * optimal performance and correct operation.
  *
  * @param d_tensors Output tensor D with shape [total_m, total_n]
  * @param a_tensors Tensor containing all A matrices (fp8_e4m3) with shape [total_m, K]
  * @param b_tensors Tensor containing all B matrices (int4 packed as int8) with shape [E, N, K/2]
  * @param a_scales Tensor containing A matrix scale factors
  * @param b_scales Tensor containing B matrix scale factors with shape [E, K//512, N*4]
  * @param expert_offsets Tensor containing expert offsets for determining group boundaries (int32)
  * @param problem_sizes Tensor containing problem sizes with shape [num_experts, 3] (M, N, K for each group) (int32)
  * @param a_strides Stride information for A tensors
  * @param b_strides Stride information for B tensors
  * @param d_strides Stride information for D tensors
  * @param s_strides Stride information for scale tensors
  * @param chunk_size Size of each chunk for scales (K / number of scale chunks)
  */
 void int4_fp8_grouped_gemm(
     torch::Tensor& d_tensors,
     torch::Tensor const& a_tensors,
     torch::Tensor const& b_tensors,
     torch::Tensor const& a_scales,
     torch::Tensor const& b_scales,
     torch::Tensor const& expert_offsets,
     torch::Tensor const& problem_sizes,
     torch::Tensor const& a_strides,
     torch::Tensor const& b_strides,
     torch::Tensor const& d_strides,
     torch::Tensor const& s_strides,
     int64_t chunk_size)
 {
     int num_experts = static_cast<int>(expert_offsets.size(0));
     bool per_act_token = a_scales.numel() != 1;
     bool per_out_ch = b_scales.numel() != num_experts;

     // Check inputs
     TORCH_CHECK(a_tensors.dim() == 2, "A tensor must be 2D");
     TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
     TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
     TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
     TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

     // Check tensor shapes
     TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
     TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
     TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
     TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
     TORCH_CHECK(b_tensors.size(2) * 2 == a_tensors.size(1), "B tensor K/2 dimension must match A tensor K dimension");
     TORCH_CHECK(b_scales.size(1) == a_tensors.size(1) / 512, "Scale tensor second dimension must be K//512");
     TORCH_CHECK(b_scales.size(2) == 4 * b_tensors.size(1), "Scale tensor last dimension must be 4*N");

     // Check tensor types
     TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
     TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
     TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
     TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");

     auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
     auto options_int =
         torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

     torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
     torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
     torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
     torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
     torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

     // int debug_node = 0;
     // int current_device = a_tensors.device().index();
     //  if (current_device == debug_node) {
     //      printf("=== Debug Info ===\n");
     //      printf("Debug node: %d\n", debug_node);
     //      int k_size = a_tensors.size(1);
     //      int n_size = b_tensors.size(1);
     //      printf("\n=== Input Tensor Information (Device %d) ===\n", current_device);
     //      printf("a_tensors shape: [%ld, %ld], device: %s\n", a_tensors.size(0), a_tensors.size(1), a_tensors.device().str().c_str());
     //      printf("b_tensors shape: [%ld, %ld, %ld], device: %s\n", b_tensors.size(0), b_tensors.size(1), b_tensors.size(2), b_tensors.device().str().c_str());
     //      printf("a_scales shape: [%ld], device: %s\n", a_scales.size(0), a_scales.device().str().c_str());
     //      printf("b_scales shape: [%ld, %ld, %ld], device: %s\n", b_scales.size(0), b_scales.size(1), b_scales.size(2), b_scales.device().str().c_str());
     //      printf("expert_offsets shape: [%ld], device: %s\n", expert_offsets.size(0), expert_offsets.device().str().c_str());
     //      printf("chunk_size: %ld\n", chunk_size);
     //
     //      printf("\n=== Derived Parameters ===\n");
     //      printf("num_experts: %d\n", num_experts);
     //      printf("k_size: %d\n", k_size);
     //      printf("n_size: %d\n", n_size);
     //      printf("per_act_token: %d\n", per_act_token);
     //      printf("per_out_ch: %d\n", per_out_ch);
     //  }

     cutlass::KernelHardwareInfo hw_info;
     hw_info.device_id = a_tensors.device().index();
     hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

     // Set up fusion arguments
     using Args = typename GemmScaleOnly::Arguments;
     Args arguments;
     decltype(arguments.epilogue.thread) fusion_args;
     fusion_args.alpha = 0;
     fusion_args.beta = 0;
     fusion_args.alpha_ptr = static_cast<const ElementAccumulator **>(a_scales_ptrs.data_ptr());
     fusion_args.beta_ptr = nullptr;
     fusion_args.alpha_ptr_array = nullptr;
     fusion_args.beta_ptr_array = nullptr;
     fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
     fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};

     ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
         static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

     //  if (current_device == debug_node) {
     //      auto problem_sizes_cpu = problem_sizes.to(torch::kCPU);
     //      auto* problem_sizes_cpu_ptr = problem_sizes_cpu.data_ptr<int32_t>();
     //      printf("\n=== Problem Sizes ===\n");
     //      for (int i = 0; i < num_experts; ++i) {
     //          printf("Expert %d: N=%d, M=%d, K=%d\n",
     //              i,
     //              problem_sizes_cpu_ptr[i * 3],     // N
     //              problem_sizes_cpu_ptr[i * 3 + 1], // M
     //              problem_sizes_cpu_ptr[i * 3 + 2]  // K
     //          );
     //      }
     //      printf("Expert Offsets:\n");
     //      auto expert_offsets_cpu = expert_offsets.to(torch::kCPU);
     //      auto* expert_offsets_cpu_ptr = expert_offsets_cpu.data_ptr<int32_t>();
     //      for (int i = 0; i < std::min(32, num_experts); ++i) {
     //          printf("  expert_offsets[%d]: %d\n", i, expert_offsets_cpu_ptr[i]);
     //      }
     //  }

     run_int4_fp8_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
             a_scales_ptrs, b_scales_ptrs, a_tensors, b_tensors,
             d_tensors, a_scales, b_scales);

     //  if (current_device == debug_node) {
     //      printf("\n=== Pointer Arrays (Before GEMM) ===\n");
     //      // Copy pointer arrays to CPU
     //      auto a_ptrs_cpu = a_ptrs.to(torch::kCPU);
     //      auto b_ptrs_cpu = b_ptrs.to(torch::kCPU);
     //      auto out_ptrs_cpu = out_ptrs.to(torch::kCPU);
     //      auto a_scales_ptrs_cpu = a_scales_ptrs.to(torch::kCPU);
     //      auto b_scales_ptrs_cpu = b_scales_ptrs.to(torch::kCPU);
     //
     //      auto* a_ptrs_cpu_ptr = a_ptrs_cpu.data_ptr<int64_t>();
     //      auto* b_ptrs_cpu_ptr = b_ptrs_cpu.data_ptr<int64_t>();
     //      auto* out_ptrs_cpu_ptr = out_ptrs_cpu.data_ptr<int64_t>();
     //      auto* a_scales_ptrs_cpu_ptr = a_scales_ptrs_cpu.data_ptr<int64_t>();
     //      auto* b_scales_ptrs_cpu_ptr = b_scales_ptrs_cpu.data_ptr<int64_t>();
     //
     //      for (int i = 0; i < std::min(32, num_experts); ++i) {
     //          printf("Expert %d:\n", i);
     //          printf("  a_ptrs[%d]: %p\n", i, reinterpret_cast<void*>(a_ptrs_cpu_ptr[i]));
     //          printf("  b_ptrs[%d]: %p\n", i, reinterpret_cast<void*>(b_ptrs_cpu_ptr[i]));
     //          printf("  out_ptrs[%d]: %p\n", i, reinterpret_cast<void*>(out_ptrs_cpu_ptr[i]));
     //          printf("  a_scales_ptrs[%d]: %p\n", i, reinterpret_cast<void*>(a_scales_ptrs_cpu_ptr[i]));
     //          printf("  b_scales_ptrs[%d]: %p\n", i, reinterpret_cast<void*>(b_scales_ptrs_cpu_ptr[i]));
     //      }

     //  printf("\n=== Stride Information ===\n");
     //  auto a_strides_cpu = a_strides.to(torch::kCPU);
     //  auto b_strides_cpu = b_strides.to(torch::kCPU);
     //  auto d_strides_cpu = d_strides.to(torch::kCPU);
     //  auto s_strides_cpu = s_strides.to(torch::kCPU);

     //  auto* a_strides_cpu_ptr = a_strides_cpu.data_ptr<int64_t>();
     //  auto* b_strides_cpu_ptr = b_strides_cpu.data_ptr<int64_t>();
     //  auto* d_strides_cpu_ptr = d_strides_cpu.data_ptr<int64_t>();
     //  auto* s_strides_cpu_ptr = s_strides_cpu.data_ptr<int64_t>();

     //  for (int i = 0; i < std::min(32, num_experts); ++i) {
     //      printf("Expert %d:\n", i);
     //      printf("  a_strides[%d]: %ld %ld %ld\n", i,
     //          a_strides_cpu_ptr[i * 3], a_strides_cpu_ptr[i * 3 + 1], a_strides_cpu_ptr[i * 3 + 2]);
     //      printf("  b_strides[%d]: %ld %ld %ld\n", i,
     //          b_strides_cpu_ptr[i * 3], b_strides_cpu_ptr[i * 3 + 1], b_strides_cpu_ptr[i * 3 + 2]);
     //      printf("  d_strides[%d]: %ld %ld %ld\n", i,
     //          d_strides_cpu_ptr[i * 3], d_strides_cpu_ptr[i * 3 + 1], d_strides_cpu_ptr[i * 3 + 2]);
     //      printf("  s_strides[%d]: %ld %ld %ld\n", i,
     //          s_strides_cpu_ptr[i * 3], d_strides_cpu_ptr[i * 3 + 1], d_strides_cpu_ptr[i * 3 + 2]);
     //  }

     arguments = Args {
         cutlass::gemm::GemmUniversalMode::kGrouped,
         {num_experts, problem_sizes_as_shapes, nullptr},
         {static_cast<const QuantType **>(b_ptrs.data_ptr()), static_cast<StrideB*>(b_strides.data_ptr()),
             static_cast<const MmaType **>(a_ptrs.data_ptr()), static_cast<StrideA*>(a_strides.data_ptr()),
             static_cast<const ElementScalePacked **>(b_scales_ptrs.data_ptr()),
             static_cast<StrideS*>(s_strides.data_ptr()),
             static_cast<int>(chunk_size)},
         {fusion_args, nullptr, nullptr, static_cast<ElementD **>(out_ptrs.data_ptr()),
             static_cast<StrideD*>(d_strides.data_ptr())},
         hw_info
     };

     // Instantiate and run GEMM
     GemmScaleOnly gemm;
     size_t workspace_size = GemmScaleOnly::get_workspace_size(arguments);
     auto const workspace_options =
         torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
     auto workspace = torch::empty(workspace_size, workspace_options);

     cutlass::Status status = gemm.can_implement(arguments);
     if (status != cutlass::Status::kSuccess) {
         TORCH_CHECK(false, "GEMM implementation not supported");
     }

     status = gemm.initialize(arguments, workspace.data_ptr(), stream);
     if (status != cutlass::Status::kSuccess) {
         TORCH_CHECK(false, "GEMM initialization failed");
     }

     status = gemm.run(stream);
     if (status != cutlass::Status::kSuccess) {
         //  if (current_device == debug_node) {
         //      printf("%d GEMM execution failed. Status: %d\n", current_device, static_cast<int>(status));
         //  }
         TORCH_CHECK(false, "GEMM execution failed");
     }
 }
