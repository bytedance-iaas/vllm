#include <torch/all.h>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

// Core types
using MmaType = cutlass::float_e4m3_t;               // FP8
using QuantType = cutlass::int4b_t;                  // INT4
using ElementD = cutlass::half_t;                    // FP16 output
using CutlassArrayType = cutlass::Array<MmaType, 8>;  // scales_packed
using ElementAccumulator = float;                    // FP32 accumulation
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// Matrix configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128,_128,cute::Int<TileShapeK>>;
using ClusterShape = Shape<_1,_1,_1>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// Define the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    void, void, 0,  // C matrix is void since we don't need it
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Define the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<QuantType, cutlass::Array<MmaType, 8>>, LayoutB_Transpose, AlignmentB,
    MmaType, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

// Define the GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideD = typename GemmKernel::StrideD;
using StrideS = typename CollectiveMainloop::StrideScale;

// Define the GEMM adapter
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor int4_fp8_gemm(
    torch::Tensor const& A,  // fp8
    torch::Tensor const& B,  // int4 packed
    torch::Tensor const& scales,  // fp8 scales
    int64_t group_size
) { 
    // Get dimensions
    int m = A.size(0);
    int n = B.size(1);
    int k = A.size(1);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(A.device());
    torch::Tensor D = torch::empty({m, n}, options);

    // Get raw pointers with proper type casting
    auto A_ptr = reinterpret_cast<MmaType*>(A.data_ptr());
    auto B_ptr = reinterpret_cast<QuantType*>(B.data_ptr());
    auto scales_ptr = reinterpret_cast<MmaType*>(scales.data_ptr());
    auto D_ptr = reinterpret_cast<ElementD*>(D.data_ptr());
    int const scale_k = (k + group_size - 1) / group_size;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create temporary buffers for encoded B and packed scales
    torch::Tensor B_encoded = torch::empty_like(B);
    torch::Tensor scales_packed = torch::empty({scales.size(0), scales.size(1), 8}, scales.options());
    
    // Encode INT4 values using CUTLASS's function
    cutlass::unified_encode_int4b(B_ptr, reinterpret_cast<QuantType*>(B_encoded.data_ptr()), B.numel());
    
    // Pack FP8 scales using CUTLASS's function
    cutlass::pack_scale_fp8(scales_ptr, reinterpret_cast<CutlassArrayType*>(scales_packed.data_ptr()), scales.numel());
    
    // Configure GEMM arguments
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        // problem size
        {n, m, k, 1},
        {
            // B matrix (now using encoded values)
            reinterpret_cast<QuantType*>(B_encoded.data_ptr()),
            // stride B (column-major)
            stride_B,
            // A matrix
            A_ptr,
            // stride A matrix (row-major)
            stride_A,
            // scales (packed)
            reinterpret_cast<CutlassArrayType*>(scales_packed.data_ptr()),
            // stride scales
            stride_S,
            // group size
            static_cast<int>(group_size)
        },
        {
            // alpha, beta
            {1.0f, 0.0f},
            // C matrix (void since we don't need it)
            nullptr,
            // stride C (not used)
            stride_D,
            // D matrix
            D_ptr,
            // stride D (row-major)
            stride_D,
        }
    };
    
    // Create and initialize GEMM
    Gemm gemm;
    gemm.initialize(args, stream);
    
    // Run GEMM
    gemm.run(stream);
    
    // Synchronize stream
    cudaStreamSynchronize(stream);
    
    return D;
}