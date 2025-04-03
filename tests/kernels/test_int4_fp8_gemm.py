# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Tuple, Dict
import pytest
from torch.testing import assert_close
from vllm._custom_ops import int4_fp8_gemm

def pack_int4(matrix):
    """Pack INT4 values into INT8 storage.
    Each INT8 value stores two INT4 values.
    """
    # Ensure input is in range [-8, 7]
    matrix = torch.clamp(matrix, -8, 7)
    
    # Reshape to group INT4 values in pairs
    k, n = matrix.shape
    packed = torch.zeros((k, (n + 1) // 2), dtype=torch.int8, device=matrix.device)
    
    # Pack each pair of INT4 values into one INT8
    for i in range(0, n, 2):
        if i + 1 < n:
            packed[:, i//2] = (matrix[:, i] & 0xF) | ((matrix[:, i+1] & 0xF) << 4)
        else:
            # Handle odd number of columns
            packed[:, i//2] = matrix[:, i] & 0xF

    flat = packed.flatten()
    pad_count = k * n // 2
    flat_padded = torch.cat([flat, torch.zeros(pad_count, dtype=packed.dtype, device=packed.device)])
    packed = flat_padded.reshape(k, n)
    #print("packed", packed)
    return packed

def unpack_int4(packed, n):
    """Unpack INT4 values from INT8 storage."""
    k = packed.shape[0]
    unpacked = torch.zeros((k, n), dtype=torch.int8, device=packed.device)

    flat = packed.flatten()[:(k * n // 2)]
    unpacked = packed.reshape(k, n)
    
    for i in range(0, n, 2):
        if i + 1 < n:
            unpacked[:, i] = packed[:, i//2] & 0xF
            unpacked[:, i+1] = (packed[:, i//2] >> 4) & 0xF
        else:
            unpacked[:, i] = packed[:, i//2] & 0xF
    
    # Sign extend from 4 bits to 8 bits
    unpacked = (unpacked << 4).to(torch.int8) >> 4
    print(unpacked)
    print(unpacked.shape)
    return unpacked

def reference_gemm(A, B, scales, group_size):
    """Reference implementation using PyTorch."""
    # Unpack INT4 values
    k, n = B.shape
    B_unpacked = unpack_int4(B, n)
    B_unpacked = B_unpacked.to(torch.float16)
    print("ref orig B", B.shape)
    print("B_unpacked", B_unpacked.shape)
    
    # Apply scaling
    scale_k = (k + group_size - 1) // group_size
    for i in range(n):
        for j in range(scale_k):
            start_idx = j * group_size
            end_idx = min((j + 1) * group_size, k)
            B_unpacked[start_idx:end_idx, i] *= scales[i, j]
    
    # Perform matrix multiplication
    return torch.matmul(A.to(torch.float16), B_unpacked)

@pytest.mark.parametrize("m,k,n,group_size", [
    (128, 256, 512, 128),    # Small matrices
    # (512, 1024, 2048, 128),  # Medium matrices
    # (1024, 2048, 4096, 128), # Large matrices
    # (2048, 4096, 8192, 128), # Very large matrices
])
def test_int4_fp8_gemm_correctness(m, k, n, group_size):
    """Test correctness of INT4-FP8 GEMM implementation."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate test data
    #### RANDOM
    A_fp32 = torch.randn(m, k, dtype=torch.float32, device='cuda')
    A_fp32 = torch.clamp(A_fp32, -448, 448)  # FP8 range
    ### ALL ONES
    #A_fp32 = torch.ones(m, k, dtype=torch.float32, device='cuda')
    ### IDENTITY MATRIX
    # A_fp32 = torch.zeros(m, k, dtype=torch.int8, device='cuda')
    # for i in range(min(m, k)):
    #    A_fp32[i, i] = 1
    A = A_fp32.to(torch.float8_e4m3fn)

    B = torch.randint(-8, 8, (k, n), dtype=torch.int8, device='cuda')
    # Pack INT4 values
    B_packed = pack_int4(B)
    print("orig B", B.shape)
    print("B_packed", B_packed.shape)

    scales_k = (k + group_size - 1) // group_size
    scales_fp32 = torch.randn(n, scales_k, dtype=torch.float32, device='cuda')
    scales_fp32 = torch.clamp(scales_fp32, -448, 448)  # FP8 range
    #scales_fp32 = torch.ones(n, scales_k, dtype=torch.float32, device='cuda')
    scales = scales_fp32.to(torch.float8_e4m3fn)

    # Run CUDA kernel
    result_cuda = int4_fp8_gemm(A.clone(), B_packed.clone(), scales.clone(), group_size)
    print(result_cuda.shape)
    
    # Run reference implementation
    result_ref = reference_gemm(A, B_packed.clone(), scales, group_size)
    print(result_ref.shape)
    
    # Compare results
    # Using a larger tolerance for FP8 operations
    assert_close(result_cuda, result_ref, rtol=1e-1, atol=1e-1)

# @pytest.mark.parametrize("m,k,n,group_size", [
#     (128, 256, 512, 128),
#     (2048, 4096, 8192, 128),
# ])
# def test_int4_fp8_gemm_performance(m, k, n, group_size):
#     """Test performance of INT4-FP8 GEMM implementation."""
#     A, B, scales = create_test_tensors(m, n, k, group_size)

#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()
    
#     # Warm-up run
#     _ = int4_fp8_gemm(A, B, scales, group_size)
    
#     # Performance measurement
#     num_runs = 10
#     torch.cuda.synchronize()
#     start_time = time.time()
    
#     for _ in range(num_runs):
#         _ = int4_fp8_gemm(A, B, scales, group_size)
    
#     torch.cuda.synchronize()
#     end_time = time.time()
    
#     avg_time = (end_time - start_time) / num_runs
#     print(f"Average time for shape m={m}, n={n}, k={k}, group_size={group_size}: {avg_time*1000:.2f}ms")

# def test_input_validation():
#     """Test input validation for INT4-FP8 GEMM."""

#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()

#     # Test invalid shapes
#     with pytest.raises(RuntimeError):
#         A = torch.randn(64, 64, dtype=torch.float8_e4m3fn, device='cuda')
#         B = torch.randint(-8, 8, (32, 32), dtype=torch.int8, device='cuda')
#         scales = torch.randn(32, dtype=torch.float8_e4m3fn, device='cuda')
#         _ = int4_fp8_gemm(A, B, scales, group_size=128)
    
#     # Test invalid data types
#     with pytest.raises(RuntimeError):
#         A = torch.randn(64, 64, dtype=torch.float32, device='cuda')
#         B = torch.randint(-8, 8, (64, 64), dtype=torch.int8, device='cuda')
#         scales = torch.randn(64, dtype=torch.float8_e4m3fn, device='cuda')
#         _ = int4_fp8_gemm(A, B, scales, group_size=128)
    
#     # Test invalid device
#     with pytest.raises(RuntimeError):
#         A = torch.randn(64, 64, dtype=torch.float8_e4m3fn, device='cpu')
#         B = torch.randint(-8, 8, (64, 64), dtype=torch.int8, device='cpu')
#         scales = torch.randn(64, dtype=torch.float8_e4m3fn, device='cpu')
#         _ = int4_fp8_gemm(A, B, scales, group_size=128)
    
#     # Test invalid group size
#     with pytest.raises(RuntimeError):
#         A = torch.randn(64, 64, dtype=torch.float8_e4m3fn, device='cuda')
#         B = torch.randint(-8, 8, (64, 64), dtype=torch.int8, device='cuda')
#         scales = torch.randn(64, dtype=torch.float8_e4m3fn, device='cuda')
#         _ = int4_fp8_gemm(A, B, scales, group_size=0)

if __name__ == "__main__":
    pytest.main([__file__])
