import torch
import triton
import triton.language as tl

@triton.jit
def rearrange_kernel(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H 
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset
    
    tl.store(t2_ptr + dst_pos, tl.load(t1_ptr + src_pos))

def rearrange_tensors(t1: torch.Tensor, t2: torch.Tensor, d: int):
    N, B, H, C = t1.shape
    
    assert t2.shape == (N, B, H, C), "Destination tensor must have same shape as source"
    assert H % d == 0, "H must be divisible by d"

    block_size = B * H * C
    token_size = H * C
    tensor_size = N * block_size
    tensor_subset_size = tensor_size // d
    
    BLOCK_SIZE = 1024
    grid = ((N * B * H * C + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    rearrange_kernel[grid](
        t1, t2,
        N, B, H, C,
        d,
        tensor_subset_size,
        block_size,
        token_size,
        BLOCK_SIZE=BLOCK_SIZE
    )