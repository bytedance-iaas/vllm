import triton

import triton.language  as tl
import torch
import math

'''
This Triton swap_blocks kernel works only for gpu->gpu direction as Triton does not support cpu at this time. 
'''
@triton.jit
def swap_kernel(
    src_tensor_ptr, dst_tensor_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    
    
    # Get block and thread indices
    
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a tensor of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, 2048)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load src and dst from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    
    #src_tensor = tl.load(src_tensor_ptr + offsets + block_idx + 0, mask=mask)
    src_tensor = tl.load(src_tensor_ptr + offsets, mask=mask)
    #tl.store(dst_tensor_ptr + offsets + block_idx, src_tensor, mask=mask) 
    tl.store(dst_tensor_ptr + offsets, src_tensor, mask=mask) 


# Driver function to call the Triton kernel
def swap_blocks(src_tensor: torch.Tensor, dst_tensor: torch.Tensor, block_mapping_tensor: torch.Tensor):
    
    block_mapping = block_mapping_tensor.tolist()

    for key, value in block_mapping:
        block_size = src_tensor[key].size(2)
        n_elements = torch.empty_like(src_tensor[key]).numel()
        num_blocks = math.ceil(n_elements / block_size)
        swap_kernel[(num_blocks,)](src_tensor_ptr=src_tensor[key], dst_tensor_ptr=dst_tensor[value], n_elements=n_elements, BLOCK_SIZE=block_size)