
import triton

import triton.language  as tl
import torch
import math


@triton.jit
def copy_blocks_kernel(
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
    

def copy_blocks(key_caches, value_caches, block_mapping_tensor: torch.Tensor):
    """
    Copy blocks using PyTorch.
    
    Args:
        key_caches (list of torch.Tensor): List of key cache tensors.
        value_caches (list of torch.Tensor): List of value cache tensors.
        block_mapping (torch.Tensor): A 2D tensor with shape (num_pairs, 2).
    """
    num_layers = len(key_caches)
    assert num_layers == len(value_caches), "Key and Value caches must have the same number of layers."
    if num_layers == 0:
        return
    
    cache_device = key_caches[0].device
    assert cache_device.type == 'cuda', "Caches must be on a CUDA device."
    
    # Create arrays of pointers to the key and value caches.
    key_cache_ptrs = torch.tensor(
        [key_cache.data_ptr() for key_cache in key_caches],
        dtype=torch.int64,
        device='cpu'  # Initial tensor on CPU.
    )
    
    block_mapping = block_mapping_tensor.tolist()
    
    for index in range(len(key_caches)):
        for key, value in block_mapping:
            n_elements = int(key_caches[index][key].numel())
            block_size = key_caches[index][key].size(2)
            num_blocks = math.ceil(n_elements / block_size)
            copy_blocks_kernel[(num_blocks,)](src_tensor_ptr=key_caches[index][key], dst_tensor_ptr=key_caches[index][value], n_elements=n_elements, BLOCK_SIZE=block_size)
            copy_blocks_kernel[(num_blocks,)](src_tensor_ptr=value_caches[index][key], dst_tensor_ptr=value_caches[index][value], n_elements=n_elements, BLOCK_SIZE=block_size)