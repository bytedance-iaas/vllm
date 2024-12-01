import triton
import triton.language as tl
import torch
import math

@triton.jit
def reshape_and_cache_flash_kernel(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr, num_tokens, n: tl.constexpr, kv_dt,
    block_stride, key_stride, value_stride, num_heads, head_size, block_size: tl.constexpr,
    k_scale, v_scale
):
    
    pid = tl.program_id(axis=0)  # Parallelize over tokens
    token_idx = pid

    if token_idx >= num_tokens:
        return

    # Load slot mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx, mask=True, other=-1)
    
    # If slot_idx is -1 (padded), skip processing
    if slot_idx < 0:
        return

    # Compute block index and offsets
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    thread_idx = tl.arange(0, 16384)
    #thread_idx = tl.arange(0, n)

    # Compute source indices
    src_key_idx = token_idx * key_stride + thread_idx
    src_value_idx = token_idx * value_stride + thread_idx

    # Compute target indices
    head_idx = thread_idx // head_size
    head_offset = thread_idx % head_size
    tgt_idx = (
        block_idx * block_stride
        + block_offset * num_heads * head_size
        + head_idx * head_size
        + head_offset
    )

    # Load key and value
    key = tl.load(key_ptr + src_key_idx, mask=thread_idx < n)
    value = tl.load(value_ptr + src_value_idx, mask=thread_idx < n)

    # Apply scaling and store into cache
    if kv_dt == 0:  # Handle auto data type
        tl.store(key_cache_ptr + tgt_idx, key, mask=thread_idx < n)
        tl.store(value_cache_ptr + tgt_idx, value, mask=thread_idx < n)
    else:
        # Convert and scale
        key_scaled = key * k_scale
        value_scaled = value * v_scale
        tl.store(key_cache_ptr + tgt_idx, key_scaled, mask=thread_idx < n)
        tl.store(value_cache_ptr + tgt_idx, value_scaled, mask=thread_idx < n)

# Driver function to invoke the Triton kernel
def reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping,
        kv_cache_dtype, k_scale, v_scale
):
    num_tokens = int(key.size(0))
    num_heads = int(key.size(1))
    head_size = int(key.size(2))
    block_size = int(key_cache.size(2))
    block_stride = int(key_cache.stride(0))
    key_stride = int(key.stride(0))
    value_stride = int(value.stride(0))
    
    n = num_heads * head_size
    
    if kv_cache_dtype == "fp8":
        kv_dt = 1
    else:
        kv_dt = 0
    

    # Launch Triton kernel
    grid = (num_tokens,)
    
    reshape_and_cache_flash_kernel[grid](
        key_ptr=key, value_ptr=value, key_cache_ptr=key_cache,
        value_cache_ptr=value_cache, slot_mapping_ptr=slot_mapping, num_tokens=num_tokens, n=n, kv_dt=kv_dt,
        block_stride=block_stride, key_stride=key_stride,
        value_stride=value_stride, num_heads=num_heads, head_size=head_size,
        block_size=block_size, 
         k_scale=k_scale, v_scale=v_scale
    )
    
