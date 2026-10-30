# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-aligned SM90 packed KV records for the encoder-only PP relay."""

import torch

from vllm.triton_utils import tl, triton

SM90_PACKED_KV_RECORD_BYTES = 584


def gather_packed_kv_cache_rows(
    kv_cache: torch.Tensor,
    positions: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Gather completed compressed records; invalid and null pages emit zero."""
    assert compress_ratio == 2 and kv_cache.dtype == torch.uint8
    assert kv_cache.ndim == 3 and kv_cache.shape[-1] == SM90_PACKED_KV_RECORD_BYTES
    assert kv_cache.stride(-1) == 1
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert req_id_per_token.dtype == torch.int32 and req_id_per_token.is_contiguous()
    assert block_table.dtype == torch.int32 and block_table.is_contiguous()
    num_tokens = positions.numel()
    assert num_tokens <= req_id_per_token.numel()
    packed_rows = torch.empty(
        (num_tokens, SM90_PACKED_KV_RECORD_BYTES),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    if num_tokens:
        _gather_packed_kv_cache_rows_kernel[(num_tokens,)](
            kv_cache,
            positions,
            req_id_per_token,
            block_table,
            packed_rows,
            block_table.stride(0),
            NUM_REQS=block_table.shape[0],
            NUM_PAGES=kv_cache.shape[0],
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            num_warps=8,
        )
    return packed_rows


def scatter_packed_kv_cache_rows(
    packed_rows: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """Install completed records without touching invalid or null pages."""
    assert compress_ratio == 2 and kv_cache.dtype == torch.uint8
    assert kv_cache.ndim == 3 and kv_cache.shape[-1] == SM90_PACKED_KV_RECORD_BYTES
    assert kv_cache.stride(-1) == 1
    assert packed_rows.dtype == torch.uint8 and packed_rows.is_contiguous()
    assert (
        packed_rows.ndim == 2 and packed_rows.shape[-1] == SM90_PACKED_KV_RECORD_BYTES
    )
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(packed_rows.shape[0], positions.numel())
    if num_tokens:
        _scatter_packed_kv_cache_rows_kernel[(num_tokens,)](
            packed_rows,
            positions,
            kv_cache,
            slot_mapping,
            NUM_PAGES=kv_cache.shape[0],
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            num_warps=8,
        )


@triton.jit(do_not_specialize=["block_table_stride"])
def _gather_packed_kv_cache_rows_kernel(
    cache,
    positions,
    req_ids,
    block_table,
    packed_rows,
    block_table_stride,
    NUM_REQS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    byte = tl.arange(0, 1024)
    position = tl.load(positions + token)
    request = tl.load(req_ids + token)
    compressed_position = position // 2
    logical_block = compressed_position // CACHE_BLOCK
    valid = (
        (position >= 0)
        & ((position + 1) % 2 == 0)
        & (request >= 0)
        & (request < NUM_REQS)
        & (logical_block < block_table_stride)
    )
    physical_block = tl.load(
        block_table + request * block_table_stride + logical_block,
        mask=valid,
        other=-1,
    )
    valid = valid & (physical_block > 0) & (physical_block < NUM_PAGES)
    row = compressed_position % CACHE_BLOCK
    page = cache + physical_block.to(tl.int64) * CACHE_STRIDE
    value = tl.load(page + row * 576 + byte, mask=valid & (byte < 576), other=0)
    scale = tl.load(
        page + CACHE_BLOCK * 576 + row * 8 + byte - 576,
        mask=valid & (byte >= 576) & (byte < 584),
        other=0,
    )
    tl.store(
        packed_rows + token.to(tl.int64) * 584 + byte,
        tl.where(byte < 576, value, scale),
        mask=byte < 584,
    )


@triton.jit
def _scatter_packed_kv_cache_rows_kernel(
    packed_rows,
    positions,
    cache,
    cache_slots,
    NUM_PAGES: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(cache_slots + token)
    position = tl.load(positions + token)
    valid = (
        (slot >= CACHE_BLOCK)
        & (slot < NUM_PAGES * CACHE_BLOCK)
        & (position >= 0)
        & ((position + 1) % 2 == 0)
    )
    byte = tl.arange(0, 1024)
    packed = tl.load(
        packed_rows + token.to(tl.int64) * 584 + byte,
        mask=valid & (byte < 584),
        other=0,
    )
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    row = slot % CACHE_BLOCK
    tl.store(page + row * 576 + byte, packed, mask=valid & (byte < 576))
    tl.store(
        page + CACHE_BLOCK * 576 + row * 8 + byte - 576,
        packed,
        mask=valid & (byte >= 576) & (byte < 584),
    )
