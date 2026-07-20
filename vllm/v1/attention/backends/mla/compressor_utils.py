# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        pos_after_compress = pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


@triton.jit
def _pcp_compressed_slot_mapping_kernel(
    # [num_tokens]
    logical_slot_mapping_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_tokens]
    compressed_slot_mapping_ptr,
    num_tokens,
    storage_block_size,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    mask = offsets < num_tokens
    logical_slots = tl.load(logical_slot_mapping_ptr + offsets, mask=mask)
    positions = tl.load(positions_ptr + offsets, mask=mask)

    is_valid = mask & (logical_slots >= 0) & ((positions + 1) % COMPRESS_RATIO == 0)
    logical_block_ids = logical_slots // LOGICAL_BLOCK_SIZE
    logical_block_offsets = logical_slots % LOGICAL_BLOCK_SIZE
    compressed_slots = (
        logical_block_ids * storage_block_size + logical_block_offsets // COMPRESS_RATIO
    )
    compressed_slots = tl.where(is_valid, compressed_slots, PAD_ID)
    tl.store(compressed_slot_mapping_ptr + offsets, compressed_slots, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping


def get_pcp_compressed_slot_mapping(
    logical_slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    logical_block_size: int,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a rank-concatenated PCP logical cache view into one layer.

    A uniform KV cache group may contain layers with different compression
    ratios. PCP therefore gathers the shared logical slot mapping once, while
    each layer maps it into its own physical storage layout here.
    """
    num_tokens = logical_slot_mapping.numel()
    assert positions.numel() == num_tokens
    assert logical_block_size % compress_ratio == 0
    assert storage_block_size == logical_block_size // compress_ratio

    if out is None:
        compressed_slot_mapping = torch.empty_like(logical_slot_mapping)
    else:
        assert out.numel() >= num_tokens
        compressed_slot_mapping = out[:num_tokens]

    if num_tokens == 0:
        return compressed_slot_mapping

    triton_block_size = 256
    _pcp_compressed_slot_mapping_kernel[(triton.cdiv(num_tokens, triton_block_size),)](
        logical_slot_mapping,
        positions,
        compressed_slot_mapping,
        num_tokens,
        storage_block_size,
        LOGICAL_BLOCK_SIZE=logical_block_size,
        COMPRESS_RATIO=compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=triton_block_size,
    )
    return compressed_slot_mapping
