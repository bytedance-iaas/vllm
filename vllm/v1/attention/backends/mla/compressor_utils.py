# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    num_tokens,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_table_width,
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
        token_idx = query_start + offset
        token_mask = (offset < query_len) & (token_idx < num_tokens)

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        # During profiling / CUDA graph capture the dummy metadata can contain
        # padded requests whose seq_len is smaller than query_len. Negative
        # positions would produce negative block ids and can trigger illegal
        # memory accesses in the block table load.
        is_valid = (pos >= 0) & ((pos + 1) % COMPRESS_RATIO == 0)
        pos_after_compress = pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        load_mask = (
            token_mask & is_valid & (block_ids >= 0) & (block_ids < block_table_width)
        )
        safe_block_ids = tl.where(load_mask, block_ids, 0)
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + safe_block_ids,
            mask=load_mask,
            other=0,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size

        # NOTE
        slot_ids = tl.where(load_mask, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + token_idx, slot_ids, mask=token_mask)


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

    # Cudagraph dummy metadata can pad block_table to a larger capture shape
    # than query_start_loc/seq_lens. Launch only over rows present in all inputs.
    num_reqs = min(
        block_table.shape[0], seq_lens.shape[0], query_start_loc.shape[0] - 1
    )
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        block_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
