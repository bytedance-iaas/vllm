# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DSpark draft context masking under prefix caching."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    shift_draft_block_tables,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
MAX_BLOCKS = 64
MAX_NUM_REQS = 8


def _make_block_table(num_reqs: int) -> torch.Tensor:
    table = torch.arange(MAX_NUM_REQS * MAX_BLOCKS, dtype=torch.int32).view(
        MAX_NUM_REQS, MAX_BLOCKS
    )
    return table[:num_reqs].contiguous()


@pytest.mark.parametrize(
    ("num_cached", "expected_shift"),
    [
        (0, 0),
        (BLOCK_SIZE * 3, 3),
        (BLOCK_SIZE * 3 + 5, 3),
        (BLOCK_SIZE - 1, 0),
    ],
)
def test_shift_single_request(num_cached: int, expected_shift: int):
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    num_cached_tokens = torch.full((MAX_NUM_REQS,), num_cached, dtype=torch.int32)
    seq_lens = torch.full((1,), MAX_BLOCKS * BLOCK_SIZE, dtype=torch.int32)

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    kept = MAX_BLOCKS - expected_shift
    torch.testing.assert_close(block_table[0, :kept], original[0, expected_shift:])


def test_shift_uses_request_state_mapping():
    num_reqs = 4
    block_table = _make_block_table(num_reqs)
    original = block_table.clone()
    idx_mapping = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
    num_cached_tokens = torch.arange(4, dtype=torch.int32) * BLOCK_SIZE
    seq_lens = torch.full((num_reqs,), MAX_BLOCKS * BLOCK_SIZE, dtype=torch.int32)

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    for batch_idx in range(num_reqs):
        shift = int(idx_mapping[batch_idx])
        kept = MAX_BLOCKS - shift
        torch.testing.assert_close(
            block_table[batch_idx, :kept],
            original[batch_idx, shift:],
        )


def test_shift_large_row_handles_overlap():
    max_blocks = 4096
    block_table = torch.arange(max_blocks, dtype=torch.int32).unsqueeze(0)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    num_cached_tokens = torch.full((1,), 7 * BLOCK_SIZE, dtype=torch.int32)
    seq_lens = torch.full((1,), max_blocks * BLOCK_SIZE, dtype=torch.int32)

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(block_table[0, : max_blocks - 7], original[0, 7:])


def test_shift_copy_is_bounded_by_seq_len():
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    num_cached_tokens = torch.full((MAX_NUM_REQS,), 4 * BLOCK_SIZE, dtype=torch.int32)
    seq_lens = torch.full((1,), 3 * BLOCK_SIZE + BLOCK_SIZE // 2, dtype=torch.int32)

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(block_table[0, :4], original[0, 4:8])
    torch.testing.assert_close(block_table[0, 4:], original[0, 4:])
