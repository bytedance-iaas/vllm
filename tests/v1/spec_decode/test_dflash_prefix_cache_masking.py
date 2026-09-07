# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DSpark draft context masking under prefix caching."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    _set_draft_query_padding_mask,
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


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output

    def precompute_and_store_context_kv(self, *args, **kwargs):
        return None


def _make_input_buffers(max_num_tokens: int = 8) -> InputBuffers:
    return InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=max_num_tokens,
        device=torch.device("cpu"),
    )


def test_set_draft_query_padding_mask_overwrites_stale_rows_in_place():
    input_buffers = _make_input_buffers()
    data_ptr = input_buffers.is_padding.data_ptr()

    dummy_mask = _set_draft_query_padding_mask(
        input_buffers,
        num_query_tokens=5,
        num_tokens_padded=8,
        dummy_run=True,
    )
    assert dummy_mask.tolist() == [True] * 8

    active_mask = _set_draft_query_padding_mask(
        input_buffers,
        num_query_tokens=5,
        num_tokens_padded=8,
        dummy_run=False,
    )
    assert active_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert active_mask.data_ptr() == data_ptr


def test_dflash_run_model_passes_padding_slice_to_forward_context(monkeypatch):
    captured = {}

    def fake_set_forward_context(*args, **kwargs):
        captured["is_padding"] = kwargs["is_padding"]
        captured["num_tokens"] = kwargs["num_tokens"]
        return nullcontext()

    monkeypatch.setattr(spec_module, "set_forward_context", fake_set_forward_context)

    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(8, dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int64),
        is_padding=torch.tensor(
            [False, False, False, False, False, True, True, True],
            dtype=torch.bool,
        ),
    )
    output = torch.ones(8, 4)
    speculator.model = _DraftModel(output)

    actual = speculator._run_model(
        8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual is output
    assert captured["num_tokens"] == 8
    assert captured["is_padding"].data_ptr() == (
        speculator.input_buffers.is_padding.data_ptr()
    )
    assert captured["is_padding"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_dflash_propose_dummy_skip_attn_marks_all_query_rows_as_padding():
    speculator = object.__new__(DFlashSpeculator)
    speculator.num_speculative_steps = 5
    speculator.sample_from_anchor = False
    speculator.max_model_len = 32
    speculator.hidden_states = torch.zeros(16, 4)
    speculator.context_positions = torch.zeros(16, dtype=torch.int64)
    speculator.input_buffers = _make_input_buffers(max_num_tokens=16)
    speculator.draft_tokens = torch.zeros((MAX_NUM_REQS, 5), dtype=torch.int32)
    speculator.model = _DraftModel(torch.ones(16, 4))
    speculator._copy_request_inputs = lambda *args, **kwargs: None
    speculator._prepare_eplb_forward = lambda *args, **kwargs: None

    captured = {}

    def fake_generate_draft(*args, **kwargs):
        captured["mask"] = speculator.input_buffers.is_padding[
            : kwargs["num_query_per_req"]
        ].clone()

    speculator._generate_draft = fake_generate_draft

    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
    )
    scalar = torch.tensor([0], dtype=torch.int32)

    speculator.propose(
        input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.ones(1, 4),
        aux_hidden_states=None,
        num_sampled=scalar,
        num_rejected=scalar,
        last_sampled=scalar,
        next_prefill_tokens=scalar,
        temperature=torch.ones(1),
        seeds=scalar,
        num_tokens_across_dp=None,
        dummy_run=True,
        skip_attn_for_dummy_run=True,
    )

    assert captured["mask"].tolist() == [True] * 6


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
