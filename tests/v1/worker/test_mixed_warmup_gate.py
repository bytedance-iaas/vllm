# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V2 worker warmup gating and speculative warmup inputs."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup, warmup_kernels


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def _make_warmup_runner(method: str | None) -> SimpleNamespace:
    disabled_states: list[bool] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        num_speculative_steps=5,
        decode_query_len=6,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8))
            ],
            num_blocks=32,
        ),
        speculative_config=(None if method is None else SimpleNamespace(method=method)),
        model_state=SimpleNamespace(
            num_new_sampled_tokens_per_step=1, max_encoder_len=0
        ),
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 128),
        kv_connector=SimpleNamespace(
            set_disabled=lambda disabled: disabled_states.append(disabled)
        ),
    )
    runner._disabled_states = disabled_states
    return runner


def _record_execute_call(execute_calls: list, output) -> None:
    execute_calls.append(copy.deepcopy(output))


def _record_sample_call(sample_calls: list, grammar) -> None:
    sample_calls.append(grammar)


def test_warmup_kernels_runs_parallel_draft_full_k_for_dflash_dspark(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    for method in ("dflash", "dspark"):
        runner = _make_warmup_runner(method)
        execute_calls: list = []
        sample_calls: list = []

        warmup_kernels(
            runner,
            worker_execute_model=lambda output,
            execute_calls=execute_calls: _record_execute_call(execute_calls, output),
            worker_sample_tokens=lambda grammar,
            sample_calls=sample_calls: _record_sample_call(sample_calls, grammar),
        )

        assert runner._disabled_states == [True, False]
        assert [call.total_num_scheduled_tokens for call in execute_calls] == [
            14,
            12,
            2,
            1,
            0,
        ]
        assert len(sample_calls) == 4
        assert sample_calls[0] is not None
        assert sample_calls[1:] == [None, None, None]

        parallel_decode = execute_calls[3]
        assert parallel_decode.total_num_scheduled_tokens == 1
        assert parallel_decode.num_scheduled_tokens == {
            "_warmup_parallel_draft_full_k_0_": 1
        }
        assert parallel_decode.num_spec_tokens_to_schedule == 5
        assert parallel_decode.scheduled_cached_reqs.req_ids == [
            "_warmup_parallel_draft_full_k_0_"
        ]
        assert parallel_decode.scheduled_cached_reqs.num_output_tokens == [1]


def test_warmup_kernels_skips_parallel_draft_full_k_for_unrelated_method(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("eagle3")
    execute_calls: list = []
    sample_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(execute_calls, output),
        worker_sample_tokens=lambda grammar: _record_sample_call(sample_calls, grammar),
    )

    assert runner._disabled_states == [True, False]
    assert [call.total_num_scheduled_tokens for call in execute_calls] == [14, 12, 0]
    assert len(sample_calls) == 2
    assert sample_calls[0] is not None
    assert sample_calls[1] is None
    assert all(call.total_num_scheduled_tokens != 1 for call in execute_calls)
