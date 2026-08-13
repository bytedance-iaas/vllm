# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V2 worker warmup gating and speculative warmup inputs."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
)
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
        last_completed_num_spec_tokens_to_schedule=0,
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


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_warmup_kernels_cleans_generic_before_parallel_draft_and_restores_k(
    monkeypatch, method
):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner(method)
    active_req_ids: set[str] = set()
    events: list[tuple[str, tuple[str, ...] | int]] = []
    sample_calls: list = []

    def execute(output):
        new_req_ids = tuple(req.req_id for req in output.scheduled_new_reqs)
        finished_req_ids = tuple(sorted(output.finished_req_ids))

        for req_id in output.scheduled_cached_reqs.req_ids:
            assert req_id in active_req_ids

        if finished_req_ids:
            events.append(("finish", finished_req_ids))
            for req_id in finished_req_ids:
                active_req_ids.remove(req_id)

        if new_req_ids:
            events.append(("new", new_req_ids))
            for req_id in new_req_ids:
                active_req_ids.add(req_id)

        assert len(active_req_ids) <= runner.max_num_reqs

        if output.num_spec_tokens_to_schedule:
            runner.last_completed_num_spec_tokens_to_schedule = (
                output.num_spec_tokens_to_schedule
            )
            events.append(("spec_k", output.num_spec_tokens_to_schedule))

    warmup_kernels(
        runner,
        worker_execute_model=execute,
        worker_sample_tokens=lambda grammar: _record_sample_call(sample_calls, grammar),
    )

    generic_req_ids = ("_warmup_0_", "_warmup_1_")
    parallel_req_ids = ("_warmup_parallel_draft_full_k_0_",)

    assert runner._disabled_states == [True, False]
    assert active_req_ids == set()
    assert runner.last_completed_num_spec_tokens_to_schedule == 0
    assert len(sample_calls) == 4
    assert sample_calls[0] is not None
    assert sample_calls[1:] == [None, None, None]
    assert events.index(("finish", generic_req_ids)) < events.index(
        ("new", parallel_req_ids)
    )
    assert events.index(("new", parallel_req_ids)) < events.index(
        ("finish", parallel_req_ids)
    )
    assert ("spec_k", 5) in events


def test_warmup_kernels_restores_k_and_cleans_parallel_req_on_failure(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dflash")
    active_req_ids: set[str] = set()
    cleanup_events: list[tuple[str, ...]] = []

    def execute(output):
        finished_req_ids = tuple(sorted(output.finished_req_ids))
        if finished_req_ids:
            cleanup_events.append(finished_req_ids)
            for req_id in finished_req_ids:
                active_req_ids.remove(req_id)

        for req_id in output.scheduled_cached_reqs.req_ids:
            assert req_id in active_req_ids

        for req in output.scheduled_new_reqs:
            active_req_ids.add(req.req_id)

        assert len(active_req_ids) <= runner.max_num_reqs

        if output.num_spec_tokens_to_schedule:
            runner.last_completed_num_spec_tokens_to_schedule = (
                output.num_spec_tokens_to_schedule
            )
            raise RuntimeError("parallel full-K warmup failed")

    with pytest.raises(RuntimeError, match="parallel full-K warmup failed"):
        warmup_kernels(
            runner,
            worker_execute_model=execute,
            worker_sample_tokens=lambda grammar: None,
        )

    assert runner._disabled_states == [True, False]
    assert runner.last_completed_num_spec_tokens_to_schedule == 0
    assert active_req_ids == set()
    assert cleanup_events[-1] == ("_warmup_parallel_draft_full_k_0_",)


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
            0,
            2,
            1,
            0,
        ]
        assert len(sample_calls) == 4
        assert sample_calls[0] is not None
        assert sample_calls[1:] == [None, None, None]

        assert execute_calls[2].finished_req_ids == {"_warmup_0_", "_warmup_1_"}

        parallel_decode = execute_calls[4]
        assert parallel_decode.total_num_scheduled_tokens == 1
        assert parallel_decode.num_scheduled_tokens == {
            "_warmup_parallel_draft_full_k_0_": 1
        }
        assert parallel_decode.num_spec_tokens_to_schedule == 5
        assert parallel_decode.scheduled_cached_reqs.req_ids == [
            "_warmup_parallel_draft_full_k_0_"
        ]
        assert parallel_decode.scheduled_cached_reqs.num_output_tokens == [1]
        assert execute_calls[5].finished_req_ids == {"_warmup_parallel_draft_full_k_0_"}


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


def test_deepseek_v4_mixed_warmup_skipped_for_pcp():
    class _Backend:
        @staticmethod
        def get_name():
            return "DEEPSEEK_SPARSE_SWA"

    runner = SimpleNamespace(
        is_pooling_model=False,
        attn_groups=[[SimpleNamespace(backend=_Backend())]],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
        ),
        _dummy_run=_fail,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
    )

    assert deepseek_v4_sparse_mla_attention_warmup(worker) is None
