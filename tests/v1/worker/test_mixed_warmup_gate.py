# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V2 worker warmup gating and speculative warmup inputs."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu import warmup as warmup_module
from vllm.v1.worker.gpu.warmup import (
    _run_parallel_draft_full_k_warmup,
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)


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


def _make_warmup_runner(
    method: str | None,
    *,
    block_sizes: tuple[int, ...] = (8,),
    is_last_pp_rank: bool = True,
) -> SimpleNamespace:
    disabled_states: list[bool] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        num_speculative_steps=5,
        decode_query_len=6,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size))
                for size in block_sizes
            ],
            num_blocks=64,
        ),
        kv_block_zeroer=None,
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(method=method, num_drafter_query_tokens=6)
        ),
        model_state=SimpleNamespace(
            num_new_sampled_tokens_per_step=1, max_encoder_len=0
        ),
        last_completed_num_spec_tokens_to_schedule=0,
        is_encoder_decoder=False,
        is_last_pp_rank=is_last_pp_rank,
        model_config=SimpleNamespace(get_vocab_size=lambda: 128),
        kv_connector=SimpleNamespace(
            set_disabled=lambda disabled: disabled_states.append(disabled)
        ),
    )
    runner._disabled_states = disabled_states
    return runner


def _record_execute_call(execute_calls: list, output) -> None:
    execute_calls.append(copy.deepcopy(output))


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_warmup_cleans_generic_before_parallel_draft_and_restores_k(
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
            active_req_ids.difference_update(finished_req_ids)
        if new_req_ids:
            events.append(("new", new_req_ids))
            active_req_ids.update(new_req_ids)
        assert len(active_req_ids) <= runner.max_num_reqs
        if output.num_spec_tokens_to_schedule:
            runner.last_completed_num_spec_tokens_to_schedule = (
                output.num_spec_tokens_to_schedule
            )
            events.append(("spec_k", output.num_spec_tokens_to_schedule))

    warmup_kernels(
        runner,
        worker_execute_model=execute,
        worker_sample_tokens=sample_calls.append,
    )

    generic_req_ids = ("_warmup_0_", "_warmup_1_")
    parallel_req_ids = ("_warmup_parallel_draft_full_k_0_",)
    assert runner._disabled_states == [True, False]
    assert active_req_ids == set()
    assert runner.last_completed_num_spec_tokens_to_schedule == 0
    assert len(sample_calls) == 8
    assert sample_calls[0] is not None
    assert sample_calls[1:] == [None] * 7
    assert events.index(("finish", generic_req_ids)) < events.index(
        ("new", parallel_req_ids)
    )
    assert events.index(("new", parallel_req_ids)) < events.index(
        ("finish", parallel_req_ids)
    )
    assert ("spec_k", 5) in events


def test_warmup_runs_expected_generic_and_parallel_shapes(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dflash")
    execute_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=lambda grammar: None,
    )

    assert [call.total_num_scheduled_tokens for call in execute_calls] == [
        14,
        12,
        7,
        2,
        6,
        1,
        0,
        2,
        1,
        0,
    ]
    assert execute_calls[6].finished_req_ids == {"_warmup_0_", "_warmup_1_"}
    parallel_decode = execute_calls[8]
    assert parallel_decode.num_scheduled_tokens == {
        "_warmup_parallel_draft_full_k_0_": 1
    }
    assert parallel_decode.num_spec_tokens_to_schedule == 5
    assert execute_calls[9].finished_req_ids == {
        "_warmup_parallel_draft_full_k_0_"
    }


def test_parallel_draft_warmup_allocates_each_kv_group():
    runner = _make_warmup_runner("dspark", block_sizes=(2, 4))
    execute_calls: list = []

    req_id = _run_parallel_draft_full_k_warmup(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=lambda grammar: None,
    )

    assert req_id == "_warmup_parallel_draft_full_k_0_"
    assert execute_calls[0].scheduled_new_reqs[0].block_ids == (
        [1, 2, 3, 4],
        [5, 6],
    )
    assert execute_calls[0].num_common_prefix_blocks == [0, 0]
    assert execute_calls[1].scheduled_cached_reqs.new_block_ids == [([7], [8])]
    assert execute_calls[1].num_common_prefix_blocks == [0, 0]


def test_parallel_draft_warmup_reserves_mamba_align_blocks():
    runner = _make_warmup_runner("dflash")
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(
            kv_cache_spec=MambaSpec(
                block_size=4,
                shapes=((2, 8),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=2,
            )
        )
    ]
    execute_calls: list = []

    _run_parallel_draft_full_k_warmup(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=lambda grammar: None,
    )

    assert execute_calls[0].scheduled_new_reqs[0].block_ids == ([1, 2, 3, 4],)
    assert execute_calls[1].scheduled_cached_reqs.new_block_ids == [([5],)]


def test_warmup_restores_state_after_parallel_draft_failure(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dflash")
    active_req_ids: set[str] = set()
    cleanup_events: list[tuple[str, ...]] = []

    def execute(output):
        finished_req_ids = tuple(sorted(output.finished_req_ids))
        if finished_req_ids:
            cleanup_events.append(finished_req_ids)
            active_req_ids.difference_update(finished_req_ids)
        for req_id in output.scheduled_cached_reqs.req_ids:
            assert req_id in active_req_ids
        active_req_ids.update(req.req_id for req in output.scheduled_new_reqs)
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


def test_warmup_preserves_primary_error_when_cleanup_also_fails(
    monkeypatch, caplog
):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dflash")

    def execute(output):
        if output.finished_req_ids == {"_warmup_parallel_draft_full_k_0_"}:
            raise RuntimeError("cleanup failed")
        if output.num_spec_tokens_to_schedule:
            raise RuntimeError("primary warmup failure")

    with pytest.raises(RuntimeError, match="primary warmup failure"):
        warmup_kernels(
            runner,
            worker_execute_model=execute,
            worker_sample_tokens=lambda grammar: None,
        )

    assert runner._disabled_states == [True, False]
    assert runner.last_completed_num_spec_tokens_to_schedule == 0
    assert "Failed to clean up parallel-draft warmup requests" in caplog.text


def test_warmup_skips_parallel_draft_for_unrelated_method(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("eagle3")
    execute_calls: list = []
    sample_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=sample_calls.append,
    )

    assert runner._disabled_states == [True, False]
    assert [call.total_num_scheduled_tokens for call in execute_calls] == [
        14,
        12,
        7,
        2,
        6,
        1,
        0,
    ]
    assert len(sample_calls) == 6
    assert all(
        not any(
            req.req_id.startswith("_warmup_parallel_draft")
            for req in call.scheduled_new_reqs
        )
        for call in execute_calls
    )


def test_warmup_skips_parallel_draft_when_runtime_k_is_zero(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dflash")
    runner.num_speculative_steps = 0
    execute_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=lambda grammar: None,
    )

    assert all(
        not any(
            req.req_id.startswith("_warmup_parallel_draft")
            for req in call.scheduled_new_reqs
        )
        for call in execute_calls
    )


def test_warmup_preserves_pooling_and_kv_zeroer_paths(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    verified_models: list[object] = []

    class FakePoolingParams:
        def __init__(self, task):
            self.task = task
            self.skip_reading_prefix_cache = None

        def verify(self, model_config):
            verified_models.append(model_config)

    monkeypatch.setattr(warmup_module, "PoolingParams", FakePoolingParams)

    runner = _make_warmup_runner("dflash")
    runner.is_pooling_model = True
    runner.num_speculative_steps = 0
    runner.decode_query_len = 1
    runner.get_supported_tasks = lambda: ("embed",)
    runner.model_config.get_pooling_task = lambda tasks: "embed"
    zeroed: list[int] = []
    runner.kv_block_zeroer = SimpleNamespace(warmup=zeroed.append)
    execute_calls: list = []
    sample_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=sample_calls.append,
    )

    assert zeroed == [64]
    assert verified_models == [runner.model_config]
    assert [call.total_num_scheduled_tokens for call in execute_calls] == [4, 0]
    assert sample_calls == []
    assert all(
        not any(
            req.req_id.startswith("_warmup_parallel_draft")
            for req in call.scheduled_new_reqs
        )
        for call in execute_calls
    )


def test_warmup_non_last_pp_rank_keeps_callback_order(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_warmup_runner("dspark", is_last_pp_rank=False)
    execute_calls: list = []
    sample_calls: list = []

    warmup_kernels(
        runner,
        worker_execute_model=lambda output: _record_execute_call(
            execute_calls, output
        ),
        worker_sample_tokens=sample_calls.append,
    )

    assert len(execute_calls) == 10
    assert sample_calls == [None] * 8
