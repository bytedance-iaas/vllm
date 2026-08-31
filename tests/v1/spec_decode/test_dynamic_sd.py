# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging
from types import SimpleNamespace

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.utils import replace
from vllm.v1.core.sched.scheduler import Scheduler, _DynamicSDBudgetPolicy
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def _model_output(scheduler: Scheduler, output, sampled: list[list[int]]) -> None:
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={request_id: i for i, request_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler._dynamic_sd is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_dsd_batch_size_override():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 4, 3), (5, 16, 1)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    scheduler.set_dynamic_sd_batch_size_override(8)
    output = _add_requests_and_schedule(scheduler, 2)

    assert len(output.num_scheduled_tokens) == 2
    assert output.num_spec_tokens_to_schedule == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_dynamic_sd_pads_first_decode_with_verification_k(async_scheduling):
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=160,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=[(1, 16, 7), (17, 60, 5)],
        speculative_method="ngram_gpu" if async_scheduling else None,
        async_scheduling=async_scheduling,
        enable_prefix_caching=True,
        block_size=16,
    )
    r1, r2 = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=16,
    )

    scheduler.add_request(r1)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, output, [[100]])

    scheduler.update_draft_token_ids(
        DraftTokenIds([r1.request_id], [[1, 2, 3, 4, 5]])
    )
    scheduler.add_request(r2)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[r1.request_id] == 6
    assert output.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3, 4, 5]
    assert output.num_scheduled_tokens[r2.request_id] == 6
    assert output.scheduled_spec_decode_tokens[r2.request_id] == [-1] * 5
    # Proposal K is selected independently from the current verification K.
    assert output.num_spec_tokens_to_schedule == 7


@pytest.mark.parametrize(
    ("draft_widths", "prefill_scheduled", "expected_k"),
    [
        ([5, 5], False, 5),
        ([5, 3], False, None),
        ([0, 0], False, None),
        ([5, 5], True, None),
    ],
)
def test_dynamic_sd_padding_requires_uniform_running_decode(
    draft_widths,
    prefill_scheduled,
    expected_k,
):
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 7), (17, 60, 5)],
        runtime_num_speculative_tokens=7,
    )
    running = [
        SimpleNamespace(request_id=f"req-{i}", is_prefill_chunk=False)
        for i in range(len(draft_widths))
    ]
    scheduled_tokens = {
        request.request_id: draft_width + 1
        for request, draft_width in zip(running, draft_widths)
    }
    scheduled_drafts = {
        request.request_id: list(range(draft_width))
        for request, draft_width in zip(running, draft_widths)
        if draft_width
    }

    running_k = scheduler._get_uniform_running_decode_k(
        running,
        scheduled_tokens,
        scheduled_drafts,
        prefill_scheduled,
    )

    assert running_k == expected_k
    assert scheduler._get_waiting_decode_padding(
        1, running_k, running, prefill_scheduled
    ) == ((1 + expected_k, expected_k) if expected_k else (1, 0))


def test_static_spec_decode_does_not_pad_diffusion_batches():
    scheduler = create_scheduler(
        num_speculative_tokens=3,
    )
    scheduler.num_sampled_tokens_per_step = 0
    running = [SimpleNamespace(request_id="req-0", is_prefill_chunk=False)]

    assert scheduler._get_waiting_decode_padding(
        1,
        running_decode_k=None,
        scheduled_running_reqs=running,
        prefill_scheduled=False,
    ) == (1, 0)


def test_dynamic_sd_clears_padding_when_later_constraint_changes_width():
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=160,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=[(1, 16, 7), (17, 60, 5)],
        enable_prefix_caching=True,
        block_size=16,
    )
    r1, r2 = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=16,
    )

    scheduler.add_request(r1)
    output = scheduler.schedule()
    _model_output(scheduler, output, [[100]])
    scheduler.update_draft_token_ids(
        DraftTokenIds([r1.request_id], [[1, 2, 3, 4, 5]])
    )

    original_split = scheduler._mamba_block_aligned_split

    def fake_split(request, num_new_tokens, *args):
        if request.request_id == r2.request_id and num_new_tokens == 6:
            return 4
        return original_split(request, num_new_tokens, *args)

    scheduler.need_mamba_block_aligned_split = True
    scheduler._mamba_block_aligned_split = fake_split  # type: ignore[method-assign]

    scheduler.add_request(r2)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[r1.request_id] == 6
    assert output.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3, 4, 5]
    assert output.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in output.scheduled_spec_decode_tokens


def _set_parallel_drafting_budget_reclaim(
    scheduler: Scheduler,
    *,
    auto_derived: bool,
    reclaim_parallel_drafting_slots: bool = True,
) -> None:
    scheduler.max_num_scheduled_tokens = 2048 - 6 * 96
    scheduler.scheduler_config.max_num_scheduled_tokens_auto_derived = auto_derived
    assert scheduler._dynamic_sd is not None
    can_reclaim_token_budget = (
        auto_derived
        and reclaim_parallel_drafting_slots
        and scheduler.max_num_scheduled_tokens == 2048 - 6 * 96
    )
    scheduler._dynamic_sd.budget_policy = (
        _DynamicSDBudgetPolicy(max_num_batched_tokens=2048, max_num_seqs=96)
        if can_reclaim_token_budget
        else None
    )


def test_scheduler_reclaims_dynamic_sd_auto_token_budget_with_override():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 7), (17, 60, 5), (61, 96, 3)],
        max_num_seqs=96,
        max_num_batched_tokens=2048,
        runtime_num_speculative_tokens=7,
    )
    _set_parallel_drafting_budget_reclaim(scheduler, auto_derived=True)

    scheduler.set_dynamic_sd_batch_size_override(61)
    output = _add_requests_and_schedule(scheduler, 1, num_tokens=4000)

    assert output.num_spec_tokens_to_schedule == 3
    assert output.total_num_scheduled_tokens == 2048 - 2 * 96


@pytest.mark.parametrize(
    ("explicit_budget", "expected_scheduled"),
    [(2048 - 6 * 96, 2048 - 6 * 96), (1024, 1024)],
)
def test_scheduler_does_not_reclaim_explicit_token_budget(
    explicit_budget,
    expected_scheduled,
):
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 7), (17, 60, 5), (61, 96, 3)],
        max_num_seqs=96,
        max_num_batched_tokens=2048,
        runtime_num_speculative_tokens=7,
    )
    _set_parallel_drafting_budget_reclaim(scheduler, auto_derived=False)
    scheduler.max_num_scheduled_tokens = explicit_budget

    scheduler.set_dynamic_sd_batch_size_override(61)
    output = _add_requests_and_schedule(scheduler, 1, num_tokens=4000)

    assert output.num_spec_tokens_to_schedule == 3
    assert output.total_num_scheduled_tokens == expected_scheduled


def test_scheduler_dynamic_sd_budget_reclaim_handles_kmax_and_kzero():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 7), (17, 60, 0)],
        max_num_seqs=96,
        max_num_batched_tokens=2048,
        runtime_num_speculative_tokens=7,
    )
    _set_parallel_drafting_budget_reclaim(scheduler, auto_derived=True)

    assert scheduler._get_effective_max_num_scheduled_tokens(7) == 2048 - 6 * 96
    assert scheduler._get_effective_max_num_scheduled_tokens(0) == 2048


def test_dynamic_sd_token_budget_provenance_survives_config_replace():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 7), (17, 60, 5), (61, 96, 3)],
        max_num_seqs=96,
        max_num_batched_tokens=2048,
        runtime_num_speculative_tokens=7,
    )
    vllm_config = scheduler.vllm_config
    assert vllm_config.speculative_config is not None
    vllm_config.speculative_config.parallel_drafting = True
    auto_derived_budget = 2048 - 6 * 96
    vllm_config.scheduler_config.max_num_scheduled_tokens = auto_derived_budget

    vllm_config.scheduler_config.max_num_scheduled_tokens_auto_derived = True
    replaced_config = replace(vllm_config, cache_config=vllm_config.cache_config)
    assert replaced_config.scheduler_config.max_num_scheduled_tokens == (
        auto_derived_budget
    )
    assert replaced_config.scheduler_config.max_num_scheduled_tokens_auto_derived

    vllm_config.scheduler_config.max_num_scheduled_tokens_auto_derived = False
    replaced_config = replace(vllm_config, cache_config=vllm_config.cache_config)
    assert replaced_config.scheduler_config.max_num_scheduled_tokens == (
        auto_derived_budget
    )
    assert not replaced_config.scheduler_config.max_num_scheduled_tokens_auto_derived


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler._dynamic_sd is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert scheduler._dynamic_sd is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler._dynamic_sd is not None
    assert len(scheduler._dynamic_sd.lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3
