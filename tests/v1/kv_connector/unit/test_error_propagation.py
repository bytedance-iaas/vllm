# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def fail_scheduler():
    """scheduler with kv_load_failure_policy='fail'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "fail"
    return create_scheduler(vllm_config)


def test_error_propagation_sync_load(fail_scheduler: Scheduler):
    """test invalid_block_ids with fail policy -> FINISHED_ERROR (sync load)"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    scheduler_output = fail_scheduler.schedule()

    assert len(fail_scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert fail_scheduler.connector.get_num_new_matched_tokens.call_count == 1

    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(fail_scheduler.running) == 0


def test_error_propagation_async_load(fail_scheduler: Scheduler):
    """test invalid_block_ids with fail policy -> FINISHED_ERROR (async load)"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, True)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    scheduler_output = fail_scheduler.schedule()

    assert len(fail_scheduler.skipped_waiting) == 1
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == num_external_computed_tokens

    (req_block_ids,) = fail_scheduler.kv_cache_manager.get_block_ids(request.request_id)
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving=set(),
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(fail_scheduler.waiting) == 0
    assert len(fail_scheduler.skipped_waiting) == 0


@pytest.fixture(params=[False, True], ids=["sync-scheduler", "async-scheduler"])
def v41_fail_scheduler(request):
    specs = [
        SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
        ),
        MLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
        MLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=144,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
        CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=1024,
            head_size_v=0,
            dtype=torch.float32,
        ),
    ]
    groups = [
        KVCacheGroupSpec([name], spec)
        for name, spec in zip(("swa", "kv", "indexer", "ring"), specs)
    ]
    config = create_vllm_config(
        block_size=32,
        max_num_batched_tokens=512,
        kv_connector="MooncakeConnector",
        disable_hybrid_kv_cache_manager=False,
    )
    config.scheduler_config.async_scheduling = request.param
    scheduler = create_scheduler(
        config,
        kv_cache_config=KVCacheConfig(
            num_blocks=10000, kv_cache_tensors=[], kv_cache_groups=groups
        ),
    )
    scheduler.connector = Mock(spec=MooncakeConnector)
    scheduler.connector.request_finished_all_groups.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    return scheduler


@pytest.mark.parametrize("async_load", [False, True])
@pytest.mark.parametrize("failed_group", [0, 1, 2, 3, None])
def test_v41_fail_policy_isolates_failed_request(
    v41_fail_scheduler: Scheduler, async_load: bool, failed_group: int | None
):
    scheduler = v41_fail_scheduler
    requests = [create_request(num_tokens=320, block_size=32) for _ in range(2)]
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {req.request_id: 288 for req in requests}, async_load
        )
    )
    for req in requests:
        scheduler.add_request(req)
    scheduled = scheduler.schedule()
    victim, sibling = requests
    groups = scheduler.kv_cache_manager.get_block_ids(victim.request_id)
    assert len(groups) == 4
    assert len(groups[3]) == 1
    null_id = scheduler.kv_cache_manager.block_pool.null_block.block_id
    invalid_id = (
        null_id
        if failed_group is None
        else next(block_id for block_id in groups[failed_group] if block_id != null_id)
    )
    evict = scheduler.kv_cache_manager.evict_blocks = Mock(
        wraps=scheduler.kv_cache_manager.evict_blocks
    )
    outputs = scheduler.update_from_output(
        scheduled,
        create_model_runner_output(
            [] if async_load else requests,
            invalid_block_ids={invalid_id},
            finished_recving={req.request_id for req in requests}
            if async_load
            else None,
            use_eos=True,
        ),
    )
    if failed_group is None:
        if async_load:
            assert all(not batch.outputs for batch in outputs.values())
            assert victim.num_computed_tokens == 288
        else:
            assert victim.status == RequestStatus.FINISHED_STOPPED
        evict.assert_not_called()
    else:
        assert victim.status == RequestStatus.FINISHED_ERROR
        assert victim.request_id not in scheduler.requests
        errors = [
            output
            for batch in outputs.values()
            for output in batch.outputs
            if output.finish_reason == FinishReason.ERROR
        ]
        assert [output.request_id for output in errors] == [victim.request_id]
        assert not victim.output_token_ids
        if async_load:
            evict.assert_not_called()
        else:
            evict.assert_called_once_with(
                {
                    block_id
                    for group in groups
                    for block_id in group
                    if block_id != null_id
                }
            )
    assert sibling.status != RequestStatus.FINISHED_ERROR
    assert not scheduler.failed_recving_kv_req_ids
    if async_load:
        resumed = scheduler.schedule()
        if failed_group is not None:
            assert victim.request_id not in resumed.num_scheduled_tokens
        live = [req for req in requests if not req.is_finished()]
        scheduler.update_from_output(
            resumed, create_model_runner_output(live, use_eos=True)
        )
    assert sibling.is_finished()
    scheduler.schedule()
    assert not scheduler.requests
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    assert not scheduler.running
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.free_block_queue.num_free_blocks == pool.num_gpu_blocks - 1
    assert all(block.ref_cnt == 0 for block in pool.blocks if not block.is_null)


@pytest.mark.parametrize("policy", ["fail", "recompute"])
@pytest.mark.parametrize("failed_group", range(4))
def test_v41_failed_load_keeps_buffers_until_receive_finishes(
    v41_fail_scheduler: Scheduler, policy: str, failed_group: int
):
    scheduler = v41_fail_scheduler
    scheduler.recompute_kv_load_failures = policy == "recompute"
    request = create_request(num_tokens=320, block_size=32)
    scheduler.connector.get_num_new_matched_tokens.return_value = (288, True)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    groups = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.free_block_queue.num_free_blocks
    invalid_id = next(
        block_id
        for block_id in groups[failed_group]
        if block_id != pool.null_block.block_id
    )
    scheduler.update_from_output(
        scheduled,
        create_model_runner_output([], invalid_block_ids={invalid_id}),
    )
    assert pool.free_block_queue.num_free_blocks == free_before
    assert request.request_id in scheduler.requests
    assert request.num_computed_tokens == 0
    if policy == "fail":
        assert request.status == RequestStatus.FINISHED_ERROR
        assert not scheduler.failed_recving_kv_req_ids
    else:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert pool.free_block_queue.num_free_blocks == free_before
    scheduler.update_from_output(
        scheduler.schedule(),
        create_model_runner_output([], finished_recving={request.request_id}),
    )
    assert request.request_id not in scheduler.requests
    assert pool.free_block_queue.num_free_blocks == pool.num_gpu_blocks - 1
