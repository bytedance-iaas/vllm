# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm.v1.utils import ConstantList
from vllm.v1.worker.gpu_input_batch import InputBatch

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_async_spec_decode_does_not_split_draft_group(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_model="ngram_gpu",
        max_num_batched_tokens=6,
        max_num_seqs=2,
    )
    requests = create_requests(num_requests=2, num_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    req_ids = [request.request_id for request in requests]
    scheduler.update_draft_token_ids(
        DraftTokenIds(req_ids=req_ids, draft_token_ids=[[1, 2, 3], [4, 5, 6]])
    )

    decode_output = scheduler.schedule()

    first_req_id, second_req_id = req_ids
    assert decode_output.num_scheduled_tokens[first_req_id] == 4
    assert decode_output.num_scheduled_tokens[second_req_id] == 1
    assert decode_output.scheduled_spec_decode_tokens[first_req_id] == [1, 2, 3]
    assert second_req_id not in decode_output.scheduled_spec_decode_tokens


def test_async_spec_decode_rejects_group_truncated_by_model_length(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_model="ngram_gpu",
        max_model_len=3,
        max_num_batched_tokens=8,
        max_num_seqs=1,
    )
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            req_ids=[request.request_id],
            draft_token_ids=[[1, 2, 3]],
        )
    )

    decode_output = scheduler.schedule()

    assert request.request_id not in decode_output.num_scheduled_tokens


def test_update_draft_token_ids_trims_at_first_negative_token(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_model="ngram_gpu",
    )
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )

    scheduler.update_draft_token_ids(
        DraftTokenIds(
            req_ids=[request.request_id],
            draft_token_ids=[[11, -1, 12]],
        )
    )

    assert request.spec_token_ids == [11]

    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={request.request_id: [-1, -1, -1, -1]},
        num_invalid_spec_tokens={},
    )
    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds(
            req_ids=[request.request_id],
            draft_token_ids=[[11, -1, 12]],
        ),
        scheduler_output,
    )
    assert scheduler_output.scheduled_spec_decode_tokens[request.request_id] == [
        11,
        -1,
        -1,
        -1,
    ]
    assert scheduler_output.num_invalid_spec_tokens == {request.request_id: 3}


def test_batch_change_ignores_out_of_range_previous_rows():
    num_computed_tokens = torch.tensor([10, 20, 30], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([7, 8, 9], dtype=torch.int32)
    prev_positions = torch.tensor([-1, 3, 1], dtype=torch.int64)
    valid_sampled_token_count = torch.tensor([2, 4, 6], dtype=torch.int32)
    prev_num_draft_tokens = torch.tensor([1, 1, 1], dtype=torch.int32)
    cpu_num_computed_tokens = torch.tensor([100, 200, 300], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
    )

    assert num_computed_tokens.tolist() == [100, 200, 24]
    assert num_accepted_tokens.tolist() == [7, 8, 4]


def test_async_input_updates_ignore_stale_previous_rows():
    input_batch = object.__new__(InputBatch)
    input_batch._req_ids = ["req-0"]
    input_batch.prev_req_id_to_index = {"req-0": 1}
    input_batch.sampling_metadata = SimpleNamespace(
        output_token_ids=[[-1]],
        spec_token_ids=[[9]],
    )
    input_batch.sampled_token_ids_cpu = torch.tensor([[7]])
    input_batch.async_copy_ready_event = Mock()

    input_batch.update_async_output_token_ids()
    input_batch.update_async_spec_token_ids([[8]])

    assert input_batch.sampling_metadata.output_token_ids == [[-1]]
    assert input_batch.sampling_metadata.spec_token_ids == [[9]]


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_preempt():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert (
        sched_output.num_scheduled_tokens[req1.request_id]
        == num_prompt_tokens % BLOCK_SIZE
    )

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert (
                    sched_output.num_scheduled_tokens[added_req.request_id]
                    == num_prompt_tokens % BLOCK_SIZE
                )

    assert scheduler.get_num_unfinished_requests() == 0


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(
            requests[i].output_token_ids
        )
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (
            req.num_prompt_tokens % BLOCK_SIZE
        )


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


def test_no_placeholder_underflow_on_discarded_spec_frame():
    num_spec = 5
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    req.status = RequestStatus.RUNNING

    req.num_output_placeholders = 1
    req.async_tokens_to_discard = num_spec
    computed_before = req.num_computed_tokens

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: num_spec + 1},
        total_num_scheduled_tokens=num_spec + 1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={req.request_id: [10] * num_spec},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert req.num_output_placeholders == 1
    assert req.num_computed_tokens == computed_before
    assert req.async_tokens_to_discard == num_spec - 1
    assert req.status == RequestStatus.RUNNING
