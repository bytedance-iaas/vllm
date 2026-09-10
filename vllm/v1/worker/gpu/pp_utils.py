# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    # Draft proposals for the step this slot feeds, when spec decoding is on.
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


@dataclass
class PendingSend:
    """Per-step send buffers kept alive until the side-stream broadcast ends."""

    event: torch.cuda.Event
    tensors: tuple[torch.Tensor, ...]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )
        self.pending_sends: deque[PendingSend] = deque()

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1
        for slot in self.queue:
            if slot is not None:
                slot.need_sampled_mask[slot.idx_mapping_np == req_idx] = False

    def _drain_pending_sends(self) -> None:
        while self.pending_sends and self.pending_sends[0].event.query():
            self.pending_sends.popleft()

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Wait before inspecting or dropping the receive buffers. If all rows are
        # filtered out, returning before this wait can release tensors while the
        # side-stream NCCL broadcast is still writing into them.
        self.main_stream.wait_event(slot.event)

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            # A freed index may already belong to a new request.
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            # `receive()` has already optimistically advanced PP0
            # num_computed_tokens for this query width. The delayed sampled
            # postprocess should only subtract speculative rejections.
            query_start_loc=None,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        idx_mapping_np = input_batch.idx_mapping_np.copy()
        gen_at_receive_np = self.req_idx_gen_np[idx_mapping_np].copy()

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                self.max_num_reqs,
                self.max_sample_len,
                dtype=torch.int64,
                device=self.device,
            )
            combined = torch.empty(
                2, self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        sampled_tokens = sampled_tokens[:num_reqs]
        combined = combined[:, :num_reqs]
        if draft_tokens is not None:
            draft_tokens = draft_tokens[:num_reqs]
        num_sampled, num_rejected = combined.unbind(dim=0)
        if need_sampled_mask is None:
            # Keep no-sample receive buffers alive until the matching NCCL work
            # completes. `get_prev_sampled_outputs` will wait on the event and
            # then drop this all-filtered frame.
            need_sampled_mask = np.zeros(num_reqs, dtype=bool)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping.clone(),
            idx_mapping_np,
            need_sampled_mask.copy(),
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor | None = None,
    ) -> None:
        assert self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        self._drain_pending_sends()

        assert sampled_token_ids.dtype == torch.int64
        if self.num_speculative_steps > 0:
            assert draft_tokens is not None

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        num_reqs = input_batch.num_reqs
        send_tokens = sampled_token_ids.new_zeros(
            self.max_num_reqs, self.max_sample_len
        )
        combined = torch.zeros(
            2, self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        send_drafts = None
        if self.num_speculative_steps > 0:
            send_drafts = draft_tokens.new_zeros(
                self.max_num_reqs, self.num_speculative_steps
            )
        if need_sampled_mask is not None:
            width = sampled_token_ids.shape[-1]
            assert width <= self.max_sample_len
            send_tokens[:num_reqs, :width] = sampled_token_ids
            combined[0, :num_reqs] = num_sampled
            combined[1, :num_reqs] = num_rejected
            if send_drafts is not None:
                active_drafts = draft_tokens[input_batch.idx_mapping].contiguous()
                draft_width = active_drafts.shape[-1]
                assert draft_width <= self.num_speculative_steps
                send_drafts[:num_reqs, :draft_width] = active_drafts

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                send_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            if send_drafts is not None:
                torch.distributed.broadcast(
                    send_drafts, src=self.last_rank, group=self.broadcast_group
                )

            event = self.broadcast_stream.record_event()
            for tensor in (send_tokens, combined):
                tensor.record_stream(self.broadcast_stream)
            if send_drafts is not None:
                send_drafts.record_stream(self.broadcast_stream)
                self.pending_sends.append(
                    PendingSend(event, (send_tokens, combined, send_drafts))
                )
            else:
                self.pending_sends.append(PendingSend(event, (send_tokens, combined)))
        self._drain_pending_sends()
