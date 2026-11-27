# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP v2 PCIe adapter using the Magic kernels from DeepEP PR #821."""

from dataclasses import dataclass

import deep_ep
import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PCIeDispatchHandle:
    num_source_tokens: int
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    num_recv_tokens_per_expert_list: list[int]


class DeepEPV2PCIeBuffer:
    """Expose the graph-safe ElasticBuffer operations used by Decode."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        use_fp8_dispatch: bool,
        allow_hybrid_mode: bool,
        prefer_overlap_with_compute: bool,
        allow_multiple_reduction: bool,
        explicitly_destroy: bool,
    ) -> None:
        import deep_ep_magic_C
        from deep_ep.utils.comm import get_nccl_comm_handle

        if allow_hybrid_mode:
            raise ValueError("DeepEP v2 PCIe requires direct communication.")
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.hidden = hidden
        self.num_topk = num_topk
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.nccl_handle = get_nccl_comm_handle(group)
        # The pinned Magic runtime divides each workspace into two stages and
        # one slot per peer. Budget for a BF16 payload, routing IDs and weights,
        # a 128-byte slot header, and the worst-case per-row bulk alignment.
        row_bytes = hidden * 2 + num_topk * 8
        aligned_row_bytes = (row_bytes + 127) // 128 * 128
        rank_slot_bytes = 128 + num_max_tokens_per_rank * aligned_row_bytes
        workspace_bytes = max(256 << 20, 2 * self.world_size * rank_slot_bytes)
        self.runtime = deep_ep_magic_C.MagicBuffer(
            self.rank,
            self.world_size,
            self.nccl_handle.get(),
            num_max_tokens_per_rank,
            workspace_bytes,
        )
        # Match MagicBuffer's collective lifecycle before peers can send.
        dist.barrier(group=group)
        logger.info(
            "DeepEP v2 SM120 PCIe Magic buffer initialized, rank=%d, world=%d, "
            "capacity=%d, hidden=%d, workspace_bytes=%d.",
            self.rank,
            self.world_size,
            num_max_tokens_per_rank,
            hidden,
            workspace_bytes,
        )

    def get_theoretical_num_sms(self, **kwargs) -> int:
        # Default single-node Magic grid has 16 send and world_size receive CTAs.
        return 16 + self.world_size

    def dispatch(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        num_max_tokens_per_rank: int | None = None,
        do_expand: bool = False,
        do_cpu_sync: bool = False,
        async_with_compute_stream: bool = False,
    ):
        if do_expand or do_cpu_sync:
            raise ValueError("DeepEP v2 PCIe currently requires the Decode layout.")
        if num_experts % self.world_size:
            raise ValueError("The number of experts must divide the EP group.")
        payload, scales = x if isinstance(x, tuple) else (x, None)
        capacity = self.world_size * (
            num_max_tokens_per_rank or self.num_max_tokens_per_rank
        )
        recv_payload = payload.new_empty((capacity, payload.shape[1]))
        recv_scales = (
            scales.new_empty((capacity, scales.shape[1]))
            if scales is not None
            else None
        )
        recv_ids = torch.full(
            (capacity, self.num_topk),
            -1,
            dtype=torch.int32,
            device=payload.device,
        )
        recv_weights = topk_weights.new_zeros((capacity, self.num_topk))
        psum = torch.zeros(self.world_size, dtype=torch.int32, device=payload.device)
        event = self.runtime.dispatch(
            payload,
            scales,
            topk_idx.to(torch.int32).contiguous(),
            topk_weights.contiguous(),
            recv_payload,
            recv_scales,
            recv_ids,
            recv_weights,
            num_experts // self.world_size,
            async_with_compute_stream,
            psum,
        )
        overlap = deep_ep.EventOverlap(event)
        # The ID conversion consumes the received routing table.
        if event is not None:
            overlap.current_stream_wait()
        offset = self.rank * (num_experts // self.world_size)
        local_ids = torch.where(
            (recv_ids >= offset) & (recv_ids < offset + num_experts // self.world_size),
            recv_ids - offset,
            -1,
        ).to(torch.int64)
        handle = PCIeDispatchHandle(payload.shape[0], psum, [])
        recv_x = (
            (recv_payload, recv_scales) if recv_scales is not None else recv_payload
        )
        return recv_x, local_ids, recv_weights, handle, deep_ep.EventOverlap()

    def combine(
        self,
        x: torch.Tensor,
        handle: PCIeDispatchHandle,
        topk_weights=None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        output = x.new_empty((handle.num_source_tokens, self.hidden))
        event = self.runtime.combine(
            x.contiguous(), output, async_with_compute_stream, 0
        )
        return output, None, deep_ep.EventOverlap(event)

    def destroy(self) -> None:
        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None
        self.nccl_handle = None
