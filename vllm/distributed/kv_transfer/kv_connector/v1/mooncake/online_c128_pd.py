# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake PD transfer helpers for C128 online state."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class _C128SlotPool:
    """Fixed-capacity transfer_id-keyed fp32 staging slots."""

    def __init__(
        self,
        capacity: int,
        num_layers: int,
        row_width: int,
        device: torch.device | str,
    ):
        self.capacity = capacity
        self.num_layers = num_layers
        self.row_width = row_width
        self.buffer = torch.empty(
            (capacity, num_layers, row_width),
            dtype=torch.float32,
            device=device,
        )
        self._lock = threading.Condition()
        self._free: list[int] = list(range(capacity))
        self._by_transfer: dict[str, int] = {}

    @property
    def base_addr(self) -> int:
        return self.buffer.data_ptr()

    @property
    def total_bytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()

    @property
    def slot_bytes(self) -> int:
        return self.num_layers * self.row_width * self.buffer.element_size()

    def slot_offset_bytes(self, slot: int) -> int:
        return slot * self.slot_bytes

    def acquire(self, transfer_id: str, timeout: float | None = None) -> int:
        """Reserve a slot for ``transfer_id``, blocking if the pool is full.

        Blocking forms backpressure so committed state is never dropped.
        """
        with self._lock:
            existing = self._by_transfer.get(transfer_id)
            if existing is not None:
                return existing
            while not self._free:
                if not self._lock.wait(timeout=timeout):
                    raise TimeoutError(
                        "C128 slot pool exhausted; no slot freed within "
                        f"{timeout}s for transfer {transfer_id}."
                    )
            slot = self._free.pop()
            self._by_transfer[transfer_id] = slot
            return slot

    def get(self, transfer_id: str) -> int | None:
        with self._lock:
            return self._by_transfer.get(transfer_id)

    def release(self, transfer_id: str) -> None:
        with self._lock:
            slot = self._by_transfer.pop(transfer_id, None)
            if slot is None:
                return
            self._free.append(slot)
            self._lock.notify()


# Kept as named subclasses for clarity at call sites / logging.
class C128ExportSlotPool(_C128SlotPool):
    """P-side RDMA source pool (snapshot of committed bank0)."""


class C128ImportSlotPool(_C128SlotPool):
    """D-side RDMA destination pool (staged remote bank0 awaiting admission)."""


def snapshot_bank0_to_slot(
    online_states: list,
    export_pool: C128ExportSlotPool,
    slot: int,
    req_state_idx: int,
) -> None:
    """Copy a request's committed bank0 rows (all layers) into an export slot.

    Must run before the live request slot is recycled on the P side.
    """
    for layer_pos, state in enumerate(online_states):
        bank0_row = state.state[req_state_idx]  # bank0 = first max_num_reqs rows
        export_pool.buffer[slot, layer_pos].copy_(bank0_row)


def restore_bank0_from_slot(
    online_states: list,
    import_pool: C128ImportSlotPool,
    slot: int,
    req_state_idx: int,
) -> None:
    """D-side: copy a staged import slot (RDMA-written by P) into live bank0."""
    for layer_pos, state in enumerate(online_states):
        state.state[req_state_idx].copy_(import_pool.buffer[slot, layer_pos])


def reset_bank0(online_states: list, req_state_idx: int) -> None:
    """Reset live bank0 rows to the online-softmax identity (128-aligned prompt
    needs no partial carry)."""
    for state in online_states:
        row = state.state[req_state_idx]
        head_dim = state.head_dim
        row[:head_dim] = float("-inf")
        row[head_dim:] = 0.0


@dataclass
class C128StateTransferPlan:
    """Source/destination pointer descriptors for one request's state transfer.

    Appended to the KV block descriptors so they ride the same
    ``batch_transfer_sync_write`` call.
    """

    src_ptrs: list[int] = field(default_factory=list)
    dst_ptrs: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)


def build_online_c128_state_descriptors(
    export_pool: C128ExportSlotPool,
    export_slot: int,
    remote_import_base_addr: int,
    remote_slot_bytes: int,
    remote_import_slot: int,
    num_layers: int,
    row_width_bytes: int,
    layer_pos_pairs: list[tuple[int, int]] | None = None,
) -> C128StateTransferPlan:
    """Build per-layer src/dst/len descriptors for one request's bank0 transfer.

    Source: the P export slot's per-layer rows.
    Destination: the D import slot's per-layer rows (NOT a live bank0 row; the
    runner copies the staged slot into bank0 after the request is admitted).
    ``layer_pos_pairs`` maps producer-local layer positions to
    consumer-local layer positions; identity mapping is used when omitted.
    """
    plan = C128StateTransferPlan()
    src_base = export_pool.base_addr + export_pool.slot_offset_bytes(export_slot)
    dst_base = remote_import_base_addr + remote_import_slot * remote_slot_bytes
    pairs = (
        layer_pos_pairs
        if layer_pos_pairs is not None
        else [(layer_pos, layer_pos) for layer_pos in range(num_layers)]
    )
    for src_layer_pos, dst_layer_pos in pairs:
        plan.src_ptrs.append(src_base + src_layer_pos * row_width_bytes)
        plan.dst_ptrs.append(dst_base + dst_layer_pos * row_width_bytes)
        plan.lengths.append(row_width_bytes)
    return plan
