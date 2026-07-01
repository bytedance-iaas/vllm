# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (
    C128ExportSlotPool,
    C128ImportSlotPool,
    build_c128_aux_descriptors,
    reset_bank0,
    restore_bank0_from_slot,
    snapshot_bank0_to_slot,
)


def test_c128_slot_pool_reuses_existing_transfer_and_releases_slot():
    pool = C128ImportSlotPool(
        capacity=1,
        num_layers=2,
        row_width=4,
        device="cpu",
    )

    slot = pool.acquire("transfer-a", timeout=0.0)

    assert pool.acquire("transfer-a", timeout=0.0) == slot
    with pytest.raises(TimeoutError, match="C128 slot pool exhausted"):
        pool.acquire("transfer-b", timeout=0.0)

    pool.release("transfer-a")

    assert pool.acquire("transfer-b", timeout=0.0) == slot


def test_build_c128_aux_descriptors_uses_slot_and_layer_offsets():
    pool = C128ExportSlotPool(
        capacity=2,
        num_layers=3,
        row_width=5,
        device="cpu",
    )
    export_slot = 1
    import_slot = 7
    row_width_bytes = 5 * pool.buffer.element_size()
    remote_import_base_addr = 0x1000
    remote_slot_bytes = 3 * row_width_bytes

    plan = build_c128_aux_descriptors(
        export_pool=pool,
        export_slot=export_slot,
        remote_import_base_addr=remote_import_base_addr,
        remote_slot_bytes=remote_slot_bytes,
        remote_import_slot=import_slot,
        num_layers=3,
        row_width_bytes=row_width_bytes,
    )

    src_base = pool.base_addr + export_slot * pool.slot_bytes
    dst_base = remote_import_base_addr + import_slot * remote_slot_bytes
    assert plan.src_ptrs == [
        src_base + layer_pos * row_width_bytes for layer_pos in range(3)
    ]
    assert plan.dst_ptrs == [
        dst_base + layer_pos * row_width_bytes for layer_pos in range(3)
    ]
    assert plan.lengths == [row_width_bytes] * 3


def test_snapshot_restore_and_reset_bank0_helpers_round_trip_cpu_state():
    states = [
        SimpleNamespace(
            state=torch.arange(24, dtype=torch.float32).reshape(4, 6),
            head_dim=2,
        ),
        SimpleNamespace(
            state=(torch.arange(24, dtype=torch.float32).reshape(4, 6) + 100),
            head_dim=2,
        ),
    ]
    export_pool = C128ExportSlotPool(
        capacity=1,
        num_layers=len(states),
        row_width=6,
        device="cpu",
    )
    import_pool = C128ImportSlotPool(
        capacity=1,
        num_layers=len(states),
        row_width=6,
        device="cpu",
    )

    snapshot_bank0_to_slot(states, export_pool, slot=0, req_state_idx=2)
    import_pool.buffer[0].copy_(export_pool.buffer[0])
    for state in states:
        state.state[1].zero_()

    restore_bank0_from_slot(states, import_pool, slot=0, req_state_idx=1)

    assert torch.equal(states[0].state[1], torch.tensor([12, 13, 14, 15, 16, 17.0]))
    assert torch.equal(
        states[1].state[1],
        torch.tensor([112, 113, 114, 115, 116, 117.0]),
    )

    reset_bank0(states, req_state_idx=1)

    for state in states:
        assert torch.isneginf(state.state[1, : state.head_dim]).all()
        assert torch.equal(
            state.state[1, state.head_dim :],
            torch.zeros(4, dtype=torch.float32),
        )
