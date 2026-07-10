# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (
    C128ExportSlotPool,
    C128ImportSlotPool,
    build_online_c128_state_descriptors,
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


def test_build_online_c128_state_descriptors_uses_slot_and_layer_offsets():
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

    plan = build_online_c128_state_descriptors(
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


def test_build_online_c128_state_descriptors_maps_producer_to_consumer_layers():
    pool = C128ExportSlotPool(
        capacity=2,
        num_layers=2,
        row_width=5,
        device="cpu",
    )
    export_slot = 1
    import_slot = 7
    row_width_bytes = 5 * pool.buffer.element_size()
    remote_import_base_addr = 0x1000
    remote_slot_bytes = 4 * row_width_bytes

    plan = build_online_c128_state_descriptors(
        export_pool=pool,
        export_slot=export_slot,
        remote_import_base_addr=remote_import_base_addr,
        remote_slot_bytes=remote_slot_bytes,
        remote_import_slot=import_slot,
        num_layers=2,
        row_width_bytes=row_width_bytes,
        layer_pos_pairs=[(0, 1), (1, 3)],
    )

    src_base = pool.base_addr + export_slot * pool.slot_bytes
    dst_base = remote_import_base_addr + import_slot * remote_slot_bytes
    assert plan.src_ptrs == [
        src_base,
        src_base + row_width_bytes,
    ]
    assert plan.dst_ptrs == [
        dst_base + row_width_bytes,
        dst_base + 3 * row_width_bytes,
    ]
    assert plan.lengths == [row_width_bytes] * 2


def test_append_online_c128_state_descriptors_maps_layer_indices_across_pp():
    worker = object.__new__(MooncakeConnectorWorker)
    worker._c128_export_pool = C128ExportSlotPool(
        capacity=2,
        num_layers=2,
        row_width=5,
        device="cpu",
    )
    worker._c128_num_layers = 2
    worker._c128_state_row_bytes = 5 * worker._c128_export_pool.buffer.element_size()
    worker._c128_layer_indices = [1, 3]
    worker.tp_size = 1
    worker.tp_rank = 0
    worker._producer_cache_is_replicated = lambda: True
    worker._c128_export_events = {}

    send_meta = SendBlockMeta(
        p_req_id="p",
        transfer_id="t",
        local_block_ids=[[1]],
        ready=None,
        c128_export_slot=1,
    )
    agent_meta = MooncakeXferMetadata(
        remote_hostname="host",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("t", [[1]])},
        kv_caches_base_addr=[],
        block_lens=[],
        c128_import_base_addr=0x1000,
        c128_import_slot_bytes=4 * worker._c128_state_row_bytes,
        c128_num_layers=4,
        c128_state_row_bytes=worker._c128_state_row_bytes,
        c128_layer_indices=[0, 1, 2, 3],
        c128_req_import_slot={"d": 7},
        c128_req_needs_partial={"d": True},
    )
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    lengths: list[int] = []

    err_msg = worker._append_online_c128_state_descriptors(
        ready_reqs=[("d", send_meta)],
        agent_meta=agent_meta,
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        lengths=lengths,
        err_reqs=[],
        err_msg=None,
        state_transfer_events=[],
    )

    row_bytes = worker._c128_state_row_bytes
    src_base = worker._c128_export_pool.base_addr + worker._c128_export_pool.slot_bytes
    dst_base = agent_meta.c128_import_base_addr + 7 * agent_meta.c128_import_slot_bytes
    assert err_msg is None
    assert src_ptrs == [src_base, src_base + row_bytes]
    assert dst_ptrs == [dst_base + row_bytes, dst_base + 3 * row_bytes]
    assert lengths == [row_bytes, row_bytes]


def test_append_online_c128_state_descriptors_errors_on_producer_superset_layers():
    worker = object.__new__(MooncakeConnectorWorker)
    worker._c128_export_pool = C128ExportSlotPool(
        capacity=1,
        num_layers=4,
        row_width=5,
        device="cpu",
    )
    worker._c128_num_layers = 4
    worker._c128_state_row_bytes = 5 * worker._c128_export_pool.buffer.element_size()
    worker._c128_layer_indices = [0, 1, 2, 3]
    worker.tp_size = 1
    worker.tp_rank = 0
    worker._producer_cache_is_replicated = lambda: True

    send_meta = SendBlockMeta(
        p_req_id="p",
        transfer_id="t",
        local_block_ids=[[1]],
        ready=None,
        c128_export_slot=0,
    )
    agent_meta = MooncakeXferMetadata(
        remote_hostname="host",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("t", [[1]])},
        kv_caches_base_addr=[],
        block_lens=[],
        c128_import_base_addr=0x1000,
        c128_import_slot_bytes=2 * worker._c128_state_row_bytes,
        c128_num_layers=2,
        c128_state_row_bytes=worker._c128_state_row_bytes,
        c128_layer_indices=[1, 3],
        c128_req_import_slot={"d": 0},
        c128_req_needs_partial={"d": True},
    )
    err_reqs: list[str] = []

    err_msg = worker._append_online_c128_state_descriptors(
        ready_reqs=[("d", send_meta)],
        agent_meta=agent_meta,
        src_ptrs=[],
        dst_ptrs=[],
        lengths=[],
        err_reqs=err_reqs,
        err_msg=None,
        state_transfer_events=[],
    )

    assert "strictly contains the consumer" in err_msg
    assert err_reqs == ["d"]


def test_append_online_c128_state_descriptors_fails_on_missing_consumer_layer():
    worker = object.__new__(MooncakeConnectorWorker)
    worker._c128_export_pool = C128ExportSlotPool(
        capacity=1,
        num_layers=2,
        row_width=5,
        device="cpu",
    )
    worker._c128_num_layers = 2
    worker._c128_state_row_bytes = 5 * worker._c128_export_pool.buffer.element_size()
    worker._c128_layer_indices = [1, 3]
    worker.tp_size = 1
    worker.tp_rank = 0
    worker._producer_cache_is_replicated = lambda: True

    send_meta = SendBlockMeta(
        p_req_id="p",
        transfer_id="t",
        local_block_ids=[[1]],
        ready=None,
        c128_export_slot=0,
    )
    agent_meta = MooncakeXferMetadata(
        remote_hostname="host",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("t", [[1]])},
        kv_caches_base_addr=[],
        block_lens=[],
        c128_import_base_addr=0x1000,
        c128_import_slot_bytes=2 * worker._c128_state_row_bytes,
        c128_num_layers=2,
        c128_state_row_bytes=worker._c128_state_row_bytes,
        c128_layer_indices=[0, 1],
        c128_req_import_slot={"d": 0},
        c128_req_needs_partial={"d": True},
    )
    err_reqs: list[str] = []

    err_msg = worker._append_online_c128_state_descriptors(
        ready_reqs=[("d", send_meta)],
        agent_meta=agent_meta,
        src_ptrs=[],
        dst_ptrs=[],
        lengths=[],
        err_reqs=err_reqs,
        err_msg=None,
        state_transfer_events=[],
    )

    assert "missing_layer_indices=[3]" in err_msg
    assert err_reqs == ["d"]


def test_bind_c128_state_index_does_not_require_export_pool_registered():
    worker = object.__new__(MooncakeConnectorWorker)
    worker._online_c128_state_transfer_enabled = True
    worker._c128_export_pool = None
    worker._c128_req_state_indices = {}

    worker.bind_c128_state_index("req-a", 3)

    assert worker._c128_req_state_indices == {"req-a": 3}


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
