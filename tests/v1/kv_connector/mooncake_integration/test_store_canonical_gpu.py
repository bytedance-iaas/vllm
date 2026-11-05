# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU descriptor probe; no model weights or remote Store required."""

import os
import uuid

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.dsv41_layout import (
    Binding,
    Component,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_canonical_gpu_descriptor_round_trip():
    cupy = pytest.importorskip("cupy")
    offsets = torch.arange(2048, device="cuda", dtype=torch.int64)
    data = (offsets // 128 * 17 + offsets % 127).to(torch.uint8)
    expected = data.clone()
    binding = Binding(
        Component("layer_8/main_kv", 64, 128, 2, "uint8"),
        0,
        data.data_ptr(),
        256,
        8,
        tensor=data,
    )
    regions = binding.regions(0, 128, [2, 5])
    host = torch.empty(256, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.current_stream()
    for i, (address, size) in enumerate(regions):
        cupy.cuda.runtime.memcpyAsync(
            host.data_ptr() + i * 128, address, size, 2, stream.cuda_stream
        )
    stream.synchronize()
    for bid in (2, 5):
        data[bid * 256 : bid * 256 + 128].zero_()
    for i, (address, size) in enumerate(regions):
        cupy.cuda.runtime.memcpyAsync(
            address, host.data_ptr() + i * 128, size, 1, stream.cuda_stream
        )
    stream.synchronize()
    assert torch.equal(data, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv("VLLM_RUN_CANONICAL_STORE_TEST") != "1",
    reason="Requires CUDA, an isolated DRAM Store and explicit opt-in",
)
def test_canonical_real_store_round_trip_into_relay_pages():
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.dsv41_layout import (  # noqa: E501
        Manifest,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.dsv41_scheduler import (  # noqa: E501
        Job,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.dsv41_worker import (  # noqa: E501
        Dsv41StoreWorker,
    )
    from vllm.distributed.mooncake_store import (
        MooncakeStoreConfig,
        setup_mooncake_store,
    )
    from vllm.utils.network_utils import get_ip

    config = MooncakeStoreConfig.load_from_config()
    assert not config.enable_offload
    store = MooncakeDistributedStore()
    source = torch.randint(0, 256, (2048,), device="cuda", dtype=torch.uint8)
    destination = torch.full_like(source, 165)
    expected = destination.clone()
    component = Component("layer_8/main_kv", 64, 128, 2, "uint8")
    manifest = Manifest(
        f"canonical-gpu-probe-{uuid.uuid4()}", "synthetic-v1", (component,), 64
    )
    producer = Dsv41StoreWorker.__new__(Dsv41StoreWorker)
    producer.manifest, producer.writer = manifest, True
    producer.bindings = [Binding(component, 0, source.data_ptr(), 256, 8)]
    receiver = Dsv41StoreWorker.__new__(Dsv41StoreWorker)
    receiver.manifest = manifest
    receiver.bindings = [
        Binding(component, 1, destination.data_ptr(), 256, 8, writer=False)
    ]
    job = Job("probe", ([2, 5], [3, 6]), (b"a", b"b"), 0, 128)
    for src, dst in ((2, 3), (5, 6)):
        expected[dst * 256 : dst * 256 + 128].copy_(source[src * 256 : src * 256 + 128])
    try:
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        for tensor in (source, destination):
            assert store.register_buffer(tensor.data_ptr(), tensor.numel()) == 0
        torch.accelerator.synchronize()
        keys, addresses, sizes = producer._batch(job, saving=True)
        assert all(
            r >= 0
            for r in store.batch_put_from_multi_buffers(
                keys, addresses, sizes, ReplicateConfig()
            )
        )
        assert list(store.batch_is_exist(keys)) == [1, 1]
        load_keys, addresses, sizes = receiver._batch(job, saving=False)
        assert load_keys == keys
        assert all(
            r >= 0 for r in store.batch_get_into_multi_buffers(keys, addresses, sizes)
        )
        torch.accelerator.synchronize()
        assert torch.equal(destination, expected)
    finally:
        # Unique keys are left to the isolated test Store's eviction policy.
        store.close()
