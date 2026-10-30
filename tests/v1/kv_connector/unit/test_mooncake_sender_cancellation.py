# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sender lifecycle regressions without a GPU or Mooncake data plane."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferResponseStatus,
    SendBlockMeta,
)


@pytest.mark.asyncio
async def test_cancelled_native_send_keeps_kv_pinned_until_terminal():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._sender_semaphore = asyncio.Semaphore(1)
    worker._build_transfer_params = AsyncMock(return_value=([1], [2], [64], [], None))
    loop = asyncio.get_running_loop()
    native_transfer = loop.create_future()
    submitted = asyncio.Event()

    def submit(*args):
        submitted.set()
        return native_transfer

    worker.sender_loop = SimpleNamespace(run_in_executor=submit)
    worker._sender_executor = None
    worker._send_blocks = None
    worker.reqs_need_send = {}
    worker.finished_sending_reqs = set()
    meta = SendBlockMeta("p", "transfer", [[1]], asyncio.Event(), need_send=1)
    worker.reqs_need_send[meta.transfer_id] = meta
    peer = SimpleNamespace(remote_hostname="consumer", remote_port=1234)

    async def run():
        meta.sending += 1
        try:
            await worker._send_ready_kv(
                [("d", meta)], peer, [], [], MooncakeXferResponseStatus.FINISH
            )
        finally:
            meta.sending -= 1

    task = asyncio.create_task(run())
    await asyncio.wait_for(submitted.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not native_transfer.cancelled()
    assert not task.done()
    assert meta.sending == 1
    assert worker._sender_semaphore.locked()
    task.cancel()
    await asyncio.sleep(0)
    assert not native_transfer.cancelled()
    assert not task.done()
    assert meta.sending == 1
    assert worker._sender_semaphore.locked()
    meta.expire_time = time.perf_counter() - 1
    assert not await worker.fetch_finished_sending_reqs()
    assert worker.reqs_need_send[meta.transfer_id] is meta
    native_transfer.set_result(0)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert meta.sending == 0
    assert not worker._sender_semaphore.locked()
    assert not worker.finished_sending_reqs


@pytest.mark.asyncio
async def test_cancelled_unready_prefill_wakes_without_a_transfer_slot():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.is_sender_worker = True
    meta = SendBlockMeta("p", "transfer", [], asyncio.Event())
    worker.reqs_need_send = {meta.transfer_id: meta}
    waiter = asyncio.create_task(meta.ready.wait())
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_not_processed.add(meta.transfer_id)
    await worker.record_send_reqs(metadata)
    await asyncio.wait_for(waiter, 1)
    assert meta.transfer_id not in worker.reqs_need_send
    assert meta.ready.is_set()
