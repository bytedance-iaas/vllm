# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.deepep_v2_pcie import DeepEPV2PCIeBuffer

rank = int(os.environ["LOCAL_RANK"])
torch.accelerator.set_device_index(rank)
dist.init_process_group(
    "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=90)
)
world = dist.get_world_size()
dist.all_reduce(torch.zeros(1, device="cuda"))
tokens, hidden, topk, experts = 8, 5120, 6, 384
buffer = DeepEPV2PCIeBuffer(
    dist.group.WORLD, 64, hidden, topk, False, False, False, False, True
)
base_ids = (
    (
        rank * 43
        + torch.arange(tokens, device="cuda")[:, None] * 11
        + torch.arange(topk, device="cuda")[None, :] * 61
    )
    % experts
).to(torch.int64)
weights = torch.full((tokens, topk), 1 / topk, device="cuda")
reference = torch.full((tokens, hidden), rank + 1, device="cuda", dtype=torch.bfloat16)


def run(payload, ids):
    recv, local_ids, recv_weights, handle, event = buffer.dispatch(
        payload, ids, weights, experts, num_max_tokens_per_rank=tokens
    )
    if isinstance(recv, tuple):
        data, recv_scales = recv
    else:
        data = recv
    total = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
    rows = torch.arange(data.shape[0], device="cuda") < total
    routed = data.to(torch.bfloat16) * torch.where(local_ids >= 0, recv_weights, 0).sum(
        1, keepdim=True
    )
    routed = torch.where(rows[:, None], routed, 0).to(torch.bfloat16)
    result, _, event = buffer.combine(routed, handle, async_with_compute_stream=True)
    if event.event is not None:
        event.current_stream_wait()
    return result


for mode in ("bf16", "mxfp8", "idle-rank"):
    ids = base_ids.clone()
    if mode == "idle-rank" and rank == world - 1:
        ids.fill_(-1)
    expected = (
        torch.zeros_like(reference)
        if mode == "idle-rank" and rank == world - 1
        else reference
    )
    scales = torch.full(
        (tokens, hidden // 128), 0x7F7F7F7F, device="cuda", dtype=torch.int32
    )
    payload = (
        (reference.to(torch.float8_e4m3fn), scales) if mode == "mxfp8" else reference
    )

    result = run(payload, ids)
    torch.accelerator.synchronize()
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.05)
    for _ in range(2):
        run(payload, ids)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(payload, ids)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0.01, atol=0.05)
    print("CASE_OK", rank, mode, flush=True)
    del graph
    dist.barrier()
buffer.destroy()
dist.destroy_process_group()
print("ALL_CASES_OK", rank, flush=True)
