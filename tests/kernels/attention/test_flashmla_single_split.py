# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 whole-request FP8 sparse FlashMLA reductions."""

import pytest
import torch

from vllm.v1.attention.backends.mla.flashmla_sparse_scheduler import (
    single_split_flashmla_schedule,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)


@pytest.mark.parametrize(
    "batch,queries,heads", [(1, 1, 64), (1, 9, 64), (16, 128, 64), (16, 9, 128)]
)
def test_single_split_request_coverage(batch, queries, heads):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    schedule, splits = single_split_flashmla_schedule(
        batch, queries, heads, 2048, sm_count, torch.device("cuda")
    )
    assert schedule.shape == (max(sm_count // queries // (heads // 64), 1), 8)
    assert splits.cpu().tolist() == list(range(batch + 1))
    covered: list[int] = []
    for (
        first,
        last,
        start,
        end,
        split,
        first_split,
        last_split,
        _,
    ) in schedule.cpu().tolist():
        assert start == 0 and end == 32
        assert split == first_split == last_split == 0
        if first < batch:
            covered.extend(range(first, last + 1))
    assert covered == list(range(batch))


@pytest.mark.parametrize(
    "batch,queries,heads,topk,sm_count",
    [
        (0, 9, 64, 2048, 78),
        (1, 0, 64, 2048, 78),
        (1, 9, 8, 2048, 78),
        (1, 9, 64, 2047, 78),
        (1, 9, 64, 2048, 0),
    ],
)
def test_single_split_rejects_invalid_native_geometry(
    batch, queries, heads, topk, sm_count
):
    with pytest.raises(ValueError):
        single_split_flashmla_schedule(
            batch, queries, heads, topk, sm_count, torch.device("cuda")
        )


@pytest.mark.parametrize("batch,queries", [(1, 9), (3, 4), (16, 16)])
def test_fp8_single_split_request_partition_and_graph(batch, queries):
    torch.manual_seed(173)
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # Real 656-byte fp8_ds_mla records: four FP32 scales, 512 FP8 NoPE
    # values and 64 BF16 RoPE values. Power-of-two scales are exact.
    cache = torch.empty((32, 64, 1, 656), dtype=torch.uint8, device=device)
    rows = cache.view(-1, 656)
    rows[:, :512].copy_(
        torch.randn((2048, 512), device=device)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    rows[:, 512:528].view(torch.float32).fill_(1 / 16)
    rows[:, 528:].copy_(
        torch.randn((2048, 64), device=device, dtype=torch.bfloat16).view(torch.uint8)
    )
    q = torch.randn((batch, queries, 64, 576), device=device, dtype=torch.bfloat16) / 8
    indices = (
        torch.arange(2048, dtype=torch.int32, device=device)
        .expand(batch, queries, -1)
        .contiguous()
    )

    def metadata(b, t):
        schedule, splits = single_split_flashmla_schedule(
            b, t, 64, 2048, sm_count, device
        )
        m = get_mla_metadata()[0]
        m.tile_scheduler_metadata = schedule
        m.num_splits = splits
        return m

    def run(query, selected, m):
        return flash_mla_with_kvcache(
            query,
            cache,
            None,
            None,
            512,
            m,
            softmax_scale=1 / 16,
            is_fp8_kvcache=True,
            indices=selected,
        )

    m = metadata(batch, queries)
    actual, actual_lse = run(q, indices, m)
    # Each request alone and a larger one-split query launch must retain
    # exactly the same reduction, including LSE.
    for i in range(batch):
        large_q = (
            q[i : i + 1].expand(1, 128, 64, 576)
            if queries == 1
            else torch.cat(
                (q[i : i + 1], q[i : i + 1, :1].expand(1, 128 - queries, 64, 576)),
                dim=1,
            )
        )
        large_idx = indices[i : i + 1, :1].expand(1, 128, -1).contiguous()
        expected, expected_lse = run(
            large_q.contiguous(), large_idx, get_mla_metadata()[0]
        )
        assert torch.equal(actual[i], expected[0, :queries])
        assert torch.equal(actual_lse[i], expected_lse[0, :, :queries])

    # Capture the real native call with the reusable schedule, change Q,
    # then restore Q and require the original output on graph replay.
    original_q = q.clone()
    run(q, indices, m)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = run(q, indices, m)
    q.zero_()
    graph.replay()
    assert not torch.equal(graph_out, actual)
    q.copy_(original_q)
    graph.replay()
    assert torch.equal(graph_out, actual)
    assert torch.equal(graph_lse, actual_lse)
