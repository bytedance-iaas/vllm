# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.nvidia.retained_replay_fixture import (
    RetainedReplayFixtureController,
    TouchedStateRegistry,
    clone_owned,
)


@dataclass
class _OwnedTree:
    tensor: torch.Tensor
    array: np.ndarray
    children: dict[str, torch.Tensor]


def test_retained_fixture_clone_owns_mutable_leaves() -> None:
    source = _OwnedTree(
        tensor=torch.arange(4),
        array=np.arange(3),
        children={"value": torch.arange(2)},
    )

    retained = clone_owned(source)
    source.tensor.zero_()
    source.array.fill(0)
    source.children["value"].zero_()

    assert retained.tensor.tolist() == [0, 1, 2, 3]
    assert retained.array.tolist() == [0, 1, 2]
    assert retained.children["value"].tolist() == [0, 1]


def test_touched_registry_restores_epochs_without_restoring_source() -> None:
    rows = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    points = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    source = torch.arange(40, dtype=torch.uint8).reshape(10, 4)
    registry = TouchedStateRegistry()
    registry.add_rows("rows", rows, torch.tensor([3, 1, 3]))
    registry.add_points("points", points, torch.tensor([1, 7, 1]))
    registry.add_source("source", source, torch.tensor([2, 5]))

    rows[[1, 3]].add_(100)
    points.view(-1)[[1, 7]].add_(50)
    registry.capture_epoch("s1")
    s1_digest = registry.digest("s1")
    source_digest = registry.source_capture_digest()

    rows[[1, 3]].zero_()
    points.view(-1)[[1, 7]].zero_()
    registry.restore("s0")
    registry.verify("s0")
    assert rows[1].tolist() == [4, 5, 6, 7]
    assert rows[3].tolist() == [12, 13, 14, 15]
    assert points.view(-1)[[1, 7]].tolist() == [1, 7]

    registry.restore("s1")
    registry.verify("s1")
    assert registry.digest() == s1_digest
    assert registry.source_digest() == source_digest

    source[2].zero_()
    registry.restore("s0")
    assert registry.source_digest() != source_digest


def test_cache_registry_canonicalizes_aliased_storage_views() -> None:
    backing = torch.arange(80, dtype=torch.uint8)
    cache_a = torch.as_strided(
        backing, size=(4, 4, 2), stride=(16, 2, 1), storage_offset=0
    )
    cache_b = torch.as_strided(
        backing, size=(4, 2, 4), stride=(16, 4, 1), storage_offset=8
    )
    cache_c = torch.as_strided(
        backing.view(torch.int32),
        size=(4, 2, 1),
        stride=(4, 1, 1),
        storage_offset=2,
    )
    cache_groups = [
        SimpleNamespace(layer_names=["layer.a"]),
        SimpleNamespace(layer_names=["layer.b"]),
        SimpleNamespace(layer_names=["layer.c"]),
    ]
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=cache_groups),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.a": SimpleNamespace(kv_cache=cache_a),
                "layer.b": SimpleNamespace(kv_cache=cache_b),
                "layer.c": SimpleNamespace(kv_cache=cache_c),
            }
        ),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(
                gpu=torch.tensor([[2], [2], [1]], dtype=torch.int32)
            ),
            kernel_block_sizes=[4, 2, 2],
        ),
        speculator=SimpleNamespace(
            draft_kv_cache_group_ids=(0,),
            num_query_per_req=0,
        ),
        num_speculative_steps=1,
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = runner
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=3,
        max_query_len=0,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8], dtype=torch.int32),
    )

    intervals = controller._collect_cache_intervals(
        input_batch,
        (
            torch.tensor([[1, 2, 0]], dtype=torch.int32),
            torch.tensor([[1, 3, 0]], dtype=torch.int32),
            torch.tensor([[1, 0, 0]], dtype=torch.int32),
        ),
        (
            (
                SimpleNamespace(
                    attn_metadata={
                        "layer.a": SimpleNamespace(
                            slot_mapping=torch.tensor([4, -1, -1], dtype=torch.int64)
                        ),
                        "layer.b": SimpleNamespace(
                            slot_mapping=torch.tensor([2, -1, -1], dtype=torch.int64)
                        ),
                        "layer.c": SimpleNamespace(
                            slot_mapping=torch.tensor([2, -1, -1], dtype=torch.int64)
                        ),
                    }
                ),
                3,
            ),
        ),
    )

    assert len(intervals) == 1
    record = next(iter(intervals.values()))
    assert record.tensor.untyped_storage().data_ptr() == backing.data_ptr()
    assert record.destination == [(16, 40)]
    assert record.source == [(56, 64)]

    registry = TouchedStateRegistry()
    registry.add_intervals("destination", record.tensor, record.destination)
    registry.add_source_intervals("source", record.tensor, record.source)
    source_digest = registry.source_capture_digest()
    backing[16:40].add_(100)
    registry.capture_epoch("s1")
    backing[56:64].zero_()
    registry.restore("s0")
    registry.verify("s0")
    assert backing[16:40].tolist() == list(range(16, 40))
    assert backing[56:64].tolist() == [0] * 8
    registry.restore("s1")
    registry.verify("s1")
    assert backing[16:40].tolist() == list(range(116, 140))
    assert backing[56:64].tolist() == [0] * 8
    assert registry.source_digest() != source_digest


def test_cache_registry_retains_replay_window_by_logical_block_order() -> None:
    cache = torch.arange(48, dtype=torch.uint8).reshape(6, 4, 2)
    layer_name = "layer"
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name])]
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                layer_name: SimpleNamespace(kv_cache=cache),
            }
        ),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(gpu=torch.tensor([[3]], dtype=torch.int32)),
            kernel_block_sizes=[4],
        ),
        speculator=None,
        num_speculative_steps=0,
        model_state=SimpleNamespace(decoder_replay_layers=SimpleNamespace(window=8)),
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = runner
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=1,
        max_query_len=4,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([12], dtype=torch.int32),
    )

    intervals = controller._collect_cache_intervals(
        input_batch,
        (torch.tensor([[1, 3, 5, 0]], dtype=torch.int32),),
        (
            (
                SimpleNamespace(
                    attn_metadata={
                        layer_name: SimpleNamespace(
                            slot_mapping=torch.tensor([12], dtype=torch.int64)
                        )
                    }
                ),
                1,
            ),
        ),
    )

    record = next(iter(intervals.values()))
    assert record.destination == [(24, 32), (40, 48)]
    assert record.source == [(8, 16)]


def test_cache_registry_retains_pad_only_request_frontier() -> None:
    cache = torch.arange(64, dtype=torch.uint8).reshape(8, 4, 2)
    layer_name = "layer"
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name])]
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                layer_name: SimpleNamespace(kv_cache=cache),
            }
        ),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(gpu=torch.tensor([[3, 3]], dtype=torch.int32)),
            kernel_block_sizes=[4],
        ),
        speculator=None,
        num_speculative_steps=0,
        model_state=SimpleNamespace(decoder_replay_layers=SimpleNamespace(window=4)),
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = runner
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens_after_padding=2,
        max_query_len=4,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([9, 12], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([12, 12], dtype=torch.int32),
    )
    full_context = SimpleNamespace(
        attn_metadata={
            layer_name: SimpleNamespace(
                slot_mapping=torch.tensor([8, -1], dtype=torch.int64)
            )
        }
    )
    replay_context = SimpleNamespace(
        attn_metadata={
            layer_name: SimpleNamespace(
                slot_mapping=torch.tensor([-1, 16], dtype=torch.int64)
            )
        }
    )

    intervals = controller._collect_cache_intervals(
        input_batch,
        (
            torch.tensor(
                [
                    [1, 2, 5, 0],
                    [3, 4, 6, 0],
                ],
                dtype=torch.int32,
            ),
        ),
        (
            (full_context, 2),
            (replay_context, 2),
        ),
    )

    record = next(iter(intervals.values()))
    assert record.destination == [(16, 24), (32, 56)]
    assert record.source == [(8, 16), (24, 32)]


def test_cache_registry_retains_next_verification_across_block_boundary() -> None:
    cache = torch.arange(64, dtype=torch.uint8).reshape(8, 4, 2)
    layer_name = "layer"
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name])]
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                layer_name: SimpleNamespace(kv_cache=cache),
            }
        ),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(gpu=torch.tensor([[5]], dtype=torch.int32)),
            kernel_block_sizes=[4],
        ),
        speculator=None,
        num_speculative_steps=5,
        model_state=SimpleNamespace(decoder_replay_layers=SimpleNamespace(window=4)),
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = runner
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=1,
        max_query_len=1,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([15], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([15], dtype=torch.int32),
    )
    context = SimpleNamespace(
        attn_metadata={
            layer_name: SimpleNamespace(
                slot_mapping=torch.tensor([-1], dtype=torch.int64)
            )
        }
    )

    intervals = controller._collect_cache_intervals(
        input_batch,
        (torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.int32),),
        ((context, 1),),
    )

    record = next(iter(intervals.values()))
    assert record.destination == [(24, 48)]
    assert record.source == [(8, 24)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_interval_registry_restores_cpu_live_snapshot_to_cuda() -> None:
    with torch.inference_mode():
        tensor = torch.arange(64, dtype=torch.uint8, device="cuda")
        registry = TouchedStateRegistry()
        registry.add_intervals("destination", tensor, [(8, 24), (32, 40)])
        expected = tensor.clone()
        registry.capture_epoch("live", cpu=True)
        tensor.zero_()

    registry.restore("live")
    registry.verify("live")

    assert torch.equal(tensor[8:24], expected[8:24])
    assert torch.equal(tensor[32:40], expected[32:40])
    assert torch.count_nonzero(tensor[:8]) == 0
    assert torch.count_nonzero(tensor[24:32]) == 0
