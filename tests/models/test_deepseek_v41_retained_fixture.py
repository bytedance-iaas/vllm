# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.replayssm.dsv41_retained_control_compare import _floating_metrics
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext
from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
from vllm.models.deepseek_v41.nvidia.retained_replay_fixture import (
    RetainedReplayFixtureController,
    RetainedReplayFixtureError,
    TouchedStateRegistry,
    _speculative_config_identity,
    canonical_decision_reference,
    clone_owned,
    typed_tree_snapshot,
    write_typed_run_bundle,
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


def test_clone_owned_preserves_aliases_and_resets_flashmla_holder(
    monkeypatch,
) -> None:
    import vllm.v1.attention.ops.flashmla as flashmla

    FlashMLASchedMeta = type("FlashMLASchedMeta", (), {})
    source = FlashMLASchedMeta()
    source.tile_scheduler_metadata = torch.tensor([1])
    source.num_splits = torch.tensor([2])
    source.have_initialized = True

    def new_holder():
        holder = FlashMLASchedMeta()
        holder.tile_scheduler_metadata = None
        holder.num_splits = None
        holder.have_initialized = False
        return holder, None

    monkeypatch.setattr(flashmla, "get_mla_metadata", new_holder)
    retained = clone_owned({"first": source, "second": source})

    assert retained["first"] is retained["second"]
    assert retained["first"] is not source
    assert retained["first"].tile_scheduler_metadata.tolist() == [1]
    assert retained["first"].num_splits.tolist() == [2]
    assert retained["first"].have_initialized is True


def test_decision_reference_ignores_unused_sample_tail() -> None:
    sampled = torch.tensor([[1, 2, 99], [3, 98, 97]], dtype=torch.int64)
    reference = canonical_decision_reference(
        sampled,
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
    )

    sampled[:, 2].zero_()
    assert [tokens.tolist() for tokens in reference["sampled_token_ids"]] == [
        [1, 2],
        [3],
    ]


def test_typed_registry_keeps_kv_bytes_and_sampling_parameters_exact() -> None:
    computed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    temperature = torch.tensor([0.7], dtype=torch.float32)
    storage = torch.arange(16, dtype=torch.uint8)
    scatter = torch.tensor([[float("-inf"), 1.0]], dtype=torch.float32)
    registry = TouchedStateRegistry()
    registry.add_rows(
        "acceptance:features",
        computed,
        torch.tensor([0, 1]),
        meaningful_count=1,
    )
    registry.add_rows("sampler:temperature", temperature, torch.tensor([0]))
    registry.add_intervals("kv-dst:test", storage, [(2, 6)])
    registry.add_rows(
        "speculator:_draft_scatter_buf",
        scatter,
        torch.tensor([0]),
        meaningful_count=1,
    )

    snapshot = registry.typed_snapshot("s1")
    comparison = {leaf["path"]: leaf["comparison"] for leaf in snapshot["leaves"]}

    assert comparison["registry/acceptance:features"] == "floating"
    assert comparison["registry/sampler:temperature"] == "exact"
    assert comparison["registry/kv-dst:test"] == "exact"
    assert comparison["registry/speculator:_draft_scatter_buf"] == "floating"
    features = next(
        leaf
        for leaf in snapshot["leaves"]
        if leaf["path"] == "registry/acceptance:features"
    )
    assert features["metadata"]["compare_prefix"] == 1


def test_numeric_comparator_accepts_matching_infinity_masks() -> None:
    metrics = _floating_metrics(
        torch.tensor([1.0, float("-inf"), 3.0]),
        torch.tensor([1.25, float("-inf"), 3.0]),
        denominator_floor=1e-12,
    )

    assert metrics["negative_infinities"] == 1
    assert metrics["max_abs"] == 0.25

    with pytest.raises(ValueError, match="negative-infinity masks differ"):
        _floating_metrics(
            torch.tensor([1.0, float("-inf")]),
            torch.tensor([1.0, 0.0]),
            denominator_floor=1e-12,
        )


def test_speculative_identity_includes_sampling_and_rejection_settings() -> None:
    @dataclass
    class SpecConfig:
        draft_sample_method: str
        rejection_sample_method: str
        dspark_draft_topk: int
        draft_model_config: object = None
        target_model_config: object = None
        draft_parallel_config: object = None
        target_parallel_config: object = None

    greedy = _speculative_config_identity(SpecConfig("greedy", "standard", 8))
    probabilistic = _speculative_config_identity(
        SpecConfig("probabilistic", "block", 16)
    )

    assert greedy != probabilistic
    assert probabilistic["draft_sample_method"] == "probabilistic"
    assert probabilistic["rejection_sample_method"] == "block"
    assert probabilistic["dspark_draft_topk"] == 16


def test_typed_bundle_writes_tensor_only_payload(tmp_path) -> None:
    snapshot = typed_tree_snapshot(
        "target",
        {
            "hidden": torch.tensor([[1.0, 2.0]]),
            "count": torch.tensor([2], dtype=torch.int32),
        },
        path="target",
    )
    result = write_typed_run_bundle(
        str(tmp_path),
        dp_rank=3,
        run_kind="a",
        ordinal=1,
        checkpoints={"target": snapshot},
        manifest={"run_id": "unit"},
    )

    manifest = __import__("json").loads(
        (tmp_path / "rank-03" / "a-001.json").read_text()
    )
    payload = torch.load(result["payload_path"], weights_only=True)
    assert manifest["complete"] is True
    assert all(isinstance(value, torch.Tensor) for value in payload.values())


def test_production_graph_observer_requires_actual_capture_replay() -> None:
    class Capture:
        num_graphs = 2
        num_eager_breaks = 1

        def __init__(self) -> None:
            self.calls = 0

        def replay(self) -> None:
            self.calls += 1

    class Manager:
        def __init__(self, descriptor, capture) -> None:
            self.capture = capture
            self.breakable_cg_runner = SimpleNamespace(
                entries={
                    descriptor: SimpleNamespace(capture=capture),
                }
            )

        def run(self, rows, inputs):
            del rows
            self.capture.replay()
            return inputs

    descriptor = BatchDescriptor(num_tokens=8)
    capture = Capture()
    manager = Manager(descriptor, capture)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=descriptor,
    )
    frozen = ReplayBatch(torch.tensor([0]), context, manager.run)
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller._graph_observations = {}

    working = controller.observed_production_replay_batch(frozen, "a-001")
    outputs = working.run_graph(torch.tensor([0]), (torch.tensor([3]),))
    observation = controller.graph_observation("a-001")

    assert outputs[0].tolist() == [3]
    assert capture.calls == 1
    assert observation["callback_calls"] == 1
    assert observation["capture_replay_calls"] == 1


def test_a_checkpoint_capture_defers_local_export_failure(monkeypatch) -> None:
    import vllm.models.deepseek_v41.nvidia.retained_replay_fixture as module

    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.state = "ready"
    controller.fixture = SimpleNamespace()
    controller._active_checkpoints = {}
    controller._active_checkpoint_error = None
    controller.boundary_error = None

    def fail_snapshot(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("export failed")

    monkeypatch.setattr(module, "typed_tree_snapshot", fail_snapshot)
    controller.capture_boundary(SimpleNamespace())

    assert controller.active_checkpoints_ready() is False
    assert controller._active_checkpoint_error == "RuntimeError: export failed"


def test_freeze_rejects_selected_front_graph() -> None:
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = SimpleNamespace(
        model_state=SimpleNamespace(
            decoder_replay_layers=SimpleNamespace(replay_batch=object()),
            _decoder_model=SimpleNamespace(decoder_replay_front_graph=object()),
        )
    )

    with pytest.raises(
        RetainedReplayFixtureError,
        match="does not support a selected front graph",
    ):
        controller.freeze_production_replay_batch()


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


def test_restore_rejects_expired_source_before_mutating_destination() -> None:
    destination = torch.arange(8, dtype=torch.int32)
    source = torch.arange(8, dtype=torch.int32)
    registry = TouchedStateRegistry()
    registry.add_rows("destination", destination, torch.tensor([1, 2]))
    registry.add_source("source", source, torch.tensor([4, 5]))
    source_digest = registry.source_capture_digest()
    destination[1:3].fill_(99)
    source[4].fill_(-1)

    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.fixture = SimpleNamespace(
        registry=registry,
        source_digest=source_digest,
    )

    with pytest.raises(
        RetainedReplayFixtureError,
        match="source KV lifetime expired before restore",
    ):
        controller.restore("s0")
    assert destination[[1, 2]].tolist() == [99, 99]


def test_target_decoder_model_unwraps_vl_language_model() -> None:
    decoder = object()
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(model=decoder),
        )
    )

    assert controller._target_decoder_model() is decoder


def test_boundary_capture_defers_clone_failure(monkeypatch) -> None:
    import vllm.models.deepseek_v41.nvidia.retained_replay_fixture as module

    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.state = "capturing"
    controller.fixture = SimpleNamespace(boundary=None)
    controller.boundary_error = None

    def fail_clone(value):
        del value
        raise RuntimeError("clone failed")

    monkeypatch.setattr(module, "clone_owned", fail_clone)
    controller.capture_boundary(SimpleNamespace())

    assert controller.fixture.boundary is None
    assert controller.boundary_error == "RuntimeError: clone failed"


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


@pytest.mark.parametrize("max_query_len", [4, None])
def test_cache_registry_retains_replay_window_by_logical_block_order(
    max_query_len: int | None,
) -> None:
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
        max_query_len=max_query_len,
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


def test_cache_registry_accepts_idle_none_query_bound() -> None:
    cache = torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2)
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
            num_blocks=SimpleNamespace(gpu=torch.zeros((1, 1), dtype=torch.int32)),
            kernel_block_sizes=[4],
        ),
        speculator=None,
        num_speculative_steps=5,
        model_state=SimpleNamespace(decoder_replay_layers=SimpleNamespace(window=8)),
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = runner
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=1,
        max_query_len=None,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
    )
    context = SimpleNamespace(
        attn_metadata={
            layer_name: SimpleNamespace(
                slot_mapping=torch.tensor([-1], dtype=torch.int64)
            )
        }
    )
    block_tables = (torch.zeros((1, 1), dtype=torch.int32),)

    idle = controller._collect_cache_intervals(
        input_batch,
        block_tables,
        ((context, 1),),
    )
    explicit_zero = controller._collect_cache_intervals(
        SimpleNamespace(**(vars(input_batch) | {"max_query_len": 0})),
        block_tables,
        ((context, 1),),
    )

    idle_record = next(iter(idle.values()))
    zero_record = next(iter(explicit_zero.values()))
    assert idle_record.destination == zero_record.destination == []
    assert idle_record.source == zero_record.source == []
    assert input_batch.max_query_len is None


@pytest.mark.parametrize("max_query_len", [1, None])
def test_cache_registry_retains_next_verification_across_block_boundary(
    max_query_len: int | None,
) -> None:
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
        max_query_len=max_query_len,
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
