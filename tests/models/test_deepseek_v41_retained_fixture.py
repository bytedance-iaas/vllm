# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.replayssm.dsv41_retained_control_compare import (
    IDENTITY_NORMALIZATION_VERSION,
    MASKED_NEGATIVE_INFINITY_PATHS,
    EvidenceRun,
    _canonicalize_rank_local_shared_config,
    _compare_runs,
    _floating_metrics,
    _identity_normalization_provenance,
    _validate_logical_payload,
    _validate_view_layout,
    _validated_scratch_rows,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext
from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
from vllm.models.deepseek_v41.nvidia.retained_replay_fixture import (
    RetainedReplayFixtureController,
    RetainedReplayFixtureError,
    TouchedStateRegistry,
    _cache_record_format,
    _CacheWriter,
    _decode_cache_record_cpu,
    _LogicalCacheView,
    _speculative_config_identity,
    canonical_decision_reference,
    clone_owned,
    merge_typed_snapshots,
    typed_tree_snapshot,
    write_typed_run_bundle,
)


@dataclass
class _OwnedTree:
    tensor: torch.Tensor
    array: np.ndarray
    children: dict[str, torch.Tensor]


def _ranked_config_manifest(
    rank: int,
    engine_id: str,
    *,
    port: int = 14579,
    connector_repr: str | None = None,
) -> dict:
    connector_repr = connector_repr or (
        "KVTransferConfig("
        "kv_connector='MooncakeConnector', "
        f"engine_id={engine_id!r}, "
        "kv_role='kv_consumer', "
        f"kv_port={port})"
    )
    shared_config = {
        "model_config": {"model": "fixture"},
        "speculative_config": {
            "method": "dspark",
            "target_kv_transfer_config": connector_repr,
        },
    }
    return {
        "dp_rank": rank,
        "rank_config": {"data_parallel_rank": rank, "device": f"cuda:{rank}"},
        "shared_config": shared_config,
        "config_digest": hashlib.sha256(
            json.dumps(shared_config, sort_keys=True).encode()
        ).hexdigest(),
    }


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


def test_decode_page_segregated_cache_records() -> None:
    fp8_one = torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8).item()

    mxfp8 = torch.zeros((1, 2, 528), dtype=torch.uint8)
    mxfp8_bytes = mxfp8.view(-1)
    mxfp8_bytes[512:1024].fill_(fp8_one)
    mxfp8_bytes[2 * 512 + 16 : 2 * 512 + 32].fill_(127)
    raw, scales, decoded = _decode_cache_record_cpu(mxfp8, 0, 1)
    assert _cache_record_format(mxfp8) == "v41_ue8m0_mxfp8"
    assert raw.shape == (512,)
    assert scales.tolist() == [127] * 16
    assert torch.equal(decoded, torch.ones(512))

    nvfp4 = torch.zeros((1, 2, 288), dtype=torch.uint8)
    nvfp4_bytes = nvfp4.view(-1)
    nvfp4_bytes[256:512].fill_(0x66)
    nvfp4_scale = (
        torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8).item()
    )
    nvfp4_bytes[2 * 256 + 32 : 2 * 256 + 64].fill_(nvfp4_scale)
    raw, scales, decoded = _decode_cache_record_cpu(nvfp4, 0, 1)
    assert _cache_record_format(nvfp4) == "v41_nvfp4"
    assert raw.shape == (256,)
    assert scales.shape == (32,)
    assert torch.equal(decoded, torch.full((512,), 4.0))


def test_decode_v4_and_indexer_cache_records() -> None:
    fp8_one = torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8).item()

    v4 = torch.zeros((1, 2, 584), dtype=torch.uint8)
    v4_bytes = v4.view(-1)
    v4_bytes[576:1024].fill_(fp8_one)
    rope = torch.ones(64, dtype=torch.bfloat16).view(torch.uint8)
    v4_bytes[576 + 448 : 2 * 576].copy_(rope)
    v4_bytes[2 * 576 + 8 : 2 * 576 + 15].fill_(127)
    _, scales, decoded = _decode_cache_record_cpu(v4, 0, 1)
    assert scales.tolist() == [127] * 7 + [0]
    assert torch.equal(decoded, torch.ones(512))

    indexer_fp8 = torch.zeros((1, 2, 132), dtype=torch.uint8)
    indexer_fp8_bytes = indexer_fp8.view(-1)
    indexer_fp8_bytes[128:256].fill_(fp8_one)
    indexer_fp8_bytes[2 * 128 + 4 : 2 * 128 + 8].copy_(
        torch.tensor([2.0], dtype=torch.float32).view(torch.uint8)
    )
    _, _, decoded = _decode_cache_record_cpu(indexer_fp8, 0, 1)
    assert torch.equal(decoded, torch.full((128,), 2.0))

    indexer_mxfp4 = torch.zeros((1, 2, 68), dtype=torch.uint8)
    indexer_mxfp4_bytes = indexer_mxfp4.view(-1)
    indexer_mxfp4_bytes[64:128].fill_(0x66)
    indexer_mxfp4_bytes[2 * 64 + 4 : 2 * 64 + 8].fill_(127)
    _, _, decoded = _decode_cache_record_cpu(indexer_mxfp4, 0, 1)
    assert torch.equal(decoded, torch.full((128,), 4.0))


def test_decode_plain_and_ring_cache_records() -> None:
    bf16 = torch.arange(1024, dtype=torch.float32).to(torch.bfloat16)
    bf16 = bf16[:512].reshape(1, 1, 512)
    raw, scales, decoded = _decode_cache_record_cpu(bf16, 0, 0)
    assert raw.numel() == 1024
    assert scales.numel() == 0
    assert torch.equal(decoded, bf16[0, 0].float())

    fp8 = torch.ones((1, 1, 512), dtype=torch.float8_e4m3fn)
    _, _, decoded = _decode_cache_record_cpu(fp8, 0, 0, fp8_scale=3.0)
    assert torch.equal(decoded, torch.full((512,), 3.0))

    ring = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 1024)
    _, _, decoded = _decode_cache_record_cpu(ring, 0, 0)
    assert torch.equal(decoded, ring[0, 0])


def test_cache_record_decoder_rejects_unknown_or_nonfinite_layout() -> None:
    with pytest.raises(RetainedReplayFixtureError, match="record width"):
        _cache_record_format(torch.zeros((1, 1, 64), dtype=torch.uint8))

    plain = torch.ones((1, 1, 512), dtype=torch.float8_e4m3fn)
    with pytest.raises(RetainedReplayFixtureError, match="dequantization scale"):
        _decode_cache_record_cpu(plain, 0, 0)
    with pytest.raises(RetainedReplayFixtureError, match="finite scalar"):
        _decode_cache_record_cpu(plain, 0, 0, fp8_scale=float("inf"))


def test_logical_cache_snapshot_binds_writer_and_preserved_rows() -> None:
    cache = torch.zeros((4, 4, 528), dtype=torch.uint8)
    storage = cache.view(-1)
    for block in (1, 3):
        page = block * cache.stride(0)
        storage[page + 4 * 512 : page + 4 * 512 + 64].fill_(127)
    view = _LogicalCacheView(
        view_id="group-0:layer",
        storage_id="storage-000",
        group_id=0,
        layer_name="layer",
        owner_type="Owner",
        spec_type="Spec",
        spec_metadata={"type": "Spec"},
        cache_kind="compressed_main",
        record_format="v41_ue8m0_mxfp8",
        tensor=cache,
        block_size=4,
        tokens_per_state=1,
        s0_extents=(),
        live_window=None,
        destination_blocks={1, 3},
        consumers=["layer"],
        aliases=[],
        fp8_scale=None,
        restoration_entry="kv-dst:layer",
    )
    writer = _CacheWriter(
        phase="target-full",
        positions=torch.tensor([4]),
        request_ordinals=torch.tensor([0]),
        request_state_slots=torch.tensor([7]),
        slots_by_layer={"layer": torch.tensor([12])},
        position_path="l20:boundary/positions",
        source_rows=torch.tensor([0]),
    )
    registry = TouchedStateRegistry()
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_computed_tokens_np=np.array([4], dtype=np.int32),
        query_start_loc_np=np.array([0, 1], dtype=np.int32),
        idx_mapping=torch.tensor([7]),
    )
    registry.configure_logical_cache(
        [view],
        [writer],
        input_batch,
        (torch.tensor([[1, 3]], dtype=torch.int32),),
        {"group-0:layer": (4,)},
        {
            "computed_extents": [4],
            "total_lengths": [4],
            "is_prefilling": [False],
            "replay_start": [0],
            "replay_window": 128,
            "speculative_group_ids": [],
            "query_start_loc": [0, 1],
        },
        [
            {
                "view_id": "group-0:layer",
                "owner_type": "Owner",
                "coverage": "logical",
                "exclusion_reason": None,
            }
        ],
    )

    page = 3 * cache.stride(0)
    storage[page : page + 512].fill_(
        torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8).item()
    )
    storage[page + 4 * 512 : page + 4 * 512 + 16].fill_(127)
    snapshot = registry.logical_cache_snapshot(
        "post_sampling",
        torch.tensor([5], dtype=torch.int32),
    )
    leaves = {leaf["path"]: leaf for leaf in snapshot["leaves"]}
    root = "logical_kv/group-0:layer"
    descriptors = snapshot["tensors"][leaves[f"{root}/descriptors"]["payload_key"]]
    decoded = snapshot["tensors"][leaves[f"{root}/decoded"]["payload_key"]]
    assert descriptors.shape == (5, 6)
    assert descriptors[-1].tolist() == [0, 7, 4, 3, 0, 0]
    assert torch.equal(decoded[-1], torch.ones(512))
    assert leaves[f"{root}/decoded"]["comparison"] == "floating"
    assert leaves[f"{root}/raw_values"]["comparison"] == "diagnostic"

    storage[cache.stride(0) : cache.stride(0) + 1].fill_(1)
    with pytest.raises(RetainedReplayFixtureError, match="unwritten live record"):
        registry.logical_cache_snapshot(
            "post_sampling",
            torch.tensor([5], dtype=torch.int32),
        )


def _logical_evidence_bundle():
    cache = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
    view = _LogicalCacheView(
        view_id="group-0:model.layers.20.self_attn",
        storage_id="storage-000",
        group_id=0,
        layer_name="model.layers.20.self_attn",
        owner_type="Owner",
        spec_type="MLAAttentionSpec",
        spec_metadata={
            "type": "MLAAttentionSpec",
            "block_size": 4,
            "head_size": 512,
            "head_size_v": 0,
            "num_kv_heads": 1,
            "dtype": "torch.bfloat16",
            "tokens_per_state": 1,
            "state_content_size_bytes": 1024,
            "num_states": 4,
            "prefix_cacheable": True,
            "prefix_replay_tokens": 0,
        },
        cache_kind="compressed_main",
        record_format="plain_bf16",
        tensor=cache,
        block_size=4,
        tokens_per_state=1,
        s0_extents=(),
        live_window=None,
        destination_blocks={1, 3},
        consumers=["model.layers.20.self_attn"],
        aliases=[],
        fp8_scale=None,
        restoration_entry="kv-dst:model.layers.20.self_attn",
    )
    writers = [
        _CacheWriter(
            phase="target-full",
            positions=torch.tensor([4]),
            request_ordinals=torch.tensor([0]),
            request_state_slots=torch.tensor([7]),
            slots_by_layer={"model.layers.20.self_attn": torch.tensor([12])},
            position_path="l20:boundary/positions",
            source_rows=torch.tensor([0]),
        ),
        _CacheWriter(
            phase="target-replay",
            positions=torch.empty(0, dtype=torch.int64),
            request_ordinals=torch.empty(0, dtype=torch.int64),
            request_state_slots=torch.empty(0, dtype=torch.int64),
            slots_by_layer={},
            position_path="l20:boundary/positions",
            source_rows=torch.empty(0, dtype=torch.int64),
        ),
    ]
    authority = {
        "computed_extents": [4],
        "total_lengths": [4],
        "is_prefilling": [False],
        "replay_start": [0],
        "replay_window": 128,
        "speculative_group_ids": [],
        "query_start_loc": [0, 1],
    }
    registry = TouchedStateRegistry()
    storage = cache.view(torch.uint8).reshape(-1)
    page_bytes = cache.stride(0) * cache.element_size()
    registry.add_intervals(
        "kv-dst:model.layers.20.self_attn",
        storage,
        [(page_bytes, 2 * page_bytes), (3 * page_bytes, 4 * page_bytes)],
    )
    computed_state = torch.zeros(8, dtype=torch.int32)
    computed_state[7] = 5
    registry.add_rows(
        "request:num_computed_tokens",
        computed_state,
        torch.tensor([7]),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([7]),
    )
    registry.configure_logical_cache(
        [view],
        writers,
        input_batch,
        (torch.tensor([[1, 3]], dtype=torch.int32),),
        {view.view_id: (4,)},
        authority,
        [
            {
                "view_id": view.view_id,
                "owner_type": "Owner",
                "coverage": "logical",
                "exclusion_reason": None,
            }
        ],
    )
    cache[3, 0].fill_(1)
    snapshot = merge_typed_snapshots(
        "post_sampling",
        registry.typed_snapshot("post_sampling"),
        registry.logical_cache_snapshot(
            "post_sampling",
            torch.tensor([5], dtype=torch.int32),
        ),
        typed_tree_snapshot(
            "post_sampling",
            {"num_rejected": torch.tensor([0], dtype=torch.int32)},
            path="decisions",
        ),
    )
    leaves = {f"post_sampling:{leaf['path']}": leaf for leaf in snapshot["leaves"]}
    authority_snapshot = typed_tree_snapshot(
        "l20",
        {
            "computed_extents": torch.tensor([4], dtype=torch.int64),
            "total_lengths": torch.tensor([4], dtype=torch.int64),
            "is_prefilling": torch.tensor([False]),
            "replay_start": torch.tensor([0], dtype=torch.int64),
            "replay_window": 128,
            "speculative_group_ids": torch.empty(0, dtype=torch.int64),
            "query_start_loc": torch.tensor([0, 1], dtype=torch.int64),
        },
        path="s0_cache_authority",
    )
    for leaf in authority_snapshot["leaves"]:
        payload_key = leaf.get("payload_key")
        if payload_key is not None:
            new_key = f"authority-{payload_key}"
            snapshot["tensors"][new_key] = authority_snapshot["tensors"][payload_key]
            leaf["payload_key"] = new_key
        leaves[f"l20:{leaf['path']}"] = leaf
    snapshot["tensors"]["boundary-positions"] = torch.tensor([4], dtype=torch.int64)
    leaves["l20:boundary/positions"] = {
        "checkpoint": "l20",
        "path": "boundary/positions",
        "family": "boundary/positions",
        "comparison": "exact",
        "dtype": "torch.int64",
        "shape": [1],
        "payload_key": "boundary-positions",
    }
    mapping = {
        "request_state_slots": [7],
        "block_tables": [[[1, 3]]],
        "query_start_loc": [0, 1],
        "s0_authority": authority,
    }
    return view.metadata(), leaves, snapshot["tensors"], mapping


def test_logical_comparator_binds_writer_destination_and_spec() -> None:
    view, leaves, payload, mapping = _logical_evidence_bundle()
    _validate_logical_payload(
        Path("in-memory"),
        "post_sampling",
        view,
        leaves,
        payload,
        mapping,
    )

    root = "post_sampling:logical_kv/group-0:model.layers.20.self_attn"
    descriptors = payload[leaves[f"{root}/descriptors"]["payload_key"]]
    descriptors[-1, 5] = 1
    with pytest.raises(ValueError, match="writer/S0 descriptors differ"):
        _validate_logical_payload(
            Path("in-memory"),
            "post_sampling",
            view,
            leaves,
            payload,
            mapping,
        )
    descriptors[-1, 5] = 0
    descriptor_metadata = leaves[f"{root}/descriptors"]["metadata"]
    identity = descriptors[-1, :5].tolist()
    descriptor_metadata["writer_records"]["target-full"] = []
    descriptor_metadata["writer_records"]["target-replay"] = [identity]
    descriptors[-1, 5] = 1
    with pytest.raises(ValueError, match="writer records differ from sources"):
        _validate_logical_payload(
            Path("in-memory"),
            "post_sampling",
            view,
            leaves,
            payload,
            mapping,
        )
    descriptor_metadata["writer_records"]["target-full"] = [identity]
    descriptor_metadata["writer_records"]["target-replay"] = []
    descriptors[-1, 5] = 0

    erased = dict(view)
    erased["destination_blocks"] = []
    with pytest.raises(ValueError, match="destination coverage differs"):
        _validate_logical_payload(
            Path("in-memory"),
            "post_sampling",
            erased,
            leaves,
            payload,
            mapping,
        )

    forged = dict(view)
    forged["spec_metadata"] = dict(view["spec_metadata"])
    forged["spec_metadata"].update(
        {
            "dtype": "torch.int32",
            "head_size": 7,
            "state_content_size_bytes": 28,
        }
    )
    with pytest.raises(ValueError, match="spec/layout agreement differs"):
        _validate_logical_payload(
            Path("in-memory"),
            "post_sampling",
            forged,
            leaves,
            payload,
            mapping,
        )

    forged_s0 = dict(view)
    forged_s0["s0_extents"] = [5]
    with pytest.raises(ValueError, match="owner-specific S0 extent differs"):
        _validate_logical_payload(
            Path("in-memory"),
            "post_sampling",
            forged_s0,
            leaves,
            payload,
            mapping,
        )


@pytest.mark.parametrize(
    "dtype,width,record_format,cache_kind,spec_type,spec_tokens,logical_tokens",
    [
        (
            "torch.uint8",
            584,
            "v4_ue8m0_fp8_bf16",
            "compressed_main",
            "MLAAttentionSpec",
            2,
            2,
        ),
        (
            "torch.uint8",
            528,
            "v41_ue8m0_mxfp8",
            "swa",
            "SlidingWindowMLASpec",
            1,
            1,
        ),
        (
            "torch.uint8",
            288,
            "v41_nvfp4",
            "compressed_main",
            "MLAAttentionSpec",
            2,
            2,
        ),
        (
            "torch.uint8",
            132,
            "indexer_fp8",
            "indexer",
            "MLAAttentionSpec",
            2,
            2,
        ),
        (
            "torch.uint8",
            68,
            "indexer_mxfp4",
            "indexer",
            "MLAAttentionSpec",
            2,
            2,
        ),
        (
            "torch.bfloat16",
            512,
            "plain_bf16",
            "compressed_main",
            "MLAAttentionSpec",
            1,
            1,
        ),
        (
            "torch.float8_e4m3fn",
            512,
            "plain_fp8",
            "compressed_main",
            "MLAAttentionSpec",
            1,
            1,
        ),
        (
            "torch.float32",
            1024,
            "compressor_ring_fp32",
            "compressor_ring",
            "CircularBufferSpec",
            1,
            2,
        ),
    ],
)
def test_logical_comparator_accepts_live_storage_layouts(
    dtype,
    width,
    record_format,
    cache_kind,
    spec_type,
    spec_tokens,
    logical_tokens,
) -> None:
    element_size = {
        "torch.uint8": 1,
        "torch.bfloat16": 2,
        "torch.float8_e4m3fn": 1,
        "torch.float32": 4,
    }[dtype]
    block_size = 8
    record_bytes = width * element_size
    _validate_view_layout(
        Path("in-memory"),
        {
            "dtype": dtype,
            "shape": [2, block_size, width],
            "strides_bytes": [
                block_size * record_bytes,
                record_bytes,
                element_size,
            ],
            "block_size": block_size,
            "tokens_per_state": logical_tokens,
            "record_format": record_format,
            "cache_kind": cache_kind,
            "spec_type": spec_type,
            "spec_metadata": {
                "type": spec_type,
                "dtype": dtype,
                "block_size": block_size * spec_tokens,
                "num_states": block_size,
                "tokens_per_state": spec_tokens,
                "state_content_size_bytes": record_bytes,
            },
        },
    )


def test_registry_builds_owner_specific_replay_s0_domains() -> None:
    layer_names = (
        "model.layers.20.self_attn",
        "model.layers.20.self_attn.swa_cache",
        "draft.layers.0.self_attn",
    )
    caches = {
        name: torch.zeros((3, 4, 512), dtype=torch.bfloat16) for name in layer_names
    }

    def view(group_id: int, layer_name: str, prefix_cacheable: bool):
        return _LogicalCacheView(
            view_id=f"group-{group_id}:{layer_name}",
            storage_id=f"storage-{group_id:03d}",
            group_id=group_id,
            layer_name=layer_name,
            owner_type="Owner",
            spec_type="MLAAttentionSpec",
            spec_metadata={
                "type": "MLAAttentionSpec",
                "prefix_cacheable": prefix_cacheable,
            },
            cache_kind="compressed_main",
            record_format="plain_bf16",
            tensor=caches[layer_name],
            block_size=4,
            tokens_per_state=1,
            s0_extents=(),
            live_window=None,
            destination_blocks={1, 2},
            consumers=[layer_name],
            aliases=[],
            fp8_scale=None,
            restoration_entry=f"kv-dst:{layer_name}",
        )

    views = [
        view(0, layer_names[0], True),
        view(1, layer_names[1], False),
        view(2, layer_names[2], True),
    ]
    groups = [SimpleNamespace(layer_names=[name]) for name in layer_names]
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = SimpleNamespace(
        req_states=SimpleNamespace(
            total_len=SimpleNamespace(gpu=torch.tensor([8], dtype=torch.int32)),
            num_computed_tokens=SimpleNamespace(
                gpu=torch.tensor([4], dtype=torch.int32)
            ),
        ),
        speculator=SimpleNamespace(draft_kv_cache_group_ids=(2,)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        compilation_config=SimpleNamespace(
            static_forward_context={
                name: SimpleNamespace(kv_cache=caches[name]) for name in layer_names
            }
        ),
    )
    controller._collect_cache_intervals = lambda *args, **kwargs: {
        ("cuda", group_id, 0, 0): SimpleNamespace(
            views=[cache_view],
            destination=[],
            source=[],
        )
        for group_id, cache_view in enumerate(views)
    }
    controller._register_runtime_state = lambda *args, **kwargs: None
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        is_prefilling_np=np.array([True]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    registry = controller._build_registry(
        input_batch,
        tuple(torch.tensor([[1, 2]], dtype=torch.int32) for _ in layer_names),
        (),
        [],
        torch.tensor([4], dtype=torch.int32),
        4,
    )

    assert registry._cache_s0_extents == {
        f"group-0:{layer_names[0]}": (8,),
        f"group-1:{layer_names[1]}": (0,),
        f"group-2:{layer_names[2]}": (0,),
    }


def test_draft_cache_writers_bind_context_and_query_rows() -> None:
    speculator = SimpleNamespace(
        draft_kv_cache_group_ids=(1,),
        _context_slot_mappings=(torch.tensor([12, -1]),),
        num_query_per_req=2,
        idx_mapping=torch.tensor([7]),
        context_positions=torch.tensor([3, 0]),
        input_buffers=SimpleNamespace(positions=torch.tensor([4, 5])),
        _context_rows=lambda input_batch: None,
        block_tables=SimpleNamespace(
            slot_mappings=torch.tensor(
                [
                    [-1, -1],
                    [16, 17],
                ]
            )
        ),
    )
    controller = RetainedReplayFixtureController.__new__(
        RetainedReplayFixtureController
    )
    controller.runner = SimpleNamespace(
        speculator=speculator,
        dspark_prefill_only=False,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=["target"]),
                SimpleNamespace(layer_names=["draft"]),
            ]
        ),
    )
    fixture = SimpleNamespace(
        is_dummy=False,
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=2,
            query_start_loc=torch.tensor([0, 2]),
            idx_mapping=torch.tensor([7]),
        ),
    )

    context, query = controller._draft_cache_writers(fixture)

    assert context.phase == "draft-context"
    assert context.positions.tolist() == [3, 0]
    assert context.request_ordinals.tolist() == [0, 0]
    assert context.request_state_slots.tolist() == [7, 7]
    assert context.slots_by_layer["draft"].tolist() == [12, -1]
    assert context.enforce_committed_extent is False
    assert query.phase == "draft-query"
    assert query.positions.tolist() == [4, 5]
    assert query.slots_by_layer["draft"].tolist() == [16, 17]
    assert query.enforce_committed_extent is False


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
    assert comparison["registry/kv-dst:test"] == "diagnostic"
    assert comparison["registry/speculator:_draft_scatter_buf"] == "floating"
    features = next(
        leaf
        for leaf in snapshot["leaves"]
        if leaf["path"] == "registry/acceptance:features"
    )
    assert features["metadata"]["compare_prefix"] == 1


def _scratch_run(
    values: torch.Tensor,
    *,
    comparison: str,
) -> EvidenceRun:
    path = "registry/model:candidate_block_buffer"
    metadata = {
        "compare_prefix": values.shape[0],
        "scratch_policy": "candidate_membership_v1",
        "producer_layer": 20,
        "cache_owner": "model.layers.20.self_attn",
        "compress_ratio": 2,
        "candidate_block_size": 2,
        "row_writer": "target-replay",
        "row_input_indices": list(range(values.shape[0])),
        "row_positions": [15] * values.shape[0],
        "row_request_ordinals": list(range(values.shape[0])),
        "row_request_state_slots": list(range(values.shape[0])),
        "sentinel": -1,
        "require_unique_valid_ids": True,
        "require_trailing_sentinels": True,
        "valid_id_exclusive_bounds": [4] * values.shape[0],
        "expected_valid_counts": [3] * values.shape[0],
    }
    leaf = {
        "checkpoint": "s1",
        "path": path,
        "family": path,
        "comparison": comparison,
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "payload_key": "values",
        "metadata": metadata,
    }
    boundary = {
        "checkpoint": "l20",
        "path": "boundary/positions",
        "family": "boundary/positions",
        "comparison": "exact",
        "dtype": "torch.int64",
        "shape": [values.shape[0]],
        "payload_key": "positions",
    }
    views = [
        {
            "layer_name": "model.layers.20.self_attn",
            "tokens_per_state": 2,
            "consumers": ["model.layers.20.self_attn"],
        },
        {
            "layer_name": "model.layers.20.self_attn.indexer.k_cache",
            "tokens_per_state": 2,
            "consumers": ["model.layers.20.self_attn.indexer.k_cache"],
        },
    ]
    return EvidenceRun(
        Path("rank-00/a-001.json"),
        {
            "dp_rank": 0,
            "ordinal": 1,
            "run_kind": "a",
            "evidence_contract": {
                "logical_mapping": {
                    "request_state_slots": list(range(values.shape[0])),
                    "query_start_loc": list(range(values.shape[0] + 1)),
                },
                "logical_cache_views": views,
            },
        },
        {
            "values": values,
            "positions": torch.full((values.shape[0],), 15, dtype=torch.int64),
        },
        {
            f"s1:{path}": leaf,
            "l20:boundary/positions": boundary,
        },
    )


def _add_topk(run: EvidenceRun, values: torch.Tensor) -> None:
    path = "registry/model:topk_indices_buffer"
    candidate = run.leaves["s1:registry/model:candidate_block_buffer"]
    metadata = {
        **candidate["metadata"],
        "scratch_policy": "topk_diagnostic_v1",
        "cache_owner": "model.layers.20.self_attn.indexer.k_cache",
        "candidate_membership_path": "registry/model:candidate_block_buffer",
        "candidate_block_size": 2,
        "valid_id_exclusive_bounds": [8] * values.shape[0],
        "expected_valid_counts": [3] * values.shape[0],
    }
    run.payload["topk"] = values
    run.leaves[f"s1:{path}"] = {
        "checkpoint": "s1",
        "path": path,
        "family": path,
        "comparison": "diagnostic_topk",
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "payload_key": "topk",
        "metadata": metadata,
    }


def test_scratch_comparator_canonicalizes_candidates_and_diagnoses_topk() -> None:
    left = _scratch_run(
        torch.tensor([[0, 1, 2, -1]], dtype=torch.int32),
        comparison="canonical_membership",
    )
    right = _scratch_run(
        torch.tensor([[2, 0, 1, -1]], dtype=torch.int32),
        comparison="canonical_membership",
    )
    assert torch.equal(
        _validated_scratch_rows(left, next(iter(left.leaves.values()))),
        _validated_scratch_rows(right, next(iter(right.leaves.values()))),
    )
    assert _compare_runs(
        left,
        right,
        denominator_floor=1e-12,
        envelope=None,
    )["passed"]

    _add_topk(
        left,
        torch.tensor([[3, 1, 2]], dtype=torch.int32),
    )
    _add_topk(
        right,
        torch.tensor([[2, 4, 1]], dtype=torch.int32),
    )
    report = _compare_runs(
        left,
        right,
        denominator_floor=1e-12,
        envelope=None,
    )
    assert report["passed"]
    assert report["diagnostic"][0]["mismatches"] == 1


def test_scratch_comparator_rejects_duplicate_or_out_of_range_ids() -> None:
    duplicate = _scratch_run(
        torch.tensor([[1, 1, 2, -1]], dtype=torch.int32),
        comparison="canonical_membership",
    )
    with pytest.raises(ValueError, match="duplicate scratch ID"):
        _validated_scratch_rows(duplicate, next(iter(duplicate.leaves.values())))

    out_of_range = _scratch_run(
        torch.tensor([[1, 2, 8, -1]], dtype=torch.int32),
        comparison="canonical_membership",
    )
    with pytest.raises(ValueError, match="exceeds visibility"):
        _validated_scratch_rows(out_of_range, next(iter(out_of_range.leaves.values())))


def test_scratch_comparator_binds_boundary_and_cache_owner() -> None:
    run = _scratch_run(
        torch.tensor([[0, 1, 2, -1]], dtype=torch.int32),
        comparison="canonical_membership",
    )
    leaf = run.leaves["s1:registry/model:candidate_block_buffer"]
    leaf["metadata"]["row_positions"] = [199]
    with pytest.raises(ValueError, match="positions differ from L20"):
        _validated_scratch_rows(run, leaf)

    leaf["metadata"]["row_positions"] = [15]
    leaf["metadata"]["cache_owner"] = "model.layers.999.self_attn"
    with pytest.raises(ValueError, match="cache ownership differs"):
        _validated_scratch_rows(run, leaf)


def test_numeric_comparator_accepts_matching_infinity_masks() -> None:
    metrics = _floating_metrics(
        torch.tensor([1.0, float("-inf"), 3.0]),
        torch.tensor([1.25, float("-inf"), 3.0]),
        denominator_floor=1e-12,
        allow_negative_infinity=True,
    )

    assert metrics["negative_infinities"] == 1
    assert metrics["max_abs"] == 0.25
    assert "registry/speculator:draft_logits_aux" not in (
        MASKED_NEGATIVE_INFINITY_PATHS
    )
    assert "registry/speculator:_draft_scatter_buffer" not in (
        MASKED_NEGATIVE_INFINITY_PATHS
    )

    with pytest.raises(ValueError, match="unexpected negative infinity"):
        _floating_metrics(
            torch.tensor([1.0, float("-inf")]),
            torch.tensor([1.0, float("-inf")]),
            denominator_floor=1e-12,
        )

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


def test_comparator_normalizes_valid_dp_engine_id_suffixes_only() -> None:
    manifests = [
        _ranked_config_manifest(rank, f"shared-engine_dp{rank}") for rank in range(8)
    ]
    canonical = [
        _canonicalize_rank_local_shared_config(manifest) for manifest in manifests
    ]
    assert len({json.dumps(config, sort_keys=True) for config, _ in canonical}) == 1

    runs = {
        ("a", rank, 1): SimpleNamespace(rank=rank, manifest=manifest)
        for rank, manifest in enumerate(manifests)
    }
    provenance = _identity_normalization_provenance(runs)
    assert provenance["version"] == IDENTITY_NORMALIZATION_VERSION
    assert len(provenance["rank_identities"]) == 8
    assert {record["engine_id"] for record in provenance["rank_identities"]} == {
        f"shared-engine_dp{rank}" for rank in range(8)
    }


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            _ranked_config_manifest(0, "shared-engine"),
            "lacks a DP suffix",
        ),
        (
            _ranked_config_manifest(0, "shared-engine_dp1"),
            "names DP rank 1",
        ),
        (
            _ranked_config_manifest(0, "shared-engine_dp00"),
            "uses a noncanonical suffix",
        ),
        (
            _ranked_config_manifest(
                0,
                "shared-engine_dp0",
                connector_repr=(
                    "KVTransferConfig("
                    "engine_id='shared-engine_dp0', "
                    "engine_id='duplicate_dp0')"
                ),
            ),
            "has duplicate keywords",
        ),
        (
            _ranked_config_manifest(
                0,
                "shared-engine_dp0",
                connector_repr="KVTransferConfig(kv_port=14579)",
            ),
            "missing or ambiguous",
        ),
        (
            _ranked_config_manifest(
                0,
                "shared-engine_dp0",
                connector_repr=(
                    "KVTransferConfig("
                    "engine_id='shared-engine_dp0', "
                    "kv_port=14579, "
                    "kv_port=14580)"
                ),
            ),
            "has duplicate keywords",
        ),
        (
            _ranked_config_manifest(
                0,
                "shared-engine_dp0",
                connector_repr=(
                    "KVTransferConfig("
                    "engine_id=None, "
                    "kv_connector_extra_config={"
                    "'note': \"prefix, engine_id='shared-engine_dp0', suffix\"})"
                ),
            ),
            "is not a string",
        ),
        (
            _ranked_config_manifest(
                0,
                "shared-engine_dp0",
                connector_repr=(
                    "KVTransferConfig(engine_id='shared-engine_dp0', bad=)"
                ),
            ),
            "is malformed",
        ),
    ],
)
def test_comparator_rejects_invalid_dp_engine_identity(
    manifest: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _canonicalize_rank_local_shared_config(manifest)


def test_comparator_rejects_config_drift_and_broken_bindings() -> None:
    rank0 = _ranked_config_manifest(0, "shared-engine_dp0")
    rank1 = _ranked_config_manifest(1, "shared-engine_dp1", port=14580)
    runs = {
        ("a", 0, 1): SimpleNamespace(rank=0, manifest=rank0),
        ("a", 1, 1): SimpleNamespace(rank=1, manifest=rank1),
    }
    with pytest.raises(ValueError, match="differs after DP identity normalization"):
        _identity_normalization_provenance(runs)

    rank1 = _ranked_config_manifest(1, "other-engine_dp1")
    runs[("a", 1, 1)] = SimpleNamespace(rank=1, manifest=rank1)
    with pytest.raises(ValueError, match="differs after DP identity normalization"):
        _identity_normalization_provenance(runs)

    nested_rank0 = _ranked_config_manifest(
        0,
        "shared-engine_dp0",
        connector_repr=(
            "KVTransferConfig("
            "engine_id='shared-engine_dp0', "
            "kv_connector_extra_config={'note': 'rank-zero'})"
        ),
    )
    nested_rank1 = _ranked_config_manifest(
        1,
        "shared-engine_dp1",
        connector_repr=(
            "KVTransferConfig("
            "engine_id='shared-engine_dp1', "
            "kv_connector_extra_config={'note': 'rank-one'})"
        ),
    )
    nested_runs = {
        ("a", 0, 1): SimpleNamespace(rank=0, manifest=nested_rank0),
        ("a", 1, 1): SimpleNamespace(rank=1, manifest=nested_rank1),
    }
    with pytest.raises(ValueError, match="differs after DP identity normalization"):
        _identity_normalization_provenance(nested_runs)

    within_rank_runs = {
        ("a", 0, 1): SimpleNamespace(
            rank=0,
            manifest=_ranked_config_manifest(0, "shared-engine_dp0"),
        ),
        ("a", 0, 2): SimpleNamespace(
            rank=0,
            manifest=_ranked_config_manifest(0, "changed-engine_dp0"),
        ),
    }
    with pytest.raises(ValueError, match="differs after DP identity normalization"):
        _identity_normalization_provenance(within_rank_runs)

    broken = _ranked_config_manifest(0, "shared-engine_dp0")
    broken["config_digest"] = "0" * 64
    with pytest.raises(ValueError, match="original configuration digest differs"):
        _canonicalize_rank_local_shared_config(broken)

    wrong_rank = _ranked_config_manifest(0, "shared-engine_dp0")
    wrong_rank["rank_config"]["data_parallel_rank"] = 7
    with pytest.raises(ValueError, match="rank-local configuration is invalid"):
        _canonicalize_rank_local_shared_config(wrong_rank)


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
