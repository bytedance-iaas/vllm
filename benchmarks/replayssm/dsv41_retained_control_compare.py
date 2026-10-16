#!/usr/bin/env python3
"""Calibrate and validate typed DeepSeek V4.1 retained A-control evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA = 2
QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
DECODED_RECORD_WIDTHS = {
    "v4_ue8m0_fp8_bf16": 512,
    "v41_ue8m0_mxfp8": 512,
    "v41_nvfp4": 512,
    "indexer_fp8": 128,
    "indexer_mxfp4": 128,
    "plain_bf16": 512,
    "plain_fp8": 512,
    "compressor_ring_fp32": 1024,
}
REQUIRED_CHECKPOINTS = {"l20", "target", "post_sampling", "s1"}
REQUIRED_PATHS = {
    "l20": {
        "boundary/hidden_states",
        "boundary/positions",
        "boundary/input_ids",
        "boundary/pre_mix",
        "boundary/post_mix",
        "boundary/res_mix",
        "boundary/residual",
        "boundary/aux_hidden_by_layer/__keys__/__type__",
        "boundary/remote_aux/__type__",
        "boundary/full_num_tokens",
    },
    "target": {
        "target/hidden_states",
        "target/aux_hidden_states/__type__",
        "target/mtp_pre_collapse",
    },
    "post_sampling": {
        "decisions/skipped",
        "decisions/num_sampled",
        "decisions/num_rejected",
    },
    "s1": {
        "decisions/skipped",
        "decisions/num_sampled",
        "decisions/num_rejected",
    },
}
NUMERIC_REGISTRY_PREFIXES = (
    "registry/model:_mtp_hidden_buffer",
    "registry/model:candidate_compact_logits_buffer",
    "registry/replay_graph:input:",
    "registry/speculator:draft_token_confidence_probs",
    "registry/speculator:_draft_scatter_buf",
    "registry/speculator:hidden_states",
    "registry/speculator:draft_logits",
    "registry/speculator:model:_mtp_hidden_buffer",
    "registry/acceptance:",
    "registry/adaptive:confidence_probs",
    "registry/adaptive:stale_gpu:",
    "registry/adaptive:stale_cpu:",
)
EXACT_FLOAT_REGISTRY_PREFIXES = (
    "registry/sampler:",
    "registry/speculator:temperature",
)
MASKED_NEGATIVE_INFINITY_PATHS = {
    "registry/speculator:draft_logits",
    "registry/speculator:_draft_scatter_buf",
}
IDENTITY_NORMALIZATION_VERSION = 1
_RANK_LOCAL_ENGINE_ID_PATH = (
    "shared_config.speculative_config.target_kv_transfer_config.engine_id"
)
TRIMMABLE_REGISTRY_NAMES = {
    "replay_graph:input:0",
    "replay_graph:input:1",
    "replay_graph:input:2",
    "replay_graph:input:3",
    "replay_graph:input:4",
    "replay_graph:input:5",
    "replay_graph:input:6",
    "speculator:draft_tokens",
    "speculator:draft_token_confidence_probs",
    "speculator:idx_mapping",
    "speculator:_draft_scatter_buf",
    "model:candidate_block_buffer",
    "model:topk_indices_buffer",
}


@dataclass
class EvidenceRun:
    manifest_path: Path
    manifest: dict[str, Any]
    payload: dict[str, torch.Tensor]
    leaves: dict[str, dict[str, Any]]

    @property
    def rank(self) -> int:
        return int(self.manifest["dp_rank"])

    @property
    def ordinal(self) -> int:
        return int(self.manifest["ordinal"])

    @property
    def kind(self) -> str:
        return str(self.manifest["run_kind"])


def _config_digest(config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _canonicalize_rank_local_shared_config(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rank = int(manifest["dp_rank"])
    rank_config = manifest.get("rank_config")
    if (
        not isinstance(rank_config, dict)
        or int(rank_config.get("data_parallel_rank", -1)) != rank
    ):
        raise ValueError(f"rank {rank}: rank-local configuration is invalid")
    shared_config = manifest.get("shared_config")
    if not isinstance(shared_config, dict):
        raise ValueError(f"rank {rank}: shared configuration is invalid")
    original_digest = _config_digest(shared_config)
    if manifest.get("config_digest") != original_digest:
        raise ValueError(f"rank {rank}: original configuration digest differs")

    speculative = shared_config.get("speculative_config")
    connector_repr = (
        None
        if not isinstance(speculative, dict)
        else speculative.get("target_kv_transfer_config")
    )
    prefix = "KVTransferConfig("
    if (
        not isinstance(connector_repr, str)
        or not connector_repr.startswith(prefix)
        or not connector_repr.endswith(")")
    ):
        raise ValueError(f"rank {rank}: target KV transfer identity is invalid")
    try:
        expression = ast.parse(connector_repr, mode="eval")
    except SyntaxError as error:
        raise ValueError(
            f"rank {rank}: target KV transfer identity is malformed"
        ) from error
    call = expression.body
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "KVTransferConfig"
        or call.args
        or any(keyword.arg is None for keyword in call.keywords)
    ):
        raise ValueError(f"rank {rank}: target KV transfer identity is invalid")
    keyword_names = [keyword.arg for keyword in call.keywords]
    if len(keyword_names) != len(set(keyword_names)):
        raise ValueError(
            f"rank {rank}: target KV transfer identity has duplicate keywords"
        )
    engine_keywords = [
        (index, keyword)
        for index, keyword in enumerate(call.keywords)
        if keyword.arg == "engine_id"
    ]
    if len(engine_keywords) != 1:
        raise ValueError(
            f"rank {rank}: target KV engine identity is missing or ambiguous"
        )
    engine_index, engine_keyword = engine_keywords[0]
    if not (
        isinstance(engine_keyword.value, ast.Constant)
        and isinstance(engine_keyword.value.value, str)
    ):
        raise ValueError(f"rank {rank}: target KV engine identity is not a string")
    engine_id = engine_keyword.value.value
    stem, marker, suffix_text = engine_id.rpartition("_dp")
    if not marker or not stem or not suffix_text.isdigit():
        raise ValueError(f"rank {rank}: target KV engine identity lacks a DP suffix")
    suffix_rank = int(suffix_text)
    if suffix_rank != rank:
        raise ValueError(
            f"rank {rank}: target KV engine identity names DP rank {suffix_rank}"
        )
    if suffix_text != str(rank):
        raise ValueError(
            f"rank {rank}: target KV engine identity uses a noncanonical suffix"
        )

    canonical = copy.deepcopy(shared_config)
    canonical_speculative = canonical["speculative_config"]
    canonical_expression = copy.deepcopy(expression)
    canonical_call = canonical_expression.body
    assert isinstance(canonical_call, ast.Call)
    canonical_call.keywords[engine_index].value = ast.Constant(value=f"{stem}_dp<RANK>")
    canonical_speculative["target_kv_transfer_config"] = ast.dump(
        canonical_expression,
        annotate_fields=True,
        include_attributes=False,
    )
    canonical_digest = _config_digest(canonical)
    return canonical, {
        "version": IDENTITY_NORMALIZATION_VERSION,
        "path": _RANK_LOCAL_ENGINE_ID_PATH,
        "rank": rank,
        "engine_id": engine_id,
        "engine_id_stem": stem,
        "original_config_digest": original_digest,
        "canonical_config_digest": canonical_digest,
    }


def _identity_normalization_provenance(
    runs: dict[tuple[str, int, int], EvidenceRun],
) -> dict[str, Any]:
    canonical_configs: set[str] = set()
    canonical_digests: set[str] = set()
    rank_records: dict[int, list[dict[str, Any]]] = {}
    for run in runs.values():
        canonical, record = _canonicalize_rank_local_shared_config(run.manifest)
        canonical_configs.add(json.dumps(canonical, sort_keys=True))
        canonical_digests.add(record["canonical_config_digest"])
        rank_records.setdefault(run.rank, []).append(record)
    if len(canonical_configs) != 1 or len(canonical_digests) != 1:
        raise ValueError(
            "evidence shared configuration differs after DP identity normalization"
        )

    rank_identities = []
    for rank, records in sorted(rank_records.items()):
        stable_fields = (
            "engine_id",
            "engine_id_stem",
            "original_config_digest",
            "canonical_config_digest",
        )
        for field in stable_fields:
            if len({record[field] for record in records}) != 1:
                raise ValueError(f"rank {rank}: connector identity changed: {field}")
        representative = records[0]
        rank_identities.append(
            {
                "rank": rank,
                **{field: representative[field] for field in stable_fields},
            }
        )
    stems = {record["engine_id_stem"] for record in rank_identities}
    if len(stems) != 1:
        raise ValueError("target KV engine identity stems differ across DP ranks")
    return {
        "version": IDENTITY_NORMALIZATION_VERSION,
        "path": _RANK_LOCAL_ENGINE_ID_PATH,
        "canonical_config_digest": next(iter(canonical_digests)),
        "rank_identities": rank_identities,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path_component(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _validate_container_children(
    path: Path,
    leaves: dict[str, dict[str, Any]],
) -> None:
    for key, leaf in leaves.items():
        leaf_path = leaf["path"]
        if not leaf_path.endswith("/__len__"):
            continue
        length = leaf.get("value")
        if not isinstance(length, int) or length < 0:
            raise ValueError(f"{path}: invalid container length at {key}")
        if length == 0:
            continue
        checkpoint = leaf["checkpoint"]
        base = leaf_path[: -len("/__len__")]
        for index in range(length):
            child = f"{checkpoint}:{base}/{index}"
            if not any(
                candidate == child or candidate.startswith(f"{child}/")
                for candidate in leaves
            ):
                raise ValueError(f"{path}: container child is absent: {child}")

        if not base.endswith("/__keys__"):
            continue
        mapping_base = base[: -len("/__keys__")]
        children = {
            candidate.split(":", 1)[1][len(mapping_base) + 1 :].split("/", 1)[0]
            for candidate in leaves
            if candidate.startswith(f"{checkpoint}:{mapping_base}/")
            and not candidate.startswith(f"{checkpoint}:{mapping_base}/__")
        }
        if len(children) < length:
            raise ValueError(
                f"{path}: mapping declares {length} keys but has "
                f"{len(children)} children"
            )


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.int64)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    values = magnitudes.index_select(0, (codes & 7).reshape(-1)).reshape(codes.shape)
    return torch.where(codes >= 8, -values, values)


def _decode_logical_raw(
    raw_values: torch.Tensor,
    raw_scales: torch.Tensor,
    view: dict[str, Any],
) -> torch.Tensor:
    record_format = view["record_format"]
    if raw_values.shape[0] == 0:
        try:
            width = DECODED_RECORD_WIDTHS[record_format]
        except KeyError as error:
            raise ValueError(
                f"unsupported logical cache format: {record_format}"
            ) from error
        return torch.empty((0, width), dtype=torch.float32)
    if record_format == "v4_ue8m0_fp8_bf16":
        fp8 = raw_values[:, :448].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw_scales[:, :7].float() - 127.0)
        nope = fp8.reshape(-1, 7, 64) * scales[:, :, None]
        rope = raw_values[:, 448:].contiguous().view(torch.bfloat16).float()
        if bool((raw_scales[:, 7] != 0).any()):
            raise ValueError("V4 packed cache padding scale is nonzero")
        return torch.cat((nope.reshape(-1, 448), rope), dim=1)
    if record_format == "v41_ue8m0_mxfp8":
        fp8 = raw_values.view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw_scales.float() - 127.0)
        return (fp8.reshape(-1, 16, 32) * scales[:, :, None]).reshape(-1, 512)
    if record_format in ("v41_nvfp4", "indexer_mxfp4"):
        codes = torch.stack((raw_values & 0xF, raw_values >> 4), dim=2).reshape(
            raw_values.shape[0], -1
        )
        block = 16 if record_format == "v41_nvfp4" else 32
        scales = (
            raw_scales.view(torch.float8_e4m3fn).float()
            if record_format == "v41_nvfp4"
            else torch.exp2(raw_scales.float() - 127.0)
        )
        return (
            _decode_e2m1(codes).reshape(raw_values.shape[0], -1, block)
            * scales[:, :, None]
        ).reshape(raw_values.shape[0], -1)
    if record_format == "indexer_fp8":
        scales = raw_scales.contiguous().view(torch.float32)
        return raw_values.view(torch.float8_e4m3fn).float() * scales
    if record_format == "plain_bf16":
        return raw_values.contiguous().view(torch.bfloat16).float()
    if record_format == "plain_fp8":
        scale = view.get("fp8_scale")
        if not isinstance(scale, (int, float)):
            raise ValueError("plain FP8 logical view lacks its scale")
        return raw_values.view(torch.float8_e4m3fn).float() * float(scale)
    if record_format == "compressor_ring_fp32":
        return raw_values.contiguous().view(torch.float32)
    raise ValueError(f"unsupported logical cache format: {record_format}")


def _validate_view_layout(path: Path, view: dict[str, Any]) -> int:
    dtype_sizes = {
        "torch.uint8": 1,
        "torch.bfloat16": 2,
        "torch.float8_e4m3fn": 1,
        "torch.float32": 4,
    }
    shape = view.get("shape")
    strides = view.get("strides_bytes")
    dtype = view.get("dtype")
    if (
        dtype not in dtype_sizes
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(strides, list)
        or len(strides) != 3
    ):
        raise ValueError(f"{path}: logical cache tensor layout is invalid")
    element_size = dtype_sizes[dtype]
    if (
        shape[1] != int(view["block_size"])
        or strides[2] != element_size
        or strides[1] != shape[2] * element_size
        or strides[0] < shape[1] * strides[1]
    ):
        raise ValueError(f"{path}: logical cache strides differ")
    expected_formats = {
        ("torch.uint8", 584): "v4_ue8m0_fp8_bf16",
        ("torch.uint8", 528): "v41_ue8m0_mxfp8",
        ("torch.uint8", 288): "v41_nvfp4",
        ("torch.uint8", 132): "indexer_fp8",
        ("torch.uint8", 68): "indexer_mxfp4",
        ("torch.bfloat16", 1024): "plain_bf16",
        ("torch.float8_e4m3fn", 512): "plain_fp8",
        ("torch.float32", 4096): "compressor_ring_fp32",
    }
    record_bytes = shape[2] * element_size
    if expected_formats.get((dtype, record_bytes)) != view.get("record_format"):
        raise ValueError(f"{path}: logical record format differs from tensor layout")

    spec = view.get("spec_metadata")
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: logical cache spec is absent")
    spec_tokens = spec.get("tokens_per_state")
    logical_tokens = int(view["tokens_per_state"])
    expected_head_size = {
        "v4_ue8m0_fp8_bf16": 512,
        "v41_ue8m0_mxfp8": 512,
        "v41_nvfp4": 512,
        "indexer_fp8": 132,
        "indexer_mxfp4": 68,
        "plain_bf16": 512,
        "plain_fp8": 512,
        "compressor_ring_fp32": 1024,
    }[view["record_format"]]
    head_size = int(spec.get("head_size", -1))
    head_size_v = int(spec.get("head_size_v", -1))
    num_kv_heads = int(spec.get("num_kv_heads", -1))
    num_heads = int(spec.get("num_heads", -1))
    num_head_slots = spec.get("num_head_slots")
    state_content_bytes = spec.get("state_content_bytes")
    if (
        spec.get("type") != view.get("spec_type")
        or spec.get("dtype") != dtype
        or int(spec.get("block_size", -1)) <= 0
        or int(spec.get("num_states", -1)) != shape[1]
        or int(spec.get("state_content_size_bytes", -1)) != record_bytes
        or head_size != expected_head_size
        or head_size_v != 0
        or num_kv_heads != 1
        or num_heads != 1
    ):
        raise ValueError(f"{path}: logical spec/layout agreement differs")
    if (
        num_head_slots is not None
        and (
            not isinstance(num_head_slots, int)
            or num_head_slots <= 0
            or num_head_slots != num_heads
        )
    ) or (num_head_slots is None and num_heads != num_kv_heads):
        raise ValueError(f"{path}: logical spec head-slot geometry differs")
    derived_state_bytes = (
        (head_size + head_size_v) * element_size
        if state_content_bytes is None
        else state_content_bytes
    )
    if not isinstance(derived_state_bytes, int) or derived_state_bytes != record_bytes:
        raise ValueError(f"{path}: logical spec state-content geometry differs")
    if view.get("cache_kind") == "compressor_ring":
        if (
            view.get("spec_type") != "CircularBufferSpec"
            or int(spec_tokens) != 1
            or view.get("record_format") != "compressor_ring_fp32"
        ):
            raise ValueError(f"{path}: compressor-ring spec/layout differs")
    elif int(spec_tokens) != logical_tokens:
        raise ValueError(f"{path}: logical spec compression ratio differs")
    if int(spec["block_size"]) // int(spec_tokens) != shape[1]:
        raise ValueError(f"{path}: logical spec state cardinality differs")
    unpadded_page_bytes = shape[1] * record_bytes
    alignment = spec.get("alignment")
    if alignment is None:
        expected_page_bytes = unpadded_page_bytes
    elif not isinstance(alignment, int) or alignment <= 0:
        raise ValueError(f"{path}: logical spec page alignment is invalid")
    else:
        expected_page_bytes = (
            (unpadded_page_bytes + alignment - 1) // alignment
        ) * alignment
    expected_padding = (
        expected_page_bytes if expected_page_bytes != unpadded_page_bytes else None
    )
    if (
        int(spec.get("real_page_size_bytes", -1)) != unpadded_page_bytes
        or int(spec.get("page_size_bytes", -1)) != expected_page_bytes
        or spec.get("page_size_padded") != expected_padding
    ):
        raise ValueError(f"{path}: logical spec page geometry differs")
    return element_size


def _validate_s0_authority(
    path: Path,
    view: dict[str, Any],
    leaves: dict[str, dict[str, Any]],
    payload: dict[str, torch.Tensor],
    logical_mapping: dict[str, Any],
) -> tuple[int, ...]:
    request_slots = logical_mapping["request_state_slots"]
    authority = logical_mapping.get("s0_authority")
    if not isinstance(authority, dict):
        raise ValueError(f"{path}: S0 cache authority is absent")

    runtime_fields = (
        "request_state_slots",
        "query_start_loc",
        "computed_extents",
        "total_lengths",
        "is_prefilling",
        "replay_start",
        "allocated_block_counts",
        "replay_rows",
    )
    runtime: dict[str, Any] = {}
    for field in runtime_fields:
        leaf = leaves.get(f"l20:cache_runtime_authority/{field}")
        if leaf is None or leaf.get("payload_key") is None:
            raise ValueError(f"{path}: cache runtime authority is absent: {field}")
        runtime[field] = payload[leaf["payload_key"]].detach().cpu().tolist()
    runtime["is_prefilling"] = [bool(value) for value in runtime["is_prefilling"]]
    if runtime["request_state_slots"] != request_slots:
        raise ValueError(f"{path}: runtime request-state mapping differs")
    if runtime["query_start_loc"] != logical_mapping.get("query_start_loc"):
        raise ValueError(f"{path}: runtime query boundaries differ")
    if runtime["allocated_block_counts"] != logical_mapping.get(
        "allocated_block_counts"
    ):
        raise ValueError(f"{path}: runtime allocated-block counts differ")
    if runtime["replay_rows"] != logical_mapping.get("replay_rows"):
        raise ValueError(f"{path}: runtime replay rows differ")
    for field in (
        "computed_extents",
        "total_lengths",
        "is_prefilling",
        "replay_start",
        "query_start_loc",
    ):
        if runtime[field] != authority.get(field):
            raise ValueError(f"{path}: runtime S0 authority differs: {field}")

    evidence_fields = (
        "computed_extents",
        "total_lengths",
        "is_prefilling",
        "replay_start",
        "replay_window",
        "speculative_group_ids",
        "query_start_loc",
    )
    for field in evidence_fields:
        leaf = leaves.get(f"l20:s0_cache_authority/{field}")
        if leaf is None:
            raise ValueError(f"{path}: S0 cache authority evidence is absent")
        payload_key = leaf.get("payload_key")
        value = (
            leaf.get("value")
            if payload_key is None
            else payload[payload_key].detach().cpu().tolist()
        )
        if field == "is_prefilling":
            value = [bool(item) for item in value]
        if value != authority.get(field):
            raise ValueError(f"{path}: S0 cache authority evidence differs")
    fields = (
        "computed_extents",
        "total_lengths",
        "is_prefilling",
        "replay_start",
    )
    if any(
        not isinstance(authority.get(field), list)
        or len(authority[field]) != len(request_slots)
        for field in fields
    ):
        raise ValueError(f"{path}: S0 cache authority cardinality differs")
    query_start = authority.get("query_start_loc")
    if (
        query_start != logical_mapping.get("query_start_loc")
        or not isinstance(query_start, list)
        or len(query_start) != len(request_slots) + 1
        or query_start[0] != 0
        or any(not isinstance(value, int) or value < 0 for value in query_start)
        or any(left > right for left, right in zip(query_start, query_start[1:]))
    ):
        raise ValueError(f"{path}: S0 query boundaries differ")

    post_computed = leaves.get("post_sampling:registry/request:num_computed_tokens")
    post_total = leaves.get("post_sampling:registry/request:total_len")
    sampled = leaves.get("post_sampling:decisions/num_sampled")
    rejected = leaves.get("post_sampling:decisions/num_rejected")
    skipped = leaves.get("post_sampling:decisions/skipped")
    if (
        post_computed is None
        or post_total is None
        or sampled is None
        or rejected is None
        or skipped is None
    ):
        raise ValueError(f"{path}: S0 checkpoint binding is absent")
    computed_values = payload[post_computed["payload_key"]].reshape(-1).tolist()
    computed_slots = post_computed.get("metadata", {}).get("indices")
    total_values = payload[post_total["payload_key"]].reshape(-1).tolist()
    total_slots = post_total.get("metadata", {}).get("indices")
    sampled_values = payload[sampled["payload_key"]].reshape(-1).tolist()
    rejected_values = payload[rejected["payload_key"]].reshape(-1).tolist()
    skipped_value = skipped.get("value")
    if (
        not isinstance(computed_slots, list)
        or len(computed_slots) != len(computed_values)
        or not isinstance(total_slots, list)
        or len(total_slots) != len(total_values)
        or len(sampled_values) != len(request_slots)
        or len(rejected_values) != len(request_slots)
        or not isinstance(skipped_value, bool)
        or any(int(value) < 0 for value in sampled_values)
    ):
        raise ValueError(f"{path}: S0 checkpoint binding cardinality differs")
    post_by_slot = dict(zip(map(int, computed_slots), map(int, computed_values)))
    total_by_slot = dict(zip(map(int, total_slots), map(int, total_values)))
    for request, slot in enumerate(request_slots):
        query_len = int(query_start[request + 1]) - int(query_start[request])
        expected_post = int(authority["computed_extents"][request]) + (
            0 if skipped_value else query_len - int(rejected_values[request])
        )
        if post_by_slot.get(int(slot)) != expected_post:
            raise ValueError(f"{path}: S0 computed extent differs from GPU state")
        expected_total = int(authority["total_lengths"][request]) + (
            0 if skipped_value else int(sampled_values[request])
        )
        if total_by_slot.get(int(slot)) != expected_total:
            raise ValueError(f"{path}: S0 total length differs from GPU state")

    replay_window = authority.get("replay_window")
    speculative_groups = authority.get("speculative_group_ids")
    if (
        not isinstance(replay_window, int)
        or replay_window <= 0
        or not isinstance(speculative_groups, list)
        or any(not isinstance(group, int) or group < 0 for group in speculative_groups)
    ):
        raise ValueError(f"{path}: S0 replay authority is invalid")
    expected = []
    prefix_cacheable = view["spec_metadata"].get("prefix_cacheable")
    if not isinstance(prefix_cacheable, bool):
        raise ValueError(f"{path}: cache prefix policy is absent")
    for request, computed in enumerate(authority["computed_extents"]):
        computed = int(computed)
        total = int(authority["total_lengths"][request])
        replay = int(authority["replay_start"][request])
        if not 0 <= replay <= total or not 0 <= computed <= total:
            raise ValueError(f"{path}: S0 request extent is invalid")
        extent = computed
        if bool(authority["is_prefilling"][request]):
            if int(view["group_id"]) in speculative_groups:
                extent = 0
            elif replay > 0:
                extent = min(total, replay + replay_window) if prefix_cacheable else 0
        expected.append(extent)
    if view.get("s0_extents") != expected:
        raise ValueError(f"{path}: owner-specific S0 extent differs")
    return tuple(expected)


def _validate_draft_writer_runtime_authority(
    path: Path,
    phase: str,
    group_id: int,
    source_rows: list[int],
    slots: list[int],
    leaves: dict[str, dict[str, Any]],
    payload: dict[str, torch.Tensor],
    logical_mapping: dict[str, Any],
) -> None:
    speculative_groups = logical_mapping["s0_authority"].get("speculative_group_ids")
    if not isinstance(speculative_groups, list):
        raise ValueError(f"{path}: draft writer group authority is absent")
    if group_id not in speculative_groups:
        if slots:
            raise ValueError(
                f"{path}: draft writer escaped its cache group for {phase}"
            )
        return
    if phase == "draft-context":
        runtime_slots_leaf = leaves.get(
            f"s1:registry/speculator:context_slots:{group_id}"
        )
        if (
            runtime_slots_leaf is None
            or runtime_slots_leaf.get("payload_key") is None
            or not isinstance(
                runtime_slots_leaf.get("metadata", {}).get("indices"),
                list,
            )
        ):
            raise ValueError(f"{path}: draft-context runtime slots are absent")
        runtime_indices = runtime_slots_leaf["metadata"]["indices"]
        runtime_values = payload[runtime_slots_leaf["payload_key"]].reshape(-1).tolist()
        runtime_by_row = dict(zip(map(int, runtime_indices), map(int, runtime_values)))
        expected_slots = [runtime_by_row.get(int(row)) for row in source_rows]
        if None in expected_slots or slots != expected_slots:
            raise ValueError(
                f"{path}: draft writer runtime authority differs for {phase}"
            )
        return
    if phase != "draft-query":
        raise ValueError(f"{path}: unsupported draft writer phase {phase}")
    runtime_slots_leaf = leaves.get("s1:registry/speculator:query_slot_mappings")
    runtime_indices = (
        None
        if runtime_slots_leaf is None
        else runtime_slots_leaf.get("metadata", {}).get("indices")
    )
    runtime_values = (
        None
        if runtime_slots_leaf is None or runtime_slots_leaf.get("payload_key") is None
        else payload[runtime_slots_leaf["payload_key"]]
    )
    if (
        not isinstance(runtime_indices, list)
        or group_id not in runtime_indices
        or not isinstance(runtime_values, torch.Tensor)
        or runtime_values.ndim != 2
    ):
        raise ValueError(f"{path}: draft-query runtime slots are absent")
    runtime_row = runtime_indices.index(group_id)
    expected_slots = (
        runtime_values[runtime_row, : len(source_rows)].reshape(-1).tolist()
    )
    if slots != expected_slots:
        raise ValueError(f"{path}: draft writer runtime authority differs for {phase}")


def _validate_logical_payload(
    path: Path,
    checkpoint: str,
    view: dict[str, Any],
    leaves: dict[str, dict[str, Any]],
    payload: dict[str, torch.Tensor],
    logical_mapping: dict[str, Any],
) -> None:
    root = f"{checkpoint}:logical_kv/{_safe_path_component(view['view_id'])}"
    descriptors = payload[leaves[f"{root}/descriptors"]["payload_key"]]
    raw_values = payload[leaves[f"{root}/raw_values"]["payload_key"]]
    raw_scales = payload[leaves[f"{root}/raw_scales"]["payload_key"]]
    decoded = payload[leaves[f"{root}/decoded"]["payload_key"]]
    descriptor_metadata = leaves[f"{root}/descriptors"].get("metadata", {})
    request_slots = logical_mapping["request_state_slots"]
    extents = descriptor_metadata.get("committed_extents")
    computed_key = f"{checkpoint}:registry/request:num_computed_tokens"
    computed_leaf = leaves.get(computed_key)
    if computed_leaf is None or not isinstance(extents, list):
        raise ValueError(f"{path}: checkpoint extent evidence is absent for {root}")
    computed_values = payload[computed_leaf["payload_key"]].reshape(-1).tolist()
    computed_slots = computed_leaf.get("metadata", {}).get("indices")
    if not isinstance(computed_slots, list) or len(computed_slots) != len(
        computed_values
    ):
        raise ValueError(f"{path}: checkpoint computed-state mapping differs")
    computed_by_slot = dict(zip(map(int, computed_slots), map(int, computed_values)))
    if len(extents) != len(request_slots) or any(
        int(extent) != computed_by_slot.get(int(slot))
        for extent, slot in zip(extents, request_slots)
    ):
        raise ValueError(f"{path}: committed extents differ from GPU state")
    if descriptors.ndim != 2 or descriptors.shape[1] != 6:
        raise ValueError(f"{path}: logical descriptors are invalid for {root}")
    element_size = _validate_view_layout(path, view)
    s0_extents = _validate_s0_authority(
        path,
        view,
        leaves,
        payload,
        logical_mapping,
    )
    rows = descriptors.shape[0]
    if any(
        tensor.shape[0] != rows
        for tensor in (
            raw_values,
            raw_scales,
            decoded,
        )
    ):
        raise ValueError(f"{path}: logical row cardinality differs for {root}")
    if torch.unique(descriptors, dim=0).shape[0] != rows:
        raise ValueError(f"{path}: logical descriptors are duplicated for {root}")
    block_size = int(view["block_size"])
    block_tables = logical_mapping["block_tables"]
    allocated_block_counts = logical_mapping.get("allocated_block_counts")
    group_id = int(view["group_id"])
    if (
        not isinstance(allocated_block_counts, list)
        or len(allocated_block_counts) != len(block_tables)
        or not 0 <= group_id < len(block_tables)
    ):
        raise ValueError(f"{path}: logical group mapping is absent for {root}")
    table = block_tables[group_id]
    allocated = allocated_block_counts[group_id]
    if (
        not isinstance(table, list)
        or not isinstance(allocated, list)
        or len(table) != len(request_slots)
        or len(allocated) != len(request_slots)
        or any(
            not isinstance(count, int)
            or count < 0
            or not isinstance(row, list)
            or count > len(row)
            for row, count in zip(table, allocated)
        )
    ):
        raise ValueError(f"{path}: allocated block-table mapping differs for {root}")
    ratio = int(view["tokens_per_state"])

    restoration_entry = view.get("restoration_entry")
    storage_id = view.get("storage_id")
    if (
        not isinstance(storage_id, str)
        or not storage_id
        or restoration_entry != f"kv-dst:{storage_id}"
    ):
        raise ValueError(f"{path}: restoration identity is invalid for {root}")
    restoration_key = f"{checkpoint}:registry/{_safe_path_component(restoration_entry)}"
    restoration_leaf = leaves.get(restoration_key)
    if (
        restoration_leaf is None
        or restoration_leaf.get("payload_key") is None
        or restoration_leaf.get("metadata", {}).get("kind") != "byte_intervals"
    ):
        raise ValueError(f"{path}: restoration entry is absent for {root}")
    intervals = restoration_leaf.get("metadata", {}).get("intervals")
    if not isinstance(intervals, list) or any(
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(not isinstance(value, int) for value in interval)
        for interval in intervals
    ):
        raise ValueError(f"{path}: restoration binding is invalid for {root}")
    intervals = [(int(begin), int(end)) for begin, end in intervals]
    if any(
        begin < 0 or begin >= end or (index and intervals[index - 1][1] >= begin)
        for index, (begin, end) in enumerate(intervals)
    ):
        raise ValueError(f"{path}: restoration intervals are invalid for {root}")
    restoration_payload = payload[restoration_leaf["payload_key"]]
    if restoration_payload.numel() != sum(end - begin for begin, end in intervals):
        raise ValueError(f"{path}: restoration payload size differs for {root}")

    def covered(begin: int, end: int) -> bool:
        return any(lo <= begin and end <= hi for lo, hi in intervals)

    storage_offset = int(view["storage_offset_bytes"])
    stride0, stride1, _ = map(int, view["strides_bytes"])
    page_content_bytes = int(view["shape"][1]) * int(view["shape"][2]) * element_size
    candidate_blocks = {
        int(block)
        for request_table, count in zip(table, allocated)
        for block in request_table[:count]
        if int(block) > 0
    }
    bound_destination_blocks = {
        block
        for block in candidate_blocks
        if covered(
            storage_offset + block * stride0,
            storage_offset + block * stride0 + page_content_bytes,
        )
    }
    destination_blocks = set(int(value) for value in view["destination_blocks"])
    if destination_blocks != bound_destination_blocks:
        raise ValueError(f"{path}: logical destination coverage differs for {root}")

    writer_count = descriptor_metadata.get("writer_count")
    writer_phases = descriptor_metadata.get("writer_phases")
    writer_records = descriptor_metadata.get("writer_records")
    writer_enforce = descriptor_metadata.get("writer_enforce_committed")
    draft_writer_authority = logical_mapping.get("draft_writer_authority")
    draft_source_rows: dict[str, list[int]] = {}
    if checkpoint == "post_sampling":
        expected_phases = ["target-full", "target-replay"]
    else:
        runtime_root = "s1:draft_writer_runtime_authority"

        def runtime_value(field: str) -> Any:
            leaf = leaves.get(f"{runtime_root}/{field}")
            if leaf is None:
                raise ValueError(
                    f"{path}: draft writer runtime authority is absent: {field}"
                )
            payload_key = leaf.get("payload_key")
            return (
                leaf.get("value")
                if payload_key is None
                else payload[payload_key].reshape(-1).tolist()
            )

        context_enabled = runtime_value("context_enabled")
        query_enabled = runtime_value("query_enabled")
        context_rows = runtime_value("context_source_rows")
        query_rows = runtime_value("query_source_rows")
        runtime_authority = {
            "context_enabled": context_enabled,
            "query_enabled": query_enabled,
            "context_source_rows": context_rows,
            "query_source_rows": query_rows,
        }
        if (
            not isinstance(context_enabled, bool)
            or not isinstance(query_enabled, bool)
            or not isinstance(context_rows, list)
            or not isinstance(query_rows, list)
            or (query_enabled and not context_enabled)
            or (not context_enabled and context_rows)
            or (not query_enabled and query_rows)
            or draft_writer_authority != runtime_authority
        ):
            raise ValueError(
                f"{path}: draft writer runtime policy is invalid for {root}"
            )
        expected_phases = ["target-full", "target-replay"]
        if context_enabled:
            expected_phases.append("draft-context")
            draft_source_rows["draft-context"] = context_rows
        if query_enabled:
            expected_phases.append("draft-query")
            draft_source_rows["draft-query"] = query_rows
    if (
        not isinstance(writer_count, int)
        or writer_count < 1
        or writer_phases != expected_phases
        or writer_count != len(writer_phases)
        or not isinstance(writer_records, dict)
        or set(writer_records) != set(writer_phases)
        or not isinstance(writer_enforce, dict)
        or set(writer_enforce) != set(writer_phases)
        or any(
            writer_enforce[phase] != (not phase.startswith("draft-"))
            for phase in writer_phases
        )
    ):
        raise ValueError(f"{path}: logical writer inventory is absent for {root}")

    def validate_identity(identity: Any, *, committed: bool) -> tuple[int, ...]:
        if (
            not isinstance(identity, list)
            or len(identity) != 5
            or any(not isinstance(value, int) for value in identity)
        ):
            raise ValueError(f"{path}: logical writer record is invalid")
        request, request_slot, position, block, row = identity
        if (
            not 0 <= request < len(request_slots)
            or request_slot != int(request_slots[request])
            or position < 0
            or block <= 0
            or not 0 <= row < block_size
            or block not in destination_blocks
            or (committed and position >= int(extents[request]))
        ):
            raise ValueError(f"{path}: logical writer identity differs")
        if view["cache_kind"] == "compressor_ring":
            if allocated[request] < 1:
                raise ValueError(f"{path}: compressor ring block is absent")
            expected_block = int(table[request][0])
            expected_row = position % block_size
        else:
            if (position + 1) % ratio:
                raise ValueError(f"{path}: logical compression boundary differs")
            state = position // ratio
            logical_block = state // block_size
            if logical_block >= allocated[request]:
                raise ValueError(f"{path}: logical block-table index is invalid")
            expected_block = int(table[request][logical_block])
            expected_row = state % block_size
        if (block, row) != (expected_block, expected_row):
            raise ValueError(f"{path}: descriptor does not match block tables")
        return tuple(identity)

    final_writer_records: dict[tuple[int, ...], int] = {}
    for writer_index, phase in enumerate(writer_phases):
        authority_root = f"{root}/writer_authority/{_safe_path_component(phase)}"

        def authority_value(
            field: str,
            *,
            authority_root: str = authority_root,
            phase: str = phase,
        ) -> Any:
            leaf = leaves.get(f"{authority_root}/{field}")
            if leaf is None:
                raise ValueError(
                    f"{path}: logical writer authority is absent: {phase}/{field}"
                )
            payload_key = leaf.get("payload_key")
            return (
                leaf.get("value")
                if payload_key is None
                else payload[payload_key].reshape(-1).tolist()
            )

        position_path = authority_value("position_path")
        source_rows = authority_value("source_rows")
        positions = authority_value("positions")
        request_ordinals = authority_value("request_ordinals")
        request_state_slots = authority_value("request_state_slots")
        slots = authority_value("slots")
        if phase.startswith("target-"):
            runtime_writer_root = (
                "l20:cache_runtime_authority/target_writers/"
                f"{_safe_path_component(phase)}"
            )
            runtime_source_leaf = leaves.get(f"{runtime_writer_root}/source_rows")
            runtime_slots_leaf = leaves.get(
                f"{runtime_writer_root}/slots_by_view/"
                f"{_safe_path_component(view['view_id'])}"
            )
            if (
                runtime_source_leaf is None
                or runtime_source_leaf.get("payload_key") is None
                or runtime_slots_leaf is None
                or runtime_slots_leaf.get("payload_key") is None
                or source_rows
                != payload[runtime_source_leaf["payload_key"]].reshape(-1).tolist()
                or slots
                != payload[runtime_slots_leaf["payload_key"]].reshape(-1).tolist()
            ):
                raise ValueError(
                    f"{path}: target writer runtime authority differs for {phase}"
                )
        elif phase.startswith("draft-"):
            expected_source_rows = draft_source_rows.get(phase)
            if (
                not isinstance(expected_source_rows, list)
                or source_rows != expected_source_rows
            ):
                raise ValueError(f"{path}: draft writer source rows differ for {phase}")
            _validate_draft_writer_runtime_authority(
                path,
                phase,
                group_id,
                source_rows,
                slots,
                leaves,
                payload,
                logical_mapping,
            )
        source_leaf = (
            None if not isinstance(position_path, str) else leaves.get(position_path)
        )
        if (
            source_leaf is None
            or not all(
                isinstance(values, list)
                for values in (
                    source_rows,
                    positions,
                    request_ordinals,
                    request_state_slots,
                    slots,
                )
            )
            or len(
                {
                    len(source_rows),
                    len(positions),
                    len(request_ordinals),
                    len(request_state_slots),
                }
            )
            != 1
            or (slots and len(slots) != len(positions))
        ):
            raise ValueError(f"{path}: logical writer source is incomplete")
        source_positions = payload[source_leaf["payload_key"]].reshape(-1)
        expected_position_paths = {
            "target-full": "l20:boundary/positions",
            "target-replay": "l20:boundary/positions",
            "draft-context": "s1:registry/speculator:context_positions",
            "draft-query": "s1:registry/speculator:input:positions",
        }
        if position_path != expected_position_paths[phase]:
            raise ValueError(f"{path}: logical writer phase binding differs")
        if phase == "target-full" and source_rows != list(
            range(int(logical_mapping["query_start_loc"][-1]))
        ):
            raise ValueError(f"{path}: target-full source rows are not canonical")
        if phase == "target-replay" and source_rows != logical_mapping.get(
            "replay_rows"
        ):
            raise ValueError(f"{path}: target-replay rows differ from runtime")
        if phase == "draft-query" and source_rows != list(range(len(source_rows))):
            raise ValueError(f"{path}: draft-query source rows are not canonical")
        if (
            any(
                not isinstance(row, int) or not 0 <= row < source_positions.numel()
                for row in source_rows
            )
            or list(map(int, positions))
            != source_positions[torch.tensor(source_rows, dtype=torch.int64)].tolist()
        ):
            raise ValueError(f"{path}: logical writer positions differ")
        if position_path in (
            "l20:boundary/positions",
            "s1:registry/speculator:context_positions",
        ):
            expected_requests = [
                next(
                    (
                        request
                        for request in range(len(request_slots))
                        if int(logical_mapping["query_start_loc"][request])
                        <= row
                        < int(logical_mapping["query_start_loc"][request + 1])
                    ),
                    -1,
                )
                for row in source_rows
            ]
            if expected_requests != list(map(int, request_ordinals)):
                raise ValueError(f"{path}: logical writer request rows differ")
        if [
            int(request_slots[int(request)])
            if 0 <= int(request) < len(request_slots)
            else -1
            for request in request_ordinals
        ] != list(map(int, request_state_slots)):
            raise ValueError(f"{path}: logical writer request slots differ")
        if not slots:
            continue
        if len(slots) != len(positions):
            raise ValueError(f"{path}: logical writer slot rows differ")
        for position, request, request_slot, slot in zip(
            positions,
            request_ordinals,
            request_state_slots,
            slots,
        ):
            request = int(request)
            request_slot = int(request_slot)
            position = int(position)
            if not 0 <= request < len(request_slots) or request_slot != int(
                request_slots[request]
            ):
                raise ValueError(f"{path}: logical writer request identity differs")
            if bool(writer_enforce[phase]) and position >= int(extents[request]):
                continue
            slot = int(slot)
            if slot < 0:
                continue
            identity = [
                request,
                request_slot,
                position,
                slot // block_size,
                slot % block_size,
            ]
            final_writer_records[
                validate_identity(
                    identity,
                    committed=bool(writer_enforce[phase]),
                )
            ] = writer_index

    declared_writer_records = {phase: [] for phase in writer_phases}
    for identity, writer_index in final_writer_records.items():
        declared_writer_records[writer_phases[writer_index]].append(list(identity))
    for records in declared_writer_records.values():
        records.sort()
    if declared_writer_records != writer_records:
        raise ValueError(f"{path}: logical writer records differ from sources")

    expected_records: dict[tuple[int, ...], int] = {}
    for request, extent in enumerate(s0_extents):
        request_slot = int(request_slots[request])
        if view["cache_kind"] == "compressor_ring":
            if allocated[request] < 1:
                continue
            block = int(table[request][0])
            if block <= 0 or block not in destination_blocks:
                continue
            for position in range(extent - extent % ratio, extent):
                identity = (
                    request,
                    request_slot,
                    position,
                    block,
                    position % block_size,
                )
                expected_records[identity] = -1
            continue
        for logical_block, block_value in enumerate(
            table[request][: allocated[request]]
        ):
            block = int(block_value)
            if block <= 0 or block not in destination_blocks:
                continue
            for row in range(block_size):
                state = logical_block * block_size + row
                position = state * ratio + ratio - 1
                if position >= extent:
                    break
                live_window = view.get("live_window")
                if live_window is not None and position < max(
                    0, extent - int(live_window)
                ):
                    continue
                expected_records[(request, request_slot, position, block, row)] = -1

    for identity, writer_index in final_writer_records.items():
        if view["cache_kind"] == "compressor_ring":
            request, _, _, block, row = identity
            stale = [
                current
                for current in expected_records
                if current[0] == request and current[3] == block and current[4] == row
            ]
            for current in stale:
                del expected_records[current]
        expected_records[identity] = writer_index
    if view["cache_kind"] == "compressor_ring":
        ephemeral = {
            index
            for index, phase in enumerate(writer_phases)
            if not bool(writer_enforce[phase])
        }
        expected_records = {
            identity: writer
            for identity, writer in expected_records.items()
            if (
                writer in ephemeral
                or identity[2]
                >= int(extents[identity[0]]) - int(extents[identity[0]]) % ratio
            )
        }
        for request, extent in enumerate(extents):
            block = int(table[request][0]) if allocated[request] else -1
            if (
                extent % ratio
                and block in destination_blocks
                and not any(identity[0] == request for identity in expected_records)
            ):
                raise ValueError(f"{path}: empty ring has an open committed group")
    expected_descriptors = [
        [*identity, writer] for identity, writer in sorted(expected_records.items())
    ]
    if descriptors.tolist() != expected_descriptors:
        raise ValueError(f"{path}: logical writer/S0 descriptors differ for {root}")

    format_layout = {
        "v4_ue8m0_fp8_bf16": (576, 8),
        "v41_ue8m0_mxfp8": (512, 16),
        "v41_nvfp4": (256, 32),
        "indexer_fp8": (128, 4),
        "indexer_mxfp4": (64, 4),
    }
    storage_offset = int(view["storage_offset_bytes"])
    stride0, stride1, _ = map(int, view["strides_bytes"])
    record_format = view["record_format"]

    def restoration_bytes(begin: int, end: int) -> torch.Tensor:
        cursor = 0
        for lo, hi in intervals:
            length = hi - lo
            if lo <= begin and end <= hi:
                offset = cursor + begin - lo
                return restoration_payload[offset : offset + end - begin]
            cursor += length
        raise ValueError(f"{path}: logical raw span escapes restoration bytes")

    for record_index, (_, _, _, block, row, _) in enumerate(descriptors.tolist()):
        page = storage_offset + int(block) * stride0
        if record_format in format_layout:
            value_bytes, scale_bytes = format_layout[record_format]
            value_begin = page + int(row) * value_bytes
            scale_begin = page + block_size * value_bytes + int(row) * scale_bytes
            spans = (
                (value_begin, value_begin + value_bytes),
                (scale_begin, scale_begin + scale_bytes),
            )
        else:
            value_begin = page + int(row) * stride1
            spans = ((value_begin, value_begin + raw_values.shape[1]),)
        if any(not covered(begin, end) for begin, end in spans):
            raise ValueError(f"{path}: logical raw span escapes restoration bytes")
        if not torch.equal(
            restoration_bytes(*spans[0]),
            raw_values[record_index],
        ):
            raise ValueError(f"{path}: logical values differ from restoration bytes")
        if len(spans) == 2 and not torch.equal(
            restoration_bytes(*spans[1]),
            raw_scales[record_index],
        ):
            raise ValueError(f"{path}: logical scales differ from restoration bytes")

    reconstructed = _decode_logical_raw(raw_values, raw_scales, view)
    if reconstructed.shape != decoded.shape or not torch.equal(reconstructed, decoded):
        raise ValueError(f"{path}: logical decoded values do not match raw bytes")


def _validate_evidence_contract(
    path: Path,
    manifest: dict[str, Any],
    leaves: dict[str, dict[str, Any]],
    payload: dict[str, torch.Tensor],
) -> None:
    contract = manifest.get("evidence_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: evidence contract is absent")
    checkpoint_paths = contract.get("checkpoint_paths")
    comparison_classes = contract.get("comparison_classes")
    registry_entries = contract.get("registry_entries")
    meaningful_counts = contract.get("meaningful_counts")
    graph_input_count = contract.get("graph_input_count")
    logical_cache_views = contract.get("logical_cache_views")
    cache_owner_inventory = contract.get("cache_owner_inventory")
    logical_mapping = contract.get("logical_mapping")
    if not (
        isinstance(checkpoint_paths, dict)
        and isinstance(comparison_classes, dict)
        and isinstance(registry_entries, list)
        and isinstance(meaningful_counts, dict)
        and isinstance(graph_input_count, int)
        and isinstance(logical_cache_views, list)
        and isinstance(cache_owner_inventory, list)
        and isinstance(logical_mapping, dict)
    ):
        raise ValueError(f"{path}: evidence contract is incomplete")

    for checkpoint in REQUIRED_CHECKPOINTS:
        actual = sorted(
            leaf["path"] for leaf in leaves.values() if leaf["checkpoint"] == checkpoint
        )
        if actual != checkpoint_paths.get(checkpoint):
            raise ValueError(f"{path}: {checkpoint} inventory differs from contract")
    actual_classes = {key: leaf["comparison"] for key, leaf in leaves.items()}
    if actual_classes != comparison_classes:
        raise ValueError(f"{path}: comparison classes differ from contract")

    for checkpoint in ("post_sampling", "s1"):
        checkpoint_leaf_paths = {
            leaf["path"] for leaf in leaves.values() if leaf["checkpoint"] == checkpoint
        }
        for name in registry_entries:
            root = f"registry/{_safe_path_component(name)}"
            if not any(
                candidate == root or candidate.startswith(f"{root}/")
                for candidate in checkpoint_leaf_paths
            ):
                raise ValueError(f"{path}: {checkpoint} misses registry entry {name}")

    graph_names = {
        name for name in registry_entries if name.startswith("replay_graph:input:")
    }
    if graph_input_count != 7:
        raise ValueError(f"{path}: expected seven replay graph inputs")
    expected_graph_names = {
        f"replay_graph:input:{index}" for index in range(graph_input_count)
    }
    if graph_names != expected_graph_names:
        raise ValueError(f"{path}: graph input inventory is incomplete")

    if manifest.get("active_role") == "active" and not logical_cache_views:
        raise ValueError(f"{path}: active evidence has no logical cache views")
    logical_view_ids = set()
    for view in logical_cache_views:
        if not isinstance(view, dict) or view.get("schema") != 1:
            raise ValueError(f"{path}: logical cache view metadata is invalid")
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in logical_view_ids:
            raise ValueError(f"{path}: logical cache view identity is invalid")
        logical_view_ids.add(view_id)
        if (
            not isinstance(view.get("aliases"), list)
            or not isinstance(view.get("consumers"), list)
            or not isinstance(view.get("destination_blocks"), list)
            or not isinstance(view.get("spec_metadata"), dict)
        ):
            raise ValueError(f"{path}: logical cache view contract is incomplete")
        root = f"logical_kv/{_safe_path_component(view_id)}"
        for checkpoint in ("post_sampling", "s1"):
            required = {
                f"{checkpoint}:{root}/descriptors": "exact",
                f"{checkpoint}:{root}/decoded": "floating",
                f"{checkpoint}:{root}/raw_values": "diagnostic",
                f"{checkpoint}:{root}/raw_scales": "diagnostic",
            }
            for key, comparison in required.items():
                leaf = leaves.get(key)
                if leaf is None or leaf.get("comparison") != comparison:
                    raise ValueError(
                        f"{path}: logical cache evidence is absent or misclassified: "
                        f"{key}"
                    )
            _validate_logical_payload(
                path,
                checkpoint,
                view,
                leaves,
                payload,
                logical_mapping,
            )
    inventory_ids = set()
    covered_ids = set()
    for item in cache_owner_inventory:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: cache owner inventory is invalid")
        view_id = item.get("view_id")
        coverage = item.get("coverage")
        if (
            not isinstance(view_id, str)
            or view_id in inventory_ids
            or coverage not in ("logical", "source_only")
        ):
            raise ValueError(f"{path}: cache owner inventory identity differs")
        inventory_ids.add(view_id)
        if coverage == "logical":
            if item.get("exclusion_reason") is not None:
                raise ValueError(f"{path}: logical owner has an exclusion reason")
            view = next(
                (
                    candidate
                    for candidate in logical_cache_views
                    if candidate["view_id"] == view_id
                ),
                None,
            )
            if (
                view is None
                or item.get("owner_type") != view.get("owner_type")
                or item.get("storage_id") != view.get("storage_id")
                or item.get("restoration_entry") != view.get("restoration_entry")
            ):
                raise ValueError(f"{path}: logical owner storage binding differs")
            covered_ids.add(view_id)
        else:
            raise ValueError(f"{path}: bound cache owner lacks a logical view")
    if logical_view_ids != covered_ids or not logical_view_ids.issubset(inventory_ids):
        raise ValueError(f"{path}: logical cache owner inventory differs")
    for view in logical_cache_views:
        for alias in view["aliases"]:
            if alias not in logical_view_ids:
                raise ValueError(f"{path}: logical cache alias is unknown")
            peer = next(
                candidate
                for candidate in logical_cache_views
                if candidate["view_id"] == alias
            )
            if view["view_id"] not in peer["aliases"]:
                raise ValueError(f"{path}: logical cache alias is not reciprocal")

    required_registry = {
        "model:_mtp_hidden_buffer",
        "request:last_sampled_tokens",
        "request:draft_tokens",
        "request:num_computed_tokens",
        "request:total_len",
        "request:next_prefill_tokens",
        "request:all_token_ids",
        "sampler:output_bin_counts",
        "sampler:temperature",
        "sampler:top_k",
        "sampler:top_p",
        "sampler:min_p",
        "sampler:seeds",
        "speculator:draft_tokens",
        "speculator:draft_token_confidence_probs",
        "speculator:hidden_states",
        "speculator:context_positions",
        "speculator:idx_mapping",
    }
    speculative = manifest.get("shared_config", {}).get("speculative_config", {})
    if speculative.get("draft_sample_method") == "probabilistic":
        required_registry.add("speculator:draft_logits")
    if speculative.get("enable_adaptive_verification") is True:
        required_registry.update(
            {
                "adaptive:confidence_probs",
                "adaptive:_batch_draft_capacity",
                "adaptive:_num_non_draft_tokens",
                "adaptive:cu_num_logits",
                "adaptive:_stale_idx",
                "adaptive:_pending_resets",
                "adaptive:_batch_budget",
            }
        )
        if not any(name.startswith("adaptive:stale_gpu:") for name in registry_entries):
            raise ValueError(f"{path}: adaptive GPU stale state is absent")
        if not any(name.startswith("adaptive:stale_cpu:") for name in registry_entries):
            raise ValueError(f"{path}: adaptive CPU stale state is absent")
    missing_configured = required_registry - set(registry_entries)
    if missing_configured:
        raise ValueError(
            f"{path}: configured registry entries are absent: "
            f"{sorted(missing_configured)}"
        )
    non_tensor_runtime_entries = {
        "adaptive:_stale_idx",
        "adaptive:_pending_resets",
        "adaptive:_batch_budget",
    }
    required_tensors = (
        (required_registry - non_tensor_runtime_entries)
        | expected_graph_names
        | set(meaningful_counts)
    )
    if speculative.get("enable_adaptive_verification") is True:
        required_tensors.update(
            name
            for name in registry_entries
            if name.startswith(("adaptive:stale_gpu:", "adaptive:stale_cpu:"))
        )
    for checkpoint in ("post_sampling", "s1"):
        for name in required_tensors:
            key = f"{checkpoint}:registry/{_safe_path_component(name)}"
            leaf = leaves.get(key)
            if leaf is None or leaf.get("payload_key") is None:
                raise ValueError(f"{path}: required tensor evidence is absent: {key}")
            if not leaf.get("shape") or any(int(size) <= 0 for size in leaf["shape"]):
                raise ValueError(f"{path}: required tensor evidence is empty: {key}")

    if not set(meaningful_counts).issubset(TRIMMABLE_REGISTRY_NAMES):
        raise ValueError(f"{path}: unsupported meaningful-row trimming family")
    if any(
        not isinstance(count, int) or count <= 0 for count in meaningful_counts.values()
    ):
        raise ValueError(f"{path}: meaningful-row counts must be positive")
    for checkpoint in ("post_sampling", "s1"):
        for name, expected_prefix in meaningful_counts.items():
            leaf = leaves[f"{checkpoint}:registry/{_safe_path_component(name)}"]
            actual_prefix = leaf.get("metadata", {}).get("compare_prefix")
            if actual_prefix != expected_prefix:
                raise ValueError(f"{path}: meaningful-row prefix differs for {name}")
        for leaf in leaves.values():
            if leaf["checkpoint"] != checkpoint:
                continue
            actual_prefix = leaf.get("metadata", {}).get("compare_prefix")
            if actual_prefix is None:
                continue
            matching_names = [
                name
                for name in meaningful_counts
                if leaf["path"] == f"registry/{_safe_path_component(name)}"
            ]
            if len(matching_names) != 1:
                raise ValueError(
                    f"{path}: undeclared meaningful-row prefix at {leaf['path']}"
                )
    _validate_container_children(path, leaves)


def _load_run(path: Path) -> EvidenceRun:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError(f"{path}: incomplete or unsupported manifest")
    try:
        directory_rank = int(path.parent.name.removeprefix("rank-"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid rank directory") from error
    if path.parent.name != f"rank-{directory_rank:02d}":
        raise ValueError(f"{path}: noncanonical rank directory")
    if int(manifest.get("dp_rank", -1)) != directory_rank:
        raise ValueError(f"{path}: manifest rank differs from directory")
    payload_path = path.parent / manifest["payload"]
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise ValueError(f"{path}: payload checksum mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError(f"{path}: payload is not tensor-only")
    if set(manifest.get("checkpoints", ())) != REQUIRED_CHECKPOINTS:
        raise ValueError(f"{path}: required checkpoints are absent")
    leaves = {}
    for leaf in manifest["leaves"]:
        key = f"{leaf['checkpoint']}:{leaf['path']}"
        if key in leaves:
            raise ValueError(f"{path}: duplicate leaf {key}")
        payload_key = leaf.get("payload_key")
        if payload_key is not None:
            tensor = payload.get(payload_key)
            if tensor is None:
                raise ValueError(f"{path}: missing payload tensor {payload_key}")
            if str(tensor.dtype) != leaf["dtype"]:
                raise ValueError(f"{path}: payload dtype differs for {key}")
            if list(tensor.shape) != leaf["shape"]:
                raise ValueError(f"{path}: payload shape differs for {key}")
        leaves[key] = leaf
    _validate_evidence_contract(path, manifest, leaves, payload)
    for checkpoint, required_paths in REQUIRED_PATHS.items():
        actual_paths = {
            leaf["path"]
            for leaf in manifest["leaves"]
            if leaf["checkpoint"] == checkpoint
        }
        missing = required_paths - actual_paths
        if missing:
            raise ValueError(
                f"{path}: {checkpoint} misses required paths {sorted(missing)}"
            )
    for checkpoint in ("post_sampling", "s1"):
        paths = {
            leaf["path"]
            for leaf in manifest["leaves"]
            if leaf["checkpoint"] == checkpoint
        }
        if not any(path.startswith("registry/") for path in paths):
            raise ValueError(f"{path}: {checkpoint} registry evidence is absent")
        if not (
            "decisions/sampled_token_ids" in paths
            or "decisions/sampled_token_ids/__type__" in paths
        ):
            raise ValueError(f"{path}: {checkpoint} sampled decisions are absent")
        mandatory_registry = {
            "registry/replay_graph:input:0",
            "registry/model:_mtp_hidden_buffer",
            "registry/request:last_sampled_tokens",
            "registry/speculator:draft_tokens",
            "registry/speculator:draft_token_confidence_probs",
        }
        missing_registry = mandatory_registry - paths
        if missing_registry:
            raise ValueError(
                f"{path}: {checkpoint} misses registry families "
                f"{sorted(missing_registry)}"
            )
        if manifest.get("active_role") == "active" and not any(
            item.startswith("registry/kv-dst:") for item in paths
        ):
            raise ValueError(f"{path}: {checkpoint} KV destination evidence is absent")
    for key, leaf in leaves.items():
        comparison = leaf["comparison"]
        path_name = leaf["path"]
        if "compare_prefix" in leaf.get("metadata", {}) and not (
            leaf["checkpoint"] in ("post_sampling", "s1")
            and path_name.startswith("registry/")
        ):
            raise ValueError(f"{path}: compare_prefix is not allowed for {key}")
        dtype = str(leaf["dtype"])
        is_float = dtype.startswith("torch.float") or dtype.startswith("torch.bfloat")
        is_tensor = leaf.get("payload_key") is not None
        is_logical_raw = path_name.startswith("logical_kv/") and path_name.endswith(
            ("/raw_values", "/raw_scales")
        )
        is_restoration_raw = path_name.startswith("registry/kv-dst:")
        is_scratch = comparison in (
            "canonical_membership",
            "diagnostic_topk",
        )
        if (
            is_tensor
            and not is_float
            and comparison != "exact"
            and not (is_logical_raw and comparison == "diagnostic")
            and not (is_restoration_raw and comparison == "diagnostic")
            and not is_scratch
        ):
            raise ValueError(f"{path}: non-floating tensor must be exact: {key}")
        if path_name.startswith("registry/kv-") and comparison != "diagnostic":
            raise ValueError(f"{path}: restoration KV bytes must be diagnostic: {key}")
        if path_name.startswith("decisions/") and comparison != "exact":
            raise ValueError(f"{path}: decisions must remain exact: {key}")
        if is_scratch:
            expected_scratch = {
                "registry/model:candidate_block_buffer": "canonical_membership",
                "registry/model:topk_indices_buffer": "diagnostic_topk",
            }.get(path_name)
            if expected_scratch != comparison or dtype not in (
                "torch.int32",
                "torch.int64",
            ):
                raise ValueError(f"{path}: scratch comparison is unsupported: {key}")
            metadata = leaf.get("metadata", {})
            required_metadata = {
                "compare_prefix",
                "scratch_policy",
                "producer_layer",
                "cache_owner",
                "compress_ratio",
                "row_writer",
                "row_input_indices",
                "row_positions",
                "row_request_ordinals",
                "row_request_state_slots",
                "sentinel",
                "require_unique_valid_ids",
                "require_trailing_sentinels",
                "valid_id_exclusive_bounds",
                "expected_valid_counts",
            }
            if not required_metadata.issubset(metadata):
                raise ValueError(f"{path}: scratch metadata is incomplete: {key}")
            if comparison == "diagnostic_topk" and not {
                "candidate_membership_path",
                "candidate_block_size",
            }.issubset(metadata):
                raise ValueError(
                    f"{path}: top-k candidate binding is incomplete: {key}"
                )
        if path_name.startswith("logical_kv/"):
            if path_name.endswith("/decoded"):
                if not (is_tensor and is_float and comparison == "floating"):
                    raise ValueError(
                        f"{path}: logical decoded values must be floating: {key}"
                    )
            elif path_name.endswith(("/raw_values", "/raw_scales")):
                if not (
                    is_tensor and dtype == "torch.uint8" and comparison == "diagnostic"
                ):
                    raise ValueError(
                        f"{path}: logical raw bytes must be diagnostic uint8: {key}"
                    )
            elif comparison != "exact":
                raise ValueError(
                    f"{path}: logical metadata/descriptors must be exact: {key}"
                )
        if not (is_tensor and is_float):
            continue
        expected = None
        if (
            path_name.startswith("logical_kv/") and path_name.endswith("/decoded")
        ) or leaf["checkpoint"] in ("l20", "target"):
            expected = "floating"
        elif path_name.startswith("registry/"):
            if any(
                path_name.startswith(prefix) for prefix in NUMERIC_REGISTRY_PREFIXES
            ):
                expected = "floating"
            elif any(
                path_name.startswith(prefix) for prefix in EXACT_FLOAT_REGISTRY_PREFIXES
            ):
                expected = "exact"
        if expected is None:
            raise ValueError(f"{path}: unclassified floating family: {key}")
        if comparison != expected:
            raise ValueError(f"{path}: expected {expected} comparison for {key}")
    target_hidden = leaves["target:target/hidden_states"]
    if not target_hidden["shape"] or target_hidden["shape"][0] < 1:
        raise ValueError(f"{path}: target meaningful rows are absent")
    expected_skipped = manifest.get("active_role") == "idle"
    for checkpoint in ("post_sampling", "s1"):
        skipped = leaves[f"{checkpoint}:decisions/skipped"]
        if skipped.get("value") is not expected_skipped:
            raise ValueError(f"{path}: {checkpoint} skipped role is inconsistent")
    return EvidenceRun(path, manifest, payload, leaves)


def _load_directory(
    root: Path,
    *,
    allowed_kinds: set[str] | None = None,
    max_a_ordinal: int | None = None,
) -> dict[tuple[str, int, int], EvidenceRun]:
    runs = {}
    for path in sorted(root.glob("rank-*/*.json")):
        if path.name == "transaction-summary.json":
            continue
        try:
            kind, ordinal_text = path.stem.rsplit("-", 1)
            ordinal = int(ordinal_text)
        except ValueError as error:
            raise ValueError(f"{path}: invalid evidence filename") from error
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        if kind == "a" and max_a_ordinal is not None and ordinal > max_a_ordinal:
            continue
        run = _load_run(path)
        key = (run.kind, run.rank, run.ordinal)
        if key in runs:
            raise ValueError(f"duplicate run {key}")
        runs[key] = run
    if not runs:
        raise ValueError(f"no evidence manifests below {root}")
    return runs


def _load_summaries(root: Path) -> dict[int, dict[str, Any]]:
    summaries = {}
    for path in sorted(root.glob("rank-*/transaction-summary.json")):
        document = json.loads(path.read_text())
        if document.get("schema") != SCHEMA or not document.get("complete"):
            raise ValueError(f"{path}: incomplete transaction summary")
        rank = int(document["dp_rank"])
        if rank in summaries:
            raise ValueError(f"duplicate transaction summary for rank {rank}")
        summaries[rank] = document
    return summaries


def _validate_cohort(
    runs: dict[tuple[str, int, int], EvidenceRun],
    summaries: dict[int, dict[str, Any]],
    ranks: list[int],
    *,
    require_complete_files: bool,
) -> None:
    if set(summaries) != set(ranks):
        raise ValueError(
            f"transaction summaries do not match ranks: "
            f"summaries={sorted(summaries)} ranks={ranks}"
        )
    run_ids = {summary["run_id"] for summary in summaries.values()}
    if len(run_ids) != 1:
        raise ValueError(f"transaction run IDs differ: {sorted(run_ids)}")
    dp_sizes = {run.manifest["dp_size"] for run in runs.values()}
    if len(dp_sizes) != 1:
        raise ValueError("DP topology differs across evidence")
    dp_size = int(next(iter(dp_sizes)))
    if ranks != list(range(dp_size)):
        raise ValueError(f"expected ranks 0..{dp_size - 1}, found {ranks}")
    _identity_normalization_provenance(runs)
    global_fields = (
        "run_id",
        "code_hashes",
        "model",
        "draft_model",
    )
    for field in global_fields:
        values = {
            json.dumps(run.manifest.get(field), sort_keys=True) for run in runs.values()
        }
        if len(values) != 1:
            raise ValueError(f"evidence field differs across cohort: {field}")
    for rank in ranks:
        rank_runs = [run for run in runs.values() if run.rank == rank]
        for field in (
            "worker_incarnation",
            "fixture_id",
            "fixture_digest",
            "input_digest",
            "active_role",
            "s0_digest",
            "source_digest",
            "evidence_contract",
        ):
            values = {
                json.dumps(run.manifest.get(field), sort_keys=True) for run in rank_runs
            }
            if len(values) != 1:
                raise ValueError(f"rank {rank}: evidence field changed: {field}")
        if any(
            int(run.manifest["rank_config"]["data_parallel_rank"]) != rank
            for run in rank_runs
        ):
            raise ValueError(f"rank {rank}: rank-local configuration is invalid")
        graph_identities = set()
        for run in rank_runs:
            graph = run.manifest["graph"]
            if graph["callback_calls"] != 1 or graph["capture_replay_calls"] != 1:
                raise ValueError(f"{run.manifest_path}: graph was not replayed once")
            if graph["num_graphs"] < 1:
                raise ValueError(f"{run.manifest_path}: graph has no segments")
            graph_identities.add(
                (
                    graph["manager_id"],
                    graph["capture_id"],
                    graph["descriptor_rows"],
                    graph["num_graphs"],
                )
            )
        if len(graph_identities) != 1:
            raise ValueError(f"rank {rank}: selected graph identity changed")
        summary = summaries[rank]
        representative = rank_runs[0].manifest
        for field in (
            "run_id",
            "worker_incarnation",
            "fixture_id",
            "fixture_digest",
            "source_digest",
        ):
            if summary.get(field) != representative.get(field):
                raise ValueError(f"rank {rank}: summary binding differs: {field}")
        if not summary.get("live_restored"):
            raise ValueError(f"rank {rank}: LIVE restoration was not proven")
        if int(summary["completed_runs"]) < max(run.ordinal for run in rank_runs):
            raise ValueError(f"rank {rank}: transaction summary is incomplete")
        summary_files = {
            (str(item["kind"]), int(item["ordinal"])): item
            for item in summary.get("files", ())
        }
        loaded_files = {(run.kind, run.ordinal): run for run in rank_runs}
        if require_complete_files and set(summary_files) != set(loaded_files):
            raise ValueError(f"rank {rank}: summary file set differs")
        for key, run in loaded_files.items():
            item = summary_files.get(key)
            if item is None:
                raise ValueError(f"rank {rank}: summary misses {key}")
            if Path(item["manifest_path"]).name != run.manifest_path.name:
                raise ValueError(
                    f"rank {rank}: summary manifest path differs for {key}"
                )
            if item["manifest_sha256"] != _sha256(run.manifest_path):
                raise ValueError(
                    f"rank {rank}: summary manifest hash differs for {key}"
                )
            payload_path = run.manifest_path.parent / run.manifest["payload"]
            if item["payload_sha256"] != _sha256(payload_path):
                raise ValueError(f"rank {rank}: summary payload hash differs for {key}")


def _leaf_schema(leaf: dict[str, Any]) -> dict[str, Any]:
    return {
        key: leaf.get(key)
        for key in (
            "checkpoint",
            "path",
            "family",
            "comparison",
            "dtype",
            "shape",
            "metadata",
        )
    }


def _envelope_key(leaf: dict[str, Any]) -> str:
    return json.dumps(
        {
            "checkpoint": leaf["checkpoint"],
            "family": leaf["family"],
            "dtype": leaf["dtype"],
        },
        sort_keys=True,
    )


def _exact_value(run: EvidenceRun, leaf: dict[str, Any]) -> Any:
    payload_key = leaf.get("payload_key")
    if payload_key is None:
        return leaf.get("value")
    return _meaningful_tensor(run, leaf)


def _meaningful_tensor(
    run: EvidenceRun,
    leaf: dict[str, Any],
) -> torch.Tensor:
    tensor = run.payload[leaf["payload_key"]]
    compare_prefix = leaf.get("metadata", {}).get("compare_prefix")
    if compare_prefix is not None:
        if tensor.ndim < 1 or not 0 < int(compare_prefix) <= tensor.shape[0]:
            raise ValueError(f"{leaf['path']}: invalid meaningful prefix")
        tensor = tensor[: int(compare_prefix)]
    return tensor


def _validated_scratch_rows(
    run: EvidenceRun,
    leaf: dict[str, Any],
) -> torch.Tensor:
    tensor = _meaningful_tensor(run, leaf)
    metadata = leaf.get("metadata", {})
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{leaf['path']}: scratch tensor must be nonempty 2D")
    rows = tensor.shape[0]
    row_fields = (
        "row_input_indices",
        "row_positions",
        "row_request_ordinals",
        "row_request_state_slots",
        "valid_id_exclusive_bounds",
        "expected_valid_counts",
    )
    if any(
        not isinstance(metadata.get(field), list) or len(metadata[field]) != rows
        for field in row_fields
    ):
        raise ValueError(f"{leaf['path']}: scratch row metadata differs")
    if metadata.get("sentinel") != -1:
        raise ValueError(f"{leaf['path']}: unsupported scratch sentinel")
    if metadata.get("require_unique_valid_ids") is not True:
        raise ValueError(f"{leaf['path']}: scratch uniqueness gate is absent")
    if metadata.get("require_trailing_sentinels") is not True:
        raise ValueError(f"{leaf['path']}: scratch sentinel-order gate is absent")

    contract = run.manifest.get("evidence_contract")
    logical_mapping = (
        None if not isinstance(contract, dict) else contract.get("logical_mapping")
    )
    views = (
        None if not isinstance(contract, dict) else contract.get("logical_cache_views")
    )
    boundary_leaf = run.leaves.get("l20:boundary/positions")
    replay_rows_leaf = run.leaves.get("l20:cache_runtime_authority/replay_rows")
    if (
        not isinstance(logical_mapping, dict)
        or not isinstance(views, list)
        or boundary_leaf is None
        or replay_rows_leaf is None
        or replay_rows_leaf.get("payload_key") is None
    ):
        raise ValueError(f"{leaf['path']}: scratch authority is absent")
    boundary_positions = _meaningful_tensor(run, boundary_leaf).reshape(-1)
    input_rows = metadata["row_input_indices"]
    runtime_replay_rows = (
        run.payload[replay_rows_leaf["payload_key"]].reshape(-1).tolist()
    )
    if (
        metadata.get("row_writer") != "target-replay"
        or input_rows != runtime_replay_rows
        or input_rows != logical_mapping.get("replay_rows")
    ):
        raise ValueError(f"{leaf['path']}: scratch replay rows differ")
    if any(
        not isinstance(index, int) or not 0 <= index < boundary_positions.numel()
        for index in input_rows
    ):
        raise ValueError(f"{leaf['path']}: scratch input row is invalid")
    authoritative_positions = boundary_positions[
        torch.tensor(input_rows, dtype=torch.int64)
    ].tolist()
    if list(map(int, metadata["row_positions"])) != list(
        map(int, authoritative_positions)
    ):
        raise ValueError(f"{leaf['path']}: scratch positions differ from L20")
    query_start = logical_mapping.get("query_start_loc")
    request_slots = logical_mapping.get("request_state_slots")
    if (
        not isinstance(query_start, list)
        or not isinstance(request_slots, list)
        or len(query_start) != len(request_slots) + 1
    ):
        raise ValueError(f"{leaf['path']}: scratch request mapping is absent")
    expected_requests = [
        next(
            (
                request
                for request in range(len(request_slots))
                if int(query_start[request]) <= index < int(query_start[request + 1])
            ),
            -1,
        )
        for index in input_rows
    ]
    if expected_requests != list(map(int, metadata["row_request_ordinals"])) or [
        int(request_slots[request]) if request >= 0 else -1
        for request in expected_requests
    ] != list(map(int, metadata["row_request_state_slots"])):
        raise ValueError(f"{leaf['path']}: scratch request ownership differs")

    cache_owner = metadata.get("cache_owner")
    producer_layer = metadata.get("producer_layer")
    producer_key = (
        "candidate" if leaf["comparison"] == "canonical_membership" else "topk"
    )
    producer_root = f"l20:cache_runtime_authority/scratch_producers/{producer_key}"
    for field in (
        "producer_layer",
        "cache_owner",
        "compress_ratio",
        "candidate_block_size",
    ):
        authority_leaf = run.leaves.get(f"{producer_root}/{field}")
        authority_value = (
            None if authority_leaf is None else _exact_value(run, authority_leaf)
        )
        if authority_value != metadata.get(field):
            raise ValueError(f"{leaf['path']}: scratch producer authority differs")
    matching_views = [
        view
        for view in views
        if isinstance(view, dict) and view.get("layer_name") == cache_owner
    ]
    if (
        not isinstance(cache_owner, str)
        or not isinstance(producer_layer, int)
        or len(matching_views) != 1
        or int(matching_views[0].get("tokens_per_state", -1))
        != int(metadata.get("compress_ratio", -1))
    ):
        raise ValueError(f"{leaf['path']}: scratch cache ownership differs")

    def layer_id(name: Any) -> int | None:
        if not isinstance(name, str):
            return None
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    owner_view = matching_views[0]
    if producer_layer not in {
        layer_id(name)
        for name in [owner_view.get("layer_name"), *owner_view.get("consumers", [])]
    }:
        raise ValueError(f"{leaf['path']}: scratch producer ownership differs")

    canonical = torch.full_like(tensor, -1)
    for row in range(rows):
        values = tensor[row]
        invalid = values < 0
        if bool((values[invalid] != -1).any()):
            raise ValueError(f"{leaf['path']}: invalid negative scratch ID")
        invalid_indices = torch.nonzero(invalid, as_tuple=False).flatten()
        if invalid_indices.numel():
            first = int(invalid_indices[0])
            if bool((values[first:] != -1).any()):
                raise ValueError(f"{leaf['path']}: scratch sentinels are not trailing")
            valid = values[:first]
        else:
            valid = values
        expected_count = int(metadata["expected_valid_counts"][row])
        bound = int(metadata["valid_id_exclusive_bounds"][row])
        position = int(metadata["row_positions"][row])
        ratio = int(metadata["compress_ratio"])
        visible_states = (position + 1) // ratio
        expected_bound = (
            (visible_states + int(metadata["candidate_block_size"]) - 1)
            // int(metadata["candidate_block_size"])
            if leaf["comparison"] == "canonical_membership"
            else visible_states
        )
        if bound != expected_bound:
            raise ValueError(f"{leaf['path']}: scratch visibility bound differs")
        if leaf["comparison"] != "diagnostic_topk":
            derived_count = min(tensor.shape[1], expected_bound)
            if expected_count != derived_count or valid.numel() != derived_count:
                raise ValueError(f"{leaf['path']}: scratch valid count differs")
        if bound < 0 or bool((valid >= bound).any()):
            raise ValueError(f"{leaf['path']}: scratch ID exceeds visibility")
        if torch.unique(valid).numel() != valid.numel():
            raise ValueError(f"{leaf['path']}: duplicate scratch ID")
        canonical[row, : valid.numel()] = torch.sort(valid).values
    if leaf["comparison"] == "diagnostic_topk":
        candidate_path = metadata["candidate_membership_path"]
        candidate_key = f"{leaf['checkpoint']}:{candidate_path}"
        candidate_leaf = run.leaves.get(candidate_key)
        if (
            candidate_leaf is None
            or candidate_leaf.get("comparison") != "canonical_membership"
        ):
            raise ValueError(f"{leaf['path']}: candidate membership is absent")
        candidates = _validated_scratch_rows(run, candidate_leaf)
        if candidates.shape[0] != rows:
            raise ValueError(f"{leaf['path']}: candidate row identity differs")
        candidate_metadata = candidate_leaf["metadata"]
        for field in (
            "row_positions",
            "row_request_ordinals",
            "row_request_state_slots",
        ):
            if candidate_metadata.get(field) != metadata.get(field):
                raise ValueError(f"{leaf['path']}: candidate ownership differs")
        block_size = int(metadata["candidate_block_size"])
        if block_size <= 0:
            raise ValueError(f"{leaf['path']}: candidate block size is invalid")
        for row in range(rows):
            candidate_blocks = {
                int(value) for value in candidates[row] if int(value) >= 0
            }
            selected = canonical[row]
            bound = int(metadata["valid_id_exclusive_bounds"][row])
            legal = {
                value
                for candidate_block in candidate_blocks
                for value in range(
                    candidate_block * block_size,
                    min((candidate_block + 1) * block_size, bound),
                )
            }
            expected = min(tensor.shape[1], len(legal))
            if (
                int(metadata["expected_valid_counts"][row]) != expected
                or int((selected >= 0).sum()) != expected
            ):
                raise ValueError(
                    f"{leaf['path']}: candidate-limited top-k count differs"
                )
            if any(int(value) >= 0 and int(value) not in legal for value in selected):
                raise ValueError(
                    f"{leaf['path']}: top-k ID is outside candidate membership"
                )
    return canonical


def _floating_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    denominator_floor: float,
    allow_negative_infinity: bool = False,
) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("floating tensor shape or dtype mismatch")
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    if bool(torch.isnan(left64).any()) or bool(torch.isnan(right64).any()):
        raise ValueError("unexpected NaN in floating evidence")
    left_positive = torch.isposinf(left64)
    right_positive = torch.isposinf(right64)
    if bool(left_positive.any()) or bool(right_positive.any()):
        raise ValueError("unexpected positive infinity in floating evidence")
    left_negative = torch.isneginf(left64)
    right_negative = torch.isneginf(right64)
    if not torch.equal(left_negative, right_negative):
        raise ValueError("negative-infinity masks differ")
    if not allow_negative_infinity and bool(left_negative.any()):
        raise ValueError("unexpected negative infinity in floating evidence")
    left_finite = torch.isfinite(left64)
    right_finite = torch.isfinite(right64)
    if not torch.equal(left_finite, right_finite):
        raise ValueError("finite masks differ")
    flat_left = left64.reshape(-1)
    flat_right = right64.reshape(-1)
    finite_indices = torch.nonzero(left_finite.reshape(-1), as_tuple=False).flatten()
    finite_left = flat_left.index_select(0, finite_indices)
    finite_right = flat_right.index_select(0, finite_indices)
    difference = (finite_left - finite_right).abs()
    if difference.numel() == 0:
        return {
            "elements": 0,
            "positive_infinities": int(torch.isposinf(left64).sum().item()),
            "negative_infinities": int(torch.isneginf(left64).sum().item()),
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "abs_quantiles": {str(q): 0.0 for q in QUANTILES},
            "worst_flat_index": None,
            "worst_left": None,
            "worst_right": None,
        }
    max_abs, worst_index = difference.max(dim=0)
    denominator = max(
        float(torch.linalg.vector_norm(finite_left).item()),
        float(torch.linalg.vector_norm(finite_right).item()),
        denominator_floor,
    )
    relative_l2 = float(torch.linalg.vector_norm(difference).item()) / denominator
    quantiles = torch.quantile(
        difference,
        torch.tensor(QUANTILES, dtype=torch.float64),
    )
    index = int(finite_indices[int(worst_index.item())].item())
    return {
        "elements": difference.numel(),
        "positive_infinities": int(torch.isposinf(left64).sum().item()),
        "negative_infinities": int(torch.isneginf(left64).sum().item()),
        "max_abs": float(max_abs.item()),
        "relative_l2": relative_l2,
        "abs_quantiles": {
            str(q): float(value) for q, value in zip(QUANTILES, quantiles.tolist())
        },
        "worst_flat_index": index,
        "worst_left": float(flat_left[index].item()),
        "worst_right": float(flat_right[index].item()),
    }


def _compare_runs(
    left: EvidenceRun,
    right: EvidenceRun,
    *,
    denominator_floor: float,
    envelope: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    if left.rank != right.rank:
        raise ValueError("comparisons must stay within one DP rank")
    if set(left.leaves) != set(right.leaves):
        missing_left = sorted(set(right.leaves) - set(left.leaves))
        missing_right = sorted(set(left.leaves) - set(right.leaves))
        raise ValueError(
            f"leaf schema differs: missing_left={missing_left[:4]} "
            f"missing_right={missing_right[:4]}"
        )
    floating = []
    diagnostic = []
    exact_mismatches = []
    for key in sorted(left.leaves):
        left_leaf = left.leaves[key]
        right_leaf = right.leaves[key]
        if _leaf_schema(left_leaf) != _leaf_schema(right_leaf):
            raise ValueError(f"{key}: leaf schema differs")
        if left_leaf["comparison"] == "exact":
            left_value = _exact_value(left, left_leaf)
            right_value = _exact_value(right, right_leaf)
            equal = (
                torch.equal(left_value, right_value)
                if isinstance(left_value, torch.Tensor)
                and isinstance(right_value, torch.Tensor)
                else left_value == right_value
            )
            if not equal:
                exact_mismatches.append(key)
            continue
        if left_leaf["comparison"] == "diagnostic":
            left_tensor = _meaningful_tensor(left, left_leaf)
            right_tensor = _meaningful_tensor(right, right_leaf)
            if left_tensor.shape != right_tensor.shape or left_tensor.dtype != (
                right_tensor.dtype
            ):
                raise ValueError(f"{key}: diagnostic tensor shape or dtype differs")
            mismatches = int(torch.count_nonzero(left_tensor != right_tensor).item())
            diagnostic.append(
                {
                    "leaf": key,
                    "elements": left_tensor.numel(),
                    "mismatches": mismatches,
                }
            )
            continue
        if left_leaf["comparison"] in (
            "canonical_membership",
            "diagnostic_topk",
        ):
            left_tensor = _validated_scratch_rows(left, left_leaf)
            right_tensor = _validated_scratch_rows(right, right_leaf)
            equal = torch.equal(left_tensor, right_tensor)
            if left_leaf["comparison"] == "canonical_membership":
                if not equal:
                    exact_mismatches.append(key)
            else:
                diagnostic.append(
                    {
                        "leaf": key,
                        "elements": left_tensor.numel(),
                        "mismatches": int(
                            torch.count_nonzero(left_tensor != right_tensor).item()
                        ),
                    }
                )
            continue
        if left_leaf["comparison"] != "floating":
            raise ValueError(f"{key}: invalid comparison class")
        left_tensor = _meaningful_tensor(left, left_leaf)
        right_tensor = _meaningful_tensor(right, right_leaf)
        metrics = _floating_metrics(
            left_tensor,
            right_tensor,
            denominator_floor=denominator_floor,
            allow_negative_infinity=(
                left_leaf["path"] in MASKED_NEGATIVE_INFINITY_PATHS
            ),
        )
        family = _envelope_key(left_leaf)
        passed = None
        if envelope is not None:
            threshold = envelope.get(family)
            if threshold is None:
                raise ValueError(f"{key}: frozen envelope is absent")
            passed = (
                metrics["max_abs"] <= threshold["max_abs"]
                and metrics["relative_l2"] <= threshold["relative_l2"]
            )
        floating.append(
            {
                "leaf": key,
                "envelope_key": family,
                "passed": passed,
                **metrics,
            }
        )
    return {
        "left": {
            "kind": left.kind,
            "rank": left.rank,
            "ordinal": left.ordinal,
        },
        "right": {
            "kind": right.kind,
            "rank": right.rank,
            "ordinal": right.ordinal,
        },
        "exact_mismatches": exact_mismatches,
        "floating": floating,
        "diagnostic": diagnostic,
        "passed": not exact_mismatches
        and all(row["passed"] is not False for row in floating),
    }


def _require_matrix(
    runs: dict[tuple[str, int, int], EvidenceRun],
    *,
    a_runs: int,
) -> list[int]:
    ranks = sorted({rank for kind, rank, _ in runs if kind == "a"})
    if not ranks:
        raise ValueError("no A-control ranks found")
    expected_ordinals = set(range(1, a_runs + 1))
    for rank in ranks:
        found = {
            ordinal
            for kind, candidate_rank, ordinal in runs
            if kind == "a" and candidate_rank == rank
        }
        if found != expected_ordinals:
            raise ValueError(
                f"rank {rank}: expected A ordinals {sorted(expected_ordinals)}, "
                f"found {sorted(found)}"
            )
    return ranks


def _calibration_provenance(
    runs: dict[tuple[str, int, int], EvidenceRun],
    summaries: dict[int, dict[str, Any]],
    ranks: list[int],
) -> dict[str, Any]:
    representative = runs[("a", ranks[0], 1)].manifest
    identity_normalization = _identity_normalization_provenance(runs)
    return {
        "run_id": representative["run_id"],
        "config_digest": identity_normalization["canonical_config_digest"],
        "identity_normalization": identity_normalization,
        "code_hashes": representative["code_hashes"],
        "model": representative["model"],
        "draft_model": representative["draft_model"],
        "dp_size": representative["dp_size"],
        "ranks": ranks,
        "manifests": [
            {
                "rank": run.rank,
                "ordinal": run.ordinal,
                "sha256": _sha256(run.manifest_path),
            }
            for run in sorted(
                runs.values(),
                key=lambda item: (item.rank, item.ordinal),
            )
        ],
        "summaries": [
            {
                "rank": rank,
                "run_id": summaries[rank]["run_id"],
                "fixture_digest": summaries[rank]["fixture_digest"],
                "worker_incarnation": summaries[rank]["worker_incarnation"],
            }
            for rank in ranks
        ],
    }


def calibrate(args: argparse.Namespace) -> None:
    if args.calibration_runs != 10:
        raise ValueError("calibration must use exactly A1-A10")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite frozen envelope: {args.output}")
    runs = _load_directory(
        args.input_dir,
        allowed_kinds={"a"},
        max_a_ordinal=args.calibration_runs,
    )
    ranks = _require_matrix(runs, a_runs=args.calibration_runs)
    summaries = _load_summaries(args.input_dir)
    _validate_cohort(
        runs,
        summaries,
        ranks,
        require_complete_files=False,
    )
    envelope: dict[str, dict[str, float]] = {}
    pair_reports = []
    for rank in ranks:
        controls = [
            runs[("a", rank, ordinal)]
            for ordinal in range(1, args.calibration_runs + 1)
        ]
        for left, right in itertools.combinations(controls, 2):
            report = _compare_runs(
                left,
                right,
                denominator_floor=args.denominator_floor,
                envelope=None,
            )
            if report["exact_mismatches"]:
                raise ValueError(
                    f"calibration exact mismatch rank={rank}: "
                    f"{report['exact_mismatches'][:4]}"
                )
            pair_reports.append(report)
            for row in report["floating"]:
                threshold = envelope.setdefault(
                    row["envelope_key"],
                    {"max_abs": 0.0, "relative_l2": 0.0},
                )
                threshold["max_abs"] = max(threshold["max_abs"], row["max_abs"])
                threshold["relative_l2"] = max(
                    threshold["relative_l2"],
                    row["relative_l2"],
                )
    document = {
        "schema": SCHEMA,
        "comparator_sha256": _sha256(Path(__file__)),
        "calibration_runs": args.calibration_runs,
        "calibration_pairs_per_rank": (
            args.calibration_runs * (args.calibration_runs - 1) // 2
        ),
        "ranks": ranks,
        "provenance": _calibration_provenance(runs, summaries, ranks),
        "denominator_floor": args.denominator_floor,
        "envelope": envelope,
        "pairs": pair_reports,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True).encode()
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": "calibrated",
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "families": len(envelope),
                "pairs": len(pair_reports),
            },
            sort_keys=True,
        )
    )


def validate(args: argparse.Namespace) -> None:
    if args.a_runs != 20:
        raise ValueError("validation must use exactly 20 A controls")
    envelope_document = json.loads(args.envelope.read_text())
    if envelope_document.get("schema") != SCHEMA:
        raise ValueError("unsupported envelope schema")
    comparator_sha256 = _sha256(Path(__file__))
    if envelope_document.get("comparator_sha256") != comparator_sha256:
        raise ValueError("frozen envelope comparator revision differs")
    calibration_runs = int(envelope_document["calibration_runs"])
    if calibration_runs != 10:
        raise ValueError("frozen envelope was not calibrated from A1-A10")
    denominator_floor = float(envelope_document["denominator_floor"])
    envelope = envelope_document["envelope"]
    provenance = envelope_document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("frozen envelope provenance is absent")
    runs = _load_directory(args.input_dir)
    ranks = _require_matrix(runs, a_runs=args.a_runs)
    summaries = _load_summaries(args.input_dir)
    _validate_cohort(
        runs,
        summaries,
        ranks,
        require_complete_files=True,
    )
    calibration_subset = {
        key: run
        for key, run in runs.items()
        if run.kind == "a" and 1 <= run.ordinal <= calibration_runs
    }
    current_provenance = _calibration_provenance(
        calibration_subset,
        summaries,
        ranks,
    )
    if current_provenance != provenance:
        raise ValueError("frozen envelope provenance does not match evidence")
    reports = []
    failures = []
    for rank in ranks:
        controls = [runs[("a", rank, ordinal)] for ordinal in range(1, args.a_runs + 1)]
        production = runs.get(("production", rank, 0))
        if production is None:
            raise ValueError(f"rank {rank}: production evidence is absent")
        for left, right in itertools.combinations(controls, 2):
            report = _compare_runs(
                left,
                right,
                denominator_floor=denominator_floor,
                envelope=envelope,
            )
            reports.append(report)
            if not report["passed"]:
                failures.append(report)
        for control in controls:
            report = _compare_runs(
                production,
                control,
                denominator_floor=denominator_floor,
                envelope=envelope,
            )
            reports.append(report)
            if not report["passed"]:
                failures.append(report)
    expected_pairs = args.a_runs * (args.a_runs - 1) // 2
    document = {
        "schema": SCHEMA,
        "comparator_sha256": comparator_sha256,
        "status": "passed" if not failures else "failed",
        "a_runs": args.a_runs,
        "calibration_runs": calibration_runs,
        "a_pairs_per_rank": expected_pairs,
        "production_comparisons_per_rank": args.a_runs,
        "ranks": ranks,
        "envelope_path": str(args.envelope),
        "envelope_sha256": _sha256(args.envelope),
        "failures": failures,
        "comparisons": reports,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(
            f"validation failed: {len(failures)} comparison(s); report={args.output}"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "ranks": ranks,
                "a_pairs_per_rank": expected_pairs,
                "production_comparisons_per_rank": args.a_runs,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--input-dir", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--calibration-runs", type=int, default=10)
    calibrate_parser.add_argument("--denominator-floor", type=float, default=1e-12)
    calibrate_parser.set_defaults(func=calibrate)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument("--envelope", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--a-runs", type=int, default=20)
    validate_parser.set_defaults(func=validate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
