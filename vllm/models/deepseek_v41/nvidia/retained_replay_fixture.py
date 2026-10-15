# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off retained replay fixture for the DeepSeek V4.1 L20 harness."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
    from vllm.models.deepseek_v41.nvidia.model import DecoderBoundaryState
    from vllm.models.deepseek_v41.nvidia.model_state import (
        OrdinaryDecodeBatch,
        RetainedReplayBatch,
    )
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class RetainedReplayFixtureError(RuntimeError):
    pass


_ByteInterval = tuple[int, int]

_PACKED_CACHE_FORMATS = {
    584: "v4_ue8m0_fp8_bf16",
    528: "v41_ue8m0_mxfp8",
    288: "v41_nvfp4",
    132: "indexer_fp8",
    68: "indexer_mxfp4",
}
_CACHE_RECORD_LAYOUTS = {
    "v4_ue8m0_fp8_bf16": (576, 8, 512),
    "v41_ue8m0_mxfp8": (512, 16, 512),
    "v41_nvfp4": (256, 32, 512),
    "indexer_fp8": (128, 4, 128),
    "indexer_mxfp4": (64, 4, 128),
    "plain_bf16": (1024, 0, 512),
    "plain_fp8": (512, 0, 512),
    "compressor_ring_fp32": (4096, 0, 1024),
}


def _clone_flashmla_scheduler(value: Any, memo: dict[int, Any]) -> Any | None:
    if type(value).__name__ != "FlashMLASchedMeta":
        return None
    from vllm.v1.attention.ops.flashmla import get_mla_metadata

    clone = get_mla_metadata()[0]
    memo[id(value)] = clone
    for name in ("tile_scheduler_metadata", "num_splits", "have_initialized"):
        if hasattr(value, name):
            setattr(clone, name, clone_owned(getattr(value, name), memo))
    return clone


def clone_owned(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Clone mutable tensor/array leaves while preserving immutable owners."""
    if memo is None:
        memo = {}
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        clone = value.clone()
        memo[value_id] = clone
        return clone
    if isinstance(value, np.ndarray):
        clone = value.copy()
        memo[value_id] = clone
        return clone
    scheduler = _clone_flashmla_scheduler(value, memo)
    if scheduler is not None:
        return scheduler
    if is_dataclass(value) and not isinstance(value, type):
        clone = copy.copy(value)
        memo[value_id] = clone
        for field in fields(value):
            object.__setattr__(
                clone,
                field.name,
                clone_owned(getattr(value, field.name), memo),
            )
        return clone
    if isinstance(value, dict):
        clone: dict[Any, Any] = {}
        memo[value_id] = clone
        clone.update(
            {
                clone_owned(key, memo): clone_owned(item, memo)
                for key, item in value.items()
            }
        )
        return clone
    if isinstance(value, list):
        clone_list: list[Any] = []
        memo[value_id] = clone_list
        clone_list.extend(clone_owned(item, memo) for item in value)
        return clone_list
    if isinstance(value, tuple):
        clone_tuple = (
            type(value)(*(clone_owned(item, memo) for item in value))
            if hasattr(value, "_fields")
            else tuple(clone_owned(item, memo) for item in value)
        )
        memo[value_id] = clone_tuple
        return clone_tuple
    if isinstance(value, set):
        clone_set = {clone_owned(item, memo) for item in value}
        memo[value_id] = clone_set
        return clone_set
    return value


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


def _storage_bytes_view(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _merge_byte_intervals(
    intervals: list[_ByteInterval],
) -> list[_ByteInterval]:
    merged: list[_ByteInterval] = []
    for begin, end in sorted(intervals):
        if begin < 0 or end < begin:
            raise RetainedReplayFixtureError(
                f"invalid storage byte interval [{begin}, {end})"
            )
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def _subtract_byte_intervals(
    intervals: list[_ByteInterval],
    excluded: list[_ByteInterval],
) -> list[_ByteInterval]:
    intervals = _merge_byte_intervals(intervals)
    excluded = _merge_byte_intervals(excluded)
    result: list[_ByteInterval] = []
    excluded_idx = 0
    for begin, end in intervals:
        while excluded_idx < len(excluded) and excluded[excluded_idx][1] <= begin:
            excluded_idx += 1
        cursor = begin
        index = excluded_idx
        while index < len(excluded) and excluded[index][0] < end:
            excluded_begin, excluded_end = excluded[index]
            if excluded_begin > cursor:
                result.append((cursor, min(excluded_begin, end)))
            cursor = max(cursor, excluded_end)
            if cursor >= end:
                break
            index += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def _cache_record_format(cache: torch.Tensor) -> str:
    if cache.ndim != 3:
        raise RetainedReplayFixtureError(
            f"unsupported logical cache shape {tuple(cache.shape)}"
        )
    width = int(cache.shape[-1])
    if cache.dtype == torch.uint8:
        try:
            return _PACKED_CACHE_FORMATS[width]
        except KeyError as error:
            raise RetainedReplayFixtureError(
                f"unsupported packed cache record width {width}"
            ) from error
    if width == 512 and cache.dtype == torch.bfloat16:
        return "plain_bf16"
    if width == 512 and cache.dtype == torch.float8_e4m3fn:
        return "plain_fp8"
    if width == 1024 and cache.dtype == torch.float32:
        return "compressor_ring_fp32"
    raise RetainedReplayFixtureError(
        f"unsupported cache record dtype/width {cache.dtype}/{width}"
    )


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.int64)
    magnitude_code = codes & 7
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    values = magnitudes.index_select(0, magnitude_code.reshape(-1)).reshape(
        magnitude_code.shape
    )
    return torch.where(codes >= 8, -values, values)


def _decode_cache_record_cpu(
    cache: torch.Tensor,
    block: int,
    row: int,
    *,
    record_format: str | None = None,
    fp8_scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy one actual cache record and decode its stored value to FP32."""
    if cache.ndim != 3:
        raise RetainedReplayFixtureError(
            f"unsupported logical cache shape {tuple(cache.shape)}"
        )
    if not 0 <= block < cache.shape[0] or not 0 <= row < cache.shape[1]:
        raise RetainedReplayFixtureError(
            f"cache record ({block}, {row}) exceeds {tuple(cache.shape[:2])}"
        )
    record_format = record_format or _cache_record_format(cache)

    if record_format in ("plain_bf16", "plain_fp8", "compressor_ring_fp32"):
        raw_row = cache[block, row].detach().contiguous().cpu()
        raw_values = raw_row.view(torch.uint8).clone()
        raw_scales = torch.empty(0, dtype=torch.uint8)
        decoded = raw_row.to(torch.float32)
        if record_format == "plain_fp8":
            if fp8_scale is None:
                raise RetainedReplayFixtureError(
                    "plain FP8 cache record lacks its dequantization scale"
                )
            scale = torch.as_tensor(fp8_scale, dtype=torch.float32).reshape(-1)
            if scale.numel() != 1 or not bool(torch.isfinite(scale).all()):
                raise RetainedReplayFixtureError(
                    "plain FP8 cache scale must be one finite scalar"
                )
            decoded = decoded * scale.item()
    else:
        if cache.dtype != torch.uint8:
            raise RetainedReplayFixtureError(
                f"{record_format}: packed cache must use uint8 storage"
            )
        page = (cache.storage_offset() + block * cache.stride(0)) * cache.element_size()
        rows_per_page = int(cache.shape[1])
        storage = _storage_bytes_view(cache)

        if record_format == "v4_ue8m0_fp8_bf16":
            value_bytes, scale_bytes = 576, 8
        elif record_format == "v41_ue8m0_mxfp8":
            value_bytes, scale_bytes = 512, 16
        elif record_format == "v41_nvfp4":
            value_bytes, scale_bytes = 256, 32
        elif record_format == "indexer_fp8":
            value_bytes, scale_bytes = 128, 4
        elif record_format == "indexer_mxfp4":
            value_bytes, scale_bytes = 64, 4
        else:
            raise RetainedReplayFixtureError(
                f"unsupported packed cache format {record_format}"
            )

        value_begin = page + row * value_bytes
        scale_begin = page + rows_per_page * value_bytes + row * scale_bytes
        raw_values = (
            storage[value_begin : value_begin + value_bytes].detach().cpu().clone()
        )
        raw_scales = (
            storage[scale_begin : scale_begin + scale_bytes].detach().cpu().clone()
        )

        if record_format == "v4_ue8m0_fp8_bf16":
            fp8 = raw_values[:448].view(torch.float8_e4m3fn).to(torch.float32)
            scales = torch.exp2(raw_scales[:7].to(torch.float32) - 127.0)
            nope = fp8.reshape(7, 64) * scales[:, None]
            rope = raw_values[448:].view(torch.bfloat16).to(torch.float32)
            decoded = torch.cat((nope.reshape(-1), rope))
            if int(raw_scales[7]) != 0:
                raise RetainedReplayFixtureError(
                    "V4 packed cache padding scale is nonzero"
                )
        elif record_format == "v41_ue8m0_mxfp8":
            fp8 = raw_values.view(torch.float8_e4m3fn).to(torch.float32)
            scales = torch.exp2(raw_scales.to(torch.float32) - 127.0)
            decoded = (fp8.reshape(16, 32) * scales[:, None]).reshape(-1)
        elif record_format == "v41_nvfp4":
            codes = torch.stack(
                (raw_values & 0xF, raw_values >> 4),
                dim=1,
            ).reshape(-1)
            scales = raw_scales.view(torch.float8_e4m3fn).to(torch.float32)
            decoded = (_decode_e2m1(codes).reshape(32, 16) * scales[:, None]).reshape(
                -1
            )
        elif record_format == "indexer_fp8":
            scale = raw_scales.view(torch.float32)
            if scale.numel() != 1:
                raise RetainedReplayFixtureError(
                    "indexer FP8 record has an invalid scale"
                )
            decoded = (
                raw_values.view(torch.float8_e4m3fn).to(torch.float32) * scale.item()
            )
        else:
            assert record_format == "indexer_mxfp4"
            codes = torch.stack(
                (raw_values & 0xF, raw_values >> 4),
                dim=1,
            ).reshape(-1)
            scales = torch.exp2(raw_scales.to(torch.float32) - 127.0)
            decoded = (_decode_e2m1(codes).reshape(4, 32) * scales[:, None]).reshape(-1)

    if not bool(torch.isfinite(decoded).all()):
        raise RetainedReplayFixtureError(
            f"{record_format}: decoded cache record is nonfinite"
        )
    return raw_values, raw_scales, decoded


def _decode_cache_records_cpu(
    cache: torch.Tensor,
    coordinates: list[tuple[int, int]],
    *,
    record_format: str,
    fp8_scale: torch.Tensor | float | None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not coordinates:
        return []
    blocks = sorted({block for block, _ in coordinates})
    block_indices = torch.tensor(
        blocks,
        dtype=torch.int64,
        device=cache.device,
    )
    copied = cache.index_select(0, block_indices).detach().cpu()
    block_rows = {block: index for index, block in enumerate(blocks)}
    scale = (
        None if fp8_scale is None else torch.as_tensor(fp8_scale).detach().float().cpu()
    )
    return [
        _decode_cache_record_cpu(
            copied,
            block_rows[block],
            row,
            record_format=record_format,
            fp8_scale=scale,
        )
        for block, row in coordinates
    ]


def _cache_block_byte_intervals(
    name: str,
    cache: torch.Tensor,
    blocks: set[int],
) -> list[_ByteInterval]:
    if cache.ndim != 3:
        raise RetainedReplayFixtureError(
            f"{name}: unsupported bound KV shape {tuple(cache.shape)}"
        )
    intervals: list[_ByteInterval] = []
    rows_per_block = cache.shape[1]
    for block in sorted(blocks):
        if block < 0 or block >= cache.shape[0]:
            raise RetainedReplayFixtureError(
                f"{name}: KV block {block} exceeds {cache.shape[0]} blocks"
            )
        block_view = cache[block]
        if block_view.is_contiguous():
            begin = (
                cache.storage_offset() + block * cache.stride(0)
            ) * cache.element_size()
            intervals.append((begin, begin + block_view.numel() * cache.element_size()))
        else:
            for row in range(rows_per_block):
                row_view = cache[block, row]
                if not row_view.is_contiguous():
                    raise RetainedReplayFixtureError(
                        f"{name}: KV block row is not contiguous"
                    )
                begin = (
                    cache.storage_offset()
                    + block * cache.stride(0)
                    + row * cache.stride(1)
                ) * cache.element_size()
                intervals.append(
                    (
                        begin,
                        begin + row_view.numel() * cache.element_size(),
                    )
                )
    return _merge_byte_intervals(intervals)


def _update_tensor_hash(digest: Any, tensor: torch.Tensor) -> None:
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    if tensor.numel() == 0:
        return
    contiguous = tensor.detach().contiguous()
    raw = contiguous.view(torch.uint8).reshape(-1)
    chunk_bytes = 16 * 1024 * 1024
    for start in range(0, raw.numel(), chunk_bytes):
        chunk = raw[start : start + chunk_bytes].cpu()
        digest.update(memoryview(chunk.numpy()))


def _byte_intervals_digest(
    tensor: torch.Tensor,
    intervals: list[_ByteInterval],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tensor")
    digest.update(str(torch.uint8).encode())
    total_bytes = sum(end - begin for begin, end in intervals)
    digest.update(str((total_bytes,)).encode())
    chunk_bytes = 16 * 1024 * 1024
    for begin, end in intervals:
        for start in range(begin, end, chunk_bytes):
            chunk = tensor[start : min(start + chunk_bytes, end)].cpu()
            digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def _tree_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            _update_tensor_hash(digest, item)
        elif isinstance(item, np.ndarray):
            digest.update(b"array")
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
        elif is_dataclass(item) and not isinstance(item, type):
            digest.update(type(item).__qualname__.encode())
            for field in fields(item):
                digest.update(field.name.encode())
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: repr(key)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, set):
            digest.update(b"set")
            for child in sorted(item, key=repr):
                visit(child)
        elif isinstance(item, (str, bytes, int, float, bool, type(None))):
            digest.update(repr(item).encode())
        else:
            digest.update(type(item).__qualname__.encode())

    visit(value)
    return digest.hexdigest()


def canonical_decision_reference(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> dict[str, Any]:
    sampled_counts = num_sampled.cpu().tolist()
    return {
        "sampled_token_ids": [
            sampled_token_ids[index, : int(count)].clone()
            for index, count in enumerate(sampled_counts)
        ],
        "num_sampled": num_sampled.clone(),
        "num_rejected": num_rejected.clone(),
    }


_TYPED_EVIDENCE_SCHEMA = 2
_FLOATING_REGISTRY_PREFIXES = (
    "model:_mtp_hidden_buffer",
    "model:candidate_compact_logits_buffer",
    "replay_graph:input:",
    "speculator:draft_token_confidence_probs",
    "speculator:_draft_scatter_buf",
    "speculator:hidden_states",
    "speculator:draft_logits",
    "speculator:model:_mtp_hidden_buffer",
    "acceptance:slope",
    "acceptance:intercepts",
    "acceptance:info",
    "acceptance:grad",
    "acceptance:features",
    "acceptance:predictions",
    "adaptive:confidence_probs",
    "adaptive:stale_gpu:",
    "adaptive:stale_cpu:",
)


def _safe_path_component(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _json_exact(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"float": repr(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.generic):
        return _json_exact(value.item())
    if isinstance(value, dict):
        return {
            str(key): _json_exact(item)
            for key, item in sorted(value.items(), key=lambda item: repr(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_exact(item) for item in value]
    if isinstance(value, set):
        return [_json_exact(item) for item in sorted(value, key=repr)]
    return {
        "type": type(value).__qualname__,
        "digest": _tree_digest(value),
    }


def _manifest_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _manifest_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _manifest_value(item)
            for key, item in sorted(value.items(), key=lambda item: repr(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)


def _model_config_identity(config: Any) -> Any:
    if config is None:
        return None
    hf_config = getattr(config, "hf_config", None)
    return {
        "model": _manifest_value(getattr(config, "model", None)),
        "revision": _manifest_value(getattr(config, "revision", None)),
        "code_revision": _manifest_value(getattr(config, "code_revision", None)),
        "tokenizer_revision": _manifest_value(
            getattr(config, "tokenizer_revision", None)
        ),
        "dtype": str(getattr(config, "dtype", None)),
        "quantization": _manifest_value(getattr(config, "quantization", None)),
        "max_model_len": _manifest_value(getattr(config, "max_model_len", None)),
        "architectures": _manifest_value(getattr(config, "architectures", None)),
        "compute_hash": (
            config.compute_hash()
            if callable(getattr(config, "compute_hash", None))
            else None
        ),
        "hf": {
            name: _manifest_value(getattr(hf_config, name, None))
            for name in (
                "model_type",
                "architectures",
                "vocab_size",
                "draft_vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "dspark_draft_topk",
                "dspark_target_layer_ids",
                "eagle_aux_hidden_state_layer_ids",
                "index_share_for_mtp_iteration",
            )
        },
    }


def _parallel_config_identity(config: Any) -> Any:
    if config is None:
        return None
    return {
        name: _manifest_value(getattr(config, name, None))
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "expert_parallel_size",
        )
    }


def _speculative_config_identity(config: Any) -> Any:
    if config is None:
        return None
    identity = {}
    for field in fields(config):
        name = field.name
        value = getattr(config, name)
        if name in ("target_model_config", "draft_model_config"):
            identity[name] = _model_config_identity(value)
        elif name in ("target_parallel_config", "draft_parallel_config"):
            identity[name] = _parallel_config_identity(value)
        else:
            identity[name] = _manifest_value(value)
    return identity


class _TypedSnapshotBuilder:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.tensors: dict[str, torch.Tensor] = {}
        self.leaves: list[dict[str, Any]] = []

    def add(
        self,
        value: Any,
        path: str,
        *,
        comparison: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().clone()
            payload_key = f"tensor_{len(self.tensors):06d}"
            self.tensors[payload_key] = tensor
            compare_as = comparison
            if compare_as is None:
                compare_as = (
                    "floating"
                    if tensor.is_floating_point() or tensor.is_complex()
                    else "exact"
                )
            self.leaves.append(
                {
                    "checkpoint": self.checkpoint,
                    "path": path,
                    "family": path,
                    "comparison": compare_as,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "payload_key": payload_key,
                    "metadata": metadata or {},
                }
            )
            return
        if isinstance(value, np.ndarray):
            self.add(
                torch.from_numpy(np.ascontiguousarray(value)),
                path,
                comparison=comparison,
                metadata=metadata,
            )
            return
        if is_dataclass(value) and not isinstance(value, type):
            self.add(type(value).__qualname__, f"{path}/__type__", comparison="exact")
            for field in fields(value):
                self.add(
                    getattr(value, field.name),
                    f"{path}/{_safe_path_component(field.name)}",
                    comparison=comparison,
                )
            return
        if isinstance(value, dict):
            self.add(
                [repr(key) for key in sorted(value, key=repr)],
                f"{path}/__keys__",
                comparison="exact",
            )
            for key in sorted(value, key=repr):
                self.add(
                    value[key],
                    f"{path}/{_safe_path_component(key)}",
                    comparison=comparison,
                )
            return
        if isinstance(value, (list, tuple)):
            self.add(type(value).__name__, f"{path}/__type__", comparison="exact")
            self.add(len(value), f"{path}/__len__", comparison="exact")
            for index, item in enumerate(value):
                self.add(item, f"{path}/{index}", comparison=comparison)
            return
        if isinstance(value, set):
            self.add(
                sorted((_json_exact(item) for item in value), key=repr),
                path,
                comparison="exact",
            )
            return
        self.leaves.append(
            {
                "checkpoint": self.checkpoint,
                "path": path,
                "family": path,
                "comparison": "exact",
                "dtype": type(value).__qualname__,
                "shape": None,
                "value": _json_exact(value),
                "metadata": metadata or {},
            }
        )

    def finish(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "leaves": self.leaves,
            "tensors": self.tensors,
        }


def typed_tree_snapshot(
    checkpoint: str,
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    builder = _TypedSnapshotBuilder(checkpoint)
    builder.add(value, path)
    return builder.finish()


def merge_typed_snapshots(
    checkpoint: str,
    *snapshots: dict[str, Any],
) -> dict[str, Any]:
    merged = {"checkpoint": checkpoint, "leaves": [], "tensors": {}}
    for snapshot_index, snapshot in enumerate(snapshots):
        key_map = {}
        for key, tensor in snapshot["tensors"].items():
            merged_key = f"part_{snapshot_index:02d}/{key}"
            key_map[key] = merged_key
            merged["tensors"][merged_key] = tensor
        for leaf in snapshot["leaves"]:
            leaf = dict(leaf)
            leaf["checkpoint"] = checkpoint
            if payload_key := leaf.get("payload_key"):
                leaf["payload_key"] = key_map[payload_key]
            merged["leaves"].append(leaf)
    return merged


def _registry_comparison(name: str, value: Any) -> str:
    if not isinstance(value, (torch.Tensor, np.ndarray)):
        return "exact"
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return "exact"
    return (
        "floating"
        if any(name.startswith(prefix) for prefix in _FLOATING_REGISTRY_PREFIXES)
        else "exact"
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_typed_run_bundle(
    output_dir: str,
    *,
    dp_rank: int,
    run_kind: str,
    ordinal: int,
    checkpoints: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    rank_dir = Path(output_dir) / f"rank-{dp_rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_kind}-{ordinal:03d}"
    payload_path = rank_dir / f"{stem}.pt"
    manifest_path = rank_dir / f"{stem}.json"

    payload: dict[str, torch.Tensor] = {}
    leaves: list[dict[str, Any]] = []
    checkpoint_names = []
    for checkpoint, snapshot in checkpoints.items():
        checkpoint_names.append(checkpoint)
        key_map = {}
        for key, tensor in snapshot["tensors"].items():
            merged_key = f"{checkpoint}/{key}"
            key_map[key] = merged_key
            payload[merged_key] = tensor
        for leaf in snapshot["leaves"]:
            leaf = dict(leaf)
            if payload_key := leaf.get("payload_key"):
                leaf["payload_key"] = key_map[payload_key]
            leaves.append(leaf)

    with tempfile.NamedTemporaryFile(
        dir=rank_dir, prefix=f".{stem}.", suffix=".pt", delete=False
    ) as temporary:
        temporary_payload = Path(temporary.name)
    try:
        torch.save(payload, temporary_payload)
        os.replace(temporary_payload, payload_path)
    finally:
        temporary_payload.unlink(missing_ok=True)
    payload_sha256 = _sha256_path(payload_path)

    document = {
        "schema": _TYPED_EVIDENCE_SCHEMA,
        **manifest,
        "run_kind": run_kind,
        "ordinal": ordinal,
        "dp_rank": dp_rank,
        "checkpoints": checkpoint_names,
        "payload": payload_path.name,
        "payload_sha256": payload_sha256,
        "leaves": leaves,
        "complete": True,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True).encode()
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_bytes(encoded)
    os.replace(temporary_manifest, manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha256,
    }


def write_typed_rank_summary(
    output_dir: str,
    *,
    dp_rank: int,
    document: dict[str, Any],
) -> dict[str, Any]:
    rank_dir = Path(output_dir) / f"rank-{dp_rank:02d}"
    path = rank_dir / "transaction-summary.json"
    encoded = json.dumps(
        {
            "schema": _TYPED_EVIDENCE_SCHEMA,
            "dp_rank": dp_rank,
            **document,
            "complete": True,
        },
        indent=2,
        sort_keys=True,
    ).encode()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return {
        "summary_path": str(path),
        "summary_sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass
class _TensorSelection:
    name: str
    tensor: torch.Tensor
    indices: torch.Tensor
    dim: int = 0
    s0: torch.Tensor | None = None
    s1: torch.Tensor | None = None
    live: torch.Tensor | None = None

    def _selected(self) -> torch.Tensor:
        return self.tensor.index_select(self.dim, self.indices)

    def capture(self, epoch: str, *, cpu: bool = False) -> None:
        value = self._selected().clone()
        if cpu:
            value = value.cpu()
        setattr(self, epoch, value)

    def restore(self, epoch: str) -> None:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        self.tensor.index_copy_(
            self.dim,
            self.indices,
            value.to(self.tensor.device, non_blocking=False),
        )

    def current_digest(self) -> str:
        return _tree_digest(self._selected())

    def epoch_digest(self, epoch: str) -> str:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        return _tree_digest(value)

    @property
    def bytes(self) -> int:
        assert self.s0 is not None
        return self.s0.numel() * self.s0.element_size()


@dataclass
class _TensorPoints:
    name: str
    tensor: torch.Tensor
    indices: torch.Tensor
    s0: torch.Tensor | None = None
    s1: torch.Tensor | None = None
    live: torch.Tensor | None = None

    def _flat(self) -> torch.Tensor:
        if not self.tensor.is_contiguous():
            raise RetainedReplayFixtureError(
                f"{self.name}: point tensor is not contiguous"
            )
        return self.tensor.view(-1)

    def _selected(self) -> torch.Tensor:
        return self._flat().index_select(0, self.indices)

    def capture(self, epoch: str, *, cpu: bool = False) -> None:
        value = self._selected().clone()
        if cpu:
            value = value.cpu()
        setattr(self, epoch, value)

    def restore(self, epoch: str) -> None:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        self._flat().index_copy_(
            0, self.indices, value.to(self.tensor.device, non_blocking=False)
        )

    def current_digest(self) -> str:
        return _tree_digest(self._selected())

    def epoch_digest(self, epoch: str) -> str:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        return _tree_digest(value)

    @property
    def bytes(self) -> int:
        assert self.s0 is not None
        return self.s0.numel() * self.s0.element_size()


@dataclass
class _TensorIntervals:
    name: str
    tensor: torch.Tensor
    intervals: list[_ByteInterval]
    s0: torch.Tensor | None = None
    s1: torch.Tensor | None = None
    live: torch.Tensor | None = None

    def capture(self, epoch: str, *, cpu: bool = False) -> None:
        device = torch.device("cpu") if cpu else self.tensor.device
        value = torch.empty(self.bytes, dtype=torch.uint8, device=device)
        offset = 0
        for begin, end in self.intervals:
            length = end - begin
            value[offset : offset + length].copy_(
                self.tensor[begin:end], non_blocking=False
            )
            offset += length
        setattr(self, epoch, value)

    def restore(self, epoch: str) -> None:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        offset = 0
        for begin, end in self.intervals:
            length = end - begin
            self.tensor[begin:end].copy_(
                value[offset : offset + length], non_blocking=False
            )
            offset += length

    def current_digest(self) -> str:
        return _byte_intervals_digest(self.tensor, self.intervals)

    def epoch_digest(self, epoch: str) -> str:
        value = getattr(self, epoch)
        if value is None:
            raise RetainedReplayFixtureError(f"{self.name}: epoch {epoch} is absent")
        return _tree_digest(value)

    @property
    def bytes(self) -> int:
        return sum(end - begin for begin, end in self.intervals)


@dataclass
class _ObjectAttribute:
    name: str
    owner: Any
    attribute: str
    s0: Any = None
    s1: Any = None
    live: Any = None

    def capture(self, epoch: str, *, cpu: bool = False) -> None:
        del cpu
        setattr(self, epoch, copy.deepcopy(getattr(self.owner, self.attribute)))

    def restore(self, epoch: str) -> None:
        setattr(self.owner, self.attribute, copy.deepcopy(getattr(self, epoch)))

    def current_digest(self) -> str:
        return _tree_digest(getattr(self.owner, self.attribute))

    def epoch_digest(self, epoch: str) -> str:
        return _tree_digest(getattr(self, epoch))

    @property
    def bytes(self) -> int:
        return 0


@dataclass
class _SourceRows:
    name: str
    tensor: torch.Tensor
    rows: torch.Tensor
    capture_digest: str

    @classmethod
    def capture(
        cls, name: str, tensor: torch.Tensor, rows: torch.Tensor
    ) -> _SourceRows:
        entry = cls(name, tensor, rows, "")
        entry.capture_digest = entry.current_digest()
        return entry

    def current_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.tensor.dtype).encode())
        digest.update(str((self.rows.numel(), *self.tensor.shape[1:])).encode())
        for start in range(0, self.rows.numel(), 4096):
            selected = self.tensor.index_select(0, self.rows[start : start + 4096])
            _update_tensor_hash(digest, selected)
        return digest.hexdigest()

    @property
    def bytes(self) -> int:
        if self.rows.numel() == 0:
            return 0
        row_bytes = self.tensor[0].numel() * self.tensor.element_size()
        return self.rows.numel() * row_bytes


@dataclass
class _SourceIntervals:
    name: str
    tensor: torch.Tensor
    intervals: list[_ByteInterval]
    capture_digest: str

    @classmethod
    def capture(
        cls,
        name: str,
        tensor: torch.Tensor,
        intervals: list[_ByteInterval],
    ) -> _SourceIntervals:
        entry = cls(name, tensor, intervals, "")
        entry.capture_digest = entry.current_digest()
        return entry

    def current_digest(self) -> str:
        return _byte_intervals_digest(self.tensor, self.intervals)

    @property
    def bytes(self) -> int:
        return sum(end - begin for begin, end in self.intervals)


class TouchedStateRegistry:
    def __init__(self) -> None:
        self.entries: list[
            _TensorSelection | _TensorPoints | _TensorIntervals | _ObjectAttribute
        ] = []
        self.sources: list[_SourceRows | _SourceIntervals] = []
        self._names: set[str] = set()
        self._meaningful_counts: dict[str, int] = {}
        self._comparison_overrides: dict[str, str] = {}
        self._selection_metadata: dict[str, dict[str, Any]] = {}
        self._cache_views: list[_LogicalCacheView] = []
        self._cache_writers: list[_CacheWriter] = []
        self._cache_block_tables: tuple[torch.Tensor, ...] = ()
        self._cache_base_extents: tuple[int, ...] = ()
        self._cache_query_lens: tuple[int, ...] = ()
        self._cache_request_slots: tuple[int, ...] = ()
        self._cache_s0_records: dict[
            str, dict[tuple[int, int, int, int, int], _LogicalCacheRecord]
        ] = {}
        self._cache_s0_raw_digests: dict[
            tuple[str, tuple[int, int, int, int, int]], str
        ] = {}

    def _claim(self, name: str) -> None:
        if name in self._names:
            raise RetainedReplayFixtureError(f"duplicate state entry: {name}")
        self._names.add(name)

    def add_rows(
        self,
        name: str,
        tensor: torch.Tensor,
        rows: torch.Tensor,
        *,
        dim: int = 0,
        meaningful_count: int | None = None,
    ) -> None:
        self._claim(name)
        rows = torch.unique(rows.to(device=tensor.device, dtype=torch.int64))
        entry = _TensorSelection(name, tensor, rows, dim)
        entry.capture("s0")
        self.entries.append(entry)
        if meaningful_count is not None:
            if dim != 0 or not 0 <= meaningful_count <= rows.numel():
                raise RetainedReplayFixtureError(
                    f"{name}: invalid meaningful row prefix"
                )
            self._meaningful_counts[name] = meaningful_count

    def add_points(
        self, name: str, tensor: torch.Tensor, indices: torch.Tensor
    ) -> None:
        self._claim(name)
        indices = torch.unique(indices.to(device=tensor.device, dtype=torch.int64))
        entry = _TensorPoints(name, tensor, indices)
        entry.capture("s0")
        self.entries.append(entry)

    def add_intervals(
        self,
        name: str,
        tensor: torch.Tensor,
        intervals: list[_ByteInterval],
    ) -> None:
        self._claim(name)
        intervals = _merge_byte_intervals(intervals)
        entry = _TensorIntervals(name, tensor, intervals)
        entry.capture("s0")
        self.entries.append(entry)

    def add_attribute(self, name: str, owner: Any, attribute: str) -> None:
        self._claim(name)
        entry = _ObjectAttribute(name, owner, attribute)
        entry.capture("s0")
        self.entries.append(entry)

    def add_source(self, name: str, tensor: torch.Tensor, rows: torch.Tensor) -> None:
        self._claim(name)
        rows = torch.unique(rows.to(device=tensor.device, dtype=torch.int64))
        self.sources.append(_SourceRows.capture(name, tensor, rows))

    def add_source_intervals(
        self,
        name: str,
        tensor: torch.Tensor,
        intervals: list[_ByteInterval],
    ) -> None:
        self._claim(name)
        intervals = _merge_byte_intervals(intervals)
        self.sources.append(_SourceIntervals.capture(name, tensor, intervals))

    def set_tensor_policy(
        self,
        name: str,
        *,
        comparison: str,
        compare_prefix: int,
        metadata: dict[str, Any],
    ) -> None:
        matching = [
            entry
            for entry in self.entries
            if isinstance(entry, _TensorSelection) and entry.name == name
        ]
        if len(matching) != 1:
            raise RetainedReplayFixtureError(
                f"{name}: scratch policy tensor is absent or ambiguous"
            )
        entry = matching[0]
        if entry.dim != 0 or not 0 < compare_prefix <= entry.indices.numel():
            raise RetainedReplayFixtureError(
                f"{name}: invalid scratch meaningful-row prefix"
            )
        if comparison not in ("canonical_membership", "diagnostic_topk"):
            raise RetainedReplayFixtureError(
                f"{name}: unsupported scratch comparison {comparison}"
            )
        self._meaningful_counts[name] = compare_prefix
        self._comparison_overrides[name] = comparison
        self._selection_metadata[name] = metadata

    def configure_logical_cache(
        self,
        views: list[_LogicalCacheView],
        writers: list[_CacheWriter],
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        base_extents: tuple[int, ...],
    ) -> None:
        if self._cache_views:
            raise RetainedReplayFixtureError(
                "logical cache evidence is configured twice"
            )
        num_reqs = int(input_batch.num_reqs)
        self._cache_views = sorted(views, key=lambda view: view.view_id)
        self._cache_writers = writers
        self._cache_block_tables = tuple(
            table[:num_reqs].detach().cpu().clone() for table in block_tables
        )
        if len(base_extents) != num_reqs:
            raise RetainedReplayFixtureError(
                "logical cache S0 extent cardinality differs"
            )
        self._cache_base_extents = base_extents
        self._cache_query_lens = tuple(
            int(value)
            for value in np.diff(
                input_batch.query_start_loc_np[: num_reqs + 1]
            ).tolist()
        )
        self._cache_request_slots = tuple(
            int(value)
            for value in input_batch.idx_mapping[:num_reqs].detach().cpu().tolist()
        )
        for view in self._cache_views:
            records = self._logical_records_for_extents(
                view,
                self._cache_base_extents,
                writer="s0",
            )
            self._cache_s0_records[view.view_id] = {
                record.identity: record for record in records
            }
            decoded_records = _decode_cache_records_cpu(
                view.tensor,
                [(record.physical_block, record.physical_row) for record in records],
                record_format=view.record_format,
                fp8_scale=view.fp8_scale,
            )
            for record, (raw_values, raw_scales, _) in zip(
                records,
                decoded_records,
            ):
                self._cache_s0_raw_digests[(view.view_id, record.identity)] = (
                    _tree_digest((raw_values, raw_scales))
                )

    def _logical_records_for_extents(
        self,
        view: _LogicalCacheView,
        extents: tuple[int, ...],
        *,
        writer: str,
    ) -> list[_LogicalCacheRecord]:
        table = self._cache_block_tables[view.group_id]
        records: list[_LogicalCacheRecord] = []
        for request_ordinal, extent in enumerate(extents):
            if extent <= 0:
                continue
            request_slot = self._cache_request_slots[request_ordinal]
            if view.cache_kind == "compressor_ring":
                physical_block = int(table[request_ordinal, 0])
                if physical_block <= 0 or physical_block not in view.destination_blocks:
                    continue
                open_rows = extent % view.tokens_per_state
                lower = extent - open_rows
                for position in range(lower, extent):
                    records.append(
                        _LogicalCacheRecord(
                            request_ordinal=request_ordinal,
                            request_state_slot=request_slot,
                            position=position,
                            physical_block=physical_block,
                            physical_row=position % view.block_size,
                            writer=writer,
                        )
                    )
                continue

            ratio = view.tokens_per_state
            if ratio <= 0:
                raise RetainedReplayFixtureError(
                    f"{view.view_id}: invalid tokens_per_state={ratio}"
                )
            for logical_block, physical_value in enumerate(
                table[request_ordinal].tolist()
            ):
                physical_block = int(physical_value)
                if physical_block <= 0 or physical_block not in view.destination_blocks:
                    continue
                for physical_row in range(view.block_size):
                    state_index = logical_block * view.block_size + physical_row
                    position = state_index * ratio + ratio - 1
                    if position >= extent:
                        break
                    if view.live_window is not None and position < max(
                        0, extent - view.live_window
                    ):
                        continue
                    records.append(
                        _LogicalCacheRecord(
                            request_ordinal=request_ordinal,
                            request_state_slot=request_slot,
                            position=position,
                            physical_block=physical_block,
                            physical_row=physical_row,
                            writer=writer,
                        )
                    )
        return records

    def _target_writer_records(
        self,
        view: _LogicalCacheView,
        extents: tuple[int, ...],
        writers: list[_CacheWriter],
    ) -> list[_LogicalCacheRecord]:
        table = self._cache_block_tables[view.group_id]
        records: dict[tuple[int, int, int, int, int], _LogicalCacheRecord] = {}
        for writer in writers:
            slots = writer.slots_by_layer.get(view.layer_name)
            if slots is None:
                continue
            count = min(
                writer.positions.numel(),
                writer.request_ordinals.numel(),
                writer.request_state_slots.numel(),
                slots.numel(),
            )
            for row in range(count):
                slot = int(slots[row])
                if slot < 0:
                    continue
                request_ordinal = int(writer.request_ordinals[row])
                request_slot = int(writer.request_state_slots[row])
                position = int(writer.positions[row])
                if not 0 <= request_ordinal < len(extents):
                    raise RetainedReplayFixtureError(
                        f"{writer.phase}: invalid request ordinal {request_ordinal}"
                    )
                if request_slot != self._cache_request_slots[request_ordinal]:
                    raise RetainedReplayFixtureError(
                        f"{writer.phase}: request state slot changed"
                    )
                if (
                    writer.enforce_committed_extent
                    and position >= extents[request_ordinal]
                ):
                    continue
                ratio = view.tokens_per_state
                if view.cache_kind != "compressor_ring":
                    if (position + 1) % ratio:
                        continue
                    state_index = position // ratio
                    logical_block = state_index // view.block_size
                    if logical_block >= table.shape[1]:
                        raise RetainedReplayFixtureError(
                            f"{writer.phase}: logical cache block is out of range"
                        )
                    expected_block = int(table[request_ordinal, logical_block])
                    expected_row = state_index % view.block_size
                else:
                    expected_block = int(table[request_ordinal, 0])
                    expected_row = position % view.block_size
                physical_block = slot // view.block_size
                physical_row = slot % view.block_size
                if (physical_block, physical_row) != (
                    expected_block,
                    expected_row,
                ):
                    raise RetainedReplayFixtureError(
                        f"{writer.phase}: {view.view_id} slot {slot} differs from "
                        f"block-table slot "
                        f"{expected_block * view.block_size + expected_row}"
                    )
                if physical_block not in view.destination_blocks:
                    raise RetainedReplayFixtureError(
                        f"{writer.phase}: {view.view_id} writer escaped destination "
                        f"coverage"
                    )
                record = _LogicalCacheRecord(
                    request_ordinal=request_ordinal,
                    request_state_slot=request_slot,
                    position=position,
                    physical_block=physical_block,
                    physical_row=physical_row,
                    writer=writer.phase,
                )
                records[record.identity] = record
        return list(records.values())

    def logical_cache_snapshot(
        self,
        checkpoint: str,
        num_rejected: torch.Tensor,
        *,
        extra_writers: list[_CacheWriter] | None = None,
    ) -> dict[str, Any]:
        rejected = tuple(int(value) for value in num_rejected.detach().cpu().tolist())
        if len(rejected) != len(self._cache_base_extents):
            raise RetainedReplayFixtureError(
                f"{checkpoint}: rejected-count request cardinality differs"
            )
        extents = tuple(
            base + query_len - rejected_count
            for base, query_len, rejected_count in zip(
                self._cache_base_extents,
                self._cache_query_lens,
                rejected,
            )
        )
        writers = [*self._cache_writers, *(extra_writers or ())]
        writer_indices = {writer.phase: index for index, writer in enumerate(writers)}
        if len(writer_indices) != len(writers):
            raise RetainedReplayFixtureError(
                f"{checkpoint}: logical writer phase identity is duplicated"
            )
        builder = _TypedSnapshotBuilder(checkpoint)
        for view in self._cache_views:
            records = dict(self._cache_s0_records[view.view_id])
            written = self._target_writer_records(view, extents, writers)
            for record in written:
                if view.cache_kind == "compressor_ring":
                    stale = [
                        identity
                        for identity, current in records.items()
                        if (
                            current.request_ordinal == record.request_ordinal
                            and current.physical_block == record.physical_block
                            and current.physical_row == record.physical_row
                        )
                    ]
                    for identity in stale:
                        del records[identity]
                records[record.identity] = record
            if view.cache_kind == "compressor_ring":
                ephemeral_phases = {
                    writer.phase
                    for writer in writers
                    if not writer.enforce_committed_extent
                }
                records = {
                    identity: record
                    for identity, record in records.items()
                    if (
                        record.writer in ephemeral_phases
                        or record.position
                        >= extents[record.request_ordinal]
                        - extents[record.request_ordinal] % view.tokens_per_state
                    )
                }
            ordered = sorted(records.values(), key=lambda record: record.identity)
            if not ordered and view.cache_kind != "compressor_ring":
                raise RetainedReplayFixtureError(
                    f"{view.view_id}: logical destination coverage is empty"
                )

            descriptors = (
                torch.tensor(
                    [
                        [
                            record.request_ordinal,
                            record.request_state_slot,
                            record.position,
                            record.physical_block,
                            record.physical_row,
                            (
                                writer_indices[record.writer]
                                if record.writer != "s0"
                                else -1
                            ),
                        ]
                        for record in ordered
                    ],
                    dtype=torch.int64,
                )
                if ordered
                else torch.empty((0, 6), dtype=torch.int64)
            )
            raw_values = []
            raw_scales = []
            decoded = []
            written_identities = {record.identity for record in written}
            decoded_records = _decode_cache_records_cpu(
                view.tensor,
                [(record.physical_block, record.physical_row) for record in ordered],
                record_format=view.record_format,
                fp8_scale=view.fp8_scale,
            )
            for record, (raw_value, raw_scale, value) in zip(
                ordered,
                decoded_records,
            ):
                if record.identity not in written_identities:
                    expected = self._cache_s0_raw_digests[
                        (view.view_id, record.identity)
                    ]
                    if _tree_digest((raw_value, raw_scale)) != expected:
                        raise RetainedReplayFixtureError(
                            f"{checkpoint}: unwritten live record changed in "
                            f"{view.view_id}"
                        )
                raw_values.append(raw_value)
                raw_scales.append(raw_scale)
                decoded.append(value)

            root = f"logical_kv/{_safe_path_component(view.view_id)}"
            metadata = {
                **view.metadata(),
                "descriptor_columns": [
                    "request_ordinal",
                    "request_state_slot",
                    "position",
                    "physical_block",
                    "physical_row",
                    "writer_index",
                ],
                "writers": [writer.phase for writer in writers],
            }
            builder.add(metadata, f"{root}/metadata", comparison="exact")
            builder.add(
                descriptors,
                f"{root}/descriptors",
                comparison="exact",
            )
            raw_width, scale_width, decoded_width = _CACHE_RECORD_LAYOUTS[
                view.record_format
            ]
            decoded_rows = (
                torch.stack(decoded)
                if decoded
                else torch.empty((0, decoded_width), dtype=torch.float32)
            )
            builder.add(
                decoded_rows,
                f"{root}/decoded",
                comparison="floating",
                metadata={"record_format": view.record_format},
            )
            raw_value_rows = (
                torch.stack(raw_values)
                if raw_values
                else torch.empty((0, raw_width), dtype=torch.uint8)
            )
            builder.add(
                raw_value_rows,
                f"{root}/raw_values",
                comparison="diagnostic",
                metadata={"record_format": view.record_format},
            )
            if raw_scales and raw_scales[0].numel():
                scale_rows = torch.stack(raw_scales)
            else:
                scale_rows = torch.empty(
                    (len(raw_scales), scale_width),
                    dtype=torch.uint8,
                )
            builder.add(
                scale_rows,
                f"{root}/raw_scales",
                comparison="diagnostic",
                metadata={"record_format": view.record_format},
            )
        return builder.finish()

    def capture_epoch(self, epoch: str, *, cpu: bool = False) -> None:
        for entry in self.entries:
            entry.capture(epoch, cpu=cpu)

    @torch.inference_mode()
    def restore(self, epoch: str) -> None:
        for entry in self.entries:
            entry.restore(epoch)

    def verify(self, epoch: str) -> None:
        mismatches = [
            entry.name
            for entry in self.entries
            if entry.current_digest() != entry.epoch_digest(epoch)
        ]
        if mismatches:
            raise RetainedReplayFixtureError(
                f"{epoch} restore mismatches: {mismatches[:8]}"
            )

    def digest(self, epoch: str | None = None) -> str:
        rows = []
        for entry in self.entries:
            value = (
                entry.current_digest() if epoch is None else entry.epoch_digest(epoch)
            )
            rows.append((entry.name, value))
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def typed_snapshot(
        self,
        checkpoint: str,
        *,
        epoch: str | None = None,
    ) -> dict[str, Any]:
        builder = _TypedSnapshotBuilder(checkpoint)
        for entry in self.entries:
            selection: dict[str, Any]
            if isinstance(entry, _TensorSelection):
                value = entry._selected() if epoch is None else getattr(entry, epoch)
                selection = {
                    "kind": "index_select",
                    "dim": entry.dim,
                    "indices": entry.indices.cpu().tolist(),
                }
                if entry.name in self._meaningful_counts:
                    selection["compare_prefix"] = self._meaningful_counts[entry.name]
                selection.update(self._selection_metadata.get(entry.name, {}))
            elif isinstance(entry, _TensorPoints):
                value = entry._selected() if epoch is None else getattr(entry, epoch)
                selection = {
                    "kind": "flat_points",
                    "indices": entry.indices.cpu().tolist(),
                }
            elif isinstance(entry, _TensorIntervals):
                if epoch is None:
                    value = torch.empty(entry.bytes, dtype=torch.uint8)
                    offset = 0
                    for begin, end in entry.intervals:
                        length = end - begin
                        value[offset : offset + length].copy_(
                            entry.tensor[begin:end], non_blocking=False
                        )
                        offset += length
                else:
                    value = getattr(entry, epoch)
                selection = {
                    "kind": "byte_intervals",
                    "intervals": entry.intervals,
                }
            else:
                assert isinstance(entry, _ObjectAttribute)
                value = (
                    getattr(entry.owner, entry.attribute)
                    if epoch is None
                    else getattr(entry, epoch)
                )
                selection = {
                    "kind": "attribute",
                    "attribute": entry.attribute,
                }
            if value is None and not isinstance(entry, _ObjectAttribute):
                raise RetainedReplayFixtureError(
                    f"{entry.name}: epoch {epoch} is absent"
                )
            comparison = self._comparison_overrides.get(entry.name)
            if comparison is None:
                comparison = (
                    "diagnostic"
                    if isinstance(entry, _TensorIntervals)
                    else _registry_comparison(entry.name, value)
                )
            builder.add(
                value,
                f"registry/{_safe_path_component(entry.name)}",
                comparison=comparison,
                metadata=selection,
            )
        return builder.finish()

    def source_digest(self) -> str:
        rows = [(source.name, source.current_digest()) for source in self.sources]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def source_capture_digest(self) -> str:
        rows = [(source.name, source.capture_digest) for source in self.sources]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    @property
    def snapshot_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)

    @property
    def source_bytes(self) -> int:
        return sum(source.bytes for source in self.sources)


@dataclass
class RetainedFixture:
    fixture_id: int
    input_batch: InputBatch
    model_inputs: dict[str, Any]
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    full_forward_context: ForwardContext
    replay_batch: RetainedReplayBatch
    ordinary_batch: OrdinaryDecodeBatch | None
    production_replay_batch: ReplayBatch
    dp_sync: DPSyncState | None
    is_dummy: bool
    registry: TouchedStateRegistry
    source_digest: str
    boundary: DecoderBoundaryState | None = None
    production_checkpoints: dict[str, dict[str, Any]] | None = None
    production_graph: dict[str, Any] | None = None
    reference: Any = None
    fixture_digest: str | None = None


@dataclass
class _CacheIntervals:
    name: str
    tensor: torch.Tensor
    destination: list[_ByteInterval]
    source: list[_ByteInterval]
    views: list[_LogicalCacheView]


@dataclass
class _LogicalCacheView:
    view_id: str
    storage_id: str
    group_id: int
    layer_name: str
    owner_type: str
    spec_type: str
    spec_metadata: dict[str, Any]
    cache_kind: str
    record_format: str
    tensor: torch.Tensor
    block_size: int
    tokens_per_state: int
    live_window: int | None
    destination_blocks: set[int]
    consumers: list[str]
    aliases: list[str]
    fp8_scale: torch.Tensor | None

    def metadata(self) -> dict[str, Any]:
        element_size = self.tensor.element_size()
        return {
            "schema": 1,
            "view_id": self.view_id,
            "storage_id": self.storage_id,
            "group_id": self.group_id,
            "layer_name": self.layer_name,
            "owner_type": self.owner_type,
            "spec_type": self.spec_type,
            "spec_metadata": self.spec_metadata,
            "cache_kind": self.cache_kind,
            "record_format": self.record_format,
            "dtype": str(self.tensor.dtype),
            "shape": list(self.tensor.shape),
            "strides_bytes": [
                int(stride) * element_size for stride in self.tensor.stride()
            ],
            "storage_offset_bytes": self.tensor.storage_offset() * element_size,
            "storage_nbytes": self.tensor.untyped_storage().nbytes(),
            "block_size": self.block_size,
            "tokens_per_state": self.tokens_per_state,
            "live_window": self.live_window,
            "destination_blocks": sorted(self.destination_blocks),
            "consumers": self.consumers,
            "aliases": self.aliases,
            "fp8_scale": (
                None
                if self.fp8_scale is None
                else float(self.fp8_scale.detach().float().cpu().reshape(-1).item())
            ),
        }


@dataclass(frozen=True)
class _LogicalCacheRecord:
    request_ordinal: int
    request_state_slot: int
    position: int
    physical_block: int
    physical_row: int
    writer: str

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (
            self.request_ordinal,
            self.request_state_slot,
            self.position,
            self.physical_block,
            self.physical_row,
        )


@dataclass
class _CacheWriter:
    phase: str
    positions: torch.Tensor
    request_ordinals: torch.Tensor
    request_state_slots: torch.Tensor
    slots_by_layer: dict[str, torch.Tensor]
    enforce_committed_extent: bool = True


class RetainedReplayFixtureController:
    def __init__(self, runner: GPUModelRunner) -> None:
        self.runner = runner
        self.state = "armed"
        self.error: str | None = None
        self.fixture: RetainedFixture | None = None
        self.fixture_id = 0
        self.verify_runs: list[dict[str, Any]] = []
        self.boundary_error: str | None = None
        self.a_control_runs: list[dict[str, Any]] = []
        self._active_checkpoints: dict[str, dict[str, Any]] | None = None
        self._active_checkpoint_error: str | None = None
        self._graph_observations: dict[str, dict[str, Any]] = {}
        self.worker_incarnation = (
            f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
        )

    def status(self) -> dict[str, Any]:
        fixture = self.fixture
        return {
            "state": self.state,
            "error": self.error,
            "dp_rank": self.runner.dp_rank,
            "fixture_id": None if fixture is None else fixture.fixture_id,
            "num_reqs": None if fixture is None else fixture.input_batch.num_reqs,
            "num_tokens": (None if fixture is None else fixture.input_batch.num_tokens),
            "replay_tokens": (
                None if fixture is None else fixture.replay_batch.input_batch.num_tokens
            ),
            "ordinary_tokens": (
                None
                if fixture is None or fixture.ordinary_batch is None
                else fixture.ordinary_batch.input_batch.num_tokens
            ),
            "is_dummy": None if fixture is None else fixture.is_dummy,
            "production_graph": (None if fixture is None else fixture.production_graph),
            "snapshot_bytes": (
                None if fixture is None else fixture.registry.snapshot_bytes
            ),
            "source_bytes": (
                None if fixture is None else fixture.registry.source_bytes
            ),
            "source_digest": (None if fixture is None else fixture.source_digest),
            "fixture_digest": (None if fixture is None else fixture.fixture_digest),
            "boundary_error": self.boundary_error,
            "checkpoint_error": self._active_checkpoint_error,
            "verify_runs": self.verify_runs,
            "a_control_runs": self.a_control_runs,
        }

    def fail(self, error: BaseException | str) -> None:
        self.state = "error"
        self.error = (
            error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        )

    @staticmethod
    def _production_graph_components(
        replay_batch: ReplayBatch,
    ) -> tuple[Any, Any, Any]:
        run_graph = replay_batch.run_graph
        manager = getattr(run_graph, "__self__", None)
        descriptor = replay_batch.forward_context.batch_descriptor
        if manager is None or getattr(run_graph, "__name__", None) != "run":
            raise RetainedReplayFixtureError(
                "production replay batch did not select the dedicated graph manager"
            )
        if replay_batch.forward_context.cudagraph_runtime_mode.name != "PIECEWISE":
            raise RetainedReplayFixtureError("production replay batch is not PIECEWISE")
        if descriptor is None:
            raise RetainedReplayFixtureError(
                "production replay graph descriptor is absent"
            )
        entry = manager.breakable_cg_runner.entries.get(descriptor)
        capture = None if entry is None else entry.capture
        if capture is None or capture.num_graphs < 1:
            raise RetainedReplayFixtureError(
                "selected production replay graph has no captured segment"
            )
        return manager, descriptor, capture

    def freeze_production_replay_batch(self) -> ReplayBatch:
        from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch

        layers = getattr(self.runner.model_state, "decoder_replay_layers", None)
        live = None if layers is None else layers.replay_batch
        if live is None:
            raise RetainedReplayFixtureError(
                "the capture step did not select a production replay batch"
            )
        decoder_model = getattr(self.runner.model_state, "_decoder_model", None)
        if getattr(decoder_model, "decoder_replay_front_graph", None) is not None:
            raise RetainedReplayFixtureError(
                "retained A-control does not support a selected front graph"
            )
        self._production_graph_components(live)
        return ReplayBatch(
            rows=clone_owned(live.rows),
            forward_context=clone_owned(live.forward_context),
            run_graph=live.run_graph,
        )

    def observed_production_replay_batch(
        self,
        frozen: ReplayBatch,
        phase: str,
    ) -> ReplayBatch:
        from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch

        manager, descriptor, capture = self._production_graph_components(frozen)
        observation = {
            "phase": phase,
            "descriptor_rows": descriptor.num_tokens,
            "descriptor_repr": repr(descriptor),
            "manager_id": id(manager),
            "capture_id": id(capture),
            "num_graphs": capture.num_graphs,
            "num_eager_breaks": capture.num_eager_breaks,
            "callback_calls": 0,
            "capture_replay_calls": 0,
            "error": None,
        }
        self._graph_observations[phase] = observation
        original_callback = frozen.run_graph
        assert original_callback is not None

        def observed_callback(
            rows: torch.Tensor,
            inputs: tuple[Any, ...],
        ) -> tuple[torch.Tensor, ...]:
            observation["callback_calls"] += 1
            try:
                current_manager, current_descriptor, current_capture = (
                    self._production_graph_components(frozen)
                )
                if (
                    current_manager is not manager
                    or current_descriptor != descriptor
                    or current_capture is not capture
                    or current_capture.num_graphs != observation["num_graphs"]
                ):
                    raise RetainedReplayFixtureError(
                        "the selected production replay graph changed"
                    )
                had_instance_replay = "replay" in capture.__dict__
                prior_instance_replay = capture.__dict__.get("replay")
                original_replay = capture.replay
            except Exception as error:
                observation["error"] = f"{type(error).__name__}: {error}"
                return original_callback(rows, inputs)

            def observed_replay() -> None:
                observation["capture_replay_calls"] += 1
                original_replay()

            capture.replay = observed_replay
            try:
                return original_callback(rows, inputs)
            finally:
                if had_instance_replay:
                    capture.replay = prior_instance_replay
                else:
                    del capture.replay

        return ReplayBatch(
            rows=clone_owned(frozen.rows),
            forward_context=clone_owned(frozen.forward_context),
            run_graph=observed_callback,
        )

    def graph_observation(self, phase: str) -> dict[str, Any]:
        observation = self._graph_observations.get(phase)
        if observation is None:
            raise RetainedReplayFixtureError(f"{phase}: graph observation is absent")
        if observation["error"] is not None:
            raise RetainedReplayFixtureError(
                f"{phase}: graph observer failed: {observation['error']}"
            )
        if observation["callback_calls"] != 1:
            raise RetainedReplayFixtureError(
                f"{phase}: expected one replay callback, got "
                f"{observation['callback_calls']}"
            )
        if observation["capture_replay_calls"] != 1:
            raise RetainedReplayFixtureError(
                f"{phase}: expected one captured replay, got "
                f"{observation['capture_replay_calls']}"
            )
        return dict(observation)

    def _register_production_graph_inputs(
        self,
        registry: TouchedStateRegistry,
        replay_batch: ReplayBatch,
    ) -> None:
        manager, descriptor, _ = self._production_graph_components(replay_batch)
        for index, tensor in enumerate(manager.input_buffers):
            if descriptor.num_tokens > tensor.shape[0]:
                raise RetainedReplayFixtureError(
                    "production replay descriptor exceeds graph input capacity"
                )
            registry.add_rows(
                f"replay_graph:input:{index}",
                tensor,
                torch.arange(descriptor.num_tokens, device=tensor.device),
                meaningful_count=int(replay_batch.rows.numel()),
            )

    def _register_target_scratch_policies(
        self,
        registry: TouchedStateRegistry,
        input_batch: InputBatch,
        replay_batch: ReplayBatch,
    ) -> None:
        rows = replay_batch.rows.detach().to(
            device=input_batch.positions.device,
            dtype=torch.int64,
        )
        count = int(rows.numel())
        if count <= 0:
            raise RetainedReplayFixtureError(
                "production replay has no scratch ownership rows"
            )
        positions = input_batch.positions.index_select(0, rows).detach().cpu()
        query_ends = input_batch.query_start_loc[1 : input_batch.num_reqs + 1].to(
            torch.int64
        )
        request_ordinals = torch.searchsorted(
            query_ends,
            rows,
            right=True,
        )
        if bool((request_ordinals >= input_batch.num_reqs).any()):
            raise RetainedReplayFixtureError(
                "production replay scratch rows exceed request bounds"
            )
        request_slots = (
            input_batch.idx_mapping.index_select(0, request_ordinals).detach().cpu()
        )
        row_identity = {
            "row_writer": "target-replay",
            "row_positions": positions.tolist(),
            "row_request_ordinals": request_ordinals.detach().cpu().tolist(),
            "row_request_state_slots": request_slots.tolist(),
            "sentinel": -1,
            "require_unique_valid_ids": True,
            "require_trailing_sentinels": True,
        }

        model = self._target_decoder_model()
        config = getattr(model, "config", self.runner.model_config.hf_config)
        layers = getattr(model, "layers", None)
        if layers is None:
            raise RetainedReplayFixtureError(
                "target scratch producer layers are absent"
            )

        candidate_name = "model:candidate_block_buffer"
        if candidate_name in registry._names:
            source_layer = int(getattr(config, "candidate_source_layer_id", -1))
            block_size = int(getattr(config, "candidate_block_size", 0))
            tensor = getattr(model, "candidate_block_buffer", None)
            if (
                source_layer < 0
                or block_size <= 0
                or not isinstance(tensor, torch.Tensor)
            ):
                raise RetainedReplayFixtureError(
                    "candidate scratch producer configuration is invalid"
                )
            producer = getattr(layers[source_layer], "attn", None)
            ratio = int(getattr(producer, "compress_ratio", 0))
            if ratio <= 0:
                raise RetainedReplayFixtureError(
                    "candidate scratch producer ratio is invalid"
                )
            states = torch.div(
                positions + 1,
                ratio,
                rounding_mode="floor",
            )
            bounds = torch.div(
                states + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            width = int(tensor.shape[1])
            registry.set_tensor_policy(
                candidate_name,
                comparison="canonical_membership",
                compare_prefix=count,
                metadata={
                    **row_identity,
                    "scratch_policy": "candidate_membership_v1",
                    "producer_layer": source_layer,
                    "cache_owner": getattr(producer, "compressed_cache_prefix", None),
                    "compress_ratio": ratio,
                    "candidate_block_size": block_size,
                    "valid_id_exclusive_bounds": bounds.tolist(),
                    "expected_valid_counts": torch.minimum(
                        bounds,
                        torch.full_like(bounds, width),
                    ).tolist(),
                },
            )

        topk_name = "model:topk_indices_buffer"
        if topk_name in registry._names:
            source_layers = list(getattr(config, "index_source_layer_ids", ()))
            tensor = getattr(model, "topk_indices_buffer", None)
            if not source_layers or not isinstance(tensor, torch.Tensor):
                raise RetainedReplayFixtureError(
                    "top-k scratch producer configuration is invalid"
                )
            source_layer = int(source_layers[-1])
            producer = getattr(layers[source_layer], "attn", None)
            ratio = int(getattr(producer, "compress_ratio", 0))
            if ratio <= 0:
                raise RetainedReplayFixtureError(
                    "top-k scratch producer ratio is invalid"
                )
            indexer_cache_owner = getattr(
                getattr(
                    getattr(producer, "indexer", None),
                    "k_cache",
                    None,
                ),
                "prefix",
                None,
            )
            if not isinstance(indexer_cache_owner, str) or not indexer_cache_owner:
                raise RetainedReplayFixtureError("top-k indexer cache owner is absent")
            bounds = torch.div(
                positions + 1,
                ratio,
                rounding_mode="floor",
            )
            width = int(tensor.shape[1])
            registry.set_tensor_policy(
                topk_name,
                comparison="diagnostic_topk",
                compare_prefix=count,
                metadata={
                    **row_identity,
                    "scratch_policy": "topk_diagnostic_v1",
                    "producer_layer": source_layer,
                    "cache_owner": indexer_cache_owner,
                    "candidate_membership_path": (
                        "registry/model:candidate_block_buffer"
                    ),
                    "candidate_block_size": int(
                        getattr(config, "candidate_block_size", 0)
                    ),
                    "compress_ratio": ratio,
                    "valid_id_exclusive_bounds": bounds.tolist(),
                    "expected_valid_counts": torch.minimum(
                        bounds,
                        torch.full_like(bounds, width),
                    ).tolist(),
                },
            )

    def begin(
        self,
        *,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        full_forward_context: ForwardContext,
        replay_batch: RetainedReplayBatch,
        ordinary_batch: OrdinaryDecodeBatch | None,
        production_replay_batch: ReplayBatch,
        dp_sync: DPSyncState | None,
        is_dummy: bool,
    ) -> None:
        if self.state != "armed":
            raise RetainedReplayFixtureError(
                f"cannot begin fixture capture from state {self.state}"
            )
        torch.cuda.synchronize(self.runner.device)
        writer_contexts = [
            (
                full_forward_context,
                input_batch.num_tokens_after_padding,
            ),
            (
                replay_batch.forward_context,
                replay_batch.input_batch.num_tokens_after_padding,
            ),
        ]
        replay_layers = getattr(self.runner.model_state, "decoder_replay_layers", None)
        live_replay = getattr(replay_layers, "replay_batch", None)
        if live_replay is not None:
            writer_contexts.append(
                (
                    live_replay.forward_context,
                    int(live_replay.rows.numel()),
                )
            )
        if ordinary_batch is not None:
            writer_contexts.append(
                (
                    ordinary_batch.forward_context,
                    ordinary_batch.input_batch.num_tokens_after_padding,
                )
            )
        decoder_model = self._target_decoder_model()
        replay_start = int(decoder_model.decoder_replay_start)
        replay_layers = decoder_model.decoder_replay_layers
        if replay_layers is None:
            raise RetainedReplayFixtureError(
                "target decoder replay ownership is absent"
            )
        full_layers = set(decoder_model.decoder_replay_front_metadata_prefixes)
        cut_attention = decoder_model.layers[replay_start].attn
        full_layers.add(cut_attention.swa_cache_layer.prefix)
        for owner in (
            getattr(cut_attention, "prefix", None),
            getattr(cut_attention, "compressed_cache_prefix", None),
            getattr(
                getattr(cut_attention, "indexer", None),
                "k_cache",
                None,
            ),
            getattr(
                getattr(cut_attention, "compressor", None),
                "state_cache",
                None,
            ),
        ):
            prefix = getattr(owner, "prefix", owner)
            if isinstance(prefix, str) and prefix:
                full_layers.add(prefix)
        logical_writers = [
            self._cache_writer(
                "target-full",
                input_batch,
                full_forward_context,
                allowed_layers=full_layers,
            ),
            self._cache_writer(
                "target-replay",
                input_batch,
                production_replay_batch.forward_context,
                rows=production_replay_batch.rows,
                allowed_layers=set(replay_layers.metadata_prefixes),
            ),
        ]
        registry = self._build_registry(
            input_batch,
            block_tables,
            tuple(writer_contexts),
            logical_writers,
        )
        self._register_production_graph_inputs(
            registry,
            production_replay_batch,
        )
        self._register_target_scratch_policies(
            registry,
            input_batch,
            production_replay_batch,
        )
        source_digest = registry.source_capture_digest()
        self.fixture_id += 1
        self.fixture = RetainedFixture(
            fixture_id=self.fixture_id,
            input_batch=clone_owned(input_batch),
            model_inputs=clone_owned(model_inputs),
            block_tables=clone_owned(block_tables),
            slot_mappings=slot_mappings.clone(),
            full_forward_context=clone_owned(full_forward_context),
            replay_batch=clone_owned(replay_batch),
            ordinary_batch=clone_owned(ordinary_batch),
            production_replay_batch=production_replay_batch,
            dp_sync=clone_owned(dp_sync),
            is_dummy=is_dummy,
            registry=registry,
            source_digest=source_digest,
            production_checkpoints={},
        )
        self._active_checkpoint_error = None
        self.state = "capturing"

    def reset_active_checkpoints(self) -> None:
        self._active_checkpoints = {}
        self._active_checkpoint_error = None
        self.boundary_error = None

    def active_checkpoints_ready(self) -> bool:
        return self._active_checkpoint_error is None

    def _defer_checkpoint_error(self, error: BaseException | str) -> None:
        if self._active_checkpoint_error is None:
            self._active_checkpoint_error = (
                error if isinstance(error, str) else f"{type(error).__name__}: {error}"
            )

    def capture_boundary(self, boundary: DecoderBoundaryState) -> None:
        fixture = self.fixture
        if fixture is None:
            return
        if self.state == "ready" and self._active_checkpoints is not None:
            if "l20" in self._active_checkpoints:
                self._defer_checkpoint_error(
                    "the A-control L20 boundary was captured twice"
                )
                return
            try:
                self._active_checkpoints["l20"] = typed_tree_snapshot(
                    "l20",
                    boundary,
                    path="boundary",
                )
            except Exception as error:
                self._defer_checkpoint_error(error)
            return
        if self.state != "capturing":
            return
        if fixture.boundary is not None:
            self.boundary_error = "the L20 boundary was captured twice"
            return
        try:
            fixture.boundary = clone_owned(boundary)
            assert fixture.production_checkpoints is not None
            fixture.production_checkpoints["l20"] = typed_tree_snapshot(
                "l20",
                boundary,
                path="boundary",
            )
        except Exception as error:
            self.boundary_error = f"{type(error).__name__}: {error}"

    def capture_target_output(
        self,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        fixture = self.fixture
        if fixture is None:
            return
        checkpoints = (
            fixture.production_checkpoints
            if self.state == "capturing"
            else self._active_checkpoints
        )
        if checkpoints is None:
            return
        try:
            mtp_hidden = None
            get_mtp = getattr(self.runner.model, "get_mtp_target_hidden_states", None)
            if callable(get_mtp):
                mtp_hidden = get_mtp()
                if mtp_hidden is not None and hidden_states is not None:
                    mtp_hidden = mtp_hidden[: hidden_states.shape[0]]
            checkpoints["target"] = typed_tree_snapshot(
                "target",
                {
                    "hidden_states": hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "mtp_pre_collapse": mtp_hidden,
                },
                path="target",
            )
        except Exception as error:
            self._defer_checkpoint_error(error)

    def capture_post_sample(
        self,
        sampled_token_ids: torch.Tensor | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        *,
        skipped: bool = False,
    ) -> None:
        fixture = self.fixture
        if fixture is None:
            return
        checkpoints = (
            fixture.production_checkpoints
            if self.state == "capturing"
            else self._active_checkpoints
        )
        if checkpoints is None:
            return
        try:
            decisions = {
                "skipped": skipped,
                "sampled_token_ids": (
                    None
                    if sampled_token_ids is None
                    else canonical_decision_reference(
                        sampled_token_ids,
                        num_sampled,
                        num_rejected,
                    )["sampled_token_ids"]
                ),
                "num_sampled": num_sampled,
                "num_rejected": num_rejected,
            }
            checkpoints["post_sampling"] = merge_typed_snapshots(
                "post_sampling",
                fixture.registry.typed_snapshot("post_sampling"),
                fixture.registry.logical_cache_snapshot(
                    "post_sampling",
                    num_rejected,
                ),
                typed_tree_snapshot(
                    "post_sampling",
                    decisions,
                    path="decisions",
                ),
            )
        except Exception as error:
            self._defer_checkpoint_error(error)

    def finalize(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        fixture = self.fixture
        if self.state != "capturing" or fixture is None:
            return
        if self.boundary_error is not None:
            raise RetainedReplayFixtureError(
                f"the L20 boundary capture failed: {self.boundary_error}"
            )
        if self._active_checkpoint_error is not None:
            raise RetainedReplayFixtureError(
                "the production checkpoint capture failed: "
                f"{self._active_checkpoint_error}"
            )
        if fixture.boundary is None:
            raise RetainedReplayFixtureError("the real L20 boundary was not captured")
        checkpoints = fixture.production_checkpoints
        if checkpoints is None or "target" not in checkpoints:
            raise RetainedReplayFixtureError("the production target was not captured")
        if "post_sampling" not in checkpoints:
            raise RetainedReplayFixtureError(
                "the production post-sampling state was not captured"
            )
        torch.cuda.synchronize(self.runner.device)
        fixture.registry.capture_epoch("s1")
        source_digest = fixture.registry.source_digest()
        if source_digest != fixture.source_digest:
            raise RetainedReplayFixtureError(
                "source KV changed during the real capture step"
            )
        fixture.reference = canonical_decision_reference(
            sampled_token_ids,
            num_sampled,
            num_rejected,
        )
        draft_writers = self._draft_cache_writers(fixture)
        checkpoints["s1"] = merge_typed_snapshots(
            "s1",
            fixture.registry.typed_snapshot("s1", epoch="s1"),
            fixture.registry.logical_cache_snapshot(
                "s1",
                num_rejected,
                extra_writers=draft_writers,
            ),
            typed_tree_snapshot(
                "s1",
                {
                    "skipped": fixture.is_dummy,
                    **fixture.reference,
                },
                path="decisions",
            ),
        )
        fixture.production_graph = self.graph_observation("production")
        fixture.fixture_digest = _tree_digest(
            {
                "input_batch": fixture.input_batch,
                "model_inputs": fixture.model_inputs,
                "block_tables": fixture.block_tables,
                "slot_mappings": fixture.slot_mappings,
                "full_forward_context": fixture.full_forward_context,
                "replay_batch": fixture.replay_batch,
                "ordinary_batch": fixture.ordinary_batch,
                "production_replay_batch": fixture.production_replay_batch,
                "dp_sync": fixture.dp_sync,
                "is_dummy": fixture.is_dummy,
                "production_graph": fixture.production_graph,
                "boundary": fixture.boundary,
                "reference": fixture.reference,
                "s0": fixture.registry.digest("s0"),
                "s1": fixture.registry.digest("s1"),
                "source": fixture.source_digest,
            }
        )
        self.state = "ready"

    def finalize_a_checkpoints(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> dict[str, Any]:
        fixture = self.fixture
        checkpoints = self._active_checkpoints
        if self.state != "ready" or fixture is None or checkpoints is None:
            raise RetainedReplayFixtureError("no A-control execution is active")
        if self._active_checkpoint_error is not None:
            raise RetainedReplayFixtureError(
                f"A-control checkpoint capture failed: {self._active_checkpoint_error}"
            )
        missing = {"l20", "target", "post_sampling"} - checkpoints.keys()
        if missing:
            raise RetainedReplayFixtureError(
                f"A-control checkpoints are incomplete: {sorted(missing)}"
            )
        reference = canonical_decision_reference(
            sampled_token_ids,
            num_sampled,
            num_rejected,
        )
        draft_writers = self._draft_cache_writers(fixture)
        checkpoints["s1"] = merge_typed_snapshots(
            "s1",
            fixture.registry.typed_snapshot("s1"),
            fixture.registry.logical_cache_snapshot(
                "s1",
                num_rejected,
                extra_writers=draft_writers,
            ),
            typed_tree_snapshot(
                "s1",
                {
                    "skipped": fixture.is_dummy,
                    **reference,
                },
                path="decisions",
            ),
        )
        return reference

    def _evidence_contract(self) -> dict[str, Any]:
        fixture = self.fixture
        assert fixture is not None
        checkpoints = fixture.production_checkpoints
        assert checkpoints is not None
        return {
            "checkpoint_paths": {
                checkpoint: sorted(leaf["path"] for leaf in snapshot["leaves"])
                for checkpoint, snapshot in checkpoints.items()
            },
            "comparison_classes": {
                f"{checkpoint}:{leaf['path']}": leaf["comparison"]
                for checkpoint, snapshot in checkpoints.items()
                for leaf in snapshot["leaves"]
            },
            "registry_entries": sorted(
                entry.name for entry in fixture.registry.entries
            ),
            "meaningful_counts": dict(
                sorted(fixture.registry._meaningful_counts.items())
            ),
            "graph_input_count": sum(
                entry.name.startswith("replay_graph:input:")
                for entry in fixture.registry.entries
            ),
            "logical_cache_views": [
                view.metadata() for view in fixture.registry._cache_views
            ],
            "cache_owner_inventory": [
                view.view_id for view in fixture.registry._cache_views
            ],
        }

    def _run_manifest(
        self,
        *,
        run_id: str,
        run_kind: str,
        ordinal: int,
        graph: dict[str, Any],
        state_digest: str,
    ) -> dict[str, Any]:
        fixture = self.fixture
        assert fixture is not None
        model_name = getattr(self.runner.model_config, "model", None)
        draft_config = getattr(
            self.runner.speculative_config,
            "draft_model_config",
            None,
        )
        draft_name = (
            None if draft_config is None else getattr(draft_config, "model", None)
        )
        runner_module = sys.modules[type(self.runner).__module__]
        runner_path = Path(runner_module.__file__)
        shared_config = {
            "model_config": _model_config_identity(self.runner.model_config),
            "cache_config": repr(self.runner.cache_config),
            "scheduler_config": repr(self.runner.scheduler_config),
            "speculative_config": _speculative_config_identity(
                self.runner.speculative_config
            ),
            "kernel_config": repr(self.runner.vllm_config.kernel_config),
            "compilation": {
                "mode": str(self.runner.compilation_config.mode),
                "cudagraph_mode": str(self.runner.compilation_config.cudagraph_mode),
                "decoder_replay_capture_sizes": list(
                    self.runner.compilation_config.decoder_replay_cudagraph_capture_sizes
                ),
            },
            "parallel": {
                "data_parallel_size": self.runner.dp_size,
                "tensor_parallel_size": (
                    self.runner.parallel_config.tensor_parallel_size
                ),
                "pipeline_parallel_size": (
                    self.runner.parallel_config.pipeline_parallel_size
                ),
                "prefill_context_parallel_size": getattr(
                    self.runner.parallel_config,
                    "prefill_context_parallel_size",
                    1,
                ),
                "decode_context_parallel_size": getattr(
                    self.runner.parallel_config,
                    "decode_context_parallel_size",
                    1,
                ),
            },
            "features": {
                "mixed_replay_graph": os.getenv(
                    "VLLM_DSV41_MIXED_REPLAY_GRAPH",
                    "0",
                ),
                "front_graph": os.getenv(
                    "VLLM_DSV41_MIXED_FRONT_GRAPH",
                    "0",
                ),
            },
        }
        return {
            "comparator_schema": 2,
            "run_id": run_id,
            "worker_incarnation": self.worker_incarnation,
            "host": socket.gethostname(),
            "device": str(self.runner.device),
            "dp_size": self.runner.dp_size,
            "tp_size": self.runner.parallel_config.tensor_parallel_size,
            "pp_size": self.runner.parallel_config.pipeline_parallel_size,
            "model": model_name,
            "draft_model": draft_name,
            "shared_config": shared_config,
            "config_digest": hashlib.sha256(
                json.dumps(shared_config, sort_keys=True).encode()
            ).hexdigest(),
            "rank_config": {
                "data_parallel_rank": self.runner.dp_rank,
                "device": str(self.runner.device),
            },
            "evidence_contract": self._evidence_contract(),
            "code_hashes": {
                "fixture": _sha256_path(Path(__file__)),
                "model_runner": _sha256_path(runner_path),
                "overlay": os.getenv(
                    "VLLM_DSV41_RETAINED_FIXTURE_CODE_HASH",
                    "unknown",
                ),
            },
            "fixture_id": fixture.fixture_id,
            "fixture_digest": fixture.fixture_digest,
            "input_digest": _tree_digest(
                {
                    "input_batch": fixture.input_batch,
                    "model_inputs": fixture.model_inputs,
                    "production_replay_batch": fixture.production_replay_batch,
                }
            ),
            "active_role": "idle" if fixture.is_dummy else "active",
            "s0_digest": fixture.registry.digest("s0"),
            "source_digest": fixture.source_digest,
            "state_digest": state_digest,
            "graph": graph,
            "assignment": (
                "production"
                if run_kind == "production"
                else ("calibration" if ordinal <= 10 else "held_out")
            ),
        }

    def run_a_control(
        self,
        *,
        runs: int,
        output_dir: str,
        run_id: str,
    ) -> dict[str, Any]:
        fixture = self.fixture
        assert fixture is not None
        registry = fixture.registry

        torch.cuda.synchronize(self.runner.device)
        live_digest = None
        staged_block_tables = None
        written: list[dict[str, Any]] = []
        try:
            local_error = None
            try:
                registry.capture_epoch("live", cpu=True)
                live_digest = registry.digest("live")
                production_bundle = write_typed_run_bundle(
                    output_dir,
                    dp_rank=self.runner.dp_rank,
                    run_kind="production",
                    ordinal=0,
                    checkpoints=fixture.production_checkpoints,
                    manifest=self._run_manifest(
                        run_id=run_id,
                        run_kind="production",
                        ordinal=0,
                        graph=fixture.production_graph,
                        state_digest=fixture.registry.digest("s1"),
                    ),
                )
                written.append(
                    {"kind": "production", "ordinal": 0, **production_bundle}
                )
                self.a_control_runs = written
                staged_block_tables = self.runner._prepare_dsv41_retained_block_tables(
                    fixture
                )
            except Exception as error:
                local_error = error
            if not self.runner._dsv41_fixture_vote(local_error is None):
                raise RetainedReplayFixtureError(
                    str(local_error or "a peer failed A-control setup")
                )

            for ordinal in range(1, runs + 1):
                local_error = None
                prepared_cycle = None
                try:
                    registry.restore("s0")
                    registry.verify("s0")
                    self.runner._install_dsv41_retained_block_tables(fixture)
                    if registry.source_digest() != fixture.source_digest:
                        raise RetainedReplayFixtureError(
                            f"A{ordinal}: source KV changed before execution"
                        )
                    self.reset_active_checkpoints()
                    prepared_cycle = self.runner._prepare_dsv41_a_cycle(
                        fixture,
                        self,
                        phase=f"a-{ordinal:03d}",
                    )
                except Exception as error:
                    local_error = error
                if not self.runner._dsv41_fixture_vote(local_error is None):
                    raise RetainedReplayFixtureError(
                        str(
                            local_error
                            or f"A{ordinal}: a peer failed restore preflight"
                        )
                    )
                assert prepared_cycle is not None

                local_error = None
                completed = None
                try:
                    phase = f"a-{ordinal:03d}"
                    result = self.runner._run_dsv41_a_cycle(
                        fixture,
                        self,
                        phase=phase,
                        prepared=prepared_cycle,
                    )
                    torch.cuda.synchronize(self.runner.device)
                    graph = self.graph_observation(phase)
                    state_digest = registry.digest()
                    source_digest = registry.source_digest()
                    if source_digest != fixture.source_digest:
                        raise RetainedReplayFixtureError(
                            f"A{ordinal}: source KV changed during execution"
                        )
                    checkpoints = self._active_checkpoints
                    if checkpoints is None:
                        raise RetainedReplayFixtureError(
                            f"A{ordinal}: typed checkpoints disappeared"
                        )
                    bundle = write_typed_run_bundle(
                        output_dir,
                        dp_rank=self.runner.dp_rank,
                        run_kind="a",
                        ordinal=ordinal,
                        checkpoints=checkpoints,
                        manifest=self._run_manifest(
                            run_id=run_id,
                            run_kind="a",
                            ordinal=ordinal,
                            graph=graph,
                            state_digest=state_digest,
                        )
                        | {
                            "decision_digest": _tree_digest(result["reference"]),
                        },
                    )
                    completed = {
                        "kind": "a",
                        "ordinal": ordinal,
                        **bundle,
                    }
                except Exception as error:
                    local_error = error
                if not self.runner._dsv41_fixture_vote(local_error is None):
                    raise RetainedReplayFixtureError(
                        str(local_error or f"A{ordinal}: a peer failed execution")
                    )
                assert completed is not None
                written.append(completed)
                self.a_control_runs = written
                self._active_checkpoints = None
        finally:
            self._active_checkpoints = None
            torch.cuda.synchronize(self.runner.device)
            self.runner._restore_dsv41_retained_block_tables(staged_block_tables)
            if live_digest is not None:
                registry.restore("live")
                registry.verify("live")
        if live_digest is None:
            raise RetainedReplayFixtureError("LIVE state was not captured")
        if registry.digest() != live_digest:
            raise RetainedReplayFixtureError("LIVE state was not restored")
        if registry.source_digest() != fixture.source_digest:
            raise RetainedReplayFixtureError("source KV changed during A-control")
        summary = write_typed_rank_summary(
            output_dir,
            dp_rank=self.runner.dp_rank,
            document={
                "run_id": run_id,
                "worker_incarnation": self.worker_incarnation,
                "fixture_id": fixture.fixture_id,
                "fixture_digest": fixture.fixture_digest,
                "completed_runs": runs,
                "source_digest": fixture.source_digest,
                "live_digest": live_digest,
                "live_restored": True,
                "files": written,
            },
        )
        return {
            "fixture_id": fixture.fixture_id,
            "fixture_digest": fixture.fixture_digest,
            "run_id": run_id,
            "output_dir": output_dir,
            "completed_runs": runs,
            "production_manifest": written[0]["manifest_path"],
            "last_manifest": written[-1]["manifest_path"],
            **summary,
            "live_digest": live_digest,
            "live_restored": True,
        }

    def validate_a_control_preflight(
        self,
        *,
        runs: int,
        output_dir: str,
        run_id: str,
    ) -> None:
        fixture = self.fixture
        if self.state != "ready" or fixture is None:
            raise RetainedReplayFixtureError(
                f"fixture is not ready: state={self.state}"
            )
        if runs < 1:
            raise ValueError("A-control run count must be positive")
        if not run_id:
            raise ValueError("run_id is required")
        if fixture.production_checkpoints is None or set(
            fixture.production_checkpoints
        ) != {"l20", "target", "post_sampling", "s1"}:
            raise RetainedReplayFixtureError(
                "production typed checkpoints are incomplete"
            )
        if fixture.production_graph is None:
            raise RetainedReplayFixtureError("production graph observation is absent")
        if fixture.registry.source_digest() != fixture.source_digest:
            raise RetainedReplayFixtureError(
                "source KV lifetime expired before A-control"
            )
        self._production_graph_components(fixture.production_replay_batch)
        self.runner._validate_dsv41_a_cycle(fixture)
        rank_dir = Path(output_dir) / f"rank-{self.runner.dp_rank:02d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            path
            for path in rank_dir.iterdir()
            if not path.name.startswith(".preflight-")
        ]
        if existing:
            raise RetainedReplayFixtureError(
                f"A-control destination is not empty: {rank_dir}"
            )
        with tempfile.NamedTemporaryFile(dir=rank_dir, prefix=".preflight-") as probe:
            probe.write(b"ok")
            probe.flush()

    def verify_restore_cycles(self, cycles: int = 2) -> dict[str, Any]:
        fixture = self.fixture
        if self.state != "ready" or fixture is None:
            raise RetainedReplayFixtureError(
                f"fixture is not ready: state={self.state}"
            )
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        torch.cuda.synchronize(self.runner.device)
        registry = fixture.registry
        before_source = registry.source_digest()
        if before_source != fixture.source_digest:
            raise RetainedReplayFixtureError(
                "source KV lifetime expired before restore validation"
            )
        registry.capture_epoch("live", cpu=True)
        live_digest = registry.digest("live")
        results = []
        try:
            for cycle in range(1, cycles + 1):
                registry.restore("s0")
                registry.verify("s0")
                s0_digest = registry.digest()
                source_before_s1 = registry.source_digest()
                if source_before_s1 != fixture.source_digest:
                    raise RetainedReplayFixtureError(
                        f"source KV changed in restore cycle {cycle} at S0"
                    )
                registry.restore("s1")
                registry.verify("s1")
                s1_digest = registry.digest()
                source_after_s1 = registry.source_digest()
                if source_after_s1 != fixture.source_digest:
                    raise RetainedReplayFixtureError(
                        f"source KV changed in restore cycle {cycle} at S1"
                    )
                results.append(
                    {
                        "cycle": cycle,
                        "s0_digest": s0_digest,
                        "s1_digest": s1_digest,
                        "source_digest": source_after_s1,
                    }
                )
            if len({row["s0_digest"] for row in results}) != 1:
                raise RetainedReplayFixtureError("A1/A2 S0 digests differ")
            if len({row["s1_digest"] for row in results}) != 1:
                raise RetainedReplayFixtureError("A1/A2 S1 digests differ")
            if len({row["source_digest"] for row in results}) != 1:
                raise RetainedReplayFixtureError("A1/A2 source digests differ")
        finally:
            registry.restore("live")
            registry.verify("live")
        if registry.digest() != live_digest:
            raise RetainedReplayFixtureError("LIVE state was not restored")
        self.verify_runs = results
        return {
            "fixture_id": fixture.fixture_id,
            "fixture_digest": fixture.fixture_digest,
            "live_digest": live_digest,
            "cycles": results,
            "live_restored": True,
        }

    def validate_restore(self, epoch: str) -> None:
        fixture = self.fixture
        if fixture is None:
            raise RetainedReplayFixtureError("no fixture has been captured")
        if epoch not in ("s0", "s1", "live"):
            raise ValueError(f"invalid restore epoch: {epoch}")
        source_digest = fixture.registry.source_digest()
        if source_digest != fixture.source_digest:
            raise RetainedReplayFixtureError(
                "source KV lifetime expired before restore"
            )
        if any(
            getattr(entry, epoch) is None
            for entry in fixture.registry.entries
            if not isinstance(entry, _ObjectAttribute)
        ):
            raise RetainedReplayFixtureError(f"restore epoch {epoch} is absent")

    def restore(self, epoch: str) -> dict[str, Any]:
        self.validate_restore(epoch)
        fixture = self.fixture
        assert fixture is not None
        source_digest = fixture.source_digest
        fixture.registry.restore(epoch)
        fixture.registry.verify(epoch)
        return {
            "fixture_id": fixture.fixture_id,
            "epoch": epoch,
            "digest": fixture.registry.digest(),
            "source_digest": source_digest,
        }

    @staticmethod
    def _effective_layer_spec(group: Any, layer_name: str) -> Any:
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", None)
        if isinstance(per_layer, dict):
            try:
                return per_layer[layer_name]
            except KeyError as error:
                raise RetainedReplayFixtureError(
                    f"{layer_name}: per-layer KV spec is absent"
                ) from error
        return spec

    @staticmethod
    def _spec_metadata(spec: Any) -> dict[str, Any]:
        names = (
            "block_size",
            "head_size",
            "head_size_v",
            "num_kv_heads",
            "dtype",
            "tokens_per_state",
            "state_content_bytes",
            "alignment",
            "block_stride_alignment",
            "cache_dtype_str",
            "kv_quant_mode",
            "sliding_window",
            "bounded_replay",
        )
        metadata = {"type": type(spec).__qualname__}
        for name in names:
            if hasattr(spec, name):
                metadata[name] = _manifest_value(getattr(spec, name))
        for name in ("state_content_size_bytes", "page_size_bytes", "num_states"):
            try:
                metadata[name] = _manifest_value(getattr(spec, name))
            except (AssertionError, NotImplementedError):
                metadata[name] = None
        return metadata

    @staticmethod
    def _tokens_per_state(spec: Any) -> int:
        value = getattr(spec, "tokens_per_state", None)
        numerator = getattr(value, "numerator", value)
        denominator = getattr(value, "denominator", 1)
        if (
            not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or denominator != 1
            or numerator <= 0
        ):
            raise RetainedReplayFixtureError(
                f"unsupported logical tokens_per_state={value!r}"
            )
        return numerator

    def _cache_consumers_and_scale(
        self,
        layer_name: str,
        owner: Any,
        cache: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor | None]:
        static_context = self.runner.compilation_config.static_forward_context
        consumers = {layer_name}
        scale = getattr(owner, "_flashinfer_fp8_kv_scale", None)
        for name, candidate in static_context.items():
            references = (
                getattr(candidate, "swa_cache_layer", None) is owner
                or getattr(candidate, "k_cache", None) is owner
                or getattr(candidate, "state_cache", None) is owner
                or getattr(candidate, "compressed_cache_prefix", None) == layer_name
                or getattr(getattr(candidate, "compressor", None), "state_cache", None)
                is owner
            )
            if not references:
                continue
            consumers.add(name)
            candidate_scale = getattr(candidate, "_flashinfer_fp8_kv_scale", None)
            if candidate_scale is not None:
                if scale is not None and not torch.equal(
                    scale.detach().cpu(),
                    candidate_scale.detach().cpu(),
                ):
                    raise RetainedReplayFixtureError(
                        f"{layer_name}: plain FP8 consumers disagree on scale"
                    )
                scale = candidate_scale
        if cache.dtype == torch.float8_e4m3fn and scale is None:
            raise RetainedReplayFixtureError(
                f"{layer_name}: plain FP8 cache scale owner is absent"
            )
        return sorted(consumers), scale

    @staticmethod
    def _cache_kind(owner: Any, layer_name: str) -> str:
        owner_type = type(owner).__name__
        if owner_type == "CompressorStateCache":
            return "compressor_ring"
        if owner_type == "DeepseekV4IndexerCache" or layer_name.endswith(
            ".indexer.k_cache"
        ):
            return "indexer"
        if owner_type == "DeepseekV4SWACache":
            return "swa"
        return "compressed_main"

    @staticmethod
    def _view_page_span(view: _LogicalCacheView) -> tuple[int, int, int]:
        element_size = view.tensor.element_size()
        stride = view.tensor.stride(0) * element_size
        begin = view.tensor.storage_offset() * element_size % stride
        content = view.tensor.shape[1] * view.tensor.shape[2] * element_size
        return stride, begin, begin + content

    def _bind_cache_storage_views(
        self,
        records: list[_CacheIntervals],
    ) -> None:
        records = [record for record in records if record.views]
        ordered = sorted(
            records,
            key=lambda record: min(view.layer_name for view in record.views),
        )
        for storage_index, record in enumerate(ordered):
            storage_id = f"storage-{storage_index:03d}"
            for view in record.views:
                view.storage_id = storage_id
            for left_index, left in enumerate(record.views):
                left_stride, left_begin, left_end = self._view_page_span(left)
                for right in record.views[left_index + 1 :]:
                    right_stride, right_begin, right_end = self._view_page_span(right)
                    exact_alias = (
                        left.tensor.storage_offset() == right.tensor.storage_offset()
                        and left.tensor.shape == right.tensor.shape
                        and left.tensor.stride() == right.tensor.stride()
                        and left.record_format == right.record_format
                    )
                    if exact_alias:
                        left.aliases.append(right.view_id)
                        right.aliases.append(left.view_id)
                        continue
                    if left_stride != right_stride:
                        raise RetainedReplayFixtureError(
                            f"{left.view_id}/{right.view_id}: shared storage has "
                            "incompatible block strides"
                        )
                    if max(left_begin, right_begin) < min(left_end, right_end):
                        raise RetainedReplayFixtureError(
                            f"{left.view_id}/{right.view_id}: unexplained cache-view "
                            "overlap"
                        )
            for view in record.views:
                view.aliases.sort()

    @staticmethod
    def _cache_writer(
        phase: str,
        input_batch: InputBatch,
        context: ForwardContext,
        *,
        rows: torch.Tensor | None = None,
        allowed_layers: set[str] | None = None,
    ) -> _CacheWriter:
        num_tokens = int(input_batch.num_tokens)
        if rows is None:
            rows = torch.arange(
                num_tokens,
                dtype=torch.int64,
                device=input_batch.positions.device,
            )
        else:
            rows = rows[:num_tokens].to(
                device=input_batch.positions.device,
                dtype=torch.int64,
            )
        positions = input_batch.positions.index_select(0, rows).detach().cpu()
        query_ends = input_batch.query_start_loc[1 : input_batch.num_reqs + 1].to(
            torch.int64
        )
        request_ordinals = torch.searchsorted(
            query_ends,
            rows,
            right=True,
        )
        if bool((request_ordinals >= input_batch.num_reqs).any()):
            raise RetainedReplayFixtureError(
                f"{phase}: writer rows exceed request query bounds"
            )
        request_slots = (
            input_batch.idx_mapping.index_select(0, request_ordinals).detach().cpu()
        )
        attn_metadata = context.attn_metadata
        if not isinstance(attn_metadata, dict):
            raise RetainedReplayFixtureError(
                f"{phase}: attention metadata is not a dictionary"
            )
        slots_by_layer = {}
        for layer_name, metadata in attn_metadata.items():
            if allowed_layers is not None and layer_name not in allowed_layers:
                continue
            slots = getattr(metadata, "cache_slot_mapping", None)
            if not isinstance(slots, torch.Tensor):
                slots = getattr(metadata, "slot_mapping", None)
            if not isinstance(slots, torch.Tensor):
                continue
            if slots.numel() < rows.numel():
                raise RetainedReplayFixtureError(
                    f"{phase}: {layer_name} slot mapping is too short"
                )
            slots_by_layer[layer_name] = slots[: rows.numel()].detach().cpu()
        return _CacheWriter(
            phase=phase,
            positions=positions,
            request_ordinals=request_ordinals.detach().cpu(),
            request_state_slots=request_slots,
            slots_by_layer=slots_by_layer,
        )

    def _draft_cache_writers(
        self,
        fixture: RetainedFixture,
    ) -> list[_CacheWriter]:
        speculator = self.runner.speculator
        group_ids = tuple(getattr(speculator, "draft_kv_cache_group_ids", ()))
        if fixture.is_dummy or speculator is None or not group_ids:
            return []
        num_reqs = int(fixture.input_batch.num_reqs)
        num_target_tokens = int(fixture.input_batch.num_tokens)
        query_ends = (
            fixture.input_batch.query_start_loc[1 : num_reqs + 1]
            .detach()
            .cpu()
            .to(torch.int64)
        )
        target_rows = torch.arange(num_target_tokens, dtype=torch.int64)
        context_request_ordinals = torch.searchsorted(
            query_ends,
            target_rows,
            right=True,
        )
        if bool((context_request_ordinals >= num_reqs).any()):
            raise RetainedReplayFixtureError(
                "draft context rows exceed target query bounds"
            )
        fixture_request_slots = (
            fixture.input_batch.idx_mapping[:num_reqs].detach().cpu().to(torch.int64)
        )
        selected_context_rows = speculator._context_rows(fixture.input_batch)
        if selected_context_rows is not None:
            selected_context_rows = selected_context_rows.detach().cpu().to(torch.int64)
            target_rows = target_rows.index_select(0, selected_context_rows)
            context_request_ordinals = context_request_ordinals.index_select(
                0, selected_context_rows
            )
        context_request_slots = fixture_request_slots.index_select(
            0, context_request_ordinals
        )

        groups = self.runner.kv_cache_config.kv_cache_groups
        context_mappings = getattr(speculator, "_context_slot_mappings", ())
        if len(context_mappings) != len(group_ids):
            raise RetainedReplayFixtureError(
                "draft context slot-mapping groups are incomplete"
            )
        context_slots_by_layer: dict[str, torch.Tensor] = {}
        query_slots_by_layer: dict[str, torch.Tensor] = {}
        num_query_per_req = int(speculator.num_query_per_req)
        num_query_tokens = num_reqs * num_query_per_req
        for group_index, group_id in enumerate(group_ids):
            if not 0 <= group_id < len(groups):
                raise RetainedReplayFixtureError(
                    f"draft KV group {group_id} is out of range"
                )
            context_slots = context_mappings[group_index]
            query_slots = speculator.block_tables.slot_mappings[group_id]
            if (
                context_slots.numel() < num_target_tokens
                or query_slots.numel() < num_query_tokens
            ):
                raise RetainedReplayFixtureError(
                    f"draft KV group {group_id} slot mappings are too short"
                )
            context_slots = context_slots[:num_target_tokens].detach().cpu()
            if selected_context_rows is not None:
                context_slots = context_slots.index_select(0, selected_context_rows)
            query_slots = query_slots[:num_query_tokens].detach().cpu()
            for layer_name in groups[group_id].layer_names:
                context_slots_by_layer[layer_name] = context_slots
                query_slots_by_layer[layer_name] = query_slots

        query_request_ordinals = torch.arange(
            num_reqs,
            dtype=torch.int64,
        ).repeat_interleave(num_query_per_req)
        query_request_slots = fixture_request_slots.repeat_interleave(num_query_per_req)
        writers = [
            _CacheWriter(
                phase="draft-context",
                positions=speculator.context_positions[:num_target_tokens]
                .detach()
                .cpu()
                .index_select(0, target_rows),
                request_ordinals=context_request_ordinals,
                request_state_slots=context_request_slots,
                slots_by_layer=context_slots_by_layer,
                enforce_committed_extent=False,
            )
        ]
        if getattr(self.runner, "dspark_prefill_only", False):
            return writers
        writers.append(
            _CacheWriter(
                phase="draft-query",
                positions=speculator.input_buffers.positions[:num_query_tokens]
                .detach()
                .cpu(),
                request_ordinals=query_request_ordinals,
                request_state_slots=query_request_slots,
                slots_by_layer=query_slots_by_layer,
                enforce_committed_extent=False,
            )
        )
        return writers

    def _build_registry(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        writer_contexts: tuple[tuple[ForwardContext, int], ...],
        logical_writers: list[_CacheWriter],
    ) -> TouchedStateRegistry:
        registry = TouchedStateRegistry()
        cache_intervals = self._collect_cache_intervals(
            input_batch,
            block_tables,
            writer_contexts,
            collect_logical=True,
        )
        logical_views = []
        for key, cache in cache_intervals.items():
            del key
            logical_views.extend(cache.views)
            if cache.destination:
                registry.add_intervals(
                    f"kv-dst:{cache.name}", cache.tensor, cache.destination
                )
            if cache.source:
                registry.add_source_intervals(
                    f"kv-src:{cache.name}", cache.tensor, cache.source
                )
        computed_tokens = self.runner.req_states.num_computed_tokens.gpu
        base_extents = computed_tokens.index_select(
            0,
            input_batch.idx_mapping[: input_batch.num_reqs],
        )
        registry.configure_logical_cache(
            logical_views,
            logical_writers,
            input_batch,
            block_tables,
            tuple(int(value) for value in base_extents.detach().cpu().tolist()),
        )
        self._register_runtime_state(registry, input_batch)
        return registry

    def _collect_cache_intervals(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        writer_contexts: tuple[tuple[ForwardContext, int], ...],
        *,
        collect_logical: bool = False,
    ) -> dict[tuple[str, int | None, int, int], _CacheIntervals]:
        runner = self.runner
        groups = runner.kv_cache_config.kv_cache_groups
        static_context = runner.compilation_config.static_forward_context
        metadata_contexts = []
        for context, num_tokens in writer_contexts:
            attn_metadata = context.attn_metadata
            if not isinstance(attn_metadata, dict):
                raise RetainedReplayFixtureError(
                    "retained fixture requires attention metadata dictionaries"
                )
            metadata_contexts.append((attn_metadata, num_tokens))
        req_slots = input_batch.idx_mapping[: input_batch.num_reqs]
        num_blocks = runner.block_tables.num_blocks.gpu
        counts = [
            num_blocks[group_id].index_select(0, req_slots).cpu().tolist()
            for group_id in range(len(groups))
        ]
        tables_cpu = [table[: input_batch.num_reqs].cpu() for table in block_tables]
        intervals_by_storage: dict[
            tuple[str, int | None, int, int], _CacheIntervals
        ] = {}

        speculative_groups = set(
            getattr(self.runner.speculator, "draft_kv_cache_group_ids", ())
        )
        proposal_span = (
            getattr(self.runner.speculator, "num_query_per_req", 0)
            + self.runner.num_speculative_steps
            + 2
        )
        seq_lens_upper = input_batch.seq_lens_cpu_upper_bound[
            : input_batch.num_reqs
        ].tolist()
        seq_lens_exact = input_batch.seq_lens[: input_batch.num_reqs].cpu().tolist()
        replay_layers = getattr(
            getattr(runner, "model_state", None),
            "decoder_replay_layers",
            None,
        )
        mutable_tokens = max(
            int(input_batch.max_query_len or 0),
            int(getattr(replay_layers, "window", 0)),
        )
        continuation_tokens = max(
            1,
            int(input_batch.max_query_len or 0),
            int(runner.num_speculative_steps) + 1,
        )

        for group_id, group in enumerate(groups):
            group_blocks: set[int] = set()
            proposal_blocks: set[int] = set()
            request_block_rows: list[list[int]] = []
            for req_id, count in enumerate(counts[group_id]):
                physical_by_logical_block = [
                    int(block)
                    for block in tables_cpu[group_id][req_id, : int(count)].tolist()
                ]
                request_block_rows.append(physical_by_logical_block)
                physical = [block for block in physical_by_logical_block if block > 0]
                group_blocks.update(physical)
                if group_id in speculative_groups and physical_by_logical_block:
                    block_size = runner.block_tables.kernel_block_sizes[group_id]
                    seq_len = int(seq_lens_upper[req_id])
                    logical_lo = max(0, seq_len - proposal_span) // block_size
                    logical_hi = min(
                        len(physical_by_logical_block) - 1,
                        (seq_len + proposal_span) // block_size,
                    )
                    proposal_blocks.update(
                        block
                        for block in physical_by_logical_block[
                            logical_lo : logical_hi + 1
                        ]
                        if block > 0
                    )

            block_size = runner.block_tables.kernel_block_sizes[group_id]
            frontier_blocks: set[int] = set()
            for req_id, row in enumerate(request_block_rows):
                seq_len_exact = int(seq_lens_exact[req_id])
                seq_len_upper = int(seq_lens_upper[req_id])
                logical_lo = max(0, seq_len_exact - mutable_tokens) // block_size
                logical_hi = min(
                    len(row) - 1,
                    (seq_len_upper + continuation_tokens - 1) // block_size,
                )
                if logical_lo <= logical_hi:
                    frontier_blocks.update(
                        block for block in row[logical_lo : logical_hi + 1] if block > 0
                    )

            for layer_name in group.layer_names:
                owner = static_context.get(layer_name)
                cache = getattr(owner, "kv_cache", None)
                if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
                    if collect_logical:
                        raise RetainedReplayFixtureError(
                            f"{layer_name}: bound KV cache owner is absent"
                        )
                    continue
                if cache.ndim != 3:
                    raise RetainedReplayFixtureError(
                        f"{layer_name}: unsupported bound KV shape {tuple(cache.shape)}"
                    )
                key = _storage_key(cache)
                record = intervals_by_storage.get(key)
                if record is None:
                    record = _CacheIntervals(
                        name=layer_name,
                        tensor=_storage_bytes_view(cache),
                        destination=[],
                        source=[],
                        views=[],
                    )
                    intervals_by_storage[key] = record
                destination_blocks: set[int] = set()
                for attn_metadata, num_tokens in metadata_contexts:
                    metadata = attn_metadata.get(layer_name)
                    if metadata is None:
                        continue
                    effective_slots = getattr(metadata, "cache_slot_mapping", None)
                    if not isinstance(effective_slots, torch.Tensor):
                        effective_slots = getattr(metadata, "slot_mapping", None)
                    if not isinstance(effective_slots, torch.Tensor):
                        raise RetainedReplayFixtureError(
                            f"{layer_name}: effective KV slot mapping is absent"
                        )
                    effective_slots = effective_slots[:num_tokens]
                    destination_blocks.update(
                        int(slot) // cache.shape[1]
                        for slot in effective_slots[effective_slots >= 0].cpu().tolist()
                    )
                mutable_blocks = destination_blocks | frontier_blocks
                logical_destination_blocks = mutable_blocks | proposal_blocks
                record.destination.extend(
                    _cache_block_byte_intervals(layer_name, cache, mutable_blocks)
                )
                record.destination.extend(
                    _cache_block_byte_intervals(layer_name, cache, proposal_blocks)
                )
                record.source.extend(
                    _cache_block_byte_intervals(layer_name, cache, group_blocks)
                )
                if collect_logical and logical_destination_blocks:
                    spec = self._effective_layer_spec(group, layer_name)
                    consumers, fp8_scale = self._cache_consumers_and_scale(
                        layer_name,
                        owner,
                        cache,
                    )
                    cache_kind = self._cache_kind(owner, layer_name)
                    tokens_per_state = self._tokens_per_state(spec)
                    if cache_kind == "compressor_ring":
                        ring_ratios = {
                            int(candidate.compressor.compress_ratio)
                            for candidate in static_context.values()
                            if getattr(
                                getattr(candidate, "compressor", None),
                                "state_cache",
                                None,
                            )
                            is owner
                        }
                        if len(ring_ratios) != 1:
                            raise RetainedReplayFixtureError(
                                f"{layer_name}: compressor ring ratio is ambiguous"
                            )
                        tokens_per_state = next(iter(ring_ratios))
                    record.views.append(
                        _LogicalCacheView(
                            view_id=f"group-{group_id}:{layer_name}",
                            storage_id="",
                            group_id=group_id,
                            layer_name=layer_name,
                            owner_type=type(owner).__qualname__,
                            spec_type=type(spec).__qualname__,
                            spec_metadata=self._spec_metadata(spec),
                            cache_kind=cache_kind,
                            record_format=_cache_record_format(cache),
                            tensor=cache,
                            block_size=int(cache.shape[1]),
                            tokens_per_state=tokens_per_state,
                            live_window=(
                                int(spec.sliding_window)
                                if getattr(spec, "sliding_window", None) is not None
                                else None
                            ),
                            destination_blocks=logical_destination_blocks,
                            consumers=consumers,
                            aliases=[],
                            fp8_scale=fp8_scale,
                        )
                    )
        if not intervals_by_storage:
            raise RetainedReplayFixtureError("no bound KV cache rows were discovered")
        if collect_logical:
            self._bind_cache_storage_views(list(intervals_by_storage.values()))
        for record in intervals_by_storage.values():
            record.destination = _merge_byte_intervals(record.destination)
            record.source = _subtract_byte_intervals(record.source, record.destination)
        return intervals_by_storage

    def _register_runtime_state(
        self, registry: TouchedStateRegistry, input_batch: InputBatch
    ) -> None:
        runner = self.runner
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        req_slots = torch.unique(
            input_batch.idx_mapping[:num_reqs].to(dtype=torch.int64)
        )
        token_rows = torch.arange(num_tokens, device=runner.device)

        model = self._target_decoder_model()
        for name in (
            "topk_indices_buffer",
            "candidate_block_buffer",
            "_mtp_hidden_buffer",
        ):
            tensor = getattr(model, name, None)
            if isinstance(tensor, torch.Tensor):
                registry.add_rows(
                    f"model:{name}",
                    tensor,
                    token_rows[: min(num_tokens, tensor.shape[0])],
                )
        for name in (
            "candidate_compact_logits_buffer",
            "candidate_compact_indices_buffer",
            "candidate_compact_ends_buffer",
        ):
            tensor = getattr(model, name, None)
            if isinstance(tensor, torch.Tensor):
                registry.add_rows(
                    f"model:{name}",
                    tensor,
                    torch.arange(tensor.shape[0], device=tensor.device),
                )

        req_states = runner.req_states
        for name, tensor in (
            ("last_sampled_tokens", req_states.last_sampled_tokens),
            ("draft_tokens", req_states.draft_tokens),
            ("num_computed_tokens", req_states.num_computed_tokens.gpu),
            ("total_len", req_states.total_len.gpu),
        ):
            registry.add_rows(f"request:{name}", tensor, req_slots)
        registry.add_rows(
            "request:next_prefill_tokens",
            req_states.next_prefill_tokens,
            req_slots,
            dim=1,
        )
        for name in (
            "num_computed_prefill_tokens",
            "num_computed_tokens_np",
        ):
            registry.add_attribute(f"request:{name}", req_states, name)

        total_lens = req_states.total_len.gpu.index_select(0, req_slots).cpu()
        max_new = runner.num_speculative_steps + 1
        point_indices = []
        width = req_states.all_token_ids.gpu.shape[1]
        for slot, total_len in zip(req_slots.cpu().tolist(), total_lens.tolist()):
            end = min(width, int(total_len) + max_new)
            point_indices.extend(
                int(slot) * width + pos for pos in range(int(total_len), end)
            )
        registry.add_points(
            "request:all_token_ids",
            req_states.all_token_ids.gpu,
            torch.tensor(point_indices, dtype=torch.int64, device=runner.device),
        )

        sampler = runner.sampler
        if sampler is not None:
            registry.add_rows(
                "sampler:output_bin_counts",
                sampler.penalties_state.output_bin_counts,
                req_slots,
            )
            sampling = sampler.sampling_states
            for name in ("temperature", "top_k", "top_p", "min_p", "seeds"):
                state = getattr(sampling, name)
                registry.add_rows(f"sampler:{name}", state.gpu, req_slots)
            thinking = sampler.thinking_budget_state
            if getattr(thinking, "enabled", False):
                for name in (
                    "cached_last_start",
                    "cached_last_end",
                    "cached_scan_pos",
                ):
                    registry.add_rows(
                        f"sampler:thinking:{name}",
                        getattr(thinking, name),
                        req_slots,
                    )

        speculator = runner.speculator
        if speculator is not None:
            prefix_rows = torch.arange(num_reqs, device=runner.device)
            token_prefix = torch.arange(num_tokens, device=runner.device)
            for name in (
                "draft_tokens",
                "draft_token_confidence_probs",
                "hidden_states",
                "context_positions",
                "idx_mapping",
                "_draft_scatter_buf",
            ):
                tensor = getattr(speculator, name, None)
                if not isinstance(tensor, torch.Tensor):
                    continue
                if name in ("hidden_states", "context_positions"):
                    rows = token_prefix[: min(num_tokens, tensor.shape[0])]
                    meaningful_count = None
                elif name in (
                    "draft_tokens",
                    "draft_token_confidence_probs",
                    "idx_mapping",
                    "_draft_scatter_buf",
                ):
                    rows = torch.arange(tensor.shape[0], device=tensor.device)
                    meaningful_count = min(num_reqs, tensor.shape[0])
                else:
                    rows = prefix_rows[: min(num_reqs, tensor.shape[0])]
                    meaningful_count = None
                registry.add_rows(
                    f"speculator:{name}",
                    tensor,
                    rows,
                    meaningful_count=meaningful_count,
                )
            for name in ("draft_logits", "temperature", "seeds"):
                tensor = getattr(speculator, name, None)
                if isinstance(tensor, torch.Tensor):
                    registry.add_rows(f"speculator:{name}", tensor, req_slots)
            for name in ("sample_indices", "sample_pos", "sample_idx_mapping"):
                tensor = getattr(speculator, name, None)
                if isinstance(tensor, torch.Tensor):
                    registry.add_rows(
                        f"speculator:{name}",
                        tensor,
                        torch.arange(tensor.shape[0], device=tensor.device),
                    )
            input_buffers = getattr(speculator, "input_buffers", None)
            if input_buffers is not None:
                for name in (
                    "input_ids",
                    "positions",
                    "is_padding",
                    "query_start_loc",
                    "seq_lens",
                    "dcp_local_seq_lens",
                ):
                    tensor = getattr(input_buffers, name)
                    registry.add_rows(
                        f"speculator:input:{name}",
                        tensor,
                        torch.arange(tensor.shape[0], device=tensor.device),
                    )
            draft_block_tables = getattr(speculator, "block_tables", None)
            draft_group_ids = getattr(speculator, "draft_kv_cache_group_ids", ())
            if draft_block_tables is not None and draft_group_ids:
                registry.add_rows(
                    "speculator:query_slot_mappings",
                    draft_block_tables.slot_mappings,
                    torch.tensor(
                        draft_group_ids,
                        dtype=torch.int64,
                        device=draft_block_tables.slot_mappings.device,
                    ),
                )
            for group_id, tensor in enumerate(
                getattr(speculator, "_context_slot_mappings", ())
            ):
                registry.add_rows(
                    f"speculator:context_slots:{group_id}",
                    tensor,
                    token_prefix[: min(num_tokens, tensor.shape[0])],
                )
            draft_model = getattr(speculator, "model", None)
            draft_model = getattr(draft_model, "model", draft_model)
            for name in (
                "topk_indices_buffer",
                "candidate_block_buffer",
                "_mtp_hidden_buffer",
            ):
                tensor = getattr(draft_model, name, None)
                if isinstance(tensor, torch.Tensor):
                    draft_query_tokens = num_reqs * int(
                        getattr(speculator, "num_query_per_req", 1)
                    )
                    retained_rows = min(
                        max(num_tokens, draft_query_tokens),
                        tensor.shape[0],
                    )
                    registry.add_rows(
                        f"speculator:model:{name}",
                        tensor,
                        torch.arange(retained_rows, device=tensor.device),
                    )
            registry.add_attribute(
                "speculator:draft_max_seq_len", speculator, "draft_max_seq_len"
            )
            estimator = getattr(speculator, "acceptance_estimator", None)
            if estimator is not None:
                for name in (
                    "slope",
                    "intercepts",
                    "info",
                    "grad",
                    "counts",
                ):
                    tensor = getattr(estimator, name)
                    registry.add_rows(
                        f"acceptance:{name}",
                        tensor,
                        torch.arange(tensor.shape[0], device=tensor.device),
                    )
                for name in ("features", "predictions"):
                    registry.add_rows(
                        f"acceptance:{name}",
                        getattr(estimator, name),
                        req_slots,
                    )
                for name in ("_steps_since_refit", "_refits"):
                    registry.add_attribute(f"acceptance:{name}", estimator, name)

        adaptive = runner.adaptive_verification
        if adaptive is not None:
            for event in adaptive._copy_events:
                event.synchronize()
            registry.add_rows(
                "adaptive:confidence_probs",
                adaptive._confidence_probs,
                req_slots,
            )
            for name in ("_batch_draft_capacity", "_num_non_draft_tokens"):
                tensor = getattr(adaptive, name)
                registry.add_rows(
                    f"adaptive:{name}",
                    tensor,
                    torch.arange(min(num_reqs, tensor.shape[0]), device=tensor.device),
                )
            registry.add_rows(
                "adaptive:cu_num_logits",
                adaptive._cu_num_logits,
                torch.arange(
                    min(num_reqs + 1, adaptive._cu_num_logits.shape[0]),
                    device=adaptive._cu_num_logits.device,
                ),
            )
            for index, stale in enumerate(adaptive._stale_confidences):
                registry.add_rows(
                    f"adaptive:stale_gpu:{index}",
                    stale.gpu,
                    torch.arange(stale.gpu.shape[0], device=stale.gpu.device),
                )
                registry.add_rows(
                    f"adaptive:stale_cpu:{index}",
                    stale.cpu,
                    torch.arange(stale.cpu.shape[0]),
                )
            for name in ("_stale_idx", "_pending_resets", "_batch_budget"):
                registry.add_attribute(f"adaptive:{name}", adaptive, name)

        draft_handler = runner.draft_tokens_handler
        for name in ("req_ids", "draft_tokens_np", "num_draft_tokens"):
            registry.add_attribute(f"draft_handler:{name}", draft_handler, name)

    def _target_decoder_model(self) -> Any:
        model = getattr(self.runner.model, "language_model", self.runner.model)
        return getattr(model, "model", model)
