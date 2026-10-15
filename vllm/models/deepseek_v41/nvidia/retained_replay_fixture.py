# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off retained replay fixture for the DeepSeek V4.1 L20 harness."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.models.deepseek_v41.nvidia.model import DecoderBoundaryState
    from vllm.models.deepseek_v41.nvidia.model_state import (
        OrdinaryDecodeBatch,
        RetainedReplayBatch,
    )
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class RetainedReplayFixtureError(RuntimeError):
    pass


_ByteInterval = tuple[int, int]


def clone_owned(value: Any) -> Any:
    """Clone mutable tensor/array leaves while preserving immutable owners."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: clone_owned(getattr(value, field.name))
            for field in fields(value)
        }
        try:
            return replace(value, **updates)
        except TypeError:
            clone = copy.copy(value)
            for name, field_value in updates.items():
                object.__setattr__(clone, name, field_value)
            return clone
    if isinstance(value, dict):
        return {clone_owned(key): clone_owned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_owned(item) for item in value]
    if isinstance(value, tuple):
        return (
            type(value)(*(clone_owned(item) for item in value))
            if hasattr(value, "_fields")
            else tuple(clone_owned(item) for item in value)
        )
    if isinstance(value, set):
        return {clone_owned(item) for item in value}
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
    ) -> None:
        self._claim(name)
        rows = torch.unique(rows.to(device=tensor.device, dtype=torch.int64))
        entry = _TensorSelection(name, tensor, rows, dim)
        entry.capture("s0")
        self.entries.append(entry)

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
    registry: TouchedStateRegistry
    source_digest: str
    boundary: DecoderBoundaryState | None = None
    reference: Any = None
    fixture_digest: str | None = None


@dataclass
class _CacheIntervals:
    name: str
    tensor: torch.Tensor
    destination: list[_ByteInterval]
    source: list[_ByteInterval]


class RetainedReplayFixtureController:
    def __init__(self, runner: GPUModelRunner) -> None:
        self.runner = runner
        self.state = "armed"
        self.error: str | None = None
        self.fixture: RetainedFixture | None = None
        self.fixture_id = 0
        self.verify_runs: list[dict[str, Any]] = []
        self.boundary_error: str | None = None

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
            "snapshot_bytes": (
                None if fixture is None else fixture.registry.snapshot_bytes
            ),
            "source_bytes": (
                None if fixture is None else fixture.registry.source_bytes
            ),
            "source_digest": (None if fixture is None else fixture.source_digest),
            "fixture_digest": (None if fixture is None else fixture.fixture_digest),
            "boundary_error": self.boundary_error,
            "verify_runs": self.verify_runs,
        }

    def fail(self, error: BaseException | str) -> None:
        self.state = "error"
        self.error = (
            error if isinstance(error, str) else f"{type(error).__name__}: {error}"
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
        registry = self._build_registry(
            input_batch,
            block_tables,
            tuple(writer_contexts),
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
            registry=registry,
            source_digest=source_digest,
        )
        self.state = "capturing"

    def capture_boundary(self, boundary: DecoderBoundaryState) -> None:
        if self.state != "capturing" or self.fixture is None:
            return
        if self.fixture.boundary is not None:
            self.boundary_error = "the L20 boundary was captured twice"
            return
        try:
            self.fixture.boundary = clone_owned(boundary)
        except Exception as error:
            self.boundary_error = f"{type(error).__name__}: {error}"

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
        if fixture.boundary is None:
            raise RetainedReplayFixtureError("the real L20 boundary was not captured")
        torch.cuda.synchronize(self.runner.device)
        fixture.registry.capture_epoch("s1")
        source_digest = fixture.registry.source_digest()
        if source_digest != fixture.source_digest:
            raise RetainedReplayFixtureError(
                "source KV changed during the real capture step"
            )
        fixture.reference = clone_owned(
            {
                "sampled_token_ids": sampled_token_ids,
                "num_sampled": num_sampled,
                "num_rejected": num_rejected,
            }
        )
        fixture.fixture_digest = _tree_digest(
            {
                "input_batch": fixture.input_batch,
                "model_inputs": fixture.model_inputs,
                "block_tables": fixture.block_tables,
                "slot_mappings": fixture.slot_mappings,
                "full_forward_context": fixture.full_forward_context,
                "replay_batch": fixture.replay_batch,
                "ordinary_batch": fixture.ordinary_batch,
                "boundary": fixture.boundary,
                "reference": fixture.reference,
                "s0": fixture.registry.digest("s0"),
                "s1": fixture.registry.digest("s1"),
                "source": fixture.source_digest,
            }
        )
        self.state = "ready"

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

    def _build_registry(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        writer_contexts: tuple[tuple[ForwardContext, int], ...],
    ) -> TouchedStateRegistry:
        registry = TouchedStateRegistry()
        cache_intervals = self._collect_cache_intervals(
            input_batch, block_tables, writer_contexts
        )
        for key, cache in cache_intervals.items():
            del key
            if cache.destination:
                registry.add_intervals(
                    f"kv-dst:{cache.name}", cache.tensor, cache.destination
                )
            if cache.source:
                registry.add_source_intervals(
                    f"kv-src:{cache.name}", cache.tensor, cache.source
                )
        self._register_runtime_state(registry, input_batch)
        return registry

    def _collect_cache_intervals(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        writer_contexts: tuple[tuple[ForwardContext, int], ...],
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
                    )
                    intervals_by_storage[key] = record
                destination_blocks: set[int] = set()
                for attn_metadata, num_tokens in metadata_contexts:
                    metadata = attn_metadata.get(layer_name)
                    if metadata is None:
                        continue
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
                record.destination.extend(
                    _cache_block_byte_intervals(layer_name, cache, mutable_blocks)
                )
                record.destination.extend(
                    _cache_block_byte_intervals(layer_name, cache, proposal_blocks)
                )
                record.source.extend(
                    _cache_block_byte_intervals(layer_name, cache, group_blocks)
                )
        if not intervals_by_storage:
            raise RetainedReplayFixtureError("no bound KV cache rows were discovered")
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
                rows = (
                    token_prefix[: min(num_tokens, tensor.shape[0])]
                    if name in ("hidden_states", "context_positions")
                    else prefix_rows[: min(num_reqs, tensor.shape[0])]
                )
                registry.add_rows(f"speculator:{name}", tensor, rows)
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
                    registry.add_rows(
                        f"speculator:model:{name}",
                        tensor,
                        token_prefix[: min(num_tokens, tensor.shape[0])],
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
