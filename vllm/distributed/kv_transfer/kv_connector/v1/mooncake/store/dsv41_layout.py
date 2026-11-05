# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, topology-independent byte layout for the DSv4.1 global cache."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import regex as re

FORMAT = "dsv41_canonical_v1"
PROFILE = "dsv41_flash_v1"
_SOURCES = (2, 8, 14, 20)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.attn(\.indexer\.k_cache)?$")


def component_id(layer_name: str, *, relay: bool = False) -> str:
    match = _LAYER.search(layer_name)
    if match is None:
        raise ValueError(f"Not a DSv4.1 global cache: {layer_name}")
    source = int(match[1])
    if relay and source == 10 and match[2] is None:
        source = 8
    if source not in _SOURCES:
        raise ValueError(f"Unexpected global cache source: {layer_name}")
    kind = "indexer_k" if match[2] else "main_kv"
    return f"layer_{source}/{kind}"


@dataclass(frozen=True, order=True)
class Component:
    name: str
    block_tokens: int
    page_bytes: int
    tokens_per_state: int
    dtype: str
    abi: str = ""

    def __post_init__(self) -> None:
        if min(self.block_tokens, self.page_bytes, self.tokens_per_state) <= 0:
            raise ValueError("Canonical cache geometry must be positive")
        if self.block_tokens % self.tokens_per_state:
            raise ValueError("Cache blocks must contain complete compressed states")


@dataclass
class Binding:
    component: Component
    group_id: int
    base_address: int
    block_stride: int
    num_blocks: int
    writer: bool = True
    tensor: Any = field(default=None, repr=False)

    def regions(self, start: int, end: int, blocks: list[int]) -> list[tuple[int, int]]:
        span = self.component.block_tokens
        if start < 0 or end < start or start % span or end % span:
            raise ValueError("Canonical transfer must cover complete physical pages")
        if self.block_stride < self.component.page_bytes:
            raise ValueError("Canonical page exceeds its GPU allocation stride")
        regions = []
        for logical in range(start // span, end // span):
            if logical >= len(blocks):
                raise ValueError("Canonical transfer extends past the block table")
            physical = blocks[logical]
            if not 0 < physical < self.num_blocks:
                raise ValueError("Canonical transfer references a null/invalid block")
            regions.append(
                (
                    self.base_address + physical * self.block_stride,
                    self.component.page_bytes,
                )
            )
        return regions


def merge_components(worker_components: list[list[Component]]) -> tuple[Component, ...]:
    merged: dict[str, Component] = {}
    for components in worker_components:
        for component in components:
            previous = merged.setdefault(component.name, component)
            if previous != component:
                raise ValueError(f"Incompatible canonical replicas: {component.name}")
    required = {f"layer_{s}/{k}" for s in _SOURCES for k in ("main_kv", "indexer_k")}
    if set(merged) != required:
        raise ValueError(f"Incomplete canonical manifest: {required ^ set(merged)}")
    return tuple(sorted(merged.values()))


@dataclass(frozen=True)
class Manifest:
    namespace: str
    revision: str
    components: tuple[Component, ...]
    hash_tokens: int

    def __post_init__(self) -> None:
        if not self.namespace or not self.revision or self.hash_tokens <= 0:
            raise ValueError("Canonical store needs namespace, revision and hash span")
        if not self.components:
            raise ValueError("Canonical store needs a complete component manifest")

    @cached_property
    def chunk_tokens(self) -> int:
        return math.lcm(self.hash_tokens, *(c.block_tokens for c in self.components))

    @cached_property
    def prefix(self) -> str:
        geometry = [vars(c) for c in self.components]
        payload = json.dumps(
            [FORMAT, self.namespace, self.revision, self.hash_tokens, geometry],
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{FORMAT}@{hashlib.sha256(payload.encode()).hexdigest()}"

    def key(self, component: Component, block_hash: bytes) -> str:
        if component not in self.components:
            raise ValueError("Component is outside the canonical manifest")
        return f"{self.prefix}@{component.name}@{block_hash.hex()}"


def build_bindings(
    config: Any, caches: dict[str, Any], *, prefill: bool
) -> list[Binding]:
    from vllm.utils.torch_utils import is_non_overlapping_and_dense
    from vllm.v1.kv_cache_interface import group_kernel_blocks

    bindings = []
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            relay = prefill and name.endswith(".layers.10.attn")
            if not group.enable_kv_transfer and not relay:
                continue
            spec = getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
                name, group.kv_cache_spec
            )
            if not spec.prefix_cacheable:
                continue
            if name not in caches:
                raise ValueError(f"Canonical KV tensor is missing: {name}")
            cache = group_kernel_blocks(caches[name], config.num_blocks)
            if not is_non_overlapping_and_dense(cache[0]):
                raise ValueError(f"Canonical page must be contiguous: {name}")
            component = Component(
                component_id(name, relay=relay),
                spec.block_size,
                spec.unpadded_page_size_bytes,
                spec.tokens_per_state,
                str(cache.dtype),
                json.dumps(
                    {
                        key: str(getattr(spec, key, None))
                        for key in (
                            "dtype",
                            "num_heads",
                            "head_size",
                            "num_head_slots",
                            "state_content_bytes",
                            "kv_quant_mode",
                            "cache_dtype_str",
                            "model_version",
                        )
                    },
                    sort_keys=True,
                ),
            )
            stride = cache.stride(0) * cache.element_size()
            if component.page_bytes > cache[0].numel() * cache.element_size():
                raise ValueError(f"Canonical page is larger than its view: {name}")
            bindings.append(
                Binding(
                    component,
                    group_id,
                    cache.data_ptr(),
                    stride,
                    config.num_blocks,
                    writer=not relay,
                    tensor=cache,
                )
            )
    return bindings
