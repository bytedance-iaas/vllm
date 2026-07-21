# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)


@dataclass(frozen=True)
class TransferRegion:
    """A Mooncake-registered KV buffer plus vLLM cache identity metadata.

    layer_aliases mirrors KVCacheTensor.shared_by for tensors shared by
    multiple layer names. logical_group_indices and alias_group_indices
    preserve the KV cache group ownership needed to filter transfers for
    hybrid caches.
    """

    layer_name: str
    layer_index: int
    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0
    layer_aliases: tuple[str, ...] = ()
    layer_indices: tuple[int, ...] = ()
    logical_group_indices: tuple[int, ...] = ()
    alias_group_indices: tuple[tuple[int, ...], ...] = ()

    @property
    def match_layer_names(self) -> tuple[str, ...]:
        return self.layer_aliases or (self.layer_name,)

    @property
    def match_layer_indices(self) -> tuple[int, ...]:
        return self.layer_indices or (self.layer_index,)


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    is_kv_layout_blocks_first: bool,
    group_indices: list[int] | None = None,
    split_kv_regions: list[bool] | None = None,
    layer_aliases: list[list[str]] | None = None,
    layer_index_aliases: list[list[int]] | None = None,
    logical_group_indices: list[list[int]] | None = None,
    alias_group_indices: list[list[list[int]]] | None = None,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), (
        "Mooncake transfer regions require matching metadata lengths, got "
        f"base_addrs={len(base_addrs)}, block_lens={len(block_lens)}, "
        f"kv_block_lens={len(kv_block_lens)}, "
        f"layer_names={len(layer_names)}, "
        f"layer_indices={len(layer_indices)}."
    )
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names), (
        "Mooncake transfer regions require matching group metadata lengths, "
        f"got group_indices={len(group_indices)}, layer_names={len(layer_names)}."
    )
    if split_kv_regions is None:
        split_kv_regions = [is_kv_layout_blocks_first] * len(layer_names)
    assert len(split_kv_regions) == len(layer_names), (
        "Mooncake transfer regions require matching split metadata, "
        f"got split_kv_regions={len(split_kv_regions)}, "
        f"layer_names={len(layer_names)}."
    )
    regions: list[TransferRegion] = []
    for idx, (
        base_addr,
        block_len,
        kv_block_len,
        layer_name,
        layer_index,
        group_index,
        split_kv_region,
    ) in enumerate(zip(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        group_indices,
        split_kv_regions,
    )):
        if split_kv_region:
            aliases: tuple[str, ...] = ()
            index_aliases: tuple[int, ...] = ()
            region_logical_group_indices: tuple[int, ...] = ()
            region_alias_group_indices: tuple[tuple[int, ...], ...] = ()
        else:
            aliases = _get_region_layer_aliases(layer_aliases or [], idx)
            index_aliases = _get_region_layer_indices(layer_index_aliases or [], idx)
            region_logical_group_indices = _get_region_logical_group_indices(
                logical_group_indices or [], idx
            )
            region_alias_group_indices = _get_region_alias_group_indices(
                alias_group_indices or [], idx
            )
        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
                layer_aliases=aliases,
                layer_indices=index_aliases,
                logical_group_indices=region_logical_group_indices,
                alias_group_indices=region_alias_group_indices,
            )
        )
        if split_kv_region:
            regions.append(
                TransferRegion(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                    group_index=group_index,
                )
            )
    return regions


def _get_region_layer_aliases(aliases: list[list[str]], idx: int) -> tuple[str, ...]:
    if idx < len(aliases) and aliases[idx]:
        return tuple(aliases[idx])
    return ()


def _get_region_layer_indices(
    index_aliases: list[list[int]], idx: int
) -> tuple[int, ...]:
    if idx < len(index_aliases) and index_aliases[idx]:
        return tuple(index_aliases[idx])
    return ()


def _get_region_logical_group_indices(
    group_indices: list[list[int]], idx: int
) -> tuple[int, ...]:
    if idx < len(group_indices) and group_indices[idx]:
        return tuple(group_indices[idx])
    return ()


def _get_region_alias_group_indices(
    alias_group_indices: list[list[list[int]]], idx: int
) -> tuple[tuple[int, ...], ...]:
    if idx < len(alias_group_indices) and alias_group_indices[idx]:
        return tuple(tuple(groups) for groups in alias_group_indices[idx])
    return ()


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _region_has_aliases(region: TransferRegion) -> bool:
    return bool(region.layer_aliases)


def _alias_group_map(region: TransferRegion) -> dict[str, dict[int, set[int]]]:
    if (
        not region.layer_aliases
        or not region.layer_indices
        or not region.alias_group_indices
    ):
        return {}

    alias_groups: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for alias, layer_index, group_indices in zip(
        region.layer_aliases,
        region.layer_indices,
        region.alias_group_indices,
    ):
        alias_groups[alias][layer_index].update(group_indices)
    return alias_groups


def _regions_have_bound_alias_layer_indices(
    local_region: TransferRegion, remote_region: TransferRegion
) -> bool:
    local_alias_groups = _alias_group_map(local_region)
    remote_alias_groups = _alias_group_map(remote_region)
    for alias in set(local_alias_groups) & set(remote_alias_groups):
        if set(local_alias_groups[alias]) & set(remote_alias_groups[alias]):
            return True
    return False


def _regions_share_layer_identity(
    local_region: TransferRegion, remote_region: TransferRegion
) -> bool:
    return bool(
        set(local_region.match_layer_names) & set(remote_region.match_layer_names)
    )


def _align_transfer_regions_by_occurrence(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, int], TransferRegion]]:
        counts: dict[str, int] = defaultdict(int)
        keyed: list[tuple[tuple[str, int], TransferRegion]] = []
        for region in regions:
            occurrence = counts[region.layer_name]
            counts[region.layer_name] += 1
            keyed.append(((region.layer_name, occurrence), region))
        return keyed

    local_keyed = keyed_regions(local_regions)
    remote_keyed = keyed_regions(remote_regions)
    remote_by_key = dict(remote_keyed)
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in local_keyed:
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            continue
        if local_region.layer_index != remote_region.layer_index:
            return (
                [],
                [],
                (
                    "Mooncake registered layer index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.layer_index}, consumer="
                    f"{remote_region.layer_index}."
                ),
            )
        if local_region.group_index != remote_region.group_index:
            return (
                [],
                [],
                (
                    "Mooncake registered group index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.group_index}, consumer="
                    f"{remote_region.group_index}."
                ),
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)

    return aligned_local, aligned_remote, None


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    """Align KV transfer regions by vLLM cache identity.

    wrong once producer and consumer have different PP layouts. For shared
    physical tensors, alias metadata carries the KVCacheTensor.shared_by layer
    names and logical group metadata carries KVCacheGroupSpec ownership across
    the Mooncake wire boundary.
    """
    has_aliases = any(
        _region_has_aliases(region) for region in local_regions + remote_regions
    )
    if has_aliases:
        alias_local_regions = [
            region for region in local_regions if _region_has_aliases(region)
        ]
        alias_remote_regions = [
            region for region in remote_regions if _region_has_aliases(region)
        ]
        legacy_local_regions = [
            region for region in local_regions if not _region_has_aliases(region)
        ]
        legacy_remote_regions = [
            region for region in remote_regions if not _region_has_aliases(region)
        ]
        if any(
            _regions_share_layer_identity(alias_region, legacy_region)
            for alias_region in alias_local_regions
            for legacy_region in legacy_remote_regions
        ) or any(
            _regions_share_layer_identity(legacy_region, alias_region)
            for legacy_region in legacy_local_regions
            for alias_region in alias_remote_regions
        ):
            return (
                [],
                [],
                (
                    "Mooncake alias metadata is present on only one side of "
                    "matching transfer regions. Producer and consumer must use "
                    "the same Mooncake metadata schema."
                ),
            )
        has_alias_group_metadata = all(
            bool(region.alias_group_indices)
            for region in alias_local_regions + alias_remote_regions
        )
        if not has_alias_group_metadata:
            return (
                [],
                [],
                (
                    "Mooncake alias metadata is missing alias-group ownership. "
                    "Producer and consumer must use the same Mooncake metadata "
                    "schema."
                ),
            )

        # DeepSeek V4 shared-cache regions bind each alias to the layer
        # index and cache groups that own that view of the shared tensor.
        # Matching the bound identity avoids transferring unrelated groups
        # when one physical region backs multiple logical cache entries.
        alias_group_aligned_local: list[TransferRegion] = []
        alias_group_aligned_remote: list[TransferRegion] = []
        matched_local_indices: set[int] = set()
        matched_remote_alias_group_keys: set[tuple[int, tuple[str, int, int]]] = set()

        for local_idx, local_region in enumerate(alias_local_regions):
            alias_group_index_mismatch_region: TransferRegion | None = None
            matched_local_alias_group_keys: set[tuple[str, int, int]] = set()
            for remote_idx, candidate_remote_region in enumerate(alias_remote_regions):
                if not _regions_share_layer_identity(
                    local_region, candidate_remote_region
                ):
                    continue
                if not _regions_have_bound_alias_layer_indices(
                    local_region, candidate_remote_region
                ):
                    if alias_group_index_mismatch_region is None:
                        alias_group_index_mismatch_region = candidate_remote_region
                    continue
                shared_alias_group_keys = _shared_alias_group_keys(
                    local_region, candidate_remote_region
                )
                if shared_alias_group_keys is None or not shared_alias_group_keys:
                    continue
                available_alias_group_keys = [
                    alias_group_key
                    for alias_group_key in shared_alias_group_keys
                    if (remote_idx, alias_group_key)
                    not in matched_remote_alias_group_keys
                ]
                available_alias_group_keys = [
                    alias_group_key
                    for alias_group_key in available_alias_group_keys
                    if alias_group_key not in matched_local_alias_group_keys
                ]
                if not available_alias_group_keys:
                    continue
                matched_local_alias_group_keys.update(available_alias_group_keys)
                matched_remote_alias_group_keys.update(
                    (remote_idx, alias_group_key)
                    for alias_group_key in available_alias_group_keys
                )
                alias_group_aligned_local.append(local_region)
                alias_group_aligned_remote.append(candidate_remote_region)
                matched_local_indices.add(local_idx)

            if (
                local_idx not in matched_local_indices
                and alias_group_index_mismatch_region is not None
            ):
                return (
                    [],
                    [],
                    (
                        "Mooncake registered layer index mismatch for "
                        f"{local_region.match_layer_names}: producer="
                        f"{local_region.match_layer_indices}, consumer="
                        f"{alias_group_index_mismatch_region.match_layer_indices}."
                    ),
                )

        for local_idx, local_region in enumerate(alias_local_regions):
            if local_idx in matched_local_indices:
                continue
            if any(
                _regions_share_layer_identity(local_region, remote_region)
                for remote_region in alias_remote_regions
            ):
                return (
                    [],
                    [],
                    (
                        "Mooncake producer registered layer aliases have no "
                        "matching consumer alias groups: "
                        f"{sorted(local_region.match_layer_names)}."
                    ),
                )

        for remote_idx, remote_region in enumerate(alias_remote_regions):
            for local_region in alias_local_regions:
                if not _regions_share_layer_identity(local_region, remote_region):
                    continue
                shared_alias_group_keys = _shared_alias_group_keys(
                    local_region, remote_region
                )
                if not shared_alias_group_keys:
                    continue
                unmatched_alias_group_keys = [
                    alias_group_key
                    for alias_group_key in shared_alias_group_keys
                    if (remote_idx, alias_group_key)
                    not in matched_remote_alias_group_keys
                ]
                if unmatched_alias_group_keys:
                    return (
                        [],
                        [],
                        (
                            "Mooncake duplicate alias group match for "
                            f"{remote_region.match_layer_names}: "
                            f"groups={unmatched_alias_group_keys}."
                        ),
                    )

        legacy_aligned_local, legacy_aligned_remote, legacy_err = (
            _align_transfer_regions_by_occurrence(
                legacy_local_regions,
                legacy_remote_regions,
            )
        )
        if legacy_err is not None:
            return [], [], legacy_err
        return (
            alias_group_aligned_local + legacy_aligned_local,
            alias_group_aligned_remote + legacy_aligned_remote,
            None,
        )

    return _align_transfer_regions_by_occurrence(local_regions, remote_regions)


def _common_group_indices_for_regions(
    local_region: TransferRegion, remote_region: TransferRegion, num_groups: int
) -> tuple[int, ...]:
    """Return the KV cache groups shared by two aligned transfer regions."""

    if num_groups <= 0:
        return ()
    groups_from_aliases = _common_group_indices_from_aliases(
        local_region, remote_region, num_groups
    )
    if groups_from_aliases is not None:
        return groups_from_aliases
    if local_region.logical_group_indices and remote_region.logical_group_indices:
        common_group_indices = sorted(
            set(local_region.logical_group_indices)
            & set(remote_region.logical_group_indices)
        )
        return tuple(
            group_idx
            for group_idx in common_group_indices
            if 0 <= group_idx < num_groups
        )
    if bool(local_region.logical_group_indices) != bool(
        remote_region.logical_group_indices
    ):
        # Legacy peers/regions did not carry group ownership metadata.
        return tuple(range(num_groups))
    if local_region.group_index == remote_region.group_index:
        return (local_region.group_index,) if local_region.group_index < num_groups else ()
    return ()


def _common_group_indices_from_aliases(
    local_region: TransferRegion, remote_region: TransferRegion, num_groups: int
) -> tuple[int, ...] | None:
    group_indices = _shared_alias_group_indices(local_region, remote_region)
    if group_indices is None:
        return None
    return tuple(g for g in group_indices if 0 <= g < num_groups)


def _shared_alias_group_indices(
    local_region: TransferRegion, remote_region: TransferRegion
) -> tuple[int, ...] | None:
    shared_alias_group_keys = _shared_alias_group_keys(local_region, remote_region)
    if shared_alias_group_keys is None:
        return None
    return tuple(sorted({group_idx for _, _, group_idx in shared_alias_group_keys}))


def _shared_alias_group_keys(
    local_region: TransferRegion, remote_region: TransferRegion
) -> tuple[tuple[str, int, int], ...] | None:
    if not local_region.alias_group_indices or not remote_region.alias_group_indices:
        return None

    local_alias_groups = _alias_group_map(local_region)
    remote_alias_groups = _alias_group_map(remote_region)
    common_aliases = set(local_alias_groups) & set(remote_alias_groups)

    alias_group_keys: set[tuple[str, int, int]] = set()
    for alias in common_aliases:
        common_layer_indices = set(local_alias_groups[alias]) & set(
            remote_alias_groups[alias]
        )
        for layer_index in common_layer_indices:
            common_group_indices = (
                local_alias_groups[alias][layer_index]
                & remote_alias_groups[alias][layer_index]
            )
            for group_idx in common_group_indices:
                alias_group_keys.add((alias, layer_index, group_idx))
    return tuple(sorted(alias_group_keys))


def _select_region_block_ids(
    local_block_ids_per_group: list[list[int]],
    remote_block_ids_per_group: list[list[int]],
    group_indices: tuple[int, ...],
) -> tuple[list[int], list[int], str | None]:
    local_block_ids: list[int] = []
    remote_block_ids: list[int] = []

    for group_idx in group_indices:
        local_group = local_block_ids_per_group[group_idx]
        remote_group = remote_block_ids_per_group[group_idx]
        n_local = len(local_group)
        n_remote = len(remote_group)
        if n_remote == 0:
            continue
        if n_local < n_remote:
            return [], [], "P num blocks less than D"
        if n_local > n_remote:
            local_group = local_group[-n_remote:]
        local_block_ids.extend(local_group)
        remote_block_ids.extend(remote_group)

    return local_block_ids, remote_block_ids, None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    registered_layer_aliases: list[list[str]] = msgspec.field(default_factory=list)
    registered_layer_index_aliases: list[list[int]] = msgspec.field(
        default_factory=list
    )
    registered_logical_group_indices: list[list[int]] = msgspec.field(
        default_factory=list
    )
    registered_alias_group_indices: list[list[list[int]]] = msgspec.field(
        default_factory=list
    )
    # D-side C128 staging pool advertised to P.
    c128_import_base_addr: int = 0
    c128_import_slot_bytes: int = 0
    c128_num_layers: int = 0
    c128_state_row_bytes: int = 0
    c128_layer_indices: list[int] = msgspec.field(default_factory=list)
    # Per-request C128 import slot and partial-state flag.
    c128_req_import_slot: dict[ReqId, int] = msgspec.field(default_factory=dict)
    c128_req_needs_partial: dict[ReqId, bool] = msgspec.field(default_factory=dict)


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0
    pull_failed: bool = False
    # D-side C128 staging slot; absent for aligned prompts.
    c128_import_slot: int | None = None
    c128_needs_partial: bool = False
    # Shared import slot is released only after every TP/PP pull task quiesces.
    c128_pull_pending: int = 0
    c128_pull_failed: bool = False
    c128_abort_pending: bool = False


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0
    # P-side export slot for committed bank0 snapshot.
    c128_export_slot: int | None = None


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        self.reqs_not_processed: set[TransferId] = set()
        # Producer req_ids whose C128 export slot must be released without send.
        self.c128_export_release_req_ids: set[ReqId] = set()
        # Consumer req_ids whose C128 import slot is not consumed by admission.
        self.c128_import_release_req_ids: set[ReqId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        c128_needs_partial: bool = False,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
                c128_needs_partial=c128_needs_partial,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    supports_online_c128_state_transfer = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None, (
                "kv_cache_config is required for SCHEDULER role"
            )
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def snapshot_c128_state(self, req_id: str, p_req_state_idx: int) -> None:
        """P side: snapshot committed bank0 before request-slot reuse."""
        if self.connector_worker is not None:
            self.connector_worker.snapshot_c128_state(req_id, p_req_state_idx)

    def bind_c128_state_index(self, req_id: str, p_req_state_idx: int) -> None:
        """P side: remember the live request-state slot for lazy snapshot."""
        if self.connector_worker is not None:
            self.connector_worker.bind_c128_state_index(req_id, p_req_state_idx)

    def restore_c128_state(self, req_id: str, d_req_state_idx: int) -> None:
        """D side: materialize bank0 after remote-prefill admission."""
        if self.connector_worker is not None:
            self.connector_worker.restore_c128_state(req_id, d_req_state_idx)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return worker-local transfer stats since the last call.

        Note the P/D asymmetry: because Mooncake is P-push (P calls
        batch_transfer_sync_write), P records successful transfer latency,
        bytes, and descriptor counts, while D only records failures
        (recv/ZMQ errors). Aggregated NIXL-style dashboards will find
        successful-transfer metrics on the P worker, not D.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.connector_worker is None:
            return set()
        return self.connector_worker.get_block_ids_with_load_errors()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MooncakeKVConnectorStats(data=data or {})


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        # GDN is represented as a MambaSpec in vLLM. This Mooncake MambaSpec
        # path is currently tested with GDN; Mamba2 is not validated yet.
        self._has_mamba = kv_cache_config.has_mamba_layers

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()
        # Producer req_ids whose snapshotted export slot has no send.
        self._c128_export_release_req_ids: set[ReqId] = set()
        # Consumer req_ids whose import slot is not consumed by admission.
        self._c128_import_release_req_ids: set[ReqId] = set()

        # Online C128 state transfer is valid only for pure producer/consumer roles.
        self._online_c128_state_transfer_enabled = (
            bool(envs.VLLM_USE_ONLINE_C128_COMPRESS)
            and bool(envs.VLLM_USE_ONLINE_C128_PD_TRANSFER)
            and (self.is_kv_producer or self.is_kv_consumer)
        )

        # Compute sliding window block counts per KV cache group.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to
        # conservatively account for boundary overlap.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block IDs to sliding window size."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = self._get_remote_prefill_token_count(len(token_ids)) - (
                num_computed_tokens
            )
            if count > 0:
                return count, True

        if params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                # Only non-128-aligned resumes need C128 partial state.
                c128_needs_partial = (
                    self._online_c128_state_transfer_enabled
                    and (req.num_computed_tokens % 128) != 0
                )
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    c128_needs_partial=c128_needs_partial,
                )
            self._reqs_need_recv.clear()
            meta.c128_import_release_req_ids = self._c128_import_release_req_ids
            self._c128_import_release_req_ids = set()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()
            meta.c128_export_release_req_ids = self._c128_export_release_req_ids
            self._c128_export_release_req_ids = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            # No KV transfer for this request (e.g. a purely-local / exception
            # request on a producer process). The worker still snapshotted its
            # Slot was snapshotted at recycle time; release it without a send.
            if self._online_c128_state_transfer_enabled and self.is_kv_producer:
                self._c128_export_release_req_ids.add(request.request_id)
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            # Producer ran this request but it is NOT a remote-decode send (e.g.
            # No remote decode send: release the snapshotted export slot.
            if self._online_c128_state_transfer_enabled and self.is_kv_producer:
                self._c128_export_release_req_ids.add(request.request_id)
            # Consumer: a remote-prefill request whose do_remote_prefill was
            # already cleared by update_state_after_alloc lands here on abort. If
            # it was aborted AFTER recv but BEFORE admission, the admission-time
            # restore_c128_state never runs, so flag its reserved import slot for
            # release. (Harmless if no slot was reserved — release is idempotent.)
            if self._online_c128_state_transfer_enabled and self.is_kv_consumer:
                self._c128_import_release_req_ids.add(request.request_id)
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            # Aborted / non-length-capped: no send will happen, so release the
            # snapshotted export slot (by req_id) to avoid leaking it.
            if self._online_c128_state_transfer_enabled and self.is_kv_producer:
                self._c128_export_release_req_ids.add(request.request_id)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (
                request,
                self.get_sw_clipped_blocks(block_ids),
            )
        elif self._online_c128_state_transfer_enabled and self.is_kv_producer:
            # Length-capped but no blocks to send: still no Mooncake send, so the
            # snapshotted export slot must be released.
            self._c128_export_release_req_ids.add(request.request_id)

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
        # may change the current CUDA device during engine.initialize().
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(
            "mooncake_protocol", "rdma"
        )
        device_name = kv_transfer_config.kv_connector_extra_config.get(
            "device_name", ""
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", protocol, device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.block_len_per_layer: list[int] = []
        self.kv_block_len_per_layer: list[int] = []
        self.registered_layer_names: list[str] = []
        self.registered_layer_indices: list[int] = []
        self.registered_group_indices: list[int] = []
        self.registered_layer_aliases: list[list[str]] = []
        self.registered_layer_index_aliases: list[list[int]] = []
        self.registered_logical_group_indices: list[list[int]] = []
        self.registered_alias_group_indices: list[list[list[int]]] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # State-transfer pools are allocated only for online C128 PD transfer.
        import vllm.envs as envs

        self._online_c128_enabled = bool(envs.VLLM_USE_ONLINE_C128_COMPRESS)
        # Gate state-transfer pools to pure PD producer/consumer roles.
        self._online_c128_state_transfer_enabled = (
            self._online_c128_enabled
            and bool(envs.VLLM_USE_ONLINE_C128_PD_TRANSFER)
            and (self.is_kv_producer or self.is_kv_consumer)
        )
        self._c128_states: list = []  # DeepseekOnlineC128State, one per layer
        self._c128_state_row_width: int = 0
        self._c128_state_row_bytes: int = 0
        self._c128_num_layers: int = 0
        self._c128_layer_indices: list[int] = []
        # P side: export pool is keyed by req_id until transfer_id is known.
        self._c128_export_pool = None  # C128ExportSlotPool
        self._c128_export_slots: dict[ReqId, int] = {}
        # Sender thread waits on this event before RDMA-reading export slots.
        self._c128_export_events: dict[ReqId, torch.cuda.Event] = {}
        self._c128_req_state_indices: dict[ReqId, int] = {}
        # D side: import pool (RDMA destination) + per-transfer staging info.
        self._c128_import_pool = None  # C128ImportSlotPool
        # transfer_id -> (import_slot, needs_partial); set when a pull is built.
        self._c128_import_slots: dict[TransferId, tuple[int, bool]] = {}
        # req_id -> transfer_id (D side), so the runner can drive restore by
        # req_id (it knows req_id -> req_state_idx; the connector knows
        # req_id -> transfer_id).
        self._c128_req_to_transfer: dict[ReqId, TransferId] = {}
        # D side live pulls; used to defer abort releases until RDMA quiesces.
        self._c128_active_pulls: dict[ReqId, PullReqMeta] = {}
        # Pulls pending bootstrap/slot reservation.
        self._c128_pending_import_reqs: set[ReqId] = set()
        # Abort tombstones for pending-before-reserve pulls.
        self._c128_aborted_import_reqs: set[ReqId] = set()

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            # Each pool thread must be bound to the correct CUDA device
            # because CUDA device selection is thread-local.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
                initializer=self._bind_sender_thread_device,
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()

        self.xfer_stats = MooncakeKVConnectorStats()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        self.attn_backends = get_current_attn_backends(vllm_config)
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug(
            "Detected attention backends %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._layer_specs: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                self._layer_specs[layer_name] = specs_by_layer.get(
                    layer_name, group_spec
                )
        self._layer_group_indices: dict[str, int] = {
            layer: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }
        self._layer_logical_group_indices: dict[str, list[int]] = defaultdict(list)
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer in group.layer_names:
                self._layer_logical_group_indices[layer].append(group_index)
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=kv_cache_config.has_mamba_layers,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def _sync_block_size_with_kernel(self) -> None:
        # When speculative decoding (e.g. Eagle) is enabled, the main model
        # and draft model may use different attention backends with different
        # physical block sizes. Pick the common (smallest) block size so that
        # KV-cache registration and transfer work correctly for both models.
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config) and hasattr(
                self, "bootstrap_server"
            ):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr,
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
            self.registered_layer_names,
            self.registered_layer_indices,
            self.registered_group_indices,
            self.registered_layer_aliases,
            self.registered_layer_index_aliases,
            self.registered_logical_group_indices,
            self.registered_alias_group_indices,
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
            meta.registered_layer_aliases,
            meta.registered_layer_index_aliases,
            meta.registered_logical_group_indices,
            meta.registered_alias_group_indices,
        )
        local_regions, remote_regions, align_err = _align_transfer_regions(
            local_regions, remote_regions
        )
        if align_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=align_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=local_regions,
            remote_regions=remote_regions,
            local_tp_size=self.tp_size,
            remote_tp_size=meta.remote_tp_size,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                pending_state = [
                    (
                        d_req_id,
                        send_meta.transfer_id,
                        send_meta.p_req_id,
                        bool(send_meta.local_block_ids),
                        send_meta.c128_export_slot,
                        send_meta.ready.is_set(),
                    )
                    for d_req_id, send_meta in pending_reqs.items()
                ]
                logger.warning("Timeout waiting for P side ready: %s", pending_state)
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            err_reqs: list[ReqId] = []
            err_req_set: set[ReqId] = set()
            err_msg: str | None = None
            ok_ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            try:
                (
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                    err_reqs,
                    err_msg,
                    state_transfer_events,
                ) = await self._build_transfer_params(
                    ready_reqs,
                    meta,
                    local_regions,
                    remote_regions,
                )
                err_req_set = set(err_reqs)
                ok_ready_reqs = [
                    (d_req_id, send_meta)
                    for d_req_id, send_meta in ready_reqs
                    if d_req_id not in err_req_set
                ]

                if src_ptrs:
                    remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                    ret_value = await self.sender_loop.run_in_executor(
                        self._sender_executor,
                        self._send_blocks,
                        remote_session,
                        src_ptrs,
                        dst_ptrs,
                        lengths,
                        state_transfer_events,
                    )

                    if ret_value != 0:
                        transfer_err_msg = (
                            f"Mooncake transfer engine returned {ret_value}"
                        )
                        err_msg = (
                            transfer_err_msg
                            if err_msg is None
                            else f"{err_msg}; {transfer_err_msg}"
                        )
                        err_reqs = list(err_reqs)
                        for d_req_id, _ in ok_ready_reqs:
                            err_reqs.append(d_req_id)
                            err_req_set.add(d_req_id)
                        ok_ready_reqs = []
            except Exception as e:
                err_msg = f"Failed to send Mooncake KV blocks: {e}"
                err_reqs = [d_req_id for d_req_id, _ in ready_reqs]
                err_req_set = set(err_reqs)
                ok_ready_reqs = []
            finally:
                for _, send_meta in ready_reqs:
                    send_meta.sending -= 1

            for d_req_id, send_meta in ready_reqs:
                if d_req_id in err_req_set:
                    continue

                send_meta.sent += 1
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)
                    # Release the C128 export slot now that the send is done.
                    self._release_c128_export_slot(send_meta)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(
        self,
        send_meta: SendBlockMeta,
        remote_tp_ranks: list[int],
    ):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: TP ranks=%s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    def _logical_to_kernel_block_ids(
        self, block_ids: list[list[int]]
    ) -> list[list[int]]:
        # For example, if a 544-token logical block is served by 32-token
        # FA kernel blocks, FA block id k expands to [17k, ..., 17k + 16],
        # while the matching Mamba/GDN state block remains k. Only attention
        # groups need logical block ids expanded to kernel block ids; Mamba/GDN
        # state block ids stay in the logical/page-id space.
        if self._physical_blocks_per_logical_kv_block == 1:
            return block_ids

        block_arange = np.arange(self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical_kv_block,
                block_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[ReqId],
        str | None,
        list["torch.cuda.Event"],
    ]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        # Guard async bank0->export-slot copies before RDMA reads.
        state_transfer_events: list[torch.cuda.Event] = []
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]

            if not remote_block_ids_per_group or all(
                len(g) == 0 for g in remote_block_ids_per_group
            ):
                continue

            if len(send_meta.local_block_ids) != len(remote_block_ids_per_group):
                logger.error(
                    "req %s: KV group count mismatch: local=%d, remote=%d",
                    d_req_id,
                    len(send_meta.local_block_ids),
                    len(remote_block_ids_per_group),
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "KV group count mismatch"
                continue

            # Keep KV-cache group identity. Hybrid/HMA groups can carry
            # different semantics (e.g. full-attention KV pages vs GDN/Mamba
            # inner-state slots), so their block IDs must not be flattened and
            # reused for every registered region.
            local_block_ids_by_group: list[list[int]] = []
            remote_block_ids_by_group: list[list[int]] = []
            has_block_error = False
            group_specs = self.kv_cache_config.kv_cache_groups
            for group_index, (local_group, remote_group) in enumerate(
                zip(send_meta.local_block_ids, remote_block_ids_per_group)
            ):
                is_mamba_group = isinstance(
                    group_specs[group_index].kv_cache_spec,
                    MambaSpec,
                )
                if is_mamba_group:
                    # Mamba/GDN prefix caching can use null blocks only as
                    # align-mode placeholders. They do not carry transferable
                    # state, so skip them on both producer and consumer sides.
                    local_group = [
                        block_id
                        for block_id in local_group
                        if block_id != NULL_BLOCK_ID
                    ]
                    remote_group = [
                        block_id
                        for block_id in remote_group
                        if block_id != NULL_BLOCK_ID
                    ]

                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    logger.error(
                        "req %s: local blocks(%d) < remote blocks(%d) "
                        "in a KV cache group (is_mamba_group=%s)",
                        d_req_id,
                        n_local,
                        n_remote,
                        is_mamba_group,
                    )
                    has_block_error = True
                    break
                elif n_local > n_remote:
                    # Partial prefix cache hit: just read uncomputed blocks.
                    local_group = local_group[-n_remote:] if n_remote > 0 else []
                local_block_ids_by_group.append(local_group)
                remote_block_ids_by_group.append(remote_group)

            if has_block_error:
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "P num blocks less than D"
                continue

            if not any(local_block_ids_by_group):
                continue

            local_block_ids_by_group = self._logical_to_kernel_block_ids(
                local_block_ids_by_group
            )
            remote_block_ids_by_group = self._logical_to_kernel_block_ids(
                remote_block_ids_by_group
            )

            selected_region_blocks: list[
                tuple[TransferRegion, TransferRegion, list[int], list[int]]
            ] = []
            selected_block_count = 0
            num_groups = len(remote_block_ids_by_group)
            for local_region, remote_region in zip(local_regions, remote_regions):
                region_group_indices = _common_group_indices_for_regions(
                    local_region,
                    remote_region,
                    num_groups,
                )
                (
                    local_block_ids,
                    remote_block_ids,
                    select_err,
                ) = _select_region_block_ids(
                    local_block_ids_by_group,
                    remote_block_ids_by_group,
                    region_group_indices,
                )
                if select_err is not None:
                    logger.error(
                        "req %s: local blocks < remote blocks for KV groups %s",
                        d_req_id,
                        region_group_indices,
                    )
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = select_err
                    selected_region_blocks = []
                    break
                if not local_block_ids:
                    continue
                selected_block_count += len(local_block_ids)
                selected_region_blocks.append(
                    (local_region, remote_region, local_block_ids, remote_block_ids)
                )

            if not selected_region_blocks:
                continue

            logged_transfer_plan = False
            for (
                local_region,
                remote_region,
                local_block_ids,
                remote_block_ids,
            ) in selected_region_blocks:
                # Group by indices within this region's KV-cache group only.
                group_local_block_ids, group_remote_block_ids = (
                    group_concurrent_contiguous(local_block_ids, remote_block_ids)
                )
                (
                    should_transfer,
                    src_region_offset,
                    dst_region_offset,
                    transfer_len,
                ) = self._get_sender_transfer_plan(
                    local_kv_block_len=local_region.kv_block_len,
                    remote_kv_block_len=remote_region.kv_block_len,
                    remote_tp_rank=agent_meta.remote_tp_rank,
                    remote_tp_size=agent_meta.remote_tp_size,
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Destination transfer region exceeds remote KV block size."
                )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len * len(group_local_block_id))
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

                if not logged_transfer_plan:
                    logger.debug(
                        "Mooncake transfer plan for request %s: local_tp=%d "
                        "remote_tp=%d remote_tp_rank=%d local_block_len=%d "
                        "remote_block_len=%d src_offset=%d dst_offset=%d "
                        "transfer_len=%d coalesce=%s",
                        d_req_id,
                        self.tp_size,
                        agent_meta.remote_tp_size,
                        agent_meta.remote_tp_rank,
                        local_region.block_len,
                        remote_region.block_len,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                        can_coalesce,
                    )
                    logged_transfer_plan = True

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                selected_block_count,
                remote_session,
            )

        # Append C128 bank0 descriptors after KV descriptors.
        if (
            self._online_c128_state_transfer_enabled
            and agent_meta.c128_import_base_addr
        ):
            err_msg = self._append_online_c128_state_descriptors(
                ready_reqs,
                agent_meta,
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
                state_transfer_events,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg, state_transfer_events

    def _append_online_c128_state_descriptors(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        err_reqs: list[ReqId],
        err_msg: str | None,
        state_transfer_events: list["torch.cuda.Event"],
    ) -> str | None:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (  # noqa: E501
            build_online_c128_state_descriptors,
        )

        pool = self._c128_export_pool
        if pool is None:
            return err_msg
        num_layers = self._c128_num_layers
        row_bytes = self._c128_state_row_bytes

        # Use the same rank election as replicated KV: one state writer per D slot.
        if not self._producer_cache_is_replicated():
            err_msg = (
                "C128 online partial-state transfer requires a replicated online "
                "state (MLA latent). Got a non-replicated/head-sharded state, "
                "which cannot be split across producer ranks without racing the "
                "single D import slot. Disable VLLM_USE_ONLINE_C128_PD_TRANSFER "
                "or use a replicated (MLA) configuration."
            )
            for d_req_id, _ in ready_reqs:
                if agent_meta.c128_req_needs_partial.get(d_req_id, False) and (
                    d_req_id not in err_reqs
                ):
                    err_reqs.append(d_req_id)
            return err_msg
        tp_ratio = _get_tp_ratio(self.tp_size, agent_meta.remote_tp_size)
        # tp_ratio > 0 (incl. 1): one elected writer per D slot. tp_ratio < 0:
        # all P ranks write, each into its own paired D rank's distinct slot.
        sends_state = tp_ratio < 0 or (self.tp_rank % tp_ratio == 0)
        if not sends_state:
            return err_msg

        remote_layer_indices = agent_meta.c128_layer_indices or list(
            range(agent_meta.c128_num_layers)
        )
        remote_layer_pos = {
            layer_index: layer_pos
            for layer_pos, layer_index in enumerate(remote_layer_indices)
        }
        layer_pos_pairs: list[tuple[int, int]] = []
        missing_layer_indices: list[int] = []
        for src_layer_pos, layer_index in enumerate(self._c128_layer_indices):
            dst_layer_pos = remote_layer_pos.get(layer_index)
            if dst_layer_pos is None:
                missing_layer_indices.append(layer_index)
            else:
                layer_pos_pairs.append((src_layer_pos, dst_layer_pos))

        needs_any_partial = any(
            agent_meta.c128_req_needs_partial.get(d_req_id, False)
            for d_req_id, _ in ready_reqs
        )
        local_layer_index_set = set(self._c128_layer_indices)
        remote_layer_index_set = set(remote_layer_indices)
        if (
            needs_any_partial
            and missing_layer_indices
            and remote_layer_index_set < local_layer_index_set
        ):
            err_msg = (
                "C128 online partial-state transfer does not support a producer "
                "layer set that strictly contains the consumer layer set. "
                "This usually means prefill is running the full model while "
                "decode is pipeline-parallel and owns only a layer subset. "
                f"P(layer_indices={self._c128_layer_indices}) vs "
                f"D(layer_indices={remote_layer_indices}); "
                f"producer_only_layer_indices={missing_layer_indices}. "
                "Use matching PP topology for prefill/decode or disable "
                "VLLM_USE_ONLINE_C128_PD_TRANSFER."
            )
            for d_req_id, _ in ready_reqs:
                if agent_meta.c128_req_needs_partial.get(d_req_id, False) and (
                    d_req_id not in err_reqs
                ):
                    err_reqs.append(d_req_id)
            return err_msg

        expected_remote_slot_bytes = agent_meta.c128_num_layers * row_bytes
        if (
            agent_meta.c128_state_row_bytes != row_bytes
            or agent_meta.c128_import_slot_bytes != expected_remote_slot_bytes
            or missing_layer_indices
        ):
            err_msg = (
                "C128 online state descriptor mismatch between producer and consumer: "
                f"P(num_layers={num_layers}, layer_indices="
                f"{self._c128_layer_indices}, row_bytes={row_bytes}, "
                f"slot_bytes={num_layers * row_bytes}) vs "
                f"D(num_layers={agent_meta.c128_num_layers}, "
                f"layer_indices={remote_layer_indices}, "
                f"row_bytes={agent_meta.c128_state_row_bytes}, "
                f"slot_bytes={agent_meta.c128_import_slot_bytes}); "
                f"missing_layer_indices={missing_layer_indices}."
            )
            for d_req_id, _ in ready_reqs:
                if agent_meta.c128_req_needs_partial.get(d_req_id, False) and (
                    d_req_id not in err_reqs
                ):
                    err_reqs.append(d_req_id)
            return err_msg

        for d_req_id, send_meta in ready_reqs:
            needs_partial = agent_meta.c128_req_needs_partial.get(d_req_id, False)
            if not needs_partial:
                # 128-aligned resume: no partial carry; D resets bank0 locally.
                continue
            slot = send_meta.c128_export_slot
            import_slot = agent_meta.c128_req_import_slot.get(d_req_id)
            if slot is None or import_slot is None:
                # Do not let D restore from an unwritten state-transfer slot.
                if d_req_id not in err_reqs:
                    err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = (
                        "C128 online partial-state transfer missing export/import "
                        f"slot for request needing partial state ({d_req_id}: "
                        f"export_slot={slot}, import_slot={import_slot})"
                    )
                continue
            plan = build_online_c128_state_descriptors(
                export_pool=pool,
                export_slot=slot,
                remote_import_base_addr=agent_meta.c128_import_base_addr,
                remote_slot_bytes=agent_meta.c128_import_slot_bytes,
                remote_import_slot=import_slot,
                num_layers=num_layers,
                row_width_bytes=row_bytes,
                layer_pos_pairs=layer_pos_pairs,
            )
            src_ptrs.extend(plan.src_ptrs)
            dst_ptrs.extend(plan.dst_ptrs)
            lengths.extend(plan.lengths)
            # Defer the RDMA read until the snapshot copy for this request has
            # landed (the copy ran async on the runner stream; the sender reads
            # the export slot from an executor thread).
            event = self._c128_export_events.get(send_meta.p_req_id)
            if event is not None:
                state_transfer_events.append(event)
        return err_msg

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
        correct CUDA device.  CUDA device selection is thread-local, so
        without this, NVLink transfers fail for TP ranks > 0."""
        current_platform.set_device(self.device_id)

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        state_transfer_events: list["torch.cuda.Event"] | None = None,
    ) -> int:
        # State snapshots are async GPU copies; wait before RDMA reads.
        if state_transfer_events:
            for event in state_transfer_events:
                event.synchronize()

        total_bytes = sum(lengths)
        total_descs = len(src_ptrs)
        start_time = time.perf_counter()

        if not envs.VLLM_MOONCAKE_ENABLE_CHUNKED_TRANSFER:
            ret_value = self.engine.batch_transfer_sync_write(
                remote_session, src_ptrs, dst_ptrs, lengths
            )
            chunk_idx = 1
            chunk_start = 0
            chunk_end = total_descs
            chunk_bytes = total_bytes
        else:
            max_chunk_bytes = envs.VLLM_MOONCAKE_TRANSFER_CHUNK_SIZE_MB * 1024 * 1024
            if max_chunk_bytes <= 0:
                raise ValueError(
                    "VLLM_MOONCAKE_TRANSFER_CHUNK_SIZE_MB must be positive"
                )
            chunked_src_ptrs: list[int] = []
            chunked_dst_ptrs: list[int] = []
            chunked_lengths: list[int] = []
            for src_ptr, dst_ptr, length in zip(src_ptrs, dst_ptrs, lengths):
                offset = 0
                while offset < length:
                    segment_len = min(max_chunk_bytes, length - offset)
                    chunked_src_ptrs.append(src_ptr + offset)
                    chunked_dst_ptrs.append(dst_ptr + offset)
                    chunked_lengths.append(segment_len)
                    offset += segment_len

            src_ptrs = chunked_src_ptrs
            dst_ptrs = chunked_dst_ptrs
            lengths = chunked_lengths
            total_descs = len(src_ptrs)
            ret_value = 0
            chunk_start = 0
            chunk_idx = 0
            chunk_end = 0
            chunk_bytes = 0
            while chunk_start < total_descs:
                chunk_bytes = 0
                chunk_end = chunk_start
                while chunk_end < total_descs:
                    next_len = lengths[chunk_end]
                    if chunk_end > chunk_start and (
                        chunk_bytes + next_len > max_chunk_bytes
                    ):
                        break
                    chunk_bytes += next_len
                    chunk_end += 1

                ret_value = self.engine.batch_transfer_sync_write(
                    remote_session,
                    src_ptrs[chunk_start:chunk_end],
                    dst_ptrs[chunk_start:chunk_end],
                    lengths[chunk_start:chunk_end],
                )
                if ret_value != 0:
                    break
                chunk_start = chunk_end
                chunk_idx += 1

        duration = time.perf_counter() - start_time
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=total_bytes,
                num_descs=total_descs,
            )
        else:
            self.xfer_stats.record_failed_transfer()
            if envs.VLLM_MOONCAKE_ENABLE_CHUNKED_TRANSFER:
                logger.warning(
                    "Sending chunk to %s failed (ret=%s) after %s "
                    "(chunk=%d, chunk_descriptors=%d, chunk_bytes=%d, "
                    "total_descriptors=%d, total_bytes=%d)",
                    remote_session,
                    ret_value,
                    duration,
                    chunk_idx + 1,
                    chunk_end - chunk_start,
                    chunk_bytes,
                    total_descs,
                    total_bytes,
                )
            else:
                logger.warning(
                    "Sending to %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                    remote_session,
                    ret_value,
                    duration,
                    total_descs,
                    total_bytes,
                )
        return ret_value

    def _register_c128_online_state(self) -> None:
        """Register C128 state-transfer pools outside the KV block region."""
        from vllm.models.deepseek_v4.online_c128 import get_online_c128_states

        states = get_online_c128_states()
        if not states:
            logger.warning(
                "C128 online compression enabled but no online states "
                "registered; skipping state-transfer RDMA registration."
            )
            return
        self._c128_states = states
        row_width = states[0].row_width
        self._c128_state_row_width = row_width
        self._c128_state_row_bytes = row_width * states[0].state.element_size()
        self._c128_num_layers = len(states)
        self._c128_layer_indices = [state.layer_index for state in states]
        capacity = self.vllm_config.scheduler_config.max_num_seqs
        device = states[0].state.device

        # State-transfer pool is allocated after KV profiling; fail early if it
        # cannot fit.
        pool_bytes = (
            capacity
            * self._c128_num_layers
            * row_width
            * states[0].state.element_size()
        )
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            # Keep a 5% headroom so registration / fragmentation does not OOM.
            margin = int(free_bytes * 0.95)
            if pool_bytes > margin:
                raise RuntimeError(
                    "C128 PD state-transfer pool would not fit in free GPU "
                    "memory: need "
                    f"{pool_bytes} bytes (capacity={capacity}, "
                    f"num_layers={self._c128_num_layers}, row_width={row_width}, "
                    f"fp32), free={free_bytes} bytes (95% margin={margin}). "
                    "Reduce max_num_seqs or disable "
                    "VLLM_USE_ONLINE_C128_PD_TRANSFER."
                )

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (  # noqa: E501
            C128ExportSlotPool,
            C128ImportSlotPool,
        )

        if self.is_kv_consumer:
            # D: import staging pool (RDMA destination).
            self._c128_import_pool = C128ImportSlotPool(
                capacity=capacity,
                num_layers=self._c128_num_layers,
                row_width=row_width,
                device=device,
            )
            pool = self._c128_import_pool
            ret = self.engine.batch_register_memory(
                [pool.base_addr], [pool.total_bytes]
            )
            if ret != 0:
                raise RuntimeError("Mooncake C128 import pool registration (D) failed.")
            logger.info(
                "Registered C128 state-transfer IMPORT pool (D): capacity=%d "
                "num_layers=%d row_bytes=%d slot_bytes=%d total_bytes=%d",
                capacity,
                self._c128_num_layers,
                self._c128_state_row_bytes,
                pool.slot_bytes,
                pool.total_bytes,
            )
            return

        # P: export snapshot pool (RDMA source).
        self._c128_export_pool = C128ExportSlotPool(
            capacity=capacity,
            num_layers=self._c128_num_layers,
            row_width=row_width,
            device=device,
        )
        pool = self._c128_export_pool
        ret = self.engine.batch_register_memory([pool.base_addr], [pool.total_bytes])
        if ret != 0:
            raise RuntimeError("Mooncake C128 export pool registration (P) failed.")
        logger.info(
            "Registered C128 state-transfer EXPORT pool (P): capacity=%d "
            "num_layers=%d row_bytes=%d slot_bytes=%d total_bytes=%d",
            capacity,
            self._c128_num_layers,
            self._c128_state_row_bytes,
            pool.slot_bytes,
            pool.total_bytes,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""
        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        region_base_addresses: list[int] = []
        seen_storage_ptrs: set[int] = set()
        self.block_len_per_layer = []
        self.kv_block_len_per_layer = []
        self.registered_layer_names = []
        self.registered_layer_indices = []
        self.registered_group_indices = []
        self.registered_layer_aliases = []
        self.registered_layer_index_aliases = []
        self.registered_logical_group_indices = []
        self.registered_alias_group_indices = []
        overlay_key_to_region_idx: dict[tuple[int, int, int, int], int] = {}
        region_aliases_by_key: dict[
            tuple[int, int, int, int], list[str]
        ] = defaultdict(list)
        region_index_aliases_by_key: dict[
            tuple[int, int, int, int], list[int]
        ] = defaultdict(list)
        region_logical_groups_by_key: dict[
            tuple[int, int, int, int], list[int]
        ] = defaultdict(list)
        region_alias_groups_by_key: dict[
            tuple[int, int, int, int], list[list[int]]
        ] = defaultdict(list)
        speculative_config = self.vllm_config.speculative_config
        speculative_method = getattr(speculative_config, "method", None)
        is_mtp_speculative = speculative_method == "mtp" or (
            isinstance(speculative_method, str)
            and speculative_method.endswith("_mtp")
        )
        total_num_hidden_layers = self.model_config.get_total_num_hidden_layers()

        for layer_name, cache_or_caches in kv_caches.items():
            layer_index = extract_layer_index(layer_name)
            if is_mtp_speculative and layer_index >= total_num_hidden_layers:
                logger.debug(
                    "Skipping MTP speculative KV cache layer %s outside the "
                    "base model layer range [0, %d)",
                    layer_name,
                    total_num_hidden_layers,
                )
                continue
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s because no KV cache spec is present.",
                    layer_name,
                )
                continue
            if isinstance(layer_spec, MambaSpec):
                conv, _ = cache_or_caches
                cache_list = [conv]
            else:
                cache_list = self.transfer_topo.get_transfer_cache_regions(
                    cache_or_caches, layer_spec
                )

            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                block_len = cache.stride(0) * cache.element_size()

                if isinstance(layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_block_len = layer_spec.page_size_bytes
                elif self.transfer_topo.virtually_split_kv_in_blocks and not isinstance(
                    layer_spec, MambaSpec
                ):
                    kv_block_len = block_len // 2
                else:
                    kv_block_len = block_len
                storage = cache.untyped_storage()
                storage_addr = storage.data_ptr()
                if storage_addr not in seen_storage_ptrs:
                    seen_storage_ptrs.add(storage_addr)
                    kv_data_ptrs.append(storage_addr)
                    kv_data_lens.append(storage.nbytes())
                overlay_key = (
                    storage_addr,
                    base_addr,
                    block_len,
                    kv_block_len,
                )
                logical_groups = list(
                    self._layer_logical_group_indices.get(layer_name, [])
                )
                if layer_name not in region_aliases_by_key[overlay_key]:
                    region_aliases_by_key[overlay_key].append(layer_name)
                    region_index_aliases_by_key[overlay_key].append(layer_index)
                    region_alias_groups_by_key[overlay_key].append(logical_groups)
                for group_idx in logical_groups:
                    if group_idx not in region_logical_groups_by_key[overlay_key]:
                        region_logical_groups_by_key[overlay_key].append(group_idx)

                if overlay_key in overlay_key_to_region_idx:
                    region_idx = overlay_key_to_region_idx[overlay_key]
                    self.registered_layer_aliases[region_idx] = list(
                        region_aliases_by_key[overlay_key]
                    )
                    self.registered_layer_index_aliases[region_idx] = list(
                        region_index_aliases_by_key[overlay_key]
                    )
                    self.registered_logical_group_indices[region_idx] = list(
                        region_logical_groups_by_key[overlay_key]
                    )
                    self.registered_alias_group_indices[region_idx] = [
                        list(groups)
                        for groups in region_alias_groups_by_key[overlay_key]
                    ]
                    continue

                overlay_key_to_region_idx[overlay_key] = len(region_base_addresses)
                region_base_addresses.append(base_addr)
                self.block_len_per_layer.append(block_len)
                self.kv_block_len_per_layer.append(kv_block_len)
                self.registered_layer_names.append(layer_name)
                self.registered_layer_indices.append(layer_index)
                self.registered_group_indices.append(
                    self._layer_group_indices[layer_name]
                )
                self.registered_layer_aliases.append(
                    list(region_aliases_by_key[overlay_key])
                )
                self.registered_layer_index_aliases.append(
                    list(region_index_aliases_by_key[overlay_key])
                )
                self.registered_logical_group_indices.append(
                    list(region_logical_groups_by_key[overlay_key])
                )
                self.registered_alias_group_indices.append(
                    [list(groups) for groups in region_alias_groups_by_key[overlay_key]]
                )

        self.kv_caches_base_addr = region_base_addresses
        self.seen_base_addresses = kv_data_ptrs

        if not kv_data_ptrs:
            raise RuntimeError("No KV cache tensors were registered with Mooncake.")

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        # Online C128 state-transfer pools are independent from the KV block region.
        if self._online_c128_state_transfer_enabled:
            self._register_c128_online_state()
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered block_lens=%s kv_block_lens=%s",
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            send_meta = self.reqs_need_send.pop(transfer_id, None)
            if send_meta is not None:
                self._release_c128_export_slot(send_meta)

        return finished_sending_reqs

    def _release_c128_export_slot(self, send_meta: "SendBlockMeta") -> None:
        """Release the P-side C128 export slot bound to this request."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_export_pool is None
        ):
            return
        p_req_id = send_meta.p_req_id
        if p_req_id and p_req_id in self._c128_export_slots:
            self._c128_export_pool.release(p_req_id)
            self._c128_export_slots.pop(p_req_id, None)
            self._c128_export_events.pop(p_req_id, None)
        if p_req_id:
            self._c128_req_state_indices.pop(p_req_id, None)
        send_meta.c128_export_slot = None

    def _release_c128_export_by_req(self, req_id: str) -> None:
        """Release a P-side export slot for requests that will not send."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_export_pool is None
        ):
            return
        if req_id in self._c128_export_slots:
            self._c128_export_pool.release(req_id)
            self._c128_export_slots.pop(req_id, None)
            self._c128_export_events.pop(req_id, None)
        self._c128_req_state_indices.pop(req_id, None)

    def bind_c128_state_index(self, req_id: str, p_req_state_idx: int) -> None:
        """P side: track live request-state slots for lazy state snapshots."""
        if not self._online_c128_state_transfer_enabled:
            return
        self._c128_req_state_indices[req_id] = p_req_state_idx

    def snapshot_c128_state(self, req_id: str, p_req_state_idx: int) -> None:
        """P side: snapshot committed bank0 before the request slot is reused."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_export_pool is None
        ):
            return
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (  # noqa: E501
            snapshot_bank0_to_slot,
        )

        slot = self._c128_export_pool.acquire(req_id)
        snapshot_bank0_to_slot(
            self._c128_states, self._c128_export_pool, slot, p_req_state_idx
        )
        # Sender waits on this event before RDMA-reading the export slot.
        event = torch.cuda.Event()
        event.record()
        self._c128_export_events[req_id] = event
        self._c128_export_slots[req_id] = slot
        self._attach_c128_export_slot_to_pending_sends(req_id, slot)

    def _attach_c128_export_slot_to_pending_sends(self, req_id: str, slot: int) -> None:
        """Bind a freshly snapshotted C128 export slot to pending sends.

        Scheduler-side ``request_finished`` can publish block IDs before the
        worker has removed the request and snapshotted bank0.  When state
        transfer is enabled, defer sender readiness until this method attaches
        the slot.
        """
        if not self._online_c128_state_transfer_enabled:
            return
        for send_meta in self.reqs_need_send.values():
            if send_meta.p_req_id != req_id:
                continue
            send_meta.c128_export_slot = slot
            if send_meta.local_block_ids:
                send_meta.ready.set()

    def reserve_c128_import_slot(
        self, d_req_id: str, transfer_id: str, needs_partial: bool
    ) -> int | None:
        """D side: reserve the RDMA staging slot before request admission."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_import_pool is None
        ):
            return None
        slot = self._c128_import_pool.acquire(transfer_id, timeout=0.0)
        self._c128_import_slots[transfer_id] = (slot, needs_partial)
        self._c128_req_to_transfer[d_req_id] = transfer_id
        return slot

    def restore_c128_state(self, req_id: str, d_req_state_idx: int) -> None:
        """D side: copy staged partial state into live bank0, or reset it."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_import_pool is None
        ):
            return
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.online_c128_pd import (  # noqa: E501
            reset_bank0,
            restore_bank0_from_slot,
        )

        transfer_id = self._c128_req_to_transfer.pop(req_id, None)
        self._c128_active_pulls.pop(req_id, None)
        self._c128_pending_import_reqs.discard(req_id)
        self._c128_aborted_import_reqs.discard(req_id)
        entry = (
            self._c128_import_slots.pop(transfer_id, None)
            if transfer_id is not None
            else None
        )
        if entry is None:
            # No staging reserved for this request: reset bank0 to identity.
            reset_bank0(self._c128_states, d_req_state_idx)
            return
        slot, needs_partial = entry
        if needs_partial:
            restore_bank0_from_slot(
                self._c128_states, self._c128_import_pool, slot, d_req_state_idx
            )
            if self._c128_import_pool.buffer.is_cuda:
                torch.cuda.current_stream(
                    self._c128_import_pool.buffer.device
                ).synchronize()
        else:
            reset_bank0(self._c128_states, d_req_state_idx)
        self._c128_import_pool.release(transfer_id)

    def _c128_release_import_slot(self, pull_meta: "PullReqMeta") -> None:
        """Release a single reserved import staging slot (idempotent)."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_import_pool is None
        ):
            return
        self._c128_active_pulls.pop(pull_meta.d_req_id, None)
        self._c128_pending_import_reqs.discard(pull_meta.d_req_id)
        self._c128_aborted_import_reqs.discard(pull_meta.d_req_id)
        transfer_id = self._c128_req_to_transfer.pop(pull_meta.d_req_id, None)
        if transfer_id is None:
            transfer_id = pull_meta.transfer_id
        self._c128_import_slots.pop(transfer_id, None)
        self._c128_import_pool.release(transfer_id)

    def release_c128_import_by_req(self, req_ids: list[ReqId]) -> None:
        """Release import slots, deferring aborts until in-flight pulls quiesce."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_import_pool is None
        ):
            return
        for req_id in req_ids:
            pull_meta = self._c128_active_pulls.get(req_id)
            if pull_meta is not None and pull_meta.c128_pull_pending > 0:
                # Do not reuse a slot while a worker may still RDMA-write it.
                pull_meta.c128_abort_pending = True
                continue
            # No in-flight pull task: safe to release now.
            transfer_id = self._c128_req_to_transfer.pop(req_id, None)
            self._c128_active_pulls.pop(req_id, None)
            if transfer_id is None:
                if req_id in self._c128_pending_import_reqs:
                    # Abort arrived before slot reservation finished.
                    self._c128_aborted_import_reqs.add(req_id)
                continue
            self._c128_pending_import_reqs.discard(req_id)
            self._c128_aborted_import_reqs.discard(req_id)
            self._c128_import_slots.pop(transfer_id, None)
            self._c128_import_pool.release(transfer_id)

    def _c128_account_pull_tasks(
        self,
        pull_metas: dict[ReqId, "PullReqMeta"],
        *,
        failed: bool,
        req_ids: list[ReqId] | None = None,
    ) -> None:
        """Decrement pull-task refs; release failed/aborted slots at quiesce."""
        if (
            not self._online_c128_state_transfer_enabled
            or self._c128_import_pool is None
        ):
            return
        targets = req_ids if req_ids is not None else list(pull_metas.keys())
        for d_req_id in targets:
            pull_meta = pull_metas.get(d_req_id)
            if pull_meta is None:
                continue
            if failed:
                pull_meta.c128_pull_failed = True
            if pull_meta.c128_pull_pending > 0:
                pull_meta.c128_pull_pending -= 1
            if pull_meta.c128_pull_pending == 0 and (
                pull_meta.c128_pull_failed or pull_meta.c128_abort_pending
            ):
                if pull_meta.c128_pull_failed:
                    self._mark_pull_failed(pull_meta)
                self._c128_release_import_slot(pull_meta)

    def _account_failed_pull_tasks(
        self,
        pull_metas: dict[ReqId, "PullReqMeta"],
        req_ids: list[ReqId],
    ) -> None:
        """Account failed async pulls and notify scheduler when safe.

        C128 state transfer needs quiesce accounting before slots can be
        released. Non-C128 pulls also wait for all worker tasks to quiesce
        before the scheduler may release or reuse local KV blocks.
        """
        self._c128_account_pull_tasks(pull_metas, failed=True, req_ids=req_ids)
        if (
            self._online_c128_state_transfer_enabled
            and self._c128_import_pool is not None
        ):
            return
        for req_id in req_ids:
            pull_meta = pull_metas.get(req_id)
            if pull_meta is None:
                continue
            pull_meta.pull_failed = True
            if pull_meta.pull_tasks_count > 0:
                pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self._mark_pull_failed(pull_meta)

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def _mark_pull_failed(self, pull_meta: "PullReqMeta") -> None:
        """Report a failed async pull to the scheduler after RDMA quiesce."""
        invalid_blocks = {
            block_id for group in pull_meta.local_block_ids for block_id in group
        }
        if invalid_blocks:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(invalid_blocks)
        self.finished_recving_reqs.add(pull_meta.d_req_id)

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = set(self._invalid_block_ids)
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return transfer stats collected since the last call, or None
        if nothing has been recorded in this interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = list(pull_metas.keys())
        outstanding_req_ids: set[ReqId] = set(req_ids)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            registered_layer_names=self.registered_layer_names,
            registered_layer_indices=self.registered_layer_indices,
            registered_group_indices=self.registered_group_indices,
            registered_layer_aliases=self.registered_layer_aliases,
            registered_layer_index_aliases=self.registered_layer_index_aliases,
            registered_logical_group_indices=self.registered_logical_group_indices,
            registered_alias_group_indices=self.registered_alias_group_indices,
            c128_import_base_addr=(
                self._c128_import_pool.base_addr
                if self._c128_import_pool is not None
                else 0
            ),
            c128_import_slot_bytes=(
                self._c128_import_pool.slot_bytes
                if self._c128_import_pool is not None
                else 0
            ),
            c128_num_layers=self._c128_num_layers,
            c128_state_row_bytes=self._c128_state_row_bytes,
            c128_layer_indices=list(self._c128_layer_indices),
            c128_req_import_slot={
                req_id: pull_meta.c128_import_slot
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.c128_import_slot is not None
            },
            c128_req_needs_partial={
                req_id: pull_meta.c128_needs_partial
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.c128_import_slot is not None
            },
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        self.xfer_stats.record_failed_recv()
                        # Account only requests still pending on this worker.
                        self._account_failed_pull_tasks(
                            pull_metas,
                            req_ids=list(outstanding_req_ids),
                        )
                        return
                    accounted_req_ids = self.process_pulling_result(
                        response, pull_metas
                    )
                    outstanding_req_ids.difference_update(accounted_req_ids)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self.xfer_stats.record_failed_recv()
            self._account_failed_pull_tasks(
                pull_metas,
                req_ids=list(outstanding_req_ids),
            )
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ) -> set[ReqId]:
        accounted_req_ids: set[ReqId] = set()
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            if pull_meta.pull_tasks_count > 0:
                pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                if pull_meta.pull_failed:
                    self._mark_pull_failed(pull_meta)
                else:
                    self.finished_recving_reqs.add(pull_meta.d_req_id)
        # Success keeps the slot for admission unless another worker fails.
        if ok_reqs:
            self._c128_account_pull_tasks(pull_metas, failed=False, req_ids=ok_reqs)
            accounted_req_ids.update(ok_reqs)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            err_reqs = list(response.err_reqs)
            logger.error(
                "pulling kv_caches for %s failed: %s",
                err_reqs,
                response.err_msg,
            )
            # Failed requests release slots only after all worker tasks quiesce.
            self._account_failed_pull_tasks(pull_metas, req_ids=err_reqs)
            accounted_req_ids.update(err_reqs)
        return accounted_req_ids

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        aborted_before_reserve: set[ReqId] = set()
        failed_before_start: list[ReqId] = []
        # Reserve D-side state-transfer slots before advertising metadata to P.
        if (
            self._online_c128_state_transfer_enabled
            and self._c128_import_pool is not None
        ):
            for d_req_id, pull_meta in pull_metas.items():
                self._c128_pending_import_reqs.discard(d_req_id)
                if d_req_id in self._c128_aborted_import_reqs:
                    aborted_before_reserve.add(d_req_id)
                    self._c128_aborted_import_reqs.discard(d_req_id)
                if pull_meta.c128_needs_partial and pull_meta.c128_import_slot is None:
                    try:
                        pull_meta.c128_import_slot = self.reserve_c128_import_slot(
                            d_req_id,
                            pull_meta.transfer_id,
                            pull_meta.c128_needs_partial,
                        )
                    except TimeoutError as exc:
                        logger.error(
                            "C128 import slot unavailable for request %s: %s",
                            d_req_id,
                            exc,
                        )
                        self._mark_pull_failed(pull_meta)
                        failed_before_start.append(d_req_id)
            for d_req_id in failed_before_start:
                pull_metas.pop(d_req_id, None)
                self._c128_pending_import_reqs.discard(d_req_id)
                self._c128_aborted_import_reqs.discard(d_req_id)
            if not pull_metas:
                return

        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )
        worker_addrs: list[str] = []
        selected_remote_pp: dict[int, list[int]] = {}
        for remote_tp_rank in remote_tp_ranks:
            pp_to_addr = self._remote_agents[remote_engine_id][remote_tp_rank]
            if self.pp_size == len(pp_to_addr) and self.pp_rank in pp_to_addr:
                pp_ranks = [self.pp_rank]
            else:
                pp_ranks = sorted(pp_to_addr)
            selected_remote_pp[remote_tp_rank] = pp_ranks
            worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)

        count = len(worker_addrs)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s "
            "and PP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
            selected_remote_pp,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
            pull_meta.pull_failed = False
            # Separate success count from C128 slot quiesce accounting.
            pull_meta.c128_pull_pending = count
            pull_meta.c128_pull_failed = False
            pull_meta.c128_abort_pending = pull_meta.d_req_id in aborted_before_reserve
            # Track the live pull so an abort (which only knows req_id) can defer
            # the import-slot release until all pull tasks quiesce.
            if (
                self._online_c128_state_transfer_enabled
                and self._c128_import_pool is not None
                and pull_meta.c128_import_slot is not None
            ):
                self._c128_active_pulls[pull_meta.d_req_id] = pull_meta
        for worker_addr in worker_addrs:
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            for req_id in pull_metas:
                self._c128_pending_import_reqs.discard(req_id)
                self._c128_aborted_import_reqs.discard(req_id)
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if (
                self._online_c128_state_transfer_enabled
                and self._c128_import_pool is not None
            ):
                self._c128_pending_import_reqs.update(pull_metas)
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv_and_release_c128_imports(
        self,
        reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]],
        c128_release_req_ids: list[ReqId],
    ):
        if c128_release_req_ids:
            self.release_c128_import_by_req(c128_release_req_ids)
        if reqs_to_recv:
            await self._start_load_kv(reqs_to_recv)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                # Attach the req-keyed export slot to this transfer.
                if self._online_c128_state_transfer_enabled:
                    slot = self._c128_export_slots.get(p_req_id)
                    if slot is None:
                        req_state_idx = self._c128_req_state_indices.get(p_req_id)
                        if req_state_idx is not None:
                            self.snapshot_c128_state(p_req_id, req_state_idx)
                            slot = self._c128_export_slots.get(p_req_id)
                        else:
                            logger.warning(
                                "C128 online state transfer has no live state "
                                "index for request %s transfer_id=%s "
                                "known_state_indices=%s",
                                p_req_id,
                                transfer_id,
                                list(self._c128_req_state_indices),
                            )
                    if slot is not None:
                        send_meta.c128_export_slot = slot
                        send_meta.ready.set()
                    else:
                        logger.warning(
                            "Deferring Mooncake send readiness for request %s "
                            "transfer_id=%s until C128 export slot is "
                            "snapshotted. known_export_slots=%s",
                            p_req_id,
                            transfer_id,
                            list(self._c128_export_slots),
                        )
                else:
                    send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()
                self._release_c128_export_slot(send_meta)
        # Release export slots snapshotted for producer requests that will not be
        # sent (purely local / aborted / no-block). Keyed by req_id; idempotent.
        for req_id in metadata.c128_export_release_req_ids:
            self._release_c128_export_by_req(req_id)

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        c128_release_req_ids = list(metadata.c128_import_release_req_ids)
        if not self.is_kv_producer and (metadata.reqs_to_recv or c128_release_req_ids):
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv_and_release_c128_imports(
                    metadata.reqs_to_recv, c128_release_req_ids
                ),
                self.receiver_loop,
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send
            or metadata.reqs_not_processed
            or metadata.c128_export_release_req_ids
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache

    def _get_transfer_regions(
        self,
        base_addrs: list[int],
        block_lens: list[int],
        kv_block_lens: list[int],
        layer_names: list[str],
        layer_indices: list[int],
        group_indices: list[int] | None = None,
        layer_aliases: list[list[str]] | None = None,
        layer_index_aliases: list[list[int]] | None = None,
        logical_group_indices: list[list[int]] | None = None,
        alias_group_indices: list[list[list[int]]] | None = None,
    ) -> list[TransferRegion]:
        if not group_indices:
            group_indices = [
                self._layer_group_indices.get(layer_name, 0)
                for layer_name in layer_names
            ]
        split_kv_regions = None
        if self.transfer_topo.virtually_split_kv_in_blocks:
            split_kv_regions = [
                not isinstance(
                    self._layer_specs[layer_name],
                    (MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec),
                )
                for layer_name in layer_names
            ]
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            kv_block_lens=kv_block_lens,
            layer_names=layer_names,
            layer_indices=layer_indices,
            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks,
            group_indices=group_indices,
            split_kv_regions=split_kv_regions,
            layer_aliases=layer_aliases,
            layer_index_aliases=layer_index_aliases,
            logical_group_indices=logical_group_indices,
            alias_group_indices=alias_group_indices,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # Only the TP=0, PP=0 worker of the designated engine should launch it.
    if get_tensor_model_parallel_rank() != 0:
        return False
    if get_pp_group().rank_in_group != 0:
        return False

    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    if parallel_config.local_engines_only:
        return parallel_config.data_parallel_rank_local == 0

    # In internal LB mode,
    # only the first data-parallel engine should launch the bootstrap server.
    return parallel_config.data_parallel_index == 0


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    elif parallel_config.nnodes_within_dp > 1:
        # Internal LB multi-node TP/PP uses the model-parallel master as the
        # single bootstrap endpoint for all ranks in the engine.
        host = parallel_config.master_addr
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
