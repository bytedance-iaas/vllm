# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import pytest
import torch
import zmq.asyncio

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeRegionIdentity,
    MooncakeRequestPageMap,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _compute_sender_transfer_plan,
    _get_full_temporal_suffix_blocks,
    _get_owned_dcp_suffix_blocks,
    _index_full_temporal_region_page_maps,
    _pair_cp_block_ids,
    _pair_dcp_blocks_by_global_page,
    _pair_full_temporal_blocks_by_global_page,
    _pair_pcp_block_ids,
    _validate_asymmetric_region_lengths,
    _validate_full_temporal_stage_coverage,
    _validate_region_layouts,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MOONCAKE_CP_BLOCK_PAIRING_VERSION,
    MOONCAKE_KV_REGION_LAYOUT_VERSION,
    MooncakeBootstrapServer,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheChildPageMapping,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTemporalLayout,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("local_tp_rank", "expected"),
    [
        (0, (True, 0, 0, 32768)),
        (1, (False, 0, 0, 0)),
        (2, (True, 0, 32768, 32768)),
        (3, (False, 0, 0, 0)),
        (4, (True, 0, 65536, 32768)),
        (5, (False, 0, 0, 0)),
        (6, (True, 0, 98304, 32768)),
        (7, (False, 0, 0, 0)),
    ],
)
def test_sender_plan_gqa_replicas_tp8_to_tp1(local_tp_rank, expected):
    assert (
        _compute_sender_transfer_plan(
            local_tp_rank=local_tp_rank,
            local_tp_size=8,
            remote_tp_rank=0,
            remote_tp_size=1,
            local_kv_block_len=32768,
            remote_kv_block_len=131072,
            producer_cache_replicated=True,
            transfer_unique_kv_heads=True,
            total_num_kv_heads=4,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("remote_tp_rank", "expected_src_offset"),
    [
        (0, 0),
        (1, 0),
        (2, 32768),
        (3, 32768),
        (4, 65536),
        (5, 65536),
        (6, 98304),
        (7, 98304),
    ],
)
def test_sender_plan_gqa_replicas_tp1_to_tp8(
    remote_tp_rank,
    expected_src_offset,
):
    assert _compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=1,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=131072,
        remote_kv_block_len=32768,
        producer_cache_replicated=False,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=4,
    ) == (True, expected_src_offset, 0, 32768)


def test_sender_plan_fully_replicated_region_is_unchanged():
    assert _compute_sender_transfer_plan(
        local_tp_rank=2,
        local_tp_size=8,
        remote_tp_rank=0,
        remote_tp_size=1,
        local_kv_block_len=4096,
        remote_kv_block_len=4096,
        producer_cache_replicated=True,
    ) == (False, 0, 0, 4096)


@pytest.mark.parametrize("remote_tp_rank", range(8))
def test_sender_plan_fully_replicated_tp1_to_tp8(remote_tp_rank):
    assert _compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=1,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=4096,
        remote_kv_block_len=4096,
        producer_cache_replicated=True,
    ) == (True, 0, 0, 4096)


@pytest.mark.parametrize(
    ("local_tp_rank", "remote_tp_rank", "should_transfer"),
    [(0, 0, True), (1, 0, False), (2, 1, True), (3, 1, False)],
)
def test_sender_plan_fully_replicated_tp4_to_tp2(
    local_tp_rank,
    remote_tp_rank,
    should_transfer,
):
    plan = _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=4,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=2,
        local_kv_block_len=4096,
        remote_kv_block_len=4096,
        producer_cache_replicated=True,
    )
    assert plan == ((True, 0, 0, 4096) if should_transfer else (False, 0, 0, 4096))


@pytest.mark.parametrize(
    ("producer_tp_rank", "consumer_tp_rank"),
    [(producer, consumer) for producer in range(4) for consumer in range(8)],
)
def test_sender_plan_maps_tp4_draft_heads_to_tp8_consumers(
    producer_tp_rank: int,
    consumer_tp_rank: int,
) -> None:
    bytes_per_head_page = 128 * 128 * 2
    plan = _compute_sender_transfer_plan(
        local_tp_rank=producer_tp_rank,
        local_tp_size=4,
        remote_tp_rank=consumer_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=16 * bytes_per_head_page,
        remote_kv_block_len=8 * bytes_per_head_page,
        producer_cache_replicated=False,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=64,
    )

    if producer_tp_rank != consumer_tp_rank // 2:
        assert plan == (False, 0, 0, 0)
        return

    expected_src_offset = (consumer_tp_rank % 2) * 8 * bytes_per_head_page
    assert plan == (
        True,
        expected_src_offset,
        0,
        8 * bytes_per_head_page,
    )


def test_validate_asymmetric_regions_allows_fully_replicated_index_cache():
    local_region = TransferRegion(
        layer_name="model.layers.0.indexer",
        layer_index=0,
        base_addr=0x1000,
        block_len=4096,
        kv_block_len=4096,
    )
    remote_region = TransferRegion(
        layer_name="model.layers.0.indexer",
        layer_index=0,
        base_addr=0x2000,
        block_len=4096,
        kv_block_len=4096,
    )
    assert (
        _validate_asymmetric_region_lengths(
            local_regions=[local_region],
            remote_regions=[remote_region],
            local_tp_size=1,
            remote_tp_size=8,
            producer_cache_replicated=False,
            fully_replicated_layers={"model.layers.0.indexer"},
        )
        is None
    )


@pytest.mark.parametrize("local_tp_rank", range(8))
def test_sender_plan_infers_per_region_head_count(local_tp_rank):
    assert _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=8,
        remote_tp_rank=0,
        remote_tp_size=1,
        local_kv_block_len=32768,
        remote_kv_block_len=8 * 32768,
        producer_cache_replicated=True,
        transfer_unique_kv_heads=True,
        # Simulate a model-level 4-head hint for an 8-head SWA region.
        total_num_kv_heads=4,
    ) == (True, 0, local_tp_rank * 32768, 32768)


def test_validate_asymmetric_regions_allows_replicated_consumer_heads():
    local_region = TransferRegion(
        layer_name="model.layers.0.self_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=131072,
        kv_block_len=131072,
    )
    remote_region = TransferRegion(
        layer_name="model.layers.0.self_attn",
        layer_index=0,
        base_addr=0x2000,
        block_len=32768,
        kv_block_len=32768,
    )
    assert (
        _validate_asymmetric_region_lengths(
            local_regions=[local_region],
            remote_regions=[remote_region],
            local_tp_size=1,
            remote_tp_size=8,
            producer_cache_replicated=False,
            unique_kv_head_layers={"model.layers.0.self_attn"},
            total_num_kv_heads_hint=4,
        )
        is None
    )


def test_sender_plan_virtual_split_preserves_head_offsets():
    assert _compute_sender_transfer_plan(
        local_tp_rank=4,
        local_tp_size=8,
        remote_tp_rank=0,
        remote_tp_size=1,
        local_kv_block_len=16384,
        remote_kv_block_len=65536,
        producer_cache_replicated=True,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=4,
    ) == (True, 0, 32768, 16384)


@pytest.mark.parametrize(
    ("local_tp_rank", "remote_tp_rank", "should_transfer"),
    [
        (0, 0, True),
        (1, 0, False),
        (2, 1, True),
        (3, 1, False),
    ],
)
def test_sender_plan_replicated_heads_tp4_to_tp2(
    local_tp_rank,
    remote_tp_rank,
    should_transfer,
):
    plan = _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=4,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=2,
        local_kv_block_len=32768,
        remote_kv_block_len=32768,
        producer_cache_replicated=True,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=1,
    )
    assert plan == ((True, 0, 0, 32768) if should_transfer else (False, 0, 0, 0))


@pytest.mark.parametrize(
    ("local_tp_rank", "remote_tp_rank"),
    [(0, 0), (0, 1), (1, 2), (1, 3)],
)
def test_sender_plan_replicated_heads_tp2_to_tp4(
    local_tp_rank,
    remote_tp_rank,
):
    assert _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=2,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=4,
        local_kv_block_len=32768,
        remote_kv_block_len=32768,
        producer_cache_replicated=True,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=1,
    ) == (True, 0, 0, 32768)


@pytest.mark.parametrize("local_tp_rank", range(16))
def test_sender_plan_mimo_equal_region_tp16_to_tp8(local_tp_rank):
    remote_tp_rank = local_tp_rank // 2
    assert _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=16,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=32768,
        remote_kv_block_len=32768,
        producer_cache_replicated=True,
        transfer_unique_kv_heads=True,
        total_num_kv_heads=8,
    ) == ((True, 0, 0, 32768) if local_tp_rank % 2 == 0 else (False, 0, 0, 0))


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [
                    "model.layers.0.self_attn",
                    "model.layers.1.self_attn",
                    "model.layers.0.mla_attn",
                    "model.layers.1.eagle_attn",
                ],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )


@pytest.mark.parametrize("group_block_size", [256, 64, 4, 8])
def test_pair_pcp_blocks_reassembles_v4_groups(group_block_size: int):
    """PCP ranks must write disjoint pages whose union is the D cache."""

    total_tokens = 1024
    remote_blocks = list(range(1000, 1000 + total_tokens // group_block_size))
    local_count = total_tokens // 2 // group_block_size

    rank_pairs = [
        _pair_pcp_block_ids(
            list(range(rank * 100, rank * 100 + local_count)),
            remote_blocks,
            total_tokens=total_tokens,
            num_external_tokens=total_tokens,
            external_start_token=0,
            producer_pcp_size=2,
            producer_pcp_rank=rank,
            consumer_pcp_size=1,
            consumer_pcp_rank=0,
            group_block_size=group_block_size,
            interleave_size=256,
        )
        for rank in range(2)
    ]

    assert all(error is None for _, _, error in rank_pairs)
    rank0_remote = rank_pairs[0][1]
    rank1_remote = rank_pairs[1][1]
    assert set(rank0_remote).isdisjoint(rank1_remote)
    assert sorted(rank0_remote + rank1_remote) == remote_blocks


def test_pair_pcp_blocks_skips_padding_after_partial_final_chunk():
    """Only the rank owning the request tail may write its final D page."""

    total_tokens = 600
    remote_blocks = [100, 101, 102]
    rank0 = _pair_pcp_block_ids(
        [10, 11],
        remote_blocks,
        total_tokens=total_tokens,
        num_external_tokens=total_tokens,
        external_start_token=0,
        producer_pcp_size=2,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=256,
        interleave_size=256,
    )
    rank1 = _pair_pcp_block_ids(
        [20],
        remote_blocks,
        total_tokens=total_tokens,
        num_external_tokens=total_tokens,
        external_start_token=0,
        producer_pcp_size=2,
        producer_pcp_rank=1,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=256,
        interleave_size=256,
    )

    assert rank0 == ([10, 11], [100, 102], None)
    assert rank1 == ([20], [101], None)


def test_pair_pcp_blocks_skips_padding_on_rank_without_tokens():
    """A short request may allocate one page on every PCP rank."""

    rank0 = _pair_pcp_block_ids(
        [10],
        [100],
        total_tokens=9,
        num_external_tokens=9,
        external_start_token=0,
        producer_pcp_size=2,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=256,
        interleave_size=256,
    )
    rank1 = _pair_pcp_block_ids(
        [20],
        [100],
        total_tokens=9,
        num_external_tokens=9,
        external_start_token=0,
        producer_pcp_size=2,
        producer_pcp_rank=1,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=256,
        interleave_size=256,
    )

    assert rank0 == ([10], [100], None)
    assert rank1 == ([], [], None)


@pytest.mark.parametrize("group_block_size", [256, 64, 8, 4])
def test_pair_pcp_blocks_maps_exact_external_suffix(group_block_size: int):
    """Prefix hits must not shift producer-local suffix pages on Decode."""

    total_tokens = 1024
    external_start_token = 512
    num_external_tokens = total_tokens - external_start_token
    remote_blocks = list(range(100, 100 + num_external_tokens // group_block_size))
    local_suffix_pages = num_external_tokens // 2 // group_block_size

    rank_pairs = [
        _pair_pcp_block_ids(
            list(range(rank * 1000, rank * 1000 + local_suffix_pages)),
            remote_blocks,
            total_tokens=total_tokens,
            num_external_tokens=num_external_tokens,
            external_start_token=external_start_token,
            producer_pcp_size=2,
            producer_pcp_rank=rank,
            consumer_pcp_size=1,
            consumer_pcp_rank=0,
            group_block_size=group_block_size,
            interleave_size=256,
        )
        for rank in range(2)
    ]

    assert all(error is None for _, _, error in rank_pairs)
    assert set(rank_pairs[0][1]).isdisjoint(rank_pairs[1][1])
    assert sorted(rank_pairs[0][1] + rank_pairs[1][1]) == remote_blocks


def test_pair_pcp_blocks_rejects_non_suffix_external_range():
    _, _, error = _pair_pcp_block_ids(
        [10],
        [100],
        total_tokens=1024,
        num_external_tokens=256,
        external_start_token=512,
        producer_pcp_size=2,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=256,
        interleave_size=256,
    )

    assert error is not None
    assert "exact suffix" in error


def test_pair_pcp_blocks_fails_closed_for_unsupported_page_geometry():
    _, _, error = _pair_pcp_block_ids(
        [10],
        [100],
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        producer_pcp_size=2,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        group_block_size=96,
        interleave_size=256,
    )

    assert error is not None
    assert "interleave" in error


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_pair_cp_blocks_scatters_global_pages_to_consumer_dcp(dcp_rank: int):
    local_blocks = list(range(28))
    remote_blocks = list(range(100 + 100 * dcp_rank, 114 + 100 * dcp_rank))
    remote_pages = list(range(dcp_rank, 28, 2))

    paired_local, paired_remote, error = _pair_dcp_blocks_by_global_page(
        local_blocks,
        remote_blocks,
        remote_pages,
        total_tokens=3563,
        num_external_tokens=3563,
        external_start_token=0,
        consumer_dcp_size=2,
        consumer_dcp_rank=dcp_rank,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert paired_local == list(range(dcp_rank, 28, 2))
    assert paired_remote == remote_blocks


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_pair_cp_blocks_maps_consumer_dcp_prefix_suffix(dcp_rank: int):
    paired_local, paired_remote, error = _pair_dcp_blocks_by_global_page(
        [4, 5, 6, 7],
        [100, 101],
        [4 + dcp_rank, 6 + dcp_rank],
        total_tokens=1024,
        num_external_tokens=512,
        external_start_token=512,
        consumer_dcp_size=2,
        consumer_dcp_rank=dcp_rank,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert paired_local == [4 + dcp_rank, 6 + dcp_rank]
    assert paired_remote == [100, 101]


def test_pair_cp_blocks_skips_consumer_dcp_rank_without_tokens():
    result = _pair_dcp_blocks_by_global_page(
        [0],
        [],
        [],
        total_tokens=9,
        num_external_tokens=9,
        external_start_token=0,
        consumer_dcp_size=2,
        consumer_dcp_rank=1,
        page_size=128,
        interleave_size=128,
    )

    assert result == ([], [], None)


@pytest.mark.parametrize(
    ("dcp_rank", "expected_blocks", "expected_pages"),
    [
        (0, list(range(100, 114)), list(range(0, 28, 2))),
        (1, list(range(200, 214)), list(range(1, 28, 2))),
    ],
)
def test_describe_consumer_dcp_blocks_with_global_pages(
    dcp_rank: int,
    expected_blocks: list[int],
    expected_pages: list[int],
):
    blocks, pages, error = _get_owned_dcp_suffix_blocks(
        expected_blocks,
        total_tokens=3563,
        num_external_tokens=3563,
        external_start_token=0,
        dcp_size=2,
        dcp_rank=dcp_rank,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert blocks == expected_blocks
    assert pages == expected_pages


@pytest.mark.parametrize(
    ("total_tokens", "external_start_token", "expected_page"),
    [(257, 0, 1), (257, 128, 1), (513, 256, 3)],
)
def test_describe_consumer_dcp_partial_tail_uses_valid_local_page(
    total_tokens: int,
    external_start_token: int,
    expected_page: int,
):
    blocks, pages, error = _get_owned_dcp_suffix_blocks(
        [100, 101],
        total_tokens=total_tokens,
        num_external_tokens=total_tokens - external_start_token,
        external_start_token=external_start_token,
        dcp_size=2,
        dcp_rank=1,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert blocks == [100]
    assert pages == [expected_page]


def test_describe_consumer_dcp_blocks_maps_clipped_suffix_tail():
    blocks, pages, error = _get_owned_dcp_suffix_blocks(
        [100, 101],
        total_tokens=1024,
        num_external_tokens=1024,
        external_start_token=0,
        dcp_size=2,
        dcp_rank=0,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert blocks == [100, 101]
    assert pages == [4, 6]


def test_describe_consumer_dcp_blocks_rejects_missing_owned_pages():
    blocks, pages, error = _get_owned_dcp_suffix_blocks(
        [],
        total_tokens=1024,
        num_external_tokens=1024,
        external_start_token=0,
        dcp_size=2,
        dcp_rank=0,
        page_size=128,
        interleave_size=128,
    )

    assert blocks == []
    assert pages == []
    assert error is not None
    assert "missing owned global pages" in error


@pytest.mark.parametrize(
    ("prompt_tokens", "parent_blocks", "expected_blocks", "expected_pages"),
    [
        (1, [100], [], []),
        (128, [100], [200], [0]),
        (129, [100], [200], [0]),
        (256, [100], [200, 201], [0, 1]),
        (257, [100, 101], [200, 201], [0, 1]),
    ],
)
def test_describe_full_temporal_draft_excludes_final_prompt_token(
    prompt_tokens: int,
    parent_blocks: list[int],
    expected_blocks: list[int],
    expected_pages: list[int],
):
    transferred_tokens = max(prompt_tokens - 1, 0)

    blocks, pages, error = _get_full_temporal_suffix_blocks(
        parent_blocks,
        total_tokens=transferred_tokens,
        num_external_tokens=transferred_tokens,
        external_start_token=0,
        child_page_factor=2,
        page_size=128,
    )

    assert error is None
    assert blocks == expected_blocks
    assert pages == expected_pages


def test_describe_full_temporal_factor1_preserves_absolute_suffix_pages():
    blocks, pages, error = _get_full_temporal_suffix_blocks(
        [10, 11, 12],
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        child_page_factor=1,
        page_size=128,
    )

    assert error is None
    assert blocks == [10, 11]
    assert pages == [0, 1]


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_external_tokens"),
    [(0, 0), (1, 0), (2, 1), (257, 256)],
)
def test_replicated_draft_remote_prefill_leaves_one_token_local(
    prompt_tokens: int,
    expected_external_tokens: int,
):
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler._has_full_temporal_draft = True
    scheduler._has_mamba = False

    assert (
        scheduler._get_remote_prefill_token_count(prompt_tokens)
        == expected_external_tokens
    )


@pytest.mark.parametrize(
    ("num_external_tokens", "expected_start"),
    [
        (512, 0),
        (256, 256),
        (0, 512),
    ],
)
def test_full_temporal_prefix_uses_atomic_target_draft_suffix(
    num_external_tokens: int,
    expected_start: int,
):
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler._has_full_temporal_draft = True
    scheduler._has_mamba = False

    total_tokens, external_start_token = scheduler._get_remote_prefill_transfer_range(
        num_prompt_tokens=513,
        num_external_tokens=num_external_tokens,
    )

    assert total_tokens == 512
    assert external_start_token == expected_start
    assert total_tokens - external_start_token == num_external_tokens


def test_pair_full_temporal_pages_uses_absolute_source_page_at_boundary():
    paired_local, paired_remote, error = _pair_full_temporal_blocks_by_global_page(
        [10, 11, 12],
        [200, 201],
        [0, 1],
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        page_size=128,
    )

    assert error is None
    assert paired_local == [10, 11]
    assert paired_remote == [200, 201]


@pytest.mark.parametrize(
    ("dcp_rank", "remote_pages", "expected_local"),
    [
        (0, [0], [10]),
        (1, [1], [11]),
    ],
)
def test_pair_dcp_pages_uses_absolute_source_page_at_recompute_boundary(
    dcp_rank: int,
    remote_pages: list[int],
    expected_local: list[int],
):
    paired_local, paired_remote, error = _pair_dcp_blocks_by_global_page(
        [10, 11, 12],
        [200 + dcp_rank],
        remote_pages,
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        consumer_dcp_size=2,
        consumer_dcp_rank=dcp_rank,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert paired_local == expected_local
    assert paired_remote == [200 + dcp_rank]


def test_index_full_temporal_region_page_maps_fails_closed():
    target = TransferRegion(
        layer_name="model.layers.0.self_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=128,
        kv_block_len=128,
        group_index=0,
        region_index=0,
        identity=MooncakeRegionIdentity(
            layer_name="model.layers.0.self_attn",
            temporal_layout=KVCacheTemporalLayout.SHARDED_DCP.value,
            protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
            child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
            child_page_factor=1,
        ),
    )
    draft = TransferRegion(
        layer_name="model.layers.60.self_attn.attn",
        layer_index=60,
        base_addr=0x2000,
        block_len=128,
        kv_block_len=128,
        group_index=0,
        region_index=1,
        identity=MooncakeRegionIdentity(
            layer_name="model.layers.60.self_attn.attn",
            temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
            protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
            child_page_mapping=(KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value),
            child_page_factor=2,
        ),
    )
    valid = MooncakeRequestPageMap(
        region_index=1,
        group_index=0,
        valid_start_token=0,
        valid_end_token_exclusive=256,
        global_page_ids=[0, 1],
        dst_physical_block_ids=[200, 201],
    )

    indexed, error = _index_full_temporal_region_page_maps(
        [valid],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is None
    assert indexed == {1: valid}

    _, error = _index_full_temporal_region_page_maps(
        [],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is not None
    assert "missing" in error

    _, error = _index_full_temporal_region_page_maps(
        [valid, valid],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is not None
    assert "duplicate" in error

    invalid_child = msgspec.structs.replace(
        valid,
        dst_physical_block_ids=[201, 200],
    )
    _, error = _index_full_temporal_region_page_maps(
        [invalid_child],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is not None
    assert "parity" in error

    duplicate_child = msgspec.structs.replace(
        valid,
        dst_physical_block_ids=[200, 200],
    )
    _, error = _index_full_temporal_region_page_maps(
        [duplicate_child],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is not None
    assert "unique" in error

    duplicate_page = msgspec.structs.replace(
        valid,
        global_page_ids=[0, 0],
    )
    _, error = _index_full_temporal_region_page_maps(
        [duplicate_page],
        [target, draft],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )
    assert error is not None
    assert "unique" in error


def test_index_full_temporal_factor1_page_map():
    layer_name = "model.layers.60.self_attn.attn"
    region = TransferRegion(
        layer_name=layer_name,
        layer_index=60,
        base_addr=0x2000,
        block_len=128,
        kv_block_len=128,
        group_index=0,
        region_index=0,
        identity=MooncakeRegionIdentity(
            layer_name=layer_name,
            temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
            protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
            child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
            child_page_factor=1,
        ),
    )
    page_map = MooncakeRequestPageMap(
        region_index=0,
        group_index=0,
        valid_start_token=0,
        valid_end_token_exclusive=256,
        global_page_ids=[0, 1],
        dst_physical_block_ids=[100, 101],
    )

    indexed, error = _index_full_temporal_region_page_maps(
        [page_map],
        [region],
        valid_start_token=0,
        valid_end_token_exclusive=256,
    )

    assert error is None
    assert indexed == {0: page_map}


def test_full_temporal_region_layout_rejects_protocol_downgrade():
    layer_name = "model.layers.60.self_attn.attn"
    producer = TransferRegion(
        layer_name=layer_name,
        layer_index=60,
        base_addr=0x1000,
        block_len=256,
        kv_block_len=128,
        identity=MooncakeRegionIdentity(
            layer_name=layer_name,
            temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
            protocol_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
            child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
            child_page_factor=1,
        ),
    )
    consumer = TransferRegion(
        layer_name=layer_name,
        layer_index=60,
        base_addr=0x2000,
        block_len=128,
        kv_block_len=64,
        identity=MooncakeRegionIdentity(
            layer_name=layer_name,
            temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
            protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
            child_page_mapping=(KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value),
            child_page_factor=2,
        ),
    )

    error = _validate_region_layouts(
        [producer],
        [consumer],
        producer_dcp_size=1,
        consumer_dcp_size=2,
    )

    assert error is not None
    assert "protocol" in error


def test_full_temporal_region_layout_accepts_dcp1_identity():
    layer_name = "model.layers.60.self_attn.attn"
    identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
        child_page_factor=1,
    )
    producer = TransferRegion(
        layer_name=layer_name,
        layer_index=60,
        base_addr=0x1000,
        block_len=128,
        kv_block_len=128,
        identity=identity,
    )
    consumer = TransferRegion(
        layer_name=layer_name,
        layer_index=60,
        base_addr=0x2000,
        block_len=128,
        kv_block_len=128,
        identity=identity,
    )

    assert (
        _validate_region_layouts(
            [producer],
            [consumer],
            producer_dcp_size=1,
            consumer_dcp_size=1,
        )
        is None
    )


@pytest.mark.parametrize(
    ("producer_layers", "pp_rank"),
    [
        (
            [
                "language_model.model.layers.0.self_attn.attn",
                "language_model.model.layers.1.self_attn.attn",
                "language_model.model.layers.2.self_attn.attn",
            ],
            0,
        ),
        (["model.layers.60.self_attn.attn"], 1),
    ],
    ids=["pp0-target-dense", "pp1-draft"],
)
def test_full_temporal_region_layout_allows_stage_local_subset(
    producer_layers: list[str],
    pp_rank: int,
) -> None:
    consumer_layers = [
        "language_model.model.layers.0.self_attn.attn",
        "language_model.model.layers.1.self_attn.attn",
        "language_model.model.layers.2.self_attn.attn",
        "model.layers.60.self_attn.attn",
    ]

    def make_region(layer_name: str, *, consumer: bool) -> TransferRegion:
        layer_index = int(layer_name.split("layers.", 1)[1].split(".", 1)[0])
        return TransferRegion(
            layer_name=layer_name,
            layer_index=layer_index,
            base_addr=0x1000 + layer_index * 0x100,
            block_len=128,
            kv_block_len=64,
            group_index=0,
            identity=MooncakeRegionIdentity(
                layer_name=layer_name,
                temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
                protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
                child_page_mapping=(
                    KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value
                    if consumer
                    else KVCacheChildPageMapping.IDENTITY.value
                ),
                child_page_factor=2 if consumer else 1,
            ),
        )

    producer_regions = [
        make_region(layer_name, consumer=False) for layer_name in producer_layers
    ]
    consumer_regions = [
        make_region(layer_name, consumer=True) for layer_name in consumer_layers
    ]

    aligned_producer, aligned_consumer, align_error = _align_transfer_regions(
        producer_regions, consumer_regions
    )

    assert align_error is None
    assert [region.layer_name for region in aligned_producer] == producer_layers
    assert [region.layer_name for region in aligned_consumer] == producer_layers
    assert (
        _validate_region_layouts(
            aligned_producer,
            aligned_consumer,
            producer_dcp_size=1,
            consumer_dcp_size=2,
        )
        is None
    )
    assert (
        _validate_full_temporal_stage_coverage(
            producer_regions,
            consumer_regions,
            speculative_config=SimpleNamespace(
                enable_eagle3_target_dense_full_temporal_kv=True,
                enable_eagle3_prefill_draft_kv=True,
            ),
            total_target_layers=60,
            pp_rank=pp_rank,
            pp_size=2,
        )
        is None
    )


@pytest.mark.parametrize(
    ("producer_layers", "pp_rank", "missing_layer"),
    [
        (
            [
                "language_model.model.layers.0.self_attn.attn",
                "language_model.model.layers.1.self_attn.attn",
            ],
            0,
            2,
        ),
        ([], 1, 60),
    ],
    ids=["pp0-missing-dense", "pp1-missing-draft"],
)
def test_full_temporal_stage_coverage_rejects_missing_layer(
    producer_layers: list[str],
    pp_rank: int,
    missing_layer: int,
) -> None:
    def make_region(layer_name: str) -> TransferRegion:
        layer_index = int(layer_name.split("layers.", 1)[1].split(".", 1)[0])
        return TransferRegion(
            layer_name=layer_name,
            layer_index=layer_index,
            base_addr=0x1000 + layer_index * 0x100,
            block_len=128,
            kv_block_len=64,
            identity=MooncakeRegionIdentity(
                layer_name=layer_name,
                temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
                protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
                child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
                child_page_factor=1,
            ),
        )

    producer_regions = [make_region(name) for name in producer_layers]
    consumer_regions = [
        make_region(
            f"language_model.model.layers.{layer}.self_attn.attn"
            if layer < 3
            else "model.layers.60.self_attn.attn"
        )
        for layer in (0, 1, 2, 60)
    ]

    error = _validate_full_temporal_stage_coverage(
        producer_regions,
        consumer_regions,
        speculative_config=SimpleNamespace(
            enable_eagle3_target_dense_full_temporal_kv=True,
            enable_eagle3_prefill_draft_kv=True,
        ),
        total_target_layers=60,
        pp_rank=pp_rank,
        pp_size=2,
    )

    assert error is not None
    assert "incomplete full-temporal coverage" in error
    assert str(missing_layer) in error


def test_pair_dcp_global_pages_maps_clipped_sliding_window_tail():
    paired_local, paired_remote, error = _pair_dcp_blocks_by_global_page(
        [20, 21, 22, 23],
        [100, 101],
        [4, 6],
        total_tokens=1024,
        num_external_tokens=1024,
        external_start_token=0,
        consumer_dcp_size=2,
        consumer_dcp_rank=0,
        page_size=128,
        interleave_size=128,
    )

    assert error is None
    assert paired_local == [20, 22]
    assert paired_remote == [100, 101]


@pytest.mark.parametrize(
    ("remote_pages", "match"),
    [
        ([0, 0], "unique"),
        ([2, 0], "increasing"),
        ([0, 3], "owner suffix"),
        ([0], "counts differ"),
    ],
)
def test_pair_dcp_global_pages_fails_closed(
    remote_pages: list[int],
    match: str,
):
    _, _, error = _pair_dcp_blocks_by_global_page(
        [10, 11, 12, 13],
        [100, 101],
        remote_pages,
        total_tokens=512,
        num_external_tokens=512,
        external_start_token=0,
        consumer_dcp_size=2,
        consumer_dcp_rank=0,
        page_size=128,
        interleave_size=128,
    )

    assert error is not None
    assert match in error


@pytest.mark.parametrize(
    (
        "producer_pcp_size",
        "producer_dcp_size",
        "consumer_pcp_size",
        "consumer_dcp_size",
        "consumer_interleave_size",
        "match",
    ),
    [
        (1, 2, 1, 1, 128, "producer DCP"),
        (1, 1, 2, 2, 128, "consumer PCP and DCP"),
        (2, 1, 1, 2, 128, "consumer DCP requires"),
        (1, 1, 1, 2, 96, "interleave"),
    ],
)
def test_pair_cp_blocks_rejects_unsupported_topologies(
    producer_pcp_size: int,
    producer_dcp_size: int,
    consumer_pcp_size: int,
    consumer_dcp_size: int,
    consumer_interleave_size: int,
    match: str,
):
    _, _, error = _pair_cp_block_ids(
        [0, 1],
        [100],
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        producer_pcp_size=producer_pcp_size,
        producer_pcp_rank=0,
        consumer_pcp_size=consumer_pcp_size,
        consumer_pcp_rank=0,
        producer_dcp_size=producer_dcp_size,
        producer_dcp_rank=0,
        consumer_dcp_size=consumer_dcp_size,
        consumer_dcp_rank=0,
        group_block_size=128,
        producer_interleave_size=128,
        consumer_interleave_size=consumer_interleave_size,
        consumer_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
    )

    assert error is not None
    assert match in error


def test_pair_cp_blocks_rejects_legacy_ambiguous_block_mismatch():
    _, _, error = _pair_cp_block_ids(
        list(range(28)),
        list(range(100, 114)),
        total_tokens=3563,
        num_external_tokens=3563,
        external_start_token=0,
        producer_pcp_size=1,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        producer_dcp_size=1,
        producer_dcp_rank=0,
        consumer_dcp_size=1,
        consumer_dcp_rank=0,
        group_block_size=128,
        producer_interleave_size=128,
        consumer_interleave_size=1,
        consumer_cp_block_pairing_version=0,
    )

    assert error is not None
    assert "CP block pairing capability" in error


def test_pair_cp_blocks_accepts_v1_non_dcp_pairing():
    local, remote, error = _pair_cp_block_ids(
        list(range(28)),
        list(range(100, 114)),
        total_tokens=3563,
        num_external_tokens=3563,
        external_start_token=0,
        producer_pcp_size=1,
        producer_pcp_rank=0,
        consumer_pcp_size=1,
        consumer_pcp_rank=0,
        producer_dcp_size=1,
        producer_dcp_rank=0,
        consumer_dcp_size=1,
        consumer_dcp_rank=0,
        group_block_size=128,
        producer_interleave_size=128,
        consumer_interleave_size=1,
        consumer_cp_block_pairing_version=1,
    )

    assert error is None
    assert local == list(range(14, 28))
    assert remote == list(range(100, 114))


@pytest.mark.asyncio
@pytest.mark.parametrize("group_block_size", [256, 64, 8, 4])
@pytest.mark.parametrize("pcp_rank", [0, 1])
async def test_build_transfer_params_reassembles_pcp_pages(
    group_block_size: int,
    pcp_rank: int,
):
    """The transfer plan must scatter compact PCP pages into D global order."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = False
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.pcp_rank = pcp_rank
    worker.pcp_size = 2
    worker.cp_kv_cache_interleave_size = 256
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn"],
                FullAttentionSpec(
                    block_size=group_block_size,
                    num_kv_heads=1,
                    head_size=16,
                    dtype=torch.float16,
                ),
            )
        ],
    )

    local_base = 0x1000
    remote_base = 0xA000
    region_block_len = 128
    local_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=local_base,
            block_len=region_block_len,
            kv_block_len=region_block_len,
            logical_group_indices=(0,),
        )
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=remote_base,
            block_len=region_block_len,
            kv_block_len=region_block_len,
            logical_group_indices=(0,),
        )
    ]
    local_pages_per_chunk = 256 // group_block_size
    local_block_ids = list(range(10, 10 + 512 // group_block_size))
    remote_block_ids = list(range(100, 100 + 1024 // group_block_size))
    transfer_id = "xfer-pcp"
    send_meta = SendBlockMeta(
        p_req_id="p-req-pcp",
        transfer_id=transfer_id,
        local_block_ids=[local_block_ids],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        remote_pcp_size=1,
        remote_pcp_rank=0,
        req_blocks={"d-req-pcp": (transfer_id, [remote_block_ids])},
        kv_caches_base_addr=[remote_base],
        block_lens=[region_block_len],
        kv_block_lens=[region_block_len],
        req_total_tokens={"d-req-pcp": 1024},
        req_num_external_tokens={"d-req-pcp": 1024},
        req_external_start_tokens={"d-req-pcp": 0},
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        ready_reqs=[("d-req-pcp", send_meta)],
        agent_meta=xfer_meta,
        local_regions=local_regions,
        remote_regions=remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [
        local_base + 10 * region_block_len,
        local_base + (10 + local_pages_per_chunk) * region_block_len,
    ]
    assert dst_ptrs == [
        remote_base + (100 + pcp_rank * local_pages_per_chunk) * region_block_len,
        remote_base + (100 + (2 + pcp_rank) * local_pages_per_chunk) * region_block_len,
    ]
    assert lengths == [region_block_len * local_pages_per_chunk] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_tp_rank", range(4))
@pytest.mark.parametrize("dcp_rank", [0, 1])
async def test_build_transfer_params_scatters_consumer_dcp_pages(
    producer_tp_rank: int,
    dcp_rank: int,
):
    layer_name = "model.layers.0.self_attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = False
    worker.is_kv_producer = True
    worker.tp_rank = producer_tp_rank
    worker.tp_size = 4
    worker.pcp_rank = 0
    worker.pcp_size = 1
    worker.dcp_rank = 0
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 128
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=4,
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer_name: SimpleNamespace(total_num_kv_heads=4)}
        )
    )
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [layer_name],
                layer_spec,
            )
        ],
    )

    block_len = 128
    local_base = 0x1000
    remote_base = 0xA000
    local_regions = [
        TransferRegion(
            layer_name=layer_name,
            layer_index=0,
            base_addr=local_base,
            block_len=block_len,
            kv_block_len=block_len,
            logical_group_indices=(0,),
        )
    ]
    remote_regions = [
        TransferRegion(
            layer_name=layer_name,
            layer_index=0,
            base_addr=remote_base,
            block_len=block_len,
            kv_block_len=block_len,
            logical_group_indices=(0,),
        )
    ]
    transfer_id = "xfer-dcp"
    send_meta = SendBlockMeta(
        p_req_id="p-req-dcp",
        transfer_id=transfer_id,
        local_block_ids=[list(range(28))],
        ready=asyncio.Event(),
    )
    remote_blocks = list(range(100, 114))
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=8,
        remote_tp_rank=producer_tp_rank * 2 + dcp_rank,
        remote_pcp_size=1,
        remote_pcp_rank=0,
        remote_dcp_size=2,
        remote_dcp_rank=dcp_rank,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        req_blocks={"d-req-dcp": (transfer_id, [remote_blocks])},
        kv_caches_base_addr=[remote_base],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        req_total_tokens={"d-req-dcp": 3563},
        req_num_external_tokens={"d-req-dcp": 3563},
        req_external_start_tokens={"d-req-dcp": 0},
        req_global_page_ids={"d-req-dcp": [list(range(dcp_rank, 28, 2))]},
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        ready_reqs=[("d-req-dcp", send_meta)],
        agent_meta=xfer_meta,
        local_regions=local_regions,
        remote_regions=remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [
        local_base + block_id * block_len for block_id in range(dcp_rank, 28, 2)
    ]
    assert dst_ptrs == [
        remote_base + block_id * block_len for block_id in remote_blocks
    ]
    assert lengths == [block_len] * 14


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consumer_dcp_size", "dcp_rank", "remote_tp_rank", "src_offset"),
    [
        (1, 0, 2, 0),
        (1, 0, 3, 128),
        (2, 0, 2, 0),
        (2, 1, 3, 128),
    ],
)
async def test_build_transfer_params_combines_full_temporal_pages_and_tp_slicing(
    consumer_dcp_size: int,
    dcp_rank: int,
    remote_tp_rank: int,
    src_offset: int,
):
    producer_tp_rank = 1
    layer_name = "model.layers.60.self_attn.attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=16,
        head_size=1,
        dtype=torch.float16,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.tp_rank = producer_tp_rank
    worker.tp_size = 4
    worker.pcp_rank = 0
    worker.pcp_size = 1
    worker.dcp_rank = 0
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 128
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=64,
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer_name: SimpleNamespace(total_num_kv_heads=64)}
        )
    )
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [layer_name],
                layer_spec,
            )
        ],
    )

    producer_identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
        child_page_factor=1,
    )
    consumer_identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=(
            KVCacheChildPageMapping.IDENTITY.value
            if consumer_dcp_size == 1
            else KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value
        ),
        child_page_factor=consumer_dcp_size,
    )
    local_block_len = 512
    remote_block_len = 256
    local_base = 0x1000
    remote_base = 0xA000
    local_regions = [
        TransferRegion(
            layer_name=layer_name,
            layer_index=60,
            base_addr=local_base,
            block_len=local_block_len,
            kv_block_len=256,
            group_index=0,
            logical_group_indices=(0,),
            region_index=0,
            identity=producer_identity,
        )
    ]
    remote_regions = [
        TransferRegion(
            layer_name=layer_name,
            layer_index=60,
            base_addr=remote_base,
            block_len=remote_block_len,
            kv_block_len=128,
            group_index=0,
            logical_group_indices=(0,),
            region_index=0,
            identity=consumer_identity,
        )
    ]
    send_meta = SendBlockMeta(
        p_req_id="p-req-draft",
        transfer_id="xfer-draft",
        # Producer computed 257 prompt tokens, while Decode requests [0, 256).
        local_block_ids=[[10, 11, 12]],
        ready=asyncio.Event(),
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=8,
        remote_tp_rank=remote_tp_rank,
        remote_dcp_size=consumer_dcp_size,
        remote_dcp_rank=dcp_rank,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        req_blocks={
            "d-req-draft": (
                "xfer-draft",
                [[300 + dcp_rank]],
            )
        },
        req_global_page_ids={"d-req-draft": [[dcp_rank]]},
        req_region_page_maps={
            "d-req-draft": [
                MooncakeRequestPageMap(
                    region_index=0,
                    group_index=0,
                    valid_start_token=0,
                    valid_end_token_exclusive=256,
                    global_page_ids=[0, 1],
                    dst_physical_block_ids=[200, 201],
                )
            ]
        },
        kv_caches_base_addr=[remote_base],
        block_lens=[remote_block_len],
        kv_block_lens=[128],
        req_total_tokens={"d-req-draft": 256},
        req_num_external_tokens={"d-req-draft": 256},
        req_external_start_tokens={"d-req-draft": 0},
    )

    src, dst, lengths, err_reqs, err_msg = await worker._build_transfer_params(
        ready_reqs=[("d-req-draft", send_meta)],
        agent_meta=metadata,
        local_regions=local_regions,
        remote_regions=remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src == [
        local_base + 10 * local_block_len + src_offset,
        local_base + 11 * local_block_len + src_offset,
    ]
    assert dst == [
        remote_base + 200 * remote_block_len,
        remote_base + 201 * remote_block_len,
    ]
    assert lengths == [128, 128]


@pytest.mark.asyncio
async def test_build_transfer_params_keeps_dcp_zero_transfer_handshake():
    layer_name = "model.layers.0.self_attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = False
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 4
    worker.pcp_rank = 0
    worker.pcp_size = 1
    worker.dcp_rank = 0
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 128
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=4,
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer_name: SimpleNamespace(total_num_kv_heads=4)}
        )
    )
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], layer_spec)],
    )

    send_meta = SendBlockMeta(
        p_req_id="p-req-dcp",
        transfer_id="xfer-dcp",
        local_block_ids=[[]],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=8,
        remote_tp_rank=0,
        remote_pcp_size=1,
        remote_pcp_rank=0,
        remote_dcp_size=2,
        remote_dcp_rank=0,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        req_blocks={"d-req-dcp": ("xfer-dcp", [[]])},
        req_global_page_ids={"d-req-dcp": [[]]},
        kv_caches_base_addr=[0xA000],
        block_lens=[128],
        kv_block_lens=[128],
        req_total_tokens={"d-req-dcp": 1024},
        req_num_external_tokens={"d-req-dcp": 0},
        req_external_start_tokens={"d-req-dcp": 1024},
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        ready_reqs=[("d-req-dcp", send_meta)],
        agent_meta=xfer_meta,
        local_regions=[],
        remote_regions=[],
    )

    assert src_ptrs == []
    assert dst_ptrs == []
    assert lengths == []
    assert err_reqs == []
    assert err_msg is None


@pytest.mark.asyncio
async def test_dcp_v2_missing_page_identities_fails_before_descriptors():
    layer_name = "model.layers.0.self_attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker._physical_blocks_per_logical_kv_block = 1
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], layer_spec)],
    )
    send_meta = SendBlockMeta(
        p_req_id="p-req",
        transfer_id="xfer",
        local_block_ids=[[0, 1]],
        ready=asyncio.Event(),
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=2,
        remote_tp_rank=0,
        remote_dcp_size=2,
        remote_dcp_rank=0,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        req_blocks={"d-req": ("xfer", [[100]])},
        kv_caches_base_addr=[0xA000],
        block_lens=[128],
        kv_block_lens=[128],
        req_total_tokens={"d-req": 256},
        req_num_external_tokens={"d-req": 256},
        req_external_start_tokens={"d-req": 0},
    )

    src, dst, lengths, err_reqs, err_msg = await worker._build_transfer_params(
        ready_reqs=[("d-req", send_meta)],
        agent_meta=metadata,
        local_regions=[],
        remote_regions=[],
    )

    assert src == dst == lengths == []
    assert err_reqs == ["d-req"]
    assert err_msg is not None
    assert "global page identities" in err_msg

    worker.pcp_size = 2
    metadata.req_global_page_ids = {"d-req": [[0]]}
    src, dst, lengths, err_reqs, err_msg = await worker._build_transfer_params(
        ready_reqs=[("d-req", send_meta)],
        agent_meta=metadata,
        local_regions=[],
        remote_regions=[],
    )

    assert src == dst == lengths == []
    assert err_reqs == ["d-req"]
    assert err_msg is not None
    assert "PCP1" in err_msg


class FakeMooncakeWrapper:
    """Mock Mooncake TransferEngine for unit testing environments."""

    def __init__(self, *args, **kwargs):
        self.initialize_calls = []

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        self.initialize_calls.append(
            (local_hostname, metadata_server, protocol, device_name)
        )
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        return 0


def test_align_transfer_regions_uses_layer_name_occurrences():
    """Repeated layer names should align by occurrence order."""

    local_regions = [
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1100,
            block_len=256,
            kv_block_len=128,
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=0xA000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB100,
            block_len=256,
            kv_block_len=128,
        ),
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert [r.base_addr for r in aligned_local] == [0x1000, 0x1100]
    assert [r.base_addr for r in aligned_remote] == [0xB000, 0xB100]


@pytest.mark.parametrize(
    ("local_count", "remote_count"),
    [(2, 1), (1, 2)],
    ids=["producer_has_more", "consumer_has_more"],
)
def test_align_transfer_regions_rejects_shared_name_occurrence_count_mismatch(
    local_count: int,
    remote_count: int,
):
    layer_name = "model.layers.1.self_attn"

    def make_regions(count: int, base_addr: int):
        return [
            TransferRegion(
                layer_name=layer_name,
                layer_index=1,
                base_addr=base_addr + occurrence * 0x100,
                block_len=256,
                kv_block_len=128,
            )
            for occurrence in range(count)
        ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        make_regions(local_count, 0x1000),
        make_regions(remote_count, 0xA000),
    )

    assert aligned_local == []
    assert aligned_remote == []
    assert err == (
        "Mooncake registered layer occurrence count mismatch for "
        f"{layer_name}: producer={local_count}, consumer={remote_count}."
    )


@pytest.mark.parametrize(
    ("local_layers", "remote_layers"),
    [
        ([0, 1], [1, 2]),
        ([1, 2], [0, 1]),
    ],
    ids=["producer_subset", "consumer_subset"],
)
def test_align_transfer_regions_uses_legacy_pp_intersection(
    local_layers: list[int],
    remote_layers: list[int],
):
    def make_regions(layer_indices: list[int], base_addr: int):
        return [
            TransferRegion(
                layer_name=f"model.layers.{layer_index}.self_attn",
                layer_index=layer_index,
                base_addr=base_addr + layer_index * 0x100,
                block_len=256,
                kv_block_len=256,
            )
            for layer_index in layer_indices
        ]

    local_regions = make_regions(local_layers, 0x1000)
    remote_regions = make_regions(remote_layers, 0xA000)

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert [region.layer_index for region in aligned_local] == [1]
    assert [region.layer_index for region in aligned_remote] == [1]


def test_align_transfer_regions_matches_shared_physical_region_aliases():
    local_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x1000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=(
                "model.layers.4.self_attn",
                "model.layers.4.swa_attn",
            ),
            layer_indices=(4, 4),
            logical_group_indices=(0, 1),
            alias_group_indices=((0,), (1,)),
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.4.swa_attn",
            layer_index=4,
            base_addr=0x2000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=(
                "model.layers.4.swa_attn",
                "model.layers.4.self_attn",
            ),
            layer_indices=(4, 4),
            logical_group_indices=(1, 0),
            alias_group_indices=((1,), (0,)),
        ),
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert aligned_local == local_regions
    assert aligned_remote == remote_regions


def test_align_transfer_regions_fans_out_shared_region_to_split_aliases():
    local_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x1000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=(
                "model.layers.4.self_attn",
                "model.layers.4.swa_attn",
            ),
            layer_indices=(4, 4),
            logical_group_indices=(0, 1),
            alias_group_indices=((0,), (1,)),
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x2000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=("model.layers.4.self_attn",),
            layer_indices=(4,),
            logical_group_indices=(0,),
            alias_group_indices=((0,),),
        ),
        TransferRegion(
            layer_name="model.layers.4.swa_attn",
            layer_index=4,
            base_addr=0x3000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=("model.layers.4.swa_attn",),
            layer_indices=(4,),
            logical_group_indices=(1,),
            alias_group_indices=((1,),),
        ),
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert aligned_local == [local_regions[0], local_regions[0]]
    assert aligned_remote == remote_regions


def test_align_transfer_regions_rejects_single_alias_occurrence_mismatch():
    local_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x1000,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=("model.layers.4.self_attn",),
            layer_indices=(4,),
            logical_group_indices=(0,),
            alias_group_indices=((0,),),
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=base_addr,
            block_len=4096,
            kv_block_len=4096,
            layer_aliases=("model.layers.4.self_attn",),
            layer_indices=(4,),
            logical_group_indices=(0,),
            alias_group_indices=((0,),),
        )
        for base_addr in (0x2000, 0x3000)
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert aligned_local == []
    assert aligned_remote == []
    assert err is not None
    assert "duplicate alias group match" in err


def test_xfer_metadata_decodes_legacy_payload_with_alias_defaults():
    payload = msgspec.msgpack.encode(
        {
            "remote_hostname": "consumer-host",
            "remote_port": 54321,
            "remote_tp_size": 1,
            "remote_tp_rank": 0,
            "req_blocks": {"d-req": ("xfer", [[1]])},
            "kv_caches_base_addr": [0x1000],
            "block_lens": [4096],
            "kv_block_lens": [4096],
            "registered_layer_names": ["model.layers.0.self_attn"],
            "registered_layer_indices": [0],
            "registered_group_indices": [0],
        }
    )

    metadata = msgspec.msgpack.decode(payload, type=MooncakeXferMetadata)

    assert metadata.registered_layer_names == ["model.layers.0.self_attn"]
    assert metadata.registered_layer_aliases == []
    assert metadata.registered_layer_index_aliases == []
    assert metadata.registered_logical_group_indices == []
    assert metadata.registered_alias_group_indices == []
    assert metadata.remote_dcp_size == 1
    assert metadata.remote_dcp_rank == 0
    assert metadata.remote_cp_kv_cache_interleave_size == 1
    assert metadata.remote_cp_block_pairing_version == 0
    assert metadata.req_region_page_maps == {}
    assert metadata.registered_region_identities == []


def test_xfer_metadata_round_trips_consumer_dcp_topology():
    metadata = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=8,
        remote_tp_rank=1,
        req_blocks={"d-req": ("xfer", [[1]])},
        kv_caches_base_addr=[0x1000],
        block_lens=[4096],
        kv_block_lens=[4096],
        remote_dcp_size=2,
        remote_dcp_rank=1,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        req_global_page_ids={"d-req": [[1, 3, 5]]},
    )

    payload = msgspec.msgpack.encode(metadata)
    decoded = msgspec.msgpack.decode(payload, type=MooncakeXferMetadata)

    assert decoded.remote_dcp_size == 2
    assert decoded.remote_dcp_rank == 1
    assert decoded.remote_cp_kv_cache_interleave_size == 128
    assert decoded.remote_cp_block_pairing_version == MOONCAKE_CP_BLOCK_PAIRING_VERSION
    assert decoded.req_global_page_ids == {"d-req": [[1, 3, 5]]}


def test_xfer_metadata_round_trips_full_temporal_region_pages():
    identity = MooncakeRegionIdentity(
        layer_name="model.layers.60.self_attn.attn",
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=(KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value),
        child_page_factor=2,
    )
    page_map = MooncakeRequestPageMap(
        region_index=0,
        group_index=0,
        valid_start_token=0,
        valid_end_token_exclusive=256,
        global_page_ids=[0, 1],
        dst_physical_block_ids=[200, 201],
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=8,
        remote_tp_rank=1,
        req_blocks={"d-req": ("xfer", [[100]])},
        kv_caches_base_addr=[0x1000],
        block_lens=[4096],
        kv_block_lens=[2048],
        remote_dcp_size=2,
        remote_dcp_rank=1,
        remote_cp_kv_cache_interleave_size=128,
        remote_cp_block_pairing_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        req_global_page_ids={"d-req": [[1]]},
        req_region_page_maps={"d-req": [page_map]},
        registered_layer_names=[identity.layer_name],
        registered_layer_indices=[60],
        registered_region_identities=[identity],
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata),
        type=MooncakeXferMetadata,
    )

    assert decoded.registered_region_identities == [identity]
    assert decoded.req_region_page_maps == {"d-req": [page_map]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dcp_rank", "expected_target_blocks", "expected_target_pages"),
    [
        (0, [100], [0]),
        (1, [100], [1]),
    ],
)
async def test_consumer_requests_owner_target_and_all_full_temporal_pages(
    dcp_rank: int,
    expected_target_blocks: list[int],
    expected_target_pages: list[int],
):
    layer_name = "model.layers.60.self_attn.attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )
    identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=(KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value),
        child_page_factor=2,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.hostname = "consumer-host"
    worker.rpc_port = 54321
    worker.tp_size = 8
    worker.tp_rank = dcp_rank
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dcp_size = 2
    worker.dcp_rank = dcp_rank
    worker.cp_kv_cache_interleave_size = 128
    worker.async_zmq_ctx = MagicMock()
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [layer_name],
                layer_spec,
            )
        ],
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.kv_caches_base_addr = [0x1000]
    worker.block_len_per_layer = [4096]
    worker.kv_block_len_per_layer = [4096]
    worker.registered_layer_names = [layer_name]
    worker.registered_layer_indices = [60]
    worker.registered_group_indices = [0]
    worker.registered_layer_aliases = [[layer_name]]
    worker.registered_layer_index_aliases = [[60]]
    worker.registered_logical_group_indices = [[0]]
    worker.registered_alias_group_indices = [[[0]]]
    worker.registered_region_identities = [identity]
    worker.process_pulling_result = MagicMock(return_value={"d-req"})

    pull_meta = PullReqMeta(
        d_req_id="d-req",
        transfer_id="xfer",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        pull_tasks_count=1,
    )
    response = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["d-req"],
    )
    socket = MagicMock(spec=zmq.asyncio.Socket)
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=worker._encoder.encode(response))
    socket_context = MagicMock()
    socket_context.__enter__.return_value = socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        return_value=socket_context,
    ):
        await worker.receive_kv_from_single_worker(
            "tcp://producer:1234",
            {"d-req": pull_meta},
        )

    sent_meta = worker._xfer_meta_decoder.decode(socket.send.await_args.args[0])
    assert sent_meta.remote_cp_block_pairing_version == (
        MOONCAKE_KV_REGION_LAYOUT_VERSION
    )
    assert sent_meta.req_blocks["d-req"] == (
        "xfer",
        [expected_target_blocks],
    )
    assert sent_meta.req_global_page_ids["d-req"] == [expected_target_pages]
    assert sent_meta.req_region_page_maps["d-req"] == [
        MooncakeRequestPageMap(
            region_index=0,
            group_index=0,
            valid_start_token=0,
            valid_end_token_exclusive=256,
            global_page_ids=[0, 1],
            dst_physical_block_ids=[200, 201],
        )
    ]
    assert sent_meta.registered_region_identities == [identity]


@pytest.mark.asyncio
async def test_dcp1_consumer_requests_full_temporal_identity_pages():
    layer_name = "model.layers.60.self_attn.attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )
    identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
        child_page_factor=1,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.hostname = "consumer-host"
    worker.rpc_port = 54321
    worker.tp_size = 8
    worker.tp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dcp_size = 1
    worker.dcp_rank = 0
    worker.cp_kv_cache_interleave_size = 1
    worker.async_zmq_ctx = MagicMock()
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], layer_spec)],
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.kv_caches_base_addr = [0x1000]
    worker.block_len_per_layer = [4096]
    worker.kv_block_len_per_layer = [4096]
    worker.registered_layer_names = [layer_name]
    worker.registered_layer_indices = [60]
    worker.registered_group_indices = [0]
    worker.registered_layer_aliases = [[layer_name]]
    worker.registered_layer_index_aliases = [[60]]
    worker.registered_logical_group_indices = [[0]]
    worker.registered_alias_group_indices = [[[0]]]
    worker.registered_region_identities = [identity]
    worker.process_pulling_result = MagicMock(return_value={"d-req"})

    pull_meta = PullReqMeta(
        d_req_id="d-req",
        transfer_id="xfer",
        local_block_ids=[[100, 101, 102]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        total_tokens=256,
        num_external_tokens=256,
        external_start_token=0,
        pull_tasks_count=1,
    )
    response = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["d-req"],
    )
    socket = MagicMock(spec=zmq.asyncio.Socket)
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=worker._encoder.encode(response))
    socket_context = MagicMock()
    socket_context.__enter__.return_value = socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        return_value=socket_context,
    ):
        await worker.receive_kv_from_single_worker(
            "tcp://producer:1234",
            {"d-req": pull_meta},
        )

    sent_meta = worker._xfer_meta_decoder.decode(socket.send.await_args.args[0])
    assert sent_meta.remote_cp_block_pairing_version == (
        MOONCAKE_KV_REGION_LAYOUT_VERSION
    )
    assert sent_meta.req_blocks["d-req"] == (
        "xfer",
        [[100, 101, 102]],
    )
    assert sent_meta.req_region_page_maps["d-req"] == [
        MooncakeRequestPageMap(
            region_index=0,
            group_index=0,
            valid_start_token=0,
            valid_end_token_exclusive=256,
            global_page_ids=[0, 1],
            dst_physical_block_ids=[100, 101],
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("dcp_size", [1, 2])
async def test_full_temporal_consumer_handles_abort_before_schedule(
    dcp_size: int,
):
    layer_name = "model.layers.60.self_attn.attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )
    identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=(
            KVCacheChildPageMapping.IDENTITY.value
            if dcp_size == 1
            else KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value
        ),
        child_page_factor=dcp_size,
    )
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.hostname = "consumer-host"
    worker.rpc_port = 54321
    worker.tp_size = 8
    worker.tp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dcp_size = dcp_size
    worker.dcp_rank = 0
    worker.cp_kv_cache_interleave_size = 1 if dcp_size == 1 else 128
    worker.async_zmq_ctx = MagicMock()
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], layer_spec)],
    )
    worker._layer_specs = {layer_name: layer_spec}
    worker.kv_caches_base_addr = [0x1000]
    worker.block_len_per_layer = [4096]
    worker.kv_block_len_per_layer = [4096]
    worker.registered_layer_names = [layer_name]
    worker.registered_layer_indices = [60]
    worker.registered_group_indices = [0]
    worker.registered_layer_aliases = [[layer_name]]
    worker.registered_layer_index_aliases = [[60]]
    worker.registered_logical_group_indices = [[0]]
    worker.registered_alias_group_indices = [[[0]]]
    worker.registered_region_identities = [identity]
    worker.finished_recving_reqs = set()

    pull_meta = PullReqMeta(
        d_req_id="d-req-aborted",
        transfer_id="xfer-aborted",
        local_block_ids=[],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        total_tokens=0,
        num_external_tokens=0,
        external_start_token=0,
        pull_tasks_count=1,
    )
    response = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=[pull_meta.d_req_id],
    )
    socket = MagicMock(spec=zmq.asyncio.Socket)
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=worker._encoder.encode(response))
    socket_context = MagicMock()
    socket_context.__enter__.return_value = socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        return_value=socket_context,
    ):
        await worker.receive_kv_from_single_worker(
            "tcp://producer:1234",
            {pull_meta.d_req_id: pull_meta},
        )

    sent_meta = worker._xfer_meta_decoder.decode(socket.send.await_args.args[0])
    assert sent_meta.req_blocks[pull_meta.d_req_id] == (
        pull_meta.transfer_id,
        [[]],
    )
    assert sent_meta.req_global_page_ids[pull_meta.d_req_id] == [[]]
    assert sent_meta.req_region_page_maps[pull_meta.d_req_id] == [
        MooncakeRequestPageMap(
            region_index=0,
            group_index=0,
            valid_start_token=0,
            valid_end_token_exclusive=0,
            global_page_ids=[],
            dst_physical_block_ids=[],
        )
    ]
    assert pull_meta.pull_tasks_count == 0
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("dcp_size", [1, 2])
async def test_full_temporal_abort_completes_producer_consumer_lifecycle(
    dcp_size: int,
):
    scheduler_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    scheduler = create_scheduler(scheduler_config)
    scheduler_connector = scheduler.get_kv_connector().connector_scheduler
    request = create_request(request_id=91, do_remote_prefill=True)
    assert request.kv_transfer_params is not None
    request.kv_transfer_params.update(
        {
            "transfer_id": "xfer-aborted-lifecycle",
            "remote_bootstrap_addr": "http://bootstrap:33333",
        }
    )
    request.status = RequestStatus.FINISHED_ABORTED

    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([],))
    connector_meta = scheduler_connector.build_connector_meta(MagicMock())
    pull_metas = connector_meta.reqs_to_recv["my-engine-id"]
    pull_meta = pull_metas[request.request_id]
    assert delay_free is False
    assert pull_meta.local_block_ids == []

    layer_name = "model.layers.60.self_attn.attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )
    producer_identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=KVCacheChildPageMapping.IDENTITY.value,
        child_page_factor=1,
    )
    consumer_identity = MooncakeRegionIdentity(
        layer_name=layer_name,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
        protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
        child_page_mapping=(
            KVCacheChildPageMapping.IDENTITY.value
            if dcp_size == 1
            else KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value
        ),
        child_page_factor=dcp_size,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], layer_spec)],
    )

    producer = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    producer.shutdown = MagicMock()
    producer.hostname = "producer-host"
    producer.tp_size = 8
    producer.tp_rank = 0
    producer.pp_size = 1
    producer.pp_rank = 0
    producer.pcp_size = 1
    producer.pcp_rank = 0
    producer.dcp_size = 1
    producer.dcp_rank = 0
    producer.cp_kv_cache_interleave_size = 1
    producer.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda _size: [0],
        virtually_split_kv_in_blocks=False,
        local_replicates_kv_cache=False,
        total_num_kv_heads=8,
    )
    producer.vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    producer.kv_cache_config = kv_cache_config
    producer._layer_specs = {layer_name: layer_spec}
    producer.kv_caches_base_addr = [0x1000]
    producer.block_len_per_layer = [4096]
    producer.kv_block_len_per_layer = [4096]
    producer.registered_layer_names = [layer_name]
    producer.registered_layer_indices = [60]
    producer.registered_group_indices = [0]
    producer.registered_layer_aliases = [[layer_name]]
    producer.registered_layer_index_aliases = [[60]]
    producer.registered_logical_group_indices = [[0]]
    producer.registered_alias_group_indices = [[[0]]]
    producer.registered_region_identities = [producer_identity]
    producer._physical_blocks_per_logical_kv_block = 1
    producer._encoder = msgspec.msgpack.Encoder()
    producer._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    producer._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    producer.reqs_need_send = {}
    producer.finished_sending_reqs = set()
    producer._send_blocks = MagicMock(return_value=0)

    send_meta = SendBlockMeta(
        p_req_id="p-req-aborted-lifecycle",
        transfer_id=pull_meta.transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    send_meta.ready.set()
    producer.reqs_need_send[send_meta.transfer_id] = send_meta

    consumer = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    consumer.shutdown = MagicMock()
    consumer.hostname = "consumer-host"
    consumer.rpc_port = 54321
    consumer.tp_size = 8
    consumer.tp_rank = 0
    consumer.pp_size = 1
    consumer.pp_rank = 0
    consumer.pcp_size = 1
    consumer.pcp_rank = 0
    consumer.dcp_size = dcp_size
    consumer.dcp_rank = 0
    consumer.cp_kv_cache_interleave_size = 1 if dcp_size == 1 else 128
    consumer.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda _size: [0],
        virtually_split_kv_in_blocks=False,
    )
    consumer.kv_cache_config = kv_cache_config
    consumer._layer_specs = {layer_name: layer_spec}
    consumer.kv_caches_base_addr = [0xA000]
    consumer.block_len_per_layer = [4096]
    consumer.kv_block_len_per_layer = [4096]
    consumer.registered_layer_names = [layer_name]
    consumer.registered_layer_indices = [60]
    consumer.registered_group_indices = [0]
    consumer.registered_layer_aliases = [[layer_name]]
    consumer.registered_layer_index_aliases = [[60]]
    consumer.registered_logical_group_indices = [[0]]
    consumer.registered_alias_group_indices = [[[0]]]
    consumer.registered_region_identities = [consumer_identity]
    consumer._encoder = msgspec.msgpack.Encoder()
    consumer._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    consumer._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    consumer.async_zmq_ctx = MagicMock()
    consumer._remote_agents = {"my-engine-id": {0: {0: {0: "tcp://producer:1234"}}}}
    consumer._tp_size = {"my-engine-id": 8}
    consumer._pcp_size = {"my-engine-id": 1}
    consumer._cp_block_pairing_version = {
        "my-engine-id": MOONCAKE_KV_REGION_LAYOUT_VERSION
    }
    consumer.finished_recving_reqs = set()

    producer_socket = AsyncMock(spec=zmq.asyncio.Socket)
    producer_socket.send_multipart = AsyncMock()
    consumer_socket = MagicMock(spec=zmq.asyncio.Socket)
    consumer_socket.setsockopt = MagicMock()
    consumer_socket.send = AsyncMock()
    producer_responses: list[MooncakeXferResponse] = []

    async def receive_producer_response():
        sent_metadata = producer._xfer_meta_decoder.decode(
            consumer_socket.send.await_args.args[0]
        )
        await producer.send_kv_to_decode(
            b"consumer-id",
            producer_socket,
            sent_metadata,
        )
        _, response = producer_socket.send_multipart.await_args.args[0]
        producer_responses.append(producer._xfer_resp_decoder.decode(response))
        return response

    consumer_socket.recv = AsyncMock(side_effect=receive_producer_response)
    socket_context = MagicMock()
    socket_context.__enter__.return_value = consumer_socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        return_value=socket_context,
    ):
        consumer.receive_kv("my-engine-id", pull_metas)
        for _ in range(100):
            if request.request_id in consumer.finished_recving_reqs:
                break
            await asyncio.sleep(0)

    producer._send_blocks.assert_not_called()
    assert len(producer_responses) == 1
    assert producer_responses[0].status == MooncakeXferResponseStatus.FINISH
    assert producer_responses[0].ok_reqs == [request.request_id]
    assert not producer_responses[0].err_reqs
    assert pull_meta.pull_failed is False
    assert pull_meta.pull_tasks_count == 0
    assert consumer.finished_recving_reqs == {request.request_id}
    assert send_meta.transfer_id not in producer.reqs_need_send
    assert producer.finished_sending_reqs == {send_meta.p_req_id}


@pytest.mark.asyncio
async def test_build_transfer_params_separates_prefill_pp_layers():
    """Each producer PP stage should send only its registered layer shard."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.kv_cache_config = _make_test_kv_cache_config()
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)

    block_len = 256
    remote_regions = [
        TransferRegion(
            layer_name=f"model.layers.{layer_index}.self_attn",
            layer_index=layer_index,
            base_addr=base_addr,
            block_len=block_len,
            kv_block_len=block_len,
        )
        for layer_index, base_addr in [
            (0, 0xA000),
            (1, 0xB000),
            (2, 0xC000),
            (3, 0xD000),
        ]
    ]
    producer_pp_regions = {
        0: [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.1.self_attn",
                layer_index=1,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
        1: [
            TransferRegion(
                layer_name="model.layers.2.self_attn",
                layer_index=2,
                base_addr=0x3000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.3.self_attn",
                layer_index=3,
                base_addr=0x4000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
    }
    expected_by_pp_rank = {
        0: {
            "layers": [0, 1],
            "src_ptrs": [0x1000 + 10 * block_len, 0x2000 + 10 * block_len],
            "dst_ptrs": [0xA000 + 20 * block_len, 0xB000 + 20 * block_len],
        },
        1: {
            "layers": [2, 3],
            "src_ptrs": [0x3000 + 10 * block_len, 0x4000 + 10 * block_len],
            "dst_ptrs": [0xC000 + 20 * block_len, 0xD000 + 20 * block_len],
        },
    }

    transfer_id = "xfer-pp-split"
    send_meta = SendBlockMeta(
        p_req_id="p-req-pp",
        transfer_id=transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-req-pp": (transfer_id, [[20, 21]])},
        kv_caches_base_addr=[region.base_addr for region in remote_regions],
        block_lens=[region.block_len for region in remote_regions],
        kv_block_lens=[region.kv_block_len for region in remote_regions],
        registered_layer_names=[region.layer_name for region in remote_regions],
        registered_layer_indices=[region.layer_index for region in remote_regions],
    )
    worker._layer_specs = {
        region.layer_name: worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        for region in remote_regions
    }

    for pp_rank, local_regions in producer_pp_regions.items():
        aligned_local, aligned_remote, err = _align_transfer_regions(
            local_regions, remote_regions
        )

        assert err is None
        assert [r.layer_index for r in aligned_local] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )
        assert [r.layer_index for r in aligned_remote] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )

        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs=[("d-req-pp", send_meta)],
            agent_meta=xfer_meta,
            local_regions=aligned_local,
            remote_regions=aligned_remote,
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == expected_by_pp_rank[pp_rank]["src_ptrs"]
        assert dst_ptrs == expected_by_pp_rank[pp_rank]["dst_ptrs"]
        assert lengths == [2 * block_len, 2 * block_len]


@pytest.mark.asyncio
async def test_send_kv_to_decode_aligns_consumer_regions_by_layer_metadata(
    monkeypatch,
):
    """Producer sends its PP layer shard to the matching consumer layer address."""

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        block_len = 4096
        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.1.self_attn"]
        prefill_worker.registered_layer_indices = [1]

        class InlineSenderLoop:
            async def run_in_executor(self, executor, func, *args):
                return func(*args)

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = InlineSenderLoop()

        transfer_id = "xfer-layer-align"
        send_meta = SendBlockMeta(
            p_req_id="p-req-layer-align",
            transfer_id=transfer_id,
            local_block_ids=[[10]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"d-req-layer-align": (transfer_id, [[20]])},
            kv_caches_base_addr=[0xA000, 0xB000],
            block_lens=[block_len, block_len],
            kv_block_lens=[block_len, block_len],
            registered_layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.self_attn",
            ],
            registered_layer_indices=[0, 1],
        )
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-layer-align"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

        src_ptrs, dst_ptrs, lengths = mock_send_blocks.call_args[0][1:]
        assert src_ptrs == [0x1000 + 10 * block_len]
        assert dst_ptrs == [0xB000 + 20 * block_len]
        assert lengths == [block_len]

        sent_identity, sent_payload = mock_socket.send_multipart.call_args[0][0]
        assert sent_identity == identity
        response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
        assert response.status == MooncakeXferResponseStatus.FINISH
        assert response.ok_reqs == ["d-req-layer-align"]

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


def test_basic_interface():
    """Unit test for basic MooncakeConnector interface functionality."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
    request_id = request.request_id
    request.kv_transfer_params.update(
        {
            "transfer_id": request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv["my-engine-id"]
    req_meta = kv_connector_metadata.reqs_to_recv["my-engine-id"][request_id]

    # local_block_ids is list[list[int]] (per-group); flatten for comparison.
    all_block_ids = [bid for group in req_meta.local_block_ids for bid in group]
    for block_id, block in zip(
        all_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """Test that we can handle case where prompt is < block."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    request.kv_transfer_params.update(
        {
            "transfer_id": request.request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv["my-engine-id"]) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


@pytest.fixture
def bootstrap_server():
    """Fixture to launch and cleanup a Mooncake Bootstrap HTTP Server."""

    port = get_open_port()
    server = MooncakeBootstrapServer("127.0.0.1", port)
    server.start()
    yield server
    server.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_server(bootstrap_server: MooncakeBootstrapServer):
    """
    Tests the bootstrap server's api for worker registration and querying.

    Validates DP/TP/PP rank indexing and error handling for duplicate registrations.
    """

    import httpx

    base_url = f"http://127.0.0.1:{bootstrap_server.port}"

    # Query when empty
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        assert response.json() == {}

    # Register multiple PP workers from the same producer engine.
    payload1 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "pcp_rank": 0,
        "pcp_size": 2,
        "cp_block_pairing_version": MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        "addr": "tcp://1.1.1.1:1111",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    payload1_pcp1 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "pcp_rank": 1,
        "pcp_size": 2,
        "cp_block_pairing_version": MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        "addr": "tcp://1.1.1.2:1112",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1_pcp1)
        assert response.status_code == 200

    payload2 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 1,
        "pcp_rank": 0,
        "pcp_size": 2,
        "cp_block_pairing_version": MOONCAKE_CP_BLOCK_PAIRING_VERSION,
        "addr": "tcp://2.2.2.2:2222",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload2)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Query after registration should preserve the PP dimension.
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        data = response.json()
        assert "0" in data
        assert data["0"]["engine_id"] == "eng-1"
        assert data["0"]["pcp_size"] == 2
        assert (
            data["0"]["cp_block_pairing_version"] == MOONCAKE_CP_BLOCK_PAIRING_VERSION
        )
        assert data["0"]["worker_addr"]["0"]["0"]["0"] == ("tcp://1.1.1.1:1111")
        assert data["0"]["worker_addr"]["0"]["0"]["1"] == ("tcp://1.1.1.2:1112")
        assert data["0"]["worker_addr"]["0"]["1"]["0"] == ("tcp://2.2.2.2:2222")

    payload_version_mismatch = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 1,
        "pp_rank": 0,
        "pcp_rank": 0,
        "pcp_size": 2,
        "cp_block_pairing_version": 0,
        "addr": "tcp://3.3.3.3:3333",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/register", json=payload_version_mismatch
        )
        assert response.status_code == 400
        assert "CP block pairing version mismatch" in response.text

    # Test failure: re-registering the same worker
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 400
        assert "is already registered" in response.text

    # Test failure: engine_id mismatch for same dp_rank
    payload3_fail = {
        "engine_id": "eng-2",
        "dp_rank": 0,
        "tp_rank": 1,
        "pp_rank": 0,
        "pcp_rank": 0,
        "pcp_size": 2,
        "addr": "tcp://3.3.3.3:3333",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload3_fail)
        assert response.status_code == 400
        assert "Engine ID mismatch" in response.text


def _make_bootstrap_vllm_config(
    *,
    local_engines_only: bool = False,
    data_parallel_rank_local: int = 0,
    data_parallel_index: int = 0,
    nnodes_within_dp: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_engines_only=local_engines_only,
            data_parallel_rank_local=data_parallel_rank_local,
            data_parallel_index=data_parallel_index,
            nnodes_within_dp=nnodes_within_dp,
            master_addr="model-parallel-master",
            data_parallel_master_ip="data-parallel-master",
        )
    )


@pytest.mark.parametrize(
    (
        "tp_rank",
        "pp_rank",
        "local_engines_only",
        "data_parallel_rank_local",
        "data_parallel_index",
        "expected",
    ),
    [
        (1, 0, False, 0, 0, False),
        (0, 1, False, 0, 0, False),
        (0, 0, True, 0, 1, True),
        (0, 0, True, 1, 0, False),
        (0, 0, False, 0, 0, True),
        (0, 0, False, 0, 1, False),
    ],
    ids=[
        "nonzero_tp_rank",
        "nonzero_pp_rank",
        "local_engine_rank_zero",
        "local_engine_nonzero_rank",
        "internal_lb_first_dp_engine",
        "internal_lb_nonzero_dp_engine",
    ],
)
def test_should_launch_bootstrap_server_selects_single_owner(
    tp_rank: int,
    pp_rank: int,
    local_engines_only: bool,
    data_parallel_rank_local: int,
    data_parallel_index: int,
    expected: bool,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        data_parallel_rank_local=data_parallel_rank_local,
        data_parallel_index=data_parallel_index,
    )
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_pp_group"
        ) as mock_pp_group,
    ):
        mock_pp_group.return_value.rank_in_group = pp_rank
        assert should_launch_bootstrap_server(vllm_config) is expected


def test_should_launch_bootstrap_server_rejects_nonzero_pcp_rank():
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=True,
        data_parallel_rank_local=0,
    )
    vllm_config.parallel_config.prefill_context_parallel_size = 2
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_pp_group"
        ) as mock_pp_group,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_pcp_group",
            create=True,
        ) as mock_pcp_group,
    ):
        mock_pp_group.return_value.rank_in_group = 0
        mock_pcp_group.return_value.rank_in_group = 1
        assert should_launch_bootstrap_server(vllm_config) is False


@pytest.mark.parametrize(
    ("local_engines_only", "nnodes_within_dp", "expected_host"),
    [
        (True, 2, "127.0.0.1"),
        (False, 2, "model-parallel-master"),
        (False, 1, "data-parallel-master"),
    ],
    ids=["local_engine", "multi_node_tp_or_pp", "single_node_internal_lb"],
)
def test_get_mooncake_bootstrap_addr_selects_expected_host(
    local_engines_only: bool,
    nnodes_within_dp: int,
    expected_host: str,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        nnodes_within_dp=nnodes_within_dp,
    )

    assert get_mooncake_bootstrap_addr(vllm_config) == (
        expected_host,
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT,
    )


def test_scheduler_request_finished():
    """
    Tests the scheduler-side logic when a request finishes.

    Differentiates between 'Finished' (requires transfer)
    and 'Aborted' (immediate free).
    """

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    scheduler = create_scheduler(vllm_config)
    scheduler_connector = scheduler.get_kv_connector().connector_scheduler
    assert scheduler_connector.kv_cache_config.kv_cache_groups

    request = create_request(request_id=1, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = request.request_id

    # Case: Capped length (Successful prefill, need to send to decoder)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([10, 11],))
    assert delay_free is True
    assert "id-1" in scheduler_connector._reqs_need_send
    assert scheduler_connector._reqs_need_send["id-1"][1] == [[10, 11]]

    # Case: Aborted (No need to transfer, free blocks immediately)
    scheduler_connector._reqs_need_send.clear()
    request.status = RequestStatus.FINISHED_ABORTED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([12],))
    assert delay_free is False
    assert len(scheduler_connector._reqs_need_send) == 0
    assert "id-1" in scheduler_connector._reqs_not_processed


@contextlib.contextmanager
def patch_worker_dependencies():
    """Helper to mock all distributed and network dependencies for Worker tests."""

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
            FakeMooncakeWrapper,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_pp_group"
        ) as mock_pp,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.should_launch_bootstrap_server",
            return_value=False,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.current_platform.set_device"
        ),
        patch("torch.accelerator.current_device_index", return_value=0),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_current_attn_backends",
            return_value=[FlashAttentionBackend],
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_kv_cache_layout",
            return_value="NHD",
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.make_zmq_socket"
        ) as mock_make_zmq,
        patch("httpx.AsyncClient") as mock_async_client,
    ):
        # Mock PP group
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = 0
        mock_pp.return_value = mock_pp_group

        # Mock ZMQ socket
        mock_socket_object = AsyncMock()
        mock_socket_object.setsockopt = MagicMock()
        mock_socket_ctx = MagicMock()
        mock_socket_ctx.__enter__.return_value = mock_socket_object
        mock_make_zmq.return_value = mock_socket_ctx

        # Mock httpx client
        mock_http_client_instance = AsyncMock()
        mock_async_client.return_value = mock_http_client_instance

        yield {
            "mock_make_zmq": mock_make_zmq,
            "mock_socket_object": mock_socket_object,
            "mock_async_client": mock_async_client,
            "mock_http_client": mock_http_client_instance,
        }


@pytest.mark.parametrize(
    ("extra_config", "expected_device"),
    [
        ({"device_name": "mlx5_2"}, "mlx5_2"),
        ({}, ""),
    ],
    ids=["extra_config_device_name", "default_empty_device"],
)
def test_worker_initializes_mooncake_with_configured_device(
    extra_config: dict[str, str],
    expected_device: str,
):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config=extra_config,
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )

    worker = connector.connector_worker
    assert worker.engine.initialize_calls == [
        ("127.0.0.1", "P2PHANDSHAKE", "rdma", expected_device)
    ]
    worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_pp_size", "local_pp_rank", "expected_addrs"),
    [
        (1, 0, ["tcp://producer-pp0:1234", "tcp://producer-pp1:1234"]),
        (2, 1, ["tcp://producer-pp1:1234"]),
    ],
    ids=["heterogeneous_pp_pulls_all_remote_pp", "matching_pp_pulls_same_rank"],
)
async def test_receive_kv_selects_remote_pp_workers(
    local_pp_size: int,
    local_pp_rank: int,
    expected_addrs: list[str],
):
    """Decode workers should not hard-code producer pp_rank 0."""

    decode_worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    decode_worker.shutdown = MagicMock()
    decode_worker.pp_size = local_pp_size
    decode_worker.pp_rank = local_pp_rank
    decode_worker.pcp_size = 1
    decode_worker.pcp_rank = 0
    decode_worker.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda _size: [0]
    )
    decode_worker._remote_agents = {
        "p-engine": {
            0: {
                0: {0: "tcp://producer-pp0:1234"},
                1: {0: "tcp://producer-pp1:1234"},
            }
        }
    }
    decode_worker._tp_size = {"p-engine": 1}
    decode_worker._pcp_size = {"p-engine": 1}

    pull_metas = {
        "d-req-1": PullReqMeta(
            d_req_id="d-req-1",
            transfer_id="xfer-req-1",
            local_block_ids=[[100, 101]],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://bootstrap:33333",
        )
    }
    seen_addrs: list[str] = []

    async def fake_receive(worker_addr: str, metas: dict[str, PullReqMeta]):
        seen_addrs.append(worker_addr)
        for meta in metas.values():
            meta.pull_tasks_count -= 1

    with patch.object(
        decode_worker,
        "receive_kv_from_single_worker",
        side_effect=fake_receive,
    ):
        decode_worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(0)

    assert seen_addrs == expected_addrs
    assert pull_metas["d-req-1"].pull_tasks_count == 0


def test_receive_kv_rejects_legacy_producer_for_consumer_dcp():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.dcp_size = 2
    worker._cp_block_pairing_version = {"p-engine": 0}
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=MagicMock())
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    worker.receive_kv_from_single_worker = MagicMock()
    pull_meta = PullReqMeta(
        d_req_id="d-req-old-producer",
        transfer_id="xfer-old-producer",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
    )

    worker.receive_kv("p-engine", {pull_meta.d_req_id: pull_meta})

    worker.transfer_topo.handshake_target_ranks.assert_not_called()
    worker.receive_kv_from_single_worker.assert_not_called()
    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_receive_kv_rejects_v2_producer_for_full_temporal_consumer(
    dcp_size: int,
):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.dcp_size = dcp_size
    worker.registered_region_identities = [
        MooncakeRegionIdentity(
            layer_name="model.layers.60.self_attn.attn",
            temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL.value,
            protocol_version=MOONCAKE_KV_REGION_LAYOUT_VERSION,
            child_page_mapping=(
                KVCacheChildPageMapping.IDENTITY.value
                if dcp_size == 1
                else KVCacheChildPageMapping.GLOBAL_PAGE_MODULO.value
            ),
            child_page_factor=dcp_size,
        )
    ]
    worker._cp_block_pairing_version = {"p-engine": MOONCAKE_CP_BLOCK_PAIRING_VERSION}
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=MagicMock())
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    worker.receive_kv_from_single_worker = MagicMock()
    pull_meta = PullReqMeta(
        d_req_id="d-req-v2-producer",
        transfer_id="xfer-v2-producer",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
    )

    worker.receive_kv("p-engine", {pull_meta.d_req_id: pull_meta})

    worker.transfer_topo.handshake_target_ranks.assert_not_called()
    worker.receive_kv_from_single_worker.assert_not_called()
    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


def test_receive_kv_rejects_pcp_producer_for_consumer_dcp():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.dcp_size = 2
    worker.pcp_size = 1
    worker.pp_size = 1
    worker.pp_rank = 0
    worker._cp_block_pairing_version = {"p-engine": MOONCAKE_CP_BLOCK_PAIRING_VERSION}
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda _size: [0])
    worker._remote_agents = {
        "p-engine": {
            0: {
                0: {
                    0: "tcp://producer-pcp0:1234",
                    1: "tcp://producer-pcp1:1234",
                }
            }
        }
    }
    worker._tp_size = {"p-engine": 1}
    worker._pcp_size = {"p-engine": 2}
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    worker.receive_kv_from_single_worker = MagicMock()
    pull_meta = PullReqMeta(
        d_req_id="d-req-pcp-producer",
        transfer_id="xfer-pcp-producer",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
    )

    worker.receive_kv("p-engine", {pull_meta.d_req_id: pull_meta})

    worker.receive_kv_from_single_worker.assert_not_called()
    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


def test_receive_kv_rejects_consumer_pp_fanout():
    """A producer PP stage cannot serve multiple consumer PP stages safely."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.pp_size = 2
    worker.pp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda _size: [0])
    worker._remote_agents = {"p-engine": {0: {0: {0: "tcp://producer-pp0:1234"}}}}
    worker._tp_size = {"p-engine": 1}
    worker._pcp_size = {"p-engine": 1}
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    worker.receive_kv_from_single_worker = MagicMock()
    pull_meta = PullReqMeta(
        d_req_id="d-req-pp-fanout",
        transfer_id="xfer-pp-fanout",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
    )

    worker.receive_kv("p-engine", {pull_meta.d_req_id: pull_meta})

    worker.receive_kv_from_single_worker.assert_not_called()
    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


@pytest.mark.asyncio
async def test_receive_kv_waits_for_every_remote_pp_pcp_worker():
    """A PCP1 decoder must pull every PP/PCP shard from a PCP2 producer."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.pp_size = 1
    worker.pp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda _size: [0])
    worker._remote_agents = {
        "p-engine": {
            0: {
                0: {
                    0: "tcp://producer-pp0-pcp0:1234",
                    1: "tcp://producer-pp0-pcp1:1234",
                },
                1: {
                    0: "tcp://producer-pp1-pcp0:1234",
                    1: "tcp://producer-pp1-pcp1:1234",
                },
            }
        }
    }
    worker._tp_size = {"p-engine": 1}
    worker._pcp_size = {"p-engine": 2}
    worker.finished_recving_reqs = set()
    pull_metas = {
        "d-req-1": PullReqMeta(
            d_req_id="d-req-1",
            transfer_id="xfer-req-1",
            local_block_ids=[[100, 101]],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://bootstrap:33333",
        )
    }
    seen_addrs: list[str] = []

    async def fake_receive(worker_addr: str, metas: dict[str, PullReqMeta]):
        seen_addrs.append(worker_addr)
        worker.process_pulling_result(
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=list(metas),
            ),
            metas,
        )

    with patch.object(
        worker,
        "receive_kv_from_single_worker",
        side_effect=fake_receive,
    ):
        worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(0)

    assert seen_addrs == [
        "tcp://producer-pp0-pcp0:1234",
        "tcp://producer-pp0-pcp1:1234",
        "tcp://producer-pp1-pcp0:1234",
        "tcp://producer-pp1-pcp1:1234",
    ]
    assert pull_metas["d-req-1"].pull_tasks_count == 0
    assert worker.finished_recving_reqs == {"d-req-1"}


def test_receive_kv_rejects_incomplete_remote_pcp_workers():
    """A decoder must fail closed when any producer PCP shard is absent."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.pp_size = 1
    worker.pp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda _size: [0])
    worker._remote_agents = {
        "p-engine": {
            0: {
                0: {0: "tcp://producer-pp0-pcp0:1234"},
                1: {0: "tcp://producer-pp1-pcp0:1234"},
            }
        }
    }
    worker._tp_size = {"p-engine": 1}
    worker._pcp_size = {"p-engine": 2}
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    pull_meta = PullReqMeta(
        d_req_id="d-req-incomplete",
        transfer_id="xfer-incomplete",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
    )

    worker.receive_kv("p-engine", {pull_meta.d_req_id: pull_meta})

    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


def test_multi_shard_late_failure_waits_for_quiesce_and_invalidates_blocks():
    """A late PCP shard failure must not expose partially reconstructed KV."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._invalid_block_ids = set()
    worker.finished_recving_reqs = set()
    pull_meta = PullReqMeta(
        d_req_id="d-req-late-failure",
        transfer_id="xfer-late-failure",
        local_block_ids=[[100, 101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        pull_tasks_count=2,
    )
    pull_metas = {pull_meta.d_req_id: pull_meta}

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.CONTINUE,
            ok_reqs=[pull_meta.d_req_id],
        ),
        pull_metas,
    )
    assert pull_meta.pull_tasks_count == 1
    assert worker.finished_recving_reqs == set()

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.ERROR,
            err_reqs=[pull_meta.d_req_id],
            err_msg="producer PCP shard failed",
        ),
        pull_metas,
    )
    assert pull_meta.pull_tasks_count == 0
    assert pull_meta.pull_failed is True
    assert worker.finished_recving_reqs == {pull_meta.d_req_id}
    assert worker.get_block_ids_with_load_errors() == {100, 101}


def test_resolve_need_send_accounts_for_remote_tp_fanout():
    """Producer-side completion waits for every paired consumer TP pull."""

    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    send_meta = SendBlockMeta(
        p_req_id="p-req-1",
        transfer_id="xfer-req-1",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
    )

    worker.resolve_need_send(send_meta, remote_tp_ranks=[0, 1])

    assert send_meta.need_send == 2


def test_finish_failed_send_attempts_release_after_all_targets():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.reqs_need_send = {}
    worker.finished_sending_reqs = set()
    send_meta = SendBlockMeta(
        p_req_id="p-req-failed",
        transfer_id="xfer-failed",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
        need_send=2,
        sending=2,
    )
    worker.reqs_need_send[send_meta.transfer_id] = send_meta

    worker._finish_send_attempt(send_meta)

    assert send_meta.sent == 1
    assert send_meta.sending == 1
    assert send_meta.transfer_id in worker.reqs_need_send
    assert worker.finished_sending_reqs == set()

    worker._finish_send_attempt(send_meta)

    assert send_meta.sent == 2
    assert send_meta.sending == 0
    assert send_meta.transfer_id not in worker.reqs_need_send
    assert worker.finished_sending_reqs == {"p-req-failed"}


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_kv_producer(monkeypatch):
    """
    Simulates a Producer Worker (Prefiller) receiving a transfer request
    from a Consumer (Decoder).

    Verifies memory offset calculation: ptr = base_addr + block_id * block_len.
    """

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker
        prefill_worker.kv_caches_base_addr = [0x1000]
        block_len = 4096
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        # Override loop to use current test loop
        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        # A request is finished on Producer and ready to be sent.
        transfer_id = "xfer-req-1"
        send_meta = SendBlockMeta(
            p_req_id="p-req-1",
            transfer_id=transfer_id,
            local_block_ids=[[10, 11]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Remote consumer request metadata
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            remote_cp_block_pairing_version=MOONCAKE_CP_BLOCK_PAIRING_VERSION,
            req_blocks={"d-req-1": (transfer_id, [[20, 21]])},
            kv_caches_base_addr=[0x2000],
            block_lens=[block_len],
            kv_block_lens=[block_len],
            registered_layer_names=["model.layers.0.self_attn"],
            registered_layer_indices=[0],
        )

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-id"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:

            def expected_transfers(src_base, dst_base, src_blocks, dst_blocks):
                n = len(src_blocks)
                return (
                    [src_base + src_blocks[0] * block_len],
                    [dst_base + dst_blocks[0] * block_len],
                    [n * block_len],
                )

            # Normal case: 2 blocks to 2 blocks
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            src, dst, lens = expected_transfers(0x1000, 0x2000, [10, 11], [20, 21])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Verify the response sent back to the consumer
            sent_call = mock_socket.send_multipart.call_args[0][0]
            sent_identity, sent_payload = sent_call
            assert sent_identity == identity
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.status == MooncakeXferResponseStatus.FINISH
            assert response.ok_reqs == ["d-req-1"]

            # Verify internal state cleanup
            assert transfer_id not in prefill_worker.reqs_need_send
            assert "p-req-1" in prefill_worker.finished_sending_reqs

            # More cases:
            # Consumer only needs 1 block (less than P)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # Verify transfer parameters are correct: 11 to 20
            src, dst, lens = expected_transfers(0x1000, 0x2000, [11], [20])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Consumer needs 3 blocks (more than P, error case)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21, 22]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because error.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "P num blocks less than D"
            assert response.err_reqs == ["d-req-1"]

            # Timeout
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.clear()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because timeout.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Timeout waiting for P side ready."
            assert response.err_reqs == ["d-req-1"]

        # Transfer error
        with patch.object(
            prefill_worker, "_send_blocks", return_value=123
        ) as mock_send_blocks:
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            mock_send_blocks.assert_called_once()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Mooncake transfer engine returned 123"
            assert response.err_reqs == ["d-req-1"]

        # Clean up
        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


@pytest.mark.asyncio
async def test_kv_consumuer(monkeypatch):
    """
    Simulates a Consumer Worker (Decoder) initiating a pull from a Producer.

    Verifies that MooncakeXferMetadata is correctly serialized and sent via ZMQ.
    """

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies() as mocks:
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.kv_caches_base_addr = [0x1000]
        decode_worker.block_len_per_layer = [4096]
        decode_worker.kv_block_len_per_layer = [4096]
        decode_worker.registered_layer_names = ["model.layers.0.self_attn"]
        decode_worker.registered_layer_indices = [0]
        decode_worker.rpc_port = 54321

        # A request to pull data arrives.
        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=1,
            )
        }
        decode_worker._remote_agents = {
            "p-engine": {0: {0: {0: "tcp://producer:1234"}}}
        }
        decode_worker._pcp_size["p-engine"] = 1
        decode_worker._tp_size["p-engine"] = 1

        # Mock the response from the producer.
        mock_response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH, ok_reqs=["d-req-1"]
        )
        encoded_response = decode_worker._encoder.encode(mock_response)
        mocks["mock_socket_object"].recv.return_value = encoded_response

        # Trigger the receive logic.
        decode_worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(1)  # Allow async task to run

        # Verify the metadata sent to the producer.
        mocks["mock_make_zmq"].assert_called_with(
            decode_worker.async_zmq_ctx,
            "tcp://producer:1234",
            zmq.DEALER,
            bind=False,
            linger=0,
        )
        sent_payload = mocks["mock_socket_object"].send.call_args[0][0]
        sent_meta = decode_worker._xfer_meta_decoder.decode(sent_payload)

        assert sent_meta.remote_hostname == "127.0.0.1"
        assert sent_meta.remote_port == 54321
        assert sent_meta.req_blocks["d-req-1"] == ("xfer-req-1", [[100, 101]])
        assert sent_meta.kv_caches_base_addr == [0x1000]
        assert sent_meta.block_lens == [4096]
        assert sent_meta.registered_layer_names == ["model.layers.0.self_attn"]
        assert sent_meta.registered_layer_indices == [0]

        # Verify internal state is updated correctly.
        assert "d-req-1" in decode_worker.finished_recving_reqs

        # Clean up
        decode_worker.shutdown()


@pytest.mark.asyncio
async def test_worker_get_finished_timeout(monkeypatch):
    """Tests the cleanup mechanism for requests."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Add an expired request (expire_time is in the past).
        prefill_worker.reqs_need_send["tx-expired"] = SendBlockMeta(
            p_req_id="p-req-expired",
            transfer_id="tx-expired",
            local_block_ids=[[1, 2]],
            ready=MagicMock(),
            expire_time=time.perf_counter() - 100,
        )

        # Add a non-expired request.
        prefill_worker.reqs_need_send["tx-active"] = SendBlockMeta(
            p_req_id="p-req-active",
            transfer_id="tx-active",
            local_block_ids=[[3, 4]],
            ready=MagicMock(),
            expire_time=time.perf_counter() + 100,
        )

        finished_reqs = await prefill_worker.fetch_finished_sending_reqs()

        assert "p-req-expired" in finished_reqs
        assert "p-req-active" not in finished_reqs
        assert "tx-expired" not in prefill_worker.reqs_need_send
        assert "tx-active" in prefill_worker.reqs_need_send


def test_register_kv_caches():
    """Tests the memory registration logic with the underlying Mooncake engine."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
        )
        tensor1 = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        tensor2 = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        kv_caches = {
            "model.layers.0.self_attn": tensor1,
            "model.layers.1.self_attn": tensor2,
        }

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

            mock_batch_register.assert_called_once()
            registered_ptrs, registered_lens = mock_batch_register.call_args[0]
            expected_ptrs = {tensor.data_ptr() for tensor in kv_caches.values()}
            assert set(registered_ptrs) == expected_ptrs
            assert set(registered_lens) == {tensor1.nbytes}

            # Verify block_len_per_layer is set correctly.
            assert len(worker.block_len_per_layer) == len(registered_ptrs)
            for bl in worker.block_len_per_layer:
                assert bl == tensor1.nbytes // tensor1.shape[0]
            assert worker.registered_layer_names == list(kv_caches)
            assert worker.registered_layer_indices == [0, 1]


def test_register_kv_caches_skips_mtp_layers_outside_base_model():
    num_hidden_layers = 32
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.use_mla = False
    worker.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: num_hidden_layers
    )
    worker.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp")
    )
    worker.kv_cache_config = _make_test_kv_cache_config()
    worker.transfer_topo = SimpleNamespace(
        virtually_split_kv_in_blocks=False,
        get_transfer_cache_regions=lambda cache, _spec: [cache],
    )
    worker.engine = MagicMock()
    worker.engine.batch_register_memory.return_value = 0
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = False
    worker.receiver_loop = MagicMock()
    worker.receiver_loop.is_running.return_value = False

    kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    normal_cache = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    mtp_cache = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    normal_layer = "model.layers.0.self_attn"
    kv_caches = {
        normal_layer: normal_cache,
        f"model.layers.{num_hidden_layers}.attn.swa_cache": mtp_cache,
    }
    worker._layer_specs = {
        normal_layer: FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
        )
    }
    worker._layer_group_indices = {normal_layer: 0}
    worker._layer_logical_group_indices = {normal_layer: [0]}

    worker.register_kv_caches(kv_caches)

    worker.engine.batch_register_memory.assert_called_once_with(
        [normal_cache.data_ptr()], [normal_cache.nbytes]
    )
    assert worker.registered_layer_names == [normal_layer]
    assert worker.registered_layer_indices == [0]


def test_register_kv_caches_keeps_non_mtp_layers_outside_base_model():
    num_hidden_layers = 32
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.use_mla = False
    worker.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: num_hidden_layers
    )
    worker.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="eagle")
    )
    worker.kv_cache_config = _make_test_kv_cache_config()
    worker.transfer_topo = SimpleNamespace(
        virtually_split_kv_in_blocks=False,
        get_transfer_cache_regions=lambda cache, _spec: [cache],
    )
    worker.engine = MagicMock()
    worker.engine.batch_register_memory.return_value = 0
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = False
    worker.receiver_loop = MagicMock()
    worker.receiver_loop.is_running.return_value = False

    kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    normal_cache = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    eagle_cache = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    kv_caches = {
        "model.layers.0.self_attn": normal_cache,
        f"model.layers.{num_hidden_layers}.attn.swa_cache": eagle_cache,
    }
    worker._layer_specs = {
        name: FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
        )
        for name in kv_caches
    }
    worker._layer_group_indices = {
        name: group_index for group_index, name in enumerate(kv_caches)
    }
    worker._layer_logical_group_indices = {
        name: [group_index] for group_index, name in enumerate(kv_caches)
    }

    worker.register_kv_caches(kv_caches)

    worker.engine.batch_register_memory.assert_called_once_with(
        [normal_cache.data_ptr(), eagle_cache.data_ptr()],
        [normal_cache.nbytes, eagle_cache.nbytes],
    )
    assert worker.registered_layer_names == list(kv_caches)
    assert worker.registered_layer_indices == [0, num_hidden_layers]


def test_pd_trace_lifecycle_clears_on_success_and_failure(monkeypatch):
    """Request-scoped trace timestamps must not survive terminal outcomes."""
    monkeypatch.setattr(envs, "VLLM_MOONCAKE_PD_TRACE", True)
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.finished_recving_reqs = set()
    worker.dp_rank = worker.pp_rank = worker.tp_rank = 0
    worker._pd_trace_pull_started = {
        "d-req-ok": time.perf_counter(),
        "d-req-failed": time.perf_counter(),
    }

    ok_meta = PullReqMeta(
        d_req_id="d-req-ok",
        transfer_id="xfer-ok",
        local_block_ids=[[100]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap",
        pull_tasks_count=1,
    )
    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=[ok_meta.d_req_id],
        ),
        {ok_meta.d_req_id: ok_meta},
    )

    failed_meta = PullReqMeta(
        d_req_id="d-req-failed",
        transfer_id="xfer-failed",
        local_block_ids=[[101]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap",
    )
    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            err_reqs=[failed_meta.d_req_id],
            err_msg="transfer failed",
        ),
        {failed_meta.d_req_id: failed_meta},
    )

    assert worker._pd_trace_pull_started == {}
    assert worker.finished_recving_reqs == {"d-req-ok", "d-req-failed"}


def test_large_request_gate_uses_largest_kv_group_block_count():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._large_request_semaphore = object()
    worker.large_request_threshold_tokens = 32768
    worker.block_size = 256

    long_meta = SimpleNamespace(req_blocks={"request": ("transfer", [[0] * 256])})
    short_meta = SimpleNamespace(req_blocks={"request": ("transfer", [[0] * 14])})

    assert worker._is_large_request_meta(long_meta)
    assert not worker._is_large_request_meta(short_meta)


@pytest.mark.asyncio
async def test_node_large_request_slots_are_mutually_exclusive(tmp_path):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._node_large_request_slot_paths = (
        MooncakeConnectorWorker._get_node_large_request_slot_paths(
            str(tmp_path), "engine-a", 1
        )
    )
    assert worker._node_large_request_slot_paths != (
        MooncakeConnectorWorker._get_node_large_request_slot_paths(
            str(tmp_path), "engine-b", 1
        )
    )

    first_slot = await worker._acquire_node_large_request_slot()
    assert first_slot is not None

    waiting_for_slot = asyncio.create_task(worker._acquire_node_large_request_slot())
    await asyncio.sleep(0.01)
    assert not waiting_for_slot.done()

    worker._release_node_large_request_slot(first_slot)
    second_slot = await asyncio.wait_for(waiting_for_slot, timeout=1)
    assert second_slot is not None
    worker._release_node_large_request_slot(second_slot)


def test_register_kv_caches_aggregates_shared_overlay_aliases():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.use_mla = True
    worker.model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: 64)
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.4.attn"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["model.layers.4.attn.swa_cache"],
                SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                    sliding_window=128,
                ),
            ),
            KVCacheGroupSpec(
                ["model.layers.4.attn.compressor.state_cache"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                ),
            ),
        ],
    )
    worker._layer_specs = {
        layer_name: group.kv_cache_spec
        for group in worker.kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }
    worker._layer_group_indices = {
        layer_name: group_index
        for group_index, group in enumerate(worker.kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    worker._layer_logical_group_indices = {
        layer_name: [group_index]
        for group_index, group in enumerate(worker.kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    worker.transfer_topo = SimpleNamespace(
        virtually_split_kv_in_blocks=False,
        get_transfer_cache_regions=lambda cache, _spec: [cache],
    )
    worker.engine = MagicMock()
    worker.engine.batch_register_memory.return_value = 0
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = False
    worker.receiver_loop = MagicMock()
    worker.receiver_loop.is_running.return_value = False

    shared_cache = torch.zeros((2, 16, 4, 16), dtype=torch.float16)
    kv_caches = {
        "model.layers.4.attn": shared_cache,
        "model.layers.4.attn.swa_cache": shared_cache,
        "model.layers.4.attn.compressor.state_cache": shared_cache,
    }
    worker._layer_specs = {
        group.layer_names[0]: group.kv_cache_spec
        for group in worker.kv_cache_config.kv_cache_groups
    }
    worker._layer_group_indices = {
        name: group_index for group_index, name in enumerate(kv_caches)
    }
    worker._layer_logical_group_indices = {
        name: [group_index] for group_index, name in enumerate(kv_caches)
    }

    worker.register_kv_caches(kv_caches)

    worker.engine.batch_register_memory.assert_called_once_with(
        [shared_cache.data_ptr()], [shared_cache.nbytes]
    )
    assert worker.registered_layer_names == ["model.layers.4.attn"]
    assert worker.registered_layer_aliases == [list(kv_caches)]
    assert worker.registered_layer_index_aliases == [[4, 4, 4]]
    assert worker.registered_group_indices == [0]
    assert worker.registered_logical_group_indices == [[0, 1, 2]]
    assert worker.registered_alias_group_indices == [[[0], [1], [2]]]

    regions = worker._get_transfer_regions(
        base_addrs=worker.kv_caches_base_addr,
        block_lens=worker.block_len_per_layer,
        kv_block_lens=worker.kv_block_len_per_layer,
        layer_names=worker.registered_layer_names,
        layer_indices=worker.registered_layer_indices,
        layer_aliases=worker.registered_layer_aliases,
        layer_index_aliases=worker.registered_layer_index_aliases,
        group_indices=worker.registered_group_indices,
        logical_group_indices=worker.registered_logical_group_indices,
        alias_group_indices=worker.registered_alias_group_indices,
    )
    aligned_local, aligned_remote, err = _align_transfer_regions(regions, regions)
    assert err is None
    assert aligned_local == regions
    assert aligned_remote == regions


def test_get_transfer_regions_rejects_metadata_shape_mismatch():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker._layer_group_indices = {}
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)

    with pytest.raises(AssertionError, match="matching metadata lengths"):
        worker._get_transfer_regions(
            base_addrs=[0x1000],
            block_lens=[64],
            kv_block_lens=[64],
            layer_names=[],
            layer_indices=[],
        )


def test_register_kv_caches_supports_mixed_mla_and_eagle_shapes():
    """Mixed MLA+Eagle caches should register by byte length, not shape."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        worker.use_mla = True
        worker.transfer_topo.is_mla = True

        # MLA cache tensor: shape[-2] is the block size.
        mla_cache = torch.zeros((2, 16, 96), dtype=torch.float16)
        # Eagle3/GQA-like cache tensor: shape[-2] is num_kv_heads, not block size.
        eagle_cache = torch.zeros((2, 16, 8, 64), dtype=torch.float16)
        kv_caches = {
            "model.layers.0.mla_attn": mla_cache,
            "model.layers.1.eagle_attn": eagle_cache,
        }

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

        mock_batch_register.assert_called_once()
        registered_ptrs, registered_lens = mock_batch_register.call_args[0]
        assert registered_ptrs == [mla_cache.data_ptr(), eagle_cache.data_ptr()]
        assert registered_lens == [mla_cache.nbytes, eagle_cache.nbytes]
        assert worker.block_len_per_layer == [
            mla_cache.nbytes // mla_cache.shape[0],
            eagle_cache.nbytes // eagle_cache.shape[0],
        ]
        assert worker.registered_layer_names == [
            "model.layers.0.mla_attn",
            "model.layers.1.eagle_attn",
        ]
        assert worker.registered_layer_indices == [0, 1]


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
@pytest.mark.parametrize("d_tp_size", [1, 4], ids=["p_tp2_d_tp1", "p_tp2_d_tp4"])
async def test_kv_producer_heterogeneous_tp(monkeypatch, d_tp_size):
    """
    Tests heterogeneous TP support in the producer transfer path.

    Verifies correct pointer and offset calculation when producer TP=2
    sends to consumer with TP=1 (P>D) or TP=4 (P<D).

    Parametrized cases:
    - P TP=2 > D TP=1: one D rank receives; dst_offset based on P rank
    - P TP=2 < D TP=4: two D ranks receive; src_offset based on D rank
    """

    P_TP_SIZE = 2
    P_TP_RANK = 0
    # The fixture model has 12 KV heads, so TP2 owns 6 heads per rank.
    LOCAL_BLOCK_LEN = 6 * 1024

    local_block_len = LOCAL_BLOCK_LEN
    remote_block_len = LOCAL_BLOCK_LEN * P_TP_SIZE // d_tp_size

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Override TP rank/size to simulate P TP=2
        prefill_worker.tp_rank = P_TP_RANK
        prefill_worker.tp_size = P_TP_SIZE
        prefill_worker._tp_size[prefill_worker.engine_id] = P_TP_SIZE
        prefill_worker.transfer_topo.tp_rank = P_TP_RANK
        prefill_worker.transfer_topo.tp_size = P_TP_SIZE

        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [local_block_len]
        prefill_worker.kv_block_len_per_layer = [local_block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        transfer_id = "xfer-hetero-1"
        local_block_ids = [[10, 11]]
        send_meta = SendBlockMeta(
            p_req_id="p-req-h1",
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Compute target D ranks using the production code path
        target_d_ranks = prefill_worker.transfer_topo.handshake_target_ranks(d_tp_size)

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-hetero"

        # Assign different remote block IDs per D rank (nested per-group)
        d_rank_remote_blocks = {
            rank: [[20 + i * 10, 21 + i * 10]] for i, rank in enumerate(target_d_ranks)
        }

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            for d_rank in target_d_ranks:
                remote_block_ids = d_rank_remote_blocks[d_rank]
                xfer_meta = MooncakeXferMetadata(
                    remote_hostname="consumer-host",
                    remote_port=54321,
                    remote_tp_size=d_tp_size,
                    remote_tp_rank=d_rank,
                    req_blocks={
                        f"d-req-h1-r{d_rank}": (
                            transfer_id,
                            remote_block_ids,
                        )
                    },
                    kv_caches_base_addr=[0x2000],
                    block_lens=[remote_block_len],
                    kv_block_lens=[remote_block_len],
                    registered_layer_names=["model.layers.0.self_attn"],
                    registered_layer_indices=[0],
                )

                mock_send_blocks.reset_mock()
                mock_socket.send_multipart.reset_mock()

                await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

                # Verify _send_blocks was called
                mock_send_blocks.assert_called_once()
                call_args = mock_send_blocks.call_args[0]
                src_ptrs = call_args[1]
                dst_ptrs = call_args[2]
                lengths = call_args[3]

                # Flatten nested per-group block IDs for assertions
                flat_local = [b for g in local_block_ids for b in g]
                flat_remote = [b for g in remote_block_ids for b in g]
                num_blocks = len(flat_local)

                assert len(src_ptrs) == num_blocks
                assert len(dst_ptrs) == num_blocks
                assert len(lengths) == num_blocks

                if d_tp_size <= P_TP_SIZE:
                    tp_ratio = P_TP_SIZE // d_tp_size
                    expected_src_off = 0
                    expected_dst_off = (P_TP_RANK % tp_ratio) * local_block_len
                    expected_xfer_len = local_block_len
                else:
                    ratio_abs = d_tp_size // P_TP_SIZE
                    expected_src_off = (d_rank % ratio_abs) * remote_block_len
                    expected_dst_off = 0
                    expected_xfer_len = remote_block_len

                local_region_base = 0x1000
                remote_region_base = 0x2000
                for blk_idx, (lblk, rblk) in enumerate(zip(flat_local, flat_remote)):
                    assert src_ptrs[blk_idx] == (
                        local_region_base + lblk * local_block_len + expected_src_off
                    )
                    assert dst_ptrs[blk_idx] == (
                        remote_region_base + rblk * remote_block_len + expected_dst_off
                    )
                    assert lengths[blk_idx] == expected_xfer_len

                # Verify successful response sent back to consumer
                mock_socket.send_multipart.assert_called_once()
                _, sent_payload = mock_socket.send_multipart.call_args[0][0]
                response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
                assert response.status == MooncakeXferResponseStatus.FINISH
                assert response.ok_reqs == [f"d-req-h1-r{d_rank}"]

        # After serving all D ranks, the request should be complete
        assert transfer_id not in prefill_worker.reqs_need_send
        assert "p-req-h1" in prefill_worker.finished_sending_reqs

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()
