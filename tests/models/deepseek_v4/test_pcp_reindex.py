# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PCP_METADATA_PATH = (
    Path(__file__).parents[3] / "vllm" / "models" / "deepseek_v4" / "pcp_metadata.py"
)
_PCP_METADATA_SPEC = importlib.util.spec_from_file_location(
    "_test_deepseek_v4_pcp_metadata", _PCP_METADATA_PATH
)
assert _PCP_METADATA_SPEC is not None
_PCP_METADATA = importlib.util.module_from_spec(_PCP_METADATA_SPEC)
assert _PCP_METADATA_SPEC.loader is not None
_PCP_METADATA_SPEC.loader.exec_module(_PCP_METADATA)

build_pcp_sparse_prefill_rows = _PCP_METADATA.build_pcp_sparse_prefill_rows
build_pcp_swa_prefill_segments = _PCP_METADATA.build_pcp_swa_prefill_segments
build_pcp_compressed_slot_mapping = _PCP_METADATA.build_pcp_compressed_slot_mapping
build_pcp_full_slot_mapping = _PCP_METADATA.build_pcp_full_slot_mapping
build_pcp_query_plan = _PCP_METADATA.build_pcp_query_plan
build_pcp_restored_req_indices = _PCP_METADATA.build_pcp_restored_req_indices
build_pcp_restored_valid_mask = _PCP_METADATA.build_pcp_restored_valid_mask
compact_pcp_sparse_prefill_queries = _PCP_METADATA.compact_pcp_sparse_prefill_queries
compact_pcp_sparse_indices = _PCP_METADATA.compact_pcp_sparse_indices
overlay_pcp_restored_swa_kv_workspace = (
    _PCP_METADATA.overlay_pcp_restored_swa_kv_workspace
)


def test_compact_pcp_sparse_indices_removes_sentinels_from_valid_prefix():
    indices = torch.tensor(
        [[0, -1, 2, -1], [-1, 4, 5, -1], [7, 8, -1, -1]],
        dtype=torch.int32,
    )
    lengths = torch.tensor([3, 3, 2], dtype=torch.int32)

    compacted, new_lengths = compact_pcp_sparse_indices(indices, lengths)

    torch.testing.assert_close(
        compacted,
        torch.tensor(
            [[0, 2, -1, -1], [4, 5, -1, -1], [7, 8, -1, -1]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        new_lengths,
        torch.tensor([2, 2, 2], dtype=torch.int32),
    )


def test_build_pcp_full_slot_mapping_uses_restored_positions():
    block_table = torch.tensor(
        [
            [4, 7, -1],
            [9, 10, 11],
        ],
        dtype=torch.int32,
    )

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 31, 32, 63, 64, 0], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64),
        block_table=block_table,
        block_size=32,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([128, 159, 224, 351, 352, -1], dtype=torch.int64),
    )


def test_build_pcp_full_slot_mapping_masks_restored_padding_rows():
    block_table = torch.tensor([[4]], dtype=torch.int32)

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        block_table=block_table,
        block_size=8,
        valid_mask=torch.tensor([True, True, True, False]),
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([32, 33, 34, -1], dtype=torch.int64),
    )


def test_build_pcp_full_slot_mapping_uses_storage_block_size_for_swa_cache():
    # DeepSeek V4 SWA cache storage blocks can be smaller than the logical
    # metadata block size. Restored PCP KV writes must use the storage block
    # size so the slots match the sparse prefill read path.
    block_table = torch.tensor([[1]], dtype=torch.int32)

    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        req_indices=torch.tensor([0, 0, 0], dtype=torch.int64),
        block_table=block_table,
        block_size=64,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([64, 65, 66], dtype=torch.int64),
    )


def test_build_pcp_compressed_slot_mapping_uses_compressed_boundaries():
    block_table = torch.tensor([[3, 4]], dtype=torch.int32)

    slot_mapping = build_pcp_compressed_slot_mapping(
        positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
        req_indices=torch.zeros(8, dtype=torch.int64),
        block_table=block_table,
        block_size=4,
        compress_ratio=4,
        valid_mask=torch.tensor(
            [True, True, True, True, True, True, True, False],
        ),
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([-1, -1, -1, 12, -1, -1, -1, -1], dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("cp_rank", "physical_blocks", "expected"),
    [
        (
            0,
            [10, 11, 12, 13],
            [20, 21, 22, 23, -1, -1, -1, -1, 24, 25, 26, 27, -1, -1, -1, -1],
        ),
        (
            1,
            [20, 21, 22, 23],
            [-1, -1, -1, -1, 40, 41, 42, 43, -1, -1, -1, -1, 44, 45, 46, 47],
        ),
    ],
)
def test_build_pcp_full_slot_mapping_obeys_interleaved_owner(
    cp_rank, physical_blocks, expected
):
    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.arange(16, dtype=torch.int64),
        req_indices=torch.zeros(16, dtype=torch.int64),
        block_table=torch.tensor([physical_blocks], dtype=torch.int32),
        block_size=2,
        cp_world_size=2,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=4,
    )

    torch.testing.assert_close(slot_mapping, torch.tensor(expected, dtype=torch.int64))


def test_build_pcp_full_slot_mapping_supports_swa_pages_smaller_than_interleave():
    # Production V4 uses a 256-token CP interleave while SWA stores 64-token
    # pages. Rank 0 owns [0, 256) and [512, 768), compacted into local space.
    slot_mapping = build_pcp_full_slot_mapping(
        positions=torch.tensor([0, 63, 64, 255, 256, 319, 511, 512]),
        req_indices=torch.zeros(8, dtype=torch.int64),
        block_table=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
        block_size=64,
        cp_world_size=2,
        cp_rank=0,
        cp_kv_cache_interleave_size=256,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([64, 127, 128, 319, -1, -1, -1, 320]),
    )


@pytest.mark.parametrize(
    ("cp_rank", "physical_blocks", "expected"),
    [
        (0, [3, 4], [192, 255, -1, -1, 256]),
        (1, [7, 8], [-1, -1, 448, 511, -1]),
    ],
)
def test_build_pcp_compressed_slot_mapping_uses_original_token_owner(
    cp_rank, physical_blocks, expected
):
    slot_mapping = build_pcp_compressed_slot_mapping(
        positions=torch.tensor([3, 255, 259, 511, 515]),
        req_indices=torch.zeros(5, dtype=torch.int64),
        block_table=torch.tensor([physical_blocks], dtype=torch.int32),
        block_size=64,
        compress_ratio=4,
        cp_world_size=2,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=256,
    )

    torch.testing.assert_close(slot_mapping, torch.tensor(expected))


def test_build_pcp_compressed_slot_mapping_rejects_split_compression_group():
    with pytest.raises(ValueError, match="divisible by compress_ratio"):
        build_pcp_compressed_slot_mapping(
            positions=torch.tensor([3]),
            req_indices=torch.tensor([0]),
            block_table=torch.tensor([[1]], dtype=torch.int32),
            block_size=64,
            compress_ratio=4,
            cp_world_size=2,
            cp_rank=0,
            cp_kv_cache_interleave_size=2,
        )


def test_build_pcp_restored_req_indices_uses_view_restore_lengths():
    req_indices = build_pcp_restored_req_indices(
        positions=torch.arange(7, dtype=torch.int64),
        views=[
            SimpleNamespace(req_idx=0, restore_idx=torch.arange(4)),
            SimpleNamespace(req_idx=1, restore_idx=torch.arange(2)),
        ],
    )

    torch.testing.assert_close(
        req_indices,
        torch.tensor([0, 0, 0, 0, 1, 1, -1], dtype=torch.int64),
    )


def test_build_pcp_restored_valid_mask_drops_per_rank_padding_rows():
    valid_mask = build_pcp_restored_valid_mask(
        positions=torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64),
        views=[
            SimpleNamespace(global_seq_len=3, restore_idx=torch.arange(4)),
            SimpleNamespace(global_seq_len=2, restore_idx=torch.arange(2)),
        ],
    )

    torch.testing.assert_close(
        valid_mask,
        torch.tensor([True, True, True, False, True, True]),
    )


def test_build_pcp_restored_valid_mask_accepts_absolute_chunk_positions():
    valid_mask = build_pcp_restored_valid_mask(
        positions=torch.tensor([4096, 4097, 4098, 4099]),
        views=[SimpleNamespace(global_seq_len=3, restore_idx=torch.arange(4))],
    )

    torch.testing.assert_close(
        valid_mask,
        torch.tensor([True, True, True, False]),
    )


@pytest.mark.parametrize(
    ("pcp_rank", "expected_local_indices", "expected_local_valid"),
    [
        (0, [0, 2, 4], [True, True, True, False]),
        (1, [1, 3], [True, True, False, False]),
    ],
)
def test_build_pcp_query_plan_compacts_padding_and_selects_local_rows(
    pcp_rank, expected_local_indices, expected_local_valid
):
    # Gathered rank order is [r0: 0,2,4,pad, r1: 1,3,pad,pad].  restore_idx
    # puts the rows into request-local order with padding at the end.
    restore_idx = torch.tensor([0, 4, 1, 5, 2, 3, 6, 7], dtype=torch.int64)
    plan = build_pcp_query_plan(
        pcp_allgather_restore_idx=restore_idx,
        views=[SimpleNamespace(global_seq_len=5, restore_idx=torch.arange(8))],
        pcp_world_size=2,
        pcp_rank=pcp_rank,
    )

    torch.testing.assert_close(
        plan.compact_restored_indices,
        torch.arange(5, dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan.local_compact_indices,
        torch.tensor(expected_local_indices, dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan.local_valid_mask,
        torch.tensor(expected_local_valid),
    )
    assert plan.num_local_tokens == 4


def test_overlay_pcp_restored_swa_kv_workspace_ignores_padding_rows():
    restored_positions = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 0])
    restored_valid_mask = torch.tensor(
        [True, False, True, True, True, True, True, True, True, False]
    )
    restored_kv = torch.stack(
        [restored_positions.to(torch.float32), restored_positions.to(torch.float32)],
        dim=1,
    )
    out = torch.full((1, 8, 2), -1.0)

    overlay_pcp_restored_swa_kv_workspace(
        out=out,
        restored_kv=restored_kv,
        restored_positions=restored_positions,
        restored_valid_mask=restored_valid_mask,
        views=[SimpleNamespace(restore_idx=torch.arange(10))],
        chunk_start=0,
        chunk_end=1,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([8], dtype=torch.int32),
        chunk_n=0,
        chunk_m=8,
    )

    expected = torch.stack(
        [torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32)],
        dim=1,
    )
    torch.testing.assert_close(out[0], expected)


def test_build_pcp_sparse_prefill_rows_compacts_global_query_rows():
    rows = build_pcp_sparse_prefill_rows(
        combined_lens=torch.tensor([1, 1, 0], dtype=torch.int32),
        positions=torch.tensor([5, 6, 10], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
    )

    assert rows.sparse_rows == 3
    assert (rows.rows_min, rows.rows_max) == (1, 2)
    torch.testing.assert_close(rows.q_rows, torch.tensor([1, 2, 0]))
    torch.testing.assert_close(rows.valid_query_mask, torch.tensor([True, True, False]))


def test_compact_pcp_sparse_prefill_queries_allocates_only_valid_local_rows():
    q = torch.arange(4 * 2, dtype=torch.float32).view(4, 1, 2)
    indices = torch.tensor(
        [[100, 101], [-1, -1], [900_000, 900_001], [-1, -1]],
        dtype=torch.int32,
    )
    lengths = torch.tensor([2, 0, 2, 0], dtype=torch.int32)

    compact_q, compact_indices, compact_lengths, output_rows = (
        compact_pcp_sparse_prefill_queries(q, indices, lengths)
    )

    torch.testing.assert_close(compact_q, q[[0, 2]])
    torch.testing.assert_close(compact_indices, indices[[0, 2]])
    torch.testing.assert_close(compact_lengths, torch.tensor([2, 2], dtype=torch.int32))
    torch.testing.assert_close(output_rows, torch.tensor([0, 2]))

    output = torch.zeros_like(q)
    output.index_copy_(0, output_rows, compact_q + 1)
    torch.testing.assert_close(output[output_rows], compact_q + 1)
    torch.testing.assert_close(output[[1, 3]], torch.zeros((2, 1, 2)))


def test_build_pcp_swa_prefill_segments_rebases_to_local_kv_window():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
        combined_lens=torch.tensor([2, 2], dtype=torch.int32),
        positions=torch.tensor([5, 6], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
        window_size=3,
    )

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.query_start, segment.query_end) == (0, 2)
    assert (segment.kv_start, segment.kv_end) == (0, 3)
    assert segment.sparse_rows == 3
    torch.testing.assert_close(segment.q_rows, torch.tensor([1, 2]))
    torch.testing.assert_close(
        segment.shifted_indices,
        torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        segment.topk_lens, torch.tensor([2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(segment.valid_mask, torch.tensor([True, True]))


def test_build_pcp_swa_prefill_segments_splits_dual_chunk_position_jump():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.tensor(
            [
                [2, 3, -1],
                [3, 4, -1],
                [17, 18, -1],
                [18, 19, -1],
            ],
            dtype=torch.int32,
        ),
        combined_lens=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        positions=torch.tensor([5, 6, 20, 21], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([24], dtype=torch.int32),
        gather_lens=torch.tensor([24], dtype=torch.int32),
        chunk_n=0,
        chunk_m=24,
        window_size=4,
        segment_size=64,
    )

    assert len(segments) == 2
    assert (segments[0].query_start, segments[0].query_end) == (0, 2)
    assert (segments[0].kv_start, segments[0].kv_end) == (2, 7)
    assert (segments[1].query_start, segments[1].query_end) == (2, 4)
    assert (segments[1].kv_start, segments[1].kv_end) == (17, 22)
    torch.testing.assert_close(
        segments[1].shifted_indices,
        torch.tensor([[0, 1, -1], [1, 2, -1]], dtype=torch.int32),
    )


def test_build_pcp_swa_prefill_segments_handles_empty_valid_segment():
    segments = build_pcp_swa_prefill_segments(
        combined_indices=torch.full((2, 3), -1, dtype=torch.int32),
        combined_lens=torch.zeros(2, dtype=torch.int32),
        positions=torch.tensor([9, 10], dtype=torch.int64),
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([4], dtype=torch.int32),
        chunk_n=0,
        chunk_m=4,
        window_size=3,
    )

    assert len(segments) == 1
    assert segments[0].sparse_rows == 1
    assert (segments[0].kv_start, segments[0].kv_end) == (0, 0)
    torch.testing.assert_close(segments[0].valid_mask, torch.tensor([False, False]))


def test_build_pcp_swa_prefill_segments_rejects_out_of_window_indices():
    with pytest.raises(ValueError, match="outside the rebased KV workspace"):
        build_pcp_swa_prefill_segments(
            combined_indices=torch.tensor([[10]], dtype=torch.int32),
            combined_lens=torch.tensor([1], dtype=torch.int32),
            positions=torch.tensor([5], dtype=torch.int64),
            local_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            gather_lens=torch.tensor([4], dtype=torch.int32),
            chunk_n=0,
            chunk_m=4,
            window_size=3,
        )


def test_pcp_swa_torch_sparse_fwd_keeps_overflowed_logits_finite():
    assert hasattr(_PCP_METADATA, "pcp_swa_torch_sparse_fwd")
    pcp_swa_torch_sparse_fwd = _PCP_METADATA.pcp_swa_torch_sparse_fwd
    q = torch.full((1, 1, 2), 1e20, dtype=torch.float32)
    kv = torch.full((2, 1, 2), 1e20, dtype=torch.float32)
    indices = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_length = torch.tensor([2], dtype=torch.int32)
    out = torch.empty_like(q)

    pcp_swa_torch_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        sm_scale=1.0,
        attn_sink=None,
        out=out,
    )

    assert torch.isfinite(out).all()
