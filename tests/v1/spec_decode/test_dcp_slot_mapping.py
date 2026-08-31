# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    _advance_cpu_sequence_metadata,
    compute_new_slot_mapping,
    compute_slot_mapping_from_block_table,
    expand_dcp_parent_block_table,
    extend_all_queries_by_N,
)


def _metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> CommonAttentionMetadata:
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens_cpu = seq_lens.cpu()
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu.clone(),
        seq_lens_cpu_upper_bound=seq_lens_cpu.clone(),
        num_reqs=seq_lens.shape[0],
        num_actual_tokens=int(query_start_loc_cpu[-1]),
        max_query_len=int((query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max()),
        max_seq_len=int(seq_lens_cpu.max()),
        block_table_tensor=block_table,
        slot_mapping=torch.empty(0, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [12927, -1, -1, -1, 13055, -1]),
        (1, [-1, 12800, 12927, -1, -1, 12928]),
    ],
)
def test_compute_new_slot_mapping_localizes_dcp_owner(
    rank: int,
    expected: list[int],
) -> None:
    positions = torch.tensor([127, 128, 255, 256, 383, 384])
    metadata = _metadata(
        torch.tensor([0, positions.numel()], dtype=torch.int32),
        torch.tensor([385], dtype=torch.int32),
        torch.tensor([[100, 101, 102, 103]], dtype=torch.int32),
    )
    rejected = torch.tensor([False, False, False, True, False, False])

    slots = compute_new_slot_mapping(
        cad=metadata,
        new_positions=positions,
        is_rejected_token_mask=rejected,
        block_size=128,
        num_new_tokens=0,
        max_model_len=1024,
        dcp_world_size=2,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=128,
    )

    assert slots.tolist() == expected
    assert slots[3].item() == PADDING_SLOT_ID


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [12927, -1, -1, 25728, -1]),
        (1, [-1, 12800, 12927, -1, -1]),
    ],
)
def test_compute_new_slot_mapping_localizes_expanded_multi_request(
    rank: int,
    expected: list[int],
) -> None:
    metadata = _metadata(
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([256, 384], dtype=torch.int32),
        torch.tensor(
            [[100, 101, 102], [200, 201, 202]],
            dtype=torch.int32,
        ),
    )

    slots = compute_new_slot_mapping(
        cad=metadata,
        new_positions=torch.tensor([127, 128, 255, 256, 383]),
        is_rejected_token_mask=torch.tensor([False, False, False, False, True]),
        block_size=128,
        num_new_tokens=1,
        max_model_len=1024,
        dcp_world_size=2,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=128,
    )

    assert slots.tolist() == expected


def test_full_temporal_parent_child_mapping_at_boundaries() -> None:
    parent_block_table = torch.tensor(
        [[3, 5, 0], [7, 9, 0]],
        dtype=torch.int32,
    )
    expanded = expand_dcp_parent_block_table(
        parent_block_table,
        dcp_world_size=2,
        max_model_len=512,
        kernel_block_size=128,
    )
    assert expanded.tolist() == [[6, 7, 10, 11], [14, 15, 18, 19]]
    assert set(expanded[0].tolist()).isdisjoint(set(expanded[1].tolist()))
    assert torch.equal(
        expanded,
        expand_dcp_parent_block_table(
            parent_block_table,
            dcp_world_size=2,
            max_model_len=512,
            kernel_block_size=128,
        ),
    )

    positions = torch.tensor([0, 127, 128, 255, 256], dtype=torch.int64)
    metadata = _metadata(
        torch.tensor([0, 5], dtype=torch.int32),
        torch.tensor([257], dtype=torch.int32),
        expanded[:1],
    )
    slots = compute_new_slot_mapping(
        cad=metadata,
        new_positions=positions,
        is_rejected_token_mask=torch.zeros(5, dtype=torch.bool),
        block_size=128,
        num_new_tokens=0,
        max_model_len=512,
    )
    assert slots.tolist() == [768, 895, 896, 1023, 1280]

    physical_pages = torch.div(slots, 128, rounding_mode="floor")
    offsets = slots % 128
    assert physical_pages.tolist() == [6, 6, 7, 7, 10]
    assert offsets.tolist() == [0, 127, 0, 127, 0]

    rejected_metadata = _metadata(
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([256], dtype=torch.int32),
        expanded[:1],
    )
    rejected_slots = compute_new_slot_mapping(
        cad=rejected_metadata,
        new_positions=torch.tensor([127, 128, 255], dtype=torch.int64),
        is_rejected_token_mask=torch.tensor([False, True, False]),
        block_size=128,
        num_new_tokens=0,
        max_model_len=512,
    )
    assert rejected_slots.tolist() == [895, PADDING_SLOT_ID, 1023]

    expanded_with_null = expand_dcp_parent_block_table(
        parent_block_table[:1],
        dcp_world_size=2,
        max_model_len=768,
        kernel_block_size=128,
    )
    assert expanded_with_null.tolist() == [[6, 7, 10, 11, 0, 0]]


def test_full_temporal_parent_mapping_is_identity_at_dcp1() -> None:
    parent = torch.tensor([[3, 5, 0]], dtype=torch.int32)

    expanded = expand_dcp_parent_block_table(
        parent,
        dcp_world_size=1,
        max_model_len=384,
        kernel_block_size=128,
    )

    assert expanded.data_ptr() == parent.data_ptr()
    assert expanded.tolist() == [[3, 5, 0]]


def test_full_temporal_slot_mapping_uses_global_positions_across_requests() -> None:
    block_table = torch.tensor(
        [
            [6, 7, 10, 11],
            [14, 15, 18, 19],
            [0, 1, 0, 1],
        ],
        dtype=torch.int32,
    )

    slots = compute_slot_mapping_from_block_table(
        query_start_loc=torch.tensor([0, 3, 5, 5], dtype=torch.int32),
        block_table_tensor=block_table,
        positions=torch.tensor([0, 127, 128, 255, 256]),
        block_size=128,
        max_model_len=512,
    )

    assert slots.tolist() == [768, 895, 896, 2047, 2304]


def test_extend_queries_updates_cpu_sequence_shadows() -> None:
    metadata = _metadata(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([127, 255], dtype=torch.int32),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
    )
    metadata.dcp_local_seq_lens_cpu = torch.tensor(
        [127, 128],
        dtype=torch.int32,
    )
    original_local_seq_lens_cpu = metadata.dcp_local_seq_lens_cpu.clone()

    extended = extend_all_queries_by_N(
        metadata,
        N=3,
        arange=torch.arange(3, dtype=torch.int32),
        new_slot_mapping=torch.arange(8, dtype=torch.int64),
    )

    assert extended.query_start_loc_cpu.tolist() == [0, 4, 8]
    assert extended.seq_lens.tolist() == [130, 258]
    assert extended._seq_lens_cpu is not None
    assert extended._seq_lens_cpu.tolist() == [130, 258]
    assert extended.seq_lens_cpu_upper_bound is not None
    assert extended.seq_lens_cpu_upper_bound.tolist() == [130, 258]
    assert extended.dcp_local_seq_lens_cpu is None
    assert metadata.dcp_local_seq_lens_cpu.tolist() == (
        original_local_seq_lens_cpu.tolist()
    )


def test_advance_cpu_sequence_metadata_handles_unpadded_alias() -> None:
    metadata = _metadata(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([127, 128], dtype=torch.int32),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
    )
    shared_seq_lens = torch.tensor([127, 128, 999], dtype=torch.int32)
    num_computed_tokens = torch.tensor([126, 127, 998], dtype=torch.int32)
    metadata._seq_lens_cpu = shared_seq_lens
    metadata.seq_lens_cpu_upper_bound = shared_seq_lens
    metadata._num_computed_tokens_cpu = num_computed_tokens

    unpadded = metadata.unpadded(num_actual_tokens=2, num_actual_reqs=2)
    seq_lens = unpadded._seq_lens_cpu
    upper_bound = unpadded.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    assert upper_bound is not None
    assert seq_lens is not upper_bound
    assert seq_lens.data_ptr() == upper_bound.data_ptr()

    _advance_cpu_sequence_metadata(unpadded, max_model_len=128)

    assert shared_seq_lens.tolist() == [128, 1, 999]
    assert num_computed_tokens.tolist() == [127, 0, 998]
