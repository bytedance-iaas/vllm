# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
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


def test_extend_queries_updates_cpu_sequence_shadows() -> None:
    metadata = _metadata(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([127, 255], dtype=torch.int32),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
    )

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
