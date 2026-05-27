# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.attention.backends.utils import (
    get_cp_local_seq_lens,
    get_dcp_local_seq_lens,
    get_pcp_kv_indices,
    get_pcp_query_indices,
    pcp_kv_allgather_and_restore,
)
from vllm.v1.worker.cp_utils import PCPManager


def test_pcp_manager_dual_chunk_swap_positions_rank0():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )
    tokens = np.array([1, 5, 8], dtype=np.int32)
    arange_np = np.arange(64, dtype=np.int32)

    pcp_tokens, positions = manager.update_tokens_for_pcp(
        tokens,
        arange_np,
        num_reqs=3,
        reorder_batch_threshold=1,
    )

    np.testing.assert_array_equal(pcp_tokens, np.array([1, 4, 4], dtype=np.int32))
    np.testing.assert_array_equal(
        positions, np.array([0, 0, 1, 6, 7, 0, 1, 6, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        manager.num_pcp_pads_cpu[:3], np.array([1, 3, 0], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        manager.pcp_unpad_mask_cpu[:18],
        np.array(
            [
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        manager.pcp_allgather_restore_idx.cpu[:18].numpy(),
        np.array([0, 9, 1, 2, 10, 11, 12, 13, 3, 4, 5, 6, 14, 15, 16, 17, 7, 8]),
    )


def test_pcp_manager_dual_chunk_swap_positions_rank1():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )
    tokens = np.array([1, 5, 8], dtype=np.int32)
    arange_np = np.arange(64, dtype=np.int32)

    pcp_tokens, positions = manager.update_tokens_for_pcp(
        tokens,
        arange_np,
        num_reqs=3,
        reorder_batch_threshold=1,
    )

    np.testing.assert_array_equal(pcp_tokens, np.array([1, 4, 4], dtype=np.int32))
    np.testing.assert_array_equal(
        positions, np.array([0, 2, 3, 4, 5, 2, 3, 4, 5], dtype=np.int32)
    )


def test_get_cp_local_seq_lens_preserves_dcp_helper_behavior():
    seq_lens = torch.tensor([1, 5, 8, 9], dtype=torch.int64)

    for rank in range(2):
        torch.testing.assert_close(
            get_cp_local_seq_lens(
                seq_lens,
                cp_world_size=2,
                cp_rank=rank,
                cp_kv_cache_interleave_size=1,
            ),
            get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=2,
                dcp_rank=rank,
                cp_kv_cache_interleave_size=1,
            ),
        )


def test_get_pcp_query_and_kv_indices():
    cu_num_tokens = torch.tensor([0, 4, 12], dtype=torch.int32)

    query_head, query_tail = get_pcp_query_indices(cu_num_tokens)
    torch.testing.assert_close(query_head, torch.tensor([0, 1, 4, 5, 6, 7]))
    torch.testing.assert_close(query_tail, torch.tensor([2, 3, 8, 9, 10, 11]))

    kv_head_rank0, kv_tail_rank0 = get_pcp_kv_indices(
        cu_num_tokens, pcp_rank=0, pcp_size=2
    )
    torch.testing.assert_close(kv_head_rank0, torch.tensor([0, 4, 5]))
    torch.testing.assert_close(
        kv_tail_rank0, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )

    kv_head_rank1, kv_tail_rank1 = get_pcp_kv_indices(
        cu_num_tokens, pcp_rank=1, pcp_size=2
    )
    torch.testing.assert_close(kv_head_rank1, torch.tensor([0, 1, 4, 5, 6, 7]))
    torch.testing.assert_close(
        kv_tail_rank1, torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9])
    )


def test_pcp_kv_allgather_and_restore():
    class FakePCPGroup:
        def __init__(self, other: torch.Tensor) -> None:
            self.other = other

        def all_gather(self, local: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cat([local, self.other.to(local.dtype)], dim=dim)

    key = torch.tensor([[10], [30], [999]], dtype=torch.float32)
    value = torch.tensor([[100], [300], [999]], dtype=torch.float32)
    other = torch.tensor([[20], [40]], dtype=torch.float32)
    restore_idx = torch.tensor([0, 2, 1, 3], dtype=torch.int64)

    restored_key, restored_value = pcp_kv_allgather_and_restore(
        key,
        value,
        num_actual_tokens=2,
        pcp_allgather_restore_idx=restore_idx,
        pcp_group=FakePCPGroup(other),
    )

    torch.testing.assert_close(restored_key, torch.tensor([[10], [20], [30], [40.0]]))
    torch.testing.assert_close(
        restored_value, torch.tensor([[100], [20], [300], [40.0]])
    )
