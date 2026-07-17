# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import ParallelConfig
from vllm.v1.attention.backends.utils import (
    get_cp_local_seq_lens,
    get_dcp_local_seq_lens,
    get_pcp_kv_indices,
    get_pcp_local_indices_after_restore,
    get_pcp_max_buffer_num_tokens,
    get_pcp_num_local_tokens_from_restore_idx,
    get_pcp_query_indices,
    pcp_allgather_and_restore,
    pcp_kv_allgather_and_restore,
    restore_pcp_local_tensor_to_padded_tokens,
)
from vllm.v1.worker.cp_utils import (
    DSV4_PCP_PREFILL_UNSUPPORTED_ERROR,
    PCPManager,
    guard_dsv4_pcp_prefill_runtime_metadata,
)


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
        manager.pcp_local_unpad_mask_cpu[:9],
        np.array(
            [
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        manager.pcp_local_token_indices_cpu[:9],
        np.array([0, 1, 2, 0, 0, 6, 7, 12, 13], dtype=np.int64),
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
    np.testing.assert_array_equal(
        manager.pcp_local_unpad_mask_cpu[:9],
        np.array(
            [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        manager.pcp_local_token_indices_cpu[:9],
        np.array([0, 3, 4, 5, 0, 8, 9, 10, 11], dtype=np.int64),
    )


def test_pcp_restore_idx_length_uses_local_total_for_odd_request():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )
    tokens = np.array([9], dtype=np.int32)
    arange_np = np.arange(64, dtype=np.int32)

    pcp_tokens, positions = manager.update_tokens_for_pcp(
        tokens,
        arange_np,
        num_reqs=1,
        reorder_batch_threshold=1,
    )

    original_total = int(tokens.sum())
    local_total = int(pcp_tokens.sum())
    assert original_total == 9
    assert local_total == 6
    assert positions.tolist() == [3, 4, 5, 6, 7, 8]

    restore_len = local_total * manager.pcp_world_size
    restore_idx = manager.pcp_allgather_restore_idx.cpu[:restore_len]
    assert restore_len == 12
    assert int(restore_idx.max()) < restore_len


def test_pcp_restore_idx_derives_actual_local_tokens_with_padding():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )
    tokens = np.array([11], dtype=np.int32)
    arange_np = np.arange(64, dtype=np.int32)

    pcp_tokens, _ = manager.update_tokens_for_pcp(
        tokens,
        arange_np,
        num_reqs=1,
        reorder_batch_threshold=1,
    )

    actual_local_tokens = int(pcp_tokens.sum())
    padded_local_tokens = 8
    restore_idx = manager.pcp_allgather_restore_idx.cpu[
        : actual_local_tokens * manager.pcp_world_size
    ]

    assert actual_local_tokens == 6
    assert padded_local_tokens > actual_local_tokens
    assert (
        get_pcp_num_local_tokens_from_restore_idx(
            restore_idx,
            manager.pcp_world_size,
        )
        == actual_local_tokens
    )

    local_indices = get_pcp_local_indices_after_restore(
        num_local_tokens=actual_local_tokens,
        pcp_rank=manager.pcp_rank,
        pcp_allgather_restore_idx=restore_idx,
    )
    assert local_indices.numel() == actual_local_tokens
    assert int(local_indices.max()) < restore_idx.numel()


def test_pcp_local_restore_preserves_padded_token_shape():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )
    tokens = np.array([11], dtype=np.int32)
    arange_np = np.arange(64, dtype=np.int32)

    pcp_tokens, _ = manager.update_tokens_for_pcp(
        tokens,
        arange_np,
        num_reqs=1,
        reorder_batch_threshold=1,
    )

    actual_local_tokens = int(pcp_tokens.sum())
    padded_local_tokens = 8
    restore_idx = manager.pcp_allgather_restore_idx.cpu[
        : actual_local_tokens * manager.pcp_world_size
    ]
    local_indices = get_pcp_local_indices_after_restore(
        num_local_tokens=actual_local_tokens,
        pcp_rank=manager.pcp_rank,
        pcp_allgather_restore_idx=restore_idx,
    )
    restored = torch.arange(restore_idx.numel() * 2).reshape(restore_idx.numel(), 2)

    local_padded = restore_pcp_local_tensor_to_padded_tokens(
        restored,
        local_indices,
        padded_local_tokens,
    )

    assert local_padded.shape == (padded_local_tokens, 2)
    torch.testing.assert_close(
        local_padded[:actual_local_tokens],
        torch.index_select(restored, 0, local_indices),
    )
    torch.testing.assert_close(
        local_padded[actual_local_tokens:],
        torch.zeros(padded_local_tokens - actual_local_tokens, 2, dtype=torch.int64),
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


def test_get_pcp_max_buffer_num_tokens():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=3),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    assert get_pcp_max_buffer_num_tokens(config) == 16

    config.parallel_config.prefill_context_parallel_size = 2
    assert get_pcp_max_buffer_num_tokens(config) == 28


@pytest.mark.parametrize(
    "microbatch_config",
    [
        {"enable_dbo": True},
        {"ubatch_size": 2},
    ],
)
def test_pcp_rejects_microbatching(microbatch_config):
    with pytest.raises(ValueError, match="does not support DBO or microbatching"):
        ParallelConfig(prefill_context_parallel_size=2, **microbatch_config)


def test_deepseek_v4_pcp_prefill_guard_raises_without_runtime_metadata():
    with pytest.raises(NotImplementedError, match=DSV4_PCP_PREFILL_UNSUPPORTED_ERROR):
        guard_dsv4_pcp_prefill_runtime_metadata(
            pcp_allgather_restore_idx=torch.tensor([0, 1], dtype=torch.int64),
            num_prefill_tokens=2,
            runtime_metadata=None,
        )


def test_deepseek_v4_pcp_prefill_guard_allows_non_legacy_paths():
    guard_dsv4_pcp_prefill_runtime_metadata(
        pcp_allgather_restore_idx=None,
        num_prefill_tokens=2,
        runtime_metadata=None,
    )
    guard_dsv4_pcp_prefill_runtime_metadata(
        pcp_allgather_restore_idx=torch.tensor([0, 1], dtype=torch.int64),
        num_prefill_tokens=0,
        runtime_metadata=None,
    )
    guard_dsv4_pcp_prefill_runtime_metadata(
        pcp_allgather_restore_idx=torch.tensor([0, 1], dtype=torch.int64),
        num_prefill_tokens=2,
        runtime_metadata=object(),
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
    torch.testing.assert_close(kv_tail_rank1, torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9]))


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


def test_pcp_allgather_restore_local_indices():
    class FakePCPGroup:
        def __init__(self, other: torch.Tensor) -> None:
            self.other = other

        def all_gather(self, local: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cat([local, self.other.to(local.dtype)], dim=dim)

    local = torch.tensor([[10], [30]], dtype=torch.float32)
    other = torch.tensor([[20], [40]], dtype=torch.float32)
    restore_idx = torch.tensor([0, 2, 1, 3], dtype=torch.int64)

    restored = pcp_allgather_and_restore(
        local,
        num_actual_tokens=2,
        pcp_allgather_restore_idx=restore_idx,
        pcp_group=FakePCPGroup(other),
    )
    torch.testing.assert_close(restored, torch.tensor([[10], [20], [30], [40.0]]))

    local_indices = get_pcp_local_indices_after_restore(
        num_local_tokens=2,
        pcp_rank=0,
        pcp_allgather_restore_idx=restore_idx,
    )
    torch.testing.assert_close(local_indices, torch.tensor([0, 2]))
    torch.testing.assert_close(
        torch.index_select(restored, 0, local_indices),
        local,
    )
