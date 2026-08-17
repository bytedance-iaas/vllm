# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.minimax_m3.common.indexer as indexer
import vllm.models.minimax_m3.common.sparse_attention as sparse_attention


class _SM100Platform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_device_capability_family(family: int) -> bool:
        return family == 100


def test_sm100_dcp_forces_triton_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2)
    )
    monkeypatch.setattr(indexer, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(sparse_attention, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(indexer, "current_platform", _SM100Platform())
    monkeypatch.setattr(sparse_attention, "current_platform", _SM100Platform())

    indexer_impl_cls = indexer.select_indexer_impl_cls(
        topk_blocks=16,
        indexer_kv_dtype="bf16",
    )
    main_impl_cls = sparse_attention.select_main_impl_cls(
        topk_blocks=16,
        kv_cache_dtype="bfloat16",
        num_kv_heads=1,
    )

    assert indexer_impl_cls is indexer.MiniMaxM3IndexerTritonImpl
    assert main_impl_cls is sparse_attention.MiniMaxM3SparseTritonImpl
    with pytest.raises(NotImplementedError, match="requires the BF16 Triton indexer"):
        indexer.select_indexer_impl_cls(
            topk_blocks=16,
            indexer_kv_dtype="fp8",
        )


class _FakeDCPGroup:
    def __init__(self, gathered: torch.Tensor) -> None:
        self.gathered = gathered

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert input_.shape == (1, 1, 3, 2)
        assert dim == 2
        return self.gathered.clone()


@pytest.mark.parametrize(
    ("rank", "score", "expected"),
    [
        (0, [1.0, 9.0, 3.0], [1, -1, -1]),
        (1, [8.0, 2.0, 7.0], [2, 0, -1]),
    ],
)
def test_dcp_global_topk_localizes_selected_owners(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    score: list[float],
    expected: list[int],
) -> None:
    # Rank 0 owns global blocks 0/2/4; rank 1 owns 1/3/5.
    # The global top-3 is {2, 1, 5}.
    gathered = torch.tensor(
        [[[[9.0, 2.0], [3.0, 4.0], [1.0, 0.0], [8.0, 1.0], [7.0, 5.0], [2.0, 3.0]]]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(indexer, "get_dcp_group", lambda: _FakeDCPGroup(gathered))

    impl = SimpleNamespace(
        dcp_world_size=2,
        dcp_rank=rank,
        block_size=128,
        init_blocks=0,
        local_blocks=0,
        topk_blocks=3,
    )
    out = torch.empty((1, 1, 3), dtype=torch.int32)

    result = indexer.MiniMaxM3IndexerTritonImpl._select_dcp_global_topk(
        impl,
        torch.tensor(score, dtype=torch.float32).view(1, 1, 3),
        torch.tensor([6 * 128], dtype=torch.int32),
        out,
        max_local_blocks=3,
    )

    assert result.tolist() == [[expected]]
