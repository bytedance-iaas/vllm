# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.distributed.parallel_state import (
    _get_attention_parallel_group_ranks,
)
from vllm.model_executor.layers.linear import RowParallelLinear


def test_attention_context_parallel_default_preserves_world_size():
    config = ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)

    assert config.attention_context_parallel_size == 1
    assert config.world_size == 8


def test_attention_context_parallel_does_not_increase_world_size():
    config = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        attention_context_parallel_size=2,
    )

    assert config.world_size == 8


def test_attention_context_parallel_requires_divisible_tp():
    with pytest.raises(
        ValueError,
        match="tp_size=4 must be divisible by attention_context_parallel_size=3",
    ):
        ParallelConfig(
            tensor_parallel_size=4,
            attention_context_parallel_size=3,
        )


@pytest.mark.parametrize(
    "incompatible_config",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_attention_context_parallel_rejects_other_context_parallel_modes(
    incompatible_config: dict[str, int],
):
    with pytest.raises(ValueError, match="cannot be combined"):
        ParallelConfig(
            tensor_parallel_size=4,
            attention_context_parallel_size=2,
            **incompatible_config,
        )


def test_attention_parallel_groups_match_tp8_layout():
    ranks = torch.arange(8).reshape(1, 1, 1, 1, 8)

    attn_tp_groups, attn_cp_groups = _get_attention_parallel_group_ranks(
        ranks,
        tensor_model_parallel_size=8,
        attention_context_model_parallel_size=2,
    )

    assert attn_tp_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert attn_cp_groups == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_attention_parallel_groups_preserve_outer_dimensions():
    # ExternalDP=1, DP=2, PP=2, PCP=1, TP=4.
    ranks = torch.arange(16).reshape(1, 2, 2, 1, 4)

    attn_tp_groups, attn_cp_groups = _get_attention_parallel_group_ranks(
        ranks,
        tensor_model_parallel_size=4,
        attention_context_model_parallel_size=2,
    )

    assert attn_tp_groups == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
    ]
    assert attn_cp_groups == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [8, 10],
        [9, 11],
        [12, 14],
        [13, 15],
    ]


def test_row_parallel_linear_uses_override_group(monkeypatch):
    class FakeGroup:
        rank_in_group = 1
        world_size = 2
        called = False

        def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
            self.called = True
            return tensor + 7

    group = FakeGroup()
    layer = RowParallelLinear(
        8,
        3,
        bias=False,
        return_bias=False,
        tp_group=group,
    )
    assert (layer.tp_rank, layer.tp_size, tuple(layer.weight.shape)) == (
        1,
        2,
        (3, 4),
    )
    monkeypatch.setattr(
        layer.quant_method,
        "apply",
        lambda layer, x, bias: x.sum(-1, keepdim=True).expand(-1, 3),
    )

    output = layer(torch.ones(2, 4))

    assert group.called
    assert torch.equal(output, torch.full((2, 3), 11.0))
