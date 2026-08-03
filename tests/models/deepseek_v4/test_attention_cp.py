# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.parameter as parameter_module
import vllm.models.deepseek_v4.attention as attention_module
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


@dataclass
class FakeGroup:
    rank_in_group: int
    world_size: int
    all_reduce_calls: int = 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        return tensor + 7


@pytest.fixture
def uninitialized_parameter_tp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def test_column_parallel_linear_loads_custom_group_shard(
    uninitialized_parameter_tp,
):
    group = FakeGroup(rank_in_group=1, world_size=2)
    layer = ColumnParallelLinear(
        input_size=4,
        output_size=8,
        bias=False,
        tp_group=group,
    )
    loaded_weight = torch.arange(32, dtype=layer.weight.dtype).reshape(8, 4)

    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert layer.tp_rank == 1
    assert layer.tp_size == 2
    torch.testing.assert_close(layer.weight, loaded_weight[4:8])


def test_row_parallel_linear_loads_and_reduces_custom_group_shard(
    uninitialized_parameter_tp,
):
    group = FakeGroup(rank_in_group=1, world_size=2)
    layer = RowParallelLinear(
        input_size=8,
        output_size=3,
        bias=False,
        return_bias=False,
        tp_group=group,
    )
    loaded_weight = torch.arange(24, dtype=layer.weight.dtype).reshape(3, 8)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    layer.weight.data.fill_(1)

    output = layer(torch.ones(2, 4))

    torch.testing.assert_close(output, torch.full((2, 3), 11.0))
    assert group.all_reduce_calls == 1
    torch.testing.assert_close(layer.weight, torch.ones(3, 4))


@pytest.mark.parametrize(
    ("attn_tp_rank", "expected"),
    [(0, (0, 8)), (1, (8, 16))],
)
def test_attention_tp_head_range_replicates_by_attn_tp_rank(
    monkeypatch: pytest.MonkeyPatch,
    attn_tp_rank: int,
    expected: tuple[int, int],
):
    group = FakeGroup(rank_in_group=attn_tp_rank, world_size=2)
    monkeypatch.setattr(attention_module, "get_attn_tp_group", lambda: group)

    assert attention_module.get_attention_tp_head_range(16) == expected
