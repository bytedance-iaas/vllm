# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.parameter as parameter_module
import vllm.models.deepseek_v4.attention as attention_module
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


@dataclass
class FakeGroup:
    rank_in_group: int
    world_size: int
    all_reduce_calls: int = 0
    all_reduce_offset: float = 7

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        return tensor + self.all_reduce_offset


def make_forward_context_config() -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=False,
        ),
    )


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


@pytest.mark.parametrize(
    ("cp_rank", "expected_lens", "expected_effective_lens", "expected_indices"),
    [
        (
            0,
            [256, 128],
            [956, 128],
            [*range(0, 256), *range(300, 428)],
        ),
        (1, [44, 0], [1000, 128], [*range(256, 300)]),
    ],
)
def test_attention_cp_plan_splits_each_request_on_aligned_blocks(
    cp_rank: int,
    expected_lens: list[int],
    expected_effective_lens: list[int],
    expected_indices: list[int],
):
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        query_start_loc_cpu=torch.tensor([0, 300, 428], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([1000, 128], dtype=torch.int32),
    )

    plan = attention_module.AttentionCPPlan.build(
        metadata,
        cp_rank=cp_rank,
        cp_size=2,
        alignment=256,
        device=torch.device("cpu"),
    )

    assert plan.local_query_lens_cpu.tolist() == expected_lens
    assert plan.effective_seq_lens.tolist() == expected_effective_lens
    assert plan.token_indices.tolist() == expected_indices


def test_attention_cp_plan_restores_original_packed_order(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = attention_module.AttentionCPPlan(
        token_indices=torch.tensor([1, 3]),
        local_query_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        local_query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        local_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        effective_seq_lens=torch.tensor([2, 4], dtype=torch.int32),
    )
    group = FakeGroup(rank_in_group=0, world_size=2, all_reduce_offset=0)
    monkeypatch.setattr(attention_module, "get_attn_cp_group", lambda: group)
    local_output = torch.tensor([[10.0, 11.0], [30.0, 31.0]])

    restored = plan.restore_output(local_output, num_global_tokens=4)

    torch.testing.assert_close(
        restored,
        torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 11.0],
                [0.0, 0.0],
                [30.0, 31.0],
            ]
        ),
    )
    assert group.all_reduce_calls == 1


def test_attention_cp_plan_supports_empty_shard():
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        query_start_loc_cpu=torch.tensor([0, 128], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([128], dtype=torch.int32),
    )

    plan = attention_module.AttentionCPPlan.build(
        metadata,
        cp_rank=1,
        cp_size=2,
        alignment=256,
        device=torch.device("cpu"),
    )

    assert plan.num_local_tokens == 0
    assert plan.local_query_lens_cpu.tolist() == [0]
    assert plan.token_indices.numel() == 0


def test_attention_cp_plan_rejects_decode_rows():
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1)

    with pytest.raises(NotImplementedError, match="pure prefill"):
        attention_module.AttentionCPPlan.build(
            metadata,
            cp_rank=0,
            cp_size=2,
            alignment=256,
            device=torch.device("cpu"),
        )


def test_forward_context_exposes_profile_runs():
    with set_forward_context({}, make_forward_context_config(), is_profile=True):
        assert get_forward_context().is_profile is True


def test_attention_cp_plan_bypasses_profile_mixed_metadata():
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1)
    layer = SimpleNamespace(
        attn_cp_size=2,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        attn_cp_alignment=256,
    )

    with set_forward_context(
        {"swa": metadata},
        make_forward_context_config(),
        is_profile=True,
    ):
        assert (
            attention_module.DeepseekV4Attention._build_attention_cp_plan(
                layer, torch.device("cpu")
            )
            is None
        )


def test_attention_cp_plan_rejects_production_mixed_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1)
    layer = SimpleNamespace(
        attn_cp_size=2,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        attn_cp_alignment=256,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attn_cp_group",
        lambda: FakeGroup(rank_in_group=0, world_size=2),
    )

    with set_forward_context({"swa": metadata}, make_forward_context_config()):
        assert get_forward_context().is_profile is False
        with pytest.raises(NotImplementedError, match="pure prefill"):
            attention_module.DeepseekV4Attention._build_attention_cp_plan(
                layer, torch.device("cpu")
            )
