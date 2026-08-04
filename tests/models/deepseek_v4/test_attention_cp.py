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


def test_attention_cp_projects_only_local_queries_before_full_kv_insert():
    plan = attention_module.AttentionCPPlan(
        token_indices=torch.tensor([1, 3]),
        local_query_lens_cpu=torch.tensor([2], dtype=torch.int32),
        local_query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        effective_seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    qr = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kv = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    observed: dict[str, object] = {}

    def wq_b(local_qr: torch.Tensor) -> torch.Tensor:
        observed["wq_b_input"] = local_qr.clone()
        return local_qr[:, :2]

    def fused_insert(
        q: torch.Tensor,
        full_kv: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        attn_metadata: object,
        *,
        split_q: bool,
    ) -> torch.Tensor:
        observed["q_positions"] = q_positions.clone()
        observed["kv"] = full_kv
        observed["kv_positions"] = kv_positions
        observed["split_q"] = split_q
        return q

    def forward_mqa(
        q: torch.Tensor,
        full_kv: torch.Tensor,
        full_positions: torch.Tensor,
        output: torch.Tensor,
        cp_plan: attention_module.AttentionCPPlan,
    ) -> None:
        observed["forward_q"] = q
        observed["forward_kv"] = full_kv
        observed["forward_positions"] = full_positions
        observed["forward_plan"] = cp_plan

    layer = SimpleNamespace(
        indexer=None,
        compressor=None,
        wq_b=wq_b,
        n_local_heads=1,
        head_dim=2,
        _fused_qnorm_rope_kv_insert=fused_insert,
        forward_mqa=forward_mqa,
    )
    output = torch.empty(2, 1, 2)

    with set_forward_context({}, make_forward_context_config()):
        attention_module.DeepseekV4Attention.attention_impl(
            layer,
            hidden_states=torch.empty(4, 1),
            qr=qr,
            kv=kv,
            kv_score=torch.empty(4, 1),
            indexer_kv_score=None,
            indexer_weights=None,
            positions=positions,
            out=output,
            cp_plan=plan,
        )

    torch.testing.assert_close(
        observed["wq_b_input"],
        qr.index_select(0, plan.token_indices),
    )
    torch.testing.assert_close(
        observed["q_positions"],
        positions.index_select(0, plan.token_indices),
    )
    assert observed["kv"] is kv
    assert observed["kv_positions"] is positions
    assert observed["split_q"] is True
    assert observed["forward_kv"] is kv
    assert observed["forward_positions"] is positions
    assert observed["forward_plan"] is plan


def test_attention_cp_empty_shard_skips_projection_but_inserts_full_kv():
    plan = attention_module.AttentionCPPlan(
        token_indices=torch.empty(0, dtype=torch.int64),
        local_query_lens_cpu=torch.tensor([0], dtype=torch.int32),
        local_query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
        local_query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
        effective_seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    kv = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    observed: dict[str, object] = {}

    def wq_b(local_qr: torch.Tensor) -> torch.Tensor:
        raise AssertionError("wq_b must not run for an empty CP shard")

    def fused_insert(
        q: torch.Tensor,
        full_kv: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        attn_metadata: object,
        *,
        split_q: bool,
    ) -> torch.Tensor:
        observed["q"] = q
        observed["kv"] = full_kv
        observed["q_positions"] = q_positions
        observed["kv_positions"] = kv_positions
        observed["split_q"] = split_q
        return q

    def forward_mqa(
        q: torch.Tensor,
        full_kv: torch.Tensor,
        full_positions: torch.Tensor,
        output: torch.Tensor,
        cp_plan: attention_module.AttentionCPPlan,
    ) -> None:
        observed["forward_q"] = q

    layer = SimpleNamespace(
        indexer=None,
        compressor=None,
        wq_b=wq_b,
        n_local_heads=1,
        head_dim=2,
        _fused_qnorm_rope_kv_insert=fused_insert,
        forward_mqa=forward_mqa,
    )

    with set_forward_context({}, make_forward_context_config()):
        attention_module.DeepseekV4Attention.attention_impl(
            layer,
            hidden_states=torch.empty(4, 1),
            qr=torch.empty(4, 3),
            kv=kv,
            kv_score=torch.empty(4, 1),
            indexer_kv_score=None,
            indexer_weights=None,
            positions=positions,
            out=torch.empty(0, 1, 2),
            cp_plan=plan,
        )

    q = observed["q"]
    assert isinstance(q, torch.Tensor)
    assert q.shape == (0, 1, 2)
    assert observed["kv"] is kv
    q_positions = observed["q_positions"]
    assert isinstance(q_positions, torch.Tensor)
    assert q_positions.numel() == 0
    assert observed["kv_positions"] is positions
    assert observed["split_q"] is True
    assert observed["forward_q"] is q


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
    class ProductionMixedMetadata:
        num_decodes = 1
        num_decode_tokens = 1

        @property
        def query_start_loc_cpu(self):
            raise AssertionError("query_start_loc_cpu should not be accessed")

        @property
        def prefill_seq_lens_cpu(self):
            raise AssertionError("prefill_seq_lens_cpu should not be accessed")

    metadata = ProductionMixedMetadata()
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


def test_attention_cp_plan_reuses_cached_plan_within_forward_context(
    monkeypatch: pytest.MonkeyPatch,
):
    query_start_loc_cpu = torch.tensor([0, 300, 428], dtype=torch.int32)
    prefill_seq_lens_cpu = torch.tensor([1000, 128], dtype=torch.int32)
    metadata_a = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        query_start_loc_cpu=query_start_loc_cpu,
        prefill_seq_lens_cpu=prefill_seq_lens_cpu,
    )
    metadata_b = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        query_start_loc_cpu=query_start_loc_cpu,
        prefill_seq_lens_cpu=prefill_seq_lens_cpu,
    )
    layer_a = SimpleNamespace(
        attn_cp_size=2,
        swa_cache_layer=SimpleNamespace(prefix="swa_a"),
        attn_cp_alignment=256,
    )
    layer_b = SimpleNamespace(
        attn_cp_size=2,
        swa_cache_layer=SimpleNamespace(prefix="swa_b"),
        attn_cp_alignment=256,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attn_cp_group",
        lambda: FakeGroup(rank_in_group=0, world_size=2),
    )

    build_calls = 0
    original_build = attention_module.AttentionCPPlan.build

    def counting_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(attention_module.AttentionCPPlan, "build", counting_build)

    with set_forward_context(
        {"swa_a": metadata_a, "swa_b": metadata_b},
        make_forward_context_config(),
    ):
        plan_a = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer_a, torch.device("cpu")
        )
        plan_b = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer_b, torch.device("cpu")
        )

    assert plan_a is plan_b
    assert build_calls == 1


@pytest.mark.parametrize(
    ("field_name", "replace_tensor"),
    [
        (
            "query_start_loc_cpu",
            lambda: torch.tensor([0, 300, 428], dtype=torch.int32),
        ),
        (
            "prefill_seq_lens_cpu",
            lambda: torch.tensor([1000, 0, 128], dtype=torch.int32).as_strided(
                (2,), (2,), 0
            ),
        ),
    ],
)
def test_attention_cp_plan_invalidates_cache_on_metadata_tensor_replacement(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    replace_tensor,
):
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        query_start_loc_cpu=torch.tensor([0, 300, 428], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([1000, 128], dtype=torch.int32),
    )
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

    build_calls = 0
    original_build = attention_module.AttentionCPPlan.build

    def counting_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(attention_module.AttentionCPPlan, "build", counting_build)

    with set_forward_context({"swa": metadata}, make_forward_context_config()):
        first_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )
        first_signature = attention_module._attention_cp_plan_cache_signature(
            metadata,
            cp_rank=0,
            cp_size=2,
            alignment=256,
            device=torch.device("cpu"),
        )
        setattr(metadata, field_name, replace_tensor())
        second_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )
        second_signature = attention_module._attention_cp_plan_cache_signature(
            metadata,
            cp_rank=0,
            cp_size=2,
            alignment=256,
            device=torch.device("cpu"),
        )

    assert first_plan is not second_plan
    assert first_signature != second_signature
    assert build_calls == 2, field_name


def test_attention_cp_plan_reuses_cached_plan_in_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
):
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

    build_calls = 0
    original_build = attention_module.AttentionCPPlan.build

    def counting_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(attention_module.AttentionCPPlan, "build", counting_build)

    with torch.inference_mode():
        metadata = SimpleNamespace(
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=2,
            query_start_loc_cpu=torch.tensor([0, 300, 428], dtype=torch.int32),
            prefill_seq_lens_cpu=torch.tensor([1000, 128], dtype=torch.int32),
        )

        with set_forward_context({"swa": metadata}, make_forward_context_config()):
            first_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
                layer, torch.device("cpu")
            )
            second_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
                layer, torch.device("cpu")
            )

    assert first_plan is second_plan
    assert build_calls == 1


def test_attention_cp_plan_replaces_foreign_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
):
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        query_start_loc_cpu=torch.tensor([0, 128], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([256], dtype=torch.int32),
    )
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
        forward_context = get_forward_context()
        forward_context.additional_kwargs[
            attention_module._ATTENTION_CP_PLAN_CACHE_KEY
        ] = "foreign-value"

        plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )

        cache_entry = forward_context.additional_kwargs[
            attention_module._ATTENTION_CP_PLAN_CACHE_KEY
        ]

    assert isinstance(cache_entry, attention_module.AttentionCPPlanCacheEntry)
    assert cache_entry.plan is plan


def test_attention_cp_plan_does_not_reuse_across_forward_contexts(
    monkeypatch: pytest.MonkeyPatch,
):
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        query_start_loc_cpu=torch.tensor([0, 128], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([256], dtype=torch.int32),
    )
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

    build_calls = 0
    original_build = attention_module.AttentionCPPlan.build

    def counting_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(attention_module.AttentionCPPlan, "build", counting_build)

    with set_forward_context({"swa": metadata}, make_forward_context_config()):
        first_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )
    with set_forward_context({"swa": metadata}, make_forward_context_config()):
        second_plan = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )

    assert first_plan is not second_plan
    assert build_calls == 2
