# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v41.attention as attention_module
from vllm.forward_context import get_forward_context, set_forward_context


@dataclass
class FakeGroup:
    rank_in_group: int
    world_size: int
    all_gatherv_result: torch.Tensor | None = None
    all_gatherv_sizes: list[int] | None = None

    def all_gatherv(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        assert dim == 0
        self.all_gatherv_sizes = sizes
        return tensor if self.all_gatherv_result is None else self.all_gatherv_result


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


@pytest.mark.parametrize(
    ("cp_rank", "expected_lens", "expected_indices"),
    [
        (0, [256, 128], [*range(0, 256), *range(300, 428)]),
        (1, [44, 0], [*range(256, 300)]),
    ],
)
def test_attention_cp_plan_splits_each_request_on_aligned_blocks(
    cp_rank: int,
    expected_lens: list[int],
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

    assert torch.diff(plan.local_query_start_loc_cpu).tolist() == expected_lens
    assert plan.token_indices.tolist() == expected_indices
    assert plan.output_gather_sizes == [384, 44]
    assert plan.gathered_token_indices.tolist() == [
        *range(0, 256),
        *range(300, 428),
        *range(256, 300),
    ]


def test_attention_cp_plan_restores_original_packed_order(monkeypatch):
    plan = attention_module.AttentionCPPlan(
        token_indices=torch.tensor([1, 3]),
        output_gather_sizes=[2, 2],
        gathered_token_indices=torch.tensor([0, 2, 1, 3]),
        local_query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    group = FakeGroup(
        rank_in_group=1,
        world_size=2,
        all_gatherv_result=torch.tensor(
            [
                [0.0, 1.0],
                [20.0, 21.0],
                [10.0, 11.0],
                [30.0, 31.0],
            ]
        ),
    )
    monkeypatch.setattr(attention_module, "get_attn_cp_group", lambda: group)

    restored = plan.restore_output(
        torch.tensor([[10.0, 11.0], [30.0, 31.0]]),
        num_global_tokens=4,
    )

    torch.testing.assert_close(
        restored,
        torch.tensor(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        ),
    )
    assert group.all_gatherv_sizes == [2, 2]


def test_attention_cp_plan_supports_empty_shard(monkeypatch):
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
    gathered = torch.arange(256, dtype=torch.float32).reshape(128, 2)
    group = FakeGroup(rank_in_group=1, world_size=2, all_gatherv_result=gathered)
    monkeypatch.setattr(attention_module, "get_attn_cp_group", lambda: group)

    restored = plan.restore_output(torch.empty(0, 2), num_global_tokens=128)

    assert plan.num_local_tokens == 0
    assert plan.output_gather_sizes == [128, 0]
    torch.testing.assert_close(restored, gathered)


def test_attention_cp_projects_only_local_queries_before_full_kv_insert():
    plan = attention_module.AttentionCPPlan(
        token_indices=torch.tensor([1, 3]),
        output_gather_sizes=[2, 2],
        gathered_token_indices=torch.tensor([0, 2, 1, 3]),
        local_query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
    )
    qr = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kv = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    observed: dict[str, object] = {}

    def project(local_qr: torch.Tensor, qr_scale: None) -> torch.Tensor:
        observed["project_input"] = local_qr
        return local_qr[:, :2]

    def insert(
        q: torch.Tensor,
        full_kv: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        attn_metadata: object,
        *,
        split_q: bool,
    ) -> torch.Tensor:
        observed.update(
            q=q,
            kv=full_kv,
            q_positions=q_positions,
            kv_positions=kv_positions,
            split_q=split_q,
        )
        return q

    def sparse_and_attn(*args) -> None:
        observed["attention_q"] = args[4]

    layer = SimpleNamespace(
        indexer=None,
        compressor=None,
        aux_stream_list=None,
        n_local_heads=1,
        head_dim=2,
        _wq_b_proj=project,
        _fused_qnorm_rope_kv_insert=insert,
        _sparse_indexer_and_attn=sparse_and_attn,
    )

    with set_forward_context({}, make_forward_context_config()):
        attention_module.DeepseekV4Attention._prepare_and_attn(
            layer,
            hidden_states=torch.empty(4, 1),
            qr=qr,
            kv=kv,
            qr_scale=None,
            kv_score=torch.empty(4, 1),
            indexer_weights=torch.empty(4, 1),
            positions=positions,
            attn_out=torch.empty(2, 1, 2),
            cp_plan=plan,
        )

    torch.testing.assert_close(
        observed["project_input"],
        qr.index_select(0, plan.token_indices),
    )
    assert observed["kv"] is kv
    assert observed["kv_positions"] is positions
    assert observed["split_q"] is True
    assert observed["attention_q"] is observed["q"]


def test_attention_cp_profile_bypass_and_decode_rejection(monkeypatch):
    mixed_metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1)
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

    with set_forward_context(
        {"swa": mixed_metadata},
        make_forward_context_config(),
        is_profile=True,
    ):
        assert get_forward_context().is_profile
        assert (
            attention_module.DeepseekV4Attention._build_attention_cp_plan(
                layer, torch.device("cpu")
            )
            is None
        )

    with (
        set_forward_context({"swa": mixed_metadata}, make_forward_context_config()),
        pytest.raises(NotImplementedError, match="pure prefill"),
    ):
        attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer, torch.device("cpu")
        )


def test_attention_cp_plan_is_reused_within_forward_context(monkeypatch):
    query_start_loc_cpu = torch.tensor([0, 300, 428], dtype=torch.int32)
    prefill_seq_lens_cpu = torch.tensor([1000, 128], dtype=torch.int32)
    metadata = SimpleNamespace(
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

    with set_forward_context(
        {"swa_a": metadata, "swa_b": metadata},
        make_forward_context_config(),
    ):
        plan_a = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer_a, torch.device("cpu")
        )
        plan_b = attention_module.DeepseekV4Attention._build_attention_cp_plan(
            layer_b, torch.device("cpu")
        )

    assert plan_a is plan_b


def test_attention_cp_uses_local_q_and_full_kv_kernel_modes(monkeypatch):
    calls: list[tuple[str, tuple[int, ...], int]] = []

    def q_only(
        q,
        kv,
        cache,
        slots,
        positions,
        cos_sin,
        padded_heads,
        *args,
    ):
        assert cache.ndim == 2
        assert slots.numel() == 0
        assert positions.shape[0] == q.shape[0] == kv.shape[0]
        calls.append(("q", tuple(q.shape), slots.numel()))
        return q.new_zeros((q.shape[0], padded_heads, q.shape[-1]))

    def kv_only(kv, cache, slots, positions, *args):
        assert cache.ndim == 3
        assert slots.shape[0] == positions.shape[0] == kv.shape[0]
        calls.append(("kv", tuple(kv.shape), slots.numel()))

    monkeypatch.setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        q_only,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C,
        "fused_deepseek_v4_kv_rope_insert",
        kv_only,
        raising=False,
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_size=4,
    )
    layer = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            prefix="swa",
            kv_cache=torch.empty(2, 4, 584, dtype=torch.uint8),
        ),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(16, 64)),
        accepts_unnormed_unroped_query=False,
        n_local_heads=16,
        padded_heads=64,
        head_dim=512,
        eps=1e-6,
        kv_mxfp8=False,
    )
    q = torch.empty(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.empty(4, 512, dtype=torch.bfloat16)
    q_positions = torch.tensor([0, 1], dtype=torch.int64)
    kv_positions = torch.arange(4, dtype=torch.int64)

    output = attention_module.DeepseekV4Attention._fused_qnorm_rope_kv_insert(
        layer,
        q,
        kv,
        q_positions,
        kv_positions,
        {"swa": metadata},
        split_q=True,
    )

    assert output.shape == (2, 64, 512)
    assert calls == [("q", (2, 16, 512), 0), ("kv", (4, 512), 4)]
