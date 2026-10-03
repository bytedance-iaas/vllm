# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v41.attention as attention_module
import vllm.models.deepseek_v41.compressor as compressor_module
import vllm.models.deepseek_v41.nvidia.flashmla as flashmla_module
import vllm.models.deepseek_v41.sparse_mla as sparse_mla_module
import vllm.v1.attention.ops.pcp as pcp_ops
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.deepseek_v41.compressor import CompressorBackend
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLABackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@dataclass
class FakeGroup:
    rank_in_group: int
    world_size: int
    all_gather_result: torch.Tensor | None = None
    all_gather_input: torch.Tensor | None = None
    all_gatherv_result: torch.Tensor | None = None
    all_gatherv_sizes: list[int] | None = None

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        self.all_gather_input = tensor
        return tensor if self.all_gather_result is None else self.all_gather_result

    def all_gatherv(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        assert dim == 0
        self.all_gatherv_sizes = sizes
        return tensor if self.all_gatherv_result is None else self.all_gatherv_result


def test_dsv41_cache_backends_advertise_pcp_support():
    assert CompressorBackend.supports_pcp()
    assert DeepseekSparseSWABackend.supports_pcp()
    assert DeepseekV4FlashMLABackend.supports_pcp()


def test_pcp_cache_inputs_gather_only_partitioned_prefill(monkeypatch):
    class GatherGroup:
        world_size = 2

        def all_gather(self, tensor, dim=0):
            assert dim == 0
            return torch.cat((tensor, tensor + 100), dim=0)

    monkeypatch.setattr(pcp_ops, "get_pcp_group", GatherGroup)
    values = torch.tensor([10, 20, 21])
    positions = torch.tensor([0, 1, 2])
    slots = torch.arange(6)

    (cache_values, cache_positions), cache_slots = (
        pcp_ops.maybe_gather_pcp_cache_inputs(
            (values, positions),
            slots,
            num_decode_tokens=1,
            use_pcp=True,
        )
    )

    torch.testing.assert_close(cache_values, torch.tensor([10, 20, 21, 120, 121]))
    torch.testing.assert_close(cache_positions, torch.tensor([0, 1, 2, 101, 102]))
    torch.testing.assert_close(cache_slots, torch.tensor([0, 1, 2, 4, 5]))


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
    ("num_tokens", "tp_rank"),
    [
        (8, 2),
        (6, 3),
        (2, 3),
    ],
)
def test_attention_input_projection_shards_and_restores_token_rows(
    monkeypatch,
    num_tokens: int,
    tp_rank: int,
):
    hidden_states = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(
        num_tokens, 3
    )
    tp_size = 4
    shard_size = (num_tokens + tp_size - 1) // tp_size
    padded = hidden_states.new_zeros((shard_size * tp_size, 3))
    padded[:num_tokens].copy_(hidden_states)
    projected_inputs: list[torch.Tensor] = []

    def project(rows: torch.Tensor) -> tuple[torch.Tensor, None]:
        projected_inputs.append(rows)
        return projected(rows), None

    def projected(rows: torch.Tensor) -> torch.Tensor:
        return torch.cat((rows, rows + 100), dim=1)

    gathered = projected(padded)
    group = FakeGroup(
        rank_in_group=tp_rank,
        world_size=tp_size,
        all_gather_result=gathered,
    )
    monkeypatch.setattr(attention_module, "get_tp_group", lambda: group)
    layer = SimpleNamespace(fused_wqa_wkv=project)

    output = attention_module.DeepseekV4Attention._fused_wqa_wkv_gemm_token_sharded(
        layer, hidden_states
    )

    start = tp_rank * shard_size
    local_hidden_states = padded[start : start + shard_size]
    torch.testing.assert_close(projected_inputs[0], local_hidden_states)
    torch.testing.assert_close(
        group.all_gather_input,
        projected(local_hidden_states),
    )
    torch.testing.assert_close(output, projected(hidden_states))


def test_attention_input_projection_uses_replicated_path_when_disabled():
    projected_inputs: list[torch.Tensor] = []

    def project(rows: torch.Tensor) -> tuple[torch.Tensor, None]:
        projected_inputs.append(rows)
        return rows + 1, None

    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    layer = SimpleNamespace(
        fused_wqa_wkv=project,
        _can_shard_fused_wqa_wkv=lambda _: False,
    )

    output = attention_module.DeepseekV4Attention._fused_wqa_wkv_gemm(
        layer, hidden_states
    )

    assert projected_inputs[0] is hidden_states
    torch.testing.assert_close(output, hidden_states + 1)


def test_attention_input_projection_accepts_native_humming_mxfp8():
    from vllm.model_executor.kernels.linear.mxfp8.humming import (
        HummingMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTensorSym,
        kMxfp8Dynamic,
    )

    quant_method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    quant_method.kernel = HummingMxfp8LinearKernel.__new__(HummingMxfp8LinearKernel)
    layer = SimpleNamespace(
        fused_wqa_wkv=SimpleNamespace(
            quant_method=quant_method,
            params_dtype=torch.bfloat16,
        )
    )
    support_check = vars(attention_module.DeepseekV4Attention)[
        "_input_projection_quant_supports_token_sharding"
    ].func

    quant_method.spec = SimpleNamespace(activation=kMxfp8Dynamic)
    assert support_check(layer)

    quant_method.spec = SimpleNamespace(activation=kFp8DynamicTensorSym)
    assert not support_check(layer)


def test_attention_input_projection_guard_accepts_unpadded_v2_mask(monkeypatch):
    num_tokens = 2048
    layer = SimpleNamespace(
        input_proj_token_shard_enabled=True,
        input_proj_token_shard_min_tokens=num_tokens,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        _input_projection_quant_supports_token_sharding=True,
    )
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=num_tokens,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with set_forward_context(
        {"swa": metadata},
        make_forward_context_config(),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
    ):
        assert attention_module.DeepseekV4Attention._can_shard_fused_wqa_wkv(
            layer, torch.empty(num_tokens, 1)
        )
        metadata.num_prefill_tokens -= 1
        assert not attention_module.DeepseekV4Attention._can_shard_fused_wqa_wkv(
            layer, torch.empty(num_tokens, 1)
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


def test_attention_cp_plan_rotates_short_requests_across_ranks():
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=8,
        query_start_loc_cpu=torch.arange(9, dtype=torch.int32) * 64,
        prefill_seq_lens_cpu=torch.full((8,), 64, dtype=torch.int32),
    )

    plans = [
        attention_module.AttentionCPPlan.build(
            metadata,
            cp_rank=rank,
            cp_size=8,
            alignment=64,
            device=torch.device("cpu"),
        )
        for rank in range(8)
    ]

    assert [plan.num_local_tokens for plan in plans] == [64] * 8
    assert [plan.output_gather_sizes for plan in plans] == [[64] * 8] * 8
    all_indices = torch.cat([plan.token_indices for plan in plans]).sort().values
    torch.testing.assert_close(all_indices, torch.arange(512))


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


def test_flashmla_empty_pcp_shard_is_a_noop():
    layer = SimpleNamespace(
        compressed_cache_prefix=None,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
    )
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=0,
        num_prefill_tokens=0,
    )
    q = torch.empty(8, 1, 4)
    output = torch.full_like(q, float("nan"))

    with set_forward_context(
        {"swa": metadata},
        make_forward_context_config(),
    ):
        flashmla_module.DeepseekV4FlashMLAAttention.forward_mqa(
            layer,
            q,
            torch.empty(8, 2),
            torch.arange(8),
            output,
        )

    torch.testing.assert_close(output, torch.zeros_like(output))


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
        use_pcp=False,
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


@pytest.mark.parametrize("attn_metadata", [None, {"swa": SimpleNamespace()}])
def test_pcp_profile_bypasses_cache_gather(monkeypatch, attn_metadata):
    qr = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kv = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.arange(4, dtype=torch.int64)
    observed: dict[str, object] = {}

    def project(local_qr: torch.Tensor, qr_scale: None) -> torch.Tensor:
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
            kv=full_kv,
            kv_positions=kv_positions,
            attn_metadata=attn_metadata,
            split_q=split_q,
        )
        return q

    layer = SimpleNamespace(
        indexer=None,
        compressor=None,
        aux_stream_list=None,
        use_pcp=True,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        n_local_heads=1,
        head_dim=2,
        _wq_b_proj=project,
        _fused_qnorm_rope_kv_insert=insert,
        _sparse_indexer_and_attn=lambda *args: None,
    )
    monkeypatch.setattr(
        attention_module,
        "maybe_gather_pcp_cache_inputs",
        lambda *args, **kwargs: pytest.fail("profile must not gather PCP cache inputs"),
    )

    with set_forward_context(
        attn_metadata,
        make_forward_context_config(),
        is_profile=True,
    ):
        attention_module.DeepseekV4Attention._prepare_and_attn(
            layer,
            hidden_states=torch.empty(4, 1),
            qr=qr,
            kv=kv,
            qr_scale=None,
            kv_score=torch.empty(4, 1),
            indexer_weights=torch.empty(4, 1),
            positions=positions,
            attn_out=torch.empty(4, 1, 2),
            cp_plan=None,
        )

    assert observed["kv"] is kv
    assert observed["kv_positions"] is positions
    assert observed["attn_metadata"] is attn_metadata
    assert observed["split_q"] is True


def test_pcp_compressor_profile_bypasses_cache_gather(monkeypatch):
    latent = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.arange(4, dtype=torch.int64)
    compressor = SimpleNamespace(use_pcp=True)
    monkeypatch.setattr(
        attention_module,
        "maybe_gather_pcp_cache_inputs",
        lambda *args, **kwargs: pytest.fail("profile must not gather PCP cache inputs"),
    )

    with set_forward_context(
        {"cache": SimpleNamespace()},
        make_forward_context_config(),
        is_profile=True,
    ):
        cache_latent, cache_positions, slot_mapping = (
            attention_module.DeepseekCompressor.prepare_cache_inputs(
                compressor,
                latent,
                positions,
            )
        )

    assert cache_latent is latent
    assert cache_positions is positions
    assert slot_mapping is None


def test_pcp_compressor_uses_explicit_swa_decode_count(monkeypatch):
    latent = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    positions = torch.arange(4, dtype=torch.int64)
    slot_mapping = torch.arange(4, dtype=torch.int64)
    compressor = SimpleNamespace(use_pcp=True, k_cache_prefix="main")
    observed: dict[str, object] = {}

    def gather(tensors, slots, num_decode_tokens, use_pcp):
        observed.update(
            tensors=tensors,
            slots=slots,
            num_decode_tokens=num_decode_tokens,
            use_pcp=use_pcp,
        )
        return tensors, slots

    monkeypatch.setattr(compressor_module, "maybe_gather_pcp_cache_inputs", gather)

    with set_forward_context(
        {"main": SimpleNamespace(slot_mapping=slot_mapping)},
        make_forward_context_config(),
    ):
        cache_latent, cache_positions, cache_slots = (
            attention_module.DeepseekCompressor.prepare_cache_inputs(
                compressor,
                latent,
                positions,
                num_decode_tokens=3,
            )
        )

    gathered_tensors = observed["tensors"]
    assert isinstance(gathered_tensors, tuple)
    assert gathered_tensors[0] is latent
    assert gathered_tensors[1] is positions
    assert observed["slots"] is slot_mapping
    assert observed["num_decode_tokens"] == 3
    assert observed["use_pcp"] is True
    assert cache_latent is latent
    assert cache_positions is positions
    assert cache_slots is slot_mapping


@pytest.mark.parametrize("pcp_rank", range(4))
def test_ratio1_pcp_compressor_uses_rank_local_cache_mapping(monkeypatch, pcp_rank):
    kv_score = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    positions = torch.arange(8, dtype=torch.int64)
    global_slots = torch.full((32,), -1, dtype=torch.int64)
    global_slots[:8] = torch.arange(8, dtype=torch.int64)
    observed: dict[str, object] = {}
    compressor = SimpleNamespace(
        state_cache=None,
        k_cache_prefix="main",
        use_pcp=True,
        head_dim=2,
        norm=SimpleNamespace(weight=torch.ones(2)),
        rms_norm_eps=1e-6,
        compress_ratio=1,
    )

    monkeypatch.setattr(
        compressor_module,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=pcp_rank),
    )

    def save_compress(
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        rms_norm_eps,
        compress_ratio,
        latent,
    ):
        observed.update(
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            token_to_req_indices=token_to_req_indices,
        )
        latent.zero_()

    monkeypatch.setattr(compressor_module, "fused_save_compress_norm", save_compress)

    with set_forward_context(
        {"main": SimpleNamespace(slot_mapping=global_slots)},
        make_forward_context_config(),
    ):
        latent = attention_module.DeepseekCompressor.forward(
            compressor, kv_score, positions
        )

    assert latent.shape == (8, 2)
    rank_start = pcp_rank * positions.numel()
    torch.testing.assert_close(
        observed["slot_mapping"],
        global_slots[rank_start : rank_start + positions.numel()],
    )
    assert observed["state_cache"] is None
    assert observed["query_start_loc"] is None
    assert observed["token_to_req_indices"] is None


def test_ratio2_pcp_compressor_keeps_local_state_mapping(monkeypatch):
    kv_score = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    positions = torch.arange(8, dtype=torch.int64)
    local_slots = torch.arange(8, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    token_to_req_indices = torch.zeros(8, dtype=torch.int32)
    state_cache = torch.empty(1)
    observed: dict[str, object] = {}
    compressor = SimpleNamespace(
        state_cache=SimpleNamespace(prefix="state", kv_cache=state_cache),
        k_cache_prefix="main",
        use_pcp=True,
        head_dim=2,
        norm=SimpleNamespace(weight=torch.ones(2)),
        rms_norm_eps=1e-6,
        compress_ratio=2,
    )

    monkeypatch.setattr(
        compressor_module,
        "get_pcp_group",
        lambda: pytest.fail("ratio-2 state mapping is already rank-local"),
    )

    def save_compress(
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        rms_norm_eps,
        compress_ratio,
        latent,
    ):
        observed.update(
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            token_to_req_indices=token_to_req_indices,
        )
        latent.zero_()

    monkeypatch.setattr(compressor_module, "fused_save_compress_norm", save_compress)

    with set_forward_context(
        {
            "state": SimpleNamespace(
                slot_mapping=local_slots,
                query_start_loc=query_start_loc,
                token_to_req_indices=token_to_req_indices,
            )
        },
        make_forward_context_config(),
    ):
        latent = attention_module.DeepseekCompressor.forward(
            compressor, kv_score, positions
        )

    assert latent.shape == (8, 2)
    assert observed["state_cache"] is state_cache
    assert observed["slot_mapping"] is local_slots
    assert observed["query_start_loc"] is query_start_loc
    assert observed["token_to_req_indices"] is token_to_req_indices


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


@pytest.mark.parametrize(
    ("attention_cp_size", "pcp_size", "expected_threshold"),
    [(2, 1, 0), (1, 4, 0), (1, 1, 1)],
)
def test_context_parallel_keeps_short_prompt_tails_on_prefill_path(
    monkeypatch, attention_cp_size, pcp_size, expected_threshold
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            attention_context_parallel_size=attention_cp_size,
            prefill_context_parallel_size=pcp_size,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 1, 2])
        ),
    )

    def initialize_base(builder, *args, **kwargs):
        builder.vllm_config = config
        builder.decode_threshold = 1

    monkeypatch.setattr(
        sparse_mla_module.DeepseekSparseSWAMetadataBuilder,
        "__init__",
        initialize_base,
    )
    monkeypatch.setattr(
        sparse_mla_module,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )

    builder = sparse_mla_module.DeepseekV41SparseSWAMetadataBuilder()

    assert builder.decode_threshold == expected_threshold


def test_pcp_swa_metadata_uses_local_slots_and_keeps_cache_write_view(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            attention_context_parallel_size=1,
            prefill_context_parallel_size=4,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 1, 2])
        ),
    )
    observed: dict[str, torch.Tensor] = {}

    def initialize_base(builder, *args, **kwargs):
        builder.vllm_config = config
        builder.decode_threshold = 1

    def build_base(
        builder,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        replay_start=None,
    ):
        observed["slot_mapping"] = common_attn_metadata.slot_mapping
        return SimpleNamespace(cache_slot_mapping=None)

    monkeypatch.setattr(
        sparse_mla_module.DeepseekSparseSWAMetadataBuilder,
        "__init__",
        initialize_base,
    )
    monkeypatch.setattr(
        sparse_mla_module.DeepseekSparseSWAMetadataBuilder,
        "build",
        build_base,
    )
    monkeypatch.setattr(
        sparse_mla_module,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=2),
    )

    slots = torch.arange(16, dtype=torch.int64)
    common = SimpleNamespace(slot_mapping=slots)
    common.replace = lambda **kwargs: SimpleNamespace(**kwargs)
    metadata = sparse_mla_module.DeepseekV41SparseSWAMetadataBuilder().build(0, common)

    torch.testing.assert_close(observed["slot_mapping"], slots[8:12])
    assert metadata.cache_slot_mapping.data_ptr() == slots.data_ptr()


def test_cp1_keeps_opaque_quantized_query_path():
    class OpaqueQuery:
        pass

    query = OpaqueQuery()
    observed: dict[str, object] = {}

    def project(local_qr, qr_scale):
        observed["query"] = local_qr
        return torch.empty(4, 1, 2)

    def insert(q, kv, q_positions, kv_positions, metadata, *, split_q):
        observed["split_q"] = split_q
        return q

    def sparse_and_attn(*args):
        observed["attention_q"] = args[4]

    layer = SimpleNamespace(
        indexer=None,
        compressor=None,
        aux_stream_list=None,
        use_pcp=False,
        n_local_heads=1,
        head_dim=2,
        _wq_b_proj=project,
        _fused_qnorm_rope_kv_insert=insert,
        _sparse_indexer_and_attn=sparse_and_attn,
    )
    positions = torch.arange(4, dtype=torch.int64)

    with set_forward_context({}, make_forward_context_config()):
        attention_module.DeepseekV4Attention._prepare_and_attn(
            layer,
            hidden_states=torch.empty(4, 1),
            qr=query,
            kv=torch.empty(4, 2),
            qr_scale=torch.empty(1),
            kv_score=torch.empty(4, 1),
            indexer_weights=torch.empty(4, 1),
            positions=positions,
            attn_out=torch.empty(4, 1, 2),
            cp_plan=None,
        )

    assert observed["query"] is query
    assert observed["split_q"] is False
