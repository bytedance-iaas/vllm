# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-safe guard tests for DeepSeek V4 SM90/SM100 MegaMoE experts.

These tests intentionally avoid CUDA so they run in CPU CI. They cover the
loader-side parameter shapes and FP8 scale sharding logic, which are pure
PyTorch/host operations and do not require a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm_utils
from vllm.forward_context import override_forward_context
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MegaMoEExperts,
    DeepseekV4MoE,
)
from vllm.utils.torch_utils import _encode_layer_name


def _make_vllm_config(
    max_num_batched_tokens: int = 4,
    max_num_seqs: int = 4,
    num_speculative_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        ),
    )


def _make_fp8_experts(
    hidden_size: int = 256,
    intermediate_size: int = 256,
    num_experts: int = 4,
    num_local_experts: int = 2,
    experts_start_idx: int = 2,
    top_k: int = 2,
) -> DeepseekV4MegaMoEExperts:
    return DeepseekV4MegaMoEExperts(
        _make_vllm_config(),
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="model.layers.0.ffn.experts",
        expert_dtype="fp8",
    )


def _make_fp4_experts(
    hidden_size: int = 256,
    intermediate_size: int = 256,
    num_experts: int = 4,
    num_local_experts: int = 2,
    experts_start_idx: int = 2,
    top_k: int = 2,
) -> DeepseekV4MegaMoEExperts:
    return DeepseekV4MegaMoEExperts(
        _make_vllm_config(),
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="model.layers.0.ffn.experts",
        expert_dtype="fp4",
    )


def test_resolve_mega_moe_decode_capacity_defaults_to_decode_capacity():
    cfg = _make_vllm_config(max_num_batched_tokens=512, max_num_seqs=16)
    assert DeepseekV4MegaMoEExperts._resolve_mega_moe_decode_capacity(cfg) == 16


def test_resolve_mega_moe_decode_capacity_accounts_for_spec_decode():
    cfg = _make_vllm_config(
        max_num_batched_tokens=256,
        max_num_seqs=16,
        num_speculative_tokens=4,
    )
    assert DeepseekV4MegaMoEExperts._resolve_mega_moe_decode_capacity(cfg) == 80


def test_resolve_mega_moe_decode_capacity_default_clamped_to_batched():
    cfg = _make_vllm_config(
        max_num_batched_tokens=64,
        max_num_seqs=16,
        num_speculative_tokens=4,
    )
    assert DeepseekV4MegaMoEExperts._resolve_mega_moe_decode_capacity(cfg) == 64


def test_resolve_mega_moe_decode_capacity_accounts_for_sequence_parallel():
    cfg = _make_vllm_config(
        max_num_batched_tokens=256,
        max_num_seqs=16,
        num_speculative_tokens=4,
    )
    assert (
        DeepseekV4MegaMoEExperts._resolve_mega_moe_decode_capacity(
            cfg,
            sequence_parallel_size=4,
        )
        == 20
    )


def test_get_symm_buffer_for_num_tokens_uses_decode_buffer(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    experts._capacity_buffers = None
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256
    calls = []

    def fake_get_symm_buffer(max_num_tokens=None, *, cache=True):
        calls.append((max_num_tokens, cache))
        return object()

    monkeypatch.setattr(experts, "get_symm_buffer", fake_get_symm_buffer)

    experts.get_symm_buffer_for_num_tokens(16)

    assert calls == [(None, True)]


def test_get_symm_buffer_for_num_tokens_uses_cached_full_capacity_buffer(
    monkeypatch,
):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    experts._capacity_buffers = None
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256
    calls = []

    def fake_get_symm_buffer(max_num_tokens=None, *, cache=True):
        calls.append((max_num_tokens, cache))
        return object()

    monkeypatch.setattr(experts, "get_symm_buffer", fake_get_symm_buffer)

    experts.get_symm_buffer_for_num_tokens(256)

    assert calls == [(256, True)]


def test_get_symm_buffer_for_num_tokens_rounds_oversized_to_full_capacity(
    monkeypatch,
):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    experts._capacity_buffers = None
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256
    calls = []

    def fake_get_symm_buffer(max_num_tokens=None, *, cache=True):
        calls.append((max_num_tokens, cache))
        return object()

    monkeypatch.setattr(experts, "get_symm_buffer", fake_get_symm_buffer)

    experts.get_symm_buffer_for_num_tokens(128)

    assert calls == [(256, True)]


def test_get_symm_buffer_for_num_tokens_uses_dp_wide_capacity(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    experts._capacity_buffers = None
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256
    calls = []

    def fake_get_symm_buffer(max_num_tokens=None, *, cache=True):
        calls.append((max_num_tokens, cache))
        return object()

    monkeypatch.setattr(experts, "get_symm_buffer", fake_get_symm_buffer)
    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([16, 128], dtype=torch.int32)
    )
    forward_context = SimpleNamespace(dp_metadata=dp_metadata)

    with override_forward_context(forward_context):
        experts.get_symm_buffer_for_num_tokens(16)

    assert calls == [(256, True)]


def test_get_max_num_tokens_across_dp_localizes_sequence_parallel():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.sequence_parallel_size = 4
    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([16, 65], dtype=torch.int32)
    )

    with override_forward_context(SimpleNamespace(dp_metadata=dp_metadata)):
        assert experts._get_max_num_tokens_across_dp(8) == 17


def test_get_symm_buffer_for_num_tokens_rejects_beyond_batched():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    experts._capacity_buffers = None
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256

    with pytest.raises(ValueError):
        experts.get_symm_buffer_for_num_tokens(257)


def test_get_requested_capacity_buckets_uses_deep_gemm_c_accessor():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.max_num_tokens = 384
    experts.max_num_batched_tokens = 2048
    deep_gemm = SimpleNamespace(
        _C=SimpleNamespace(get_token_alignment_for_sm90_mega_moe=lambda: 384)
    )

    assert experts._get_requested_capacity_buckets(deep_gemm) == (
        384,
        768,
        2048,
    )


def test_get_requested_capacity_buckets_prefers_top_level_accessor():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.max_num_tokens = 384
    experts.max_num_batched_tokens = 2048
    deep_gemm = SimpleNamespace(
        get_token_alignment_for_sm90_mega_moe=lambda: 384,
        _C=SimpleNamespace(
            get_token_alignment_for_sm90_mega_moe=lambda: pytest.fail(
                "top-level accessor should win when available"
            )
        ),
    )

    assert experts._get_requested_capacity_buckets(deep_gemm) == (
        384,
        768,
        2048,
    )


def test_get_requested_capacity_buckets_stays_disabled_without_accessor():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.max_num_tokens = 384
    experts.max_num_batched_tokens = 2048

    assert experts._get_requested_capacity_buckets(SimpleNamespace()) == ()


@pytest.mark.parametrize(
    ("num_tokens", "expected_bucket"),
    [
        (0, "decode"),
        (384, "decode"),
        (385, "mid"),
        (767, "mid"),
        (768, "mid"),
        (769, "full"),
        (2048, "full"),
    ],
)
def test_get_symm_buffer_for_num_tokens_uses_first_prepared_bucket(
    monkeypatch, num_tokens, expected_bucket
):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = True
    experts.max_num_batched_tokens = 2048
    experts._capacity_buffers = (
        (384, "decode"),
        (768, "mid"),
        (2048, "full"),
    )
    monkeypatch.setattr(
        experts,
        "_get_max_num_tokens_across_dp",
        lambda tokens: tokens,
    )
    monkeypatch.setattr(
        experts,
        "get_symm_buffer",
        lambda *args, **kwargs: pytest.fail(
            "serving path should not allocate lazily after preparation"
        ),
    )

    assert experts.get_symm_buffer_for_num_tokens(num_tokens) == expected_bucket


def test_get_symm_buffer_for_num_tokens_requires_prepared_buckets():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = True
    experts._capacity_buffers = None
    experts.max_num_batched_tokens = 2048
    experts.max_num_tokens = 384

    with pytest.raises(RuntimeError, match="not prepared during initialization"):
        experts.get_symm_buffer_for_num_tokens(1)


def test_get_symm_buffer_for_num_tokens_rejects_beyond_batched_with_buckets(
    monkeypatch,
):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = True
    experts._capacity_buffers = (
        (384, "decode"),
        (768, "mid"),
        (2048, "full"),
    )
    experts.max_num_batched_tokens = 2048
    monkeypatch.setattr(
        experts,
        "_get_max_num_tokens_across_dp",
        lambda tokens: tokens,
    )

    with pytest.raises(ValueError):
        experts.get_symm_buffer_for_num_tokens(2049)


def test_prepare_capacity_buckets_allocates_and_prewarms_in_order(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._capacity_buffers = None
    experts._use_prepared_capacity_buckets = False
    experts._transformed_l1_weights = (torch.empty(1), torch.empty(1))
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        experts,
        "_supports_prepared_capacity_buckets",
        lambda: True,
    )
    monkeypatch.setattr(
        experts,
        "_get_requested_capacity_buckets",
        lambda deep_gemm: (384, 768, 2048),
    )

    buffers = {
        384: SimpleNamespace(num_max_tokens_per_rank=384),
        768: SimpleNamespace(num_max_tokens_per_rank=768),
        2048: SimpleNamespace(num_max_tokens_per_rank=2304),
    }

    def fake_get_symm_buffer(max_num_tokens=None, *, cache=True):
        assert cache is True
        calls.append(("alloc", max_num_tokens))
        return buffers[max_num_tokens]

    monkeypatch.setattr(experts, "get_symm_buffer", fake_get_symm_buffer)
    monkeypatch.setattr(
        experts,
        "_prewarm_capacity_bucket",
        lambda requested_capacity, symm_buffer, **kwargs: calls.append(
            ("warm", requested_capacity)
        ),
    )
    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: SimpleNamespace(),
    )

    experts.prepare_capacity_buckets(activation_clamp=7.5)

    assert calls == [
        ("alloc", 384),
        ("alloc", 768),
        ("alloc", 2048),
        ("warm", 384),
        ("warm", 768),
        ("warm", 2048),
    ]
    assert experts._capacity_buffers == (
        (384, buffers[384]),
        (768, buffers[768]),
        (2048, buffers[2048]),
    )
    assert experts._use_prepared_capacity_buckets is True


def test_prepare_capacity_buckets_is_idempotent(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_prepared_capacity_buckets = False
    prepared = ((384, object()), (768, object()), (2048, object()))
    experts._capacity_buffers = prepared
    monkeypatch.setattr(
        experts,
        "_supports_prepared_capacity_buckets",
        lambda: pytest.fail("should not re-check once prepared"),
    )

    experts.prepare_capacity_buckets(activation_clamp=None)

    assert experts._capacity_buffers is prepared


def test_prepare_capacity_buckets_stays_disabled_when_physical_caps_mismatch(
    monkeypatch,
):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._capacity_buffers = None
    experts._use_prepared_capacity_buckets = False
    experts._transformed_l1_weights = (torch.empty(1), torch.empty(1))

    monkeypatch.setattr(
        experts,
        "_supports_prepared_capacity_buckets",
        lambda: True,
    )
    monkeypatch.setattr(
        experts,
        "_get_requested_capacity_buckets",
        lambda deep_gemm: (384, 768, 2048),
    )
    monkeypatch.setattr(
        experts,
        "get_symm_buffer",
        lambda max_num_tokens=None, *, cache=True: SimpleNamespace(
            num_max_tokens_per_rank=max_num_tokens
        ),
    )
    monkeypatch.setattr(
        experts,
        "_prewarm_capacity_bucket",
        lambda *args, **kwargs: pytest.fail("should not prewarm mismatched buckets"),
    )
    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: SimpleNamespace(),
    )

    experts.prepare_capacity_buckets(activation_clamp=None)

    assert experts._capacity_buffers is None
    assert experts._use_prepared_capacity_buckets is False


def test_fp8_loader_params_have_expected_shapes_and_dtypes():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    assert experts.expert_dtype == "fp8"
    # FP8 path must not allocate the FP4/UE8M0 packed scale params.
    assert not hasattr(experts, "w13_weight_scale")
    assert not hasattr(experts, "w2_weight_scale")

    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight.shape == (2, 2 * intermediate_size, hidden_size)
    assert experts.w2_weight.dtype == torch.float8_e4m3fn
    assert experts.w2_weight.shape == (2, hidden_size, intermediate_size)

    scale_n = (intermediate_size + 127) // 128
    scale_h = (hidden_size + 127) // 128
    assert experts.w13_weight_scale_inv.dtype == torch.float32
    assert experts.w13_weight_scale_inv.shape == (2, 2 * scale_n, scale_h)
    assert experts.w2_weight_scale_inv.dtype == torch.float32
    assert experts.w2_weight_scale_inv.shape == (2, scale_h, scale_n)


def test_fp8_weight_loader_packs_w1_w3_and_w2():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    # Non-local expert (id=1 is not owned by experts_start_idx=2 rank) must be
    # rejected and leave the local data untouched.
    nonlocal_w1 = torch.ones(intermediate_size, hidden_size, dtype=torch.float8_e4m3fn)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_w1,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((intermediate_size, hidden_size), 2.0, dtype=torch.float8_e4m3fn)
    w3 = torch.full((intermediate_size, hidden_size), 3.0, dtype=torch.float8_e4m3fn)
    w2 = torch.full((hidden_size, intermediate_size), 4.0, dtype=torch.float8_e4m3fn)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :intermediate_size], w1)
    assert torch.equal(experts.w13_weight[0, intermediate_size:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    # Second local expert (global id 3) is untouched.
    assert torch.count_nonzero(experts.w13_weight[1].float()) == 0


def test_fp8_weight_loader_shards_scales_by_block_count():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    scale_n = (intermediate_size + 127) // 128
    scale_h = (hidden_size + 127) // 128

    w1_sf = torch.full((scale_n, scale_h), 0.5, dtype=torch.float32)
    w3_sf = torch.full((scale_n, scale_h), 0.25, dtype=torch.float32)
    w2_sf = torch.full((scale_h, scale_n), 0.125, dtype=torch.float32)

    assert experts.weight_loader(
        experts.w13_weight_scale_inv,
        w1_sf,
        "experts.w13_weight_scale_inv",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight_scale_inv,
        w3_sf,
        "experts.w13_weight_scale_inv",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight_scale_inv,
        w2_sf,
        "experts.w2_weight_scale_inv",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight_scale_inv[0, :scale_n], w1_sf)
    assert torch.equal(experts.w13_weight_scale_inv[0, scale_n:], w3_sf)
    assert torch.equal(experts.w2_weight_scale_inv[0], w2_sf)
    assert torch.count_nonzero(experts.w13_weight_scale_inv[1]) == 0


def test_fp4_loader_params_unchanged():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp4_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    assert experts.expert_dtype == "fp4"
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w13_weight.shape == (2, 2 * intermediate_size, hidden_size // 2)
    assert experts.w13_weight_scale.dtype == torch.uint8
    assert experts.w13_weight_scale.shape == (
        2,
        2 * intermediate_size,
        hidden_size // 32,
    )
    assert not hasattr(experts, "w13_weight_scale_inv")
    assert not hasattr(experts, "w2_weight_scale_inv")


def test_sm90_finalize_passes_fp8_weights_to_deep_gemm(monkeypatch):
    experts = _make_fp8_experts()

    class FakeDeepGemm:
        def transform_sf_into_required_layout(
            self,
            scale,
            rows,
            cols,
            block_shape,
            num_experts,
            *,
            disable_ue8m0_cast=False,
        ):
            assert scale.dtype == torch.float32
            assert block_shape == (128, 128)
            assert num_experts == experts.num_local_experts
            assert disable_ue8m0_cast is True
            return scale

        def transform_weights_for_mega_moe_sm90(self, l1_weight, l2_weight):
            w13, w13_sf = l1_weight
            w2, w2_sf = l2_weight

            assert w13.dtype == torch.float8_e4m3fn
            assert w2.dtype == torch.float8_e4m3fn
            assert w13.is_contiguous()
            assert w2.is_contiguous()
            assert w13_sf.dtype == torch.float32
            assert w2_sf.dtype == torch.float32
            return (w13, w13_sf), (w2, w2_sf)

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )
    experts._use_sm90_mega_moe = True

    experts._finalize_weights_sm90()

    assert experts._transformed_l1_weights[0].dtype == torch.float8_e4m3fn
    assert experts._transformed_l1_weights[1].dtype == torch.float32
    assert experts._transformed_l2_weights[0].dtype == torch.float8_e4m3fn
    assert experts._transformed_l2_weights[1].dtype == torch.float32


def test_sm90_finalize_passes_fp4_weights_to_deep_gemm(monkeypatch):
    experts = _make_fp4_experts()
    experts.w13_weight_scale.data.fill_(127)
    experts.w2_weight_scale.data.fill_(126)

    class FakeDeepGemm:
        def transform_weights_for_mega_moe_sm90_fp4(self, l1_weight, l2_weight):
            w13, w13_sf = l1_weight
            w2, w2_sf = l2_weight

            assert w13.dtype == torch.int8
            assert w2.dtype == torch.int8
            assert w13.is_contiguous()
            assert w2.is_contiguous()
            assert w13.shape == (
                experts.num_local_experts,
                2 * experts.intermediate_size,
                experts.hidden_size // 2,
            )
            assert w2.shape == (
                experts.num_local_experts,
                experts.hidden_size,
                experts.intermediate_size // 2,
            )
            assert w13_sf.dtype == torch.float32
            assert w2_sf.dtype == torch.float32
            assert torch.all(w13_sf == 1.0)
            assert torch.all(w2_sf == 0.5)
            return (w13, w13_sf), (w2, w2_sf)

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )
    experts._use_sm90_mega_moe = True
    experts._use_sm90_fp4_mega_moe = True

    experts._finalize_weights_sm90()

    assert experts._transformed_l1_weights[0].dtype == torch.int8
    assert experts._transformed_l1_weights[1].dtype == torch.float32
    assert experts._transformed_l2_weights[0].dtype == torch.int8
    assert experts._transformed_l2_weights[1].dtype == torch.float32


def test_sm90_mega_moe_uses_unfused_gate_above_threshold(monkeypatch):
    calls = []

    class FakeGate(torch.nn.Module):
        weight = torch.empty(384, 128)
        tid2eid = None
        e_score_correction_bias = None
        bias_vl = None

        def forward(self, hidden_states):
            calls.append("gate")
            return torch.empty(hidden_states.shape[0], 384), None

    class FakeExperts(torch.nn.Module):
        def forward(self, hidden_states, topk_weights, topk_ids, **kwargs):
            calls.append("experts")
            return hidden_states.clone()

    def fake_fused_topk_bias(**kwargs):
        calls.append("topk")
        num_tokens = kwargs["hidden_states"].shape[0]
        return torch.ones(num_tokens, 2), torch.zeros(num_tokens, 2, dtype=torch.int64)

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.use_fused_mega_gate = False
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = None
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 2
    moe.renormalize = True
    moe.hash_indices_dtype = torch.int64
    moe.routed_scaling_factor = 1.0
    moe.swiglu_limit = 10.0
    moe.image_sentinel_lo = 0
    monkeypatch.setattr(dsv4_model, "fused_topk_bias", fake_fused_topk_bias)

    hidden_states = torch.zeros(17, 128)
    output = moe(hidden_states)

    assert calls == ["gate", "topk", "experts"]
    torch.testing.assert_close(output, hidden_states)


@pytest.mark.parametrize(
    ("expert_dtype", "missing_symbol"),
    [
        ("fp4", "transform_weights_for_mega_moe_sm90_fp4"),
        ("fp8", "transform_weights_for_mega_moe_sm90"),
    ],
)
def test_sm90_runtime_guard_reports_missing_symbols(
    monkeypatch, expert_dtype, missing_symbol
):
    experts = _make_fp4_experts() if expert_dtype == "fp4" else _make_fp8_experts()
    del experts.w13_weight
    experts.w13_weight = SimpleNamespace(device=torch.device("cuda"))

    available_symbols = {
        "get_symm_buffer_for_mega_moe": object(),
        "fp8_fp4_mega_moe": object(),
        "fp8_mega_moe": object(),
        "transform_sf_into_required_layout": object(),
    }
    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: SimpleNamespace(**available_symbols),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    with pytest.raises(NotImplementedError, match=missing_symbol):
        experts._check_runtime_supported()


def test_symm_buffer_cache_separates_architecture_and_dtype_modes(monkeypatch):
    calls = []

    class FakeDeepGemm:
        def get_symm_buffer_for_mega_moe(self, *args, **kwargs):
            calls.append(kwargs)
            return object()

    group = SimpleNamespace(device_group=object())
    monkeypatch.setattr(deep_gemm_utils, "_import_deep_gemm", FakeDeepGemm)
    monkeypatch.setattr(dsv4_model, "get_ep_group", lambda: group)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(DeepseekV4MegaMoEExperts, "_symm_buffer_cache", {})

    sm100 = _make_fp4_experts()
    sm90_fp4 = _make_fp4_experts()
    sm90_fp4._use_sm90_mega_moe = True
    sm90_fp4._use_sm90_fp4_mega_moe = True
    sm90_fp8 = _make_fp8_experts()
    sm90_fp8._use_sm90_mega_moe = True
    sm90_fp8._use_sm90_fp8_mega_moe = True

    assert sm100.get_symm_buffer() is sm100.get_symm_buffer()
    sm90_fp4.get_symm_buffer()
    sm90_fp8.get_symm_buffer()

    assert calls == [
        {},
        {"use_fp8_dispatch": True, "activation": "swiglu"},
        {"use_fp8_dispatch": True, "activation": "swiglu"},
    ]


@pytest.mark.parametrize(
    ("peer_tokens", "expected_calls"),
    [
        (0, 0),
        (1, 1),
    ],
)
def test_empty_local_rank_only_exits_when_work_is_globally_empty(
    monkeypatch, peer_tokens, expected_calls
):
    experts = _make_fp4_experts()
    calls = []

    monkeypatch.setattr(
        experts,
        "_get_max_num_tokens_across_dp",
        lambda num_tokens: peer_tokens,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "deepseek_v4_mega_moe_experts",
        lambda *args: calls.append(args),
    )

    hidden_states = torch.empty(0, experts.hidden_size)
    topk_weights = torch.empty(0, experts.top_k)
    topk_ids = torch.empty(0, experts.top_k, dtype=torch.int64)

    output = experts(
        hidden_states,
        topk_weights,
        topk_ids,
        activation_clamp=None,
    )

    assert output.shape == hidden_states.shape
    assert len(calls) == expected_calls


def test_mega_moe_custom_op_resolves_encoded_layer_name():
    calls = []
    layer = SimpleNamespace(_run_mega_moe=lambda *args: calls.append(args))
    context = SimpleNamespace(no_compile_layers={"model.layers.0.ffn.experts": layer})
    hidden_states = torch.empty(1, 4)
    topk_weights = torch.empty(1, 2)
    topk_ids = torch.empty(1, 2, dtype=torch.int64)
    output = torch.empty_like(hidden_states)

    with override_forward_context(context):
        dsv4_model._deepseek_v4_mega_moe_experts_op(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
            _encode_layer_name("model.layers.0.ffn.experts"),
            None,
            True,
        )

    assert len(calls) == 1
    args = calls[0]
    assert args[0] is hidden_states
    assert args[1] is topk_weights
    assert args[2] is topk_ids
    assert args[3] is output
    assert args[4:] == (None, True)


def test_sm90_fp8_eplb_exposes_transformed_weight_and_scale_tensors():
    experts = _make_fp8_experts()
    l1_weight = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    l1_scale = torch.arange(24, dtype=torch.float32).view(2, 4, 3).transpose(1, 2)
    l2_weight = torch.arange(40, dtype=torch.float32).view(2, 4, 5)
    l2_scale = torch.arange(40, dtype=torch.float32).view(2, 5, 4).transpose(1, 2)
    experts._transformed_l1_weights = (l1_weight, l1_scale)
    experts._transformed_l2_weights = (l2_weight, l2_scale)
    experts._use_sm90_fp8_mega_moe = True

    eplb_weights = experts.get_expert_weights()

    assert len(eplb_weights) == 4
    assert torch.equal(eplb_weights[0], l1_weight.view(2, -1))
    assert torch.equal(eplb_weights[1], l1_scale.transpose(1, 2).view(2, -1))
    assert torch.equal(eplb_weights[2], l2_weight.view(2, -1))
    assert torch.equal(eplb_weights[3], l2_scale.transpose(1, 2).view(2, -1))


def test_sm90_fp8_dispatch_preserves_weight_scale_pairs(monkeypatch):
    experts = _make_fp8_experts()
    l1_weights = (object(), object())
    l2_weights = (object(), object())
    experts._transformed_l1_weights = l1_weights
    experts._transformed_l2_weights = l2_weights
    experts._use_sm90_mega_moe = True
    experts._use_sm90_fp8_mega_moe = True

    symm_buffer = SimpleNamespace(
        x=object(),
        x_sf=object(),
        topk_idx=object(),
        topk_weights=object(),
    )
    monkeypatch.setattr(
        experts,
        "get_symm_buffer_for_num_tokens",
        lambda num_tokens: symm_buffer,
    )
    monkeypatch.setattr(
        dsv4_model,
        "prepare_megamoe_inputs_sm90",
        lambda *args, **kwargs: None,
    )

    calls = []

    class FakeDeepGemm:
        def fp8_mega_moe(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )

    experts._run_mega_moe_sm90(
        torch.empty(1, experts.hidden_size),
        torch.empty(1, experts.top_k),
        torch.empty(1, experts.top_k, dtype=torch.int64),
        torch.empty(1, experts.hidden_size),
        activation_clamp=None,
        fast_math=True,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1] is l1_weights
    assert args[2] is l2_weights
    assert kwargs["recipe"] == (128, 128, 128)
