# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-safe guard tests for DeepSeek V4 SM90/SM100 MegaMoE experts.

These tests intentionally avoid CUDA so they run in CPU CI. They cover the
loader-side parameter shapes and FP8 scale sharding logic, which are pure
PyTorch/host operations and do not require a GPU.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm_utils
from vllm.forward_context import override_forward_context
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts


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


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 8, 9, 31, 32, 33, 383, 384])
@pytest.mark.parametrize("shape_tail", [(), (3,), (2, 2)])
def test_shard_tp_token_rows_preserves_order_and_marks_padding(
    num_tokens: int,
    shape_tail: tuple[int, ...],
):
    tp_size = 8
    shape = (num_tokens, *shape_tail)
    tensor = torch.arange(max(1, math.prod(shape)), dtype=torch.int64)[
        : math.prod(shape)
    ]
    tensor = tensor.reshape(shape)

    shards = []
    valid_counts = []
    for tp_rank in range(tp_size):
        shard, valid_count = dsv4_model._shard_tp_token_rows(
            tensor,
            tp_rank,
            tp_size,
            padding_value=-1,
        )
        shards.append(shard)
        valid_counts.append(valid_count)

        if valid_count < shard.shape[0]:
            assert torch.all(shard[valid_count:] == -1)

    reconstructed = torch.cat(shards, dim=0)[:num_tokens]
    assert torch.equal(reconstructed, tensor)
    assert sum(valid_counts) == num_tokens
    assert len(set(shard.shape[0] for shard in shards)) == 1


@pytest.mark.parametrize(
    "num_tokens,tp_rank,expected",
    [
        (9, 0, [False, True]),
        (9, 1, [False, False]),
        (9, 4, [False, True]),
        (1, 0, [False]),
        (1, 7, [True]),
    ],
)
def test_local_tp_padding_mask_combines_scheduler_and_shard_padding(
    num_tokens: int,
    tp_rank: int,
    expected: list[bool],
):
    tp_size = 8
    global_padding_mask = torch.zeros(num_tokens, dtype=torch.bool)
    if num_tokens > 1:
        global_padding_mask[1] = True
    local_input, valid_num_tokens = dsv4_model._shard_tp_token_rows(
        torch.zeros(num_tokens, 2),
        tp_rank,
        tp_size,
    )

    actual = dsv4_model._local_tp_padding_mask(
        global_padding_mask,
        local_input,
        valid_num_tokens,
        tp_rank,
        tp_size,
    )

    assert actual.tolist() == expected


def test_local_tp_padding_mask_skips_allocation_without_padding():
    local_input = torch.zeros(3, 2)

    actual = dsv4_model._local_tp_padding_mask(
        None,
        local_input,
        valid_num_tokens=3,
        tp_rank=0,
        tp_size=8,
    )

    assert actual is None


def test_local_tp_padding_mask_marks_tp_tail_without_scheduler_padding():
    local_input = torch.zeros(2, 2)

    actual = dsv4_model._local_tp_padding_mask(
        None,
        local_input,
        valid_num_tokens=1,
        tp_rank=4,
        tp_size=8,
    )

    assert actual is not None
    assert actual.tolist() == [False, True]


@pytest.mark.parametrize(
    "num_tokens,tp_rank,expected_padding",
    [
        (8, 3, None),
        (9, 4, [False, True]),
        (1, 7, [True]),
    ],
)
def test_run_mega_moe_tp_dedup_restores_routed_output(
    monkeypatch,
    num_tokens: int,
    tp_rank: int,
    expected_padding: list[bool] | None,
):
    tp_size = 8
    hidden_states = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(
        num_tokens, 2
    )
    input_ids = torch.arange(num_tokens, dtype=torch.int64)
    out = torch.empty_like(hidden_states)

    class FakeGate(torch.nn.Module):
        e_score_correction_bias = None
        tid2eid = None

        def forward(self, local_input):
            return torch.zeros(local_input.shape[0], 4), None

    class FakeExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.padding = None

        def forward(
            self,
            local_input,
            topk_weights,
            topk_ids,
            *,
            activation_clamp,
            is_padding,
        ):
            self.padding = None if is_padding is None else is_padding.tolist()
            return local_input + 10

    padded_tokens = math.ceil(num_tokens / tp_size) * tp_size
    padding = torch.zeros(padded_tokens - num_tokens, 2)
    gathered_output = torch.cat((hidden_states, padding), dim=0) + 10

    class FakeGroup:
        def all_gather(self, local_output, dim):
            assert dim == 0
            expected_local, _ = dsv4_model._shard_tp_token_rows(
                hidden_states,
                tp_rank,
                tp_size,
            )
            assert torch.equal(local_output, expected_local + 10)
            return gathered_output

    routed_input_ids = []

    def fake_fused_topk_bias(**kwargs):
        routed_input_ids.append(kwargs["input_tokens"].clone())
        return (
            torch.ones(kwargs["hidden_states"].shape[0], 2),
            torch.zeros(kwargs["hidden_states"].shape[0], 2, dtype=torch.int64),
        )

    monkeypatch.setattr(dsv4_model, "fused_topk_bias", fake_fused_topk_bias)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")

    moe = object.__new__(dsv4_model.DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.tp_rank = tp_rank
    moe.tp_size = tp_size
    moe.tp_group = FakeGroup()
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 2
    moe.renormalize = True
    moe.hash_indices_dtype = torch.int64
    moe.routed_scaling_factor = 1.0

    moe._run_mega_moe_tp_dedup(
        hidden_states,
        input_ids=input_ids,
        out=out,
        activation_clamp=None,
    )

    assert torch.equal(out, hidden_states + 10)
    assert moe.experts.padding == expected_padding
    expected_input_ids, _ = dsv4_model._shard_tp_token_rows(
        input_ids,
        tp_rank,
        tp_size,
    )
    assert torch.equal(routed_input_ids[0], expected_input_ids)


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


def test_sm90_mega_moe_rejects_unsupported_num_sms(monkeypatch):
    monkeypatch.setattr(dsv4_model.envs, "VLLM_DSV4_MEGA_MOE_NUM_SMS", 77)

    with pytest.raises(ValueError, match="0, 76, or 78"):
        _make_fp4_experts()


def test_get_symm_buffer_for_num_tokens_uses_decode_buffer(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
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


def test_get_symm_buffer_for_num_tokens_rejects_beyond_batched():
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.max_num_tokens = 80
    experts.max_num_batched_tokens = 256

    with pytest.raises(ValueError):
        experts.get_symm_buffer_for_num_tokens(257)


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


def test_sm100_run_mega_moe_forwards_padding_mask_to_prepare(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.eplb_state = SimpleNamespace(logical_to_physical_map=None)
    experts._use_sm90_mega_moe = False
    experts._transformed_l1_weights = object()
    experts._transformed_l2_weights = object()
    monkeypatch.setattr(experts, "finalize_weights", lambda: None)
    monkeypatch.setattr(
        experts,
        "get_symm_buffer_for_num_tokens",
        lambda n: SimpleNamespace(
            x=torch.empty(n, 4),
            x_sf=torch.empty(n, 1),
            topk_idx=torch.empty(n, 2, dtype=torch.int64),
            topk_weights=torch.empty(n, 2),
        ),
    )
    monkeypatch.setattr(dsv4_model.envs, "VLLM_MOE_SKIP_PADDING", True)

    captured = {}

    def fake_prepare(*args, **kwargs):
        captured["is_padding"] = kwargs["is_padding"]

    monkeypatch.setattr(dsv4_model, "prepare_megamoe_inputs", fake_prepare)

    calls = []

    class FakeDeepGemm:
        def fp8_fp4_mega_moe(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )

    hidden_states = torch.randn(3, 4)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    is_padding = torch.tensor([False, True, True], dtype=torch.bool)

    with override_forward_context(SimpleNamespace(is_padding=is_padding)):
        experts._run_mega_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            y,
            activation_clamp=None,
            fast_math=True,
        )

    assert len(calls) == 1
    torch.testing.assert_close(captured["is_padding"], is_padding)


def test_sm90_run_mega_moe_uses_skip_padding_sentinel_for_idle_rows(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts.eplb_state = SimpleNamespace(logical_to_physical_map=None)
    experts._use_sm90_mega_moe = True
    experts._use_sm90_fp4_mega_moe = True
    experts._transformed_l1_weights = object()
    experts._transformed_l2_weights = object()
    experts._sm90_mega_moe_num_sms = 0
    monkeypatch.setattr(experts, "finalize_weights", lambda: None)
    monkeypatch.setattr(
        experts,
        "get_symm_buffer_for_num_tokens",
        lambda n: SimpleNamespace(
            x=torch.empty(n, 4),
            x_sf=torch.empty(n, 1),
            topk_idx=torch.empty(n, 2, dtype=torch.int64),
            topk_weights=torch.empty(n, 2),
        ),
    )
    monkeypatch.setattr(dsv4_model.envs, "VLLM_MOE_SKIP_PADDING", True)

    captured = {}

    def fake_prepare(
        hidden_states,
        topk_weights,
        topk_ids,
        *args,
        **kwargs,
    ):
        captured["topk_ids"] = topk_ids.clone()
        captured["topk_weights"] = topk_weights.clone()

    monkeypatch.setattr(dsv4_model, "prepare_megamoe_inputs_sm90", fake_prepare)

    class FakeDeepGemm:
        def fp8_fp4_mega_moe(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )

    hidden_states = torch.randn(2, 4)
    topk_weights = torch.tensor([[0.3, 0.7], [0.4, 0.6]], dtype=torch.float32)
    topk_ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
    y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    is_padding = torch.tensor([True, True], dtype=torch.bool)

    with override_forward_context(SimpleNamespace(is_padding=is_padding)):
        experts._run_mega_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            y,
            activation_clamp=None,
            fast_math=True,
        )

    assert torch.equal(captured["topk_ids"], torch.full_like(topk_ids, -1))
    assert torch.equal(captured["topk_weights"], torch.zeros_like(topk_weights))


def test_sm90_fp4_mega_moe_passes_num_sms_override(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_sm90_fp4_mega_moe = True
    experts._transformed_l1_weights = object()
    experts._transformed_l2_weights = object()
    experts._sm90_mega_moe_num_sms = 76
    monkeypatch.setattr(
        experts,
        "get_symm_buffer_for_num_tokens",
        lambda n: SimpleNamespace(
            x=torch.empty(n, 4),
            x_sf=torch.empty(n, 1),
            topk_idx=torch.empty(n, 2, dtype=torch.int64),
            topk_weights=torch.empty(n, 2),
        ),
    )
    monkeypatch.setattr(
        dsv4_model,
        "prepare_megamoe_inputs_sm90",
        lambda *args, **kwargs: None,
    )

    calls = []

    class FakeDeepGemm:
        def fp8_fp4_mega_moe(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        deep_gemm_utils,
        "_import_deep_gemm",
        lambda: FakeDeepGemm(),
    )

    experts._run_mega_moe_sm90(
        torch.randn(2, 4),
        torch.randn(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
        torch.empty(2, 4),
        activation_clamp=None,
        fast_math=True,
    )

    assert len(calls) == 1
    assert calls[0][1]["num_sms"] == 76


def test_sm90_mega_moe_keeps_sm76_before_last_pp_stage(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._sm90_mega_moe_num_sms = 76
    monkeypatch.setattr(
        dsv4_model,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2, is_last_rank=False),
    )

    assert experts._get_effective_sm90_mega_moe_num_sms() == 76


def test_sm90_mega_moe_uses_sm78_on_last_pp_stage(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._sm90_mega_moe_num_sms = 76
    monkeypatch.setattr(
        dsv4_model,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2, is_last_rank=True),
    )

    assert experts._get_effective_sm90_mega_moe_num_sms() == 78


def test_sm90_fp8_mega_moe_rejects_num_sms_override(monkeypatch):
    experts = object.__new__(DeepseekV4MegaMoEExperts)
    experts._use_sm90_fp4_mega_moe = False
    experts._transformed_l1_weights = object()
    experts._transformed_l2_weights = object()
    experts._sm90_mega_moe_num_sms = 76
    monkeypatch.setattr(
        experts,
        "get_symm_buffer_for_num_tokens",
        lambda n: SimpleNamespace(
            x=torch.empty(n, 4),
            x_sf=torch.empty(n, 1),
            topk_idx=torch.empty(n, 2, dtype=torch.int64),
            topk_weights=torch.empty(n, 2),
        ),
    )
    monkeypatch.setattr(
        dsv4_model,
        "prepare_megamoe_inputs_sm90",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="only supported for the SM90 FP4"):
        experts._run_mega_moe_sm90(
            torch.randn(2, 4),
            torch.randn(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            torch.empty(2, 4),
            activation_clamp=None,
            fast_math=True,
        )
