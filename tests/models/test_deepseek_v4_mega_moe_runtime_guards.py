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
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
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
    output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    is_padding = torch.tensor([False, True, True], dtype=torch.bool)

    with override_forward_context(SimpleNamespace(is_padding=is_padding)):
        experts._run_mega_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
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
    output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    is_padding = torch.tensor([True, True], dtype=torch.bool)

    with override_forward_context(SimpleNamespace(is_padding=is_padding)):
        experts._run_mega_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
            activation_clamp=None,
            fast_math=True,
        )

    assert torch.equal(captured["topk_ids"], torch.full_like(topk_ids, -1))
    assert torch.equal(captured["topk_weights"], torch.zeros_like(topk_weights))
