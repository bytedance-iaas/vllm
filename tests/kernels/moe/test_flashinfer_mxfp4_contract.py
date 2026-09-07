# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
)

pytestmark = pytest.mark.cpu_test


def _install_flashinfer_interleave_stubs(monkeypatch):
    interleaved_weights: list[torch.Tensor] = []
    interleaved_scales: list[torch.Tensor] = []

    def record_weight_interleave(weight, quant_type):
        assert quant_type == "fp4"
        interleaved_weights.append(weight.clone())
        return weight

    def record_scale_interleave(scale):
        interleaved_scales.append(scale.clone())
        return scale

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.block_scale_interleave = record_scale_interleave
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.interleave_moe_weights_for_sm90_mixed_gemm = (
        record_weight_interleave
    )
    fused_moe.interleave_moe_scales_for_sm90_mixed_gemm = record_scale_interleave
    flashinfer.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    return interleaved_weights, interleaved_scales


@pytest.mark.parametrize(
    ("backend", "interleaves_weights"),
    [
        (Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8, False),
        (Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16, True),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_convert_standard_mxfp4_weights_for_flashinfer_cutlass(
    monkeypatch,
    backend,
    interleaves_weights,
    with_bias,
):
    interleaved_weights_seen, interleaved_scales_seen = (
        _install_flashinfer_interleave_stubs(monkeypatch)
    )
    w13 = torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4)
    w2 = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
    w13_scale = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
    w2_scale = torch.arange(4, dtype=torch.uint8).reshape(1, 4, 1)
    w13_bias = (
        torch.arange(4, dtype=torch.float32).reshape(1, 4) if with_bias else None
    )
    w2_bias = (
        torch.arange(4, dtype=torch.float32).reshape(1, 4) if with_bias else None
    )

    converted = convert_weight_to_mxfp4_moe_kernel_format(
        backend,
        SimpleNamespace(),
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
    )

    expected_w13 = torch.cat([w13[:, 2:], w13[:, :2]], dim=1)
    expected_w13_scale = torch.cat([w13_scale[:, 2:], w13_scale[:, :2]], dim=1)
    torch.testing.assert_close(converted[0], expected_w13)
    torch.testing.assert_close(converted[1], w2)
    torch.testing.assert_close(converted[2], expected_w13_scale)
    torch.testing.assert_close(converted[3], w2_scale)

    if with_bias:
        assert w13_bias is not None and w2_bias is not None
        expected_w13_bias = torch.cat(
            [w13_bias[:, 2:], w13_bias[:, :2]], dim=1
        ).to(torch.bfloat16)
        torch.testing.assert_close(converted[4], expected_w13_bias)
        torch.testing.assert_close(converted[5], w2_bias.to(torch.bfloat16))
    else:
        assert converted[4] is None
        assert converted[5] is None

    if interleaves_weights:
        assert len(interleaved_weights_seen) == 2
        torch.testing.assert_close(interleaved_weights_seen[0], expected_w13)
        torch.testing.assert_close(interleaved_weights_seen[1], w2)
    else:
        assert not interleaved_weights_seen
    assert len(interleaved_scales_seen) == 2
    torch.testing.assert_close(interleaved_scales_seen[0], expected_w13_scale)
    torch.testing.assert_close(interleaved_scales_seen[1], w2_scale)


def _make_flashinfer_experts(
    alpha: float | None,
    beta: float | None,
    clamp: float | None,
):
    parallel = SimpleNamespace(ep_rank=0, ep_size=1, tp_rank=0, tp_size=1, dp_size=1)
    moe_config = SimpleNamespace(
        device="cpu",
        num_local_experts=2,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        max_capture_size=8,
    )
    quant_config = SimpleNamespace(
        weight_quant_dtype="mxfp4",
        quant_dtype="mxfp8",
        use_nvfp4_w4a4=False,
        use_fp8_w8a8=False,
        is_block_quantized=False,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=clamp,
        w1_scale=torch.zeros(4, dtype=torch.uint8),
        w2_scale=torch.zeros(4, dtype=torch.uint8),
        w1_bias=None,
        w2_bias=None,
    )
    return flashinfer_cutlass_moe.FlashInferExperts(moe_config, quant_config)


@pytest.mark.parametrize(
    ("alpha", "beta", "clamp"),
    [(None, None, None), (1.702, 1.0, 7.0)],
)
def test_flashinfer_mxfp4_forwards_configured_swiglu_parameters(
    monkeypatch,
    alpha,
    beta,
    clamp,
):
    class ActivationType:
        Swiglu = "swiglu"
        Geglu = "geglu"
        Relu2 = "relu2"

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.__path__ = []
    core = types.ModuleType("flashinfer.fused_moe.core")
    core.ActivationType = ActivationType
    flashinfer.fused_moe = fused_moe
    fused_moe.core = core
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)

    calls = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        calls.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        flashinfer_cutlass_moe,
        "flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )

    experts = _make_flashinfer_experts(alpha, beta, clamp)
    experts.apply(
        output=torch.empty(1, 8, dtype=torch.bfloat16),
        hidden_states=torch.empty(1, 8, dtype=torch.bfloat16),
        w1=torch.zeros(2, 16, 8, dtype=torch.uint8),
        w2=torch.zeros(2, 8, 8, dtype=torch.uint8),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=None,
    )

    for name, expected in (
        ("swiglu_alpha", alpha),
        ("swiglu_beta", beta),
        ("swiglu_limit", clamp),
    ):
        actual = calls[name]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, torch.full((2,), expected))
