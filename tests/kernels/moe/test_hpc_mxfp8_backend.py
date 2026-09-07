# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.hpc_moe as hpc_moe
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as mxfp8_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.utils import hpc as hpc_utils


def _parallel_config(**overrides) -> FusedMoEParallelConfig:
    values = {"tp_size": 16}
    values.update(overrides)
    return replace(FusedMoEParallelConfig.make_no_parallel(), **values)


def _moe_config(**overrides) -> FusedMoEConfig:
    values = {
        "num_experts": 128,
        "experts_per_token": 4,
        "hidden_dim": 6144,
        "intermediate_size": 3072,
        "num_local_experts": 128,
        "num_logical_experts": 128,
        "moe_parallel_config": _parallel_config(),
        "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        "in_dtype": torch.bfloat16,
        "device": "cuda",
        "routing_method": RoutingMethodType.TopK,
        "moe_backend": "hpc",
    }
    values.update(overrides)
    return FusedMoEConfig(**values)


def _make_hpc_experts() -> hpc_moe.MiniMaxM3HPCExperts:
    experts = object.__new__(hpc_moe.MiniMaxM3HPCExperts)
    object.__setattr__(experts, "num_experts", 128)
    object.__setattr__(
        experts,
        "quant_config",
        SimpleNamespace(
            w1_scale=torch.empty(
                (128, 384, 192), dtype=torch.uint8, device="meta"
            ),
            w2_scale=torch.empty(
                (128, 6144, 6), dtype=torch.uint8, device="meta"
            ),
            gemm1_clamp_limit=7.0,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
        ),
    )
    return experts


def test_mxfp8_hpc_backend_is_explicit_only():
    assert mxfp8_oracle._BACKEND_NAME_MAP["hpc"] is Fp8MoeBackend.HPC
    assert Fp8MoeBackend.HPC not in mxfp8_oracle._SUPPORTED_BACKENDS
    assert mxfp8_oracle._mxfp8_backend_to_kernel_cls(Fp8MoeBackend.HPC) == [
        hpc_moe.MiniMaxM3HPCExperts
    ]


def test_minimax_hpc_supports_only_target_config(monkeypatch):
    monkeypatch.setattr(
        hpc_moe.MiniMaxM3HPCExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    supported, reason = hpc_moe.MiniMaxM3HPCExperts.is_supported_config(
        hpc_moe.MiniMaxM3HPCExperts,
        _moe_config(),
        kMxfp8Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported
    assert reason is None


@pytest.mark.parametrize(
    "parallel_config",
    [
        _parallel_config(tp_size=8),
        _parallel_config(dp_size=2),
        _parallel_config(pcp_size=2),
        _parallel_config(ep_size=16, tp_size=1, use_ep=True),
        _parallel_config(sp_size=16),
        _parallel_config(enable_eplb=True),
    ],
)
def test_minimax_hpc_rejects_unvalidated_parallel_configs(parallel_config):
    assert not hpc_moe.MiniMaxM3HPCExperts._supports_parallel_config(
        parallel_config
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"num_experts": 64, "num_local_experts": 64}, "128 local experts"),
        ({"experts_per_token": 2}, "top_k=4"),
        ({"intermediate_size": 4096}, "intermediate_size_per_partition=192"),
        ({"in_dtype": torch.float16}, "BF16 activations"),
    ],
)
def test_minimax_hpc_rejects_non_minimax_shapes(monkeypatch, overrides, reason):
    monkeypatch.setattr(
        hpc_moe.MiniMaxM3HPCExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    supported, actual_reason = hpc_moe.MiniMaxM3HPCExperts.is_supported_config(
        hpc_moe.MiniMaxM3HPCExperts,
        _moe_config(**overrides),
        kMxfp8Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert actual_reason is not None and reason in actual_reason


@pytest.mark.parametrize("op_present", [False, True])
def test_hpc_mxfp8_probe_requires_final_out_op(monkeypatch, op_present):
    hpc_utils.has_hpc_mxfp8_k32_moe.cache_clear()
    monkeypatch.setattr(hpc_utils, "has_hpc", lambda: True)
    monkeypatch.setitem(sys.modules, "hpc", ModuleType("hpc"))
    hpc_namespace = SimpleNamespace()
    if op_present:
        hpc_namespace.fuse_moe_mxfp8_k32_bf16_candidate_out = object()
    monkeypatch.setattr(
        hpc_utils,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(hpc=hpc_namespace)),
    )

    try:
        assert hpc_utils.has_hpc_mxfp8_k32_moe() is op_present
    finally:
        hpc_utils.has_hpc_mxfp8_k32_moe.cache_clear()


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 8192])
def test_minimax_hpc_workspace_shapes_include_aligned_packing(num_tokens):
    experts = _make_hpc_experts()
    routed_rows = num_tokens * 4
    routing_bytes = ((2 * routed_rows + 257) * 4 + 15) & ~15
    expected_bytes = routing_bytes + routed_rows * (6144 + 192 + 384 * 2)

    workspace13, workspace2, output = experts.workspace_shapes(
        M=num_tokens,
        N=384,
        K=6144,
        topk=4,
        global_num_experts=128,
        local_num_experts=128,
        expert_tokens_meta=None,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )

    assert workspace13 == ((expected_bytes + 1) // 2,)
    assert workspace2 == (routed_rows, 6144)
    assert output == (num_tokens, 6144)
    assert routing_bytes % 16 == 0


def test_minimax_hpc_apply_packs_aligned_reused_workspace(monkeypatch):
    experts = _make_hpc_experts()
    num_tokens = 16
    routed_rows = num_tokens * 4
    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        num_tokens,
        384,
        6144,
        4,
        128,
        128,
        None,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    workspace13 = torch.empty(workspace13_shape, dtype=torch.bfloat16)
    workspace2 = torch.empty(workspace2_shape, dtype=torch.bfloat16)
    output = torch.empty(output_shape, dtype=torch.bfloat16)
    hidden = torch.empty((num_tokens, 6144), dtype=torch.bfloat16)
    topk_ids = torch.zeros((num_tokens, 4), dtype=torch.int32)
    topk_weights = torch.full((num_tokens, 4), 0.25, dtype=torch.float32)
    w1 = torch.empty((128, 384, 6144), dtype=torch.float8_e4m3fn, device="meta")
    w2 = torch.empty((128, 6144, 192), dtype=torch.float8_e4m3fn, device="meta")
    captured = {}

    def fake_hpc_out(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        hpc_moe,
        "hpc_fuse_moe_mxfp8_k32_bf16_candidate_out",
        fake_hpc_out,
    )

    experts.apply(
        output=output,
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=128,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    grouped_hidden = captured["grouped_hidden"]
    grouped_scale = captured["grouped_hidden_scale"]
    gate_output = captured["gate_output"]
    assert captured["hidden"] is hidden
    assert captured["topk_ids"] is topk_ids
    assert captured["topk_weights"] is topk_weights
    assert captured["output"] is output
    assert captured["down_output"] is workspace2
    assert grouped_hidden.shape == (routed_rows, 6144)
    assert grouped_scale.shape == (routed_rows, 192)
    assert grouped_hidden.data_ptr() % 16 == 0
    assert grouped_scale.data_ptr() % 16 == 0
    assert gate_output.data_ptr() % 16 == 0
    assert captured["activated_output"].data_ptr() == grouped_hidden.data_ptr()
    assert captured["activated_scale"].data_ptr() == (
        grouped_hidden.data_ptr() + routed_rows * 192
    )


def test_hpc_out_wrapper_preserves_input_and_workspace_identity(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "hpc", ModuleType("hpc"))
    monkeypatch.setattr(
        torch.ops.hpc,
        "fuse_moe_mxfp8_k32_bf16_candidate_out",
        lambda *args: calls.append(args),
        raising=False,
    )
    tensors = [torch.empty(1) for _ in range(18)]

    result = hpc_utils.hpc_fuse_moe_mxfp8_k32_bf16_candidate_out(
        hidden=tensors[0],
        gate_up_weight=tensors[1],
        gate_up_weight_scale=tensors[2],
        down_weight=tensors[3],
        down_weight_scale=tensors[4],
        topk_ids=tensors[5],
        topk_weights=tensors[6],
        output=tensors[7],
        row_indices=tensors[8],
        topk_pos=tensors[9],
        seqlens=tensors[10],
        cu_seqlens=tensors[11],
        grouped_hidden=tensors[12],
        grouped_hidden_scale=tensors[13],
        gate_output=tensors[14],
        activated_output=tensors[15],
        activated_scale=tensors[16],
        down_output=tensors[17],
    )

    assert result is tensors[7]
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], tensors))


def test_modular_kernel_passes_distinct_caller_output_to_hpc(monkeypatch):
    applied = {}

    class FakeExperts:
        moe_config = SimpleNamespace(moe_parallel_config=None)
        a2_scale = None

        def moe_problem_size(self, *args):
            return 128, 2, 384, 8, 4

        def apply(self, **kwargs):
            applied.update(kwargs)

    kernel = mk.FusedMoEKernelModularImpl(None, FakeExperts())
    workspace13 = torch.empty(1)
    workspace2 = torch.empty(1)
    allocated_output = torch.empty((2, 8))
    caller_output = torch.empty_like(allocated_output)
    monkeypatch.setattr(
        kernel,
        "_allocate_buffers",
        lambda *args, **kwargs: (workspace13, workspace2, allocated_output),
    )
    monkeypatch.setattr(mk.current_platform, "is_rocm", lambda: False)

    result = kernel._fused_experts(
        in_dtype=torch.float32,
        a1q=torch.empty((2, 8)),
        a1q_scale=None,
        w1=torch.empty((128, 384, 8)),
        w2=torch.empty((128, 8, 192)),
        topk_weights=torch.empty((2, 4)),
        topk_ids=torch.empty((2, 4), dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=128,
        local_num_experts=128,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
        output_alias=caller_output,
    )

    assert result is caller_output
    assert applied["output"] is caller_output
    assert applied["output"] is not workspace13
