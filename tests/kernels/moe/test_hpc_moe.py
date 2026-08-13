# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.hpc_moe as hpc_moe
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.utils.hpc import hpc_fuse_moe_mxfp8_k32_bf16_candidate_out


def make_hpc_experts() -> hpc_moe.MiniMaxM3HPCExperts:
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


@pytest.mark.parametrize("num_tokens", [16, 64, 8192])
def test_minimax_hpc_workspace_shapes_include_aligned_packing(num_tokens: int):
    experts = make_hpc_experts()
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
    experts = make_hpc_experts()
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


def test_minimax_hpc_disables_modular_chunking():
    experts = make_hpc_experts()

    assert not experts.supports_chunking()


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

    result = hpc_fuse_moe_mxfp8_k32_bf16_candidate_out(
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


@pytest.mark.parametrize("supports_alias", [False, True])
def test_modular_kernel_uses_caller_output_only_when_supported(
    monkeypatch,
    supports_alias: bool,
):
    applied = {}

    class FakeExperts:
        moe_config = SimpleNamespace(moe_parallel_config=None)
        a2_scale = None

        def moe_problem_size(self, *args):
            return 128, 2, 384, 8, 4

        def supports_output_alias(self):
            return supports_alias

        def apply(self, **kwargs):
            applied.update(kwargs)

    experts = FakeExperts()
    kernel = mk.FusedMoEKernelModularImpl(None, experts)
    workspace13 = torch.empty(1)
    workspace2 = torch.empty(1)
    allocated_output = torch.empty((2, 8))
    caller_output = torch.empty_like(allocated_output)
    monkeypatch.setattr(
        kernel,
        "_allocate_buffers",
        lambda *args, **kwargs: (workspace13, workspace2, allocated_output),
    )

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

    expected = caller_output if supports_alias else allocated_output
    assert result is expected
    assert applied["output"] is expected
    reducer = TopKWeightAndReduceNoOP()
    assert (
        reducer.apply(
            expected,
            expected,
            torch.empty((2, 4)),
            torch.empty((2, 4), dtype=torch.int32),
            False,
        )
        is expected
    )
