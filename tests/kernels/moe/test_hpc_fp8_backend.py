# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.hpc_moe import HPCExperts
from vllm.utils import hpc as hpc_utils
from vllm.utils.hpc import hpc_fuse_moe_blockwise


def _moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=4,
        num_logical_experts=4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_backend="hpc",
    )


def _quant_config(clamp: float | None = 7.0) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        block_shape=[1, 128],
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        gemm1_clamp_limit=clamp,
    )


def test_hpc_blockwise_wrapper_passes_activation_clamp(monkeypatch):
    calls = {}
    result = object()

    def fake_fuse_moe_blockwise(*args, **kwargs):
        calls["kwargs"] = kwargs
        return result

    monkeypatch.setitem(
        sys.modules,
        "hpc",
        types.SimpleNamespace(fuse_moe_blockwise=fake_fuse_moe_blockwise),
    )

    assert (
        hpc_fuse_moe_blockwise(
            *(object() for _ in range(10)),
            output="output",
            activation_clamp=7.0,
        )
        is result
    )
    assert calls["kwargs"]["activation_clamp"] == 7.0
    assert calls["kwargs"]["output"] == "output"


def test_hpc_blockwise_wrapper_rejects_old_hpc_ops_when_clamp_needed(monkeypatch):
    def old_fuse_moe_blockwise(*args, output=None):
        return output

    monkeypatch.setitem(
        sys.modules,
        "hpc",
        types.SimpleNamespace(fuse_moe_blockwise=old_fuse_moe_blockwise),
    )

    with pytest.raises(RuntimeError, match="activation_clamp support"):
        hpc_fuse_moe_blockwise(*(object() for _ in range(10)), activation_clamp=7.0)


def test_hpc_blockwise_wrapper_allows_old_hpc_ops_without_clamp(monkeypatch):
    calls = []

    def old_fuse_moe_blockwise(*args, output=None):
        calls.append(output)
        return output

    monkeypatch.setitem(
        sys.modules,
        "hpc",
        types.SimpleNamespace(fuse_moe_blockwise=old_fuse_moe_blockwise),
    )

    assert (
        hpc_fuse_moe_blockwise(
            *(object() for _ in range(10)),
            output="output",
            activation_clamp=None,
        )
        == "output"
    )
    assert (
        hpc_fuse_moe_blockwise(
            *(object() for _ in range(10)),
            output="output-2",
            activation_clamp=0.0,
        )
        == "output-2"
    )
    assert calls == ["output", "output-2"]


def test_hpc_blockwise_clamp_probe_fails_closed_when_signature_is_unavailable(
    monkeypatch,
):
    def fuse_moe_blockwise():
        pass

    def unavailable_signature(_):
        raise ValueError("no signature")

    hpc_utils._hpc_blockwise_supports_activation_clamp.cache_clear()
    monkeypatch.setattr(hpc_utils.inspect, "signature", unavailable_signature)

    assert not hpc_utils._hpc_blockwise_supports_activation_clamp(fuse_moe_blockwise)


def test_hpc_blockwise_clamp_probe_rejects_positional_only_parameter():
    def fuse_moe_blockwise(activation_clamp, /):
        pass

    hpc_utils._hpc_blockwise_supports_activation_clamp.cache_clear()

    assert not hpc_utils._hpc_blockwise_supports_activation_clamp(fuse_moe_blockwise)


def test_hpc_experts_apply_forwards_activation_clamp(monkeypatch):
    calls = {}
    experts = HPCExperts(_moe_config(), _quant_config(clamp=7.0))

    def fake_hpc_fuse_moe_blockwise(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.hpc_moe.hpc_fuse_moe_blockwise",
        fake_hpc_fuse_moe_blockwise,
    )

    experts.apply(
        output=torch.empty(1, 128),
        hidden_states=torch.empty(1, 128),
        w1=torch.empty(4, 256, 128),
        w2=torch.empty(4, 128, 128),
        topk_weights=torch.empty(1, 2),
        topk_ids=torch.empty(1, 2, dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=None,
        a1q_scale=torch.empty(1, 1),
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=None,
    )

    assert calls["activation_clamp"] == 7.0
    assert calls["num_expert_total"] == 4
    assert calls["rank_ep"] == 0
