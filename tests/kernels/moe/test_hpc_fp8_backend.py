# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import sys
import types
from unittest.mock import patch

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
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.hpc import hpc_fuse_moe_blockwise


def _moe_config(
    *,
    moe_backend: str = "hpc",
    parallel_config: FusedMoEParallelConfig | None = None,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=4,
        num_logical_experts=4,
        moe_parallel_config=parallel_config or FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_backend=moe_backend,
    )


def _quant_config(clamp: float | None = 7.0) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        block_shape=[1, 128],
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        gemm1_clamp_limit=clamp,
    )


def test_explicit_hpc_fp8_backend_selects_hpc_experts():
    with patch.object(HPCExperts, "_supports_current_device", return_value=True):
        backend, experts_cls = select_fp8_moe_backend(
            _moe_config(),
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )

    assert backend == Fp8MoeBackend.HPC
    assert experts_cls is HPCExperts


@pytest.mark.parametrize(
    ("parallel_config", "supported"),
    [
        (FusedMoEParallelConfig.make_no_parallel(), True),
        (dataclasses.replace(FusedMoEParallelConfig.make_no_parallel(), dp_size=2), False),
        (
            dataclasses.replace(
                FusedMoEParallelConfig.make_no_parallel(), ep_size=2, use_ep=True
            ),
            False,
        ),
        (dataclasses.replace(FusedMoEParallelConfig.make_no_parallel(), sp_size=2), False),
    ],
)
def test_hpc_fp8_backend_limits_parallel_config(parallel_config, supported):
    assert HPCExperts._supports_parallel_config(parallel_config) is supported


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
