# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
from types import SimpleNamespace

import pytest
import torch
from torch import fx

import vllm.model_executor.layers.fused_moe.runner.moe_runner  # noqa: F401
from vllm.compilation.passes.fusion import collective_fusion
from vllm.compilation.passes.fusion.collective_fusion import AsyncTPPass
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _moe_forward_shared_w4a8_chunked_finalize_rs_fake,
)
from vllm.utils.torch_utils import LayerName


def make_graph(
    *,
    extra_getitem_user: bool = False,
    include_layer_meta: bool = True,
) -> fx.Graph:
    graph = fx.Graph()
    hidden = graph.placeholder("hidden")
    router_logits = graph.placeholder("router_logits")
    layer_name = graph.placeholder("layer_name")
    if include_layer_meta:
        layer_name.meta["val"] = SimpleNamespace(
            real_obj=LayerName("layers.3.block_sparse_moe.experts")
        )
    moe = graph.call_function(
        torch.ops.vllm.moe_forward_shared.default,
        args=(hidden, router_logits, hidden, None, layer_name, 0),
    )
    shared = graph.call_function(operator.getitem, args=(moe, 0))
    routed = graph.call_function(operator.getitem, args=(moe, 1))
    added = graph.call_function(torch.ops.aten.add.Tensor, args=(shared, routed))
    reduced = graph.call_function(
        torch.ops.vllm.reduce_scatter.default,
        args=(added, 0, 4, "tp:0"),
    )
    reduced.meta["val"] = SimpleNamespace(
        dtype=torch.bfloat16,
        ndim=2,
        shape=(4, 6144),
    )
    if extra_getitem_user:
        extra = graph.call_function(torch.ops.aten.clone.default, args=(shared,))
        graph.output((reduced, extra))
    else:
        graph.output(reduced)
    return graph


@pytest.fixture
def rewrite_pass(monkeypatch):
    monkeypatch.setattr(
        collective_fusion,
        "get_tp_group",
        lambda: SimpleNamespace(unique_name="tp:0"),
    )
    pass_ = AsyncTPPass.__new__(AsyncTPPass)
    pass_.w4a8_chunked_layers = {
        "layers.3.block_sparse_moe.experts": 1.0,
    }
    pass_.w4a8_chunked_wildcard_scale = 1.0
    return pass_


def test_w4a8_moe_finalize_rs_rewrite(rewrite_pass):
    graph = make_graph()

    assert rewrite_pass._rewrite_w4a8_chunked_finalize_rs(graph) == 1

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets == [
        torch.ops.vllm.moe_forward_shared_w4a8_chunked_finalize_rs.default
    ]
    replacement = next(node for node in graph.nodes if node.op == "call_function")
    assert replacement.args[-4:] == (
        1.0,
        256,
        4,
        "compiler_sp_async_tp_w4a8",
    )


def test_w4a8_moe_finalize_rs_rewrite_rejects_extra_user(rewrite_pass):
    graph = make_graph(extra_getitem_user=True)

    assert rewrite_pass._rewrite_w4a8_chunked_finalize_rs(graph) == 0
    assert any(
        node.target == torch.ops.vllm.reduce_scatter.default
        for node in graph.nodes
        if node.op == "call_function"
    )


def test_w4a8_moe_finalize_rs_rewrite_uses_safe_wildcard(rewrite_pass):
    graph = make_graph(include_layer_meta=False)

    assert rewrite_pass._rewrite_w4a8_chunked_finalize_rs(graph) == 1


def test_w4a8_moe_finalize_rs_custom_op_schema_and_fake():
    hidden = torch.empty((8, 16), dtype=torch.bfloat16)
    result = _moe_forward_shared_w4a8_chunked_finalize_rs_fake(
        hidden,
        torch.empty((8, 4), dtype=torch.float32),
        hidden,
        None,
        LayerName("layers.3.block_sparse_moe.experts"),
        0,
        1.0,
        2,
        4,
        "compiler_sp_async_tp_w4a8",
    )

    assert result.shape == (2, 16)
    assert result.is_contiguous()
    assert "str state_key" in str(
        torch.ops.vllm.moe_forward_shared_w4a8_chunked_finalize_rs.default._schema
    )
