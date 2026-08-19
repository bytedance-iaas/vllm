# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.minimax_m3.nvidia.model as minimax_model
from vllm.config import CompilationConfig


def _make_sparse_layer() -> minimax_model.MiniMaxM3SparseAttention:
    layer = minimax_model.MiniMaxM3SparseAttention.__new__(
        minimax_model.MiniMaxM3SparseAttention
    )
    nn.Module.__init__(layer)
    layer.layer_name = "model.layers.3.self_attn.attn"
    layer.q_size = 4
    layer.kv_size = 2
    layer.index_q_size = 2
    layer.num_heads = 1
    layer.num_kv_heads = 1
    layer.num_idx_heads = 1
    layer.hidden_size = 8
    layer.head_dim = 4
    layer.idx_head_dim = 2
    layer.kv_cache_dtype = "auto"
    layer.q_norm = SimpleNamespace(
        weight=torch.ones(4),
        variance_epsilon=1e-6,
    )
    layer.k_norm = SimpleNamespace(weight=torch.ones(4))
    layer.index_q_norm = SimpleNamespace(weight=torch.ones(2))
    layer.index_k_norm = SimpleNamespace(weight=torch.ones(2))
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.empty(0),
        rotary_dim=4,
    )
    layer.topk_indices_buffer = torch.empty(3, 1, 1, dtype=torch.int32)
    return layer


def _set_forward_context(
    monkeypatch: pytest.MonkeyPatch,
    layer: minimax_model.MiniMaxM3SparseAttention,
    slot_mapping: object,
) -> None:
    context = SimpleNamespace(
        no_compile_layers={layer.layer_name: layer},
        slot_mapping=slot_mapping,
    )
    monkeypatch.setattr(minimax_model, "get_forward_context", lambda: context)


def test_sparse_attention_custom_op_schema_mutates_qkv_and_output() -> None:
    schema = torch.ops.vllm.minimax_m3_sparse_attention_with_output.default._schema
    assert str(schema).endswith("-> ()")
    assert "Tensor(a0!) qkv" in str(schema)
    assert "Tensor(a2!) output" in str(schema)


def test_sparse_attention_custom_op_is_an_attention_split_point() -> None:
    assert (
        "vllm::minimax_m3_sparse_attention_with_output"
        in CompilationConfig._attention_ops
    )


def test_sparse_attention_custom_op_profiles_with_zero_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_sparse_layer()
    layer.kv_cache = torch.empty(0)
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="model.layers.3.self_attn.indexer",
            kv_cache=torch.empty(0),
        )
    )
    _set_forward_context(monkeypatch, layer, {})
    monkeypatch.setattr(
        minimax_model.ops,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        lambda *args, **kwargs: pytest.fail("profile path ran cache insert"),
    )

    qkv = torch.randn(3, 12)
    output = torch.full((3, 4), torch.nan)
    minimax_model.minimax_m3_sparse_attention_with_output(
        qkv,
        torch.arange(3),
        output,
        layer.layer_name,
    )

    assert torch.equal(output, torch.zeros_like(output))


def test_sparse_attention_custom_op_accepts_multihead_index_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_sparse_layer()
    layer.num_idx_heads = 2
    layer.index_q_size = 4
    layer.kv_cache = torch.empty(0)
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="model.layers.3.self_attn.indexer",
            kv_cache=torch.empty(0),
        )
    )
    _set_forward_context(monkeypatch, layer, {})

    output = torch.full((3, 4), torch.nan)
    minimax_model.minimax_m3_sparse_attention_with_output(
        torch.randn(3, 14),
        torch.arange(3),
        output,
        layer.layer_name,
    )

    assert torch.equal(output, torch.zeros_like(output))


def test_sparse_attention_custom_op_preserves_runtime_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    layer = _make_sparse_layer()
    layer.kv_cache = torch.empty(1, 1, 1)
    index_cache = SimpleNamespace(
        prefix="model.layers.3.self_attn.indexer",
        kv_cache=torch.empty(1, 1, 1),
        dtype=torch.float32,
    )

    class _Indexer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.index_cache = index_cache

        def forward(self, index_query: torch.Tensor) -> None:
            events.append("indexer")

    class _Impl:
        def forward(self, layer, query, kv_cache, output):
            events.append("attention")
            output.copy_(query)
            return output

    layer.indexer = _Indexer()
    layer.impl = _Impl()
    _set_forward_context(
        monkeypatch,
        layer,
        {
            layer.layer_name: torch.arange(3),
            index_cache.prefix: torch.arange(3),
        },
    )

    def _cache_insert(*args, **kwargs) -> None:
        events.append("cache_insert")
        args[17].copy_(args[0][:, :4])
        args[18].copy_(args[0][:, 8:10])

    monkeypatch.setattr(
        minimax_model.ops,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        _cache_insert,
    )

    qkv = torch.randn(3, 12)
    output = torch.empty(3, 4)
    minimax_model.minimax_m3_sparse_attention_with_output(
        qkv,
        torch.arange(3),
        output,
        layer.layer_name,
    )

    assert events == ["cache_insert", "indexer", "attention"]
    assert torch.equal(output, qkv[:, :4])


@pytest.mark.parametrize(
    "slot_mapping",
    [
        {"model.layers.3.self_attn.attn": torch.arange(3)},
        {"model.layers.3.self_attn.indexer": torch.arange(3)},
    ],
)
def test_sparse_attention_custom_op_requires_both_runtime_mappings(
    monkeypatch: pytest.MonkeyPatch,
    slot_mapping: dict[str, torch.Tensor],
) -> None:
    layer = _make_sparse_layer()
    layer.kv_cache = torch.empty(1, 1, 1)
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="model.layers.3.self_attn.indexer",
            kv_cache=torch.empty(1, 1, 1),
        )
    )
    _set_forward_context(monkeypatch, layer, slot_mapping)

    with pytest.raises(RuntimeError, match="slot mappings"):
        minimax_model.minimax_m3_sparse_attention_with_output(
            torch.randn(3, 12),
            torch.arange(3),
            torch.empty(3, 4),
            layer.layer_name,
        )


def test_sparse_attention_custom_op_rejects_incompatible_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_sparse_layer()
    layer.kv_cache = torch.empty(1, 1, 1)
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="model.layers.3.self_attn.indexer",
            kv_cache=torch.empty(1, 1, 1),
        )
    )
    _set_forward_context(
        monkeypatch,
        layer,
        {
            layer.layer_name: torch.arange(3),
            layer.indexer.index_cache.prefix: torch.arange(3),
        },
    )

    with pytest.raises(RuntimeError, match="output shape"):
        minimax_model.minimax_m3_sparse_attention_with_output(
            torch.randn(3, 12),
            torch.arange(3),
            torch.empty(3, 5),
            layer.layer_name,
        )


def test_sparse_attention_custom_op_rejects_wrong_layer_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.3.self_attn.attn"
    context = SimpleNamespace(
        no_compile_layers={layer_name: nn.Identity()},
        slot_mapping={},
    )
    monkeypatch.setattr(minimax_model, "get_forward_context", lambda: context)

    with pytest.raises(RuntimeError, match="incompatible type"):
        minimax_model.minimax_m3_sparse_attention_with_output(
            torch.randn(3, 12),
            torch.arange(3),
            torch.empty(3, 4),
            layer_name,
        )
