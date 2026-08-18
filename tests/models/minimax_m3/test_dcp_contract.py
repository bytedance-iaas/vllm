# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.minimax_m3.common.indexer as indexer
import vllm.models.minimax_m3.common.sparse_attention as sparse_attention
import vllm.models.minimax_m3.nvidia.model as minimax_model
import vllm.v1.spec_decode.dcp_eagle_trace as eagle_trace


class _SM100Platform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_device_capability_family(family: int) -> bool:
        return family == 100

    @staticmethod
    def is_device_capability(capability: int) -> bool:
        return capability == 100


class _SM90Platform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_device_capability_family(family: int) -> bool:
        return family == 90

    @staticmethod
    def is_device_capability(capability: int) -> bool:
        return capability == 90


def test_force_unfused_post_attn_norm_is_sm90_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_FORCE_UNFUSED_POST_ATTN_NORM", "1")
    monkeypatch.setattr(minimax_model, "current_platform", _SM90Platform())
    assert minimax_model._force_unfused_post_attn_norm()

    monkeypatch.setattr(minimax_model, "current_platform", _SM100Platform())
    assert not minimax_model._force_unfused_post_attn_norm()

    monkeypatch.delenv("VLLM_MINIMAX_M3_FORCE_UNFUSED_POST_ATTN_NORM")
    monkeypatch.setattr(minimax_model, "current_platform", _SM90Platform())
    assert not minimax_model._force_unfused_post_attn_norm()


def test_sm100_dcp_forces_triton_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2)
    )
    monkeypatch.setattr(indexer, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(sparse_attention, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(indexer, "current_platform", _SM100Platform())
    monkeypatch.setattr(sparse_attention, "current_platform", _SM100Platform())

    indexer_impl_cls = indexer.select_indexer_impl_cls(
        topk_blocks=16,
        indexer_kv_dtype="bf16",
    )
    main_impl_cls = sparse_attention.select_main_impl_cls(
        topk_blocks=16,
        kv_cache_dtype="bfloat16",
        num_kv_heads=1,
    )

    assert indexer_impl_cls is indexer.MiniMaxM3IndexerTritonImpl
    assert main_impl_cls is sparse_attention.MiniMaxM3SparseTritonImpl
    with pytest.raises(NotImplementedError, match="requires the BF16 Triton indexer"):
        indexer.select_indexer_impl_cls(
            topk_blocks=16,
            indexer_kv_dtype="fp8",
        )


class _FakeDCPGroup:
    def __init__(self, gathered: torch.Tensor) -> None:
        self.gathered = gathered

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert input_.shape == (1, 1, 3, 2)
        assert dim == 2
        return self.gathered.clone()


@pytest.mark.parametrize(
    ("rank", "score", "expected"),
    [
        (0, [1.0, 9.0, 3.0], [1, -1, -1]),
        (1, [8.0, 2.0, 7.0], [2, 0, -1]),
    ],
)
def test_dcp_global_topk_localizes_selected_owners(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    score: list[float],
    expected: list[int],
) -> None:
    # Rank 0 owns global blocks 0/2/4; rank 1 owns 1/3/5.
    # The global top-3 is {2, 1, 5}.
    gathered = torch.tensor(
        [[[[9.0, 2.0], [3.0, 4.0], [1.0, 0.0], [8.0, 1.0], [7.0, 5.0], [2.0, 3.0]]]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(indexer, "get_dcp_group", lambda: _FakeDCPGroup(gathered))

    impl = SimpleNamespace(
        dcp_world_size=2,
        dcp_rank=rank,
        block_size=128,
        init_blocks=0,
        local_blocks=0,
        topk_blocks=3,
    )
    out = torch.empty((1, 1, 3), dtype=torch.int32)

    result = indexer.MiniMaxM3IndexerTritonImpl._select_dcp_global_topk(
        impl,
        torch.tensor(score, dtype=torch.float32).view(1, 1, 3),
        torch.tensor([6 * 128], dtype=torch.int32),
        out,
        max_local_blocks=3,
    )

    assert result.tolist() == [[expected]]


class _FakeTraceGroup:
    rank_in_group = 0
    rank = 0


def test_dense_attention_snapshot_preserves_floating_dtype() -> None:
    source = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

    snapshot = eagle_trace._exact_snapshot(source)

    assert snapshot.dtype == source.dtype
    assert torch.equal(snapshot.view(torch.int16), source.view(torch.int16))


class _TripwireAttention(torch.nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states * 2


class _TripwireFFN(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 3


class _TripwireLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_moe_layer = False
        self.self_attn = _TripwireAttention()
        self.mlp = _TripwireFFN()
        self.ffn_all_reduce_deferred = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        return self.mlp(hidden_states)


def test_target_layer_tripwire_captures_four_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VLLM_DCP_EAGLE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_DCP_EAGLE_TRIPWIRE_POSITION", "7")
    monkeypatch.setenv("VLLM_DCP_EAGLE_TRIPWIRE_TOKEN", "11")
    monkeypatch.setenv("VLLM_DCP_EAGLE_TRIPWIRE_MAX_LAYER", "0")
    monkeypatch.setenv("VLLM_DCP_EAGLE_ATTN_DECOMP_LAYER", "0")
    monkeypatch.setattr(eagle_trace, "get_tp_group", _FakeTraceGroup)
    monkeypatch.setattr(eagle_trace, "get_dcp_group", _FakeTraceGroup)
    monkeypatch.setattr(eagle_trace, "get_world_group", _FakeTraceGroup)
    monkeypatch.setattr(eagle_trace, "current_platform", _SM90Platform())
    monkeypatch.setattr(eagle_trace, "_TARGET_LAYER_TRIPWIRE_CAPTURED", False)
    monkeypatch.setattr(eagle_trace, "_TARGET_LAYER_TRIPWIRE_ACTIVE", False)
    monkeypatch.setattr(eagle_trace, "_DENSE_ATTN_DECOMP_CAPTURED", False)

    layer = _TripwireLayer()
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=torch.nn.ModuleList([layer]))
        )
    )
    positions = torch.tensor([7])
    hidden_states = torch.tensor([[1.0, 2.0]])
    tripwire = eagle_trace.prepare_target_layer_tripwire(
        torch.tensor([11]),
        positions,
    )
    assert not eagle_trace.dense_attn_decomp_enabled(
        "model.layers.0.self_attn.attn",
        max_seq_len=8,
        num_actual_tokens=1,
    )

    with eagle_trace.capture_target_layer_tripwire(
        model,
        tripwire,
    ):
        assert eagle_trace.dense_attn_decomp_enabled(
            "model.layers.0.self_attn.attn",
            max_seq_len=8,
            num_actual_tokens=1,
        )
        output = layer(positions, hidden_states)

    assert not eagle_trace.dense_attn_decomp_enabled(
        "model.layers.0.self_attn.attn",
        max_seq_len=8,
        num_actual_tokens=1,
    )
    assert output.tolist() == [[6.0, 12.0]]
    trace_path = next(tmp_path.glob("*/target-layer-tripwire-pos7.pt"))
    trace = torch.load(trace_path, weights_only=False)
    assert trace["checkpoint_names"] == [
        "A_attention_input",
        "B_attention_return",
        "C_ffn_input",
        "D_ffn_return",
    ]
    assert trace["ffn_output_reduced"] == [False]
    assert trace["layer0_post_attention_residual"].tolist() == [1.0, 2.0]
    assert trace["checkpoints"].tolist() == [
        [[1.0, 2.0], [2.0, 4.0], [2.0, 4.0], [6.0, 12.0]]
    ]


def test_decoder_layer_toggles_dense_and_moe_ffn_reduction() -> None:
    dense = SimpleNamespace(
        is_moe_layer=False,
        mlp=SimpleNamespace(down_proj=SimpleNamespace(reduce_results=False)),
    )
    moe = SimpleNamespace(
        is_moe_layer=True,
        block_sparse_moe=SimpleNamespace(
            experts=SimpleNamespace(
                moe_config=SimpleNamespace(skip_final_all_reduce=True)
            )
        ),
    )

    minimax_model.MiniMaxM3DecoderLayer.set_ffn_all_reduce_deferred(dense, False)
    minimax_model.MiniMaxM3DecoderLayer.set_ffn_all_reduce_deferred(moe, False)
    assert dense.mlp.down_proj.reduce_results
    assert not moe.block_sparse_moe.experts.moe_config.skip_final_all_reduce

    minimax_model.MiniMaxM3DecoderLayer.set_ffn_all_reduce_deferred(dense, True)
    minimax_model.MiniMaxM3DecoderLayer.set_ffn_all_reduce_deferred(moe, True)
    assert not dense.mlp.down_proj.reduce_results
    assert moe.block_sparse_moe.experts.moe_config.skip_final_all_reduce


class _FakeDecoderLayer:
    def __init__(self, deferred: bool) -> None:
        self.deferred = deferred
        self.fuse_input_allreduce = False

    @property
    def ffn_all_reduce_deferred(self) -> bool:
        return self.deferred

    def set_ffn_all_reduce_deferred(self, defer: bool) -> None:
        self.deferred = defer


class _MutatingFirstLayer:
    def __call__(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert residual is None
        residual = hidden_states
        residual.add_(10)
        return hidden_states, residual


def test_eagle_input_boundary_survives_first_layer_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        minimax_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    model = SimpleNamespace(
        aux_hidden_state_layers=(0,),
        start_layer=0,
        end_layer=1,
        layers=[_MutatingFirstLayer()],
        fuse_final_norm_allreduce=False,
        norm=lambda hidden_states, residual: (hidden_states, residual),
    )
    model._maybe_add_hidden_state = lambda states, idx, hidden, residual: (
        minimax_model.EagleModelMixin._maybe_add_hidden_state(
            model, states, idx, hidden, residual
        )
    )
    inputs = torch.tensor([[1.0, 2.0]])

    _, aux_hidden_states = minimax_model.MiniMaxM3Model.forward(
        model,
        input_ids=None,
        positions=torch.zeros(1, dtype=torch.int64),
        intermediate_tensors=None,
        inputs_embeds=inputs,
    )

    assert inputs.tolist() == [[11.0, 12.0]]
    assert aux_hidden_states[0].tolist() == [[1.0, 2.0]]
    assert aux_hidden_states[0].data_ptr() != inputs.data_ptr()


def test_eagle_aux_boundaries_reconfigure_deferred_allreduce() -> None:
    layers = [
        _FakeDecoderLayer(True),
        _FakeDecoderLayer(False),
        _FakeDecoderLayer(True),
        _FakeDecoderLayer(True),
    ]
    model = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        _original_ffn_all_reduce_deferred=(True, False, True, True),
    )
    model._configure_deferred_allreduce = lambda boundaries: (
        minimax_model.MiniMaxM3Model._configure_deferred_allreduce(model, boundaries)
    )

    minimax_model.MiniMaxM3Model._set_aux_hidden_state_layers(model, (0, 2, 4))
    assert model.aux_hidden_state_layers == (0, 2, 4)
    assert [layer.deferred for layer in layers] == [True, False, True, False]
    assert [layer.fuse_input_allreduce for layer in layers] == [
        False,
        True,
        False,
        True,
    ]
    assert not model.fuse_final_norm_allreduce

    minimax_model.MiniMaxM3Model._set_aux_hidden_state_layers(model, (1,))
    assert [layer.deferred for layer in layers] == [False, False, True, True]
    assert [layer.fuse_input_allreduce for layer in layers] == [
        False,
        False,
        False,
        True,
    ]
    assert model.fuse_final_norm_allreduce

    minimax_model.MiniMaxM3Model._set_aux_hidden_state_layers(model, ())
    assert [layer.deferred for layer in layers] == [True, False, True, True]
    assert [layer.fuse_input_allreduce for layer in layers] == [
        False,
        True,
        False,
        True,
    ]
    assert model.fuse_final_norm_allreduce
