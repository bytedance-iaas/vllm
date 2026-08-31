# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

import vllm.models.minimax_m3.common.indexer as indexer
import vllm.models.minimax_m3.common.sparse_attention as sparse_attention
import vllm.models.minimax_m3.nvidia.model as minimax_model
from vllm.sequence import IntermediateTensors


class _SM100Platform:
    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_device_capability_family(family: int) -> bool:
        return family == 100


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


@pytest.mark.parametrize(
    ("target_parity_enabled", "expected"),
    [(False, False), (True, True)],
)
def test_target_parity_flag_enables_sparse_bf16_partials(
    monkeypatch: pytest.MonkeyPatch,
    target_parity_enabled: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(
        sparse_attention,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                cp_kv_cache_interleave_size=128,
                dcp_comm_backend="a2a",
            ),
            speculative_config=SimpleNamespace(
                enable_eagle3_target_dense_full_temporal_kv=(target_parity_enabled)
            ),
        ),
    )
    monkeypatch.setattr(
        sparse_attention.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )

    impl = sparse_attention.MiniMaxM3SparseTritonImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        kv_cache_dtype="fp8",
        topk_blocks=16,
        sparse_block_size=128,
    )

    assert impl.dcp_bf16_partials is expected


class _FakeDCPGroup:
    def __init__(self, gathered: torch.Tensor) -> None:
        self.gathered = gathered

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert input_.shape == (1, 1, 3, 3)
        assert dim == 2
        return self.gathered.clone()


@pytest.mark.parametrize(
    ("rank", "score", "expected", "aligned"),
    [
        (0, [1.0, 9.0, 3.0], [1, -1, -1], [1, -1, -1]),
        (1, [8.0, 2.0, 7.0], [-1, 0, 2], [-1, 0, 2]),
    ],
)
def test_dcp_global_topk_localizes_selected_owners(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    score: list[float],
    expected: list[int],
    aligned: list[int],
) -> None:
    # Rank 0 owns global blocks 0/2/4; rank 1 owns 1/3/5.
    # The global top-3 is {2, 1, 5}.
    gathered = torch.tensor(
        [
            [
                [
                    [9.0, 2.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [8.0, 1.0, 0.0],
                    [7.0, 5.0, 0.0],
                    [2.0, 3.0, 0.0],
                ]
            ]
        ],
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
    assert impl._dcp_canonical_global_topk.tolist() == [[[2, 1, 5]]]
    assert impl._dcp_aligned_local_topk.tolist() == [[aligned]]


def test_dcp_topk_uses_strict_lexicographic_order() -> None:
    scores = torch.tensor([[[5.0, 5.0, 999.0, 5.0]]])
    global_ids = torch.tensor([[[3, 1, 2, 0]]], dtype=torch.int32)
    tiers = torch.tensor([[[0, 0, 2, 1]]], dtype=torch.int32)

    _, selected_ids, selected_tiers = indexer._stable_lexicographic_topk(
        scores,
        global_ids,
        tiers,
        4,
    )

    assert selected_ids.tolist() == [[[2, 0, 1, 3]]]
    assert selected_tiers.tolist() == [[[2, 1, 0, 0]]]


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


class _IncrementLayer:
    def __call__(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states + 1, residual


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


@pytest.mark.parametrize(
    ("start_layer", "end_layer", "expected_local", "expected_input_keys"),
    [
        (0, 30, (2, 30), ("hidden_states", "residual")),
        (
            30,
            60,
            (27,),
            (
                "hidden_states",
                "residual",
                "aux_hidden_state_2",
                "aux_hidden_state_30",
            ),
        ),
    ],
)
def test_eagle_aux_boundaries_map_global_to_pp_local(
    start_layer: int,
    end_layer: int,
    expected_local: tuple[int, ...],
    expected_input_keys: tuple[str, ...],
) -> None:
    model = SimpleNamespace(
        start_layer=start_layer,
        end_layer=end_layer,
        layers=[None] * 60,
        config=SimpleNamespace(hidden_size=2),
    )
    model._configure_deferred_allreduce = lambda boundaries: setattr(
        model, "configured_boundaries", boundaries
    )
    make_empty_intermediate_tensors = MethodType(
        minimax_model.MiniMaxM3Model.make_empty_intermediate_tensors,
        model,
    )

    minimax_model.MiniMaxM3Model._set_aux_hidden_state_layers(model, (2, 30, 57))

    assert model.aux_hidden_state_layers == (2, 30, 57)
    assert model.configured_boundaries == expected_local
    empty = make_empty_intermediate_tensors(
        batch_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert tuple(empty.tensors) == expected_input_keys


def test_eagle_aux_states_cross_pp_boundary_in_config_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr(minimax_model, "get_pp_group", lambda: pp_group)

    layers = [_IncrementLayer() for _ in range(60)]

    def make_partition(start_layer: int, end_layer: int) -> SimpleNamespace:
        model = SimpleNamespace(
            start_layer=start_layer,
            end_layer=end_layer,
            layers=layers,
            aux_hidden_state_layers=(2, 30, 57),
            fuse_final_norm_allreduce=False,
            norm=lambda hidden, residual: (hidden + residual, None),
        )
        model._maybe_add_hidden_state = lambda states, idx, hidden, residual: (
            minimax_model.EagleModelMixin._maybe_add_hidden_state(
                model, states, idx, hidden, residual
            )
        )
        return model

    positions = torch.zeros(1, dtype=torch.int64)
    inputs = torch.zeros((1, 1))
    stage0 = make_partition(0, 30)
    stage0_output = minimax_model.MiniMaxM3Model.forward(
        stage0,
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=inputs,
    )

    assert isinstance(stage0_output, IntermediateTensors)
    assert tuple(stage0_output.tensors) == (
        "hidden_states",
        "residual",
        "aux_hidden_state_2",
        "aux_hidden_state_30",
    )

    pp_group.is_first_rank = False
    pp_group.is_last_rank = True
    stage1 = make_partition(30, 60)
    _, pp_aux = minimax_model.MiniMaxM3Model.forward(
        stage1,
        input_ids=None,
        positions=positions,
        intermediate_tensors=stage0_output,
    )

    pp_group.is_first_rank = True
    reference = make_partition(0, 60)
    _, reference_aux = minimax_model.MiniMaxM3Model.forward(
        reference,
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=inputs,
    )

    assert len(pp_aux) == len(reference_aux) == 3
    for actual, expected in zip(pp_aux, reference_aux):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_minimax_pp_output_without_aux_states_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        minimax_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[_IncrementLayer()],
        aux_hidden_state_layers=(),
    )
    model._maybe_add_hidden_state = lambda states, idx, hidden, residual: (
        minimax_model.EagleModelMixin._maybe_add_hidden_state(
            model, states, idx, hidden, residual
        )
    )

    output = minimax_model.MiniMaxM3Model.forward(
        model,
        input_ids=None,
        positions=torch.zeros(1, dtype=torch.int64),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros((1, 1)),
    )

    assert isinstance(output, IntermediateTensors)
    assert tuple(output.tensors) == ("hidden_states", "residual")


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
        config=SimpleNamespace(hidden_size=2),
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
