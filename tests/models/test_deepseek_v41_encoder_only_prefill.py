# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.interfaces import requires_raw_input_tokens
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.pipeline import PipelineCacheRelay
from vllm.models.deepseek_v41.nvidia import model as dsv41_model
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_prefill_pp_keeps_input_ids_for_second_stage_engram():
    assert requires_raw_input_tokens(DeepseekV41ForCausalLM)


def test_official_index_sources_do_not_move_encoder_only_boundary(monkeypatch):
    """Later indexers share L20 K; they must not move the producer cut."""

    class ConstructionReached(Exception):
        pass

    def stop_after_topology_check():
        raise ConstructionReached

    monkeypatch.setattr(dsv41_model, "_use_sequence_parallel", lambda _: False)
    monkeypatch.setattr(
        dsv41_model, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(dsv41_model, "get_sharing_dependencies", lambda *_: ())
    monkeypatch.setattr(
        dsv41_model, "validate_local_sharing", lambda _, relay=None: None
    )
    monkeypatch.setattr(dsv41_model.torch.cuda, "Stream", stop_after_topology_check)
    hf_config = SimpleNamespace(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        vocab_size=128,
        hc_eps=1e-6,
        hc_mult=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        kernel_config=SimpleNamespace(moe_backend="CUTLASS"),
        is_dsv41_encoder_only_prefill=True,
    )

    with pytest.raises(ConstructionReached):
        dsv41_model.DeepseekV4Model(vllm_config=vllm_config)


class _Embedding(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(1).expand(-1, 2)


class _RecordingLayer(dsv41_model.DeepseekV4DecoderLayer):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.full_calls = 0
        self.cache_calls = 0
        self.global_cache: torch.Tensor | None = None

    def forward(self, x, *args, **kwargs):
        self.full_calls += 1
        x = x + 1
        state = torch.ones_like(x)
        return x, state, state, state, state, None

    def write_global_cache(self, x, *args, **kwargs):
        self.cache_calls += 1
        self.global_cache = x.clone()
        return x


def test_encoder_only_prefill_stops_at_global_cache_boundary(monkeypatch):
    """The 40-layer path runs 20 full layers plus the L20 producer only."""
    monkeypatch.setattr(
        dsv41_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    layers = [_RecordingLayer() for _ in range(40)]
    model.embed_tokens = _Embedding()
    model.layers = torch.nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.decoder_replay_start = 21
    model.use_mega_moe = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = set()
    model.encoder_only_prefill = True
    model.encoder_only_boundary_layer = 20

    input_ids = torch.tensor([2, 3])
    output = model(input_ids, torch.arange(2), None)

    expected_boundary_input = model.embed_input_ids(input_ids) + 20
    torch.testing.assert_close(output, expected_boundary_input)
    torch.testing.assert_close(layers[20].global_cache, expected_boundary_input)
    assert sum(layer.full_calls for layer in layers) == 20
    assert sum(layer.cache_calls for layer in layers) == 1
    assert all(layer.full_calls == 0 for layer in layers[20:])


@pytest.mark.parametrize("cut", [8, 14])
def test_encoder_only_prefill_pp_hands_off_before_shared_source(monkeypatch, cut):
    """Both stages match the single-stage L20 cache-writer input."""
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr(dsv41_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(dsv41_model, "mhc_post_tilelang", lambda x, *_: x)
    layers = [_RecordingLayer() for _ in range(40)]
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.embed_tokens = _Embedding()
    model.layers = torch.nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = cut
    model.decoder_replay_start = cut
    model.fuse_mhc_all_reduce = False
    model.use_mega_moe = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = set()
    model.encoder_only_prefill = True
    model.encoder_only_boundary_layer = 20

    input_ids = torch.tensor([2, 3])
    intermediate = model(input_ids, torch.arange(2), None)
    assert isinstance(intermediate, dsv41_model.IntermediateTensors)
    assert sum(layer.full_calls for layer in layers) == cut
    assert sum(layer.cache_calls for layer in layers) == 0

    pp_group.is_first_rank = False
    pp_group.is_last_rank = True
    model.start_layer = cut
    model.end_layer = 40
    model.decoder_replay_start = 21
    output = model(input_ids, torch.arange(2), intermediate)

    expected = model.embed_input_ids(input_ids) + 20
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(layers[20].global_cache, expected)
    assert sum(layer.full_calls for layer in layers) == 20
    assert sum(layer.cache_calls for layer in layers) == 1


def test_global_cache_writer_publishes_same_latent_to_main_and_indexer():
    positions = torch.arange(4)
    hidden_states = torch.randn(4, 8)
    score = torch.randn(4, 8)
    latent = torch.randn(4, 8)
    compressor = Mock(return_value=latent)
    indexer = SimpleNamespace(owns_k=True, insert_cache=Mock())
    attention = SimpleNamespace(
        compressor=compressor,
        indexer=indexer,
        is_kv_source=True,
        layer_id=20,
        aux_stream_list=None,
        ln_events=[None, None],
        indexer_rotary_emb=object(),
        rotary_emb=object(),
        _compressor_kv_score=Mock(return_value=score),
    )

    DeepseekV4Attention.write_global_cache(attention, positions, hidden_states)

    attention._compressor_kv_score.assert_called_once_with(hidden_states)
    compressor.assert_called_once_with(score, positions)
    compressor.insert_cache.assert_called_once_with(
        latent, positions, attention.rotary_emb
    )
    indexer.insert_cache.assert_called_once_with(
        latent, positions, attention.indexer_rotary_emb
    )


def _cache_relay() -> PipelineCacheRelay:
    return PipelineCacheRelay(
        source_layer=8,
        receiver_layer=10,
        source_stage=0,
        receiver_stage=1,
        consumer_layers=(10, 11, 12, 13),
    )


def test_pipeline_cache_relay_allocates_typed_intermediates(monkeypatch):
    monkeypatch.setattr(
        dsv41_model,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    monkeypatch.setattr(
        dsv41_model,
        "get_forward_context",
        lambda: SimpleNamespace(is_profile=False),
    )
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.config = SimpleNamespace(hidden_size=16, index_topk=32)
    model.pipeline_cache_relay = _cache_relay()

    tensors = model.make_empty_intermediate_tensors(
        batch_size=7, dtype=torch.bfloat16, device=torch.device("cpu")
    )

    assert tensors["hidden_states"].shape == (7, 4, 16)
    assert tensors["pre_mix"].shape == (7, 4)
    assert tensors["dsv41_relay_l8_main_kv"].shape == (7, 584)
    assert tensors["dsv41_relay_l8_main_kv"].dtype == torch.uint8
    assert tensors["dsv41_relay_l8_topk"].shape == (7, 32)
    assert tensors["dsv41_relay_l8_topk"].dtype == torch.int32


def test_pipeline_cache_relay_imports_before_consumers(monkeypatch):
    monkeypatch.setattr(
        dsv41_model,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    monkeypatch.setattr(
        dsv41_model,
        "get_forward_context",
        lambda: SimpleNamespace(is_profile=False),
    )
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    layers = [_RecordingLayer() for _ in range(40)]
    layers[10].attn = SimpleNamespace(import_cross_stage_kv=Mock())
    model.layers = torch.nn.ModuleList(layers)
    model.config = SimpleNamespace(index_topk=4)
    model.pipeline_cache_relay = _cache_relay()
    model.topk_indices_buffer = torch.empty((8, 4), dtype=torch.int32)
    positions = torch.arange(3)
    packed_kv = torch.randint(0, 256, (3, 584), dtype=torch.uint8)
    topk = torch.arange(12, dtype=torch.int32).view(3, 4)
    intermediate = dsv41_model.IntermediateTensors(
        {
            "dsv41_relay_l8_main_kv": packed_kv,
            "dsv41_relay_l8_topk": topk,
        }
    )

    model._import_pipeline_cache_relay(intermediate, positions)

    layers[10].attn.import_cross_stage_kv.assert_called_once_with(packed_kv, positions)
    torch.testing.assert_close(model.topk_indices_buffer[:3], topk)


def test_pipeline_cache_relay_exports_source_payload(monkeypatch):
    monkeypatch.setattr(
        dsv41_model,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(
        dsv41_model,
        "get_forward_context",
        lambda: SimpleNamespace(is_profile=False),
    )
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    layers = [_RecordingLayer() for _ in range(40)]
    packed_kv = torch.randint(0, 256, (3, 584), dtype=torch.uint8)
    layers[8].attn = SimpleNamespace(export_cross_stage_kv=Mock(return_value=packed_kv))
    model.layers = torch.nn.ModuleList(layers)
    model.config = SimpleNamespace(index_topk=4)
    model.pipeline_cache_relay = _cache_relay()
    model.topk_indices_buffer = torch.arange(32, dtype=torch.int32).view(8, 4)
    positions = torch.arange(3)

    payload = model._export_pipeline_cache_relay(positions)

    layers[8].attn.export_cross_stage_kv.assert_called_once_with(positions)
    assert payload["dsv41_relay_l8_main_kv"] is packed_kv
    torch.testing.assert_close(
        payload["dsv41_relay_l8_topk"], model.topk_indices_buffer[:3]
    )


def test_pipeline_cache_relay_import_precedes_layer_10_forward(monkeypatch):
    events = []
    pp_group = SimpleNamespace(
        is_first_rank=False,
        is_last_rank=True,
        rank_in_group=1,
    )
    monkeypatch.setattr(dsv41_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        dsv41_model,
        "get_forward_context",
        lambda: SimpleNamespace(is_profile=False),
    )
    monkeypatch.setattr(dsv41_model, "mhc_post_tilelang", lambda x, *_: x)
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    layers = [_RecordingLayer() for _ in range(40)]
    receiver_forward = layers[10].forward

    def ordered_forward(*args, **kwargs):
        events.append("layer10")
        return receiver_forward(*args, **kwargs)

    layers[10].forward = ordered_forward
    layers[10].attn = SimpleNamespace(
        import_cross_stage_kv=Mock(side_effect=lambda *_: events.append("import"))
    )
    model.layers = torch.nn.ModuleList(layers)
    model.start_layer = 10
    model.end_layer = 40
    model.decoder_replay_start = 21
    model.fuse_mhc_all_reduce = False
    model.use_mega_moe = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = set()
    model.encoder_only_prefill = True
    model.encoder_only_boundary_layer = 20
    model.config = SimpleNamespace(index_topk=4)
    model.pipeline_cache_relay = _cache_relay()
    model.topk_indices_buffer = torch.empty((8, 4), dtype=torch.int32)
    positions = torch.arange(2)
    intermediate = dsv41_model.IntermediateTensors(
        {
            "hidden_states": torch.ones((2, 2)),
            "pre_mix": torch.ones((2, 2)),
            "dsv41_relay_l8_main_kv": torch.zeros((2, 584), dtype=torch.uint8),
            "dsv41_relay_l8_topk": torch.zeros((2, 4), dtype=torch.int32),
        }
    )

    model(torch.tensor([2, 3]), positions, intermediate)

    assert events[:2] == ["import", "layer10"]
