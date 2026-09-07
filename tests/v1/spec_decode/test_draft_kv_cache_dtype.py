# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch.nn as nn

import vllm.model_executor.models.utils as model_utils
import vllm.v1.worker.gpu.spec_decode.dflash.utils as dflash_utils
import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils
from vllm.config import CacheConfig
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4Model
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_draft_cache_config

pytestmark = pytest.mark.cpu_test


def _make_config(
    target_dtype: str,
    draft_dtype: str | None,
    *,
    use_mla: bool,
):
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype=target_dtype,
    )
    speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype,
        draft_model_config=SimpleNamespace(use_mla=use_mla),
    )
    return SimpleNamespace(
        cache_config=cache_config,
        speculative_config=speculative_config,
    )


@pytest.mark.parametrize("draft_dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_explicit_draft_kv_dtype_wins_over_dense_fallback(draft_dtype):
    vllm_config = _make_config("fp8_ds_mla", draft_dtype, use_mla=False)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config.cache_dtype == draft_dtype
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache_config is not vllm_config.cache_config


def test_dense_draft_falls_back_from_inherited_fp8_ds_mla():
    vllm_config = _make_config("fp8_ds_mla", None, use_mla=False)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config.cache_dtype == "auto"
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache_config is not vllm_config.cache_config


@pytest.mark.parametrize(
    ("target_dtype", "use_mla"),
    [("fp8_ds_mla", True), ("fp8", False), ("auto", False)],
)
def test_compatible_draft_inherits_target_cache_config(target_dtype, use_mla):
    vllm_config = _make_config(target_dtype, None, use_mla=use_mla)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config is vllm_config.cache_config


def _replace_namespace(obj, **kwargs):
    return SimpleNamespace(**(obj.__dict__ | kwargs))


def _make_load_config(*, target_moe_backend: str, draft_moe_backend: str | None):
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=object()),
        attention_backend="flash_attn",
        kv_cache_dtype=None,
        moe_backend=draft_moe_backend,
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend="target_attention",
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kernel_config=SimpleNamespace(moe_backend=target_moe_backend),
        quant_config=object(),
    )


def _make_language_model(embed_name: str = "embed_tokens"):
    return SimpleNamespace(model=SimpleNamespace(**{embed_name: object()}))


def _install_draft_loader_stubs(monkeypatch, *, has_non_causal: bool):
    @contextmanager
    def fake_set_model_tag(_tag):
        yield

    backend_module = ModuleType("vllm.compilation.backends")
    backend_module.set_model_tag = fake_set_model_tag
    monkeypatch.setitem(sys.modules, "vllm.compilation.backends", backend_module)

    qwen3_dflash_module = ModuleType("vllm.model_executor.models.qwen3_dflash")
    qwen3_dflash_module.dflash_has_any_non_causal = (
        lambda _hf_config: has_non_causal
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_dflash",
        qwen3_dflash_module,
    )


def _fake_loaded_model():
    return SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
    )


@pytest.mark.parametrize(
    ("draft_backend", "expected_backend"),
    [("marlin", "marlin"), (None, "deep_gemm_mega_moe")],
)
def test_load_dflash_model_uses_draft_moe_backend_without_mutating_target(
    monkeypatch,
    draft_backend,
    expected_backend,
):
    captured = {}

    def fake_get_model(*, vllm_config, model_config):
        captured["config"] = vllm_config
        assert model_config is vllm_config.speculative_config.draft_model_config
        return _fake_loaded_model()

    monkeypatch.setattr(dflash_utils, "replace", _replace_namespace)
    monkeypatch.setattr(dflash_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(dflash_utils, "_should_share", lambda *_args: False)
    _install_draft_loader_stubs(monkeypatch, has_non_causal=True)

    target_model = SimpleNamespace(
        get_language_model=lambda: _make_language_model(),
        lm_head=object(),
    )
    config = _make_load_config(
        target_moe_backend="deep_gemm_mega_moe",
        draft_moe_backend=draft_backend,
    )
    target_attention = config.attention_config
    target_cache = config.cache_config
    target_kernel = config.kernel_config

    dflash_utils.load_dflash_model(target_model, config)

    draft_config = captured["config"]
    assert draft_config.kernel_config.moe_backend == expected_backend
    assert config.kernel_config is target_kernel
    assert config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    assert draft_config.attention_config.backend == "flash_attn"
    assert draft_config.attention_config.use_non_causal is True
    assert draft_config.cache_config is target_cache
    assert config.attention_config is target_attention
    assert config.attention_config.backend == "target_attention"


@pytest.mark.parametrize(
    ("draft_backend", "expected_backend"),
    [("marlin", "marlin"), (None, "deep_gemm_mega_moe")],
)
def test_load_dspark_model_preserves_draft_kernel_and_quant_config(
    monkeypatch,
    draft_backend,
    expected_backend,
):
    captured = {}
    draft_quant = object()

    def fake_get_model(*, vllm_config, model_config):
        captured["config"] = vllm_config
        assert model_config is vllm_config.speculative_config.draft_model_config
        return _fake_loaded_model()

    monkeypatch.setattr(dspark_utils, "replace", _replace_namespace)
    monkeypatch.setattr(dspark_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        dspark_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(dspark_utils, "_should_share", lambda *_args: False)
    monkeypatch.setattr(dflash_utils, "replace", _replace_namespace)
    monkeypatch.setattr(
        model_utils,
        "get_draft_quant_config",
        lambda _config: draft_quant,
    )
    _install_draft_loader_stubs(monkeypatch, has_non_causal=False)

    target_model = SimpleNamespace(
        get_language_model=lambda: _make_language_model(),
        lm_head=object(),
    )
    config = _make_load_config(
        target_moe_backend="deep_gemm_mega_moe",
        draft_moe_backend=draft_backend,
    )
    target_attention = config.attention_config
    target_cache = config.cache_config
    target_kernel = config.kernel_config
    target_quant = config.quant_config

    dspark_utils.load_dspark_model(target_model, config)

    draft_config = captured["config"]
    assert draft_config.kernel_config.moe_backend == expected_backend
    assert draft_config.quant_config is draft_quant
    assert draft_config.attention_config.backend == "flash_attn"
    assert draft_config.attention_config.use_non_causal is False
    assert draft_config.cache_config is target_cache
    assert config.kernel_config is target_kernel
    assert config.quant_config is target_quant
    assert config.attention_config is target_attention
    assert config.attention_config.backend == "target_attention"


def test_dspark_layers_use_passed_draft_config_and_topk_buffer(monkeypatch):
    captured_layers = []

    class FakeModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeDecoderLayer(nn.Module):
        def __init__(self, vllm_config, prefix, *args, **kwargs):
            super().__init__()
            captured_layers.append(
                (vllm_config, prefix, kwargs["topk_indices_buffer"])
            )

    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.VocabParallelEmbedding",
        FakeModule,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.ReplicatedLinear",
        FakeModule,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.RMSNorm",
        FakeModule,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.DSparkMarkovHead",
        FakeModule,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.DeepseekV4DecoderLayer",
        FakeDecoderLayer,
    )

    draft_hf_config = SimpleNamespace(
        hidden_size=16,
        hc_mult=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=61,
        dspark_target_layer_ids=(58, 59, 60),
        n_mtp_layers=2,
        vocab_size=128,
        dspark_markov_rank=4,
        index_topk=8,
    )
    draft_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        ),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="marlin"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=2,
            data_parallel_size=1,
        ),
    )

    model = DSparkDeepseekV4Model(vllm_config=draft_config)

    assert len(captured_layers) == draft_hf_config.n_mtp_layers
    assert not model.use_sequence_parallel
    assert model.topk_indices_buffer.shape == (32, 8)
    assert all(config is draft_config for config, _, _ in captured_layers)
    assert all(buffer is model.topk_indices_buffer for _, _, buffer in captured_layers)
    assert isinstance(model.layers, nn.ModuleList)
