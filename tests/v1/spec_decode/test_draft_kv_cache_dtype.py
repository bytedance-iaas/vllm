# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch.nn as nn

import vllm.v1.worker.gpu.spec_decode.dflash.utils as dflash_utils
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


def _make_load_dflash_config(*, target_moe_backend: str, draft_moe_backend: str | None):
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
    )


def _make_language_model(embed_name: str = "embed_tokens"):
    return SimpleNamespace(model=SimpleNamespace(**{embed_name: object()}))


def _install_load_dflash_import_stubs(monkeypatch, *, has_non_causal: bool):
    @contextmanager
    def fake_set_model_tag(_tag):
        yield

    backend_module = ModuleType("vllm.compilation.backends")
    backend_module.set_model_tag = fake_set_model_tag
    monkeypatch.setitem(sys.modules, "vllm.compilation.backends", backend_module)

    qwen3_dflash_module = ModuleType("vllm.model_executor.models.qwen3_dflash")
    qwen3_dflash_module.dflash_has_any_non_causal = lambda _hf_config: has_non_causal
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_dflash",
        qwen3_dflash_module,
    )


def test_load_dflash_model_applies_explicit_draft_moe_backend(monkeypatch):
    captured_config = None

    def fake_get_model(*, vllm_config, model_config):
        nonlocal captured_config
        captured_config = vllm_config
        assert model_config is vllm_config.speculative_config.draft_model_config
        return SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object()),
            lm_head=object(),
        )

    monkeypatch.setattr(dflash_utils, "replace", _replace_namespace)
    monkeypatch.setattr(dflash_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(dflash_utils, "_should_share", lambda *_args: False)
    _install_load_dflash_import_stubs(monkeypatch, has_non_causal=True)

    target_model = SimpleNamespace(
        get_language_model=lambda: _make_language_model(),
        lm_head=object(),
    )
    vllm_config = _make_load_dflash_config(
        target_moe_backend="deep_gemm_mega_moe",
        draft_moe_backend="marlin",
    )

    dflash_utils.load_dflash_model(target_model, vllm_config)

    assert captured_config is not None
    assert captured_config.kernel_config.moe_backend == "marlin"
    assert vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    assert captured_config.attention_config.backend == "flash_attn"
    assert captured_config.attention_config.use_non_causal is True


def test_load_dflash_model_inherits_target_moe_backend_when_unset(monkeypatch):
    captured_config = None

    def fake_get_model(*, vllm_config, model_config):
        nonlocal captured_config
        captured_config = vllm_config
        assert model_config is vllm_config.speculative_config.draft_model_config
        return SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object()),
            lm_head=object(),
        )

    monkeypatch.setattr(dflash_utils, "replace", _replace_namespace)
    monkeypatch.setattr(dflash_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(dflash_utils, "_should_share", lambda *_args: False)
    _install_load_dflash_import_stubs(monkeypatch, has_non_causal=False)

    target_model = SimpleNamespace(
        get_language_model=lambda: _make_language_model(),
        lm_head=object(),
    )
    vllm_config = _make_load_dflash_config(
        target_moe_backend="deep_gemm_mega_moe",
        draft_moe_backend=None,
    )

    dflash_utils.load_dflash_model(target_model, vllm_config)

    assert captured_config is not None
    assert captured_config.kernel_config is vllm_config.kernel_config
    assert captured_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    assert captured_config.attention_config.backend == "flash_attn"
    assert captured_config.attention_config.use_non_causal is False


def test_dspark_layers_use_passed_draft_vllm_config(monkeypatch):
    captured_layer_configs = []

    class FakeModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeDecoderLayer(nn.Module):
        def __init__(self, vllm_config, prefix, *args, **kwargs):
            super().__init__()
            captured_layer_configs.append((vllm_config, prefix))

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
    )
    draft_vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        ),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="marlin"),
    )
    target_vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
    )

    model = DSparkDeepseekV4Model(vllm_config=draft_vllm_config)

    assert len(captured_layer_configs) == draft_hf_config.n_mtp_layers
    assert all(config is draft_vllm_config for config, _ in captured_layer_configs)
    assert all(
        config.kernel_config.moe_backend == "marlin"
        for config, _ in captured_layer_configs
    )
    assert all(
        config.kernel_config.moe_backend != target_vllm_config.kernel_config.moe_backend
        for config, _ in captured_layer_configs
    )
    assert isinstance(model.layers, nn.ModuleList)
