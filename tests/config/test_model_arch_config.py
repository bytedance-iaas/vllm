# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelArchitectureConfig and its integration with ModelConfig."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig, PretrainedConfig

import vllm.config.speculative as speculative_config_module
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

BASE_TRUST_REMOTE_CODE_MODELS = {
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "XiaomiMiMo/MiMo-7B-RL",
    "stepfun-ai/Step-3.5-Flash",
    # Excluded: Not available online right now
    # "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1",
    "meituan-longcat/LongCat-Flash-Chat",
}

BASE_MODELS_TO_TEST = [
    "state-spaces/mamba-130m-hf",
    "mistralai/Mamba-Codestral-7B-v0.1",
    # Excluded: terratorch/torchgeo version mismatch in CPU CI environment
    # (NonGeoDataset import error). Tested in model initialization tests.
    # "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "Zyphra/Zamba2-7B-instruct",
    # FIXME: mosaicml/mpt-7b has been deleted
    # "mosaicml/mpt-7b",
    # FIXME: databricks/dbrx-instruct has been deleted
    # "databricks/dbrx-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "luccafong/deepseek_mtp_main_random",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "tiny-random/qwen3-next-moe",
    "zai-org/GLM-4.5",
    "baidu/ERNIE-4.5-21B-A3B-PT",
    # Models using base convertor
    "lmsys/gpt-oss-20b-bf16",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
] + list(BASE_TRUST_REMOTE_CODE_MODELS)

# (target_model, draft_model, trust_remote_code)
SPECULATIVE_MODELS = [
    ("JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False),
    ("luccafong/deepseek_mtp_main_random", "luccafong/deepseek_mtp_draft_random", True),
    ("eagle618/deepseek-v3-random", "eagle618/eagle-deepseek-v3-random", True),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "yuhuili/EAGLE-LLaMA3-Instruct-8B", True),
    ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", True),
]


def _load_groundtruth(filename: str) -> dict:
    """Load groundtruth JSON from the test directory."""
    groundtruth_path = Path(__file__).parent / filename
    with open(groundtruth_path) as f:
        return json.load(f)


def _assert_model_arch_config(
    model_config, expected: dict, check_head_size: bool = True
):
    """Assert model_arch_config matches expected values."""
    model_arch_config = model_config.model_arch_config
    assert model_arch_config.architectures == expected["architectures"]
    assert model_arch_config.model_type == expected["model_type"]
    assert model_arch_config.text_model_type == expected["text_model_type"]
    assert model_arch_config.hidden_size == expected["hidden_size"]
    assert (
        model_arch_config.total_num_hidden_layers == expected["total_num_hidden_layers"]
    )
    assert (
        model_arch_config.total_num_attention_heads
        == expected["total_num_attention_heads"]
    )
    assert model_arch_config.vocab_size == expected["vocab_size"]
    assert model_arch_config.total_num_kv_heads == expected["total_num_kv_heads"]
    assert model_arch_config.num_experts == expected["num_experts"]
    assert model_arch_config.is_deepseek_mla == expected["is_deepseek_mla"]

    torch_dtype = ModelArchConfigConvertorBase.get_torch_dtype(
        model_config.hf_config,
        model_config.model,
        revision=model_config.revision,
        config_format="hf",
    )
    assert str(torch_dtype) == expected["dtype"]

    if check_head_size:
        assert model_arch_config.head_size == expected["head_size"]


def _assert_model_config_methods(
    model_config, expected: dict, check_head_size: bool = True
):
    """Assert model_config methods return expected values."""
    assert model_config.architectures == expected["architectures"]
    assert model_config.get_vocab_size() == expected["vocab_size"]
    assert model_config.get_hidden_size() == expected["hidden_size"]
    assert model_config.get_total_num_kv_heads() == expected["total_num_kv_heads"]
    assert model_config.get_num_experts() == expected["num_experts"]
    assert (
        model_config.get_total_num_hidden_layers()
        == expected["total_num_hidden_layers"]
    )

    if check_head_size:
        assert model_config.get_head_size() == expected["head_size"]


def test_head_size_falls_back_when_head_dim_is_zero():
    """Regression test for configs that materialize missing head_dim as 0."""
    hf_config = PretrainedConfig(
        model_type="deepseek_vl_v2",
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=10,
        head_dim=0,
        kv_lora_rank=None,
    )

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_head_size() == 128


def test_legacy_modelopt_config_without_producer_is_normalized():
    quantization_config = {
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "modelopt_quant_config": {"quant_cfg": {}},
        }
    }
    hf_config = PretrainedConfig(quantization_config=quantization_config)

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_quantization_config()["quant_method"] == "modelopt_fp4"


@pytest.mark.parametrize("model", BASE_MODELS_TO_TEST)
def test_base_model_arch_config(model: str):
    """Test model architecture config for base models."""
    groundtruth = _load_groundtruth("base_model_arch_groundtruth.json")
    expected = groundtruth[model]

    model_config = ModelConfig(
        model, trust_remote_code=model in BASE_TRUST_REMOTE_CODE_MODELS
    )

    _assert_model_arch_config(model_config, expected)
    _assert_model_config_methods(model_config, expected)


@pytest.mark.parametrize(
    "target_model,draft_model,trust_remote_code", SPECULATIVE_MODELS
)
def test_draft_model_arch_config(
    target_model: str, draft_model: str, trust_remote_code: bool
):
    """Test model architecture config for draft/speculative models."""
    groundtruth = _load_groundtruth("draft_model_arch_groundtruth.json")
    expected = groundtruth[draft_model]

    target_model_config = ModelConfig(target_model, trust_remote_code=trust_remote_code)
    speculative_config = SpeculativeConfig(
        model=draft_model,
        num_speculative_tokens=1,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )
    model_config = speculative_config.draft_model_config

    # For medusa models, head_size may cause division by zero before
    # model_arch_config was introduced, so we conditionally check it
    check_head_size = isinstance(expected["head_size"], int)

    _assert_model_arch_config(model_config, expected, check_head_size=check_head_size)
    _assert_model_config_methods(
        model_config, expected, check_head_size=check_head_size
    )


def _make_local_llama_model_config(tmp_path: Path) -> ModelConfig:
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=8,
        vocab_size=32000,
    ).save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), runner="generate", max_model_len=100)


def test_extract_hidden_states_allows_supported_pipeline_parallel_target(
    tmp_path,
    monkeypatch,
):
    model_config = _make_local_llama_model_config(tmp_path)
    model_config.hf_text_config.supports_pp_aux_hidden_state_transport = True
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )
    parallel_config = ParallelConfig(pipeline_parallel_size=2)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
            }
        },
    )

    assert MiniMaxM3TextConfig.supports_pp_aux_hidden_state_transport
    assert speculative_config.draft_parallel_config is parallel_config


def test_extract_hidden_states_rejects_unsupported_pipeline_parallel_target(
    tmp_path,
    monkeypatch,
):
    model_config = _make_local_llama_model_config(tmp_path)
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    with pytest.raises(
        NotImplementedError,
        match="does not support pipeline-parallel auxiliary hidden-state transport",
    ):
        SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(pipeline_parallel_size=2),
            method="extract_hidden_states",
            num_speculative_tokens=1,
            draft_model_config={
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
                }
            },
        )


def test_extract_hidden_states_rejects_non_cuda_pipeline_parallel_target(
    tmp_path,
    monkeypatch,
):
    model_config = _make_local_llama_model_config(tmp_path)
    model_config.hf_text_config.supports_pp_aux_hidden_state_transport = True
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: False,
    )

    with pytest.raises(
        NotImplementedError,
        match="does not support pipeline-parallel auxiliary hidden-state transport",
    ):
        SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(pipeline_parallel_size=2),
            method="extract_hidden_states",
            num_speculative_tokens=1,
            draft_model_config={
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
                }
            },
        )


def _prefill_draft_kv_config() -> SimpleNamespace:
    return SimpleNamespace(
        method="eagle3",
        num_speculative_tokens=1,
        target_model_config=SimpleNamespace(
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            get_total_num_hidden_layers=lambda: 60,
        ),
        target_parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
        ),
        parallel_drafting=False,
        uses_dynamic_speculative_decoding=lambda: False,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=1,
                num_attention_heads=64,
                num_key_value_heads=64,
                head_dim=128,
            )
        ),
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )


def test_minimax_eagle3_prefill_draft_kv_accepts_supported_config(
    monkeypatch,
) -> None:
    config = _prefill_draft_kv_config()
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    SpeculativeConfig._verify_eagle3_prefill_draft_kv(config)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("method", "eagle", "method must be eagle3"),
        ("num_speculative_tokens", 3, "num_speculative_tokens must be 1"),
    ],
)
def test_minimax_eagle3_prefill_draft_kv_rejects_unsupported_config(
    monkeypatch,
    field: str,
    value,
    expected: str,
) -> None:
    config = _prefill_draft_kv_config()
    setattr(config, field, value)
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    with pytest.raises(ValueError, match=expected):
        SpeculativeConfig._verify_eagle3_prefill_draft_kv(config)


def _replicated_draft_kv_config(dcp_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        method="eagle3",
        num_speculative_tokens=3,
        target_model_config=SimpleNamespace(
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            get_total_num_hidden_layers=lambda: 60,
        ),
        target_parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=8,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            use_ubatching=False,
        ),
        parallel_drafting=False,
        uses_dynamic_speculative_decoding=lambda: False,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=1,
                num_attention_heads=64,
                num_key_value_heads=64,
                head_dim=128,
            )
        ),
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_minimax_eagle3_replicated_draft_kv_accepts_supported_config(
    monkeypatch,
    dcp_size: int,
) -> None:
    config = _replicated_draft_kv_config(dcp_size)
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    SpeculativeConfig._verify_eagle3_replicated_draft_kv(config)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("num_speculative_tokens", 1, "num_speculative_tokens must be 3"),
        (
            "target_parallel_config.decode_context_parallel_size",
            3,
            "decode_context_parallel_size must be 1 or 2",
        ),
    ],
)
def test_minimax_eagle3_replicated_draft_kv_rejects_unsupported_config(
    monkeypatch,
    field: str,
    value,
    expected: str,
) -> None:
    config = _replicated_draft_kv_config()
    target = config
    parts = field.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    monkeypatch.setattr(
        speculative_config_module.current_platform,
        "is_cuda",
        lambda: True,
    )

    with pytest.raises(ValueError, match=expected):
        SpeculativeConfig._verify_eagle3_replicated_draft_kv(config)


def test_minimax_eagle3_target_dense_full_temporal_kv_defers_execution_mode() -> None:
    eager_config = SimpleNamespace(
        enable_eagle3_replicated_draft_kv=True,
        target_model_config=SimpleNamespace(enforce_eager=True),
    )
    SpeculativeConfig._verify_eagle3_target_dense_full_temporal_kv(eager_config)

    graph_config = SimpleNamespace(
        enable_eagle3_replicated_draft_kv=True,
        target_model_config=SimpleNamespace(enforce_eager=False),
    )
    SpeculativeConfig._verify_eagle3_target_dense_full_temporal_kv(graph_config)

    prefill_graph_config = SimpleNamespace(
        enable_eagle3_replicated_draft_kv=False,
        target_model_config=SimpleNamespace(enforce_eager=False),
    )
    SpeculativeConfig._verify_eagle3_target_dense_full_temporal_kv(prefill_graph_config)
