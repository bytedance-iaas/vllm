# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from transformers import PretrainedConfig

from vllm.transformers_utils import config as config_utils


@pytest.mark.parametrize("quant_method", ["fp8", "deepseek_v4_fp8"])
def test_resolve_dsv4_expert_dtype_from_selected_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    quant_method: str,
):
    observed: dict[str, object] = {}
    expert_name = "model.layers.0.mlp.experts.0.weight"

    def fake_metadata(model: str, *, revision: str | None = None):
        observed.update(model=model, revision=revision)
        return {
            expert_name: {"dtype": "F8_E4M3"},
        }

    def fake_index(file_name, model, revision):
        observed.update(
            index_file=file_name,
            index_model=model,
            index_revision=revision,
        )
        return {"weight_map": {expert_name: "model-00001-of-00002.safetensors"}}

    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        config_utils,
        "get_hf_file_to_dict",
        fake_index,
    )
    config = PretrainedConfig(
        quantization_config={"quant_method": quant_method},
    )
    config.model_type = "deepseek_v4"

    result = config_utils._maybe_resolve_dsv4_expert_dtype(
        config,
        "org/deepseek-v4-fp8",
        revision="production",
    )

    assert result is config
    assert config.expert_dtype == "fp8"
    assert observed == {
        "model": "org/deepseek-v4-fp8",
        "revision": "production",
        "index_file": config_utils.constants.SAFETENSORS_INDEX_FILE,
        "index_model": "org/deepseek-v4-fp8",
        "index_revision": "production",
    }


def test_resolve_dsv4_expert_dtype_keeps_explicit_value(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_metadata(*args, **kwargs):
        raise AssertionError("metadata lookup should not run")

    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        unexpected_metadata,
    )
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8"},
        expert_dtype="fp4",
    )
    config.model_type = "deepseek_v4"

    result = config_utils._maybe_resolve_dsv4_expert_dtype(config, "unused")

    assert result is config
    assert config.expert_dtype == "fp4"


def test_resolve_dsv4_expert_dtype_ignores_non_fp8_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: {
            "model.layers.0.mlp.experts.0.weight": {"dtype": "U8"},
        },
    )
    monkeypatch.setattr(
        config_utils,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: None,
    )
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8"},
    )
    config.model_type = "deepseek_v4"

    result = config_utils._maybe_resolve_dsv4_expert_dtype(config, "model")

    assert result is config
    assert not hasattr(config, "expert_dtype")


def test_resolve_dsv4_expert_dtype_ignores_unselected_fp8_expert(
    monkeypatch: pytest.MonkeyPatch,
):
    expert_name = "model.layers.0.mlp.experts.0.weight"
    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: {
            expert_name: {"dtype": "F8_E4M3"},
        },
    )
    monkeypatch.setattr(
        config_utils,
        "get_hf_file_to_dict",
        lambda file_name, model, revision: {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": (
                    "model-00001-of-00002.safetensors"
                )
            }
        },
    )
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8"},
    )
    config.model_type = "deepseek_v4"

    result = config_utils._maybe_resolve_dsv4_expert_dtype(config, "model")

    assert result is config
    assert not hasattr(config, "expert_dtype")


def test_get_config_infers_dsv4_expert_dtype_from_legacy_quant_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
            }
        )
    )
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quant_method": "fp8"}))
    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: {
            "model.layers.0.mlp.experts.0.weight": {"dtype": "F8_E4M3"},
        },
    )

    config = config_utils.get_config(tmp_path, trust_remote_code=False)

    assert config.quantization_config == {"quant_method": "fp8"}
    assert config.expert_dtype == "fp8"
