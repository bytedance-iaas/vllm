# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.transformers_utils import config as config_utils


@pytest.mark.parametrize("quant_method", ["fp8", "deepseek_v4_fp8"])
def test_resolve_dsv4_expert_dtype_from_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    quant_method: str,
):
    observed: dict[str, str | None] = {}

    def fake_metadata(model: str, *, revision: str | None = None):
        observed.update(model=model, revision=revision)
        return {
            "model.layers.0.mlp.experts.0.weight": {"dtype": "F8_E4M3"},
        }

    monkeypatch.setattr(
        config_utils,
        "get_safetensors_params_metadata",
        fake_metadata,
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
    config = PretrainedConfig(
        quantization_config={"quant_method": "fp8"},
    )
    config.model_type = "deepseek_v4"

    result = config_utils._maybe_resolve_dsv4_expert_dtype(config, "model")

    assert result is config
    assert not hasattr(config, "expert_dtype")
