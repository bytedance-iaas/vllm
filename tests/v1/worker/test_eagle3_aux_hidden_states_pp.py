# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
    verify_supports_aux_hidden_states_over_pp,
)


def test_aux_layers_are_sorted_and_deduplicated():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24, 48))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


def test_mimo_does_not_inherit_aux_hidden_state_pp_support():
    inner = MiMoModel.__new__(MiMoModel)
    target = SimpleNamespace(model=inner)

    assert not inner.supports_aux_hidden_states_over_pp
    with pytest.raises(ValueError, match="does not support eagle3"):
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


def _raw_layer_config(target_architecture: str):
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(eagle_config={"layer_ids": [0, 10, 20]})
        ),
        target_model_config=SimpleNamespace(architectures=[target_architecture]),
    )


def test_generic_aux_layers_preserve_embedding_layer_zero():
    config = _raw_layer_config("LlamaForCausalLM")

    assert get_eagle3_aux_layers_from_config(config) == (0, 10, 20)


def test_deepseek_v4_aux_layers_drop_unavailable_embedding_state():
    config = _raw_layer_config("DeepseekV4ForCausalLM")

    assert get_eagle3_aux_layers_from_config(config) == (10, 20)
