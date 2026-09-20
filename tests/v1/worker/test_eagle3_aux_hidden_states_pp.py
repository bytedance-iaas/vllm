# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
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


def test_deepseek_v41_supports_aux_hidden_state_pp_relay():
    inner = DeepseekV4Model.__new__(DeepseekV4Model)
    target = SimpleNamespace(model=inner)

    assert inner.supports_aux_hidden_states_over_pp
    verify_supports_aux_hidden_states_over_pp(target, "dspark")
