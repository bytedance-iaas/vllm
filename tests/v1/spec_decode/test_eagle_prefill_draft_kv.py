# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

from vllm.v1.kv_cache_interface import HiddenStateCacheSpec
from vllm.v1.spec_decode.eagle import EagleProposer


class _FakeAttentionLayer:
    def __init__(self, spec) -> None:
        self.spec = spec

    def get_kv_cache_spec(self, _):
        return self.spec


def test_eagle3_prefill_draft_cache_has_no_hidden_state_spec() -> None:
    proposer = EagleProposer.__new__(EagleProposer)
    proposer.speculative_config = mock.MagicMock(enable_eagle3_prefill_draft_kv=True)
    proposer.vllm_config = mock.MagicMock()
    proposer.vllm_config.model_config.get_total_num_hidden_layers.return_value = 60
    proposer._draft_attn_layer_names = {"model.layers.60.self_attn.attn"}
    draft_layer = _FakeAttentionLayer(mock.MagicMock())

    proposer._validate_prefill_draft_kv_layers(
        {"model.layers.60.self_attn.attn": draft_layer}
    )

    hidden_layer = _FakeAttentionLayer(
        HiddenStateCacheSpec(
            block_size=128,
            num_kv_heads=3,
            head_size=6144,
            dtype=torch.bfloat16,
        )
    )
    with pytest.raises(ValueError, match="must not allocate hidden-state cache"):
        proposer._validate_prefill_draft_kv_layers(
            {
                "model.layers.60.self_attn.attn": draft_layer,
                "cache_only_layers.60": hidden_layer,
            }
        )
