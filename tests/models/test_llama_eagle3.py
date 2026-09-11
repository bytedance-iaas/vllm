# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.llama_eagle3 import (
    _get_eagle3_target_layer_count,
)


def _config(*, prefill_draft_kv: bool, total_layers: int, local_layers: int):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            enable_eagle3_prefill_draft_kv=prefill_draft_kv
        ),
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: total_layers,
            get_num_layers=lambda _: local_layers,
        ),
        parallel_config=SimpleNamespace(),
    )


def test_eagle3_prefill_draft_uses_global_target_layer_count() -> None:
    config = _config(
        prefill_draft_kv=True,
        total_layers=60,
        local_layers=30,
    )

    assert _get_eagle3_target_layer_count(config) == 60


def test_eagle3_default_path_keeps_local_target_layer_count() -> None:
    config = _config(
        prefill_draft_kv=False,
        total_layers=60,
        local_layers=30,
    )

    assert _get_eagle3_target_layer_count(config) == 30


def test_eagle3_prefill_draft_rejects_non_minimax_layer_count() -> None:
    config = _config(
        prefill_draft_kv=True,
        total_layers=30,
        local_layers=15,
    )

    with pytest.raises(ValueError, match="60 global target layers"):
        _get_eagle3_target_layer_count(config)
