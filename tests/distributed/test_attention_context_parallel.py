# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.distributed.parallel_state import (
    _get_attention_parallel_group_ranks,
)


def test_attention_context_parallel_default_preserves_world_size():
    config = ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)

    assert config.attention_context_parallel_size == 1
    assert config.world_size == 8


def test_attention_context_parallel_does_not_increase_world_size():
    config = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        attention_context_parallel_size=2,
    )

    assert config.world_size == 8


def test_attention_context_parallel_requires_divisible_tp():
    with pytest.raises(
        ValueError,
        match="tp_size=4 must be divisible by attention_context_parallel_size=3",
    ):
        ParallelConfig(
            tensor_parallel_size=4,
            attention_context_parallel_size=3,
        )


@pytest.mark.parametrize(
    "incompatible_config",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_attention_context_parallel_rejects_other_context_parallel_modes(
    incompatible_config: dict[str, int],
):
    with pytest.raises(ValueError, match="cannot be combined"):
        ParallelConfig(
            tensor_parallel_size=4,
            attention_context_parallel_size=2,
            **incompatible_config,
        )


def test_attention_parallel_groups_match_target_tp4_layout():
    ranks = torch.arange(4).reshape(1, 1, 1, 1, 4)

    attn_tp_groups, attn_cp_groups = _get_attention_parallel_group_ranks(
        ranks,
        tensor_model_parallel_size=4,
        attention_context_model_parallel_size=2,
    )

    assert attn_tp_groups == [[0, 1], [2, 3]]
    assert attn_cp_groups == [[0, 2], [1, 3]]


def test_attention_parallel_groups_preserve_outer_dimensions():
    # ExternalDP=1, DP=2, PP=2, PCP=1, TP=4.
    ranks = torch.arange(16).reshape(1, 2, 2, 1, 4)

    attn_tp_groups, attn_cp_groups = _get_attention_parallel_group_ranks(
        ranks,
        tensor_model_parallel_size=4,
        attention_context_model_parallel_size=2,
    )

    assert attn_tp_groups == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
    ]
    assert attn_cp_groups == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [8, 10],
        [9, 11],
        [12, 14],
        [13, 15],
    ]
