# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _compute_mhc_pre_num_split,
    _find_first_mhc_layer,
    _select_nvidia_fused_mhc_warmup_token_sizes,
    _select_nvidia_mhc_warmup_token_sizes,
    _select_split_representative_token_sizes,
)


@pytest.mark.parametrize("max_tokens", [1, 3500, 8192, 32769])
@pytest.mark.parametrize("k", [4096, 4 * 4096])
def test_split_representatives_are_exhaustive(max_tokens: int, k: int) -> None:
    num_sms = 78
    representatives = _select_split_representative_token_sizes(
        max_tokens=max_tokens,
        k=k,
        num_sms=num_sms,
    )

    expected = {
        _compute_mhc_pre_num_split(
            num_tokens=num_tokens,
            k=k,
            num_sms=num_sms,
        )
        for num_tokens in range(1, max_tokens + 1)
    }
    actual = {
        _compute_mhc_pre_num_split(
            num_tokens=num_tokens,
            k=k,
            num_sms=num_sms,
        )
        for num_tokens in representatives
    }

    assert actual == expected
    assert representatives == sorted(set(representatives))
    assert all(1 <= size <= max_tokens for size in representatives)


def test_nvidia_warmup_covers_normal_and_broadcast_split_sets() -> None:
    normal, broadcast = _select_nvidia_mhc_warmup_token_sizes(
        max_tokens=8192,
        hidden_size=4096,
        hc_mult=4,
        num_sms=78,
    )

    normal_splits = {
        _compute_mhc_pre_num_split(
            num_tokens=size,
            k=4 * 4096,
            num_sms=78,
        )
        for size in normal
    }
    broadcast_splits = {
        _compute_mhc_pre_num_split(
            num_tokens=size,
            k=4096,
            num_sms=78,
        )
        for size in broadcast
    }

    assert normal_splits == {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        15,
        19,
        26,
        39,
        64,
    }
    assert broadcast_splits == {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        15,
        16,
    }


@pytest.mark.parametrize(
    ("max_tokens", "expected"),
    [
        (1, [1]),
        (8, [1, 8]),
        (16, [1, 8]),
        (64, [1, 8, 17]),
    ],
)
def test_nvidia_fused_warmup_boundaries(
    max_tokens: int,
    expected: list[int],
) -> None:
    assert _select_nvidia_fused_mhc_warmup_token_sizes(max_tokens) == expected


def test_finds_nvidia_mhc_layer_without_custom_ops() -> None:
    class DeepseekV4DecoderLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "attn_norm",
                "ffn_norm",
            ):
                setattr(self, name, SimpleNamespace())

    model = torch.nn.Module()
    model.layer = DeepseekV4DecoderLayer()

    assert _find_first_mhc_layer(model) is model.layer
