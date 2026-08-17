# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.v1.attention.backends.flash_attn as flash_attn


@pytest.mark.parametrize(
    ("scale", "expected_group_shape"),
    [
        (torch.tensor(0.5), None),
        (torch.tensor([0.5, 1.5]), (-1, 4)),
    ],
)
def test_quantize_dcp_live_kv_uses_static_scale(
    monkeypatch: pytest.MonkeyPatch,
    scale: torch.Tensor,
    expected_group_shape: tuple[int, int] | None,
) -> None:
    calls = []

    def fake_scaled_fp8_quant(
        tensor: torch.Tensor,
        actual_scale: torch.Tensor,
        *,
        group_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((tensor.shape, actual_scale, group_shape))
        return torch.empty_like(tensor, dtype=torch.float8_e4m3fn), actual_scale

    monkeypatch.setattr(
        flash_attn.ops,
        "scaled_fp8_quant",
        fake_scaled_fp8_quant,
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    tensor = torch.randn(3, 2, 4, dtype=torch.bfloat16)

    result = impl._quantize_dcp_live_kv(tensor, scale)

    assert result.shape == tensor.shape
    assert result.dtype == torch.float8_e4m3fn
    assert calls == [(torch.Size([3, 8]), scale, expected_group_shape)]


def test_quantize_dcp_live_kv_does_not_requantize_fp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("already-FP8 live K/V must not be requantized")

    monkeypatch.setattr(
        flash_attn.ops,
        "scaled_fp8_quant",
        fail_if_called,
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    tensor = torch.empty(3, 2, 4, dtype=torch.float8_e4m3fn)

    result = impl._quantize_dcp_live_kv(tensor, torch.tensor([0.5, 1.5]))

    assert result is tensor
