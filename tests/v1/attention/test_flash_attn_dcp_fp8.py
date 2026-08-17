# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.v1.attention.backends.flash_attn as flash_attn


def test_trace_paged_cache_rows_follows_block_table() -> None:
    cache = torch.arange(3 * 4 * 2).view(3, 4, 1, 2)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)

    rows = flash_attn._trace_paged_cache_rows(cache, block_table, max_rows=6)

    expected = torch.cat((cache[2], cache[0][:2]))
    assert torch.equal(rows, expected)


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
    fp8_dtype = flash_attn.current_platform.fp8_dtype()

    def fake_scaled_fp8_quant(
        tensor: torch.Tensor,
        actual_scale: torch.Tensor,
        *,
        group_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(
            (
                tensor.shape,
                tensor.stride(),
                tensor.is_contiguous(),
                actual_scale,
                group_shape,
            )
        )
        return torch.empty_like(tensor, dtype=fp8_dtype), actual_scale

    monkeypatch.setattr(
        flash_attn.ops,
        "scaled_fp8_quant",
        fake_scaled_fp8_quant,
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    qkv = torch.randn(3, 16, dtype=torch.bfloat16)
    tensor = qkv[:, 4:12].view(3, 2, 4)
    assert not tensor.is_contiguous()

    result = impl._quantize_dcp_live_kv(tensor, scale)

    assert result.shape == tensor.shape
    assert result.dtype == fp8_dtype
    assert calls == [
        (
            torch.Size([3, 8]),
            (16, 1),
            False,
            scale,
            expected_group_shape,
        )
    ]


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
    tensor = torch.empty(
        3,
        2,
        4,
        dtype=flash_attn.current_platform.fp8_dtype(),
    )

    result = impl._quantize_dcp_live_kv(tensor, torch.tensor([0.5, 1.5]))

    assert result is tensor
