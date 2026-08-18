# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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
    ("world_size", "interleave", "context_len", "max_local_len"),
    [
        (2, 1, 7, 4),
        (2, 2, 11, 6),
        (4, 1, 13, 4),
        (4, 2, 19, 6),
    ],
)
def test_restore_dcp_global_context_rows(
    world_size: int,
    interleave: int,
    context_len: int,
    max_local_len: int,
) -> None:
    rows = torch.arange(context_len * 2).view(context_len, 2)
    gathered = torch.full((world_size * max_local_len, 2), -1)
    for pos in range(context_len):
        rank = (pos // interleave) % world_size
        local_idx = pos // (world_size * interleave) * interleave + pos % interleave
        gathered[rank * max_local_len + local_idx] = rows[pos]

    restored = flash_attn._restore_dcp_global_context_rows(
        gathered,
        context_len=context_len,
        dcp_world_size=world_size,
        cp_kv_cache_interleave_size=interleave,
        max_local_len=max_local_len,
    )

    torch.testing.assert_close(restored, rows)


@pytest.mark.parametrize(
    ("enabled", "layer_name", "max_query_len", "max_context_len", "expected"),
    [
        (True, "language_model.model.layers.0.self_attn.attn", 1, 128, True),
        (True, "language_model.model.layers.0.self_attn.attn", 4, 128, True),
        (False, "language_model.model.layers.0.self_attn.attn", 4, 128, False),
        (True, "model.layers.60.self_attn.attn", 4, 128, False),
        (True, "language_model.model.layers.0.self_attn.attn", 5, 128, False),
        (True, "language_model.model.layers.0.self_attn.attn", 4, 0, False),
    ],
)
def test_dcp_eagle_fp32_combine_gate(
    enabled: bool,
    layer_name: str,
    max_query_len: int,
    max_context_len: int,
    expected: bool,
) -> None:
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl._dcp_eagle_fp32_combine = enabled
    impl._dcp_eagle_max_query_len = 4
    layer = SimpleNamespace(layer_name=layer_name)
    metadata = SimpleNamespace(
        max_query_len=max_query_len,
        max_dcp_context_kv_len=max_context_len,
    )
    output = torch.empty((1, 8, 128), dtype=torch.bfloat16)

    assert impl._use_dcp_eagle_fp32_combine(layer, metadata, output) is expected


@pytest.mark.parametrize(
    (
        "enabled",
        "layer_name",
        "num_reqs",
        "max_query_len",
        "max_context_len",
        "has_context_lens",
        "is_capturing",
        "expected",
    ),
    [
        (True, "language_model.model.layers.0.self_attn.attn", 1, 4, 128, True, False, True),
        (False, "language_model.model.layers.0.self_attn.attn", 1, 4, 128, True, False, False),
        (True, "model.layers.0.self_attn.attn", 1, 4, 128, True, False, False),
        (True, "language_model.model.layers.0.self_attn.attn", 2, 4, 128, True, False, False),
        (True, "language_model.model.layers.0.self_attn.attn", 1, 5, 128, True, False, False),
        (True, "language_model.model.layers.0.self_attn.attn", 1, 4, 0, True, False, False),
        (True, "language_model.model.layers.0.self_attn.attn", 1, 4, 128, False, False, False),
        (True, "language_model.model.layers.0.self_attn.attn", 1, 4, 128, True, True, False),
    ],
)
def test_dcp_eagle_full_context_attn_gate(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    layer_name: str,
    num_reqs: int,
    max_query_len: int,
    max_context_len: int,
    has_context_lens: bool,
    is_capturing: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        flash_attn.torch.cuda,
        "is_current_stream_capturing",
        lambda: is_capturing,
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl._dcp_eagle_full_context_attn = enabled
    impl._dcp_eagle_max_query_len = 4
    layer = SimpleNamespace(layer_name=layer_name)
    metadata = SimpleNamespace(
        query_start_loc=torch.zeros(num_reqs + 1, dtype=torch.int32),
        max_query_len=max_query_len,
        max_dcp_context_kv_len=max_context_len,
        dcp_context_kv_lens=torch.ones(num_reqs) if has_context_lens else None,
    )
    output = torch.empty((1, 8, 128), dtype=torch.bfloat16)

    assert impl._use_dcp_eagle_full_context_attn(layer, metadata, output) is expected


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
