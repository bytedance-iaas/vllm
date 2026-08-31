# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.flash_attn as flash_attn
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTemporalLayout,
)


def test_full_temporal_builder_uses_global_dcp1_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda: 2)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _parallel_config: 8,
            get_num_kv_heads=lambda _parallel_config: 1,
            get_head_size=lambda: 128,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=128),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: False,
            ),
            max_cudagraph_capture_size=None,
        ),
        attention_config=SimpleNamespace(),
    )
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )
    builder = flash_attn.FlashAttentionMetadataBuilder(
        spec,
        ["language_model.model.layers.0.self_attn.attn"],
        config,
        torch.device("cpu"),
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([260], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([260], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([260], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=260,
        block_table_tensor=torch.tensor([[6, 7, 10]], dtype=torch.int32),
        slot_mapping=torch.tensor([895, 896, 1023, 1280]),
    )

    metadata = builder.build(0, common)

    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0
    assert metadata.dcp_context_kv_lens is None
    assert metadata.max_dcp_context_kv_len == 0
    assert torch.equal(metadata.block_table, common.block_table_tensor)
    assert torch.equal(metadata.slot_mapping, common.slot_mapping)


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
