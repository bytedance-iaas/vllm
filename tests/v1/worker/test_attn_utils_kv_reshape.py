# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 GPU runner KV-cache reshape: layer-local cache dtype resolution.

Guards the DeepSeek-V4 ``fp8_ds_mla`` path used by DSpark (which forces the V2
runner). The reshape must pass the layer's own ``cache_dtype_str`` to the
backend's ``get_kv_cache_shape`` so the 584B/token SWA layout is honored,
instead of the global cache dtype (which would produce the 512 semantic-head
shape and mismatch the FlashMLA layout).
"""

import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVQuantMode, SlidingWindowMLASpec
from vllm.v1.worker.gpu.attn_utils import _layer_cache_dtype_str, _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup


def _new_swa_mla_spec(
    kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    cache_dtype_str="fp8_ds_mla",
):
    return SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        # fp8_ds_mla stores 1 byte/elem (uint8), 584 bytes/token; matches the
        # real DeepseekV4 SWA cache dtype so the reshape view math lines up.
        dtype=torch.uint8,
        kv_quant_mode=kv_quant_mode,
        cache_dtype_str=cache_dtype_str,
        sliding_window=128,
        model_version="deepseek_v4",
    )


def test_layer_cache_dtype_str_prefers_layer_local_for_quantized():
    spec = _new_swa_mla_spec()
    # Global dtype is intentionally mismatched; the layer-local value wins.
    assert _layer_cache_dtype_str(spec, "auto") == "fp8_ds_mla"


def test_layer_cache_dtype_str_auto_for_unquantized():
    spec = _new_swa_mla_spec(kv_quant_mode=KVQuantMode.NONE)
    assert _layer_cache_dtype_str(spec, "fp8") == "auto"


def test_layer_cache_dtype_str_falls_back_to_global():
    spec = _new_swa_mla_spec(cache_dtype_str=None)
    assert _layer_cache_dtype_str(spec, "fp8_ds_mla") == "fp8_ds_mla"


class _RecordingBackend(AttentionBackend):
    """Minimal backend that records the cache_dtype_str it is asked to shape."""

    seen_cache_dtype_str: str | None = None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str="auto",
    ):
        _RecordingBackend.seen_cache_dtype_str = cache_dtype_str
        # DeepSeek-V4 fp8_ds_mla stores 584 bytes/token; mirror that so the
        # reshape's raw-tensor view matches the raw allocation below.
        return (num_blocks, block_size, 584)

    @staticmethod
    def get_kv_cache_stride_order():
        raise NotImplementedError


def test_v2_reshape_passes_layer_local_fp8_ds_mla_to_backend():
    _RecordingBackend.seen_cache_dtype_str = None
    spec = _new_swa_mla_spec()
    layer_name = "model.layers.0.self_attn.attn"
    group = AttentionGroup(
        backend=_RecordingBackend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )

    num_blocks = 4
    # Raw tensor sized to an integer number of pages (fp8_ds_mla: 584B/token).
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)

    _reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors={layer_name: raw},
        # Global cache dtype is "auto" on purpose: a regression that used it
        # would produce the 512 semantic-head shape and this assert would fail.
        cache_dtype="auto",
        kernel_block_sizes=[spec.storage_block_size],
        shared_kv_cache_layers={},
    )

    assert _RecordingBackend.seen_cache_dtype_str == "fp8_ds_mla"
