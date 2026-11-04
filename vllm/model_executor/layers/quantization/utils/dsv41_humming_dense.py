# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected H20 projections using the unified Humming ordered Stream-K path."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_SHAPES = {
    "attn.fused_wqa_wkv": (1792, 5120),
    "attn.wq_a": (1280, 5120),
    "attn.wkv": (512, 5120),
    "attn.wq_b": (32768, 1280),
    "attn.wo_b": (5120, 8192),
    "attn.indexer.wq_b": (4096, 1280),
    "engram.wkv": (25600, 6144),
    "ffn.shared_experts.gate_up_proj": (4608, 5120),
    "ffn.shared_experts.down_proj": (5120, 2304),
    "main_proj": (5120, 15360),
}
_TARGET_ROWS = (96, 104, 108, 112)
_DRAFT_ROWS = (80, 88, 90, 96)
_VALIDATED_TARGET_ROWS = (64, 256, 384)
_VALIDATED_TARGET_PROJECTIONS = frozenset(
    (
        "attn.wq_b",
        "attn.wo_b",
        "ffn.shared_experts.gate_up_proj",
        "ffn.shared_experts.down_proj",
    )
)
_LEGACY_STREAM_K_PROJECTIONS = frozenset(
    (
        "attn.wq_b",
        "attn.wo_b",
        "engram.wkv",
        "ffn.shared_experts.down_proj",
        "main_proj",
    )
)


@triton.jit
def _quantize_groups(X, Q, S, GROUPS: tl.constexpr, GROUPS_PER_CTA: tl.constexpr):
    groups = tl.program_id(0) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    value = tl.load(X + offsets, groups[:, None] < GROUPS, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
    scale_raw = amax * (1.0 / 448.0)
    bits = scale_raw.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF).to(tl.int32) - 127
    exponent += (bits & 0x7FFFFF) != 0
    scale = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    inv_scale = ((127 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    quant = tl.minimum(tl.maximum(value * inv_scale[:, None], -448.0), 448.0)
    tl.store(Q + offsets, quant, groups[:, None] < GROUPS)
    tl.store(S + groups, scale, groups < GROUPS)


def quantize_input(x):
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float32
    )
    groups = x.numel() // 32
    _quantize_groups[(triton.cdiv(groups, 16),)](x, q, scale, groups, 16, num_warps=4)
    return q, scale


def _projection(prefix):
    if prefix in ("model.main_proj", "language_model.model.main_proj"):
        return "main_proj", _TARGET_ROWS
    parts = prefix.split(".")
    if parts[:3] == ["language_model", "model", "layers"]:
        index, suffix, rows = parts[3:4], parts[4:], _TARGET_ROWS
        allowed = range(40)
    elif parts[:2] == ["model", "layers"]:
        index, suffix, rows = parts[2:3], parts[3:], _DRAFT_ROWS
        allowed = range(40, 43)
    else:
        return None
    if not index or not index[0].isdigit() or int(index[0]) not in allowed:
        return None
    name = ".".join(suffix)
    if name not in _SHAPES:
        return None
    if rows == _TARGET_ROWS and name in ("attn.wq_a", "attn.wkv"):
        return None
    if rows == _DRAFT_ROWS and name == "attn.fused_wqa_wkv":
        return None
    return name, rows


def _legacy_stream_k_tuning(projection, m):
    block_m = 40 if m == 80 else 48 if m <= 96 else 56
    force_tma = projection == "attn.wq_b" or m != 80
    raster_group_m = 1
    if projection == "engram.wkv":
        raster_group_m = {40: 153, 48: 128, 56: 109}[block_m]
    elif projection == "main_proj":
        raster_group_m = {40: 61, 48: 51, 56: 43}[block_m]
    return {
        "block_shape": (block_m, 128, 128),
        "warp_shape": (block_m, 32, 128),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_sms": 64 if projection == "attn.wq_b" else 78,
        "num_stages": 3 if force_tma else 4,
        "num_ctas_per_sm": 2,
        "use_tma": force_tma,
        "use_warp_spec": force_tma,
        "mma_type": "wgmma",
        "raster_group_m": raster_group_m,
        "use_tma_c": False,
        "stream_k_max_slices": 3,
    }


def _validated_target_tuning(projection, m):
    if (
        projection not in _VALIDATED_TARGET_PROJECTIONS
        or m not in _VALIDATED_TARGET_ROWS
    ):
        return None
    if m == 64:
        wq_b = projection == "attn.wq_b"
        tuning = {
            "block_shape": (64, 128, 128),
            "warp_shape": (64, 32, 128 if wq_b else 64),
            "use_stream_k": not wq_b,
            "use_f16_accum": False,
            "num_sms": 64 if wq_b else 78,
            "num_stages": 3,
            "num_ctas_per_sm": 2 if wq_b else 1,
        }
    elif projection == "attn.wo_b":
        tuning = {
            "block_shape": (64, 128, 128),
            "warp_shape": (64, 32, 128),
            "use_stream_k": True,
            "use_f16_accum": False,
            "num_sms": 78,
            "num_stages": 3,
            "num_ctas_per_sm": 2,
        }
    else:
        tuning = {
            "block_shape": (64, 128, 128),
            "warp_shape": (64, 16, 128),
            "use_stream_k": False,
            "use_f16_accum": False,
            "num_sms": 78,
            "num_stages": 4,
            "num_ctas_per_sm": 1,
            "multi_cast_size_a": 1,
            "multi_cast_size_b": 1,
        }
    tuning.update(
        use_tma=True,
        use_warp_spec=True,
        raster_group_m=1,
        use_tma_c=False,
        stream_k_max_slices=3,
    )
    return tuning


class Dsv41HummingDense(torch.nn.Module):
    @classmethod
    @torch.no_grad()
    def from_layer(cls, layer):
        projection = _projection(layer.prefix)
        device_name = current_platform.get_device_name().upper()
        if (
            projection is None
            or "H20" not in device_name
            or "H200" in device_name
            or getattr(layer, "tp_size", None) != 1
            or layer.has_bias
            or layer.params_dtype != torch.bfloat16
        ):
            return None
        name, rows = projection
        n, k = _SHAPES[name]
        weight = layer.weight.detach()
        scale = layer.weight_scale.detach().view(torch.uint8)
        if (
            weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (n, k)
            or not weight.is_contiguous()
            or tuple(scale.shape) != (n, k // 32)
            or not scale.is_contiguous()
            or not bool(((scale > 0) & (scale < 255)).all())
        ):
            return None
        grouped = scale.view(n // 32, 32, k // 32)
        first = grouped[:, :1, :]
        if not torch.equal(grouped, first.expand_as(grouped)):
            return None
        block_scale = first[:, 0, :].contiguous().view(torch.float8_e8m0fnu).float()
        return cls(weight, block_scale, name, rows)

    def __init__(self, weight, block_scale, projection, rows):
        super().__init__()
        from humming.config import TuningConfig
        from humming.layer import HummingLayer
        from humming.tune import get_heuristics_config

        if "stream_k_max_slices" not in TuningConfig.get_field_names():
            raise RuntimeError(
                "DSV4.1 ordered Decode requires the unified Humming wheel "
                "with native stream_k_max_slices support."
            )
        validated_rows = (
            _VALIDATED_TARGET_ROWS
            if tuple(rows) == _TARGET_ROWS
            and projection in _VALIDATED_TARGET_PROJECTIONS
            else ()
        )
        self.rows = (*rows, *validated_rows)
        self.m = self.rows[0]
        self.n, self.k = weight.shape
        with torch.device(weight.device):
            packed = HummingLayer(
                shape_n=self.n,
                shape_k=self.k,
                weight_config={
                    "quant_method": "fp8",
                    "weight_block_size": [32, 32],
                },
                input_config={
                    "dtype": "float8e4m3",
                    "group_size": 32,
                    "scale_dtype": "float32",
                },
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                torch_dtype=torch.bfloat16,
            )
        packed.load_from_tensors({"weight": weight, "weight_scale_inv": block_scale})
        packed.transform()
        self.layer_config = packed.humming_config
        for name in ("weight", "weight_scale", "locks"):
            self.register_buffer(name, getattr(packed, name).detach(), persistent=False)
        self.tuning_configs = {}
        grid_caps = {
            "attn.fused_wqa_wkv": 56,
            "attn.wq_a": 40,
            "attn.wkv": 16,
            "attn.wq_b": 64,
        }
        for m in self.rows:
            tuning = _validated_target_tuning(projection, m)
            if tuning is None:
                tuning = dict(get_heuristics_config(self.layer_config, shape_m=m))
                tuning.update(use_tma_c=False, stream_k_max_slices=3)
                if projection in _LEGACY_STREAM_K_PROJECTIONS:
                    tuning = _legacy_stream_k_tuning(projection, m)
                if projection in grid_caps:
                    tuning["num_sms"] = grid_caps[projection]
            if self.layer_config.use_block_scaled_mma:
                raise RuntimeError("Ordered Decode requires ordinary FP32 scales.")
            self.tuning_configs[m] = tuning

    def supports(self, x):
        m = x.numel() // x.shape[-1]
        return (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == self.k
            and m in self.tuning_configs
        )

    def forward(self, x):
        from humming.forward import humming_forward

        if not self.supports(x):
            raise ValueError("Unsupported DSV4.1 Humming projection input")
        m = x.numel() // x.shape[-1]
        q, scale = quantize_input(x.view(m, self.k))
        output = humming_forward(
            self.layer_config,
            inputs=q,
            input_scale=scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            locks=self.locks,
            compute_config={"gemm_type": "dense", "use_batch_invariant": False},
            tuning_config=self.tuning_configs[m],
        )
        return output.view(*x.shape[:-1], self.n)
