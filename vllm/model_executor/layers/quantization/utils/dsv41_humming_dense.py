# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape H20 block-FP8 projections with ordered Stream-K reductions."""

import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_SHAPES = {
    "attn.fused_wqa_wkv": (1792, 5120),
    "attn.wq_a": (1280, 5120),
    "attn.wkv": (512, 5120),
    "attn.wq_b": (32768, 1280),
    "attn.wo_b": (5120, 8192),
    "ffn.shared_experts.gate_up_proj": (4608, 5120),
    "ffn.shared_experts.down_proj": (5120, 2304),
}


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
    parts = prefix.split(".")
    if parts[:3] == ["language_model", "model", "layers"]:
        index, suffix, m = parts[3:4], parts[4:], 96
        allowed = range(40)
    elif parts[:2] == ["model", "layers"]:
        index, suffix, m = parts[2:3], parts[3:], 80
        allowed = range(40, 43)
    else:
        return None
    if not index or not index[0].isdigit() or int(index[0]) not in allowed:
        return None
    name = ".".join(suffix)
    if name not in _SHAPES:
        return None
    if m == 96 and name in ("attn.wq_a", "attn.wkv"):
        return None
    if m == 80 and name == "attn.fused_wqa_wkv":
        return None
    return name, m


def _max_stream_slices(m, n, k, tuning):
    """Mirror the dense scheduler's stage-aligned Stream-K partition."""
    bm, bn, bk = tuning["block_shape"]
    tiles = math.ceil(m / bm) * (n // bn)
    grid = tuning["num_sms"] * tuning["num_ctas_per_sm"]
    tail = tiles
    if tiles > grid:
        tail = tiles % grid
        if tail and tail * 10 <= grid:
            tail += grid
    if not tail:
        return 1
    k_blocks = k // bk
    stages = tuning["num_stages"]
    chunk = math.ceil(math.ceil(tail * k_blocks / grid) / stages) * stages
    return max(
        ((tile + 1) * k_blocks - 1) // chunk - (tile * k_blocks) // chunk + 1
        for tile in range(tail)
    )


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
        name, m = projection
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
        return cls(weight, block_scale, name, m)

    def __init__(self, weight, block_scale, projection, m):
        super().__init__()
        import humming
        from humming.layer import HummingLayer
        from humming.tune import get_heuristics_config

        if humming.__version__.split("+")[0] != "0.1.12":
            raise RuntimeError("DSv4.1 ordered W8A8 requires humming-kernels 0.1.12")
        self.m = m
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
        tuning = dict(get_heuristics_config(self.layer_config, shape_m=m))
        tuning["use_tma_c"] = False
        grid_caps = {
            "attn.fused_wqa_wkv": 56,
            "attn.wq_a": 40,
            "attn.wkv": 16,
        }
        if projection in grid_caps:
            tuning["num_sms"] = grid_caps[projection]
        if projection == "attn.wq_b":
            tuning.update(
                num_sms=64,
                use_tma=True,
                use_warp_spec=True,
                use_mbarrier=True,
                num_stages=3,
            )
        # With at most three slices, the non-TMA epilogue serializes BF16 adds.
        if _max_stream_slices(m, self.n, self.k, tuning) > 3:
            raise RuntimeError("DSv4.1 Humming launch would use unordered atomics")
        self.tuning_config = tuning

    def supports(self, x):
        return (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == self.k
            and x.numel() == self.m * self.k
        )

    def forward(self, x):
        from humming.forward import humming_forward

        if not self.supports(x):
            raise ValueError("Unsupported DSv4.1 Humming projection input")
        q, scale = quantize_input(x.view(self.m, self.k))
        output = humming_forward(
            self.layer_config,
            inputs=q,
            input_scale=scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            locks=self.locks,
            compute_config={"gemm_type": "dense", "use_batch_invariant": False},
            tuning_config=self.tuning_config,
        )
        return output.view(*x.shape[:-1], self.n)
