# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mxfp8_dense(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    split = tl.program_id(2)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK * SPLIT_K)):
        ks = (block * SPLIT_K + split) * BK + ki
        x = tl.load(
            X + mi[:, None] * K + ks[None, :],
            (mi[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + ni[:, None] * K + ks[None, :],
            (ni[:, None] < N) & (ks[None, :] < K),
            other=0,
        )
        e = tl.load(
            S + ni[:, None] * (K // 32) + ks[None, :] // 32,
            (ni[:, None] < N) & (ks[None, :] < K),
            other=127,
        )
        scale = (e.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        w = w.to(tl.float8e4nv, bitcast=True)
        b = (w.to(tl.float32) * scale).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(b))
    offset = split * M * N + mi[:, None] * N + ni[None, :]
    tl.store(Y + offset, acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _reduce_splits(P, Y, MN: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        value += tl.load(P + split * MN + idx, idx < MN, other=0)
    tl.store(Y + idx, value, idx < MN)


@triton.jit
def _reduce_swiglu(
    P,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = N // 2
    offset = idx // d * N + idx % d
    gate = tl.full((BLOCK,), 0, tl.float32)
    up = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        gate += tl.load(P + split * M * N + offset, idx < M * d, other=0)
        up += tl.load(P + split * M * N + offset + d, idx < M * d, other=0)
    # The unfused GEMM materializes BF16 before the FP32 activation.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    # The CUDA activation also rounds SiLU before multiplying by up.
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value = activated * up
    tl.store(Y + idx, value, idx < M * d)


def mxfp8_dense(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 32,
    split_k: int = 1,
    num_warps: int = 4,
    num_stages: int = 3,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """BF16 GEMM for contiguous MXFP8 weights and E8M0 codes 1 through 254."""
    n, k = weight.shape
    m = x.numel() // k
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
    assert scale.dtype == torch.uint8 and scale.shape == (n, k // 32)
    assert k % 32 == 0 and scale.is_contiguous()
    out_n = n if swiglu_limit is None else n // 2
    assert swiglu_limit is None or n % 2 == 0
    output = torch.empty((m, out_n), device=x.device, dtype=x.dtype)
    if m == 0:
        return output.view(*x.shape[:-1], out_n)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1 or swiglu_limit is not None
        else output
    )
    _mxfp8_dense[(triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)](
        x,
        weight.view(torch.uint8),
        scale,
        partial,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if swiglu_limit is not None:
        _reduce_swiglu[(triton.cdiv(m * out_n, 256),)](
            partial,
            output,
            m,
            n,
            split_k,
            swiglu_limit,
            256,
            enable_fp_fusion=False,
        )
    elif split_k > 1:
        _reduce_splits[(triton.cdiv(m * n, 512),)](
            partial,
            output,
            m * n,
            split_k,
            512,
        )
    return output.view(*x.shape[:-1], out_n)


class Dsv41DenseMxfp8(torch.nn.Module):
    """Retained checkpoint layout for selected SM90 Decode projections."""

    _CONFIGS = {
        "attn.fused_wqa_wkv": ((1792, 5120), 128, 16),
        "ffn.shared_experts.gate_up_proj": ((4608, 5120), 128, 8),
        "ffn.shared_experts.down_proj": ((5120, 2304), 64, 4),
    }

    @classmethod
    def from_layer(cls, layer):
        parts = layer.prefix.split(".")
        if (
            len(parts) < 6
            or parts[:3] != ["language_model", "model", "layers"]
            or not parts[3].isdigit()
            or not 0 <= int(parts[3]) < 40
            or getattr(layer, "tp_size", None) != 1
            or layer.has_bias
            or layer.params_dtype != torch.bfloat16
        ):
            return None
        config = cls._CONFIGS.get(".".join(parts[4:]))
        if config is None or tuple(layer.weight.shape) != config[0]:
            return None
        scale = layer.weight_scale.detach().view(torch.uint8)
        # The kernel's bit expansion requires normal finite E8M0 scales.
        if (
            layer.weight.dtype != torch.float8_e4m3fn
            or not layer.weight.is_contiguous()
            or scale.shape != (config[0][0], config[0][1] // 32)
            or not scale.is_contiguous()
            or not bool(((scale > 0) & (scale < 255)).all())
        ):
            return None
        return cls(layer.weight, scale, config[1], config[2])

    def __init__(self, weight, scale, block_n, split_k):
        super().__init__()
        self.register_buffer("weight", weight.detach(), persistent=False)
        self.register_buffer(
            "scale", scale.detach().view(torch.uint8), persistent=False
        )
        self.block_n = block_n
        self.split_k = split_k

    def supports(self, x):
        return (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == self.weight.shape[1]
            and x.numel() // x.shape[-1] in (24, 32, 40, 48, 56, 64, 72, 80, 96, 128)
        )

    def forward(self, x, swiglu_limit=None):
        return mxfp8_dense(
            x,
            self.weight,
            self.scale,
            block_m=64,
            block_n=self.block_n,
            block_k=32,
            split_k=self.split_k,
            swiglu_limit=swiglu_limit,
        )
