# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


@triton.jit
def _qnorm_rope_kv_kernel(
    q_ptr,
    kv_ptr,
    kv_out_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    if head_idx < num_heads:
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM
        offs = tl.arange(0, HEAD_DIM)
        q_vals = tl.load(q_base + offs).to(tl.float32)
        rms = tl.rsqrt(tl.sum(q_vals * q_vals, axis=0) / HEAD_DIM + eps)
        q_vals *= rms
        tl.store(
            q_base + offs,
            q_vals.to(q_ptr.type.element_ty),
            mask=offs < NOPE_DIM,
        )

        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = even_offs + 1
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms
        tl.store(
            q_base + even_offs,
            (q_even * cos_val - q_odd * sin_val).to(q_ptr.type.element_ty),
        )
        tl.store(
            q_base + odd_offs,
            (q_even * sin_val + q_odd * cos_val).to(q_ptr.type.element_ty),
        )
    else:
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM
        offs = tl.arange(0, HEAD_DIM)
        tl.store(
            kv_out_base + offs,
            tl.load(kv_base + offs),
            mask=offs < NOPE_DIM,
        )

        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = even_offs + 1
        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)
        tl.store(
            kv_out_base + even_offs,
            (kv_even * cos_val - kv_odd * sin_val).to(kv_out_ptr.type.element_ty),
        )
        tl.store(
            kv_out_base + odd_offs,
            (kv_even * sin_val + kv_odd * cos_val).to(kv_out_ptr.type.element_ty),
        )


def qnorm_rope_kv(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize/rotate local Q and return the rotated local KV rows."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    kv_roped = torch.empty_like(kv)
    _qnorm_rope_kv_kernel[(num_tokens, num_heads + 1)](
        q,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )
    return kv_roped
