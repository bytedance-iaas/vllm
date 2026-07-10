# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for DeepSeek V4 MegaMoE.

Quantizes hidden states to fp8 with E8M0 group scales and repacks the
routing top-k tensors into the int64/float32 layout that the DeepGEMM
MegaMoE kernels consume.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding, -1, ids)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        top_k,
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


@triton.jit
def _prepare_megamoe_sm90_quant_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    GROUP_K: tl.constexpr,
) -> None:
    # Launched over valid tokens only (grid = (num_tokens, hidden / GROUP_K)).
    # Padded rows are handled by the lightweight topk pad kernel below, so this
    # kernel never wastes hidden-group programs on padding.
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * GROUP_K + tl.arange(0, GROUP_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(hidden), axis=0)
    amax = tl.maximum(amax, 1.0e-10)
    scale = amax / 448.0

    fp8 = (hidden * (1.0 / scale)).to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )
    # Store raw FP32 per-128 activation scale.
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        scale,
    )


@triton.jit
def _prepare_megamoe_sm90_topk_copy_kernel(
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    routed_scaling_factor,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    # One program per *valid* token (grid = (num_tokens,)). Padded rows are
    # written by the flattened pad kernel below.
    token_id = tl.program_id(0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    topk_mask = topk_offsets < top_k

    ids = tl.load(
        topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
        mask=topk_mask,
        other=0,
    ).to(tl.int64)
    weights = tl.load(
        topk_weights
        + token_id * topk_weights_stride_m
        + topk_offsets * topk_weights_stride_k,
        mask=topk_mask,
        other=0.0,
    ).to(tl.float32)
    weights = weights * routed_scaling_factor

    tl.store(
        topk_idx_out + token_id * topk_idx_stride_m + topk_offsets * topk_idx_stride_k,
        ids,
        mask=topk_mask,
    )
    tl.store(
        topk_weights_out
        + token_id * topk_weights_out_stride_m
        + topk_offsets * topk_weights_out_stride_k,
        weights,
        mask=topk_mask,
    )


@triton.jit
def _prepare_megamoe_sm90_topk_pad_kernel(
    topk_idx_out,
    topk_weights_out,
    pad_base,
    total_slots,
    topk_idx_stride_m,
    topk_idx_stride_k,
    topk_weights_out_stride_m,
    topk_weights_out_stride_k,
    top_k,
    BLOCK: tl.constexpr,
) -> None:
    # Flattened padding fill: grid = (ceil(pad_slots / BLOCK),), where
    # pad_slots = (max_num_tokens - num_tokens) * top_k. The launch size scales
    # with the *padding amount*, not with max_num_tokens.
    pid = tl.program_id(0)
    slot = pad_base + pid * BLOCK + tl.arange(0, BLOCK)
    mask = slot < total_slots

    row = slot // top_k
    col = slot % top_k
    idx_off = row * topk_idx_stride_m + col * topk_idx_stride_k
    w_off = row * topk_weights_out_stride_m + col * topk_weights_out_stride_k

    tl.store(topk_idx_out + idx_off, tl.full([BLOCK], -1, tl.int64), mask=mask)
    tl.store(topk_weights_out + w_off, tl.zeros([BLOCK], tl.float32), mask=mask)


def _prepare_megamoe_inputs_sm90_triton(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Triton implementation for SM90 MegaMoE input staging.

    Split into quant (over valid tokens) + topk copy (over valid tokens) +
    flattened padding fill, so the launch size scales with the real token /
    padding counts rather than ``max_num_tokens``.
    """
    num_tokens, hidden_size = hidden_states.shape
    max_num_tokens = topk_idx_out.shape[0]
    if num_tokens > max_num_tokens:
        raise ValueError(
            "DeepSeek V4 SM90 MegaMoE input staging got "
            f"{num_tokens} tokens, but the output buffers are sized for "
            f"{max_num_tokens}."
        )
    if max_num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 SM90 MegaMoE input staging requires hidden_size to "
            "be a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 SM90 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    group_k = 128
    block_topk = triton.next_power_of_2(top_k)

    if num_tokens > 0:
        _prepare_megamoe_sm90_quant_kernel[
            (num_tokens, triton.cdiv(hidden_size, group_k))
        ](
            hidden_states,
            x_fp8,
            x_sf,
            hidden_states.stride(0),
            hidden_states.stride(1),
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_sf.stride(0),
            x_sf.stride(1),
            hidden_size,
            GROUP_K=group_k,
            num_warps=4,
        )
        _prepare_megamoe_sm90_topk_copy_kernel[(num_tokens,)](
            topk_ids,
            topk_weights,
            topk_idx_out,
            topk_weights_out,
            float(routed_scaling_factor),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk_idx_out.stride(0),
            topk_idx_out.stride(1),
            topk_weights_out.stride(0),
            topk_weights_out.stride(1),
            top_k,
            BLOCK_TOPK=block_topk,
            num_warps=4,
        )

    pad_slots = (max_num_tokens - num_tokens) * top_k
    if pad_slots > 0:
        pad_block = 1024
        _prepare_megamoe_sm90_topk_pad_kernel[(triton.cdiv(pad_slots, pad_block),)](
            topk_idx_out,
            topk_weights_out,
            num_tokens * top_k,
            max_num_tokens * top_k,
            topk_idx_out.stride(0),
            topk_idx_out.stride(1),
            topk_weights_out.stride(0),
            topk_weights_out.stride(1),
            top_k,
            BLOCK=pad_block,
            num_warps=4,
        )


def prepare_megamoe_inputs_sm90(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
) -> None:
    """SM90 (Hopper) MegaMoE input staging.

    Uses the Triton staging implementation. This path:

    - quantize BF16 hidden states into FP8 E4M3 with a fixed group size of 128
      and write the *raw* FP32 activation scale (not packed UE8M0);
    - receive the *full* symmetric buffers (size ``max_num_tokens``) and fill
      padded topk rows ``[num_tokens:]`` with ``-1`` ids and ``0.0`` weights;
    - fold ``routed_scaling_factor`` into the topk weights.

    The ``x_fp8``/``x_sf`` rows beyond ``num_tokens`` are left untouched; the
    DeepGEMM kernel skips them because their topk ids are ``-1``.
    """
    _prepare_megamoe_inputs_sm90_triton(
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
        routed_scaling_factor=routed_scaling_factor,
    )
