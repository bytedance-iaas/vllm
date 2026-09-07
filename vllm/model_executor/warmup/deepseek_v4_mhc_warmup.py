# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _compute_mhc_pre_num_split(
    *,
    num_tokens: int,
    k: int,
    num_sms: int,
) -> int:
    grid_size = cdiv(num_tokens, 64)
    split_k = min(num_sms // grid_size, cdiv(k, 64) // 4)
    return max(split_k, 1)


def _select_split_representative_token_sizes(
    *,
    max_tokens: int,
    k: int,
    num_sms: int,
) -> list[int]:
    representatives: dict[int, int] = {}
    for grid_size in range(1, cdiv(max_tokens, 64) + 1):
        num_tokens = (grid_size - 1) * 64 + 1
        split_k = _compute_mhc_pre_num_split(
            num_tokens=num_tokens,
            k=k,
            num_sms=num_sms,
        )
        representatives.setdefault(split_k, num_tokens)
    return sorted(representatives.values())


def _select_nvidia_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> tuple[list[int], list[int]]:
    normal = _select_split_representative_token_sizes(
        max_tokens=max_tokens,
        k=hc_mult * hidden_size,
        num_sms=num_sms,
    )
    broadcast = _select_split_representative_token_sizes(
        max_tokens=max_tokens,
        k=hidden_size,
        num_sms=num_sms,
    )
    return normal, broadcast


def _select_nvidia_fused_mhc_warmup_token_sizes(max_tokens: int) -> list[int]:
    return _normalize_token_sizes((1, 8, 17), max_tokens=max_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "attn_norm",
                "ffn_norm",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_custom_op_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_nvidia_layer_mhc(
    layer: torch.nn.Module,
    *,
    normal_token_sizes: list[int],
    broadcast_token_sizes: list[int],
    warmup_broadcast: bool,
    max_tokens: int,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_broadcast_tilelang,
        mhc_pre_tilelang,
    )

    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    common_args = (
        layer.rms_norm_eps,
        layer.hc_eps,
        layer.hc_eps,
        layer.hc_post_alpha,
        layer.hc_sinkhorn_iters,
    )
    parameter_pairs = (
        (
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.attn_norm,
        ),
        (
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.ffn_norm,
        ),
    )

    fused_token_sizes = _select_nvidia_fused_mhc_warmup_token_sizes(max_tokens)
    residual = torch.zeros(
        max(*normal_token_sizes, *fused_token_sizes),
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    for size in normal_token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base, norm in parameter_pairs:
            post_mix, comb_mix, layer_input = mhc_pre_tilelang(
                residual_slice,
                fn,
                scale,
                base,
                *common_args,
                norm_weight=norm.weight.data,
                norm_eps=norm.variance_epsilon,
            )
            mhc_post_tilelang(layer_input, residual_slice, post_mix, comb_mix)

    # The fused path has separate small-FMA configurations below 17 tokens.
    for size in fused_token_sizes:
        residual_slice = residual[:size]
        attn_fn, attn_scale, attn_base, attn_norm = parameter_pairs[0]
        post_mix, comb_mix, layer_input = mhc_pre_tilelang(
            residual_slice,
            attn_fn,
            attn_scale,
            attn_base,
            *common_args,
            norm_weight=attn_norm.weight.data,
            norm_eps=attn_norm.variance_epsilon,
        )
        ffn_fn, ffn_scale, ffn_base, ffn_norm = parameter_pairs[1]
        residual_out, post_mix, comb_mix, layer_input = (
            mhc_fused_post_pre_tilelang(
                layer_input,
                residual_slice,
                post_mix,
                comb_mix,
                ffn_fn,
                ffn_scale,
                ffn_base,
                *common_args,
                n_splits=1,
                tile_n=1,
                norm_weight=ffn_norm.weight.data,
                norm_eps=ffn_norm.variance_epsilon,
            )
        )
        mhc_post_tilelang(layer_input, residual_out, post_mix, comb_mix)

    if not warmup_broadcast:
        return

    fn_broadcast = getattr(layer, "hc_attn_fn_broadcast", None)
    if fn_broadcast is None:
        raise RuntimeError(
            "DeepSeek V4 first PP stage is missing finalized mHC broadcast weights"
        )

    hidden_states = torch.zeros(
        max(broadcast_token_sizes),
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    attn_fn, attn_scale, attn_base, attn_norm = parameter_pairs[0]
    for size in broadcast_token_sizes:
        mhc_pre_broadcast_tilelang(
            hidden_states[:size],
            attn_fn,
            attn_scale,
            attn_base,
            *common_args,
            norm_weight=attn_norm.weight.data,
            norm_eps=attn_norm.variance_epsilon,
            fn_broadcast=fn_broadcast,
        )


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
    *,
    use_nvidia_kernels: bool,
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        if use_nvidia_kernels:
            from vllm.model_executor.kernels.mhc.tilelang import (
                hc_head_fused_kernel_tilelang,
            )

            hc_head_fused_kernel_tilelang(
                hidden_states[:size],
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.rms_norm_eps,
                model.hc_eps,
            )
        else:
            hc_head_op = getattr(model, "hc_head_op", None)
            if hc_head_op is None:
                return
            hc_head_op(
                hidden_states[:size],
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.rms_norm_eps,
                model.hc_eps,
            )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    from vllm.distributed import get_pp_group

    pp_group = get_pp_group()
    deepseek_model = _find_deepseek_v4_model(model)
    use_nvidia_kernels = current_platform.is_cuda()
    if use_nvidia_kernels:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        normal_token_sizes, broadcast_token_sizes = (
            _select_nvidia_mhc_warmup_token_sizes(
                max_tokens=max_tokens,
                hidden_size=int(layer.hidden_size),
                hc_mult=int(layer.hc_mult),
                num_sms=num_sms,
            )
        )
        token_sizes = normal_token_sizes
    else:
        token_sizes = _select_mhc_warmup_token_sizes(
            max_tokens=max_tokens,
            cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        )
        normal_token_sizes = token_sizes
        broadcast_token_sizes = []
    if not normal_token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels: backend=%s, "
        "normal_token_sizes=%s, broadcast_token_sizes=%s",
        "nvidia" if use_nvidia_kernels else "custom_op",
        normal_token_sizes,
        broadcast_token_sizes,
    )
    with torch.inference_mode():
        if use_nvidia_kernels:
            _warmup_nvidia_layer_mhc(
                layer,
                normal_token_sizes=normal_token_sizes,
                broadcast_token_sizes=broadcast_token_sizes,
                warmup_broadcast=pp_group.is_first_rank,
                max_tokens=max_tokens,
            )
        else:
            _warmup_custom_op_layer_mhc(layer, token_sizes)
        if deepseek_model is not None and pp_group.is_last_rank:
            _warmup_hc_head(
                deepseek_model,
                [1] if use_nvidia_kernels else token_sizes,
                use_nvidia_kernels=use_nvidia_kernels,
            )
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
