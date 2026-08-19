# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS based Fused MoE kernels."""

import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.profiler import record_function

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
    moe_unpermute_range,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kMxfp4Dynamic,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_group_gemm_supported,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

logger = init_logger(__name__)

_W4A8_DEBUG_RECORDS = 0
_W4A8_SUPPORTED_SCHEDULES = {
    "Kernel_128x16_1x1x1_Coop",
    "Kernel_128x16_2x1x1_Coop",
    "Kernel_128x256_2x1x1_Coop",
    "Kernel_256x16_1x1x1_Coop",
    "Kernel_256x16_2x1x1_Coop",
    "Kernel_256x32_1x1x1_Coop",
    "Kernel_256x64_1x1x1_Coop",
    "Kernel_256x128_2x1x1_Coop",
}


def _w4a8_debug_schedule_override_value() -> str | None:
    schedule = os.environ.get("VLLM_W4A8_DEBUG_SCHEDULE_OVERRIDE")
    if not schedule:
        return None
    return schedule.strip()


def _w4a8_debug_enabled() -> bool:
    return os.environ.get("VLLM_W4A8_DEBUG_INSTRUMENT", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _w4a8_debug_max_records() -> int:
    value = os.environ.get("VLLM_W4A8_DEBUG_MAX_RECORDS", "128")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning_once("Invalid VLLM_W4A8_DEBUG_MAX_RECORDS=%r", value)
        return 128


def _w4a8_debug_schedule_override() -> str | None:
    schedule = _w4a8_debug_schedule_override_value()
    if not schedule:
        return None
    if schedule == "heuristic":
        return None
    if schedule not in _W4A8_SUPPORTED_SCHEDULES:
        logger.warning_once(
            "Ignoring unsupported VLLM_W4A8_DEBUG_SCHEDULE_OVERRIDE=%r",
            schedule,
        )
        return None
    return schedule


def _w4a8_debug_force_heuristic() -> bool:
    return _w4a8_debug_schedule_override_value() == "heuristic"


def _w4a8_float_eq(value: float | None, expected: float) -> bool:
    return value is not None and math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def _w4a8_debug_profile_phases() -> bool:
    return os.environ.get("VLLM_W4A8_DEBUG_PROFILE_PHASES", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _w4a8_debug_zero_skip_enabled() -> bool:
    return os.environ.get("VLLM_W4A8_DEBUG_ZERO_SKIP", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _w4a8_debug_active_threshold() -> int:
    value = os.environ.get("VLLM_W4A8_DEBUG_ACTIVE_THRESHOLD", "4")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning_once("Invalid VLLM_W4A8_DEBUG_ACTIVE_THRESHOLD=%r", value)
        return 4


def _w4a8_debug_scope(name: str):
    if _w4a8_debug_profile_phases():
        return record_function(name)
    return nullcontext()


def _w4a8_debug_tensor_stats(
    tensor: torch.Tensor | None,
    *,
    token_column: int | None = None,
) -> dict[str, int | float | None]:
    if tensor is None:
        return {}

    if token_column is not None:
        tensor = tensor[:, token_column]
    flat = tensor.detach().to(device="cpu", dtype=torch.int64).flatten()
    if flat.numel() == 0:
        return {"numel": 0}

    values = flat.tolist()
    values.sort()
    numel = len(values)
    return {
        "numel": numel,
        "min": values[0],
        "p50": values[numel // 2],
        "p90": values[min(numel - 1, int(numel * 0.9))],
        "max": values[-1],
        "sum": sum(values),
        "mean": float(sum(values)) / numel,
        "nonzero": sum(1 for value in values if value != 0),
        "zero": sum(1 for value in values if value == 0),
    }


def _w4a8_debug_active_stats(
    expert_token_counts: torch.Tensor | None,
) -> dict[str, int | bool | None]:
    stats: dict[str, int | bool | None] = {
        "active_threshold": _w4a8_debug_active_threshold(),
        "active_le_threshold": None,
        "active_experts": None,
        "total_tokens": None,
    }
    if expert_token_counts is None:
        return stats

    flat = expert_token_counts.detach().to(device="cpu", dtype=torch.int64).flatten()
    active_experts = sum(1 for value in flat.tolist() if value != 0)
    total_tokens = int(flat.sum().item())
    threshold = _w4a8_debug_active_threshold()
    stats.update(
        {
            "active_experts": active_experts,
            "total_tokens": total_tokens,
            "active_le_threshold": active_experts <= threshold,
        }
    )
    return stats


def _w4a8_debug_rank_info() -> dict[str, int | None]:
    info: dict[str, int | None] = {
        "pid": os.getpid(),
        "pp_rank": None,
        "pp_size": None,
        "tp_rank": None,
        "tp_size": None,
    }
    try:
        pp = get_pp_group()
        info["pp_rank"] = pp.rank_in_group
        info["pp_size"] = pp.world_size
    except (AssertionError, RuntimeError, ValueError):
        pass
    try:
        tp = get_tp_group()
        info["tp_rank"] = tp.rank_in_group
        info["tp_size"] = tp.world_size
    except (AssertionError, RuntimeError, ValueError):
        pass
    return info


def _w4a8_debug_log_metadata(
    *,
    path: str,
    schedule: str | None,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    local_E: int,
    global_num_experts: int,
    K: int,
    N: int,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    expert_token_counts: torch.Tensor | None,
) -> None:
    global _W4A8_DEBUG_RECORDS
    if not _w4a8_debug_enabled():
        return
    if _w4a8_debug_max_records() <= _W4A8_DEBUG_RECORDS:
        return
    if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
        logger.warning_once(
            "Skipping W4A8 debug instrumentation during CUDA graph capture."
        )
        return

    payload = {
        "event": "w4a8_debug_metadata",
        "record": _W4A8_DEBUG_RECORDS,
        "path": path,
        "schedule": schedule or "heuristic",
        "schedule_override": os.environ.get("VLLM_W4A8_DEBUG_SCHEDULE_OVERRIDE"),
        "hidden_shape": tuple(hidden_states.shape),
        "topk_shape": tuple(topk_ids.shape),
        "topk": topk_ids.size(1),
        "local_E": local_E,
        "global_E": global_num_experts,
        "K": K,
        "N": N,
        "expert_tokens": _w4a8_debug_tensor_stats(expert_token_counts),
        "active_stats": _w4a8_debug_active_stats(expert_token_counts),
        # W4A8 currently builds problem shapes with swap_ab=True in both paths,
        # so the useful expert M dimension is column 1.
        "problem_m_mm1": _w4a8_debug_tensor_stats(problem_sizes1, token_column=1),
        "problem_m_mm2": _w4a8_debug_tensor_stats(problem_sizes2, token_column=1),
    }
    payload.update(_w4a8_debug_rank_info())
    logger.info("W4A8_DEBUG %s", json.dumps(payload, sort_keys=True))
    _W4A8_DEBUG_RECORDS += 1


def _require_w4a8_swigluoai_params(
    gemm1_alpha: float | None,
    gemm1_beta: float | None,
    gemm1_clamp_limit: float | None,
) -> tuple[float, float, float]:
    params = {
        "gemm1_alpha": gemm1_alpha,
        "gemm1_beta": gemm1_beta,
        "gemm1_clamp_limit": gemm1_clamp_limit,
    }
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise ValueError("SWIGLUOAI_UNINTERLEAVE requires " + ", ".join(missing))

    assert gemm1_alpha is not None
    assert gemm1_beta is not None
    assert gemm1_clamp_limit is not None
    return gemm1_alpha, gemm1_beta, gemm1_clamp_limit


def _estimate_w4a8_batched_m(
    total_num_tokens: int,
    topk: int,
    global_num_experts: int,
) -> int:
    assert total_num_tokens >= 0
    assert topk > 0
    assert global_num_experts > 0

    return (total_num_tokens * topk + global_num_experts - 1) // global_num_experts


def _select_w4a8_batched_schedule(
    total_num_tokens: int,
    topk: int,
    global_num_experts: int,
) -> str:
    """Select a grouped-GEMM schedule from useful, rather than padded, M."""
    m_expert = _estimate_w4a8_batched_m(
        total_num_tokens,
        topk,
        global_num_experts,
    )

    if m_expert <= 1:
        return "Kernel_128x16_1x1x1_Coop"
    if m_expert <= 16:
        return "Kernel_256x16_1x1x1_Coop"
    if m_expert <= 32:
        return "Kernel_256x32_1x1x1_Coop"
    if m_expert <= 64:
        return "Kernel_256x64_1x1x1_Coop"
    if m_expert <= 128:
        return "Kernel_256x128_2x1x1_Coop"
    return "Kernel_128x256_2x1x1_Coop"


def _select_w4a8_standard_schedule(
    *,
    input_tokens: int,
    topk: int,
    local_num_experts: int,
    global_num_experts: int,
    n: int,
    k: int,
    activation: MoEActivation,
    gemm1_alpha: float | None,
    gemm1_beta: float | None,
    gemm1_clamp_limit: float | None,
) -> str | None:
    # MiniMax-M3's full long-prefill chunks favor the narrower N tile. Keep the
    # specialization within the exact model semantics and chunk size measured.
    if (
        input_tokens == 8192
        and topk == 4
        and local_num_experts == 32
        and global_num_experts == 128
        and n == 3072
        and k == 6144
        and activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        and _w4a8_float_eq(gemm1_alpha, 1.702)
        and _w4a8_float_eq(gemm1_beta, 1.0)
        and _w4a8_float_eq(gemm1_clamp_limit, 7.0)
    ):
        return "Kernel_256x32_1x1x1_Coop"
    return None


def _select_w4a8_compact_programs(
    total_num_tokens: int,
    topk: int,
    global_num_experts: int,
) -> int:
    m_expert = _estimate_w4a8_batched_m(
        total_num_tokens,
        topk,
        global_num_experts,
    )
    if m_expert <= 16:
        return 16
    if m_expert <= 32:
        return 64
    if m_expert <= 64:
        return 128
    return 256


def _w4a8_batched_total_num_tokens(
    local_num_tokens: int,
    global_num_experts: int,
    num_local_experts: int,
) -> int:
    dp_metadata = (
        get_forward_context().dp_metadata if is_forward_context_available() else None
    )
    if dp_metadata is not None:
        return int(dp_metadata.num_tokens_across_dp_cpu.sum().item())

    assert global_num_experts % num_local_experts == 0
    num_dispatchers = global_num_experts // num_local_experts
    return local_num_tokens * num_dispatchers


@triton.jit
def _masked_per_token_fp8_quant_kernel(
    src,
    dst,
    scales,
    expert_num_tokens,
    hidden: tl.constexpr,
    src_stride_e: tl.constexpr,
    src_stride_m: tl.constexpr,
    dst_stride_e: tl.constexpr,
    dst_stride_m: tl.constexpr,
    scale_stride_e: tl.constexpr,
    scale_stride_m: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    min_scale: tl.constexpr,
    block_n: tl.constexpr,
    programs_per_expert: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    valid_tokens = tl.load(expert_num_tokens + expert)
    offsets = tl.arange(0, block_n)
    mask = offsets < hidden
    for token in tl.range(lane, valid_tokens, programs_per_expert):
        src_ptr = src + expert * src_stride_e + token * src_stride_m + offsets
        values = tl.load(src_ptr, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.max(tl.abs(values), axis=0)
        scale = tl.maximum(absmax / fp8_max, min_scale)
        quantized = tl.clamp(values / scale, fp8_min, fp8_max).to(tl.float8e4nv)
        dst_ptr = dst + expert * dst_stride_e + token * dst_stride_m + offsets
        tl.store(dst_ptr, quantized, mask=mask)
        tl.store(
            scales + expert * scale_stride_e + token * scale_stride_m,
            scale,
        )


@triton.jit
def _masked_swigluoai_quant_kernel(
    src,
    dst,
    scales,
    expert_num_tokens,
    hidden: tl.constexpr,
    src_stride_e: tl.constexpr,
    src_stride_m: tl.constexpr,
    dst_stride_e: tl.constexpr,
    dst_stride_m: tl.constexpr,
    scale_stride_e: tl.constexpr,
    scale_stride_m: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    clamp_limit: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    min_scale: tl.constexpr,
    block_n: tl.constexpr,
    programs_per_expert: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    valid_tokens = tl.load(expert_num_tokens + expert)
    offsets = tl.arange(0, block_n)
    mask = offsets < hidden
    for token in tl.range(lane, valid_tokens, programs_per_expert):
        src_ptr = src + expert * src_stride_e + token * src_stride_m
        gate = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(src_ptr + hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.minimum(gate, clamp_limit)
        up = tl.minimum(tl.maximum(up, -clamp_limit), clamp_limit)
        gate = (gate / (1.0 + tl.exp(-alpha * gate))).to(tl.bfloat16).to(tl.float32)
        activated = gate * (up + beta)
        activated = activated.to(tl.bfloat16).to(tl.float32)
        absmax = tl.max(tl.abs(activated), axis=0)
        scale = tl.maximum(absmax / fp8_max, min_scale)
        quantized = tl.clamp(activated / scale, fp8_min, fp8_max).to(tl.float8e4nv)
        dst_ptr = dst + expert * dst_stride_e + token * dst_stride_m + offsets
        tl.store(dst_ptr, quantized, mask=mask)
        tl.store(
            scales + expert * scale_stride_e + token * scale_stride_m,
            scale,
        )


def _masked_per_token_fp8_quant(
    src: torch.Tensor,
    dst: torch.Tensor,
    scales: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    programs_per_expert: int = 16,
) -> None:
    assert src.dim() == 3 and src.is_contiguous()
    assert src.is_cuda and src.dtype == torch.bfloat16
    assert dst.shape == src.shape and dst.is_contiguous()
    assert dst.is_cuda and dst.dtype == torch.float8_e4m3fn
    assert scales.shape == (*src.shape[:2], 1) and scales.is_contiguous()
    assert scales.is_cuda and scales.dtype == torch.float32
    assert expert_num_tokens.shape == (src.shape[0],)
    assert expert_num_tokens.is_cuda and expert_num_tokens.dtype == torch.int32
    assert expert_num_tokens.is_contiguous()
    _, _, hidden = src.shape
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    min_scale = 1.0 / (fp8_info.max * 512.0)
    _masked_per_token_fp8_quant_kernel[(src.shape[0], programs_per_expert)](
        src,
        dst,
        scales,
        expert_num_tokens,
        hidden,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        scales.stride(0),
        scales.stride(1),
        fp8_info.min,
        fp8_info.max,
        min_scale,
        triton.next_power_of_2(hidden),
        programs_per_expert,
        num_warps=8,
    )


def _masked_swigluoai_quant(
    src: torch.Tensor,
    dst: torch.Tensor,
    scales: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    alpha: float,
    beta: float,
    clamp_limit: float,
    programs_per_expert: int = 16,
) -> None:
    assert src.dim() == 3 and src.is_contiguous()
    assert src.is_cuda and src.dtype == torch.bfloat16
    assert src.shape[-1] == dst.shape[-1] * 2
    assert dst.shape[:2] == src.shape[:2] and dst.is_contiguous()
    assert dst.is_cuda and dst.dtype == torch.float8_e4m3fn
    assert scales.shape == (*dst.shape[:2], 1) and scales.is_contiguous()
    assert scales.is_cuda and scales.dtype == torch.float32
    assert expert_num_tokens.shape == (src.shape[0],)
    assert expert_num_tokens.is_cuda and expert_num_tokens.dtype == torch.int32
    assert expert_num_tokens.is_contiguous()
    _, _, two_hidden = src.shape
    hidden = two_hidden // 2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    min_scale = 1.0 / (fp8_info.max * 512.0)
    _masked_swigluoai_quant_kernel[(src.shape[0], programs_per_expert)](
        src,
        dst,
        scales,
        expert_num_tokens,
        hidden,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        scales.stride(0),
        scales.stride(1),
        alpha,
        beta,
        clamp_limit,
        fp8_info.min,
        fp8_info.max,
        min_scale,
        triton.next_power_of_2(hidden),
        programs_per_expert,
        num_warps=8,
    )


def _w4a8_batched_quant_workspace(
    workspace: torch.Tensor,
    num_experts: int,
    padded_m: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert workspace.dtype == torch.bfloat16 and workspace.is_contiguous()
    storage = workspace.view(torch.uint8).flatten()
    quant_bytes = num_experts * padded_m * hidden
    scale_offset = (quant_bytes + 3) // 4 * 4
    scale_bytes = num_experts * padded_m * torch.float32.itemsize
    assert storage.numel() >= scale_offset + scale_bytes
    quant = storage[:quant_bytes].view(torch.float8_e4m3fn)
    scales = storage[scale_offset : scale_offset + scale_bytes].view(torch.float32)
    return (
        quant.view(num_experts, padded_m, hidden),
        scales.view(num_experts, padded_m, 1),
    )


def _apply_w4a8_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    gemm1_alpha: float | None,
    gemm1_beta: float | None,
    gemm1_clamp_limit: float | None,
) -> None:
    if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        gemm1_alpha, gemm1_beta, gemm1_clamp_limit = _require_w4a8_swigluoai_params(
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
        )

    apply_moe_activation(
        activation,
        output,
        input,
        clamp_limit=gemm1_clamp_limit,
        alpha=1.0 if gemm1_alpha is None else gemm1_alpha,
        beta=0.0 if gemm1_beta is None else gemm1_beta,
    )


def run_cutlass_moe_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    ab_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    permute_scratch: MoEPermuteScratch | None,
):
    a1q = hidden_states

    assert activation.is_gated, "Only gated activation is supported"
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert a1q.size(-1) == w1.size(2), "Hidden size mismatch w1"
    assert w1.size(1) == w2.size(2) * 2, "Hidden size mismatch w2"
    assert (
        w1_scale.dim() == 1 or w1_scale.size(1) == 1 or w1_scale.shape[1] == w1.size(1)
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1 or w2_scale.size(1) == 1 or w2_scale.shape[1] == w2.size(1)
    ), "W2 scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Expert number mismatch"
    assert (
        a1q_scale is None
        or a1q_scale.dim() == 0
        or a1q_scale.size(0) == 1
        or a1q_scale.size(0) == a1q.shape[0]
    ), "Input scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert w1.size(0) == w1_scale.size(0), "w1 scales expert number mismatch"
    assert w1.size(0) == w2_scale.size(0), "w2 scales expert number mismatch"
    assert (
        a2_scale is None
        or a2_scale.dim() == 0
        or a2_scale.size(0) == 1
        or a2_scale.size(0) == a1q.shape[0]
    ), "Intermediate scale shape mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    # NOTE(rob): the expert_map is used for the STANDARD case and
    # the batched format is used by the BATCHED case.
    # TODO(rob): update the MK interface to only pass the expert_map
    # during the STANDARD case to make this clearer across all kernels.
    if use_batched_format:
        assert expert_num_tokens is not None
    else:
        assert expert_num_tokens is None

    # We have two modes: batched experts and non-batched experts.
    # In the non-batched mode, the input tokens are not padded: thus, the shape
    # of the input is [total_num_tokens, hidden_size]. The input and output
    # require shuffling by a_map and c_map such that the tokens assigned to
    # each expert are contiguous.
    # In the batched mode, the input tokens are padded per expert to ensure that
    # the batched dispatch and combine functions work correctly: thus, the shape
    # of the input is [num_experts, max_num_tokens_per_expert, hidden_size].
    # The batched input and output require no shuffling by a_map and c_map since
    # their tokens are already contiguous for each expert as a result of
    # the dispatch function.

    M = a1q.size(0)  # non batched expert M
    padded_M = a1q.size(1)  # batched expert M
    _, K, N = w2.shape
    device = a1q.device

    assert w1.size(2) == K
    assert global_num_experts != -1
    assert a1q_scale is not None

    topk = topk_ids.size(1)
    local_E = w1.size(0)

    if use_batched_format:
        mm1_out = _resize_cache(workspace13, (local_E * padded_M, N * 2))
        act_out = _resize_cache(workspace2, (local_E * padded_M, N))
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (local_E * padded_M, N)
        )
        mm2_out = _resize_cache(workspace2, (local_E * padded_M, K))
    else:
        a1q_perm = _resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K)
        )
        mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
        act_out = _resize_cache(workspace2, (M * topk, N))
        # original workspace are based on input hidden_states dtype (bf16)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
        )
        mm2_out = _resize_cache(workspace2, (M * topk, K))

    if use_batched_format:
        assert expert_num_tokens is not None

        expert_offsets = torch.empty((local_E), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        ops.get_cutlass_batched_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            local_E,
            padded_M,
            N,
            K,
            local_E * padded_M <= 64,
        )

        w1_scale = w1_scale.reshape(w1_scale.size(0), -1)
        w2_scale = w2_scale.reshape(w2_scale.size(0), -1)
        a1q = a1q.reshape(-1, a1q.size(2))
        a1q_scale = a1q_scale.reshape(-1, a1q_scale.size(2)).contiguous()
        # c3x get_group_gemm_starts expects int64 to avoid overflow
        # during offset calculations
        expert_offsets = expert_offsets.to(torch.int64)
    else:
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        num_expert = global_num_experts if expert_map is None else expert_map.size(0)
        # permuted a1q reuses workspace2
        a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
            a1q,
            a1q_scale,
            topk_ids,
            num_expert,
            local_E,
            expert_map,
            permuted_hidden_states=a1q_perm,
            scratch=permute_scratch,
        )
        # swap_ab is a CUTLASS grouped-GEMM optimization (M <= 64 reduces padding).
        swap_ab = a1q.size(0) <= 64
        ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
            expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab
        )
        expert_offsets = expert_first_token_offset[:-1]

    if not per_act_token and (expert_map is not None or use_batched_format):
        # this is necessary to avoid imprecise scale calculation caused by
        # random data in the unused workspace. The workspace is unused when
        # this rank handles only partial tokens, or when it is batched .
        mm1_out.fill_(0)

    ops.cutlass_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_scale,
        expert_offsets,
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
        per_act_token,
        per_out_ch,
    )

    apply_moe_activation(activation, act_out, mm1_out)

    a2q, a2q_scale = ops.scaled_fp8_quant(
        act_out, a2_scale, use_per_token_if_dynamic=per_act_token, output=quant_out
    )

    ops.cutlass_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_scale,
        expert_offsets,
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
        per_act_token,
        per_out_ch,
    )

    if use_batched_format:
        output.copy_(mm2_out.reshape(local_E, padded_M, K), non_blocking=True)
    else:
        # for non-chunking mode the output is resized from workspace13
        # so we need to make sure mm2_out uses workspace2.
        moe_unpermute(
            out=output,
            permuted_hidden_states=mm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_first_token_offset,
        )


class CutlassExpertsFp8Base(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.use_fp8_w8a8

        e = moe_config.num_local_experts
        n = moe_config.intermediate_size_per_partition
        k = moe_config.hidden_dim
        device = moe_config.device
        ab_strides1_c_strides2 = torch.full((e,), k, device=device, dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device=device, dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device=device, dtype=torch.int64)

        self.out_dtype = moe_config.in_dtype
        self.ab_strides1 = ab_strides1_c_strides2
        self.ab_strides2 = ab_strides2
        self.c_strides1 = c_strides1
        self.c_strides2 = ab_strides1_c_strides2
        self._permute_scratch: MoEPermuteScratch | None = None

    @staticmethod
    def _supports_current_device() -> bool:
        return cutlass_group_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
        ]

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            max_num_tokens = self.moe_config.max_num_tokens
            if self.activation_format() == mk.FusedMoEActivationFormat.Standard:
                parallel_config = self.moe_config.moe_parallel_config
                num_dispatchers = (
                    parallel_config.ep_size
                    if parallel_config.use_ep
                    else parallel_config.dp_size
                )
                max_num_tokens *= num_dispatchers
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
            )
        return self._permute_scratch

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None
        if expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens

        use_batched_format = (
            self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
        )

        in_dtype = hidden_states.dtype
        run_cutlass_moe_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.ab_strides1,
            self.ab_strides2,
            self.c_strides1,
            self.c_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
            self._get_permute_scratch(),
        )


class CutlassExpertsFp8(CutlassExpertsFp8Base):
    """CUTLASS FP8 fused MoE expert implementation."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp8 does not support expert map, which is
        # needed for STANDARD activation format kernels in DP/EP mode.
        # Note that the BATCHED activation format does not use
        # the expert map for identifying experts.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_deepep_ht_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)


class CutlassBatchedExpertsFp8(CutlassExpertsFp8Base):
    """Batched CUTLASS FP8 fused MoE expert implementation."""

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # BATCHED activation format works with EP because
        # expert_map is not used to identify experts (the
        # info is encoded/managed by the P/F logic).
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        num_dp = self.num_dispatchers
        assert num_dp is not None
        experts_per_worker = self.moe_config.num_local_experts
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (experts_per_worker, M * num_dp, max(N, K))
        workspace2 = (
            experts_per_worker,
            M * num_dp,
            max(activation_out_dim, K),
        )
        output = (experts_per_worker, M, K)
        return (workspace1, workspace2, output)


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None:
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1 (not an argument to cutlass_moe_fp4): [e, w1_n, k]
    w1_fp4: [e, w1_n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    where w1_n = 2*n for gated activations (gate+up), n for non-gated (up only).
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, w1_n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int

    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    is_gated = activation.is_gated
    # For gated activations (e.g. SiLU), w1 output is 2*n (gate + up).
    # For non-gated activations (e.g. SiLU_NO_MUL), w1 output is n (up only).
    w1_n = n * 2 if is_gated else n

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape
    e_w1, w1_n_actual, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e, (
        "Number of experts must match",
        f" between weights. {e_w1}, {e_w2}, {e}",
    )
    assert k_a == half_k_w1 * 2 and k == k_w2, (
        "Hidden size mismatch between a, w1 and w2"
    )
    assert w1_n_actual == w1_n and half_n_w2 * 2 == n, "mismatch in expected `n`"
    assert m == m_a, "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m, (
        "topk must be provided for each row of a"
    )
    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if apply_router_weight_on_input:
        # TODO: this only works for topK=1, will need to update for topK>1
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=is_gated,
    )

    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
    )
    c1 = _resize_cache(workspace13, (m * topk, w1_n))
    c2 = _resize_cache(workspace2, (m * topk, n))
    c3 = _resize_cache(workspace13, (m * topk, k))
    ops.cutlass_fp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    if activation == MoEActivation.SILU:
        # Fused SiLU+Mul+NVFP4 quantization
        # Note: c2 workspace is no longer needed since SiLU is fused with quantization.
        # c3 reuses workspace13 after c1 is consumed.
        int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )
    else:
        apply_moe_activation(activation, c2, c1)
        int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
            c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )

    ops.cutlass_fp4_moe_mm(
        c3,
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        w2_alphas,
        problem_sizes2,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del int_fp4, int_blockscale

    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )
    else:
        output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


class CutlassExpertsFp4(mk.FusedMoEExpertsModular):
    """CUTLASS FP4 fused MoE expert implementation."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fuse activation scales into w_scale_2 in-place so that
        # g1/g2_alphas (which reference the same tensor) stay in sync
        # when EPLB rearranges the parameter.
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and (
            p.is_device_capability_family(100)
            or p.is_device_capability_family(110)
            or p.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU uses a fused silu+mul+fp4_quant kernel path.
        # Other gated activations use the generic apply_moe_activation()
        # fallback + separate fp4 quantization in run_cutlass_moe_fp4().
        # Non-gated activations (_NO_MUL) are also supported for models
        # like Nemotron-Nano that don't use gated MLP.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp4 does not support expert map, which is
        # needed for STANDARD activation format kernels in EP mode.
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(2 * N, K))
        workspace2 = (M * topk, N)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def run_cutlass_moe_mxfp4(
    output: torch.Tensor,
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None:
    """MXFP4 x MXFP4 MoE implementation using CUTLASS grouped GEMM."""
    is_gated = activation.is_gated
    w1_n = n * 2 if is_gated else n

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_mxfp4"
    m_a, k_a = a.shape
    e_w1, w1_n_actual, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e
    assert k_a == half_k_w1 * 2 and k == k_w2
    assert w1_n_actual == w1_n and half_n_w2 * 2 == n
    assert m == m_a
    assert 2 * half_k_w1 == k_w2
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m

    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if apply_router_weight_on_input:
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=is_gated,
    )

    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.mxfp4_experts_quant(
        a,
        expert_offsets,
        blockscale_offsets,
        e,
        num_topk,
    )
    c1 = _resize_cache(workspace13, (m * topk, w1_n))
    c2 = _resize_cache(workspace2, (m * topk, n))
    c3 = _resize_cache(workspace13, (m * topk, k))

    ops.cutlass_mxfp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    if activation == MoEActivation.SILU:
        int_fp4, int_blockscale = ops.silu_and_mul_mxfp4_experts_quant(
            c1, expert_offsets, blockscale_offsets, e, num_topk
        )
    else:
        apply_moe_activation(activation, c2, c1)
        int_fp4, int_blockscale = ops.mxfp4_experts_quant(
            c2, expert_offsets, blockscale_offsets, e, num_topk
        )

    ops.cutlass_mxfp4_moe_mm(
        c3,
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        problem_sizes2,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del int_fp4, int_blockscale

    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )
    else:
        output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


def swizzle_mxfp4_scales(
    scales: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    """Swizzle flat [N, K//32] E8M0 scales to CUTLASS tiled layout.

    CUTLASS expects MX scale factors in a tiled layout:
        [numMTiles, numKTiles, 32, 4, 4]
    where numMTiles = ceil(N/128), numKTiles = ceil(K/128),
    and the inner dimensions correspond to the swizzle pattern:
        mTileIdx = mIdx / 128
        outerMIdx = mIdx % 32
        innerMIdx = (mIdx / 32) % 4
        kTileIdx = kIdx / 4
        innerKIdx = kIdx % 4
    with kIdx = col_in_scale_space (i.e., index into K//32).
    """
    assert scales.dtype == torch.uint8
    num_scale_cols = K // 32  # number of E8M0 scale values per row

    num_m_tiles = (N + 127) // 128
    num_k_tiles = (num_scale_cols + 3) // 4

    # Pad N to multiple of 128 and scale_cols to multiple of 4
    padded_N = num_m_tiles * 128
    padded_scale_cols = num_k_tiles * 4

    # Start with flat scales, pad if needed
    padded = torch.zeros(
        padded_N, padded_scale_cols, dtype=torch.uint8, device=scales.device
    )
    padded[:N, :num_scale_cols] = scales

    # Reshape to tile structure:
    # [numMTiles, 4, 32, numKTiles, 4]
    #  mTileIdx, innerMIdx, outerMIdx, kTileIdx, innerKIdx
    tiled = padded.reshape(num_m_tiles, 4, 32, num_k_tiles, 4)
    # Permute to [numMTiles, numKTiles, 32, 4, 4]
    #            (outerMIdx, innerMIdx, innerKIdx)
    tiled = tiled.permute(0, 3, 2, 1, 4).contiguous()
    return tiled.reshape(-1)


class CutlassExpertsMxfp4(mk.FusedMoEExpertsModular):
    """CUTLASS MXFP4 x MXFP4 fused MoE expert implementation."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        capability = p.get_device_capability()
        return (
            p.is_cuda()
            and capability is not None
            and ops.mxfp4_experts_quant_supported(capability.to_int())
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(2 * N, K))
        workspace2 = (M * topk, N)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_mxfp4(
            output=output,
            a=hidden_states,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


@dataclass(frozen=True)
class W4A8ChunkedFinalizeRSContext:
    shared_out: torch.Tensor
    routed_scale: float
    chunk_rows_per_destination: int
    tp_size: int
    state_key: str


# W4A8
def run_cutlass_moe_w4a8_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    gemm1_alpha: float | None,
    gemm1_beta: float | None,
    gemm1_clamp_limit: float | None,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    group_size: int,
    permute_scratch: MoEPermuteScratch | None,
    chunked_finalize_rs: W4A8ChunkedFinalizeRSContext | None = None,
) -> torch.Tensor | None:
    a1q = hidden_states
    local_E = w1.size(0)
    device = a1q.device
    _, K, N_packed = w2.shape
    N = N_packed * 8  # logical N, pack 8 int4 into 1 int32

    assert per_act_token, "W4A8 must use per-token scales"
    assert per_out_ch, "W4A8 must use per-channel scales"
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1_scale.dtype == torch.float8_e4m3fn
    assert w2_scale.dtype == torch.float8_e4m3fn
    assert w1.dtype == torch.int32
    assert w2.dtype == torch.int32
    assert w1_chan_scale.dtype == torch.float32
    assert w2_chan_scale.dtype == torch.float32
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert a2_scale is None
    assert out_dtype in [torch.bfloat16], f"Invalid output dtype: {out_dtype}"
    assert group_size == 128, f"Only group size 128 supported but got {group_size=}"

    assert global_num_experts != -1
    assert w1.size(2) * 8 == K, (
        f"w1 hidden size mismatch: got {w1.size(2) * 8}, expected {K=}"
    )

    topk = topk_ids.size(1)
    problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
    schedule = _w4a8_debug_schedule_override()
    debug_path = "standard"
    debug_expert_token_counts: torch.Tensor | None = None

    if use_batched_format:
        debug_path = "batched"
        assert expert_num_tokens is not None
        assert expert_num_tokens.dtype == torch.int32
        assert expert_num_tokens.is_cuda
        assert expert_num_tokens.shape == (local_E,)
        debug_expert_token_counts = expert_num_tokens
        assert a1q.dim() == 3
        assert a1q.shape[0] == local_E
        assert a1q.shape[2] == K
        assert a1q.is_contiguous()

        padded_M = a1q.size(1)
        assert output.dim() == 3
        assert output.shape[0] == local_E
        assert output.shape[1] == padded_M
        assert output.shape[2] == K
        assert output.is_contiguous()
        mm1_out = _resize_cache(workspace13, (local_E * padded_M, N * 2))
        total_num_tokens = _w4a8_batched_total_num_tokens(
            local_num_tokens=topk_ids.size(0),
            global_num_experts=global_num_experts,
            num_local_experts=local_E,
        )
        compact_programs = _select_w4a8_compact_programs(
            total_num_tokens=total_num_tokens,
            topk=topk,
            global_num_experts=global_num_experts,
        )

        if a1q.dtype == torch.bfloat16:
            assert a1q_scale is None
            a1q, a1q_scale = _w4a8_batched_quant_workspace(
                workspace2,
                local_E,
                padded_M,
                K,
            )
            with _w4a8_debug_scope("w4a8:input_quant"):
                _masked_per_token_fp8_quant(
                    hidden_states,
                    a1q,
                    a1q_scale,
                    expert_num_tokens,
                    compact_programs,
                )
        else:
            assert a1q.dtype == torch.float8_e4m3fn
            assert a1q_scale is not None
            assert a1q_scale.dim() == 3
            assert a1q_scale.shape == (local_E, padded_M, 1)
            assert a1q_scale.is_contiguous()

        expert_offsets = torch.empty((local_E,), dtype=torch.int32, device=device)
        with _w4a8_debug_scope("w4a8:problem_sizes"):
            ops.get_cutlass_batched_moe_mm_data(
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                expert_num_tokens,
                local_E,
                padded_M,
                N,
                K,
                True,
            )
        a1q = a1q.reshape(local_E * padded_M, K)
        assert a1q_scale is not None
        a1q_scale = a1q_scale.reshape(local_E * padded_M, 1)
        # c3x get_group_gemm_starts expects int64 to avoid overflow during
        # offset calculations. W4A8 offsets are physical padded slabs.
        expert_offsets = expert_offsets.to(torch.int64)
        if schedule is None:
            schedule = _select_w4a8_batched_schedule(
                total_num_tokens=total_num_tokens,
                topk=topk,
                global_num_experts=global_num_experts,
            )
    else:
        assert expert_num_tokens is None
        assert a1q_scale is not None
        M = a1q.size(0)
        a1q_perm = _resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K)
        )
        mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
        act_out = _resize_cache(workspace2, (M * topk, N))
        # original workspace are based on input hidden_states dtype (bf16)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
        )
        mm2_out = _resize_cache(workspace2, (M * topk, K))

        num_expert = global_num_experts if expert_map is None else expert_map.size(0)
        # permuted a1q reuses workspace2
        with _w4a8_debug_scope("w4a8:permute"):
            a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
                a1q,
                a1q_scale,
                topk_ids,
                num_expert,
                local_E,
                expert_map,
                permuted_hidden_states=a1q_perm,
                scratch=permute_scratch,
            )
        # for RS gemm SwapAB is always enabled (swap logical M, N in the problem shape).
        with _w4a8_debug_scope("w4a8:problem_sizes"):
            ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
                expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, True
            )
        expert_offsets = expert_first_token_offset[:-1]
        debug_expert_token_counts = (
            expert_first_token_offset[1:] - expert_first_token_offset[:-1]
        )
        if schedule is None and not _w4a8_debug_force_heuristic():
            schedule = _select_w4a8_standard_schedule(
                input_tokens=hidden_states.size(0),
                topk=topk,
                local_num_experts=local_E,
                global_num_experts=global_num_experts,
                n=N,
                k=K,
                activation=activation,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

    _w4a8_debug_log_metadata(
        path=debug_path,
        schedule=schedule,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        local_E=local_E,
        global_num_experts=global_num_experts,
        K=K,
        N=N,
        problem_sizes1=problem_sizes1,
        problem_sizes2=problem_sizes2,
        expert_token_counts=debug_expert_token_counts,
    )

    if (
        _w4a8_debug_zero_skip_enabled()
        and not use_batched_format
        and chunked_finalize_rs is None
        and debug_expert_token_counts is not None
        and not torch.cuda.is_current_stream_capturing()
        and int(torch.count_nonzero(debug_expert_token_counts).item()) == 0
    ):
        with _w4a8_debug_scope("w4a8:zero_skip"):
            output.zero_()
        return None

    with _w4a8_debug_scope("w4a8:mm1"):
        ops.cutlass_w4a8_moe_mm(
            mm1_out,
            a1q,
            w1,
            a1q_scale,
            w1_chan_scale,
            w1_scale,
            group_size,
            expert_offsets,
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides1,
            schedule,
        )

    use_masked_swigluoai = (
        use_batched_format and activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )
    if use_masked_swigluoai:
        assert expert_num_tokens is not None
        alpha, beta, clamp_limit = _require_w4a8_swigluoai_params(
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
        )
        a2q, a2q_scale = _w4a8_batched_quant_workspace(
            workspace2,
            local_E,
            padded_M,
            N,
        )
        with _w4a8_debug_scope("w4a8:activation_quant"):
            _masked_swigluoai_quant(
                mm1_out.view(local_E, padded_M, N * 2),
                a2q,
                a2q_scale,
                expert_num_tokens,
                alpha,
                beta,
                clamp_limit,
                compact_programs,
            )
        a2q = a2q.reshape(local_E * padded_M, N)
        a2q_scale = a2q_scale.reshape(local_E * padded_M, 1)
        mm2_out = output[:, :padded_M, :].reshape(local_E * padded_M, K)
    else:
        if use_batched_format:
            act_out = _resize_cache(workspace2, (local_E * padded_M, N))
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn),
                (local_E * padded_M, N),
            )
            mm2_out = _resize_cache(workspace2, (local_E * padded_M, K))
        with _w4a8_debug_scope("w4a8:activation_quant"):
            _apply_w4a8_moe_activation(
                activation,
                act_out,
                mm1_out,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
            )
            a2q, a2q_scale = ops.scaled_fp8_quant(
                act_out,
                a2_scale,
                use_per_token_if_dynamic=per_act_token,
                output=quant_out,
            )

    with _w4a8_debug_scope("w4a8:mm2"):
        ops.cutlass_w4a8_moe_mm(
            mm2_out,
            a2q,
            w2,
            a2q_scale,
            w2_chan_scale,
            w2_scale,
            group_size,
            expert_offsets,
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            schedule,
        )

    if use_batched_format and not use_masked_swigluoai:
        output[:, :padded_M, :].copy_(
            mm2_out.reshape(local_E, padded_M, K), non_blocking=True
        )
    elif not use_batched_format:
        # for non-chunking mode the output is resized from workspace13
        # so we need to make sure mm2_out uses workspace2.
        if chunked_finalize_rs is None:
            with _w4a8_debug_scope("w4a8:unpermute"):
                moe_unpermute(
                    out=output,
                    permuted_hidden_states=mm2_out,
                    topk_weights=topk_weights,
                    inv_permuted_idx=inv_perm,
                    expert_first_token_offset=expert_first_token_offset,
                )
        else:
            return cutlass_w4a8_chunked_finalize_rs(
                mm2_out,
                topk_weights,
                inv_perm,
                expert_first_token_offset,
                chunked_finalize_rs.shared_out,
                chunked_finalize_rs.routed_scale,
                chunked_finalize_rs.chunk_rows_per_destination,
                chunked_finalize_rs.tp_size,
                chunked_finalize_rs.state_key,
            )
    return None


class _W4A8ChunkedFinalizeRSState:
    def __init__(
        self,
        tp_size: int,
        chunk_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.comm_stream = torch.cuda.Stream(device=device)
        self.stage = [
            torch.empty(
                (tp_size, chunk_rows, hidden_size),
                dtype=dtype,
                device=device,
            )
            for _ in range(2)
        ]
        self.tail = [
            torch.empty(
                (chunk_rows, hidden_size),
                dtype=dtype,
                device=device,
            )
            for _ in range(2)
        ]
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.done = [torch.cuda.Event() for _ in range(2)]
        self.busy = [False, False]


_W4A8_CHUNKED_FINALIZE_RS_STATES: dict[
    tuple[str, str, int, int, int, int, torch.dtype], _W4A8ChunkedFinalizeRSState
] = {}


def _get_w4a8_chunked_finalize_rs_state(
    state_key: str,
    group_name: str,
    tp_size: int,
    chunk_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _W4A8ChunkedFinalizeRSState:
    key = (
        state_key,
        group_name,
        device.index or 0,
        tp_size,
        chunk_rows,
        hidden_size,
        dtype,
    )
    state = _W4A8_CHUNKED_FINALIZE_RS_STATES.get(key)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A8 chunked finalize-RS state must be initialized before "
                "CUDA graph capture."
            )
        state = _W4A8ChunkedFinalizeRSState(
            tp_size,
            chunk_rows,
            hidden_size,
            dtype,
            device,
        )
        _W4A8_CHUNKED_FINALIZE_RS_STATES[key] = state
    return state


def _w4a8_chunked_finalize_plan(
    num_tokens: int,
    tp_size: int,
    chunk_rows_per_destination: int,
) -> list[tuple[int, int]]:
    if tp_size <= 1 or num_tokens % tp_size != 0:
        raise ValueError("Token count must be divisible by TP size.")
    if chunk_rows_per_destination <= 0:
        raise ValueError("chunk_rows_per_destination must be positive.")
    local_rows = num_tokens // tp_size
    return [
        (
            row_start,
            min(chunk_rows_per_destination, local_rows - row_start),
        )
        for row_start in range(0, local_rows, chunk_rows_per_destination)
    ]


def cutlass_w4a8_chunked_finalize_rs(
    mm2_out: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None,
    shared_out: torch.Tensor,
    routed_scale: float,
    chunk_rows_per_destination: int,
    tp_size: int,
    state_key: str,
) -> torch.Tensor:
    if not mm2_out.is_cuda:
        raise RuntimeError("W4A8 chunked finalize-RS requires CUDA tensors.")
    if mm2_out.ndim != 2 or shared_out.ndim != 2 or topk_weights.ndim != 2:
        raise RuntimeError("W4A8 chunked finalize-RS requires rank-2 inputs.")
    if (
        mm2_out.dtype != torch.bfloat16
        or shared_out.dtype != torch.bfloat16
        or topk_weights.dtype != torch.float32
    ):
        raise RuntimeError(
            "W4A8 chunked finalize-RS requires BF16 outputs and FP32 routing weights."
        )
    if (
        not mm2_out.is_contiguous()
        or not shared_out.is_contiguous()
        or not topk_weights.is_contiguous()
        or not inv_permuted_idx.is_contiguous()
    ):
        raise RuntimeError(
            "W4A8 chunked finalize-RS requires contiguous input tensors."
        )
    if mm2_out.device != shared_out.device or topk_weights.device != mm2_out.device:
        raise RuntimeError(
            "W4A8 chunked finalize-RS inputs must be on the same device."
        )
    if inv_permuted_idx.device != mm2_out.device:
        raise RuntimeError(
            "W4A8 chunked finalize-RS inverse mapping must be on the same device."
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "W4A8 chunked finalize-RS does not support CUDA graph capture."
        )

    num_tokens, hidden_size = shared_out.shape
    topk = topk_weights.shape[1]
    if topk_weights.shape[0] != num_tokens:
        raise RuntimeError("Routing weights and shared output token counts differ.")
    if mm2_out.shape != (num_tokens * topk, hidden_size):
        raise RuntimeError(
            "MM2 output must have shape [num_tokens * topk, hidden_size]."
        )
    if inv_permuted_idx.numel() != num_tokens * topk:
        raise RuntimeError("Inverse permutation has an incompatible size.")
    try:
        chunk_plan = _w4a8_chunked_finalize_plan(
            num_tokens,
            tp_size,
            chunk_rows_per_destination,
        )
    except ValueError as error:
        raise RuntimeError(f"W4A8 chunked finalize-RS: {error}") from error

    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError("Requested TP size does not match the active TP group.")
    device_comm = tp_group.device_communicator
    pynccl_comm = (
        None if device_comm is None else getattr(device_comm, "pynccl_comm", None)
    )
    if pynccl_comm is None or pynccl_comm.disabled:
        raise RuntimeError("W4A8 chunked finalize-RS requires active PyNccl.")

    local_rows = num_tokens // tp_size
    output = shared_out.new_empty((local_rows, hidden_size))
    state = _get_w4a8_chunked_finalize_rs_state(
        state_key,
        tp_group.unique_name,
        tp_size,
        chunk_rows_per_destination,
        hidden_size,
        shared_out.dtype,
        shared_out.device,
    )
    compute_stream = current_stream()
    last_slot = 0

    for chunk_idx, (row_start, valid_rows) in enumerate(chunk_plan):
        slot = chunk_idx % 2
        last_slot = slot
        if state.busy[slot]:
            compute_stream.wait_event(state.done[slot])

        stage = state.stage[slot]
        if valid_rows != chunk_rows_per_destination:
            stage[:, valid_rows:].zero_()

        for destination in range(tp_size):
            token_start = destination * local_rows + row_start
            destination_out = stage[destination, :valid_rows]
            moe_unpermute_range(
                out=destination_out,
                permuted_hidden_states=mm2_out,
                topk_weights=topk_weights,
                inv_permuted_idx=inv_permuted_idx,
                expert_first_token_offset=expert_first_token_offset,
                token_start=token_start,
            )
            if routed_scale != 1.0:
                destination_out.mul_(routed_scale)
            destination_out.add_(shared_out[token_start : token_start + valid_rows])

        state.ready[slot].record(compute_stream)
        state.comm_stream.wait_event(state.ready[slot])
        rs_output = (
            output[row_start : row_start + valid_rows]
            if valid_rows == chunk_rows_per_destination
            else state.tail[slot]
        )
        pynccl_comm.reduce_scatter(
            rs_output,
            stage.view(tp_size * chunk_rows_per_destination, hidden_size),
            stream=state.comm_stream,
        )
        if valid_rows != chunk_rows_per_destination:
            with torch.cuda.stream(state.comm_stream):
                output[row_start : row_start + valid_rows].copy_(rs_output[:valid_rows])
        state.done[slot].record(state.comm_stream)
        state.busy[slot] = True

    compute_stream.wait_event(state.done[last_slot])
    return output


def cutlass_w4a8_chunked_finalize_rs_fake(
    mm2_out: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None,
    shared_out: torch.Tensor,
    routed_scale: float,
    chunk_rows_per_destination: int,
    tp_size: int,
    state_key: str,
) -> torch.Tensor:
    return shared_out.new_empty((shared_out.shape[0] // tp_size, shared_out.shape[1]))


direct_register_custom_op(
    op_name="cutlass_w4a8_chunked_finalize_rs",
    op_func=cutlass_w4a8_chunked_finalize_rs,
    mutates_args=[],
    fake_impl=cutlass_w4a8_chunked_finalize_rs_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class CutlassExpertsW4A8Fp8(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        b_strides1: torch.Tensor,
        b_strides2: torch.Tensor,
        group_size: int,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

        e = moe_config.num_local_experts
        n = moe_config.intermediate_size_per_partition
        k = moe_config.hidden_dim
        device = moe_config.device

        self.out_dtype = moe_config.in_dtype

        a_strides1_c_strides2 = torch.full((e,), k, device=device, dtype=torch.int64)
        self.a_strides1 = a_strides1_c_strides2
        self.a_strides2 = torch.full((e,), n, device=device, dtype=torch.int64)
        self.c_strides1 = torch.full((e,), 2 * n, device=device, dtype=torch.int64)
        self.c_strides2 = a_strides1_c_strides2

        self.b_strides1 = b_strides1
        self.b_strides2 = b_strides2

        # sizeof(StrideS) = 16 bytes, encoded as 2xint64.
        self.s_strides1 = torch.zeros((e, 2), device=device, dtype=torch.int64)
        self.s_strides1[:, 0] = 2 * n
        self.s_strides2 = torch.zeros((e, 2), device=device, dtype=torch.int64)
        self.s_strides2[:, 0] = k

        self.group_size = group_size
        self._permute_scratch: MoEPermuteScratch | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_output_alias(self) -> bool:
        parallel_config = self.moe_config.moe_parallel_config
        return (
            self.activation_format() == mk.FusedMoEActivationFormat.Standard
            and parallel_config.dp_size == 1
            and not parallel_config.use_all2all_kernels
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.in_dtype != torch.bfloat16:
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )

        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason

        if moe_config.hidden_dim % 256 != 0:
            return False, "kernel requires hidden_dim to be divisible by 256"
        if moe_config.intermediate_size_per_partition % 256 != 0:
            return (
                False,
                (
                    "kernel requires intermediate_size_per_partition to be "
                    "divisible by 256"
                ),
            )

        if moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            params = {
                "swiglu_alpha": moe_config.swiglu_alpha,
                "swiglu_beta": moe_config.swiglu_beta,
                "swiglu_limit": moe_config.swiglu_limit,
            }
            missing = [name for name, value in params.items() if value is None]
            if missing:
                return False, "kernel requires " + ", ".join(missing)

        return True, None

    @staticmethod
    def _supports_current_device() -> bool:
        return cutlass_group_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kInt4Static, kFp8DynamicTokenSym)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            max_num_tokens = self.moe_config.max_num_tokens
            if self.activation_format() == mk.FusedMoEActivationFormat.Standard:
                parallel_config = self.moe_config.moe_parallel_config
                num_dispatchers = (
                    parallel_config.ep_size
                    if parallel_config.use_ep
                    else parallel_config.dp_size
                )
                max_num_tokens *= num_dispatchers
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
            )
        return self._permute_scratch

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None
        use_batched_format = (
            self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
        )
        if use_batched_format and expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens

        in_dtype = hidden_states.dtype

        run_cutlass_moe_w4a8_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            self.quant_config.gemm1_alpha,
            self.quant_config.gemm1_beta,
            self.quant_config.gemm1_clamp_limit,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.g1_alphas,  # per-channel scales
            self.g2_alphas,  # per-channel scales
            self.a_strides1,
            self.a_strides2,
            self.b_strides1,
            self.b_strides2,
            self.c_strides1,
            self.c_strides2,
            self.s_strides1,
            self.s_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
            self.group_size,
            self._get_permute_scratch(),
        )


class CutlassBatchedExpertsW4A8Fp8(CutlassExpertsW4A8Fp8):
    """Batched CUTLASS W4A8 expert implementation for DeepEP low latency."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_deepep_ll_kernels

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        num_dispatchers = self.num_dispatchers
        assert num_dispatchers is not None
        max_num_tokens = self.max_num_tokens
        assert max_num_tokens is not None
        experts_per_worker = self.moe_config.num_local_experts
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        padded_m = max_num_tokens * num_dispatchers
        workspace13 = (experts_per_worker, padded_m, max(N, K))
        workspace2 = (
            experts_per_worker,
            padded_m,
            max(activation_out_dim, K),
        )
        output = (experts_per_worker, padded_m, K)
        return (workspace13, workspace2, output)

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        return None
