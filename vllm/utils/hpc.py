# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for HPC API changes.

Users of vLLM should always import **only** these wrappers.
"""

import functools
import importlib
import importlib.util
import inspect

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_hpc() -> bool:
    """Return `True` if hpc package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("hpc") is None:
        logger.warning_once(
            "HPC attention requires the hpc module to be installed. "
            "Please install it from https://github.com/Tencent/hpc-ops"
        )
        return False
    return True


@functools.cache
def has_hpc_mxfp8_k32_moe() -> bool:
    """Return True if the installed hpc package has MiniMax-M3 K32 MoE ops."""
    if not has_hpc():
        return False
    try:
        import hpc  # noqa: F401
    except Exception as err:
        logger.warning_once("Failed to import hpc package: %s", err)
        return False

    required_ops = (
        "build_mxfp8_k32_moe_routing_cache",
        "fuse_moe_mxfp8_k32_candidate",
        "fuse_moe_mxfp8_k32_bf16_candidate",
    )
    missing_ops = [op for op in required_ops if not hasattr(torch.ops.hpc, op)]
    if missing_ops:
        logger.warning_once(
            "Installed hpc package is missing MiniMax-M3 MXFP8 K32 MoE ops: %s",
            ", ".join(missing_ops),
        )
        return False
    return True


# Remove 'torch._library.custom_ops':
# The output of this custom operator (1) must not also be an input to
# this custom operator and (2) may not alias any inputs to this custom
# operator or other returns. The most common way to trigger this error
# is if we have y = custom_op(x) and y and x are the same Tensor.
# Please instead return a clone of the offending output tensor(s) (e.g.
# return x.clone()) or refactor the custom operator to not return y.
# @torch.library.custom_op(
#     "vllm::fuse_moe_impl",
#     mutates_args=[],
#     device_types="cuda",
# )
def fuse_moe_impl(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    from hpc import fuse_moe as fuse_moe_

    return fuse_moe_(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        shared_output,
        output=output,
    )


# @torch.library.register_fake(
#     "vllm::fuse_moe_impl",
# )
def fuse_moe_impl_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def hpc_fuse_moe(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    return fuse_moe_impl(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        shared_output,
        output=output,
    )


@functools.cache
def _hpc_blockwise_supports_activation_clamp(fuse_moe_blockwise: object) -> bool:
    signature = inspect.signature(fuse_moe_blockwise)
    return any(
        param_name == "activation_clamp" or param.kind == inspect.Parameter.VAR_KEYWORD
        for param_name, param in signature.parameters.items()
    )


# @torch.library.custom_op(
#     "vllm::fuse_moe_blockwise_impl",
#     mutates_args=[],
#     device_types="cuda",
# )
def fuse_moe_blockwise_impl(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    from hpc import fuse_moe_blockwise as fuse_moe_blockwise_

    clamp = 0.0 if activation_clamp is None else float(activation_clamp)

    # Preserve the original HPC-Ops call for regular SwiGLU. Only clipped
    # SwiGLU requires the newer activation_clamp argument.
    if clamp == 0.0:
        return fuse_moe_blockwise_(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            rank_ep,
            num_expert_total,
            shared_output,
            output=output,
        )

    if not _hpc_blockwise_supports_activation_clamp(fuse_moe_blockwise_):
        raise RuntimeError(
            "HPC blockwise MoE requires hpc-ops with activation_clamp "
            "support for DeepSeek-V4 clipped-SwiGLU."
        )

    return fuse_moe_blockwise_(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output=output,
        activation_clamp=clamp,
    )


# @torch.library.register_fake(
#     "vllm::fuse_moe_blockwise_impl",
# )
def fuse_moe_blockwise_impl_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def hpc_fuse_moe_blockwise(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return fuse_moe_blockwise_impl(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output=output,
        activation_clamp=activation_clamp,
    )


def hpc_build_mxfp8_k32_moe_routing_cache(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import hpc  # noqa: F401

    return torch.ops.hpc.build_mxfp8_k32_moe_routing_cache(
        topk_ids.to(torch.int32).contiguous(),
        num_experts,
    )


def hpc_fuse_moe_mxfp8_k32_candidate(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    routing_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    output: torch.Tensor | None = None,
    activation_clamp: float = 7.0,
    alpha: float = 1.702,
    beta: float = 1.0,
) -> torch.Tensor:
    import hpc  # noqa: F401

    row_indices, topk_pos, seqlens, cu_seqlens = routing_cache
    return torch.ops.hpc.fuse_moe_mxfp8_k32_candidate(
        hidden_q,
        hidden_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        row_indices,
        topk_pos,
        seqlens,
        cu_seqlens,
        topk_weights.contiguous(),
        output,
        None,
        None,
        None,
        None,
        None,
        None,
        activation_clamp,
        alpha,
        beta,
    )


def hpc_fuse_moe_mxfp8_k32_bf16_candidate(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor | None = None,
    gate_output: torch.Tensor | None = None,
    down_output: torch.Tensor | None = None,
    activation_clamp: float = 7.0,
    alpha: float = 1.702,
    beta: float = 1.0,
) -> torch.Tensor:
    import hpc  # noqa: F401

    return torch.ops.hpc.fuse_moe_mxfp8_k32_bf16_candidate(
        hidden.contiguous(),
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.float().contiguous(),
        output,
        None,
        None,
        None,
        None,
        None,
        None,
        gate_output,
        None,
        None,
        down_output,
        activation_clamp,
        alpha,
        beta,
    )


__all__ = [
    "has_hpc",
    "has_hpc_mxfp8_k32_moe",
    "hpc_build_mxfp8_k32_moe_routing_cache",
    "hpc_fuse_moe",
    "hpc_fuse_moe_blockwise",
    "hpc_fuse_moe_mxfp8_k32_bf16_candidate",
    "hpc_fuse_moe_mxfp8_k32_candidate",
]
