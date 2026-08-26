# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kInt4Static,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)


class W4A8MoeBackend(Enum):
    CUTLASS = "CUTLASS"
    HUMMING = "HUMMING"


def backend_to_kernel_cls(
    backend: W4A8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == W4A8MoeBackend.CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassBatchedExpertsW4A8Fp8,
            CutlassExpertsW4A8Fp8,
        )

        return [CutlassExpertsW4A8Fp8, CutlassBatchedExpertsW4A8Fp8]
    elif backend == W4A8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            HummingGroupedExperts,
            HummingIndexedExperts,
        )

        return [
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    else:
        raise ValueError(f"Unknown W4A8 MoE backend: {backend.value}")


def map_w4a8_backend(runner_backend: MoEBackend) -> W4A8MoeBackend:
    backend_map = {
        "auto": W4A8MoeBackend.CUTLASS,
        "cutlass": W4A8MoeBackend.CUTLASS,
        "humming": W4A8MoeBackend.HUMMING,
    }
    if runner_backend not in backend_map:
        raise NotImplementedError(
            f"moe_backend={runner_backend!r} is not supported for W4A8 MoE. "
            "Supported backends are 'auto', 'cutlass', and 'humming'."
        )
    return backend_map[runner_backend]


def select_w4a8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt4Static,
    activation_key: QuantKey | None = kFp8DynamicTokenSym,
) -> tuple[W4A8MoeBackend, type[mk.FusedMoEExperts]]:
    backend = map_w4a8_backend(config.moe_backend)

    if (
        backend == W4A8MoeBackend.HUMMING
        and (
            config.moe_parallel_config.use_batched_activation_format
            or config.moe_parallel_config.use_fi_nvl_one_sided_kernels
        )
    ):
        raise NotImplementedError(
            "Humming W4A8 does not support the selected communication format."
        )

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    last_reason: str | None = None
    for kernel_cls in backend_to_kernel_cls(backend):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            logger.info_once("Using %s W4A8 MoE backend.", backend.value)
            return backend, kernel_cls
        last_reason = reason

    raise NotImplementedError(
        f"W4A8 MoE backend {backend.value} does not support the "
        f"deployment configuration: {last_reason}."
    )


def convert_to_w4a8_moe_kernel_format(
    w13_weight_packed: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        convert_bf16_scales_to_fp8,
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    convert_packed_uint4b8_to_signed_int4_inplace(w13_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w13_weight_shuffled, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w13_weight_packed
    )

    convert_packed_uint4b8_to_signed_int4_inplace(w2_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w2_weight_shuffled, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w2_weight_packed
    )

    w13_weight_scale, w13_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w13_weight_scale
    )
    w2_weight_scale, w2_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w2_weight_scale
    )

    # Scales are stored as (E, N, K // 128), but the kernel expects
    # (E, K // 128, N) in row-major format.
    w13_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w13_weight_scale.permute(0, 2, 1).contiguous()
    )
    w2_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w2_weight_scale.permute(0, 2, 1).contiguous()
    )

    return (
        w13_weight_shuffled,
        w2_weight_shuffled,
        w13_weight_scale_packed,
        w2_weight_scale_packed,
        w13_weight_chan_scale,
        w2_weight_chan_scale,
        b_strides1,
        b_strides2,
    )


def make_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    return int4_w4afp8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=True,
        per_out_ch_quant=True,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def make_w4a8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    group_size: int,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    expert_kwargs = {}
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        expert_kwargs = {
            "max_num_tokens": prepare_finalize.max_num_tokens_per_rank(),
            "num_dispatchers": prepare_finalize.num_dispatchers(),
        }

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        group_size=group_size,
        **expert_kwargs,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )


def make_humming_w4a8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    layer: "RoutedExperts",
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        make_humming_moe_kernel,
    )

    return make_humming_moe_kernel(
        moe_quant_config=moe_quant_config,
        moe_config=moe_config,
        experts_cls=experts_cls,
        layer=layer,
        routing_tables=routing_tables,
    )


def _quant_args_to_humming_weight_config(
    quant_args: Any,
    *,
    format: str,
) -> dict[str, Any]:
    def _value(value):
        return value.value if hasattr(value, "value") else value

    config = {
        "quant_method": "compressed-tensors",
        "format": format,
        "type": _value(quant_args.type),
        "num_bits": quant_args.num_bits,
        "strategy": _value(quant_args.strategy),
        "symmetric": quant_args.symmetric,
    }
    group_size = getattr(quant_args, "group_size", None)
    if group_size is not None:
        config["group_size"] = group_size
    elif format != "pack-quantized":
        config["group_size"] = 0
    actorder = getattr(quant_args, "actorder", None)
    if actorder is not None:
        config["actorder"] = _value(actorder)
    return config


def _quant_args_to_humming_input_config(
    quant_args: Any,
    *,
    format: str,
) -> dict[str, Any]:
    def _value(value):
        return value.value if hasattr(value, "value") else value

    return {
        "quant_method": "compressed-tensors",
        "format": format,
        "type": _value(quant_args.type),
        "num_bits": quant_args.num_bits,
        "strategy": _value(quant_args.strategy),
        "dynamic": quant_args.dynamic,
        "group_size": getattr(quant_args, "group_size", None) or 0,
        "symmetric": quant_args.symmetric,
    }


def convert_to_humming_w4a8_moe_kernel_format(
    layer: "RoutedExperts",
    weight_quant: Any,
    input_quant: Any,
) -> None:
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        convert_to_humming_moe_kernel_format,
    )
    from vllm.utils.humming import BaseInputSchema, BaseWeightSchema

    weight_schema = BaseWeightSchema.from_config(
        _quant_args_to_humming_weight_config(
            weight_quant,
            format="pack-quantized",
        )
    )
    input_schema = BaseInputSchema.from_config(
        _quant_args_to_humming_input_config(
            input_quant,
            format="float-quantized",
        )
    )

    convert_to_humming_moe_kernel_format(
        layer=layer,
        quant_config=None,
        weight_schema=weight_schema,
        input_schema=input_schema,
        sublayer_configs=None,
        force_weight_schema=None,
    )
