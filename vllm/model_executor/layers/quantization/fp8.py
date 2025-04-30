# SPDX-License-Identifier: Apache-2.0

import os
import importlib.util
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)


from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)

from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d, convert_to_channelwise,
    cutlass_block_fp8_supported, cutlass_fp8_supported,
    maybe_create_device_identity, normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize, requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


def _is_col_major(x: torch.Tensor) -> bool:
    assert x.dim() == 3
    b, m, n = x.shape
    return x.stride(0) == m * n and x.stride(1) == 1 and x.stride(2) == m


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   weight_block_size=weight_block_size)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        return None


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union["Fp8Config", "Fp8MoEInt4Config"]):
        self.quant_config = quant_config
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        self.block_quant = self.quant_config.weight_block_size is not None
        if self.block_quant:
            # Marlin doesn't support block-wise fp8
            self.use_marlin = False

        self.fp8_linear = Fp8LinearOp(
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=cutlass_fp8_supported())

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.quant_config.weight_block_size is not None
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                weight_loader=weight_loader)

                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_fp8_fnuz():
                weight, weight_scale_inv, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            weight = self._maybe_pad_weight(weight)

            # Torch.compile cannot use Parameter subclasses.
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = Parameter(weight_scale_inv,
                                               requires_grad=False)
            return

        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)

            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                assert weight_scale.numel() == 1
                weight_scale = convert_to_channelwise(
                    weight_scale.expand(len(layer.logical_widths)),
                    layer.logical_widths)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp8, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                       requires_grad=False)
            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                weight = layer.weight
                weight_scale = convert_to_channelwise(layer.weight_scale,
                                                      layer.logical_widths)

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            else:
                # Dequant -> Quant with max scale so we can run per tensor.
                weight = layer.weight
                weight_scale = layer.weight_scale

                if current_platform.is_fp8_fnuz():
                    weight, weight_scale, input_scale = \
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=layer.input_scale)
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale,
                                                      requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            weight = self._maybe_pad_weight(weight)
            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return torch.ops.vllm.apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)

class Fp8MoEInt4Config(QuantizationConfig):
    """Config class for ModelOpt W4A8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        is_moe_w4a8_serialized: bool = False,
        activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_moe_w4a8_serialized = is_moe_w4a8_serialized
        if is_moe_w4a8_serialized:
            logger.warning(
                "Detected fp8 moe int4 checkpoint. Please note that"
                " the format is experimental and could change."
            )
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = weight_block_size
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8_moe_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        # return ["hf_quant_config.json"]
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8MoEInt4Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_moe_w4a8_serialized = "moe_int4" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized,
            is_moe_w4a8_serialized,
            activation_scheme,
            ignored_layers,
            weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEInt4MoEMethod(self)
        return None


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

        # Check for DeepGemm support.
        self.allow_deep_gemm = False
        if envs.VLLM_USE_DEEP_GEMM:
            if not has_deep_gemm:
                logger.warning_once("Failed to import DeepGemm kernels.")
            elif (current_platform.is_cuda()
                  and current_platform.has_device_capability(90)):
                logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
                self.allow_deep_gemm = True
            else:
                logger.warning_once(
                    "DeepGemm not supported on the current platform.")

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}.")
            if (tp_size > 1
                    and intermediate_size_per_partition % block_k != 0):
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) //
                         block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
             value} if self.block_quant else
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Lazy import to avoid importing triton too early.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            expand_weights, is_rocm_aiter_block_scaled_moe_enabled,
            is_rocm_aiter_moe_enabled, shuffle_weights)

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight
                w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                   requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                  requires_grad=False)
            if is_rocm_aiter_block_scaled_moe_enabled():
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            # DeepGemm scales need to be transposed and aligned.  We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm:
                # Lazy import to avoid CUDA initialization problems.
                import deep_gemm as dg
                if _is_col_major(layer.w13_weight_scale_inv):
                    layer.w13_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
                if _is_col_major(layer.w2_weight_scale_inv):
                    layer.w2_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()

            return

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
                layer.local_num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                        requires_grad=False)
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            if is_rocm_aiter_moe_enabled():
                # reshaping weights is required for aiter moe kernel.
                w13_scales, w2_scales = expand_weights(
                    layer.w13_weight_scale.data,
                    layer.w2_weight_scale.data,
                    expansion_dims=[
                        layer.w13_weight.shape[1], layer.w2_weight.shape[1]
                    ])
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_scales.contiguous(), requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_scales.contiguous(), requires_grad=False)

                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None
                        or layer.w2_input_scale is None):
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.w13_input_scale)
                        or not all_close_1d(layer.w2_input_scale)):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer.")
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False)
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False)
            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale,
                        layer.w2_input_scale)
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                      requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False)
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                     requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                           requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            if is_rocm_aiter_moe_enabled():
                # reshaping weights is required for aiter moe kernel.
                expansion_dims = [
                    layer.w13_weight.shape[1], layer.w2_weight.shape[1]
                ]
                max_w13_scales, w2_scales = expand_weights(
                    max_w13_scales,
                    layer.w2_weight_scale.data,
                    expansion_dims=expansion_dims)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_scales.contiguous(), requires_grad=False)

                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=(layer.w13_weight_scale_inv
                      if self.block_quant else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale_inv
                      if self.block_quant else layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
            allow_deep_gemm=self.allow_deep_gemm,
        )


class Fp8MoEInt4MoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: Fp8MoEInt4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert "weight_loader" in extra_weight_attrs

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition //
                self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w13_input_scale", w13_input_scale)
        # extra_weight_attrs.update(
        #     {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        # )
        # set_weight_attrs(w13_input_scale, {"scale_type": "input_scale"})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                       dtype=torch.float32),
                                            requires_grad=False)
        # set_weight_attrs(w2_input_scale, {"scale_type": "input_scale"})
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        return

    def _interleave_scales(self, scales: torch.Tensor) -> torch.Tensor:
        """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
        s_shape = scales.shape
        # Reshape to separate groups of 4
        scales_interleaved = scales.reshape(s_shape[0], s_shape[1],
                                            (s_shape[2] // 4), 4)
        # Permute dimensions to interleave
        scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
        # Reshape back to original dimensions but with interleaved values
        scales_interleaved = scales_interleaved.reshape(
            s_shape[0], s_shape[2] // 4, s_shape[1] * 4)
        return scales_interleaved.contiguous()

    def process_weights_after_loading(self, layer: Module) -> None:
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.w2_weight.shape[2] * 2

        # Interleave w13_weight_scale (gate_up_proj)
        # print(f"w13_weight_scale_inv shape {layer.w13_weight_scale_inv.shape}")
        w13_weight_scale = layer.w13_weight_scale_inv.to(torch.bfloat16)
        w13_weight_scale = self._interleave_scales(w13_weight_scale)
        # print(f"w13_weight_scale_inv shape after post process {w13_weight_scale.shape}")
        # layer.w13_weight_scale_inv = Parameter(w13_weight_scale.view(
        #     torch.quint4x2),
        #                                        requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale,
                                               requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        # print(f"w2_weight_scale_inv shape {layer.w2_weight_scale_inv.shape}")
        w2_weight_scale = layer.w2_weight_scale_inv.to(torch.bfloat16)
        w2_weight_scale = self._interleave_scales(w2_weight_scale)
        # print(f"w2_weight_scale_inv shape after post process {w2_weight_scale.shape}")
        # layer.w2_weight_scale_inv = Parameter(w2_weight_scale.view(
        #     torch.quint4x2),
        #                                       requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale,
                                              requires_grad=False)

        # Process input scales
        # print(f"w13_input_scale shape {layer.w13_input_scale.shape}")
        w13_input_scale_scalar = layer.w13_input_scale.max().item()
        w13_input_scale = Parameter(torch.ones(
            hidden_size,
            dtype=torch.bfloat16,
            device=layer.w13_input_scale.device),
                                    requires_grad=False)
        # layer.w13_input_scale = Parameter(w13_input_scale, requires_grad=False)
        layer.w13_input_scale = Parameter(w13_input_scale /
                                          w13_input_scale_scalar,
                                          requires_grad=False)
        # print(f"w13_input_scale shape after post process {w13_input_scale.shape}")

        # print(f"w2_input_scale shape {layer.w2_input_scale.shape}")
        w2_input_scale_scalar = layer.w2_input_scale.max().item()
        w2_input_scale = Parameter(torch.ones(
            intermediate_size_per_partition,
            dtype=torch.bfloat16,
            device=layer.w2_input_scale.device),
                                   requires_grad=False)
        # layer.w2_input_scale = Parameter(w2_input_scale, requires_grad=False)
        layer.w2_input_scale = Parameter(w2_input_scale /
                                         w2_input_scale_scalar,
                                         requires_grad=False)
        # print(f"w2_input_scale shape after post process {w2_input_scale.shape}")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        m = x.shape[0]
        k = x.shape[1]
        n = layer.w13_weight.shape[1] / 2
        device = layer.w13_weight.device

        # return ops.group_gemm_xx()
        num_experts = layer.w2_weight.shape[0]
        # a_strides1 = torch.full(
        #     (num_experts,), k, device=device, dtype=torch.int64
        # )
        a_strides1 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        a_strides1[:, 0] = k
        a_strides1[:, 1] = 1
        a_strides1[:, 2] = 0
        # print(f"a_strides1 shape {a_strides1.shape}")
        # b_strides1 = torch.full((num_experts, ),
        #                         k / 2,
        #                         device=device,
        #                         dtype=torch.int64)
        b_strides1 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        b_strides1[:, 0] = k // 2
        b_strides1[:, 1] = 1
        b_strides1[:, 2] = 0
        # print(f"b_strides1 shape {b_strides1.shape}")
        # c_strides1 = torch.full(
        #     (num_experts, ),
        #     2 * n,
        #     device=device,
        #     dtype=torch.int64,
        # )
        c_strides1 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        c_strides1[:, 0] = 1
        c_strides1[:, 1] = 2 * n
        c_strides1[:, 2] = 0
        # print(f"c_strides1 shape {c_strides1.shape}")
        # a_strides2 = torch.full(
        #     (num_experts, ),
        #     n,
        #     device=device,
        #     dtype=torch.int64,
        # )
        a_strides2 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        a_strides2[:, 0] = n
        a_strides2[:, 1] = 1
        a_strides2[:, 2] = 0
        # print(f"a_strides2 shape {a_strides2.shape}")
        # b_strides2 = torch.full(
        #     (num_experts, ),
        #     n / 2,
        #     device=device,
        #     dtype=torch.int64,
        # )
        b_strides2 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        b_strides2[:, 0] = n // 2 # Use integer division
        b_strides2[:, 1] = 1
        b_strides2[:, 2] = 0
        # print(f"b_strides2 shape {b_strides2.shape}")
        # c_strides2 = torch.full((num_experts, ),
        #                         k,
        #                         device=device,
        #                         dtype=torch.int64)
        c_strides2 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        c_strides2[:, 0] = 1
        c_strides2[:, 1] = k
        c_strides2[:, 2] = 0
        # print(f"c_strides2 shape {c_strides2.shape}")
        # s_strides13 = torch.full(
        #     (num_experts, ),
        #     n * 2,
        #     device=device,
        #     dtype=torch.int64,
        # )
        s_strides13 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        s_strides13[:, 0] = 1
        s_strides13[:, 1] = 2 * n
        s_strides13[:, 2] = 0
        # print(f"s_strides13 shape {s_strides13.shape}")
        # s_strides2 = torch.full(
        #     (num_experts, ),
        #     k,
        #     device=device,
        #     dtype=torch.int64,
        # )
        s_strides2 = torch.empty((num_experts, 3), device=device, dtype=torch.int64)
        s_strides2[:, 0] = 1
        s_strides2[:, 1] = k
        s_strides2[:, 2] = 0
        # print(f"s_strides2 shape {s_strides2.shape}")
        # print(
        #     f"w13_weight shape, dtype: {layer.w13_weight.shape, layer.w13_weight.dtype}"
        # )
        # print(
        #     f"w2_weight shape, dtype: {layer.w2_weight.shape, layer.w2_weight.dtype}"
        # )
        # print(
        #     f"w13_weight_scale_inv shape, dtype: {layer.w13_weight_scale_inv.shape, layer.w13_weight_scale_inv.dtype}"
        # )
        # print(
        #     f"w2_weight_scale_inv shape, dtype: {layer.w2_weight_scale_inv.shape, layer.w2_weight_scale_inv.dtype}"
        # )
        # print(
        #     f"w13_input_scale shape, dtype: {layer.w13_input_scale.shape, layer.w13_input_scale.dtype}"
        # )
        # print(
        #     f"w2_input_scale shape, dtype: {layer.w2_input_scale.shape, layer.w2_input_scale.dtype}"
        # )

        # 获取当前设备ID
        device_id = device.index
        # 创建保存目录
        save_dir = f"/nvme0n1/w4a8_debug_tensors/device_{device_id}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            # 定义要保存的张量列表
            tensors = {
                "x": x,
                "w13_weight": layer.w13_weight,
                "w2_weight": layer.w2_weight,
                "w13_weight_scale_inv": layer.w13_weight_scale_inv,
                "w2_weight_scale_inv": layer.w2_weight_scale_inv,
                "topk_weights": topk_weights,
                "topk_ids": topk_ids,
                "w13_input_scale": layer.w13_input_scale,
                "w2_input_scale": layer.w2_input_scale,
                "a_strides1": a_strides1,
                "b_strides1": b_strides1,
                "c_strides1": c_strides1,
                "a_strides2": a_strides2,
                "b_strides2": b_strides2,
                "c_strides2": c_strides2,
                "s_strides13": s_strides13,
                "s_strides2": s_strides2,
            }

            # 保存形状信息到单独文件
            with open(f"{save_dir}/shapes_and_dtypes.txt", "w") as f:
                for name, tensor in tensors.items():
                    f.write(
                        f"{name}: {tensor.shape}, {tensor.dtype}, {tensor.device}\n"
                    )
                f.write(
                    f"apply_router_weight_on_input: {apply_router_weight_on_input}\n"
                )

            # 保存每个张量的数据到单独文件
            for name, tensor in tensors.items():
                torch.save(tensor, f"{save_dir}/{name}.pt")

        return cutlass_w4a8_moe(
            x,
            layer.w13_weight,  # Alreay transpose
            layer.w2_weight,  # Alreay transpose
            layer.w13_weight_scale_inv,  # Already interleaved
            layer.w2_weight_scale_inv,  # Already interleaved
            topk_weights,
            topk_ids,
            a_strides1,
            b_strides1,
            c_strides1,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides13,
            s_strides2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            apply_router_weight_on_input,
        )
        # return x

class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Union[Fp8Config, Fp8MoEInt4Config]):
        super().__init__(quant_config)
