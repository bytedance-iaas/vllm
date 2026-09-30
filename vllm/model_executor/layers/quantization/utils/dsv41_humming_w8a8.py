# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in H20 W8A8 Humming path for DeepSeek-V4.1 target WQ_B."""

from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.humming import (
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)

if TYPE_CHECKING:
    from vllm.utils.humming import LayerConfig

_BLOCK_SIZE = 32
_INPUT_SIZE = 1280
_OUTPUT_SIZE = 32768
_VERIFY_BATCH_SIZE = 96


def _quantize_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (*x.shape[:-1], x.shape[-1] // _BLOCK_SIZE),
        dtype=torch.float32,
        device=x.device,
    )
    fp8_min, fp8_max = get_fp8_min_max()
    groups = x.numel() // _BLOCK_SIZE
    _per_token_group_quant_fp8[(groups,)](
        x,
        quantized,
        scale,
        _BLOCK_SIZE,
        x.shape[-1],
        x.stride(0),
        1e-10,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=True,
        BLOCK=_BLOCK_SIZE,
        num_warps=1,
        num_stages=1,
    )
    return quantized, scale


def _collapse_repeated_block_scales(weight_scale: torch.Tensor) -> torch.Tensor:
    expected_shape = (_OUTPUT_SIZE, _INPUT_SIZE // _BLOCK_SIZE)
    if weight_scale.dtype != torch.uint8 or tuple(weight_scale.shape) != expected_shape:
        raise ValueError(
            "DSv4.1 WQ_B expects expanded uint8 scales with shape "
            f"{expected_shape}, got {weight_scale.dtype} {tuple(weight_scale.shape)}"
        )

    grouped = weight_scale.view(
        _OUTPUT_SIZE // _BLOCK_SIZE,
        _BLOCK_SIZE,
        _INPUT_SIZE // _BLOCK_SIZE,
    )
    first_rows = grouped[:, :1, :]
    if not torch.equal(grouped, first_rows.expand_as(grouped)):
        raise ValueError("DSv4.1 WQ_B scales are not constant within 32-row blocks")

    return first_rows[:, 0, :].contiguous().view(torch.float8_e8m0fnu).float()


class Dsv41HummingW8A8WqB(torch.nn.Module):
    """SGLang-compatible group-32 W8A8 Humming execution for target WQ_B."""

    layer_config: "LayerConfig"

    def __init__(self) -> None:
        super().__init__()
        self.compute_config = ""

    @classmethod
    @torch.no_grad()
    def from_layer(cls, layer: torch.nn.Module) -> "Dsv41HummingW8A8WqB":
        weight = layer.weight.detach()
        if (
            weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (_OUTPUT_SIZE, _INPUT_SIZE)
            or layer.params_dtype != torch.bfloat16
            or layer.has_bias
        ):
            raise ValueError("Unsupported DSv4.1 WQ_B tensor contract")

        packed = cls()
        packed.register_parameter("weight", Parameter(weight, requires_grad=False))
        packed.register_parameter(
            "weight_scale_inv",
            Parameter(
                _collapse_repeated_block_scales(layer.weight_scale.detach()),
                requires_grad=False,
            ),
        )
        packed.input_size_per_partition = _INPUT_SIZE
        packed.output_size_per_partition = _OUTPUT_SIZE
        packed.output_partition_sizes = [_OUTPUT_SIZE]
        packed.params_dtype = torch.bfloat16
        packed.has_bias = False

        from vllm.utils.humming import (
            HummingInputSchema,
            InputQuantizationMode,
            dtypes,
        )

        input_schema = HummingInputSchema(
            a_dtype=dtypes.float8e4m3,
            input_scale_group_size=_BLOCK_SIZE,
            input_scale_dtype=dtypes.float32,
            input_quant_mode=InputQuantizationMode.DynamicGroup,
        )
        packed.layer_config = prepare_humming_linear_layer_config(
            packed,
            {
                "quant_method": "fp8",
                "weight_block_size": [_BLOCK_SIZE, _BLOCK_SIZE],
            },
            input_schema=input_schema,
            allow_input_fallback=False,
        )
        packed.compute_config = get_humming_linear_compute_config()
        packed.register_buffer(
            "locks",
            torch.zeros(1024, dtype=torch.int32, device=packed.weight.device),
            persistent=False,
        )
        return packed

    def supports(self, x: torch.Tensor) -> bool:
        return (
            x.dtype == torch.bfloat16
            and x.shape[-1] == _INPUT_SIZE
            and x.numel() == _VERIFY_BATCH_SIZE * _INPUT_SIZE
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.supports(x):
            raise ValueError("DSv4.1 WQ_B W8A8 Humming requires BF16 M96xK1280 input")

        from vllm.utils.humming import humming_forward

        flat_x = x.reshape(_VERIFY_BATCH_SIZE, _INPUT_SIZE).contiguous()
        quantized_x, input_scale = _quantize_input(flat_x)
        output = humming_forward(
            self.layer_config,
            inputs=quantized_x,
            input_scale=input_scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            locks=self.locks,
            compute_config=self.compute_config,
        )
        return output.view(*x.shape[:-1], _OUTPUT_SIZE)
