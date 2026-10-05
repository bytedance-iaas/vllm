import dataclasses
from typing import Any

import torch

from humming.schema.mxfp4 import Mxfp4WeightSchema


@dataclasses.dataclass(kw_only=True)
class GptOssMxfp4WeightSchema(Mxfp4WeightSchema):
    quant_method: str = "gpt_oss_mxfp4"

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        tensor_meta = {
            "weight": {
                "shape": (shape_n, shape_k // 2),
                "dtype": torch.uint8,
                "extra_attrs": {"output_dim": 0, "input_dim": 1},
            },
            "weight_scale": {
                "shape": (shape_n, shape_k // 32),
                "dtype": torch.uint8,
                "extra_attrs": {"output_dim": 0, "input_dim": 1, "scale_type": "group"},
            },
        }

        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta
