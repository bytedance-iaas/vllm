# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from ..op import register_op


@register_op
def sequence_parallel_boundary(x: Tensor) -> Tensor:
    """Mark a tensor that may need materializing at an SP graph boundary."""
    return x


@sequence_parallel_boundary.register_input_generator
def _sequence_parallel_boundary_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple[Tensor]:
    return (torch.randn(num_tokens, hidden_size, dtype=dtype),)
