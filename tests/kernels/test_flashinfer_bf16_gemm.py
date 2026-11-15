# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import utils as layer_utils
from vllm.utils.flashinfer import (
    autotune,
    flashinfer_bf16_bmm,
    is_flashinfer_bf16_bmm_supported,
    is_flashinfer_cutedsl_bf16_gemm_supported,
)


@pytest.fixture(scope="module")
def require_flashinfer_bf16_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not is_flashinfer_cutedsl_bf16_gemm_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 3072)])
@pytest.mark.usefixtures("require_flashinfer_bf16_cutedsl")
@pytest.mark.parametrize(
    "m,use_bias,pdl",
    [
        (1, False, False),
        (8, True, False),
        (16, False, True),
        (32, True, True),
        (64, True, True),
        (512, False, False),
    ],
)
def test_flashinfer_bf16_cutedsl_correctness(
    m: int,
    n: int,
    k: int,
    use_bias: bool,
    pdl: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1 if use_bias else None
    )

    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, bias, pdl, "flashinfer_cutedsl"
    )
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("k,n", [(192, 512), (512, 256)])
def test_flashinfer_cudnn_bmm_keeps_mla_strided_output(k: int, n: int) -> None:
    """The exact MLA activation/weight/output views must preserve the BF16 result."""
    if not is_flashinfer_bf16_bmm_supported("cudnn"):
        pytest.skip("FlashInfer cuDNN BF16 BMM is unavailable")
    torch.manual_seed(0)
    rows, heads = 1024, 64
    a = (
        torch.randn(rows, heads, k, device="cuda", dtype=torch.bfloat16) * 0.1
    ).transpose(0, 1)
    if k == 192:
        weight = torch.randn(n, heads, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = weight.permute(1, 2, 0)
    else:
        weight = torch.randn(k, heads, n, device="cuda", dtype=torch.bfloat16) * 0.1
        b = weight.transpose(0, 1)
    out = torch.empty(rows, heads, n, device="cuda", dtype=torch.bfloat16)
    out_view = out.transpose(0, 1)
    expected = torch.bmm(a, b)
    with autotune(tune_mode=True):
        flashinfer_bf16_bmm(a, b, out_view, backend="cudnn")
    actual = flashinfer_bf16_bmm(a, b, out_view, backend="cudnn")
    assert actual.data_ptr() == out_view.data_ptr()
    assert actual.stride() == out_view.stride()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
