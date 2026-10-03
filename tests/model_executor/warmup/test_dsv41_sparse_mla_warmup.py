# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.warmup.flashinfer_sparse_mla_warmup as warmup_module
import vllm.model_executor.warmup.kernel_warmup as kernel_warmup_module
import vllm.v1.worker.gpu.warmup as gpu_warmup_module

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_encoder_only_prefill_skips_mixed_sparse_mla_warmup(monkeypatch):
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(is_pooling_model=False),
        vllm_config=SimpleNamespace(is_dsv41_encoder_only_prefill=True),
    )
    monkeypatch.setattr(
        warmup_module,
        "_has_deepseek_v4_sparse_mla_backend",
        lambda runner: pytest.fail("backend inspection should be skipped"),
    )
    monkeypatch.setattr(
        warmup_module,
        "run_mixed_prefill_decode_warmup",
        lambda *args, **kwargs: pytest.fail("mixed warmup should be skipped"),
    )

    warmup_module.deepseek_v4_sparse_mla_attention_warmup(worker)


@pytest.mark.parametrize(
    ("hisparse_config", "encoder_only", "pcp_size", "expected"),
    [
        (object(), False, 1, True),
        (None, True, 4, True),
        (None, True, 1, False),
        (None, False, 4, False),
    ],
)
def test_flashinfer_autotune_attention_gate(
    hisparse_config,
    encoder_only,
    pcp_size,
    expected,
):
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=hisparse_config),
            is_dsv41_encoder_only_prefill=encoder_only,
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
            ),
        )
    )

    assert (
        kernel_warmup_module._skip_attention_in_flashinfer_autotune(runner) is expected
    )


def test_encoder_only_prefill_skips_scheduler_warmup():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            is_mm_encoder_only=False,
            is_dsv41_encoder_only_prefill=True,
        )
    )

    def fail(*args, **kwargs):
        pytest.fail("scheduler warmup must be skipped")

    gpu_warmup_module._warmup_kernels(runner, fail, fail)
