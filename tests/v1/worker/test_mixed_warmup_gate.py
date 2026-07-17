# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
)
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_deepseek_v4_mixed_warmup_skipped_for_pcp():
    class _Backend:
        @staticmethod
        def get_name():
            return "DEEPSEEK_SPARSE_SWA"

    runner = SimpleNamespace(
        is_pooling_model=False,
        attn_groups=[[SimpleNamespace(backend=_Backend())]],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
        ),
        _dummy_run=_fail,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
    )

    assert deepseek_v4_sparse_mla_attention_warmup(worker) is None
