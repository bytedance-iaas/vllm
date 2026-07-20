# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.warmup import (
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)


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


def test_mixed_warmup_skipped_for_pure_prefill_pcp():
    """A pure-prefill PCP worker must not synthesize a decode request."""
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        pcp_manager=SimpleNamespace(requires_pure_prefill=True),
    )

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_kernel_warmup_keeps_pure_prefill_pcp_batch_pure(monkeypatch):
    class FakeConnector:
        def __init__(self):
            self.disabled = []

        def set_disabled(self, disabled):
            self.disabled.append(disabled)

    connector = FakeConnector()
    runner = SimpleNamespace(
        num_speculative_steps=0,
        decode_query_len=1,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=16,
        ),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_encoder_decoder=False,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=128,
        ),
        is_pooling_model=False,
        pcp_manager=SimpleNamespace(requires_pure_prefill=True),
        kv_connector=connector,
    )
    execute_calls = []
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.warmup.torch.accelerator.synchronize", lambda: None
    )

    warmup_kernels(runner, execute_calls.append, _fail)

    assert len(execute_calls) == 2
    assert execute_calls[0].scheduled_new_reqs
    assert execute_calls[1].finished_req_ids
    assert connector.disabled == [True, False]
