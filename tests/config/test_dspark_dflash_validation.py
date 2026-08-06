# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.config import VllmConfig

pytestmark = pytest.mark.cpu_test


def _make_block_size_config(method: str):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            dcp_kv_cache_interleave_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=SimpleNamespace(method=method),
        scheduler_config=SimpleNamespace(),
    )


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_dcp_rejects_parallel_drafters(method):
    config = _make_block_size_config(method)

    with pytest.raises(
        NotImplementedError,
        match="does not support decode context parallelism",
    ):
        VllmConfig.validate_block_size(config)


def test_dcp_allows_other_speculative_methods():
    config = _make_block_size_config("mtp")

    VllmConfig.validate_block_size(config)


def test_explicit_v1_runner_rejected_for_dspark(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark"),
    )

    with pytest.raises(ValueError, match="requires the V2 model runner"):
        VllmConfig.use_v2_model_runner.fget(config)
