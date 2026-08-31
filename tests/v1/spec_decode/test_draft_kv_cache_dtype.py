# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_draft_cache_config

pytestmark = pytest.mark.cpu_test


def _make_config(
    target_dtype: str,
    draft_dtype: str | None,
    *,
    use_mla: bool,
):
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype=target_dtype,
    )
    speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype,
        draft_model_config=SimpleNamespace(use_mla=use_mla),
    )
    return SimpleNamespace(
        cache_config=cache_config,
        speculative_config=speculative_config,
    )


@pytest.mark.parametrize("draft_dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_explicit_draft_kv_dtype_wins_over_dense_fallback(draft_dtype):
    vllm_config = _make_config("fp8_ds_mla", draft_dtype, use_mla=False)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config.cache_dtype == draft_dtype
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache_config is not vllm_config.cache_config


def test_dense_draft_falls_back_from_inherited_fp8_ds_mla():
    vllm_config = _make_config("fp8_ds_mla", None, use_mla=False)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config.cache_dtype == "auto"
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache_config is not vllm_config.cache_config


@pytest.mark.parametrize(
    ("target_dtype", "use_mla"),
    [("fp8_ds_mla", True), ("fp8", False), ("auto", False)],
)
def test_compatible_draft_inherits_target_cache_config(target_dtype, use_mla):
    vllm_config = _make_config(target_dtype, None, use_mla=use_mla)

    draft_cache_config = get_draft_cache_config(vllm_config)

    assert draft_cache_config is vllm_config.cache_config
