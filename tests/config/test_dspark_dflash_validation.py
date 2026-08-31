# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig

pytestmark = pytest.mark.cpu_test


def _make_block_size_config(
    method: str,
    draft_model_type: str | None = None,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            dcp_kv_cache_interleave_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=draft_model_type)
            ),
        ),
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


def test_dcp_rejects_draft_model():
    config = _make_block_size_config("draft_model")

    with pytest.raises(
        NotImplementedError,
        match="Draft-model speculative decoding does not support",
    ):
        VllmConfig.validate_block_size(config)


def test_dcp_rejects_step3_mtp():
    config = _make_block_size_config("mtp", draft_model_type="step3p5_mtp")

    with pytest.raises(
        NotImplementedError,
        match="Step3 MTP speculative decoding does not support",
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


def test_v2_runner_rejects_replicated_eagle3_draft_kv():
    config = SimpleNamespace(
        model_config=None,
        speculative_config=SimpleNamespace(
            method="eagle3",
            parallel_drafting=False,
            enable_eagle3_replicated_draft_kv=True,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=8,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=None,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "EAGLE3 replicated draft KV" in unsupported


def test_v2_runner_rejects_eagle3_target_dense_full_temporal_kv():
    config = SimpleNamespace(
        model_config=None,
        speculative_config=SimpleNamespace(
            method="eagle3",
            parallel_drafting=False,
            enable_eagle3_replicated_draft_kv=False,
            enable_eagle3_target_dense_full_temporal_kv=True,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=8,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=None,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "EAGLE3 target dense full-temporal KV" in unsupported


def test_dspark_allows_static_k_below_checkpoint_block_size(monkeypatch):
    def _fake_draft_model_config(*args, **kwargs):
        hf_config = SimpleNamespace(
            model_type="dspark",
            architectures=["Qwen3DSparkModel"],
            dspark_block_size=5,
            n_predict=5,
        )
        return SimpleNamespace(
            model=kwargs["model"],
            hf_config=hf_config,
            architectures=["Qwen3DSparkModel"],
            max_model_len=kwargs["spec_target_max_model_len"],
        )

    monkeypatch.setattr("vllm.config.speculative.ModelConfig", _fake_draft_model_config)

    target_model_config = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4",
        quantization=None,
        tokenizer="deepseek-ai/DeepSeek-V4",
        tokenizer_mode="auto",
        trust_remote_code=True,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="float16",
        seed=0,
        tokenizer_revision=None,
        max_model_len=8192,
        enforce_eager=False,
        max_logprobs=20,
        hf_overrides={},
        config_format="hf",
    )

    speculative_config = SpeculativeConfig(
        method="dspark",
        num_speculative_tokens=4,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )

    assert speculative_config.num_speculative_tokens == 4
    assert speculative_config.draft_model_config.hf_config.dspark_block_size == 5
