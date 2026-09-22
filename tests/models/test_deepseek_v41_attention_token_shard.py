# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
)
from vllm.models.deepseek_v41 import attention as attn_module
from vllm.models.deepseek_v41.nvidia import model as model_module
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
_BATCH_SCALE_KEY = replace(kMxfp8Dynamic, scale2=kFp8DynamicTensorSym.scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "has_compressor,has_indexer",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    "tp_size,num_tokens,threshold,quant_key",
    [
        (1, 7, 4, None),
        (4, 0, 4, None),
        (4, 3, 4, None),
        (4, 4, 4, None),
        (4, 7, 4, None),
        (4, 7, 0, None),
        (4, 2, 1, None),  # Ranks with no real rows still join the gather.
        (2, 7, 4, kFp8DynamicTokenSym),
        (8, 9, 4, kMxfp8Dynamic),
        (4, 7, 4, kFp8StaticTensorSym),
        (4, 7, 4, kFp8DynamicTensorSym),
        (4, 7, 4, _BATCH_SCALE_KEY),
    ],
)
def test_forward_restores_tokens_before_attention(
    monkeypatch,
    dtype,
    has_compressor,
    has_indexer,
    tp_size,
    num_tokens,
    threshold,
    quant_key,
):
    """Only shard QKV; retain full auxiliary inputs and the stream policy."""
    torch.manual_seed(42)
    hidden = torch.randn(num_tokens, 6, dtype=dtype)
    weight = torch.randn(6, 5, dtype=dtype)
    aux_weight = torch.randn(6, 2, dtype=torch.float32)
    expected = hidden @ weight
    sharded = (
        threshold > 0
        and num_tokens >= threshold
        and tp_size > 1
        and quant_key not in (kFp8DynamicTensorSym, _BATCH_SCALE_KEY)
    )
    chunk = (num_tokens + tp_size - 1) // tp_size
    padded = F.pad(hidden, (0, 0, 0, chunk * tp_size - num_tokens))
    monkeypatch.setattr(
        attn_module, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(envs, "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", 3)
    execute = attn_module.execute_in_parallel

    def checked_execute(*args, **kwargs):
        assert kwargs["enable"] == (num_tokens <= 3)
        return execute(*args, **kwargs)

    monkeypatch.setattr(attn_module, "execute_in_parallel", checked_execute)
    for rank in range(tp_size):
        monkeypatch.setattr(
            attn_module, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        gathers: list[torch.Tensor] = []

        def gather(local, dim, rank=rank, gathers=gathers):
            assert sharded and dim == 0
            torch.testing.assert_close(
                local, padded[rank * chunk : (rank + 1) * chunk] @ weight
            )
            gathers.append(local)
            return padded @ weight

        monkeypatch.setattr(attn_module, "tensor_model_parallel_all_gather", gather)

        def project(x, rank=rank):
            reference = padded[rank * chunk : (rank + 1) * chunk] if sharded else hidden
            torch.testing.assert_close(x, reference)
            return x @ weight

        def aux_project(x):
            assert x is hidden
            return x.float() @ aux_weight

        def prepare(x, qr, kv, qr_scale, kv_score, index_weights, positions, out):
            assert x is hidden
            assert positions.shape == (num_tokens,)
            torch.testing.assert_close(qr, expected)
            if has_compressor:
                assert kv_score.dtype == torch.float32
                torch.testing.assert_close(kv_score, hidden.float() @ aux_weight)
            else:
                assert kv_score is None
            if has_indexer:
                torch.testing.assert_close(
                    index_weights, (hidden.float() @ aux_weight).to(dtype)
                )
            else:
                assert index_weights is None
            out[:, 0].copy_(qr)

        attn = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
        torch.nn.Module.__init__(attn)
        attn.token_shard_min_tokens = threshold
        attn.fused_wqa_wkv = SimpleNamespace(
            quant_method=(
                UnquantizedLinearMethod()
                if quant_key is None
                else SimpleNamespace(activation_quant_key=quant_key)
            )
        )
        attn.aux_stream_list = None
        attn.ln_events = [None, None, None]
        attn.compressor = object() if has_compressor else None
        attn.indexer = (
            SimpleNamespace(weights_proj=lambda x: (aux_project(x).to(dtype), None))
            if has_indexer
            else None
        )
        attn._fused_wqa_wkv_gemm = project
        attn._compressor_kv_score = aux_project
        attn._split_qkv_and_norm = lambda x: (x, None, x)
        attn._prepare_and_attn_fn = prepare
        attn._o_proj = lambda out, positions: out[:, 0]
        attn.padded_heads = 1
        attn.head_dim = 5

        actual = attn(torch.arange(num_tokens), hidden)
        torch.testing.assert_close(actual, expected)
        assert len(gathers) == int(sharded)


@pytest.mark.parametrize(
    "blocked",
    [None, "tp1", "sp", "pcp", "dcp", "ubatching", "lora", "spec", "invariant", "cpu"],
)
def test_configuration_preserves_unsupported_execution_modes(monkeypatch, blocked):
    """PP2 can opt in without enabling SP; incompatible modes fall back."""
    parallel = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=4,
        enable_expert_parallel=False,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_ubatching=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        kernel_config=SimpleNamespace(moe_backend="CUTLASS"),
        lora_config=object() if blocked == "lora" else None,
        speculative_config=object() if blocked == "spec" else None,
    )
    if blocked == "tp1":
        parallel.tensor_parallel_size = 1
    elif blocked == "sp":
        parallel.pipeline_parallel_size = 1
        parallel.enable_expert_parallel = True
        parallel.data_parallel_size = 2
    elif blocked == "pcp":
        parallel.prefill_context_parallel_size = 2
    elif blocked == "dcp":
        parallel.decode_context_parallel_size = 2
    elif blocked == "ubatching":
        parallel.use_ubatching = True
    monkeypatch.setattr(
        model_module.current_platform, "is_cuda", lambda: blocked != "cpu"
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", blocked == "invariant")
    monkeypatch.setattr(envs, "VLLM_DSV41_ATTN_TOKEN_SHARD_MIN_TOKENS", 2048)
    assert model_module._attention_token_shard_min_tokens(config) == (
        2048 if blocked is None else 0
    )


def test_token_sharding_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_DSV41_ATTN_TOKEN_SHARD_MIN_TOKENS", raising=False)
    assert envs.VLLM_DSV41_ATTN_TOKEN_SHARD_MIN_TOKENS == 0
    assert model_module._attention_token_shard_min_tokens(None) == 0
