# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts

pytestmark = pytest.mark.cpu_test


def test_kimi_mega_moe_preserves_full_batch_capacity():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512,
            max_num_seqs=16,
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
    )

    assert KimiK3MegaMoEExperts._resolve_mega_moe_decode_capacity(config) == 512


def test_kimi_mega_moe_remains_sm100_only(monkeypatch):
    experts = object.__new__(KimiK3MegaMoEExperts)
    experts.w13_weight = SimpleNamespace(device=torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    with pytest.raises(NotImplementedError, match="requires SM100"):
        experts._check_runtime_supported()


def test_kimi_mega_moe_fails_closed_without_situ_deepgemm(monkeypatch):
    class FakeDeepGemm:

        @staticmethod
        def get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        ):
            pass

        @staticmethod
        def transform_weights_for_mega_moe(l1_weights, l2_weights):
            pass

        @staticmethod
        def fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            activation="swiglu",
        ):
            pass

        @staticmethod
        def transform_sf_into_required_layout(*args, **kwargs):
            pass

    experts = object.__new__(KimiK3MegaMoEExperts)
    experts.w13_weight = SimpleNamespace(device=torch.device("cuda"))
    experts.expert_dtype = "fp4"
    experts.hidden_size = 128
    experts.intermediate_size = 128
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: FakeDeepGemm,
    )

    with pytest.raises(NotImplementedError, match="SITU activation support"):
        experts._check_runtime_supported()
