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
