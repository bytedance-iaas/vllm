# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-safe guard tests for DeepSeek V4 SM90/SM100 MegaMoE experts.

These tests intentionally avoid CUDA so they run in CPU CI. They cover the
loader-side parameter shapes and FP8 scale sharding logic, which are pure
PyTorch/host operations and do not require a GPU.
"""

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts


def _make_vllm_config(max_num_batched_tokens: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_fp8_experts(
    hidden_size: int = 256,
    intermediate_size: int = 256,
    num_experts: int = 4,
    num_local_experts: int = 2,
    experts_start_idx: int = 2,
    top_k: int = 2,
) -> DeepseekV4MegaMoEExperts:
    return DeepseekV4MegaMoEExperts(
        _make_vllm_config(),
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="model.layers.0.ffn.experts",
        expert_dtype="fp8",
    )


def test_fp8_loader_params_have_expected_shapes_and_dtypes():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    assert experts.expert_dtype == "fp8"
    # FP8 path must not allocate the FP4/UE8M0 packed scale params.
    assert not hasattr(experts, "w13_weight_scale")
    assert not hasattr(experts, "w2_weight_scale")

    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight.shape == (2, 2 * intermediate_size, hidden_size)
    assert experts.w2_weight.dtype == torch.float8_e4m3fn
    assert experts.w2_weight.shape == (2, hidden_size, intermediate_size)

    scale_n = (intermediate_size + 127) // 128
    scale_h = (hidden_size + 127) // 128
    assert experts.w13_weight_scale_inv.dtype == torch.float32
    assert experts.w13_weight_scale_inv.shape == (2, 2 * scale_n, scale_h)
    assert experts.w2_weight_scale_inv.dtype == torch.float32
    assert experts.w2_weight_scale_inv.shape == (2, scale_h, scale_n)


def test_fp8_weight_loader_packs_w1_w3_and_w2():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    # Non-local expert (id=1 is not owned by experts_start_idx=2 rank) must be
    # rejected and leave the local data untouched.
    nonlocal_w1 = torch.ones(
        intermediate_size, hidden_size, dtype=torch.float8_e4m3fn
    )
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_w1,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full(
        (intermediate_size, hidden_size), 2.0, dtype=torch.float8_e4m3fn
    )
    w3 = torch.full(
        (intermediate_size, hidden_size), 3.0, dtype=torch.float8_e4m3fn
    )
    w2 = torch.full(
        (hidden_size, intermediate_size), 4.0, dtype=torch.float8_e4m3fn
    )

    assert experts.weight_loader(
        experts.w13_weight, w1, "experts.w13_weight", shard_id="w1",
        expert_id=2, return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight, w3, "experts.w13_weight", shard_id="w3",
        expert_id=2, return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight, w2, "experts.w2_weight", shard_id="w2",
        expert_id=2, return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :intermediate_size], w1)
    assert torch.equal(experts.w13_weight[0, intermediate_size:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    # Second local expert (global id 3) is untouched.
    assert torch.count_nonzero(experts.w13_weight[1].float()) == 0


def test_fp8_weight_loader_shards_scales_by_block_count():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_fp8_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    scale_n = (intermediate_size + 127) // 128
    scale_h = (hidden_size + 127) // 128

    w1_sf = torch.full((scale_n, scale_h), 0.5, dtype=torch.float32)
    w3_sf = torch.full((scale_n, scale_h), 0.25, dtype=torch.float32)
    w2_sf = torch.full((scale_h, scale_n), 0.125, dtype=torch.float32)

    assert experts.weight_loader(
        experts.w13_weight_scale_inv, w1_sf, "experts.w13_weight_scale_inv",
        shard_id="w1", expert_id=2, return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight_scale_inv, w3_sf, "experts.w13_weight_scale_inv",
        shard_id="w3", expert_id=2, return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight_scale_inv, w2_sf, "experts.w2_weight_scale_inv",
        shard_id="w2", expert_id=2, return_success=True,
    )

    assert torch.equal(experts.w13_weight_scale_inv[0, :scale_n], w1_sf)
    assert torch.equal(experts.w13_weight_scale_inv[0, scale_n:], w3_sf)
    assert torch.equal(experts.w2_weight_scale_inv[0], w2_sf)
    assert torch.count_nonzero(experts.w13_weight_scale_inv[1]) == 0


def test_fp4_loader_params_unchanged():
    hidden_size = 256
    intermediate_size = 256
    experts = DeepseekV4MegaMoEExperts(
        _make_vllm_config(),
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="model.layers.0.ffn.experts",
        expert_dtype="fp4",
    )

    assert experts.expert_dtype == "fp4"
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w13_weight.shape == (2, 2 * intermediate_size, hidden_size // 2)
    assert experts.w13_weight_scale.dtype == torch.uint8
    assert experts.w13_weight_scale.shape == (
        2,
        2 * intermediate_size,
        hidden_size // 32,
    )
    assert not hasattr(experts, "w13_weight_scale_inv")
    assert not hasattr(experts, "w2_weight_scale_inv")
