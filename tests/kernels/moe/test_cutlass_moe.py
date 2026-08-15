# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import dataclasses
from math import prod
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.cutlass_moe as cutlass_moe
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassBatchedExpertsFp8,
    CutlassBatchedExpertsW4A8Fp8,
    CutlassExpertsFp4,
    CutlassExpertsFp8,
    CutlassExpertsW4A8Fp8,
    run_cutlass_moe_fp8,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    W4A8MoeBackend,
    make_w4a8_moe_quant_config,
    select_w4a8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kInt4Static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (7, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (224, 1024, 1024),
    (224, 3072, 1024),
    (224, 3072, 1536),
    (32768, 1024, 1024),
    # These sizes trigger wrong answers.
    # (7232, 2048, 5120),
    # (40000, 2048, 5120),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def test_cutlass_moe_supports_gelu_tanh_activation_metadata():
    assert CutlassExpertsFp8._supports_activation(MoEActivation.GELU_TANH)
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH)
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH_NO_MUL)


@pytest.mark.parametrize(
    "experts_cls",
    [CutlassExpertsFp8, CutlassExpertsW4A8Fp8],
)
def test_cutlass_permute_scratch_covers_naive_dp_ep_gather(experts_cls):
    config = make_dummy_moe_config()
    config.moe_parallel_config = dataclasses.replace(
        config.moe_parallel_config,
        dp_size=8,
        ep_size=8,
        use_ep=True,
    )
    experts = object.__new__(experts_cls)
    object.__setattr__(experts, "moe_config", config)
    object.__setattr__(experts, "_permute_scratch", None)
    scratch = object()

    with (
        patch.object(
            cutlass_moe,
            "moe_permute_unpermute_supported",
            return_value=True,
        ),
        patch.object(
            cutlass_moe,
            "MoEPermuteScratch",
            return_value=scratch,
        ) as scratch_cls,
    ):
        result = experts._get_permute_scratch()

    assert result is scratch
    assert scratch_cls.call_args.kwargs["max_num_tokens"] == (
        config.max_num_tokens * config.dp_size
    )


def test_cutlass_batched_permute_scratch_keeps_per_rank_capacity():
    config = make_dummy_moe_config()
    config.moe_parallel_config = dataclasses.replace(
        config.moe_parallel_config,
        dp_size=8,
        ep_size=8,
        use_ep=True,
    )
    experts = object.__new__(CutlassBatchedExpertsFp8)
    object.__setattr__(experts, "moe_config", config)
    object.__setattr__(experts, "_permute_scratch", None)

    with (
        patch.object(
            cutlass_moe,
            "moe_permute_unpermute_supported",
            return_value=True,
        ),
        patch.object(cutlass_moe, "MoEPermuteScratch") as scratch_cls,
    ):
        experts._get_permute_scratch()

    assert scratch_cls.call_args.kwargs["max_num_tokens"] == config.max_num_tokens


def make_minimax_w4a8_config(intermediate_size: int = 3072):
    config = make_dummy_moe_config(
        num_experts=128,
        num_local_experts=16,
        experts_per_token=4,
        hidden_dim=6144,
        intermediate_size=intermediate_size,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    config.moe_parallel_config = dataclasses.replace(
        config.moe_parallel_config,
        ep_size=8,
        use_ep=True,
    )
    config.swiglu_alpha = 1.702
    config.swiglu_beta = 1.0
    config.swiglu_limit = 7.0
    return config


def make_minimax_w4a8_deepep_ll_config():
    config = make_minimax_w4a8_config()
    config.moe_parallel_config = dataclasses.replace(
        config.moe_parallel_config,
        dp_size=8,
        ep_size=8,
        use_ep=True,
        all2all_backend="deepep_low_latency",
    )
    return config


@pytest.mark.parametrize(
    ("total_num_tokens", "expected"),
    [
        (0, "Kernel_128x16_1x1x1_Coop"),
        (8, "Kernel_128x16_1x1x1_Coop"),
        (128, "Kernel_256x16_1x1x1_Coop"),
        (512, "Kernel_256x16_1x1x1_Coop"),
        (1024, "Kernel_256x32_1x1x1_Coop"),
        (2048, "Kernel_256x64_1x1x1_Coop"),
        (4096, "Kernel_256x128_2x1x1_Coop"),
        (8192, "Kernel_128x256_2x1x1_Coop"),
    ],
)
def test_w4a8_batched_schedule_uses_expected_routed_rows(
    total_num_tokens: int,
    expected: str,
):
    assert (
        cutlass_moe._select_w4a8_batched_schedule(
            total_num_tokens=total_num_tokens,
            topk=4,
            global_num_experts=128,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("total_num_tokens", "expected"),
    [
        (0, 16),
        (512, 16),
        (1024, 64),
        (2048, 128),
        (4096, 256),
        (8192, 256),
    ],
)
def test_w4a8_compact_programs_scale_with_expected_m(
    total_num_tokens: int,
    expected: int,
):
    assert (
        cutlass_moe._select_w4a8_compact_programs(
            total_num_tokens=total_num_tokens,
            topk=4,
            global_num_experts=128,
        )
        == expected
    )


def test_w4a8_batched_schedule_uses_dp_group_token_total():
    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([1, 3, 7, 9], dtype=torch.int32)
    )
    context = SimpleNamespace(dp_metadata=dp_metadata)

    with (
        patch.object(cutlass_moe, "is_forward_context_available", return_value=True),
        patch.object(cutlass_moe, "get_forward_context", return_value=context),
    ):
        total_num_tokens = cutlass_moe._w4a8_batched_total_num_tokens(
            local_num_tokens=1,
            global_num_experts=128,
            num_local_experts=32,
        )

    assert total_num_tokens == 20


def test_w4a8_batched_schedule_fallback_uses_dispatcher_count():
    with patch.object(
        cutlass_moe,
        "is_forward_context_available",
        return_value=False,
    ):
        total_num_tokens = cutlass_moe._w4a8_batched_total_num_tokens(
            local_num_tokens=64,
            global_num_experts=128,
            num_local_experts=16,
        )

    assert total_num_tokens == 512


def make_minimax_w4a8_nixl_ep_config():
    config = make_minimax_w4a8_config()
    config.moe_parallel_config = dataclasses.replace(
        config.moe_parallel_config,
        dp_size=8,
        ep_size=8,
        use_ep=True,
        all2all_backend="nixl_ep",
    )
    return config


def get_w4a8_support(config):
    with patch.object(
        CutlassExpertsW4A8Fp8,
        "_supports_current_device",
        return_value=True,
    ):
        return CutlassExpertsW4A8Fp8.is_supported_config(
            CutlassExpertsW4A8Fp8,
            config,
            kInt4Static,
            kFp8DynamicTokenSym,
            mk.FusedMoEActivationFormat.Standard,
        )


def get_batched_w4a8_support(config):
    with patch.object(
        CutlassExpertsW4A8Fp8,
        "_supports_current_device",
        return_value=True,
    ):
        return CutlassBatchedExpertsW4A8Fp8.is_supported_config(
            CutlassBatchedExpertsW4A8Fp8,
            config,
            kInt4Static,
            kFp8DynamicTokenSym,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )


def test_cutlass_w4a8_supports_minimax_uninterleaved_swiglu_with_ep():
    config = make_minimax_w4a8_config()
    supported, reason = get_w4a8_support(config)

    assert supported
    assert reason is None
    with patch.object(
        CutlassExpertsW4A8Fp8,
        "_supports_current_device",
        return_value=True,
    ):
        backend, experts_cls = select_w4a8_moe_backend(config)
    assert backend is W4A8MoeBackend.CUTLASS
    assert experts_cls is CutlassExpertsW4A8Fp8


def test_cutlass_w4a8_selects_batched_experts_for_deepep_ll():
    config = make_minimax_w4a8_deepep_ll_config()
    supported, reason = get_batched_w4a8_support(config)

    assert supported
    assert reason is None
    with patch.object(
        CutlassExpertsW4A8Fp8,
        "_supports_current_device",
        return_value=True,
    ):
        backend, experts_cls = select_w4a8_moe_backend(config)
    assert backend is W4A8MoeBackend.CUTLASS
    assert experts_cls is CutlassBatchedExpertsW4A8Fp8


def test_cutlass_w4a8_rejects_batched_non_deepep_ll():
    supported, reason = get_batched_w4a8_support(make_minimax_w4a8_nixl_ep_config())

    assert not supported
    assert reason is not None
    assert "parallel config" in reason


def test_cutlass_w4a8_batched_workspace_and_finalize_contract():
    config = make_minimax_w4a8_deepep_ll_config()
    config.device = "cpu"
    quant_config = make_w4a8_moe_quant_config(
        w1_scale=torch.empty(16, 1, dtype=torch.float8_e4m3fn),
        w2_scale=torch.empty(16, 1, dtype=torch.float8_e4m3fn),
        g1_alphas=torch.empty(16, 6144, dtype=torch.float32),
        g2_alphas=torch.empty(16, 6144, dtype=torch.float32),
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = CutlassBatchedExpertsW4A8Fp8(
        moe_config=config,
        quant_config=quant_config,
        b_strides1=torch.empty(16, dtype=torch.int64),
        b_strides2=torch.empty(16, dtype=torch.int64),
        group_size=128,
        max_num_tokens=64,
        num_dispatchers=2,
    )

    assert isinstance(
        experts.finalize_weight_and_reduce_impl(), TopKWeightAndReduceDelegate
    )
    assert experts.expects_unquantized_inputs
    assert experts._get_permute_scratch() is None
    assert isinstance(
        CutlassExpertsW4A8Fp8(
            moe_config=config,
            quant_config=quant_config,
            b_strides1=torch.empty(16, dtype=torch.int64),
            b_strides2=torch.empty(16, dtype=torch.int64),
            group_size=128,
        ).finalize_weight_and_reduce_impl(),
        TopKWeightAndReduceNoOP,
    )
    assert experts.workspace_shapes(
        M=64,
        N=3072 * 2,
        K=6144,
        topk=4,
        global_num_experts=128,
        local_num_experts=16,
        expert_tokens_meta=None,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    ) == (
        (16, 128, 6144),
        (16, 128, 6144),
        (16, 128, 6144),
    )


def test_deepep_ll_receiver_defers_batched_w4a8_input_quant():
    pytest.importorskip("deep_ep")
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DeepEPLLPrepareAndFinalize,
    )

    prepare_finalize = object.__new__(DeepEPLLPrepareAndFinalize)
    prepare_finalize.use_fp8_dispatch = False
    expert_x = torch.randn((4, 8, 256), dtype=torch.bfloat16)
    expert_num_tokens = torch.tensor([0, 8, 1, 3], dtype=torch.int32)
    quant_config = make_w4a8_moe_quant_config(
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        g1_alphas=torch.empty(1),
        g2_alphas=torch.empty(1),
    )

    result = prepare_finalize._receiver(
        expert_x,
        expert_num_tokens,
        a1_scale=None,
        a1_dtype=torch.bfloat16,
        quant_config=quant_config,
        defer_input_quant=True,
    )

    received_x, received_scale, metadata, topk_ids, topk_weights = result
    assert received_x is expert_x
    assert received_scale is None
    assert metadata.expert_num_tokens is expert_num_tokens
    assert metadata.expert_num_tokens_cpu is None
    assert topk_ids is None
    assert topk_weights is None


MASKED_W4A8_ROUTING_CASES = [
    pytest.param([0, 0, 0, 0], id="empty"),
    pytest.param([8, 0, 0, 0], id="hot"),
    pytest.param([8, 8, 8, 8], id="uniform"),
    pytest.param([1, 8, 0, 3], id="skewed"),
    pytest.param([16, 17, 32, 65], id="persistent-loop-boundaries"),
    pytest.param([0, 64, 1, 17], id="persistent-loop-skewed"),
]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("counts", MASKED_W4A8_ROUTING_CASES)
def test_cutlass_w4a8_masked_per_token_quant_matches_full_flatten(counts):
    set_random_seed(7)
    num_experts = len(counts)
    padded_m = max(8, max(counts))
    hidden = 256
    src = torch.randn(
        (num_experts, padded_m, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    quant = torch.full_like(src, 1.0, dtype=torch.float8_e4m3fn)
    scales = torch.full(
        (num_experts, padded_m, 1),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    expert_num_tokens = torch.tensor(counts, dtype=torch.int32, device="cuda")

    cutlass_moe._masked_per_token_fp8_quant(
        src,
        quant,
        scales,
        expert_num_tokens,
    )

    for expert, count in enumerate(counts):
        if count:
            ref_quant, ref_scales = ops.scaled_fp8_quant(
                src[expert, :count],
                use_per_token_if_dynamic=True,
            )
            quantized = quant[expert, :count].float()
            ref_quantized = ref_quant.float()
            mismatch_rate = (quantized != ref_quantized).float().mean()
            assert mismatch_rate <= 2.0e-3
            torch.testing.assert_close(
                scales[expert, :count],
                ref_scales,
                rtol=2e-7,
                atol=1e-9,
            )
            torch.testing.assert_close(
                quantized * scales[expert, :count],
                ref_quantized * ref_scales,
                rtol=1e-1,
                atol=3e-1,
            )
        torch.testing.assert_close(
            quant[expert, count:].float(),
            torch.ones_like(quant[expert, count:].float()),
        )
        torch.testing.assert_close(
            scales[expert, count:],
            -torch.ones_like(scales[expert, count:]),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("counts", MASKED_W4A8_ROUTING_CASES)
def test_cutlass_w4a8_masked_minimax_activation_quant_matches_reference(counts):
    set_random_seed(11)
    num_experts = len(counts)
    padded_m = max(8, max(counts))
    hidden = 256
    src = torch.randn(
        (num_experts, padded_m, hidden * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    quant = torch.full(
        (num_experts, padded_m, hidden),
        1.0,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    scales = torch.full(
        (num_experts, padded_m, 1),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    expert_num_tokens = torch.tensor(counts, dtype=torch.int32, device="cuda")

    cutlass_moe._masked_swigluoai_quant(
        src,
        quant,
        scales,
        expert_num_tokens,
        alpha=1.702,
        beta=1.0,
        clamp_limit=7.0,
    )

    for expert, count in enumerate(counts):
        if count:
            ref_activation = torch.empty(
                (count, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            cutlass_moe._apply_w4a8_moe_activation(
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                ref_activation,
                src[expert, :count],
                gemm1_alpha=1.702,
                gemm1_beta=1.0,
                gemm1_clamp_limit=7.0,
            )
            ref_quant, ref_scales = ops.scaled_fp8_quant(
                ref_activation,
                use_per_token_if_dynamic=True,
            )
            torch.testing.assert_close(
                scales[expert, :count],
                ref_scales,
                rtol=5e-3,
                atol=1e-6,
            )
            torch.testing.assert_close(
                quant[expert, :count].float() * scales[expert, :count],
                ref_quant.float() * ref_scales,
                rtol=5e-3,
                atol=3e-2,
            )
        torch.testing.assert_close(
            quant[expert, count:].float(),
            torch.ones_like(quant[expert, count:].float()),
        )
        torch.testing.assert_close(
            scales[expert, count:],
            -torch.ones_like(scales[expert, count:]),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_cutlass_w4a8_masked_adapter_cuda_graph_replay():
    set_random_seed(13)
    num_experts = 4
    padded_m = 65
    hidden = 256
    counts = [0, 16, 17, 32]
    expert_num_tokens = torch.tensor(counts, dtype=torch.int32, device="cuda")
    a1 = torch.randn(
        (num_experts, padded_m, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    mm1 = torch.randn(
        (num_experts, padded_m, hidden * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    a1_quant = torch.empty_like(a1, dtype=torch.float8_e4m3fn)
    a1_scales = torch.empty(
        (num_experts, padded_m, 1),
        dtype=torch.float32,
        device="cuda",
    )
    a2_quant = torch.empty_like(a1_quant)
    a2_scales = torch.empty_like(a1_scales)

    cutlass_moe._masked_per_token_fp8_quant(
        a1,
        a1_quant,
        a1_scales,
        expert_num_tokens,
    )
    cutlass_moe._masked_swigluoai_quant(
        mm1,
        a2_quant,
        a2_scales,
        expert_num_tokens,
        alpha=1.702,
        beta=1.0,
        clamp_limit=7.0,
    )
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        cutlass_moe._masked_per_token_fp8_quant(
            a1,
            a1_quant,
            a1_scales,
            expert_num_tokens,
        )
        cutlass_moe._masked_swigluoai_quant(
            mm1,
            a2_quant,
            a2_scales,
            expert_num_tokens,
            alpha=1.702,
            beta=1.0,
            clamp_limit=7.0,
        )

    a1.copy_(torch.randn_like(a1))
    mm1.copy_(torch.randn_like(mm1))
    replay_counts = [65, 1, 32, 17]
    expert_num_tokens.copy_(
        torch.tensor(replay_counts, dtype=torch.int32, device="cuda")
    )
    graph.replay()

    eager_a1_quant = torch.empty_like(a1_quant)
    eager_a1_scales = torch.empty_like(a1_scales)
    eager_a2_quant = torch.empty_like(a2_quant)
    eager_a2_scales = torch.empty_like(a2_scales)
    cutlass_moe._masked_per_token_fp8_quant(
        a1,
        eager_a1_quant,
        eager_a1_scales,
        expert_num_tokens,
    )
    cutlass_moe._masked_swigluoai_quant(
        mm1,
        eager_a2_quant,
        eager_a2_scales,
        expert_num_tokens,
        alpha=1.702,
        beta=1.0,
        clamp_limit=7.0,
    )

    for expert, count in enumerate(replay_counts):
        if not count:
            continue
        torch.testing.assert_close(
            a1_quant[expert, :count].float(),
            eager_a1_quant[expert, :count].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            a1_scales[expert, :count],
            eager_a1_scales[expert, :count],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            a2_quant[expert, :count].float(),
            eager_a2_quant[expert, :count].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            a2_scales[expert, :count],
            eager_a2_scales[expert, :count],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("missing", ["swiglu_alpha", "swiglu_beta", "swiglu_limit"])
def test_cutlass_w4a8_rejects_missing_minimax_swiglu_params(missing):
    config = make_minimax_w4a8_config()
    setattr(config, missing, None)

    supported, reason = get_w4a8_support(config)

    assert not supported
    assert reason is not None
    assert missing in reason


def test_cutlass_w4a8_rejects_unaligned_intermediate_partition():
    supported, reason = get_w4a8_support(
        make_minimax_w4a8_config(intermediate_size=384)
    )

    assert not supported
    assert reason is not None
    assert "intermediate_size_per_partition" in reason


def test_w4a8_quant_config_preserves_minimax_swiglu_params():
    quant_config = make_w4a8_moe_quant_config(
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        g1_alphas=torch.empty(1),
        g2_alphas=torch.empty(1),
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )

    assert quant_config.gemm1_alpha == 1.702
    assert quant_config.gemm1_beta == 1.0
    assert quant_config.gemm1_clamp_limit == 7.0


def test_cutlass_w4a8_activation_forwards_minimax_swiglu_params(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_apply_moe_activation(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        cutlass_moe,
        "apply_moe_activation",
        fake_apply_moe_activation,
    )
    output = torch.empty((1, 1))
    input = torch.empty((1, 2))

    cutlass_moe._apply_w4a8_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )

    assert captured["args"] == (
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
    )
    assert captured["kwargs"] == {
        "clamp_limit": 7.0,
        "alpha": 1.702,
        "beta": 1.0,
    }


@pytest.mark.parametrize(
    "missing",
    ["gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"],
)
def test_cutlass_w4a8_activation_rejects_missing_minimax_swiglu_param(missing: str):
    params: dict[str, float | None] = {
        "gemm1_alpha": 1.702,
        "gemm1_beta": 1.0,
        "gemm1_clamp_limit": 7.0,
    }
    params[missing] = None

    with pytest.raises(ValueError, match=missing):
        cutlass_moe._apply_w4a8_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            torch.empty((1, 1)),
            torch.empty((1, 2)),
            **params,
        )


@dataclasses.dataclass
class MOETensors:
    a: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    ab_strides1: torch.Tensor
    c_strides1: torch.Tensor
    ab_strides2: torch.Tensor
    c_strides2: torch.Tensor

    @staticmethod
    def make_moe_tensors(
        m: int, k: int, n: int, e: int, dtype: torch.dtype
    ) -> "MOETensors":
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        ab_strides1 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        return MOETensors(
            a=a,
            w1=w1,
            w2=w2,
            ab_strides1=ab_strides1,
            c_strides1=c_strides1,
            ab_strides2=ab_strides2,
            c_strides2=c_strides2,
        )


@dataclasses.dataclass
class MOETensors8Bit(MOETensors):
    # quantized
    a_q: torch.Tensor | None = None  # a -> a_q
    w1_q: torch.Tensor | None = None  # w1 -> w1_q
    w2_q: torch.Tensor | None = None  # w2 -> w2_q
    a_scale: torch.Tensor | None = None
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    # dequantized
    a_d: torch.Tensor | None = None  # a -> a_q -> a_d
    w1_d: torch.Tensor | None = None  # w1 -> w1_q -> w1_d
    w2_d: torch.Tensor | None = None  # w2 -> w2_q -> w2_d

    @staticmethod
    def make_moe_tensors_8bit(
        m: int, k: int, n: int, e: int, per_act_token: bool, per_out_channel: bool
    ) -> "MOETensors8Bit":
        dtype = torch.half
        q_dtype = torch.float8_e4m3fn

        moe_tensors_fp16 = MOETensors.make_moe_tensors(m, k, n, e, dtype)

        # a -> a_q, w1 -> w1_q, w2 -> w2_q
        n_b_scales = 2 * n if per_out_channel else 1
        k_b_scales = k if per_out_channel else 1
        # Get the right scale for tests.
        a_q, a_scale = ops.scaled_fp8_quant(
            moe_tensors_fp16.a, None, use_per_token_if_dynamic=per_act_token
        )

        w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

        w1_scale = torch.empty((e, n_b_scales, 1), device="cuda", dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1), device="cuda", dtype=torch.float32)
        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w1[expert], use_per_token_if_dynamic=per_out_channel
            )
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                moe_tensors_fp16.w2[expert], use_per_token_if_dynamic=per_out_channel
            )

        # a_q -> a_d, w1_q -> w1_d, w2_q -> w2_d
        a_d = a_q.float().mul(a_scale).to(dtype)
        w1_d = torch.empty_like(moe_tensors_fp16.w1)
        w2_d = torch.empty_like(moe_tensors_fp16.w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()

        return MOETensors8Bit(
            a=moe_tensors_fp16.a,
            w1=moe_tensors_fp16.w1,
            w2=moe_tensors_fp16.w2,
            ab_strides1=moe_tensors_fp16.ab_strides1,
            c_strides1=moe_tensors_fp16.c_strides1,
            ab_strides2=moe_tensors_fp16.ab_strides2,
            c_strides2=moe_tensors_fp16.c_strides2,
            a_q=a_q,
            w1_q=w1_q,
            w2_q=w2_q,
            a_scale=a_scale,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a_d=a_d,
            w1_d=w1_d,
            w2_d=w2_d,
        )


def run_with_expert_maps(
    num_experts: int,
    num_local_experts: int,
    quant_config: FusedMoEQuantConfig,
    **cutlass_moe_kwargs,
):
    def slice_experts():
        slice_params = [
            "w1",
            "w2",
        ]
        full_tensors = {
            k: v
            for k, v in cutlass_moe_kwargs.items()
            if k in slice_params and k in cutlass_moe_kwargs
        }

        for i in range(0, num_experts, num_local_experts):
            s, e = i, i + num_local_experts

            # make expert map
            expert_map = [-1] * num_experts
            expert_map[s:e] = list(range(num_local_experts))
            expert_map = torch.tensor(expert_map, dtype=torch.int32, device="cuda")

            # update cutlass moe arg with expert_map
            cutlass_moe_kwargs["expert_map"] = expert_map
            # update cutlass moe arg tensors
            for k, t in full_tensors.items():
                cutlass_moe_kwargs[k] = t[s:e]

            new_quant_config = copy.deepcopy(quant_config)
            new_quant_config._w1.scale = quant_config.w1_scale[s:e]
            new_quant_config._w2.scale = quant_config.w2_scale[s:e]

            yield cutlass_moe_kwargs, new_quant_config

    out_tensor = torch.zeros_like(cutlass_moe_kwargs["hidden_states"])
    for kwargs, new_quant_config in slice_experts():
        w2 = kwargs["w2"]
        a = kwargs["hidden_states"]
        moe_config = make_dummy_moe_config(
            max_num_tokens=kwargs.get("hidden_states").shape[0],
            experts_per_token=kwargs.get("topk_ids").shape[1],
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_dim=w2.shape[1],
            intermediate_size=w2.shape[2],
            in_dtype=a.dtype,
        )
        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=new_quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            CutlassExpertsFp8(
                moe_config=moe_config,
                quant_config=new_quant_config,
            ),
        )
        out_tensor = out_tensor + kernel.apply(**kwargs)

    return out_tensor


def run_8_bit(
    moe_tensors: MOETensors8Bit,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
    num_local_experts: int | None = None,
) -> torch.Tensor:
    assert not any(
        [
            t is None
            for t in [
                moe_tensors.w1_q,
                moe_tensors.w2_q,
                moe_tensors.w1_scale,
                moe_tensors.w2_scale,
                moe_tensors.a_scale,
            ]
        ]
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=moe_tensors.w1_scale,
        w2_scale=moe_tensors.w2_scale,
        per_act_token_quant=per_act_token,
        per_out_ch_quant=per_out_ch,
        # Set to moe_tensors.a_scale iff static scales + per tensor.
        # This is not currently being tested.
        a1_scale=None,
    )

    num_experts = moe_tensors.w1.size(0)  # type: ignore[attr-defined]
    with_ep = num_local_experts is not None or num_local_experts == num_experts

    kwargs = {
        "hidden_states": moe_tensors.a,
        "w1": moe_tensors.w1_q,  # type: ignore[union-attr]
        "w2": moe_tensors.w2_q,  # type: ignore[union-attr]
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "global_num_experts": num_experts,
        "activation": MoEActivation.SILU,
        "expert_map": None,
        "apply_router_weight_on_input": False,
    }

    if not with_ep:
        moe_config = make_dummy_moe_config(
            max_num_tokens=moe_tensors.a.shape[0],
            experts_per_token=topk_ids.shape[1],
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_dim=moe_tensors.w2_q.shape[1],  # type: ignore[union-attr]
            intermediate_size=moe_tensors.w2_q.shape[2],  # type: ignore[union-attr]
            in_dtype=moe_tensors.a.dtype,
        )
        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            CutlassExpertsFp8(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
        return kernel.apply(**kwargs)

    assert num_local_experts is not None
    return run_with_expert_maps(
        num_experts,
        num_local_experts,  # type: ignore[arg-type]
        quant_config,
        **kwargs,
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    monkeypatch,
    workspace_init,
    ep_size: int | None = None,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.

        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        triton_output = fused_experts(
            mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids, quant_config=quant_config
        )

        if ep_size is not None:
            assert e % ep_size == 0, "Cannot distribute experts evenly"
            number_local_experts = e // ep_size
        else:
            number_local_experts = None

        cutlass_output = run_8_bit(
            mt, topk_weights, topk_ids, per_act_token, per_out_ch, number_local_experts
        )

        # Note 5.5 only needed for larger problem sizes, 5 works ok for
        # the rest.
        torch.testing.assert_close(
            triton_output, cutlass_output, atol=5.5e-2, rtol=1e-2
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_cuda_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        dtype = torch.half

        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)

        # Note that we are using the dequantized versions of the tensors.
        # Using a, w1 and w2 directly results in minor output differences.
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        triton_output = fused_experts(
            mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids, quant_config=quant_config
        )

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run_8_bit(
                mt, topk_weights, topk_ids, per_act_token, per_out_ch
            )

        torch.accelerator.synchronize()
        graph.replay()
        torch.accelerator.synchronize()

        torch.testing.assert_close(triton_output, cutlass_output, atol=9e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [64])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("per_act_token", [True])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8, 16])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_EP(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    monkeypatch,
    workspace_init,
):
    test_cutlass_moe_8_bit_no_graph(
        m,
        n,
        k,
        e,
        topk,
        per_act_token,
        per_out_channel,
        monkeypatch,
        workspace_init,
        ep_size,
    )


LARGE_MNK_FACTORS = [
    (1, 8192, 5120, 31),
    (32768, 1024, 1024, 16),
    (65536, 512, 1024, 16),
]


@pytest.mark.parametrize("m,n,k,topk", LARGE_MNK_FACTORS)
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("per_act_token", [False])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [8])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_cutlass_moe_8_bit_EP_large(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    monkeypatch,
    workspace_init,
):
    test_cutlass_moe_8_bit_no_graph(
        m,
        n,
        k,
        e,
        topk,
        per_act_token,
        per_out_channel,
        monkeypatch,
        workspace_init,
        ep_size,
    )


@pytest.mark.parametrize("m,n,k,topk", [(1, 8192, 5120, 31)])
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("per_act_token", [False])
@pytest.mark.parametrize("per_out_channel", [True])
@pytest.mark.parametrize("ep_size", [8])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()
    ),
    reason="Grouped gemm is not supported on this GPU type.",
)
def test_run_cutlass_moe_fp8(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_channel: bool,
    ep_size: int,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(
            m, k, n, e, per_act_token, per_out_channel
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)
        # we want to make sure there is at least one token that's generated in
        # this expert shard and at least one token that's NOT generated in this
        # expert shard
        topk_ids[0][0] = -1
        topk_ids[0][1] = 1

        workspace13_shape = (m * topk, max(2 * n, k))
        workspace2_shape = (m * topk, max(n, k))
        output_shape = (m, k)

        workspace13 = torch.empty(
            prod(workspace13_shape), device="cuda", dtype=mt.a.dtype
        )
        workspace2 = torch.empty(
            prod(workspace2_shape), device="cuda", dtype=mt.a.dtype
        )

        num_local_experts = e // ep_size
        start, end = 0, num_local_experts
        expert_map = [-1] * e
        expert_map[start:end] = list(range(num_local_experts))
        expert_map = torch.tensor(expert_map, dtype=torch.int32, device="cuda")

        ab_strides1 = torch.full((e,), k, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device="cuda", dtype=torch.int64)

        activation = MoEActivation.SILU
        a1q, a1q_scale = moe_kernel_quantize_input(
            mt.a, mt.a_scale, torch.float8_e4m3fn, per_act_token
        )
        global_num_experts = -1 if mt.w1_q is None else mt.w1_q.size(0)
        func = lambda output: run_cutlass_moe_fp8(
            output,
            a1q,
            mt.w1_q,
            mt.w2_q,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            mt.w1_scale,
            mt.w2_scale,
            a1q_scale,
            None,
            ab_strides1,
            ab_strides2,
            c_strides1,
            c_strides2,
            workspace13,
            workspace2,
            None,
            mt.a.dtype,
            per_act_token,
            per_out_channel,
            False,
            topk_weights,
            None,
        )

        workspace13.random_()
        output_random_workspace = torch.empty(
            output_shape, device="cuda", dtype=mt.a.dtype
        )
        func(output_random_workspace)

        workspace13.fill_(0)
        output_zero_workspace = torch.zeros(
            output_shape, device="cuda", dtype=mt.a.dtype
        )
        func(output_zero_workspace)

        torch.testing.assert_close(
            output_random_workspace, output_zero_workspace, atol=5e-3, rtol=1e-3
        )
