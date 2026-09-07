# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingExpertsBase,
    HummingGroupedExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import w4a8
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a8_fp8 as ct_w4a8,
)
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    _group_shape,
    get_humming_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8DynamicTokenSym,
    kInt4Static,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("auto", w4a8.W4A8MoeBackend.CUTLASS),
        ("cutlass", w4a8.W4A8MoeBackend.CUTLASS),
        ("humming", w4a8.W4A8MoeBackend.HUMMING),
    ],
)
def test_w4a8_backend_mapping(name, expected):
    assert w4a8.map_w4a8_backend(name) is expected


def test_w4a8_backend_mapping_rejects_unknown_backend():
    with pytest.raises(NotImplementedError, match="not supported for W4A8"):
        w4a8.map_w4a8_backend("triton")


def test_humming_w4a8_selection_uses_standard_activation_format():
    class StandardExperts:
        @staticmethod
        def is_supported_config(
            cls, config, weight_key, activation_key, activation_format
        ):
            del cls, config, weight_key, activation_key
            return activation_format == mk.FusedMoEActivationFormat.Standard, None

    config = make_dummy_moe_config()
    config.moe_backend = "humming"

    with patch.object(
        w4a8,
        "backend_to_kernel_cls",
        return_value=[StandardExperts],
    ):
        backend, experts_cls = w4a8.select_w4a8_moe_backend(config)

    assert backend is w4a8.W4A8MoeBackend.HUMMING
    assert experts_cls is StandardExperts


@pytest.mark.parametrize(
    "all2all_backend",
    ["deepep_low_latency", "flashinfer_nvlink_one_sided"],
)
def test_humming_w4a8_rejects_unsupported_communication_format(all2all_backend):
    config = make_dummy_moe_config()
    config.moe_backend = "humming"
    config.moe_parallel_config = SimpleNamespace(
        use_batched_activation_format=all2all_backend == "deepep_low_latency",
        use_fi_nvl_one_sided_kernels=(
            all2all_backend == "flashinfer_nvlink_one_sided"
        ),
    )

    with pytest.raises(NotImplementedError, match="communication format"):
        w4a8.select_w4a8_moe_backend(config)


def test_humming_w4a8_quant_config_uses_n_by_k_group_shape():
    class DType:
        num_bits = 8

        def __str__(self):
            return "float8_e4m3"

    layer = SimpleNamespace(
        input_schemas={"w13": SimpleNamespace(a_dtype=DType())},
        weight_schemas={
            "w13": SimpleNamespace(
                b_dtype="uint4",
                weight_scale_group_size=128,
                weight_scale_group_size_n=32,
            )
        },
    )

    quant_config = get_humming_moe_quant_config(
        layer,
        gemm1_alpha=0.0,
        gemm1_beta=0.0,
        gemm1_clamp_limit=7.0,
    )

    assert quant_config._w1.shape == GroupShape(row=32, col=128)
    assert quant_config.gemm1_alpha == 0.0
    assert quant_config.gemm1_beta == 0.0
    assert quant_config.gemm1_clamp_limit == 7.0


@pytest.mark.parametrize(
    ("group_size", "group_size_n", "expected"),
    [
        (128, 0, GroupShape(row=1, col=128)),
        (0, 0, GroupShape(row=-1, col=1)),
        (128, 64, GroupShape(row=64, col=128)),
    ],
)
def test_humming_group_shape_uses_n_by_k_convention(
    group_size,
    group_size_n,
    expected,
):
    assert _group_shape(group_size, group_size_n) == expected


def test_humming_w4a8_rejects_missing_activation_params():
    config = make_dummy_moe_config(
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )
    config.swiglu_alpha = None
    config.swiglu_beta = 1.0
    config.swiglu_limit = 7.0

    with patch.object(
        HummingGroupedExperts,
        "_supports_current_device",
        return_value=True,
    ):
        supported, reason = HummingGroupedExperts.is_supported_config(
            HummingGroupedExperts,
            config,
            kInt4Static,
            kFp8DynamicTokenSym,
            mk.FusedMoEActivationFormat.Standard,
        )

    assert not supported
    assert reason is not None
    assert "swiglu_alpha" in reason


def test_humming_w4a8_activation_preserves_explicit_zero_params():
    captured = {}
    experts = SimpleNamespace(
        quant_config=SimpleNamespace(
            gemm1_alpha=0.0,
            gemm1_beta=0.0,
            gemm1_clamp_limit=7.0,
        ),
        activation=lambda **kwargs: captured.update(kwargs),
    )
    input_tensor = torch.empty((1, 2))
    output_tensor = torch.empty((1, 1))

    HummingExpertsBase.apply_activation(
        experts,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output_tensor,
        input_tensor,
    )

    assert captured == {
        "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        "input": input_tensor,
        "output": output_tensor,
        "clamp_limit": 7.0,
        "alpha": 0.0,
        "beta": 0.0,
    }


def test_w4a8_humming_schema_configs_preserve_checkpoint_contract():
    weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )

    weight_config = w4a8._quant_args_to_humming_weight_config(
        weight_quant,
        format="pack-quantized",
    )
    input_config = w4a8._quant_args_to_humming_input_config(
        input_quant,
        format="float-quantized",
    )

    assert weight_config == {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "type": "int",
        "num_bits": 4,
        "strategy": "group",
        "symmetric": True,
        "group_size": 128,
    }
    assert input_config == {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "type": "float",
        "num_bits": 8,
        "strategy": "token",
        "dynamic": True,
        "group_size": 0,
        "symmetric": True,
    }


def test_compressed_tensors_w4a8_requires_int_weight_and_float_activation():
    valid_weight = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
    )
    valid_input = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )

    assert CompressedTensorsConfig._is_fp8_w4a8(valid_weight, valid_input)
    assert not CompressedTensorsConfig._is_fp8_w4a8(
        valid_weight,
        valid_input.model_copy(update={"type": QuantizationType.INT}),
    )
    assert not CompressedTensorsConfig._is_fp8_w4a8(
        valid_weight.model_copy(update={"type": QuantizationType.FLOAT}),
        valid_input,
    )


def test_humming_w4a8_weight_creation_skips_cutlass_alignment():
    method = object.__new__(ct_w4a8.CompressedTensorsW4A8Fp8MoEMethod)
    method.w4a8_backend = w4a8.W4A8MoeBackend.HUMMING
    method.packed_factor = 8
    method.group_size = 128
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=1,
        hidden_size=6144,
        intermediate_size_per_partition=384,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight_packed.shape == (1, 768, 768)
    assert layer.w2_weight_packed.shape == (1, 6144, 48)


def test_cutlass_w4a8_weight_creation_keeps_alignment_guard():
    method = object.__new__(ct_w4a8.CompressedTensorsW4A8Fp8MoEMethod)
    method.w4a8_backend = w4a8.W4A8MoeBackend.CUTLASS
    method.packed_factor = 8
    method.group_size = 128

    with pytest.raises(AssertionError, match="intermediate_size_per_partition"):
        method.create_weights(
            layer=torch.nn.Module(),
            num_experts=1,
            hidden_size=6144,
            intermediate_size_per_partition=384,
            params_dtype=torch.bfloat16,
        )


def test_humming_grouped_scratch_covers_standard_dispatch_group():
    config = make_dummy_moe_config()
    config.moe_parallel_config = SimpleNamespace(
        use_ep=True,
        ep_size=8,
        dp_size=2,
    )
    experts = object.__new__(HummingGroupedExperts)
    experts.moe_config = config
    experts._permute_scratch = None

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe."
            "moe_permute_unpermute_supported",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe."
            "MoEPermuteScratch"
        ) as scratch_cls,
    ):
        experts._get_permute_scratch()

    assert scratch_cls.call_args.kwargs["max_num_tokens"] == (
        config.max_num_tokens * config.moe_parallel_config.ep_size
    )


def test_compressed_tensors_humming_uses_converted_canonical_weights(monkeypatch):
    method = object.__new__(ct_w4a8.CompressedTensorsW4A8Fp8MoEMethod)
    method.w4a8_backend = w4a8.W4A8MoeBackend.HUMMING
    method.weight_quant = object()
    method.input_quant = object()
    method.moe = SimpleNamespace(
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    )
    method.experts_cls = object()
    method.moe_quant_config = None

    routing_tables = object()
    layer = SimpleNamespace(
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=128,
        expert_map=None,
        apply_router_weight_on_input=False,
        _expert_routing_tables=lambda: routing_tables,
    )
    quant_config = object()
    output = torch.empty(1)
    kernel = SimpleNamespace(
        apply=Mock(return_value=output),
        is_monolithic=False,
    )
    convert = Mock()
    cutlass_convert = Mock(side_effect=AssertionError("unexpected CUTLASS conversion"))

    monkeypatch.setattr(
        ct_w4a8,
        "convert_to_humming_w4a8_moe_kernel_format",
        convert,
    )
    monkeypatch.setattr(
        ct_w4a8,
        "convert_to_w4a8_moe_kernel_format",
        cutlass_convert,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.humming_utils."
        "get_humming_moe_quant_config",
        lambda *args, **kwargs: quant_config,
    )
    monkeypatch.setattr(
        ct_w4a8,
        "make_humming_w4a8_moe_kernel",
        lambda **kwargs: kernel,
    )

    method.process_weights_after_loading(layer)

    convert.assert_called_once_with(
        layer=layer,
        weight_quant=method.weight_quant,
        input_quant=method.input_quant,
    )
    cutlass_convert.assert_not_called()
    assert method.moe_quant_config is quant_config

    result = method.apply(
        layer=layer,
        x=torch.empty((1, 1)),
        topk_weights=torch.empty((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int64),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is output
    assert kernel.apply.call_args.kwargs["w1"] is layer.w13_weight
    assert kernel.apply.call_args.kwargs["w2"] is layer.w2_weight
