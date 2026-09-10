import pytest

from vllm import envs
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe


@pytest.mark.parametrize("env_value", [None, "auto"])
def test_humming_gemm_auto_uses_grouped_masked_for_batched_activation(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    monkeypatch.setattr(envs, "VLLM_HUMMING_MOE_GEMM_TYPE", env_value)

    assert (
        fused_humming_moe.get_humming_moe_gemm_type(
            mk.FusedMoEActivationFormat.BatchedExperts
        )
        == "grouped_masked"
    )


@pytest.mark.parametrize("env_value", [None, "auto"])
def test_humming_gemm_auto_does_not_filter_standard_activation(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    monkeypatch.setattr(envs, "VLLM_HUMMING_MOE_GEMM_TYPE", env_value)

    assert (
        fused_humming_moe.get_humming_moe_gemm_type(
            mk.FusedMoEActivationFormat.Standard
        )
        is None
    )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("indexed", "indexed"),
        ("grouped", "grouped_contiguous"),
        ("grouped_contiguous", "grouped_contiguous"),
        ("grouped_masked", "grouped_masked"),
        ("batched_grouped", "grouped_masked"),
        ("unexpected", "indexed"),
    ],
)
def test_humming_gemm_explicit_env_values(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: str
) -> None:
    monkeypatch.setattr(envs, "VLLM_HUMMING_MOE_GEMM_TYPE", env_value)

    assert fused_humming_moe.get_humming_moe_gemm_type() == expected
