# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class FakePCPGroup:
    def __init__(self, gathered: torch.Tensor):
        self.gathered = gathered
        self.calls = 0
        self.input_dtypes: list[torch.dtype] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        self.calls += 1
        self.input_dtypes.append(tensor.dtype)
        return self.gathered


def make_runner() -> MoERunner:
    runner = MoERunner.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        dp_size=1,
        is_sequence_parallel=False,
        pcp_size=2,
        sp_size=1,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    return runner


def test_pcp_padding_mask_is_gathered_once_per_forward(monkeypatch):
    local_mask = torch.tensor([False, False, True])
    gathered_mask = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.uint8)
    context = SimpleNamespace(
        is_padding=local_mask,
        is_profile=False,
        additional_kwargs={},
        dp_metadata=None,
    )
    group = FakePCPGroup(gathered_mask)
    monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(moe_runner_module.envs, "VLLM_MOE_SKIP_PADDING", True)

    runner = make_runner()
    for _ in range(2):
        with runner._sequence_parallel_context():
            torch.testing.assert_close(
                context.is_padding,
                gathered_mask.bool(),
            )
        assert context.is_padding is local_mask

    assert group.calls == 1
    assert group.input_dtypes == [torch.uint8]
    assert (
        context.additional_kwargs[moe_runner_module._PCP_MOE_PADDING_MASK_KEY].dtype
        == torch.bool
    )


def test_pcp_profile_padding_mask_avoids_collective(monkeypatch):
    local_mask = torch.tensor([False, False, False])
    context = SimpleNamespace(
        is_padding=local_mask,
        is_profile=True,
        additional_kwargs={},
        dp_metadata=None,
    )
    group = FakePCPGroup(torch.empty(0, dtype=torch.bool))
    monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(moe_runner_module.envs, "VLLM_MOE_SKIP_PADDING", True)

    runner = make_runner()
    with runner._sequence_parallel_context():
        torch.testing.assert_close(context.is_padding, local_mask.repeat(2))

    assert context.is_padding is local_mask
    assert group.calls == 0


def test_pcp_dispatch_gathers_input_ids_with_routed_rows(monkeypatch):
    hidden_states = torch.tensor([[1.0], [2.0]])
    router_logits = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    input_ids = torch.tensor([10, 11])
    gathered = [
        torch.cat((hidden_states, hidden_states + 100)),
        torch.cat((router_logits, router_logits + 100)),
        torch.cat((input_ids, input_ids + 100)),
    ]

    class OrderedGatherGroup:
        def __init__(self):
            self.calls = 0

        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            assert dim == 0
            expected = (hidden_states, router_logits, input_ids)[self.calls]
            assert tensor is expected
            result = gathered[self.calls]
            self.calls += 1
            return result

    group = OrderedGatherGroup()
    context = SimpleNamespace(is_padding=None)
    monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: group)

    actual_hidden, actual_router, actual_input_ids = make_runner()._maybe_dispatch(
        hidden_states,
        router_logits,
        input_ids,
    )

    torch.testing.assert_close(actual_hidden, gathered[0])
    torch.testing.assert_close(actual_router, gathered[1])
    torch.testing.assert_close(actual_input_ids, gathered[2])
    assert actual_input_ids.shape[0] == actual_hidden.shape[0]
    assert group.calls == 3
