# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import fx
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

MINIMAX_M3_PIPELINED_AG_GATE_SMALL_RANGE_END = 5024
_MINIMAX_M3_ARCHITECTURES = {
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
}


def minimax_m3_pipelined_ag_gate_compile_enabled(config: VllmConfig) -> bool:
    model_config = config.model_config
    compilation_config = config.compilation_config
    return bool(
        envs.VLLM_MINIMAX_M3_PIPELINED_AG_GATE
        and model_config is not None
        and model_config.architecture in _MINIMAX_M3_ARCHITECTURES
        and model_config.dtype == torch.bfloat16
        and config.parallel_config.tensor_parallel_size == 8
        and compilation_config.mode == CompilationMode.VLLM_COMPILE
        and compilation_config.backend == "inductor"
    )


def _minimax_m3_small_ag_gate(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    gate_weight: torch.Tensor,
    world_size: int,
    vllm_group_name: str,
    c10d_group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del c10d_group_name
    packed = torch.cat((hidden, residual), dim=-1)
    gathered = torch.ops.vllm.all_gather.default(
        packed,
        0,
        world_size,
        vllm_group_name,
    )
    gathered_hidden, gathered_residual = gathered.split(hidden.shape[-1], dim=-1)
    gathered_hidden = gathered_hidden.contiguous()
    gathered_residual = gathered_residual.contiguous()
    router_logits = torch.ops.vllm.fp32_router_gemm_dispatch.default(
        gathered_hidden,
        gate_weight,
        False,
    )
    return gathered_hidden, gathered_residual, router_logits


class MiniMaxM3PipelinedAGGateExpansionPass(VllmInductorPass):
    """Expose the small-shape fallback to Inductor for cross-layer fusion."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(self.pass_name)
        op = torch.ops.vllm.minimax_m3_pipelined_ag_gate.default
        register_graph_pattern(
            CallFunctionVarArgs([op]),
            pass_dict=self.patterns,
        )(self._expand)
        self.matched_count = 0

    def _expand(self, match: Match, *args, **kwargs) -> None:
        assert len(match.nodes) == 1
        node = match.nodes[0]
        assert not node.kwargs
        match.replace_by_example(
            _minimax_m3_small_ag_gate,
            node.args,
            run_functional_passes=True,
        )

    def is_applicable_for_range(self, compile_range) -> bool:
        if compile_range.end <= MINIMAX_M3_PIPELINED_AG_GATE_SMALL_RANGE_END:
            return True
        if compile_range.start > MINIMAX_M3_PIPELINED_AG_GATE_SMALL_RANGE_END:
            return False
        raise RuntimeError(
            "MiniMax-M3 pipelined AG+gate compile range crosses the "
            f"{MINIMAX_M3_PIPELINED_AG_GATE_SMALL_RANGE_END} token boundary: "
            f"{compile_range}"
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "Expanded %d MiniMax-M3 small AG+gate nodes",
            self.matched_count,
        )
