# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.hpc import (
    has_hpc,
    has_hpc_mxfp8_k32_moe,
    hpc_fuse_moe,
    hpc_fuse_moe_blockwise,
    hpc_fuse_moe_mxfp8_k32_bf16_candidate_out,
)

logger = init_logger(__name__)


class HPCExperts(mk.FusedMoEExpertsModular):
    """MoE implementation powered by [HPC](https://github.com/Tencent/hpc-ops).

    Only supported on NVIDIA Hopper GPUs (e.g. H20, H200), and currently limited to
    FP8 models such as Hy3-FP8, Qwen3-235B-A22B-FP8, etc.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype in (torch.float8_e4m3fn,), (
            "Only fp8 quantization is currently supported."
        )

        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype
        self.activation_clamp = quant_config.gemm1_clamp_limit

    @property
    def expects_unquantized_inputs(self) -> bool:
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and (p.is_device_capability(90) or p.is_device_capability_family(100))
            and has_hpc()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        scheme = (weight_key, activation_key)
        # The following are supported by HPCExperts:
        return scheme in [
            # fp8 static per-tensor on 9.0+
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
        ]

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # HPC fused MoE kernels process hidden_size in blocks of 128:
        # block-wise fp8 requires hidden_size % 128 == 0 (per-128 quant), and
        # the group GEMM tiles N by 128. Require 128-alignment to cover all
        # code paths.
        return hidden_dim % 128 == 0

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        # This refers to TP chunking; DP chunking is handled separately.
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        workspace1 = (M, K)
        workspace2 = (0,)
        output_shape = (M, K)
        # The workspace is determined by `aq`, since it comes after any
        # potential communication op and is involved in the expert computation.
        return (workspace1, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._supports_activation(activation), f"{activation=} not supported"
        assert self.quant_config.w1_scale is not None, (
            "w13_weight_scale must be provided"
        )
        assert self.quant_config.w2_scale is not None, (
            "w2_weight_scale must be provided"
        )

        if self.quant_config.is_block_quantized:
            hpc_fuse_moe_blockwise(
                x=hidden_states,
                x_scale=a1q_scale,
                gate_up_weight=w1,
                gate_up_weight_scale=self.quant_config.w1_scale,
                down_weight=w2,
                down_weight_scale=self.quant_config.w2_scale,
                topk_ids=topk_ids,
                topk_scale=topk_weights,
                rank_ep=self.ep_rank,
                num_expert_total=global_num_experts,
                output=output,
                activation_clamp=self.activation_clamp,
            )
        else:
            assert self.quant_config.a1_scale is not None, (
                "w13_input_scale must be provided"
            )
            assert self.quant_config.a2_scale is not None, (
                "w2_input_scale must be provided"
            )
            hpc_fuse_moe(
                x=hidden_states,
                gate_up_weight=w1,
                down_weight=w2,
                gate_up_scale=self.quant_config.g1_alphas,
                down_scale=self.quant_config.g2_alphas,
                act_and_mul_scale=self.quant_config.a2_gscale,
                topk_ids=topk_ids,
                topk_scale=topk_weights,
                rank_ep=self.ep_rank,
                num_expert_total=global_num_experts,
                output=output,
            )


class MiniMaxM3HPCExperts(mk.FusedMoEExpertsModular):
    """MiniMax-M3 MXFP8 K32 MoE body backed by hpc-ops.

    This backend is intentionally model-specific. It targets the MiniMax-M3
    routed expert shape: hidden size 6144, intermediate size 192, 128 experts,
    top-4 routing, MXFP8 E4M3 values with UE8M0 K32 scales, and SwiGLU-OAI over
    packed [gate | up] output.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype == "mxfp8", (
            "MiniMaxM3HPCExperts only supports MXFP8 weights."
        )
        assert quant_config.block_shape == [1, 32], (
            "MiniMaxM3HPCExperts requires MXFP8 K32 scales."
        )

        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability(90) and has_hpc_mxfp8_k32_moe()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim == 6144

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason

        if moe_config.num_experts != 128 or moe_config.num_local_experts != 128:
            return False, "MiniMax-M3 HPC backend requires 128 local experts"
        if moe_config.experts_per_token != 4:
            return False, "MiniMax-M3 HPC backend requires top_k=4"
        if moe_config.intermediate_size_per_partition != 192:
            return False, (
                "MiniMax-M3 HPC backend requires intermediate_size_per_partition=192"
            )
        if moe_config.in_dtype != torch.bfloat16:
            return False, "MiniMax-M3 HPC backend requires BF16 activations"
        return True, None

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.ep_size == 1

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        return True

    def supports_output_alias(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        routed_rows = M * topk
        routing_bytes = (
            (2 * routed_rows + 2 * local_num_experts + 1) * 4 + 15
        ) & ~15
        grouped_bytes = routed_rows * (K + K // 32)
        gate_output_bytes = routed_rows * N * 2
        workspace13_elements = (
            routing_bytes + grouped_bytes + gate_output_bytes + 1
        ) // 2
        return (workspace13_elements,), (routed_rows, K), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._supports_activation(activation), f"{activation=} not supported"
        assert not apply_router_weight_on_input, (
            "MiniMaxM3HPCExperts applies router weights in hpc.reduce."
        )
        assert expert_map is None, "MiniMaxM3HPCExperts does not support expert_map"
        assert a1q_scale is None and a2_scale is None
        assert topk_ids.size(1) == 4, "MiniMax-M3 expects top_k=4"
        assert w1.size(0) == w2.size(0) == 128, "MiniMax-M3 expects 128 experts"
        assert w1.size(1) == 384 and w1.size(2) == 6144
        assert w2.size(1) == 6144 and w2.size(2) == 192
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        assert hidden_states.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert workspace13 is not None and workspace2 is not None

        num_tokens = hidden_states.size(0)
        topk = topk_ids.size(1)
        routed_rows = num_tokens * topk
        hidden_dim = hidden_states.size(1)
        gate_n = w1.size(1)

        scratch = workspace13.view(torch.uint8).flatten()
        offset = 0

        def take(
            num_bytes: int,
            dtype: torch.dtype,
            shape: tuple[int, ...],
        ) -> torch.Tensor:
            nonlocal offset
            result = scratch[offset : offset + num_bytes].view(dtype).view(shape)
            offset += num_bytes
            return result

        row_indices = take(routed_rows * 4, torch.int32, (routed_rows,))
        topk_pos = take(routed_rows * 4, torch.int32, (num_tokens, topk))
        seqlens = take(self.num_experts * 4, torch.int32, (self.num_experts,))
        cu_seqlens = take(
            (self.num_experts + 1) * 4,
            torch.int32,
            (self.num_experts + 1,),
        )

        offset = (offset + 15) & ~15
        grouped_start = offset
        grouped_hidden = take(
            routed_rows * hidden_dim,
            torch.float8_e4m3fn,
            (routed_rows, hidden_dim),
        )
        grouped_hidden_scale = take(
            routed_rows * (hidden_dim // 32),
            torch.uint8,
            (routed_rows, hidden_dim // 32),
        )
        gate_output = take(
            routed_rows * gate_n * 2,
            torch.bfloat16,
            (routed_rows, gate_n),
        )
        assert offset <= scratch.numel()

        # Gate consumes grouped input before activation runs on the same stream,
        # so activation output can reuse the dead grouped-input prefix.
        activated_output_bytes = routed_rows * (gate_n // 2)
        activated_output = scratch[
            grouped_start : grouped_start + activated_output_bytes
        ].view(torch.float8_e4m3fn).view(routed_rows, gate_n // 2)
        activated_scale = scratch[
            grouped_start
            + activated_output_bytes : grouped_start
            + activated_output_bytes
            + routed_rows * (gate_n // 64)
        ].view(torch.uint8).view(routed_rows, gate_n // 64)

        clamp = self.quant_config.gemm1_clamp_limit
        alpha = self.quant_config.gemm1_alpha
        beta = self.quant_config.gemm1_beta
        hpc_fuse_moe_mxfp8_k32_bf16_candidate_out(
            hidden=hidden_states,
            gate_up_weight=w1,
            gate_up_weight_scale=self.quant_config.w1_scale,
            down_weight=w2,
            down_weight_scale=self.quant_config.w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            row_indices=row_indices,
            topk_pos=topk_pos,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            grouped_hidden=grouped_hidden,
            grouped_hidden_scale=grouped_hidden_scale,
            gate_output=gate_output,
            activated_output=activated_output,
            activated_scale=activated_scale,
            down_output=workspace2,
            activation_clamp=7.0 if clamp is None else float(clamp),
            alpha=1.702 if alpha is None else float(alpha),
            beta=1.0 if beta is None else float(beta),
        )
