# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.models.deepseek_v4.online_c128 import (
    DeepseekOnlineC128State,
    assert_online_c128_supported,
    ensure_online_c128_compressed_kv,
    online_c128_compress_enabled,
    online_c128_uses_mtp,
    register_online_c128_state,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    # [num_tokens] token -> persistent request-state slot index (padded reqs
    # map to -1). Only populated when C128 online compression is enabled; used
    # to address the independent DeepseekOnlineC128State rows.
    token_to_req_state_indices: torch.Tensor | None = None
    # Device per-request inputs for the graph-safe online C128 decode / verify
    # kernel (fixed-address; safe to capture in a FULL decode cudagraph). Only
    # populated when C128 online is enabled.
    query_start_loc: torch.Tensor | None = None  # [num_reqs + 1] int32
    req_state_indices: torch.Tensor | None = None  # [num_reqs] int32 (-1 pad)
    # Host-side per-step inputs for building the C128 online segment plan. All
    # CPU (no device sync). Only populated when C128 online is enabled.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: "np.ndarray | None" = None
    req_state_indices_cpu: "np.ndarray | None" = None
    num_draft_tokens_per_req_cpu: "np.ndarray | None" = None


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # C128 online compression addresses an independent per-request running
        # state by stable request-state slot (CommonAttentionMetadata
        # .req_state_indices). When enabled, expand that per-request mapping to a
        # per-token mapping (padded reqs / tokens carry -1). Kept in a persistent
        # buffer for CUDA-graph address stability, mirroring token_to_req_indices.
        self._online_c128 = online_c128_compress_enabled()
        if self._online_c128:
            self.token_to_req_state_indices = torch.full(
                (self.vllm_config.scheduler_config.max_num_batched_tokens,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)

        token_to_req_state_indices = None
        query_start_loc_cpu = None
        seq_lens_cpu = None
        req_state_indices_cpu = None
        num_draft_tokens_per_req_cpu = None
        device_query_start_loc = None
        device_req_state_indices = None
        if self._online_c128:
            req_state_indices = common_attn_metadata.req_state_indices
            assert req_state_indices is not None, (
                "C128 online compression requires req_state_indices in "
                "CommonAttentionMetadata (v2 GPU runner)."
            )
            # Device per-request inputs for the graph-safe decode/verify kernel
            # (fixed-address; safe to capture in a FULL decode cudagraph).
            device_query_start_loc = common_attn_metadata.query_start_loc
            device_req_state_indices = req_state_indices[:num_reqs].to(torch.int32)
            # Map each token to its persistent request-state slot via the
            # batch-local token->req index built above.
            token_state = device_req_state_indices[
                token_to_req_indices.to(torch.long)
            ]
            token_to_req_state_indices = self.token_to_req_state_indices[
                : x.shape[0]
            ]
            token_to_req_state_indices.copy_(token_state, non_blocking=True)

            # Host-side inputs for the segment planner (no device sync).
            query_start_loc_cpu = query_start_loc_cpu_t = (
                common_attn_metadata.query_start_loc_cpu
            )
            req_state_indices_cpu = common_attn_metadata.req_state_indices_cpu
            assert req_state_indices_cpu is not None, (
                "C128 online compression requires req_state_indices_cpu."
            )
            num_draft_tokens_per_req_cpu = (
                common_attn_metadata.num_draft_tokens_per_req_cpu
            )
            # seq_lens_cpu = num_computed + query_len. seq_lens_cpu_upper_bound is
            # precise for prefill rows (the only rows that emit/store via the
            # segment plan); decode rows use single-token recurrence and ignore
            # the plan's per-segment math beyond num_rows==1.
            ub = common_attn_metadata.seq_lens_cpu_upper_bound
            if ub is not None:
                seq_lens_cpu = ub[:num_reqs].cpu().numpy()
            else:
                qlen = (
                    query_start_loc_cpu_t[1:] - query_start_loc_cpu_t[:-1]
                ).numpy()
                seq_lens_cpu = qlen  # fallback: seq_len == query_len

        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            token_to_req_state_indices=token_to_req_state_indices,
            query_start_loc=device_query_start_loc,
            req_state_indices=device_req_state_indices,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            req_state_indices_cpu=req_state_indices_cpu,
            num_draft_tokens_per_req_cpu=num_draft_tokens_per_req_cpu,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
        online_c128: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

        # C128 online compression maintains its own dense per-request running
        # state and NEVER reads this paged window, so we do not need to retain
        # the full `coff * compress_ratio` sliding window — shrinking it to a
        # single page reclaims the per-request window reservation from the KV
        # pool while keeping the layer's slot-mapping/metadata pipeline intact.
        if online_c128:
            self.sliding_window = self.block_size
        else:
            self.sliding_window = coff * compress_ratio

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # FlashMLA's UE8M0 paged layout needs 576B alignment; the FlashInfer
        # full-cache path shares state pages with contiguous KV pages, so
        # padding would break page matching.
        is_flashmla = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if is_flashmla else None,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        # The real C128 layout (compress_ratio==128, head_dim==512) takes the
        # online running-state path; C4 / indexer layers stay on the legacy
        # compressor. Computed here so the state cache can shrink its sliding
        # window when the online path will own the running state.
        self._use_online_c128 = (
            online_c128_compress_enabled()
            and compress_ratio == 128
            and self.head_dim == 512
        )

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
            online_c128=self._use_online_c128,
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        # C128 online (running-state) compression: allocate the independent
        # per-request state at model-init time (NOT a KV-cache-group layer, so
        # not carved from profiled KV memory). Gated + guarded fail-closed.
        # Only the real C128 layout (compress_ratio==128, head_dim==512) takes
        # the online path; C4 / indexer layers stay on the legacy compressor.
        self.online_c128_state: DeepseekOnlineC128State | None = None
        self.online_c128_uses_mtp = False
        if self._use_online_c128:
            assert_online_c128_supported(
                vllm_config,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            )
            self.online_c128_state = DeepseekOnlineC128State(
                vllm_config=vllm_config,
                head_dim=self.head_dim,
                layer_index=extract_layer_index(prefix) if prefix else 0,
                device=self.device,
            )
            register_online_c128_state(self.online_c128_state)
            self.online_c128_uses_mtp = online_c128_uses_mtp(vllm_config)
            # Fixed-address compressed-KV scratch for the FULL decode cudagraph
            # path (shared across layers; allocated before any capture).
            ensure_online_c128_compressed_kv(
                max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                head_dim=self.head_dim,
                device=self.device,
            )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # FlashInfer V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # legacy FlashMLA path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # C128 online (running-state) path: maintain a per-request online-softmax
        # accumulator and store only at chunk boundaries, instead of paging the
        # raw 128-token window. Bypasses save_partial_states / state_cache.
        if self.online_c128_state is not None:
            self._online_forward(
                kv=kv,
                score=score,
                positions=positions,
                state_metadata=state_metadata,
                num_actual=num_actual,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
            return

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA always uses cutedsl, for both the legacy UE8M0
            # layout and the FlashInfer full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )

    def _online_forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        positions: torch.Tensor,
        state_metadata: "CompressorMetadata",
        num_actual: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        store_full_kv: bool,
        store_full_fp8: bool,
        fp8_scale: torch.Tensor | None,
    ) -> None:
        """C128 online path: graph-safe FULL decode or planned eager/PW."""
        from vllm.forward_context import get_forward_context
        from vllm.config.compilation import CUDAGraphMode

        online_state = self.online_c128_state
        assert online_state is not None
        run_state = online_state.state

        forward_context = get_forward_context()
        cg_mode = forward_context.cudagraph_runtime_mode
        if cg_mode == CUDAGraphMode.FULL:
            batch_desc = forward_context.batch_descriptor
            candidate_chain = self.online_c128_uses_mtp and bool(
                getattr(batch_desc, "online_c128_candidate_chain", False)
            )
            self._online_forward_graph_safe(
                kv=kv,
                score=score,
                positions=positions,
                state_metadata=state_metadata,
                num_actual=num_actual,
                run_state=run_state,
                online_state=online_state,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
                candidate_chain=candidate_chain,
            )
            return

        self._online_forward_planned(
            kv=kv,
            score=score,
            positions=positions,
            state_metadata=state_metadata,
            num_actual=num_actual,
            run_state=run_state,
            online_state=online_state,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            store_full_kv=store_full_kv,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
        )

    def _online_forward_graph_safe(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        positions: torch.Tensor,
        state_metadata: "CompressorMetadata",
        num_actual: int,
        run_state: torch.Tensor,
        online_state: "DeepseekOnlineC128State",
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        store_full_kv: bool,
        store_full_fp8: bool,
        fp8_scale: torch.Tensor | None,
        candidate_chain: bool,
    ) -> None:
        """FULL-decode-cudagraph path: fixed-address, on-device, plan-free."""
        from vllm.models.deepseek_v4.nvidia.ops.online_c128_cutedsl import (
            online_c128_decode,
        )
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            store_compressed_kv_cutedsl,
        )
        from vllm.models.deepseek_v4.online_c128 import online_c128_compressed_kv

        query_start_loc = state_metadata.query_start_loc
        req_state_indices = state_metadata.req_state_indices
        assert query_start_loc is not None and req_state_indices is not None, (
            "C128 online FULL-graph path requires device query_start_loc / "
            "req_state_indices in the compressor metadata."
        )

        # Stable scratch for FULL graph replay; zero non-boundary rows before store.
        compressed_kv = online_c128_compressed_kv(num_actual)
        compressed_kv.zero_()

        online_c128_decode(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            query_start_loc=query_start_loc,
            req_state_indices=req_state_indices,
            run_state=run_state,
            compressed_kv=compressed_kv,
            max_num_reqs=online_state.max_num_reqs,
            compress_ratio=self.compress_ratio,
            candidate_chain=candidate_chain,
        )

        store_compressed_kv_cutedsl(
            compressed_kv=compressed_kv,
            positions=positions,
            slot_mapping=state_metadata.slot_mapping,
            block_size=state_metadata.block_size,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            k_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping,
            kv_cache_block_size=kv_cache.shape[1],
            kv_block_stride=kv_cache.stride(0),
            head_size=self.head_dim,
            state_width=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            fp8_max=448.0,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            store_full_kv=store_full_kv,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
        )

    def _online_forward_planned(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        positions: torch.Tensor,
        state_metadata: "CompressorMetadata",
        num_actual: int,
        run_state: torch.Tensor,
        online_state: "DeepseekOnlineC128State",
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        store_full_kv: bool,
        store_full_fp8: bool,
        fp8_scale: torch.Tensor | None,
    ) -> None:
        """Eager / PIECEWISE C128 path using host-built segment plans."""
        from vllm.models.deepseek_v4.nvidia.ops.online_c128_cutedsl import (
            online_c128_merge,
        )
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            store_compressed_kv_cutedsl,
        )
        from vllm.models.deepseek_v4.online_c128 import (
            online_c128_verify_active,
            plan_online_c128_segments,
            plan_online_c128_verify,
        )

        query_start_loc_cpu = state_metadata.query_start_loc_cpu
        seq_lens_cpu = state_metadata.seq_lens_cpu
        req_state_indices_cpu = state_metadata.req_state_indices_cpu
        assert (
            query_start_loc_cpu is not None
            and seq_lens_cpu is not None
            and req_state_indices_cpu is not None
        ), "C128 online forward requires host-side segment-plan metadata."

        compressed_kv = torch.empty(
            (num_actual, self.head_dim),
            dtype=torch.float32,
            device=kv.device,
        )

        verify_mode = self.online_c128_uses_mtp and online_c128_verify_active()
        if verify_mode:
            num_draft_tokens_per_req_cpu = (
                state_metadata.num_draft_tokens_per_req_cpu
            )
            if num_draft_tokens_per_req_cpu is None:
                raise ValueError(
                    "C128 online MTP verify requires per-request draft-token "
                    "counts in compressor metadata."
                )
            num_draft_tokens_per_req_cpu = num_draft_tokens_per_req_cpu[
                : len(req_state_indices_cpu)
            ]
            verify_req_mask = num_draft_tokens_per_req_cpu > 0
            # Verify rows use candidate banks; mixed non-verify rows update bank0.
            if np.any(verify_req_mask):
                verify_segments_by_step = plan_online_c128_verify(
                    query_start_loc_cpu=query_start_loc_cpu.numpy(),
                    seq_lens_cpu=seq_lens_cpu,
                    req_state_indices_cpu=req_state_indices_cpu,
                    max_num_reqs=online_state.max_num_reqs,
                    device=kv.device,
                    compress_ratio=self.compress_ratio,
                    req_mask=verify_req_mask,
                    max_query_len=online_state.num_banks - 1,
                )
                for step_segments in verify_segments_by_step:
                    online_c128_merge(
                        kv=kv,
                        score=score,
                        ape=self.ape,
                        positions=positions,
                        run_state=run_state,
                        segments=step_segments,
                        compressed_kv=compressed_kv,
                        compress_ratio=self.compress_ratio,
                    )

            normal_req_mask = ~verify_req_mask
            if np.any(normal_req_mask):
                plan = plan_online_c128_segments(
                    query_start_loc_cpu=query_start_loc_cpu.numpy(),
                    seq_lens_cpu=seq_lens_cpu,
                    req_state_indices_cpu=req_state_indices_cpu,
                    max_num_reqs=online_state.max_num_reqs,
                    device=kv.device,
                    bank_id=0,
                    compress_ratio=self.compress_ratio,
                    req_mask=normal_req_mask,
                )
                online_c128_merge(
                    kv=kv,
                    score=score,
                    ape=self.ape,
                    positions=positions,
                    run_state=run_state,
                    segments=plan.emit_segments,
                    compressed_kv=compressed_kv,
                    compress_ratio=self.compress_ratio,
                )
                online_c128_merge(
                    kv=kv,
                    score=score,
                    ape=self.ape,
                    positions=positions,
                    run_state=run_state,
                    segments=plan.update_segments,
                    compressed_kv=compressed_kv,
                    compress_ratio=self.compress_ratio,
                )
                if plan.reset_rows.numel() > 0:
                    rows = plan.reset_rows.to(torch.long)
                    run_state[rows, : self.head_dim] = float("-inf")
                    run_state[rows, self.head_dim :] = 0.0
        else:
            plan = plan_online_c128_segments(
                query_start_loc_cpu=query_start_loc_cpu.numpy(),
                seq_lens_cpu=seq_lens_cpu,
                req_state_indices_cpu=req_state_indices_cpu,
                max_num_reqs=online_state.max_num_reqs,
                device=kv.device,
                bank_id=0,
                compress_ratio=self.compress_ratio,
            )
            # Phase 1 (emit): read committed carry read-only, write compressed_kv.
            online_c128_merge(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                run_state=run_state,
                segments=plan.emit_segments,
                compressed_kv=compressed_kv,
                compress_ratio=self.compress_ratio,
            )
            # Phase 2 (update): write the trailing partial carry back to bank0.
            online_c128_merge(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                run_state=run_state,
                segments=plan.update_segments,
                compressed_kv=compressed_kv,
                compress_ratio=self.compress_ratio,
            )
            # Reset bank rows whose step ended exactly on a 128 boundary.
            if plan.reset_rows.numel() > 0:
                rows = plan.reset_rows.to(torch.long)
                run_state[rows, : self.head_dim] = float("-inf")
                run_state[rows, self.head_dim :] = 0.0

        # Store boundary tokens (RMSNorm + RoPE + UE8M0 / FlashInfer full KV).
        store_compressed_kv_cutedsl(
            compressed_kv=compressed_kv,
            positions=positions,
            slot_mapping=state_metadata.slot_mapping,
            block_size=state_metadata.block_size,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            k_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping,
            kv_cache_block_size=kv_cache.shape[1],
            kv_block_stride=kv_cache.stride(0),
            head_size=self.head_dim,
            state_width=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            fp8_max=448.0,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            store_full_kv=store_full_kv,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
        )
