# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    combine_topk_swa_indices_with_positions,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.pcp_metadata import (
    compact_pcp_sparse_indices,
    compact_pcp_sparse_prefill_queries,
    overlay_pcp_restored_swa_kv_workspace,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.cp_utils import guard_dsv4_pcp_prefill_runtime_metadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _kv_cache_storage_block_size(
    k_cache: torch.Tensor | None,
    fallback: int,
) -> int:
    if k_cache is not None and k_cache.dim() >= 3:
        return int(k_cache.shape[1])
    return int(fallback)


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend
    supports_pcp = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        is_pcp_prefill = swa_metadata.pcp_allgather_restore_idx is not None
        guard_dsv4_pcp_prefill_runtime_metadata(
            pcp_allgather_restore_idx=swa_metadata.pcp_allgather_restore_idx,
            num_prefill_tokens=num_prefill_tokens,
            runtime_metadata=swa_metadata.pcp_prefill_metadata,
        )

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                compressed_cache_block_size = _kv_cache_storage_block_size(
                    compressed_k_cache,
                    attn_metadata.block_size // self.compress_ratio,
                )
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=compressed_cache_block_size,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            swa_cache_block_size = _kv_cache_storage_block_size(
                swa_k_cache,
                swa_metadata.block_size,
            )
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_cache_block_size,
                offset=chunk_N,
                force_triton=is_pcp_prefill,
            )
            if is_pcp_prefill:
                assert swa_metadata.pcp_prefill_metadata is not None
                pcp_metadata = swa_metadata.pcp_prefill_metadata
                assert pcp_metadata.restored_swa_kv is not None
                assert pcp_metadata.restored_swa_positions is not None
                assert pcp_metadata.restored_swa_valid_mask is not None
                overlay_pcp_restored_swa_kv_workspace(
                    out=kv[:chunk_size],
                    restored_kv=pcp_metadata.restored_swa_kv,
                    restored_positions=pcp_metadata.restored_swa_positions,
                    restored_valid_mask=pcp_metadata.restored_swa_valid_mask,
                    views=pcp_metadata.views,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    chunk_n=chunk_N,
                    chunk_m=chunk_M,
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            chunk_query_start_loc = query_start_loc[
                num_decodes + chunk_start : num_decodes + chunk_end + 1
            ]
            if is_pcp_prefill:
                combined_indices, combined_lens = (
                    combine_topk_swa_indices_with_positions(
                        topk_indices[query_start:query_end],
                        positions[query_start:query_end],
                        chunk_query_start_loc,
                        seq_lens[chunk_start:chunk_end],
                        gather_lens[chunk_start:chunk_end],
                        self.window_size,
                        self.compress_ratio,
                        top_k,
                        chunk_M,
                        chunk_N,
                    )
                )
                combined_indices, combined_lens = compact_pcp_sparse_indices(
                    combined_indices, combined_lens
                )
                chunk_output = output[query_start:query_end]
                chunk_output.zero_()
                pcp_q, pcp_indices, pcp_lens, valid_rows = (
                    compact_pcp_sparse_prefill_queries(
                        q[query_start:query_end],
                        combined_indices,
                        combined_lens,
                    )
                )
                if valid_rows.numel() > 0:
                    pcp_out = output.new_empty((pcp_q.shape[0], *output.shape[1:]))
                    flash_mla_sparse_fwd(
                        q=pcp_q,
                        kv=kv.view(-1, 1, q.shape[-1]),
                        indices=pcp_indices.unsqueeze(1),
                        sm_scale=self.scale,
                        attn_sink=self.attn_sink,
                        topk_length=pcp_lens,
                        out=pcp_out,
                    )
                    chunk_output.index_copy_(0, valid_rows, pcp_out)
                continue

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                chunk_query_start_loc,
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
            )
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
