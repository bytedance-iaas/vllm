# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


DSV4_PCP_PREFILL_UNSUPPORTED_ERROR = (
    "DeepSeek-V4 prefill PCP requires dsv4 PCP runtime metadata path; "
    "legacy sparse backend remap path is unsupported."
)


@dataclass(frozen=True)
class PCPInterleaveRequestView:
    req_idx: int
    global_seq_len: int
    local_token_count: int
    local_query_start: int
    local_query_end: int
    global_positions: torch.Tensor
    local_positions: torch.Tensor
    restore_idx: torch.Tensor
    global_slot_mapping: torch.Tensor
    local_kv_base: int
    local_kv_len: int


def guard_dsv4_pcp_prefill_runtime_metadata(
    *,
    pcp_allgather_restore_idx: torch.Tensor | None,
    num_prefill_tokens: int,
    runtime_metadata: object | None,
) -> None:
    """Fail closed before DeepSeek V4 sparse backends use legacy PCP remap."""
    if (
        pcp_allgather_restore_idx is not None
        and num_prefill_tokens > 0
        and runtime_metadata is None
    ):
        raise NotImplementedError(DSV4_PCP_PREFILL_UNSUPPORTED_ERROR)


def _cpu_long_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", dtype=torch.long)
    return torch.as_tensor(data, dtype=torch.long, device="cpu")


def build_pcp_interleave_request_views(
    *,
    original_token_counts: np.ndarray | torch.Tensor,
    local_token_counts: np.ndarray | torch.Tensor,
    local_positions: np.ndarray | torch.Tensor,
    restore_idx: torch.Tensor,
    pcp_world_size: int,
    global_slot_mapping: torch.Tensor,
    local_valid_mask: np.ndarray | torch.Tensor | None = None,
) -> list[PCPInterleaveRequestView]:
    """Build per-request PCP views from the current local-rank token layout.

    The current V1 PCP manager uses a dual-chunk head/tail layout. The view keeps
    the request-local selected global positions, compact local positions, and the
    per-request restore slice together so model-specific metadata builders do not
    need to rediscover those relationships from raw buffers.
    """
    original_counts = _cpu_long_tensor(original_token_counts)
    local_counts = _cpu_long_tensor(local_token_counts)
    positions = _cpu_long_tensor(local_positions)
    valid_mask = (
        _cpu_long_tensor(local_valid_mask).to(dtype=torch.bool)
        if local_valid_mask is not None
        else None
    )
    slots = global_slot_mapping.detach().to(device="cpu", dtype=torch.long)
    restore = restore_idx.detach().to(device="cpu", dtype=torch.long)

    num_reqs = int(original_counts.numel())
    original_starts = torch.empty(num_reqs, dtype=torch.long)
    local_starts = torch.empty(num_reqs, dtype=torch.long)
    padded_starts = torch.empty(num_reqs, dtype=torch.long)
    if num_reqs == 0:
        return []
    original_starts[0] = 0
    local_starts[0] = 0
    padded_starts[0] = 0
    if num_reqs > 1:
        original_starts[1:] = torch.cumsum(original_counts, dim=0)[:-1]
        local_starts[1:] = torch.cumsum(local_counts, dim=0)[:-1]
        padded_starts[1:] = torch.cumsum(local_counts * pcp_world_size, dim=0)[:-1]

    views: list[PCPInterleaveRequestView] = []
    compact_start = 0
    for req_idx in range(num_reqs):
        global_seq_len = int(original_counts[req_idx].item())
        local_count = int(local_counts[req_idx].item())
        local_start = int(local_starts[req_idx].item())
        local_end = local_start + local_count

        req_positions = positions[local_start:local_end]
        if valid_mask is None:
            req_valid_mask = req_positions < global_seq_len
        else:
            req_valid_mask = valid_mask[local_start:local_end]

        valid_positions = req_positions[req_valid_mask]
        valid_count = int(valid_positions.numel())
        original_start = int(original_starts[req_idx].item())
        slot_indices = original_start + valid_positions
        request_slots = slots[slot_indices] if valid_count > 0 else slots[:0]
        compact_end = compact_start + valid_count

        padded_start = int(padded_starts[req_idx].item())
        padded_end = padded_start + local_count * pcp_world_size
        views.append(
            PCPInterleaveRequestView(
                req_idx=req_idx,
                global_seq_len=global_seq_len,
                local_token_count=valid_count,
                local_query_start=compact_start,
                local_query_end=compact_end,
                global_positions=valid_positions,
                local_positions=torch.arange(valid_count, dtype=torch.long),
                restore_idx=restore[padded_start:padded_end],
                global_slot_mapping=request_slots,
                local_kv_base=compact_start,
                local_kv_len=valid_count,
            )
        )
        compact_start = compact_end
    return views


class PCPManager:
    """Build per-rank token metadata for Prefill Context Parallelism.

    PCP splits long prefill requests across ranks using a head/tail chunk
    assignment. Decode requests are not split; they are replicated on every PCP
    rank so mixed decode+prefill batches keep decode semantics unchanged.

    The manager owns the small CPU/GPU buffers needed by the model runner to:

    * replace scheduled token counts with the local PCP-rank token counts;
    * build the local token positions for this PCP rank;
    * mask padding after PCP all-gather;
    * restore all-gathered hidden/KV tensors back to original request order.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        assert pcp_world_size > 1
        assert 0 <= pcp_rank < pcp_world_size
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank

        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_padded_slot_mapping = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.pcp_padded_slot_mappings: dict[int, torch.Tensor] = {
            0: self.pcp_padded_slot_mapping,
        }
        self.pcp_padded_positions = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.pcp_padded_query_start_loc = CpuGpuBuffer(
            (max_num_reqs + 1,),
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask = CpuGpuBuffer(
            (max_buffer_num_tokens,),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_unpad_mask_cpu_tensor = self.pcp_unpad_mask.cpu
        self.pcp_unpad_mask_gpu_tensor = self.pcp_unpad_mask.gpu
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()
        self.pcp_local_unpad_mask = CpuGpuBuffer(
            (max_buffer_num_tokens,),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_local_unpad_mask_cpu_tensor = self.pcp_local_unpad_mask.cpu
        self.pcp_local_unpad_mask_gpu_tensor = self.pcp_local_unpad_mask.gpu
        self.pcp_local_unpad_mask_cpu = self.pcp_local_unpad_mask_cpu_tensor.numpy()
        self.pcp_local_token_indices = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_local_token_indices_cpu_tensor = self.pcp_local_token_indices.cpu
        self.pcp_local_token_indices_gpu_tensor = self.pcp_local_token_indices.gpu
        self.pcp_local_token_indices_cpu = (
            self.pcp_local_token_indices_cpu_tensor.numpy()
        )
        self.pcp_request_views: list[PCPInterleaveRequestView] = []

    def get_pcp_padded_slot_mapping(self, kv_cache_gid: int) -> torch.Tensor:
        slot_mapping = self.pcp_padded_slot_mappings.get(kv_cache_gid)
        if slot_mapping is None:
            slot_mapping = torch.empty_like(self.pcp_padded_slot_mapping)
            self.pcp_padded_slot_mappings[kv_cache_gid] = slot_mapping
        return slot_mapping

    @staticmethod
    def _get_cumsum_and_arange(
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative token counts and per-request aranges.

        Example: [2, 5, 3] -> ([2, 7, 10],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]).
        """
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(
            cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
        )
        arange = arange_np[:total_num_tokens] - cumsums_offsets
        return cu_num_tokens, arange

    def update_tokens_for_pcp(
        self,
        tokens: np.ndarray,
        arange_np: np.ndarray,
        num_reqs: int,
        reorder_batch_threshold: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update token counts and positions for this PCP rank.

        Args:
            tokens: Scheduled token counts per request before PCP splitting.
            arange_np: Reusable arange buffer large enough for the padded batch.
            num_reqs: Number of active requests in the prefix of ``tokens``.
            reorder_batch_threshold: Decode/prefill split threshold used by MLA
                metadata builders. Requests with scheduled tokens less than or
                equal to this threshold are treated as decode requests.

        Returns:
            ``(pcp_tokens, pcp_positions)`` for this rank.
        """
        assert reorder_batch_threshold is not None, (
            "PCP depends on reorder batch to split decode and prefill requests."
        )
        tokens = tokens[:num_reqs]
        num_decode_reqs = int(np.sum(tokens <= reorder_batch_threshold))
        num_decode_tokens = int(np.sum(tokens[:num_decode_reqs]))

        num_padded_scheduled_tokens = np.ceil(
            tokens / (2 * self.pcp_world_size)
        ).astype(np.int32) * (2 * self.pcp_world_size)
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            tokens[:num_decode_reqs] * self.pcp_world_size
        )

        self.num_pcp_pads_cpu[:num_reqs] = num_padded_scheduled_tokens - tokens

        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )
        self.pcp_unpad_mask_cpu[: pcp_padded_arange.shape[0]] = (
            pcp_padded_arange < np.repeat(tokens, num_padded_scheduled_tokens)
        )
        self.pcp_unpad_mask.copy_to_gpu(pcp_padded_arange.shape[0])

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size
        self.pcp_padded_query_start_loc.cpu[0] = 0
        pcp_padded_query_start = np.cumsum(
            pcp_tokens[:num_reqs] * self.pcp_world_size, dtype=np.int32
        )
        self.pcp_padded_query_start_loc.cpu[1 : num_reqs + 1].copy_(
            torch.from_numpy(pcp_padded_query_start)
        )
        self.pcp_padded_query_start_loc.copy_to_gpu(num_reqs + 1)
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes, arange_np)
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(
            positions_start_loc: int | np.ndarray, rank: int
        ) -> np.ndarray:
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = (
                positions_start_loc
                + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            )
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
                head_start_loc, pcp_chunk_sizes
            )
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_rank)
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                tokens[:num_decode_reqs], arange_np
            )[1]
        original_cu_tokens = np.cumsum(tokens, dtype=np.int64)
        original_start_loc = np.roll(original_cu_tokens, 1)
        original_start_loc[0] = 0
        num_local_tokens = positions.shape[0]
        local_valid_mask = positions < np.repeat(tokens, pcp_tokens)
        self.pcp_local_unpad_mask_cpu[:num_local_tokens] = local_valid_mask
        self.pcp_local_unpad_mask.copy_to_gpu(num_local_tokens)
        self.pcp_local_token_indices_cpu[:num_local_tokens] = positions.astype(
            np.int64
        ) + np.repeat(original_start_loc, pcp_tokens)
        self.pcp_local_token_indices_cpu[:num_local_tokens][~local_valid_mask] = 0
        self.pcp_local_token_indices.copy_to_gpu(num_local_tokens)

        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions = np.concatenate(
            [
                get_current_rank_positions(padded_pos_start_loc, rank)
                for rank in range(self.pcp_world_size)
            ]
        )
        restore_idx = all_positions.argsort()
        self.pcp_allgather_restore_idx.np[: restore_idx.shape[0]] = restore_idx
        self.pcp_allgather_restore_idx.copy_to_gpu(restore_idx.shape[0])
        identity_slot_mapping = torch.arange(
            int(tokens.sum(dtype=np.int64)),
            dtype=torch.long,
            device="cpu",
        )
        self.pcp_request_views = build_pcp_interleave_request_views(
            original_token_counts=tokens,
            local_token_counts=pcp_tokens[:num_reqs],
            local_positions=positions[:num_local_tokens],
            restore_idx=torch.from_numpy(restore_idx),
            pcp_world_size=self.pcp_world_size,
            global_slot_mapping=identity_slot_mapping,
            local_valid_mask=local_valid_mask,
        )

        return pcp_tokens[:num_reqs], positions


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            get_attn_backend = getattr(layer, "get_attn_backend", None)
            if pcp_size > 1 and get_attn_backend is not None:
                backend = get_attn_backend()
                assert backend.supports_pcp(), (
                    "PCP requires attention backend support, "
                    f"but {backend.get_name()} does not support PCP."
                )
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )


def get_kv_cache_shard_count() -> int:
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing.
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size


def get_dcp_dummy_context_len(
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    has_kv_cache_config: bool,
    create_mixed_batch: bool,
    is_graph_capturing: bool,
    uniform_decode: bool,
) -> int:
    if (
        dcp_world_size <= 1
        or not has_kv_cache_config
        or not (create_mixed_batch or (is_graph_capturing and uniform_decode))
    ):
        return 0
    return dcp_world_size * cp_kv_cache_interleave_size


def prepare_dcp_dummy_context_metadata(
    *,
    input_batch: Any,
    kv_cache_config: Any,
    query_pos: Any,
    positions: torch.Tensor,
    query_start_loc: Any,
    num_reqs: int,
    num_tokens_unpadded: int,
    dcp_dummy_context_len: int,
) -> None:
    """Populate valid fake KV metadata for DCP CUDA graph warmup/capture."""
    if dcp_dummy_context_len == 0:
        return

    # DCP graph warmup may exercise context attention, so block-table entries
    # must point at allocated KV blocks.
    assert kv_cache_config is not None
    max_valid_block_id = kv_cache_config.num_blocks - 1
    assert max_valid_block_id > 0
    for blk_table in input_batch.block_table.block_tables:
        max_row_blocks = (
            blk_table.max_num_blocks_per_req // blk_table.blocks_per_kv_block
        )
        block_ids = [
            (block_idx % max_valid_block_id) + 1 for block_idx in range(max_row_blocks)
        ]
        for req_idx in range(num_reqs):
            blk_table.add_row(block_ids, req_idx)
        blk_table.commit_block_table(num_reqs)

    query_pos.copy_to_gpu(num_tokens_unpadded)
    positions[:num_tokens_unpadded] = (
        query_pos.gpu[:num_tokens_unpadded] + dcp_dummy_context_len
    )
    input_batch.block_table.compute_slot_mapping(
        num_reqs,
        query_start_loc.gpu[: num_reqs + 1],
        positions[:num_tokens_unpadded],
    )


def should_skip_dcp_context_attention(context_kv_lens_cpu: torch.Tensor) -> bool:
    """Whether DCP context attention can be skipped for this batch.

    Must be computed from rank-invariant inputs only (the global context
    lengths, NOT this rank's local share from get_dcp_local_seq_lens): the
    non-skip path in _forward_with_dcp issues DCP collectives (query
    all-gather + LSE combine), so every DCP rank must take the same branch.
    A rank can hold zero local context tokens while other ranks still hold
    context for the same batch.
    """
    return int(context_kv_lens_cpu.max().item()) == 0


def split_dcp_context_queries(
    query_start_loc: torch.Tensor,
    seq_lens_cpu_upper_bound: torch.Tensor | None,
    max_query_len: int,
    num_actual_tokens: int,
) -> tuple[int, int, int, int]:
    """Split reordered DCP context queries into decode and extend regions."""
    num_reqs = query_start_loc.shape[0] - 1
    if max_query_len <= 1:
        return num_reqs, 0, num_actual_tokens, 0
    if seq_lens_cpu_upper_bound is None:
        return 0, num_reqs, 0, num_actual_tokens

    common_attn_metadata = cast(
        CommonAttentionMetadata,
        SimpleNamespace(
            max_query_len=max_query_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            query_start_loc_cpu=query_start_loc,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            is_prefilling=None,
        ),
    )
    (
        num_decodes,
        num_extends,
        _num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        _num_prefill_tokens,
    ) = split_decodes_prefills_and_extends(common_attn_metadata)
    return num_decodes, num_extends, num_decode_tokens, num_extend_tokens


def should_split_fa2_dcp_context_attention(
    fa_version: int | None,
    max_query_len: int,
    num_reqs: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
) -> bool:
    num_prefills = num_reqs - num_decode_reqs
    # TODO: Remove this FA2-only DCP compatibility path once FA4 supports
    # the Qwen3.5 head_size=256 shape on Blackwell and can be used here.
    # FA2 paged-varlen context attention can fail for DCP mixed batches when
    # decode rows, context-bearing extend rows, and zero-context pure prefill
    # rows are submitted together.
    return (
        fa_version == 2
        and max_query_len > 1
        and num_prefills > 0
        and (num_decode_reqs > 0 or num_context_prefill_reqs < num_prefills)
    )


def run_split_fa2_dcp_context_attention(
    flash_attn_varlen_func: Any,
    query_across_dcp: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dcp_context_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dcp_context_kv_lens: torch.Tensor,
    max_dcp_context_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window_size: list[int] | None,
    block_table: torch.Tensor,
    softcap: float,
    fa_version: int,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    max_num_splits: int,
    num_heads: int,
    dcp_world_size: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
    num_decode_tokens: int,
    num_context_prefill_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dcp_context_out.zero_()
    context_lse = torch.full(
        (num_heads * dcp_world_size, query_across_dcp.shape[0]),
        -torch.inf,
        dtype=torch.float32,
        device=query_across_dcp.device,
    )

    if num_decode_tokens > 0:
        _, decode_context_lse = flash_attn_varlen_func(
            q=query_across_dcp[:num_decode_tokens],
            k=key_cache,
            v=value_cache,
            out=dcp_context_out[:num_decode_tokens],
            cu_seqlens_q=cu_seqlens_q[: num_decode_reqs + 1],
            max_seqlen_q=1,
            seqused_k=dcp_context_kv_lens[:num_decode_reqs],
            max_seqlen_k=max_dcp_context_kv_len,
            softmax_scale=softmax_scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table[:num_decode_reqs],
            softcap=softcap,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale[:num_decode_reqs] if q_descale is not None else None,
            k_descale=k_descale[:num_decode_reqs] if k_descale is not None else None,
            v_descale=v_descale[:num_decode_reqs] if v_descale is not None else None,
            num_splits=max_num_splits,
        )
        context_lse[:, :num_decode_tokens] = decode_context_lse

    if num_context_prefill_tokens > 0:
        prefill_start = num_decode_tokens
        prefill_end = prefill_start + num_context_prefill_tokens
        prefill_query_start_loc = (
            cu_seqlens_q[
                num_decode_reqs : num_decode_reqs + num_context_prefill_reqs + 1
            ]
            - num_decode_tokens
        )
        prefill_req_slice = slice(
            num_decode_reqs, num_decode_reqs + num_context_prefill_reqs
        )
        _, prefill_context_lse = flash_attn_varlen_func(
            q=query_across_dcp[prefill_start:prefill_end],
            k=key_cache,
            v=value_cache,
            out=dcp_context_out[prefill_start:prefill_end],
            cu_seqlens_q=prefill_query_start_loc,
            max_seqlen_q=max_seqlen_q,
            seqused_k=dcp_context_kv_lens[prefill_req_slice],
            max_seqlen_k=max_dcp_context_kv_len,
            softmax_scale=softmax_scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table[prefill_req_slice],
            softcap=softcap,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale[prefill_req_slice] if q_descale is not None else None,
            k_descale=k_descale[prefill_req_slice] if k_descale is not None else None,
            v_descale=v_descale[prefill_req_slice] if v_descale is not None else None,
            num_splits=max_num_splits,
        )
        context_lse[:, prefill_start:prefill_end] = prefill_context_lse

    return dcp_context_out, context_lse
