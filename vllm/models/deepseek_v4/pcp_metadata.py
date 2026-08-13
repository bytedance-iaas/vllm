# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 prefill PCP metadata helpers."""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.v1.worker.cp_utils import PCPInterleaveRequestView


@dataclass(frozen=True)
class DeepseekV4PcpQueryPlan:
    """Indexer query rows after PCP all-gather, restore, and padding removal."""

    restored_valid_mask: torch.Tensor
    compact_restored_indices: torch.Tensor
    local_compact_indices: torch.Tensor
    local_valid_mask: torch.Tensor
    num_local_tokens: int


@dataclass(frozen=True)
class DeepseekV4PcpSwaSegment:
    query_start: int
    query_end: int
    kv_start: int
    kv_end: int
    sparse_rows: int
    q_rows: torch.Tensor
    shifted_indices: torch.Tensor
    topk_lens: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DeepseekV4PcpSparseRows:
    q_rows: torch.Tensor
    sparse_rows: int
    valid_query_mask: torch.Tensor
    rows_min: int
    rows_max: int


@dataclass
class DeepseekV4PcpPrefillMetadata:
    cp_size: int
    cp_rank: int
    strategy: Literal["dual_chunk"]
    views: list[PCPInterleaveRequestView]
    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    local_swa_indices: torch.Tensor | None
    local_swa_valid_lens: torch.Tensor | None
    local_c4_indices: torch.Tensor | None
    local_c128_indices: torch.Tensor | None
    global_slot_mapping: torch.Tensor
    compressor_write_locs_global: torch.Tensor | None
    restore_idx: torch.Tensor | None
    debug_global_positions: torch.Tensor | None
    sparse_rows: DeepseekV4PcpSparseRows | None = None
    swa_segments: list[DeepseekV4PcpSwaSegment] | None = None
    restored_swa_kv: torch.Tensor | None = None
    restored_swa_positions: torch.Tensor | None = None
    restored_swa_valid_mask: torch.Tensor | None = None


def build_pcp_full_slot_mapping(
    *,
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    valid_mask: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Map restored PCP positions to slots owned by this CP rank.

    DeepSeek V4 cache groups do not all use the scheduler block size: SWA uses
    64-token pages, C4/C128 use compressed storage pages, and compressor state
    groups use 4/8-token pages. CP ownership is nevertheless defined in the
    original token coordinate system by ``cp_kv_cache_interleave_size``. Map to
    that rank-local coordinate first, then address this cache group's pages.
    """
    return _build_pcp_owned_slot_mapping(
        positions=positions,
        req_indices=req_indices,
        block_table=block_table,
        block_size=block_size,
        valid_mask=valid_mask,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        compress_ratio=1,
    )


def build_pcp_compressed_slot_mapping(
    *,
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    valid_mask: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Map restored PCP boundaries to compressed slots owned by this CP rank."""
    return _build_pcp_owned_slot_mapping(
        positions=positions,
        req_indices=req_indices,
        block_table=block_table,
        block_size=block_size,
        valid_mask=valid_mask,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        compress_ratio=compress_ratio,
    )


def _build_pcp_owned_slot_mapping(
    *,
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    valid_mask: torch.Tensor | None,
    cp_world_size: int,
    cp_rank: int,
    cp_kv_cache_interleave_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Translate global token positions into one rank's local cache pages."""
    if positions.numel() == 0:
        return torch.empty_like(positions, dtype=torch.long)

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if cp_world_size <= 0 or not 0 <= cp_rank < cp_world_size:
        raise ValueError(
            f"invalid CP topology: world_size={cp_world_size}, rank={cp_rank}"
        )
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError(
            "cp_kv_cache_interleave_size must be positive, got "
            f"{cp_kv_cache_interleave_size}"
        )
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if (
        cp_world_size > 1
        and compress_ratio > 1
        and cp_kv_cache_interleave_size % compress_ratio != 0
    ):
        raise ValueError(
            "compressed PCP cache ownership requires CP interleave size to be "
            f"divisible by compress_ratio, got interleave="
            f"{cp_kv_cache_interleave_size}, ratio={compress_ratio}"
        )

    req_indices = req_indices[: positions.numel()].to(
        device=positions.device, dtype=torch.long
    )
    positions_long = positions.to(dtype=torch.long)
    slot_mapping = torch.full_like(positions_long, -1)

    valid = (positions_long >= 0) & (req_indices >= 0)
    if compress_ratio > 1:
        valid &= (positions_long + 1) % compress_ratio == 0
    if valid_mask is not None:
        valid &= valid_mask[: positions.numel()].to(
            device=positions.device, dtype=torch.bool
        )
    if not valid.any():
        return slot_mapping

    cp_cycle = cp_world_size * cp_kv_cache_interleave_size
    cycle_indices = positions_long // cp_cycle
    cycle_offsets = positions_long % cp_cycle
    owner_ranks = cycle_offsets // cp_kv_cache_interleave_size
    valid &= owner_ranks == cp_rank

    local_positions = (
        cycle_indices * cp_kv_cache_interleave_size
        + cycle_offsets % cp_kv_cache_interleave_size
    )
    storage_positions = local_positions // compress_ratio
    block_indices = storage_positions // block_size
    block_offsets = storage_positions % block_size
    valid &= block_indices < block_table.shape[1]
    valid &= req_indices < block_table.shape[0]
    if not valid.any():
        return slot_mapping

    valid_req_indices = req_indices[valid]
    valid_block_indices = block_indices[valid]
    physical_blocks = block_table[valid_req_indices, valid_block_indices].to(
        dtype=torch.long
    )
    valid_physical = physical_blocks >= 0
    if valid_physical.any():
        valid_slots = (
            physical_blocks[valid_physical] * block_size
            + block_offsets[valid][valid_physical]
        )
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()[valid_physical]
        slot_mapping[valid_positions] = valid_slots
    return slot_mapping


def build_pcp_restored_req_indices(
    *,
    positions: torch.Tensor,
    views: list[PCPInterleaveRequestView],
) -> torch.Tensor:
    """Build request indices aligned to a PCP all-gather restored buffer."""
    req_indices = torch.full_like(positions, -1, dtype=torch.long)
    row_start = 0
    for view in views:
        row_end = row_start + int(view.restore_idx.numel())
        req_indices[row_start : min(row_end, positions.numel())] = int(view.req_idx)
        row_start = row_end
        if row_start >= positions.numel():
            break
    return req_indices


def build_pcp_restored_valid_mask(
    *,
    positions: torch.Tensor,
    views: list[PCPInterleaveRequestView],
) -> torch.Tensor:
    """Mark real restored rows and exclude per-rank padding rows."""
    valid_mask = torch.zeros_like(positions, dtype=torch.bool)
    row_start = 0
    for view in views:
        row_end = row_start + int(view.restore_idx.numel())
        # Restored rows are sorted in request-local token order.  The view's
        # ``global_seq_len`` is the scheduled query length, while ``positions``
        # contains absolute model positions.  Comparing the two is wrong for a
        # chunk with an existing prefix.  Padding follows the real query rows,
        # so validity is determined by row count instead.
        valid_end = min(row_start + int(view.global_seq_len), row_end)
        valid_mask[row_start:valid_end] = True
        row_start = row_end
        if row_start >= positions.numel():
            break
    return valid_mask


def build_pcp_query_plan(
    *,
    pcp_allgather_restore_idx: torch.Tensor,
    views: list[PCPInterleaveRequestView],
    pcp_world_size: int,
    pcp_rank: int,
) -> DeepseekV4PcpQueryPlan:
    """Build compact global query rows and this rank's selection from them."""
    restore_len = pcp_allgather_restore_idx.numel()
    if pcp_world_size <= 1 or restore_len % pcp_world_size != 0:
        raise ValueError(
            "invalid PCP query topology: "
            f"restore_len={restore_len}, world_size={pcp_world_size}"
        )
    if not 0 <= pcp_rank < pcp_world_size:
        raise ValueError(f"invalid PCP rank {pcp_rank} for world_size={pcp_world_size}")

    restored_valid_mask = build_pcp_restored_valid_mask(
        positions=pcp_allgather_restore_idx,
        views=views,
    )
    compact_restored_indices = torch.nonzero(
        restored_valid_mask, as_tuple=False
    ).flatten()

    num_local_tokens = restore_len // pcp_world_size
    local_concat_indices = torch.arange(
        num_local_tokens,
        dtype=pcp_allgather_restore_idx.dtype,
        device=pcp_allgather_restore_idx.device,
    )
    local_concat_indices += pcp_rank * num_local_tokens
    inverse_restore = torch.empty_like(pcp_allgather_restore_idx)
    inverse_restore.scatter_(
        0,
        pcp_allgather_restore_idx,
        torch.arange(
            restore_len,
            dtype=pcp_allgather_restore_idx.dtype,
            device=pcp_allgather_restore_idx.device,
        ),
    )
    local_restored_indices = torch.index_select(
        inverse_restore, 0, local_concat_indices
    )

    restored_to_compact = torch.full_like(pcp_allgather_restore_idx, -1)
    restored_to_compact[compact_restored_indices] = torch.arange(
        compact_restored_indices.numel(),
        dtype=pcp_allgather_restore_idx.dtype,
        device=pcp_allgather_restore_idx.device,
    )
    local_compact_indices = torch.index_select(
        restored_to_compact, 0, local_restored_indices
    )
    local_valid_mask = local_compact_indices >= 0

    return DeepseekV4PcpQueryPlan(
        restored_valid_mask=restored_valid_mask,
        compact_restored_indices=compact_restored_indices,
        local_compact_indices=local_compact_indices[local_valid_mask],
        local_valid_mask=local_valid_mask,
        num_local_tokens=num_local_tokens,
    )


def compact_pcp_sparse_indices(
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the sparse kernel valid prefix free of padding sentinels."""
    if indices.numel() == 0:
        return indices, lengths

    offsets = torch.arange(indices.shape[1], device=indices.device)
    valid_prefix = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    valid_indices = valid_prefix & (indices >= 0)
    new_lengths = valid_indices.sum(dim=1).to(lengths.dtype)
    if torch.equal(new_lengths, lengths):
        return indices, lengths

    compacted = torch.full_like(indices, -1)
    compacted_offsets = valid_indices.to(torch.long).cumsum(dim=1) - 1
    row_ids = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1)
    row_ids = row_ids.expand_as(indices)
    compacted[row_ids[valid_indices], compacted_offsets[valid_indices]] = indices[
        valid_indices
    ]
    return compacted, new_lengths


def compact_pcp_sparse_prefill_queries(
    q: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove PCP query rows that have no locally addressable KV."""
    valid_rows = torch.nonzero(lengths > 0, as_tuple=False).flatten()
    return (
        q.index_select(0, valid_rows),
        indices.index_select(0, valid_rows),
        lengths.index_select(0, valid_rows),
        valid_rows,
    )


def overlay_pcp_restored_swa_kv_workspace(
    *,
    out: torch.Tensor,
    restored_kv: torch.Tensor,
    restored_positions: torch.Tensor,
    restored_valid_mask: torch.Tensor,
    views: list[PCPInterleaveRequestView],
    chunk_start: int,
    chunk_end: int,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    chunk_n: int,
    chunk_m: int,
) -> None:
    """Overlay all-gather restored current-step SWA KV into a dense workspace."""
    if chunk_start >= chunk_end:
        return

    flat_out = out.view(-1, out.shape[-1])
    row_start = sum(int(view.restore_idx.numel()) for view in views[:chunk_start])
    for req_offset, req_idx in enumerate(range(chunk_start, chunk_end)):
        view = views[req_idx]
        row_end = row_start + int(view.restore_idx.numel())
        req_positions = restored_positions[row_start:row_end].to(torch.long)
        req_valid = restored_valid_mask[row_start:row_end].to(torch.bool)
        req_kv = restored_kv[row_start:row_end]

        seq_len = int(seq_lens[req_offset].item())
        gather_len = int(gather_lens[req_offset].item())
        gather_start = seq_len - gather_len
        req_valid = (
            req_valid & (req_positions >= gather_start) & (req_positions < seq_len)
        )
        if req_valid.any():
            valid_positions = req_positions[req_valid]
            valid_kv = req_kv[req_valid]
            target_rows = (
                req_offset * chunk_m + chunk_n + valid_positions - gather_start
            )
            flat_out.index_copy_(0, target_rows.to(torch.long), valid_kv)

        row_start = row_end


def pcp_swa_torch_sparse_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Reference sparse SWA attention for PCP prefill segment fallback."""
    out.zero_()
    active_rows = topk_length > 0
    if not active_rows.any():
        return

    q = q[active_rows]
    indices = indices[active_rows]
    topk_length = topk_length[active_rows]

    num_rows, num_heads, _ = q.shape
    max_topk = indices.shape[1]
    valid_offsets = torch.arange(
        max_topk, device=indices.device, dtype=topk_length.dtype
    )
    valid_mask = valid_offsets.unsqueeze(0) < topk_length.unsqueeze(1)
    safe_indices = torch.where(indices >= 0, indices, torch.zeros_like(indices))
    gathered_kv = kv.index_select(0, safe_indices.reshape(-1)).view(
        num_rows, max_topk, kv.shape[1], kv.shape[-1]
    )
    gathered_kv = gathered_kv.squeeze(2).to(torch.float32)
    q_float = q.to(torch.float32)
    scores = torch.einsum("rhd,rkd->rhk", q_float, gathered_kv) * sm_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(1), -float("inf"))
    if attn_sink is not None:
        sink = attn_sink[:num_heads].to(torch.float32).view(1, num_heads, 1)
        sink = sink.expand(num_rows, -1, -1)
        scores = torch.cat([scores, sink], dim=-1)

    scores = torch.nan_to_num(
        scores,
        nan=torch.finfo(scores.dtype).min,
        posinf=torch.finfo(scores.dtype).max,
        neginf=torch.finfo(scores.dtype).min,
    )
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores, dim=-1)
    if attn_sink is not None:
        probs = probs[..., :max_topk]
    out[active_rows].copy_(
        torch.einsum("rhk,rkd->rhd", probs, gathered_kv).to(out.dtype)
    )


def build_pcp_sparse_prefill_rows(
    *,
    combined_lens: torch.Tensor,
    positions: torch.Tensor,
    local_query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    chunk_n: int,
    chunk_m: int,
) -> DeepseekV4PcpSparseRows:
    """Map PCP query rows to compact rows in the gathered KV workspace."""
    q_tokens = int(positions.numel())
    q_rows = torch.empty(q_tokens, dtype=torch.long, device=positions.device)
    chunk_size = int(seq_lens.numel())
    for req_idx in range(chunk_size):
        req_query_start = int(local_query_start_loc[req_idx].item())
        req_query_end = int(local_query_start_loc[req_idx + 1].item())
        if req_query_start == req_query_end:
            continue
        req_seq_len = int(seq_lens[req_idx].item())
        req_gather_len = int(gather_lens[req_idx].item())
        req_gather_start = req_seq_len - req_gather_len
        req_positions = positions[req_query_start:req_query_end].to(torch.long)
        req_valid = (req_positions >= req_gather_start) & (req_positions < req_seq_len)
        req_rows = req_idx * chunk_m + chunk_n + req_positions - req_gather_start
        req_rows = torch.where(req_valid, req_rows, torch.zeros_like(req_rows))
        q_rows[req_query_start:req_query_end] = req_rows

    valid_query_mask = combined_lens > 0
    rows_min = 0
    rows_max = -1
    if q_tokens > 0 and valid_query_mask.any():
        valid_rows = q_rows[valid_query_mask]
        rows_min = int(valid_rows.min().item())
        rows_max = int(valid_rows.max().item())
        if rows_min < 0 or rows_max >= chunk_size * chunk_m:
            raise ValueError(
                "DeepSeek V4 PCP sparse prefill query rows are outside the "
                "gathered KV workspace: "
                f"min={rows_min}, max={rows_max}, "
                f"workspace_rows={chunk_size * chunk_m}"
            )
        sparse_rows = rows_max + 1
    else:
        sparse_rows = 1
    return DeepseekV4PcpSparseRows(
        q_rows=q_rows,
        sparse_rows=sparse_rows,
        valid_query_mask=valid_query_mask,
        rows_min=rows_min,
        rows_max=rows_max,
    )


def build_pcp_swa_prefill_segments(
    *,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    positions: torch.Tensor,
    local_query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    chunk_n: int,
    chunk_m: int,
    window_size: int,
    segment_size: int = 64,
) -> list[DeepseekV4PcpSwaSegment]:
    """Build compact local SWA segments for PCP sparse prefill.

    FlashMLA sparse prefill consumes compact local row coordinates. PCP prefill
    positions are global within the request, so each segment is rebased to the
    smallest local KV window that covers its valid query rows.
    """
    segments: list[DeepseekV4PcpSwaSegment] = []
    chunk_size = int(seq_lens.numel())
    for req_idx in range(chunk_size):
        req_query_start = int(local_query_start_loc[req_idx].item())
        req_query_end = int(local_query_start_loc[req_idx + 1].item())
        req_seq_len = int(seq_lens[req_idx].item())
        req_gather_len = int(gather_lens[req_idx].item())
        req_gather_start = req_seq_len - req_gather_len

        seg_query_start = req_query_start
        while seg_query_start < req_query_end:
            tentative_query_end = min(seg_query_start + segment_size, req_query_end)
            seg_positions_all = positions[seg_query_start:tentative_query_end].to(
                torch.long
            )
            seg_lens_all = combined_lens[seg_query_start:tentative_query_end]
            seg_valid_all = seg_lens_all > 0
            if seg_positions_all.numel() > 1:
                adjacent_valid = seg_valid_all[:-1] & seg_valid_all[1:]
                position_breaks = adjacent_valid & (
                    seg_positions_all[1:] != seg_positions_all[:-1] + 1
                )
                if position_breaks.any():
                    first_break = int(position_breaks.nonzero()[0].item()) + 1
                    seg_query_end = seg_query_start + first_break
                else:
                    seg_query_end = tentative_query_end
            else:
                seg_query_end = tentative_query_end

            seg_positions = positions[seg_query_start:seg_query_end].to(torch.long)
            seg_lens = combined_lens[seg_query_start:seg_query_end]
            seg_valid = seg_lens > 0
            if not seg_valid.any():
                segments.append(
                    DeepseekV4PcpSwaSegment(
                        query_start=seg_query_start,
                        query_end=seg_query_end,
                        kv_start=0,
                        kv_end=0,
                        sparse_rows=1,
                        q_rows=torch.zeros_like(seg_positions),
                        shifted_indices=combined_indices[seg_query_start:seg_query_end],
                        topk_lens=seg_lens,
                        valid_mask=seg_valid,
                    )
                )
                seg_query_start = seg_query_end
                continue

            valid_positions = seg_positions[seg_valid]
            seg_base_pos = max(
                req_gather_start,
                int(valid_positions.min().item()) - window_size + 1,
            )
            seg_end_pos = int(valid_positions.max().item()) + 1
            kv_start = req_idx * chunk_m + chunk_n + seg_base_pos - req_gather_start
            kv_end = req_idx * chunk_m + chunk_n + seg_end_pos - req_gather_start
            q_rows = seg_positions - seg_base_pos
            q_rows = torch.where(seg_valid, q_rows, torch.zeros_like(q_rows))
            sparse_rows = int(q_rows[seg_valid].max().item()) + 1

            seg_indices = combined_indices[seg_query_start:seg_query_end]
            shifted_indices = torch.where(
                seg_indices >= 0,
                seg_indices - kv_start,
                seg_indices,
            )
            valid_offsets = torch.arange(
                shifted_indices.shape[1],
                device=shifted_indices.device,
                dtype=seg_lens.dtype,
            )
            valid_index_mask = valid_offsets.unsqueeze(0) < seg_lens.unsqueeze(1)
            valid_shifted_indices = shifted_indices[valid_index_mask]
            if (valid_shifted_indices < 0).any() or (
                valid_shifted_indices >= kv_end - kv_start
            ).any():
                raise ValueError(
                    "DeepSeek V4 PCP SWA prefill segment indices are outside "
                    "the rebased KV workspace"
                )

            segments.append(
                DeepseekV4PcpSwaSegment(
                    query_start=seg_query_start,
                    query_end=seg_query_end,
                    kv_start=kv_start,
                    kv_end=kv_end,
                    sparse_rows=sparse_rows,
                    q_rows=q_rows,
                    shifted_indices=shifted_indices,
                    topk_lens=seg_lens,
                    valid_mask=seg_valid,
                )
            )
            seg_query_start = seg_query_end
    return segments
