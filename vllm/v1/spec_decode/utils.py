# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection
from dataclasses import dataclass

import torch

from vllm.distributed.cp_mapping import map_cp_positions
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

PADDING_SLOT_ID = -1
MINIMAX_M3_TARGET_LAYER_COUNT = 60
MINIMAX_M3_DENSE_TARGET_LAYER_IDS = (0, 1, 2)
MINIMAX_M3_TARGET_LAYER_PREFIXES = ("language_model.model", "model")


def get_minimax_m3_target_attention_layer_name(layer_index: int) -> str:
    """Return the global attention-layer identity for a MiniMax-M3 layer."""
    if not 0 <= layer_index < MINIMAX_M3_TARGET_LAYER_COUNT:
        raise ValueError(
            "MiniMax-M3 target layer index must be in "
            f"[0, {MINIMAX_M3_TARGET_LAYER_COUNT})"
        )
    return f"language_model.model.layers.{layer_index}.self_attn.attn"


def get_minimax_m3_dense_target_attention_layer_names() -> tuple[str, ...]:
    """Return target layers requiring DCP1-equivalent EAGLE-visible parity."""
    return tuple(
        get_minimax_m3_target_attention_layer_name(layer_index)
        for layer_index in MINIMAX_M3_DENSE_TARGET_LAYER_IDS
    )


def resolve_minimax_m3_dense_target_attention_layer_names(
    available_layer_names: Collection[str],
) -> tuple[str, ...]:
    """Resolve the complete dense target layer set for either model wrapper."""
    resolved: list[tuple[str, ...]] = []
    available = set(available_layer_names)
    for prefix in MINIMAX_M3_TARGET_LAYER_PREFIXES:
        candidates = tuple(
            f"{prefix}.layers.{layer_index}.self_attn.attn"
            for layer_index in MINIMAX_M3_DENSE_TARGET_LAYER_IDS
        )
        present = set(candidates) & available
        if present and len(present) != len(candidates):
            raise ValueError(
                "MiniMax-M3 dense target attention layers are incomplete: "
                f"expected {list(candidates)!r}, got {sorted(present)!r}"
            )
        if present:
            resolved.append(candidates)
    if len(resolved) > 1:
        raise ValueError(
            "Multiple MiniMax-M3 dense target attention layer prefixes found"
        )
    return resolved[0] if resolved else ()


def get_eagle3_draft_attention_layer_name(
    total_target_layers: int,
    draft_layer_index: int = 0,
) -> str:
    """Return the global MiniMax-M3 layer identity for an EAGLE3 draft layer."""
    if total_target_layers != MINIMAX_M3_TARGET_LAYER_COUNT:
        raise ValueError("MiniMax-M3 EAGLE3 requires exactly 60 global target layers")
    if draft_layer_index < 0:
        raise ValueError("draft_layer_index must be non-negative")
    layer_index = total_target_layers + draft_layer_index
    return f"model.layers.{layer_index}.self_attn.attn"


@dataclass(frozen=True)
class PromptDraftKVCoverage:
    """Logical prompt draft-KV handoff between Prefill and Decode."""

    prompt_tokens: int
    target_prefix_tokens: int
    compatible_draft_prefix_tokens: int
    transfer_start_token: int
    transfer_end_token_exclusive: int
    decode_recompute_position: int

    @property
    def transfer_token_count(self) -> int:
        return self.transfer_end_token_exclusive - self.transfer_start_token


def get_prompt_draft_kv_coverage(
    prompt_tokens: int,
    target_prefix_tokens: int,
    compatible_draft_prefix_tokens: int,
) -> PromptDraftKVCoverage:
    """Freeze the half-open prompt draft-KV transfer and handoff contract."""
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if not 0 <= target_prefix_tokens <= prompt_tokens:
        raise ValueError("target_prefix_tokens must be within the prompt")
    if not 0 <= compatible_draft_prefix_tokens <= target_prefix_tokens:
        raise ValueError(
            "compatible_draft_prefix_tokens must be within the target prefix"
        )

    decode_recompute_position = prompt_tokens - 1
    transfer_start_token = min(
        compatible_draft_prefix_tokens,
        decode_recompute_position,
    )
    return PromptDraftKVCoverage(
        prompt_tokens=prompt_tokens,
        target_prefix_tokens=target_prefix_tokens,
        compatible_draft_prefix_tokens=compatible_draft_prefix_tokens,
        transfer_start_token=transfer_start_token,
        transfer_end_token_exclusive=decode_recompute_position,
        decode_recompute_position=decode_recompute_position,
    )


def expand_dcp_parent_block_table(
    parent_block_table: torch.Tensor,
    dcp_world_size: int,
    max_model_len: int,
    kernel_block_size: int,
) -> torch.Tensor:
    """Expand each DCP parent block into full-temporal physical child pages."""
    if dcp_world_size <= 1:
        return parent_block_table
    if max_model_len <= 0 or kernel_block_size <= 0:
        raise ValueError("max_model_len and kernel_block_size must be positive")

    child_offsets = torch.arange(
        dcp_world_size,
        dtype=parent_block_table.dtype,
        device=parent_block_table.device,
    )
    children = parent_block_table.unsqueeze(-1) * dcp_world_size + child_offsets
    # Block zero is the null block. Keep all children of padding entries null.
    children.masked_fill_(parent_block_table.unsqueeze(-1) == 0, 0)
    max_child_blocks = (max_model_len + kernel_block_size - 1) // kernel_block_size
    expanded = children.flatten(1)
    if expanded.shape[1] < max_child_blocks:
        raise ValueError(
            "DCP parent block table is too short for full-temporal KV: "
            f"{expanded.shape[1]} child blocks < {max_child_blocks}"
        )
    return expanded[:, :max_child_blocks].contiguous()


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def _advance_cpu_sequence_metadata(
    metadata: CommonAttentionMetadata,
    max_model_len: int,
) -> None:
    seq_lens = metadata._seq_lens_cpu
    upper_bound = metadata.seq_lens_cpu_upper_bound
    exceeds_max = None
    if seq_lens is not None:
        exceeds_max = seq_lens >= max_model_len
        seq_lens.add_(1)
        seq_lens.masked_fill_(exceeds_max, 1)
    elif upper_bound is not None:
        exceeds_max = upper_bound >= max_model_len

    num_computed_tokens = metadata._num_computed_tokens_cpu
    if num_computed_tokens is not None:
        num_computed_tokens.add_(1)
        if exceeds_max is not None:
            num_computed_tokens.masked_fill_(exceeds_max, 0)

    upper_bound_aliases_seq_lens = (
        upper_bound is not None
        and seq_lens is not None
        and upper_bound.data_ptr() == seq_lens.data_ptr()
        and upper_bound.shape == seq_lens.shape
        and upper_bound.stride() == seq_lens.stride()
    )
    if upper_bound is not None and not upper_bound_aliases_seq_lens:
        exceeds_upper_bound = upper_bound >= max_model_len
        upper_bound.add_(1)
        upper_bound.masked_fill_(exceeds_upper_bound, 1)


@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    positions_ptr,  # [batch_size] - current positions (1D view for M-RoPE)
    block_table_ptr,  # [batch_size, n_blocks_per_req]
    block_table_stride,  # stride for block_table dim 1
    seq_lens_ptr,  # [batch_size] - read and write
    out_clamped_positions_ptr,  # [batch_size] (output)
    out_slot_mapping_ptr,  # [input_batch_size] (output)
    dcp_world_size,
    dcp_rank,
    cp_kv_cache_interleave_size,
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    n_blocks_per_req: tl.constexpr,
    PAD_ID: tl.constexpr,
    batch_size,
):
    """
    Fused kernel for EAGLE autoregressive step: updates positions, slot mapping,
    and sequence lengths in a single kernel to reduce launch overhead.

    Launched with input_batch_size threads. Threads with req_idx >= batch_size
    are cudagraph padding slots and only write PADDING_SLOT_ID.

    Each real thread handles one request in the batch. Computes:
    - new_position = position + 1, clamped if exceeds max_model_len
    - slot_mapping from block table lookup
    - seq_lens += 1, or 1 if position exceeds max
    """
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
        return

    # Load current position and increment
    position = tl.load(positions_ptr + req_idx)
    new_position = position + 1

    # Check bounds and compute clamped position
    exceeds_max = new_position >= max_model_len
    clamped_position = tl.where(exceeds_max, 0, new_position)

    cp_cycle = dcp_world_size * cp_kv_cache_interleave_size
    owner_rank = (clamped_position % cp_cycle) // cp_kv_cache_interleave_size
    local_position = (
        clamped_position // cp_cycle * cp_kv_cache_interleave_size
        + clamped_position % cp_kv_cache_interleave_size
    )

    # The block table is compacted to this DCP rank's local token space.
    # Clamp block_number to avoid OOB when position is at max
    block_number = local_position // block_size
    block_number = tl.minimum(block_number, n_blocks_per_req - 1)

    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
    slot_id = block_id * block_size + (local_position % block_size)
    slot_id = tl.where(
        exceeds_max | (owner_rank != dcp_rank),
        PAD_ID,
        slot_id,
    )

    # Update seq_lens: +1 normally, or 1 if exceeded
    seq_len = tl.load(seq_lens_ptr + req_idx)
    new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
    new_seq_len = tl.minimum(new_seq_len, max_model_len)

    # Store outputs
    tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
    tl.store(out_slot_mapping_ptr + req_idx, slot_id)
    tl.store(seq_lens_ptr + req_idx, new_seq_len)


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    """
    Fused update of slot mapping and metadata for one EAGLE autoregressive step.
    Updates seq_lens in place. Writes to out_clamped_positions and out_slot_mapping.

    When input_batch_size > batch_size, threads beyond batch_size write
    PADDING_SLOT_ID to out_slot_mapping for cudagraph padding.

    Args:
        positions_1d: [batch_size] current positions (use positions[0] for M-RoPE)
        block_table_tensor: [batch_size, n_blocks_per_req]
        seq_lens: [batch_size] updated in place
        block_size: KV cache block size
        max_model_len: max model length for clamping
        out_clamped_positions: [batch_size] output buffer for clamped positions
        out_slot_mapping: [input_batch_size] output buffer for slot mapping
        input_batch_size: total batch size including cudagraph padding;
            defaults to batch_size (no padding)
    """
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size

    n_blocks_per_req = block_table_tensor.shape[1]
    eagle_step_slot_mapping_metadata_kernel[(input_batch_size,)](
        positions_1d,
        block_table_tensor,
        block_table_tensor.stride(0),
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        dcp_world_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        block_size=block_size,
        max_model_len=max_model_len,
        n_blocks_per_req=n_blocks_per_req,
        PAD_ID=PADDING_SLOT_ID,
        batch_size=batch_size,
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
):
    """
    Fused kernel for Eagle prepare_input_padded. This kernel computes the
    token index to sample for each request, taking into account the number
    of draft tokens and the number of valid sampled tokens (which is one more than
    the number of accepted tokens).
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected/padded tokens are negative; valid tokens are in
        # [0, vocab_size).
        is_valid_mask = (token_ids >= 0) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def compute_slot_mapping_from_block_table(
    query_start_loc: torch.Tensor,
    block_table_tensor: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    max_model_len: int,
    *,
    num_new_tokens: int = 0,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    is_rejected_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map global token positions through an explicit block table."""
    batch_size, n_blocks_per_req = block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        query_start_loc[1:] - query_start_loc[:-1] + num_new_tokens,
        output_size=len(positions),
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(positions, max=max_model_len - 1)
    owner_ranks, local_positions = map_cp_positions(
        clamped_positions,
        cp_world_size=dcp_world_size,
        interleave_size=cp_kv_cache_interleave_size,
    )
    block_table_indices = req_indices * n_blocks_per_req + local_positions // block_size
    block_nums = block_table_tensor.view(-1)[block_table_indices]
    block_offsets = local_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    new_slot_mapping.masked_fill_(owner_ranks != dcp_rank, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    if is_rejected_token_mask is not None:
        new_slot_mapping.masked_fill_(
            is_rejected_token_mask,
            PADDING_SLOT_ID,
        )
    return new_slot_mapping


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
):
    return compute_slot_mapping_from_block_table(
        query_start_loc=cad.query_start_loc,
        block_table_tensor=cad.block_table_tensor,
        positions=new_positions,
        block_size=block_size,
        max_model_len=max_model_len,
        num_new_tokens=num_new_tokens,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        is_rejected_token_mask=is_rejected_token_mask,
    )


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by N.
    Also all seq lens are increased by N.
    This is useful e.g. in speculative decoding with parallel drafting, where we
    extend each sequence by N tokens and predict all tokens in one pass.
    The slot mapping is computed externally, as it requires more information.
    """
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_seq_lens_cpu = cad._seq_lens_cpu + N if cad._seq_lens_cpu is not None else None
    new_seq_lens_cpu_upper_bound = (
        cad.seq_lens_cpu_upper_bound + N
        if cad.seq_lens_cpu_upper_bound is not None
        else None
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # each request is extended by N tokens -> batch_size * N tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # All query lens increase by N, so max query len increases by N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
        _seq_lens_cpu=new_seq_lens_cpu,
        seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
        dcp_local_seq_lens_cpu=None,
    )
    return new_cad


# Unified copy/expand kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # (Padded) Inputs from the target model
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (output)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (output)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc_ptr,  # [num_reqs + 1], last value is the total num input tokens
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # Sizing info
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Blocks along token dim to handle prefills
):
    """
    Copy and expand inputs from the target model to the drafting buffers for Eagle
    speculative decoding. This kernel handles padding slots and parallel drafting
    tokens, if enabled.
    """
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # Load query locations
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # Calculate number of valid tokens to copy and input offset
    # With shift_input_ids=True, we skip the first token
    # Output layout: each request gets (input_len + num_padding_slots_per_request) slots
    # But with shift, we lose one token per request
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # Number of rejected tokens from previous speculation
    num_rejected = next_query_start_loc - query_end_loc - 1

    # Total output tokens for this request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Process tokens in this block
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # Compute masks for different output regions:
    # [0, num_valid_tokens): valid tokens copied from input
    # [num_valid_tokens]: bonus token from next_token_ids
    # (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
    #     parallel drafting slots
    # [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
    #     rejected slots
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # Compute output indices
    out_idx = output_start + j

    # For valid tokens, compute input index
    in_idx = query_start_loc + input_offset + j
    # Clamp to avoid out-of-bounds access (masked loads still need valid addresses)
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # Load input tokens (masked to valid region)
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # Load the starting position for this request (first position in the sequence)
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # Load bonus token for this request
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # Build final token_ids based on region
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # Build final positions:
    # Positions are NOT shifted - they start from the first input position and increment
    # Output position j gets start_pos + j
    # (e.g., input positions [5,6,7] -> output [5,6,7,8,9,...])
    positions = start_pos + j
    # Rejected positions are don't-care, set to 0
    positions = tl.where(is_rejected_region, 0, positions)

    # Compute output masks
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # Compute indices of new tokens (bonus + parallel drafting) for sampling
    # New tokens are at positions
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # 0 for bonus, 1, 2, ... for parallel drafting
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # Compute hidden state mapping (source index -> destination index)
    # This maps each input position to its corresponding output position
    # Hidden states don't get shifted, so we map all input tokens (including rejected)
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # Store outputs
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )


@triton.jit
def copy_and_expand_dflash_inputs_kernel(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    """
    Fused kernel for DFlash first-pass input setup.

    Per request, this kernel:
      1. Copies context positions from target_positions to
         out_context_positions.
      2. Computes query positions (last_target_pos + 1 + offset) and writes
         them to out_query_positions.
      3. Writes input_ids for query tokens: [next_token, mask, mask, ...].
      4. Computes slot_mapping for context and query positions into separate
         buffers via block_table lookup.
      5. Writes token_indices_to_sample for the mask (speculative) tokens.
    """
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # Load context token range for this request
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    total_tokens = num_ctx + num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx  # offset within query portion (0-indexed)

    # --- Positions ---
    # Context: load from target_positions
    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)

    # Query: last_valid_pos + 1 + query_off
    # In padded mode, ctx_end includes rejected tokens; use valid_ctx_end
    # to find the last accepted context position.
    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off

    positions = tl.where(is_ctx, ctx_pos, query_pos)

    # Context and query positions go to separate buffers.
    ctx_pos_out = ctx_start + j
    tl.store(out_context_positions_ptr + ctx_pos_out, ctx_pos, mask=is_ctx)
    query_out = req_idx * num_query_per_req + query_off
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    # --- Slot mapping (block_table lookup for all positions) ---
    block_num = positions // block_size
    # # Clamp block_number to avoid OOB when position is at max
    block_num = tl.minimum(block_num, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + (positions % block_size)
    tl.store(out_context_slot_mapping_ptr + ctx_pos_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    # --- Input IDs (query tokens only) ---
    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    # --- Token indices to sample (mask tokens, skip the bonus token) ---
    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    valid_prev = (
        (prev_positions >= 0)
        & (prev_positions < valid_sampled_token_count.shape[0])
        & (prev_positions < num_computed_tokens.shape[0])
        & (prev_positions < prev_num_draft_tokens.shape[0])
    )
    # Invalid rows fall back to the CPU value just like new requests.
    gather_indices = torch.where(
        valid_prev, prev_positions, torch.zeros_like(prev_positions)
    )

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = valid_prev & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


def unconditional_to_conditional_rates(rates: list[float]) -> list[float]:
    """Convert per-position unconditional rates to per-position conditional
    rates for the early-terminating rejection loop (c_i = p_i / p_{i-1})."""
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]
