"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.forward_context import get_forward_context
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)

# from flag_attn.paged import attention as triton_flag_attention_flash_attention
# from flag_attn.split_kv import _fwd_split_kv_kernel, _fwd_combine_kv_splits, num_splits_herustic
# from flag_attn.split_kv import get_fwd_config as get_fwd_config_kv_split
import triton
import triton.language as tl

# Requires triton 2.2.0
def triton_flash_flag_attention(
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    num_splits: int = 0,
) -> None:
    out = torch.empty_like(query)

    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert padded_group_size == 1 or kv_block_size >= 16, f"kv_block_size={kv_block_size}"
    # query_group_size in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    # assert query_group_size > 0 and query_group_size & (query_group_size-1) == 0, f"query_group_size={query_group_size}"

    # config for A100
    # TODO: support more devices and optimize
    device = torch.cuda.device_of(query)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, kv_block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_block_size >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = triton.cdiv(max_context_len, num_splits)
        partition_size = triton.next_power_of_2(partition_size)

    num_warps = get_num_warps(query_group_size, head_size, kv_block_size)
    num_stages = get_num_stages(partition_size, kv_block_size)
    with torch.cuda.device(device):
        if num_splits == 1:
            grid = (num_seqs, num_kv_heads, 1)
            _paged_attn_kernel[grid](
                out,  # dummy input
                out,  # dummy input
                out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(1),
                out.stride(1),
                out.stride(2),
                head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                PARTITION_SIZE=0,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        else:
            grid = (num_seqs, num_kv_heads, num_splits)
            m_i = torch.empty(
                size=(num_seqs, num_kv_heads, num_splits, query_group_size),
                dtype=torch.float32,
                device=query.device,
            )
            l_i = torch.empty_like(m_i)
            tmp_out = torch.empty(
                size=(
                    num_seqs,
                    num_kv_heads,
                    num_splits,
                    query_group_size,
                    head_size,
                ),
                dtype=out.dtype,
                device=out.device,
            )

            assert (partition_size >= kv_block_size) and (partition_size % kv_block_size == 0), \
                f"partition_size={partition_size}, kv_block_size={kv_block_size}"
            _paged_attn_kernel[grid](
                m_i,
                l_i,
                tmp_out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                tmp_out.stride(0),
                tmp_out.stride(1),
                tmp_out.stride(2),
                tmp_out.stride(3),
                tmp_out.stride(4),
                head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                partition_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            reduce_grid = (num_seqs, num_kv_heads)
            next_num_splits = triton.next_power_of_2(num_splits)

            _paged_attn_v2_reduce_kernel[reduce_grid](
                out,
                m_i,
                l_i,
                tmp_out,
                context_lens,
                num_splits,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                head_size,
                query_group_size,
                num_kv_heads,
                partition_size,
                next_num_splits,
            )
    return out


def get_num_warps(query_group_size, head_size, kv_block_size):
    if query_group_size == 1:
        if head_size >= 128 and kv_block_size >= 32:
            return 16
        else:
            return 8
    else:
        return 4


def get_num_stages(partition_size, kv_block_size):
    if partition_size == 0:
        return 1
    else:
        if torch.cuda.get_device_capability() == (8, 0):
            if kv_block_size < 256:
                return 3
            else:
                return 2
        elif torch.cuda.get_device_capability() == (8, 6):
            if kv_block_size < 256:
                return 2
            else:
                return 1
        else:
            return 1


# @triton.heuristics(
#     {
#         "num_warps": lambda args: get_num_warps(
#             args["QUERY_GROUP_SIZE"], args["HEAD_SIZE"], args["KV_BLOCK_SIZE"]
#         ),
#         "num_stages": lambda args: get_num_stages(
#             args["QUERY_GROUP_SIZE"], args["KV_BLOCK_SIZE"]
#         ),
#     }
# )
@triton.jit
def _paged_attn_kernel(
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale,
    stride_bt0,
    stride_bt1,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_kv0,
    stride_kv1,
    stride_kv2,
    stride_kv3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx, KV_BLOCK_SIZE)
    else:
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)

    kv_offset = (
        kv_head_idx * stride_kv1
        + block_offset[:, None] * stride_kv2
        + head_offset[None, :] * stride_kv3
    )

    # Load queries.
    q_offset = (
        seq_idx * stride_q0
        + (kv_head_idx * QUERY_GROUP_SIZE + padding_group_offset[:, None]) * stride_q1
        + head_offset[None, :] * stride_q2
    )
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    # q: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    q = tl.load(q_ptr + q_offset, mask=group_mask, other=0.0)

    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(
            block_tables_ptr + seq_idx * stride_bt0 + block_idx * stride_bt1
        )

        # Load a key block.
        kv_block_offset = block_number * stride_kv0 + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        kv_mask = mask_offset[:, None] < context_len

        # k: [KV_BLOCK_SIZE, HEAD_SIZE]
        k = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
        else:
            qk = tl.dot(q, k.T, out_dtype=tl.float32)

        qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLOCK_SIZE, HEAD_SIZE]
        v = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(p.T[:, :, None] * v[:, None, :], axis=0)
        else:
            p = p.to(v.dtype)
            acc += tl.dot(p, v, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    if USE_PARTITIONING:
        part_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx)
            * max_num_partitions
            * QUERY_GROUP_SIZE
            + part_idx * QUERY_GROUP_SIZE
            + padding_group_offset
        )
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)

    out_offset = seq_idx * stride_o0
    if USE_PARTITIONING:
        out_offset += kv_head_idx * stride_o1
    else:
        out_offset += kv_head_idx * QUERY_GROUP_SIZE * stride_o1
    out_offset += (
        part_idx * stride_o2
        + padding_group_offset[:, None] * stride_o3
        + head_offset[None, :] * stride_o4
    )

    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(
    out_ptr,  # [num_seqs, NUM_KV_HEADS, QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    max_num_partitions,  # partition stride
    stride_o0,
    stride_o1,
    stride_o2,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = (
        tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    if num_partitions == 1:
        tmp_out_offset = (
            seq_idx * NUM_KV_HEADS + kv_head_idx
        ) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)

        out_offset = (
            seq_idx * stride_o0
            + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
            + group_head_offset * stride_o2
        )
        tl.store(out_ptr + out_offset, tmp_out)
        return

    # Get the global max logit.
    ml_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE
        + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    )

    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    # m_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float("-inf"))
    # m: [QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [QUERY_GROUP_SIZE]
    l = tl.sum(l_i, axis=0)
    # r: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    r = l_i / l[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, QUERY_GROUP_SIZE, 1))

    tmp_out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx)
        * max_num_partitions
        * QUERY_GROUP_SIZE
        * HEAD_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE
        + tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, None, :]
    )
    # tmp_out: [NUM_PARTITIONS, QUERY_GROUP_SIZE, HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    # out: [QUERY_GROUP_SIZE, HEAD_SIZE]
    out = tl.sum((tmp_out * r).to(tl.float32), axis=0)

    out_offset = (
        seq_idx * stride_o0
        + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
        + group_head_offset * stride_o2
    )
    tl.store(out_ptr + out_offset, out)





class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch.
    max_query_len: Optional[int]

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int]

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=self.query_start_loc[self.num_prefills:]
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        return FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        
        # if self.use_triton_flash_attn:
        #     from vllm.attention.ops.triton_flash_attention import (  # noqa: F401
        #         triton_attention)
        #     self.attn_func = triton_attention
        #     # logger.debug("Using Triton FA in ROCmBackend")
        # from vllm.attention.ops.triton_flash_attention import triton_attention
        # self.attn_func = triton_attention
        # print("Set Triton attention func")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        # from vllm.attention.ops.triton_flash_attention import triton_attention
        # self.attn_func = triton_attention
        # print("Set Triton attention func")

        # output = triton_attention()
        output = torch.ops.vllm.unified_flash_attention(
            query,
            key,
            value,
            self.num_heads,
            self.head_size,
            self.num_kv_heads,
            kv_cache,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
            self.scale,
            self.sliding_window,
            self.alibi_slopes,
            self.logits_soft_cap,
        )

        return output


@torch.library.custom_op("vllm::unified_flash_attention",
                         mutates_args=["kv_cache"])
def unified_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    window_size: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> torch.Tensor:

    current_metadata = get_forward_context()
    assert current_metadata is not None
    assert isinstance(current_metadata, FlashAttentionMetadata)
    attn_metadata: FlashAttentionMetadata = current_metadata

    num_tokens, hidden_size = query.shape
    # Reshape the query, key, and value tensors.
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    if kv_cache.numel() > 0:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Reshape the input keys and values and store them in the cache.
        # If kv_cache is not provided, the new key and value tensors are
        # not cached. This happens during the initial memory profiling run.
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[0],
            kv_cache[1],
            attn_metadata.slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    num_prefill_tokens = attn_metadata.num_prefill_tokens
    num_decode_tokens = attn_metadata.num_decode_tokens
    assert key.shape[0] == num_prefill_tokens + num_decode_tokens, \
                f"key : {key.shape} : #prefill tokens {num_prefill_tokens} : #decode tokens {num_decode_tokens}" # noqa
    assert value.shape[0] == num_prefill_tokens + num_decode_tokens, \
                f"value : {value.shape} : #prefill toks {num_prefill_tokens} : #decode toks {num_decode_tokens}" # noqa

    # Query for decode. KV is not needed because it is already cached.
    decode_query = query[num_prefill_tokens:]
    # QKV for prefill.
    query = query[:num_prefill_tokens]
    key = key[:num_prefill_tokens]
    value = value[:num_prefill_tokens]

    assert query.shape[0] == num_prefill_tokens
    assert decode_query.shape[0] == num_decode_tokens

    prefill_output: Optional[torch.Tensor] = None
    decode_output: Optional[torch.Tensor] = None

    from vllm.attention.ops.triton_flash_attention import triton_attention
    if prefill_meta := attn_metadata.prefill_metadata:
        # Prompt run.
        if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0):
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            attn_masks = None
            if alibi_slopes is not None:
                attn_masks = _make_alibi_bias(
                    alibi_slopes,
                    query.dtype,
                    attn_metadata.seq_lens,
                    make_attn_mask=False)  # type: ignore
            print("triton_attention prefill 1")
            print(prefill_meta.seq_start_loc)
            print(prefill_meta.max_prefill_seq_len)
            prefill_output, _ = triton_attention(
                query,
                key,
                value,
                None,
                prefill_meta.seq_start_loc,
                prefill_meta.seq_start_loc,
                prefill_meta.max_prefill_seq_len,
                prefill_meta.max_prefill_seq_len,
                True,
                softmax_scale,
                attn_masks[0][None]
                if attn_masks is not None else None,
            )
            # prefill_output = triton_attention(
            #     q=query,
            #     k=key,
            #     v=value,
            #     o=None,
            #     cu_seqlens_q=prefill_meta.seq_start_loc,
            #     cu_seqlens_k=prefill_meta.seq_start_loc,
            #     max_seqlen_q=prefill_meta.max_prefill_seq_len,
            #     max_seqlen_k=prefill_meta.max_prefill_seq_len,
            #     # softmax_scale=softmax_scale,
            #     causal=True,
            #     sm_scale=1.0,
            #     bias=None,
            #     # window_size=window_size,
            #     # alibi_slopes=alibi_slopes,
            #     # softcap=logits_soft_cap,
            # )
            # prefill_output = flash_attn_varlen_func(
            #     q=query,
            #     k=key,
            #     v=value,
            #     cu_seqlens_q=prefill_meta.seq_start_loc,
            #     cu_seqlens_k=prefill_meta.seq_start_loc,
            #     max_seqlen_q=prefill_meta.max_prefill_seq_len,
            #     max_seqlen_k=prefill_meta.max_prefill_seq_len,
            #     softmax_scale=softmax_scale,
            #     causal=True,
            #     window_size=window_size,
            #     alibi_slopes=alibi_slopes,
            #     softcap=logits_soft_cap,
            # )
        else:
            # prefix-enabled attention
            print("triton_attention prefill 2")
            assert prefill_meta.seq_lens is not None
            max_seq_len = max(prefill_meta.seq_lens)
            prefill_output, _ = triton_attention(
                query,
                key,
                value,
                None,
                prefill_meta.seq_start_loc,
                prefill_meta.seq_start_loc,
                prefill_meta.max_prefill_seq_len,
                prefill_meta.max_prefill_seq_len,
                True,
                softmax_scale,
                None,
                None,
            )
            # prefill_output = flash_attn_varlen_func(  # noqa
            #     q=query,
            #     k=key_cache,
            #     v=value_cache,
            #     cu_seqlens_q=prefill_meta.query_start_loc,
            #     max_seqlen_q=prefill_meta.max_query_len,
            #     cu_seqlens_k=prefill_meta.seq_start_loc,
            #     max_seqlen_k=max_seq_len,
            #     softmax_scale=softmax_scale,
            #     causal=True,
            #     window_size=window_size,
            #     alibi_slopes=alibi_slopes,
            #     block_table=prefill_meta.block_tables,
            #     softcap=logits_soft_cap,
            # )

    if decode_meta := attn_metadata.decode_metadata:
        # Decoding run.
        # Use flash_attn_varlen_func kernel for speculative decoding
        # because different queries might have different lengths.
        assert decode_meta.max_decode_query_len is not None
        if decode_meta.max_decode_query_len > 1:
            print("triton_attention decode 1")
            decode_output, _ = triton_attention(
                decode_query,
                key_cache,
                value_cache,
                None,
                decode_meta.query_start_loc,
                decode_meta.seq_start_loc,
                decode_meta.max_decode_seq_len,
                decode_meta.max_decode_seq_len,
                True,
                softmax_scale,
                None,
                # None,
            )
            # decode_output = flash_attn_varlen_func(
            #     q=decode_query,
            #     k=key_cache,
            #     v=value_cache,
            #     cu_seqlens_q=decode_meta.query_start_loc,
            #     max_seqlen_q=decode_meta.max_decode_query_len,
            #     cu_seqlens_k=decode_meta.seq_start_loc,
            #     max_seqlen_k=decode_meta.max_decode_seq_len,
            #     softmax_scale=softmax_scale,
            #     causal=True,
            #     window_size=window_size,
            #     alibi_slopes=alibi_slopes,
            #     softcap=logits_soft_cap,
            #     block_table=decode_meta.block_tables,
            # )
        else:
            # Use flash_attn_with_kvcache for normal decoding.
            from vllm.attention.ops.flash import triton_flag_attention
            print("triton_attention decode 2")
            print(decode_meta.query_start_loc)
            print(decode_meta.seq_lens_tensor)
            print(decode_meta.seq_lens)
            print(decode_meta.seq_start_loc)
            print(decode_meta.max_decode_query_len)
            print(decode_meta.max_decode_seq_len)
            print(decode_meta.max_decode_seq_len)
            print(decode_meta.max_query_len)
            print(decode_meta.max_prefill_seq_len)
            print(decode_meta.seq_lens)
            print(decode_meta.query_start_loc)
            # print(prefill_meta.max_prefill_seq_len)
            #print(decode_meta.)
            # decode_output, _ = triton_attention(
            #     decode_query,
            #     # key_cache,
            #     # value_cache,
            #     key,
            #     value,
            #     None,
            #     # decode_meta.query_start_loc,
            #     # decode_meta.seq_start_loc,
            #     decode_meta.seq_lens_tensor,
            #     decode_meta.seq_lens_tensor,
            #     decode_meta.max_decode_seq_len,
            #     decode_meta.max_decode_seq_len,
            #     True,
            #     softmax_scale,
            #     None,
            #     # None,
            # )
            # decode_output, _ = triton_flag_attention(
            #     decode_query,
            #     # key_cache,
            #     # value_cache,
            #     key,
            #     value,
            #     True,
            #     softmax_scale,
            #     0,
            #     False,
            #     False,
            #     False,
            #     # decode_meta.query_start_loc,
            #     # decode_meta.seq_start_loc,
            #     # decode_meta.seq_lens_tensor,
            #     # decode_meta.seq_lens_tensor,
            #     # decode_meta.max_decode_seq_len,
            #     # decode_meta.max_decode_seq_len,
            #     # True,
            #     # softmax_scale,
            #     # None,
            #     # None,
            # )

            # sglang

            # from vllm.attention.ops.decode_attention import decode_attention_fwd_normal
            # decode_output = torch.empty_like(decode_query)
            # decode_attention_fwd_normal(
            #     decode_query.unsqueeze(1),
            #     key_cache,
            #     value_cache,
            #     decode_output,
            #     req_to_token,
            #     b_req_idx,
            #     b_start_loc,
            #     b_seq_len,
            #     attn_logits,
            #     max_len_in_batch,
            #     sm_scale,
            #     logit_cap,
            # )
            
            # example code for 11/14/2024 version
            # decode_output = triton_flash_flag_attention(
            #     query=decode_query, 
            #     key_cache=key_cache, 
            #     value_cache=value_cache, 
            #     context_lens=decode_meta.seq_lens_tensor, 
            #     block_tables=decode_meta.block_table, 
            #     attn_scale=softmax_scale,
            #     max_context_len=4096,
            #     num_splits=0,
            # )

            # decode_output = triton_flag_attention_flash_attention(
            #     query=decode_query, 
            #     key_cache=key_cache, 
            #     value_cache=value_cache, 
            #     context_lens=decode_meta.seq_lens_tensor, 
            #     block_tables=decode_meta.block_table, 
            #     attn_scale=softmax_scale,
            #     max_context_len=4096,
            #     num_splits=0,
            # )

            decode_output = flash_attn_with_kvcache(
                q=decode_query.unsqueeze(1),
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                softcap=logits_soft_cap,
            ).squeeze(1)

    if prefill_output is None:
        assert decode_output is not None
        return decode_output.view(num_decode_tokens, hidden_size)
    if decode_output is None:
        assert prefill_output is not None
        return prefill_output.view(num_prefill_tokens, hidden_size)

    # Chunked prefill does not work with speculative decoding.
    # Therefore, the query length for decode should be 1 in chunked prefill.
    assert decode_meta is not None
    decode_output = decode_output.squeeze(1)
    output = torch.cat([prefill_output, decode_output], dim=0)
    return output.view(num_tokens, hidden_size)


@unified_flash_attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    window_size: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(query)

def _make_alibi_bias(alibi_slopes: torch.Tensor,
                     dtype: torch.dtype,
                     seq_lens: Optional[List[int]],
                     make_attn_mask: bool = True) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            # NOTE(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(seq_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]

            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat(
                (num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = torch.empty(
                    (1, seq_len, seq_len),
                    dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1).to(
                        alibi_slopes.device)
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))

    return attn_biases