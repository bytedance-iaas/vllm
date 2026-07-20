# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata, PCPAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_pcp_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_pcp_compressed_slot_mapping_accepts_empty_input():
    logical_slots = torch.empty(0, dtype=torch.int64)
    positions = torch.empty(0, dtype=torch.int64)

    physical_slots = get_pcp_compressed_slot_mapping(
        logical_slots,
        positions,
        logical_block_size=256,
        storage_block_size=64,
        compress_ratio=4,
    )

    assert physical_slots.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_pcp_compressed_slot_mapping_stays_within_physical_capacity(
    compress_ratio: int,
):
    device = torch.device("cuda")
    logical_block_size = 256
    storage_block_size = logical_block_size // compress_ratio
    num_blocks = 46820
    block_ids = torch.tensor([1, 13441, num_blocks - 1], device=device)
    block_offsets = torch.tensor(
        [compress_ratio - 1, 2 * compress_ratio - 1, 255], device=device
    )
    logical_slots = block_ids * logical_block_size + block_offsets
    positions = torch.tensor(
        [
            compress_ratio - 1,
            32768 + 2 * compress_ratio - 1,
            131072 + 255,
        ],
        device=device,
    )
    logical_slots = torch.cat((logical_slots, torch.tensor([-1], device=device)))
    positions = torch.cat((positions, torch.tensor([0], device=device)))

    physical_slots = get_pcp_compressed_slot_mapping(
        logical_slots,
        positions,
        logical_block_size,
        storage_block_size,
        compress_ratio,
    )

    expected = torch.cat(
        (
            block_ids * storage_block_size + block_offsets // compress_ratio,
            torch.tensor([-1], device=device),
        )
    )
    torch.testing.assert_close(physical_slots, expected)
    assert int(physical_slots[:-1].max()) < num_blocks * storage_block_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_projects_pcp_logical_slots_per_layer():
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024, max_num_batched_tokens=8)
    vllm_config.parallel_config.prefill_context_parallel_size = 2
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    positions = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], device=device)
    logical_slots = 256 + positions
    pcp_metadata = PCPAttentionMetadata(
        rank=0,
        world_size=2,
        local_num_tokens_padded=4,
        positions=positions,
        token_to_req_indices=torch.zeros(8, dtype=torch.int32, device=device),
        block_table_tensor=torch.tensor([[1]], dtype=torch.int32, device=device),
        cache_slot_mapping=logical_slots,
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=4,
        block_table_tensor=torch.tensor([[1]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(256, 260, dtype=torch.int64, device=device),
        causal=True,
        pcp_metadata=pcp_metadata,
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([-1, -1, -1, 65, -1, 64, -1, -1], device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # storage_block_size = block_size // compress_ratio = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.storage_block_size  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)
