# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

from tests.v1.attention.utils import create_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import (
    can_use_indexer_topk_page_table_fusion,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "enabled",
        "num_tokens",
        "num_seqs",
        "requires_padding",
        "has_global_seq_lens",
        "expected",
    ),
    [
        (True, 4, 4, False, False, True),
        (False, 4, 4, False, False, False),
        (True, 5, 5, False, False, False),
        (True, 8, 2, False, False, False),
        (True, 4, 4, True, False, False),
        (True, 4, 4, False, True, False),
    ],
)
def test_indexer_topk_page_table_fusion_gate(
    enabled: bool,
    num_tokens: int,
    num_seqs: int,
    requires_padding: bool,
    has_global_seq_lens: bool,
    expected: bool,
) -> None:
    page_table = torch.empty((4, 8), dtype=torch.int32, device="cuda")
    topk_lens = torch.empty(4, dtype=torch.int32, device="cuda")
    assert (
        can_use_indexer_topk_page_table_fusion(
            enabled=enabled,
            topk_lens_buffer=topk_lens,
            topk_tokens=2048,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            page_table_block_size=64,
            page_table=page_table,
            valid_token_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
            requires_padding=requires_padding,
            has_global_seq_lens=has_global_seq_lens,
        )
        is expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size(
    tmp_path: Path,
):
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
    model_dir = tmp_path / "model"
    LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=2048,
    ).save_pretrained(model_dir)
    vllm_config = create_vllm_config(
        model_name=str(model_dir),
        max_model_len=1024,
    )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "num_tokens",
        "query_start_loc",
        "seq_lens",
        "block_table",
        "use_padded_stride",
        "expected",
    ),
    [
        (4, [0, 4], [2], [[5], [7], [9]], False, [-1, -1, -1, -1]),
        (4, [0, 4], [260], [[5, 999]], True, [-1, -1, -1, -1]),
        (4, [0, 6], [6], [[5]], False, [-1, -1, -1, 320]),
    ],
)
def test_compressed_slot_mapping_ignores_padded_or_out_of_range_rows(
    num_tokens: int,
    query_start_loc: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    use_padded_stride: bool,
    expected: list[int],
):
    device = torch.device("cuda")
    block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device=device)
    if use_padded_stride:
        block_table_tensor = block_table_tensor[:, :1]
    result = get_compressed_slot_mapping(
        num_tokens=num_tokens,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_table=block_table_tensor,
        block_size=64,
        compress_ratio=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        result,
        torch.tensor(expected, dtype=torch.int64, device=device),
    )
