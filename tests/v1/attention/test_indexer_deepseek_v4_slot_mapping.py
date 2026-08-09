# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_module
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _make_decode_common_metadata(seq_lens: list[int], device: torch.device):
    batch_size = len(seq_lens)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens_tensor,
        seq_lens_cpu_upper_bound=seq_lens_tensor.cpu(),
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=max(seq_lens, default=0),
        block_table_tensor=torch.zeros(
            batch_size, 1, dtype=torch.int32, device=device
        ),
        slot_mapping=torch.arange(batch_size, dtype=torch.int64, device=device),
        causal=True,
    )


def _make_indexer_builder(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    *,
    compress_ratio: int,
    block_size: int = 256,
    num_sms: int = 17,
) -> DeepseekV32IndexerMetadataBuilder:
    monkeypatch.setattr(
        indexer_module,
        "num_compute_units",
        lambda _device_id=0: num_sms,
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
    )
    vllm_config = create_vllm_config(max_model_len=1024, block_size=block_size)
    return DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )


def _mock_paged_mqa_metadata(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, torch.Tensor | int]] = []

    monkeypatch.setattr(indexer_module, "has_deep_gemm", lambda: True)

    def fake_get_paged_mqa_logits_metadata(
        seq_lens: torch.Tensor,
        block_size: int,
        num_sms: int,
    ) -> torch.Tensor:
        calls.append(
            {
                "seq_lens": seq_lens.detach().clone(),
                "block_size": block_size,
                "num_sms": num_sms,
            }
        )
        return torch.full(
            (num_sms + 1, 2),
            7,
            dtype=torch.int32,
            device=seq_lens.device,
        )

    monkeypatch.setattr(
        indexer_module,
        "get_paged_mqa_logits_metadata",
        fake_get_paged_mqa_logits_metadata,
    )
    return calls


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [32, 64, 96])
def test_indexer_builder_capture_build_uses_metadata_ones_for_compressed_decodes(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
):
    device = torch.device("cuda")
    builder = _make_indexer_builder(monkeypatch, device, compress_ratio=4)
    metadata_calls = _mock_paged_mqa_metadata(monkeypatch)
    common = _make_decode_common_metadata([1] * batch_size, device)

    md = builder.build_for_cudagraph_capture(common)

    assert md.decode is not None
    assert len(metadata_calls) == 1
    torch.testing.assert_close(
        metadata_calls[0]["seq_lens"],
        torch.ones((batch_size, 1), dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        common.seq_lens,
        torch.ones(batch_size, dtype=torch.int32, device=device),
    )
    assert metadata_calls[0]["block_size"] == 64
    assert metadata_calls[0]["num_sms"] == builder.num_sms == 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_compressed_dummy_mask_preserves_real_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    builder = _make_indexer_builder(monkeypatch, device, compress_ratio=4)
    metadata_calls = _mock_paged_mqa_metadata(monkeypatch)
    common = _make_decode_common_metadata([0, 1, 3, 4, 8], device)

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert md.decode is not None
    assert len(metadata_calls) == 1
    torch.testing.assert_close(
        metadata_calls[0]["seq_lens"],
        torch.tensor([[1], [0], [0], [1], [2]], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        common.seq_lens,
        torch.tensor([0, 1, 3, 4, 8], dtype=torch.int32, device=device),
    )
    assert metadata_calls[0]["block_size"] == 64
    assert metadata_calls[0]["num_sms"] == builder.num_sms == 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepare_paged_mqa_metadata_seq_lens_none_mask_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    builder = _make_indexer_builder(monkeypatch, device, compress_ratio=1)
    num_decode_tokens = 4
    seq_lens = torch.tensor([3, 5, 7, 9], dtype=torch.int32, device=device)

    warmup = builder._prepare_paged_mqa_metadata_seq_lens(
        seq_lens=seq_lens,
        dummy_decode_mask=None,
        num_decode_tokens=num_decode_tokens,
        seq_lens_is_buffer_view=False,
    )
    torch.testing.assert_close(
        warmup,
        torch.ones(num_decode_tokens, dtype=torch.int32, device=device),
    )
    assert warmup.data_ptr() == builder.decode_seq_lens_buffer.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = builder._prepare_paged_mqa_metadata_seq_lens(
            seq_lens=seq_lens,
            dummy_decode_mask=None,
            num_decode_tokens=num_decode_tokens,
            seq_lens_is_buffer_view=False,
        )

    seq_lens.copy_(torch.tensor([11, 13, 15, 17], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.ones(num_decode_tokens, dtype=torch.int32, device=device)
    torch.testing.assert_close(captured, expected)
    torch.testing.assert_close(
        builder.decode_seq_lens_buffer[:num_decode_tokens],
        expected,
    )
    assert captured.data_ptr() == builder.decode_seq_lens_buffer.data_ptr()
