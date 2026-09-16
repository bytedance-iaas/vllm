# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import (
    indexer_kv_cache_as_quant_view,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4SparseMLABackend as DeepseekV41SparseMLABackend,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    CompressedSlotMappingKernel,
)
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
    kpool_flat_page_view,
    kpool_page_geometry,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    ConvertReqIndexToGlobalIndexKernel,
)
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    compute_layer_kv_cache_shape_bytes,
    create_kv_cache_views,
)
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import select_common_block_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_lens", [[1], [6] * 16, [0, 2, 6, 0, 4], [0, 0, 0]])
@pytest.mark.parametrize("padding", [0, 17])
def test_fused_indexer_decode_metadata(query_lens, padding):
    """Flatten device query boundaries and clear graph padding on every replay."""
    from vllm.v1.attention.ops.metadata import _indexer_decode_metadata_kernel

    device = "cuda"
    lengths = torch.tensor(query_lens, device=device, dtype=torch.int32)
    qsl = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32), lengths.cumsum(0).int()]
    )
    reqs = len(query_lens)
    n = sum(query_lens)
    tokens = n + padding
    capacity = max(tokens, reqs) + 31
    seq = lengths + 4096
    # Exercise block-table row padding and masked columns.
    bt = torch.arange(reqs * 74, device=device, dtype=torch.int32).view(reqs, 74)[
        :, :65
    ]
    out_bt = torch.full((capacity, 69), -99, device=device, dtype=torch.int32)
    outputs = [
        torch.full((capacity,), -99, device=device, dtype=torch.int32) for _ in range(4)
    ]
    out_seq, out_lens, indices, per_req = outputs
    grid = max(reqs, tokens + (capacity - tokens + 255) // 256)
    _indexer_decode_metadata_kernel[(grid,)](
        qsl,
        seq,
        bt,
        out_seq,
        out_bt,
        out_lens,
        indices,
        per_req,
        reqs,
        tokens,
        capacity,
        bt.stride(0),
        out_bt.stride(0),
        BLOCK_COLS=bt.shape[1],
        num_warps=4,
    )
    expected_req = torch.repeat_interleave(torch.arange(reqs, device=device), lengths)
    expected_seq = torch.cat(
        [torch.arange(4097, 4097 + q, device=device) for q in query_lens]
    )
    torch.testing.assert_close(out_seq[:n], expected_seq.int(), rtol=0, atol=0)
    assert torch.count_nonzero(out_seq[n:]) == 0
    torch.testing.assert_close(out_bt[:n, :65], bt[expected_req], rtol=0, atol=0)
    assert torch.count_nonzero(out_bt[n:tokens, :65]) == 0
    torch.testing.assert_close(indices[:n], expected_req.int(), rtol=0, atol=0)
    torch.testing.assert_close(
        indices[n:tokens],
        torch.arange(reqs, reqs + padding, device=device, dtype=torch.int32),
    )
    assert torch.all(out_lens[:tokens] == 1)
    torch.testing.assert_close(per_req[:reqs], lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_lens", [[1], [0, 257, 1, 0, 3], [6] * 64, [2, 6, 4, 0]])
def test_device_token_request_mapping(query_lens):
    """Graph replay follows device boundaries even when CPU lengths are stale."""
    lengths = torch.tensor(query_lens, device="cuda", dtype=torch.int32)
    qsl = torch.cat(
        [torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int()]
    )
    n = sum(query_lens)
    output = torch.full((n + 7,), -99, device="cuda", dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=lengths,
        num_reqs=len(query_lens),
        num_actual_tokens=output.numel(),
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens),
        block_table_tensor=torch.empty(
            (len(query_lens), 1), device="cuda", dtype=torch.int32
        ),
        slot_mapping=torch.full((n + 7,), -1, device="cuda", dtype=torch.int64),
    )
    result = common.token_to_req_indices(output)
    assert result.data_ptr() == output.data_ptr()
    expected = torch.repeat_interleave(
        torch.arange(len(query_lens), device="cuda", dtype=torch.int32), lengths
    )
    torch.testing.assert_close(output[:n], expected)
    assert torch.count_nonzero(output[n:]) == 0
    graph = torch.cuda.CUDAGraph()
    common._token_to_req_indices_cache = None
    with torch.cuda.graph(graph):
        common.token_to_req_indices(output)
    reversed_lens = lengths.flip(0)
    qsl[1:].copy_(reversed_lens.cumsum(0))
    graph.replay()
    expected = torch.repeat_interleave(
        torch.arange(len(query_lens), device="cuda", dtype=torch.int32), reversed_lens
    )
    torch.testing.assert_close(output[:n], expected)
    assert torch.count_nonzero(output[n:]) == 0


def test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla():
    """Packed MLA/indexer groups must retain 64 compressed rows per page."""
    kernel_block_size = select_common_block_size(
        256, [DeepseekV4SparseMLABackend, DeepseekV4IndexerBackend]
    )
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    assert compute_layer_kv_cache_shape_bytes(spec, 2, kernel_block_size) == (
        2,
        1,
        64,
        132,
    )


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_indexer_uses_64_state_pages_for_deepseek_v41_sm90(monkeypatch, compress_ratio):
    """V4.1 keeps one manager block and re-pages only when necessary."""
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 90
    )

    manager_block_size = DeepseekV41SparseMLABackend.get_preferred_block_size(16)
    kernel_block_size = select_common_block_size(
        manager_block_size,
        [DeepseekV41SparseMLABackend, DeepseekV41IndexerBackend],
    )
    spec = MLAAttentionSpec(
        block_size=manager_block_size,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
        alignment=576,
        kernel_page_rows=64,
    )

    assert manager_block_size == 128
    assert kernel_block_size == 128
    assert compute_layer_kv_cache_shape_bytes(spec, num_blocks=2)[:3] == (
        2,
        1,
        128 // compress_ratio,
    )
    assert kpool_page_geometry(
        spec.num_states, None, spec.state_content_size_bytes, spec.kernel_page_rows
    )[:2] == (64, 2 // compress_ratio)


def test_indexer_repaging_preserves_padded_manager_block_stride():
    """The ratio-1 page view must skip alignment padding between blocks."""
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=576,
        kernel_page_rows=64,
    )
    page_bytes = 64 * spec.state_content_size_bytes
    block_stride = 5 * page_bytes
    raw = torch.arange(2 * block_stride + spec.page_size_bytes, dtype=torch.int32).to(
        torch.uint8
    )
    tensor = KVCacheTensor(
        size=raw.numel(),
        layers=["indexer"],
        layer_stride=spec.page_size_bytes,
        block_stride=block_stride,
    )
    cache = create_kv_cache_views(raw, spec, 3, KVCacheLayout.BLHNC, tensor)[0].squeeze(
        1
    )

    pages = kpool_flat_page_view(cache, spec.kernel_page_rows)

    stride_pages = block_stride // page_bytes
    assert pages.shape == ((3 - 1) * stride_pages + 2, 64, 132)
    for block in range(3):
        for page in range(2):
            torch.testing.assert_close(
                pages[block * stride_pages + page],
                cache[block, page * 64 : (page + 1) * 64],
            )


def test_sparse_indexer_quant_view_uses_native_page_rows():
    raw = torch.arange(3 * 5 * 64 * 132, dtype=torch.int32).to(torch.uint8)
    cache = raw.as_strided((3, 128, 132), (5 * 64 * 132, 132, 1))

    quant_view = indexer_kv_cache_as_quant_view(
        cache, head_dim=128, use_fp4_cache=False, kernel_page_rows=64
    )

    assert quant_view.shape == (12, 64, 1, 132)
    for block in range(3):
        for page in range(2):
            torch.testing.assert_close(
                quant_view[block * 5 + page, :, 0],
                cache[block, page * 64 : (page + 1) * 64],
            )


def test_indexer_native_page_alignment_applies_to_packed_block_stride():
    ratio1 = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=576,
        kernel_page_rows=64,
    )
    ratio2 = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=2,
        alignment=576,
        kernel_page_rows=64,
    )
    specs = {"ratio1": ratio1, "ratio2": ratio2}
    groups = [
        KVCacheGroupSpec(
            list(specs),
            UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs),
        )
    ]

    dense = ratio1.page_size_bytes + ratio2.page_size_bytes
    packed = _get_kv_cache_bytes_per_block(groups, KVCacheLayout.BLHNC)

    native_page_bytes = 64 * ratio1.state_content_size_bytes
    assert packed >= dense
    assert packed % native_page_bytes == 0


def test_indexer_page_table_matches_packed_page_addresses():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        kernel_page_rows=64,
    )
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = spec
    builder.block_stride_bytes = 5 * 64 * 132
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)
    block_table = torch.tensor([[3, 7]], dtype=torch.int32)

    page_states, pages = builder._indexer_page_table(block_table)

    assert page_states == 64
    assert pages.tolist() == [[15, 16, 35, 36]]


def test_ratio2_indexer_page_table_remains_manager_block_table():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=2,
        kernel_page_rows=64,
    )
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = spec
    builder.block_stride_bytes = 5 * 64 * 132
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)
    block_table = torch.tensor([[3, 7]], dtype=torch.int32)

    page_states, pages = builder._indexer_page_table(block_table)

    assert page_states == 64
    assert pages is block_table


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                compress_ratios=[0, 0, 4, 128, 0], index_kpool=32
            )
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.compress_ratio for key in keys} == {1, 4, 32, 128}
    assert {(key.query_slice_start, key.query_slice_stop) for key in keys} == {
        (query_slice_start, query_slice_stop)
        for query_slice_start in (1, 2, 16)
        for query_slice_stop in (1, 2, 16)
    }


def test_compressed_slot_mapping_warmup_includes_index_kpool():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=32)),
    )

    keys = CompressedSlotMappingKernel().get_warmup_keys(config)
    assert {(key.compress_ratio, key.block_size) for key in keys} == {(32, 2)}


def test_index_conversion_warmup_uses_physical_block_stride():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_text_config=SimpleNamespace(index_topk=2048),
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = ConvertReqIndexToGlobalIndexKernel().get_warmup_keys(
        config,
        block_stride_rows=4096,
    )
    assert {key.block_stride_rows for key in keys} == {4096}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # num_states = block_size // tokens_per_state = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
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

    storage_bs = kv_cache_spec.num_states  # 64
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
