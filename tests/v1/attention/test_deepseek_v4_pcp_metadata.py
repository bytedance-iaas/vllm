# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v4.compressor import (
    CompressorBackend,
    CompressorMetadataBuilder,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLABackend
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata, PCPAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _make_common_metadata(
    *,
    slot_mapping: torch.Tensor,
    pcp_metadata: PCPAttentionMetadata | None = None,
) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.tensor([[3]], dtype=torch.int32),
        slot_mapping=slot_mapping,
        causal=True,
        pcp_metadata=pcp_metadata,
    )


def _make_pcp_metadata() -> PCPAttentionMetadata:
    return PCPAttentionMetadata(
        rank=1,
        world_size=2,
        local_num_tokens_padded=3,
        positions=torch.tensor([0, 1, 0, 2, 3, 4], dtype=torch.int64),
        token_to_req_indices=torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[11, 12]], dtype=torch.int32),
        cache_slot_mapping=torch.tensor([20, 21, -1, 22, 23, 24]),
    )


def test_pcp_metadata_selects_rank_local_slot_mapping() -> None:
    pcp_metadata = _make_pcp_metadata()

    torch.testing.assert_close(
        pcp_metadata.local_cache_slot_mapping(),
        torch.tensor([22, 23, 24]),
    )


def test_dsv4_backends_advertise_pcp_only_after_cache_replication_support() -> None:
    assert CompressorBackend.supports_pcp()
    assert DeepseekSparseSWABackend.supports_pcp()
    assert DeepseekV4FlashMLABackend.supports_pcp()


def test_compressor_builder_uses_complete_pcp_cache_write_view() -> None:
    pcp_metadata = _make_pcp_metadata()
    common = _make_common_metadata(
        slot_mapping=torch.tensor([1, 2, 3, 4, 5, 6]),
        pcp_metadata=pcp_metadata,
    )
    common._token_to_req_indices_cache = torch.tensor([0, 0], dtype=torch.int32)
    builder = object.__new__(CompressorMetadataBuilder)
    builder.block_size = 4
    builder.token_to_req_indices = torch.empty(8, dtype=torch.int32)

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert metadata.block_table.data_ptr() == pcp_metadata.block_table_tensor.data_ptr()
    assert (
        metadata.slot_mapping.data_ptr() == pcp_metadata.cache_slot_mapping.data_ptr()
    )
    assert (
        metadata.token_to_req_indices.data_ptr()
        == pcp_metadata.token_to_req_indices.data_ptr()
    )
    assert metadata.positions is not None
    assert metadata.positions.data_ptr() == pcp_metadata.positions.data_ptr()


def test_flashmla_builder_uses_precomputed_pcp_compressed_cache_mapping() -> None:
    pcp_metadata = _make_pcp_metadata()
    common = _make_common_metadata(
        slot_mapping=torch.tensor([1, 2, 3, 4, 5, 6]),
        pcp_metadata=pcp_metadata,
    )
    common._token_to_req_indices_cache = torch.tensor([0, 0], dtype=torch.int32)
    builder = object.__new__(DeepseekV4FlashMLAMetadataBuilder)
    builder.compress_ratio = 4
    builder.topk_tokens = 16
    builder.req_id_per_token_buffer = torch.empty(8, dtype=torch.int32)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        compress_ratio=4,
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert (
        metadata.slot_mapping.data_ptr() == pcp_metadata.cache_slot_mapping.data_ptr()
    )


def test_pcp_manager_builds_rank_concatenated_compressed_cache_view(
    monkeypatch,
) -> None:
    class FakeBlockTables:
        def gather_block_tables(
            self,
            idx_mapping,
            num_reqs_padded,
            out=None,
            out_ptrs=None,
        ):
            assert out is not None
            for group_idx, table in enumerate(out):
                table[:num_reqs_padded].copy_(
                    torch.tensor([[7 + group_idx, 9 + group_idx]])
                )
            return tuple(table[:num_reqs_padded] for table in out)

    def fake_async_copy(x, out=None, device=None):
        source = torch.as_tensor(x)
        if out is None:
            return source.to(device=device)
        return out.copy_(source)

    compressed_calls = []

    def fake_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        storage_block_size,
        compress_ratio,
        *,
        out,
    ):
        compressed_calls.append((storage_block_size, compress_ratio))
        out.fill_(-1)
        base = compress_ratio * 100
        out[:4].copy_(torch.arange(base, base + 4))
        return out[:4]

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu", fake_async_copy
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.get_compressed_slot_mapping",
        fake_compressed_slot_mapping,
    )

    manager = object.__new__(PCPManager)
    manager.pcp_rank = 1
    manager.pcp_world_size = 2
    manager._block_tables = FakeBlockTables()
    manager._kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["swa"],
                kv_cache_spec=SlidingWindowMLASpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                    sliding_window=128,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["c4"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                    compress_ratio=4,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["c128"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                    compress_ratio=128,
                ),
            ),
        ],
    )
    manager._global_batch = type(
        "GlobalBatch",
        (),
        {
            "idx_mapping": torch.tensor([0], dtype=torch.int32),
            "num_reqs": 1,
            "num_tokens": 4,
            "positions": torch.tensor([0, 1, 2, 3]),
            "query_start_loc": torch.tensor([0, 4], dtype=torch.int32),
            "query_start_loc_np": torch.tensor([0, 4], dtype=torch.int32).numpy(),
            "seq_lens": torch.tensor([4], dtype=torch.int32),
        },
    )()
    manager._global_block_tables = tuple(
        torch.zeros((1, 2), dtype=torch.int32) for _ in range(3)
    )
    manager._global_block_table_ptrs = torch.zeros(3, dtype=torch.uint64)
    manager._padded_gather_idx = torch.tensor([0, 3, 1, 0])
    manager._gathered_kv_write_mask = torch.tensor([True, True, True, False])
    manager._pcp_cache_slot_mappings = torch.empty((3, 4), dtype=torch.int64)
    manager._global_compressed_slot_mapping = torch.empty(4, dtype=torch.int64)
    manager._global_token_to_req_indices = torch.empty(4, dtype=torch.int32)
    manager._expanded_token_to_req_indices = torch.empty(4, dtype=torch.int32)
    manager._expanded_positions = torch.empty(4, dtype=torch.int64)
    manager._pad_slot_id = torch.tensor(-1, dtype=torch.int64)

    metadata = manager._prepare_attention_metadata(
        torch.tensor(
            [
                [10, 13, 11, -1],
                [20, 23, 21, -1],
                [30, 33, 31, -1],
            ],
            dtype=torch.int64,
        )
    )

    torch.testing.assert_close(metadata[0].positions, torch.tensor([0, 3, 1, 0]))
    torch.testing.assert_close(
        metadata[0].cache_slot_mapping, torch.tensor([10, 13, 11, -1])
    )
    torch.testing.assert_close(
        metadata[1].cache_slot_mapping, torch.tensor([400, 403, 401, -1])
    )
    torch.testing.assert_close(
        metadata[2].cache_slot_mapping, torch.tensor([12800, 12803, 12801, -1])
    )
    torch.testing.assert_close(
        metadata[0].local_cache_slot_mapping(), torch.tensor([11, -1])
    )
    torch.testing.assert_close(
        metadata[1].block_table_tensor, torch.tensor([[8, 10]], dtype=torch.int32)
    )
    assert compressed_calls == [(64, 4), (2, 128)]


def test_pcp_manager_dual_chunk_layout_keeps_position_jumps_in_separate_rows(
    monkeypatch,
) -> None:
    def fake_async_copy(x, out=None, device=None):
        source = torch.as_tensor(x)
        if out is None:
            return source.to(device=device)
        return out.copy_(source)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu", fake_async_copy
    )
    manager = object.__new__(PCPManager)
    manager.pcp_world_size = 2
    manager.device = torch.device("cpu")

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([15], dtype=np.int32),
        num_computed_tokens=np.array([0], dtype=np.int32),
        is_prefilling=np.array([True]),
        query_start_loc_np=np.array([0, 15], dtype=np.int32),
    )

    assert per_rank_num_tokens == [7, 8]
    assert [segment.global_batch_slice for segment in segments_by_rank[0]] == [
        slice(12, 15),
        slice(0, 4),
    ]
    assert [segment.global_batch_slice for segment in segments_by_rank[1]] == [
        slice(4, 8),
        slice(8, 12),
    ]
    torch.testing.assert_close(
        manager._padded_gather_idx,
        torch.tensor([12, 13, 14, 0, 1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11]),
    )
    torch.testing.assert_close(
        manager._gathered_kv_write_mask,
        torch.tensor([True] * 7 + [False] + [True] * 8),
    )


def test_pcp_manager_continued_prefill_layout_at_160k_boundary(
    monkeypatch,
) -> None:
    def fake_async_copy(x, out=None, device=None):
        source = torch.as_tensor(x)
        if out is None:
            return source.to(device=device)
        return out.copy_(source)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu", fake_async_copy
    )
    manager = object.__new__(PCPManager)
    manager.pcp_world_size = 2
    manager.device = torch.device("cpu")

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.array([32768], dtype=np.int32),
        num_computed_tokens=np.array([131072], dtype=np.int32),
        is_prefilling=np.array([True]),
        query_start_loc_np=np.array([0, 32768], dtype=np.int32),
    )

    assert per_rank_num_tokens == [16384, 16384]
    assert [segment.global_batch_slice for segment in segments_by_rank[0]] == [
        slice(0, 8192),
        slice(24576, 32768),
    ]
    assert [segment.global_batch_slice for segment in segments_by_rank[1]] == [
        slice(8192, 16384),
        slice(16384, 24576),
    ]
    starts_by_rank = [
        [131072 + segment.global_batch_slice.start for segment in rank_segments]
        for rank_segments in segments_by_rank
    ]
    assert starts_by_rank == [[131072, 155648], [139264, 147456]]
    expected_gather_idx = torch.cat(
        (
            torch.arange(0, 8192),
            torch.arange(24576, 32768),
            torch.arange(8192, 24576),
        )
    )
    torch.testing.assert_close(manager._padded_gather_idx, expected_gather_idx)
    torch.testing.assert_close(
        manager._gathered_kv_write_mask,
        torch.ones(32768, dtype=torch.bool),
    )


def test_pcp_manager_builds_complete_padding_only_dummy_cache_view() -> None:
    manager = object.__new__(PCPManager)
    manager.pcp_rank = 1
    manager.pcp_world_size = 2
    manager._expanded_token_to_req_indices = torch.empty(8, dtype=torch.int32)
    manager._expanded_positions = torch.empty(8, dtype=torch.int64)
    block_tables = (torch.tensor([[7, 9]], dtype=torch.int32),)
    slot_mappings = torch.full((1, 8), -1, dtype=torch.int64)

    metadata = manager._prepare_dummy_attention_metadata(
        num_local_tokens=4,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
    )[0]

    assert metadata.local_num_tokens_padded == 4
    torch.testing.assert_close(metadata.positions, torch.zeros(8, dtype=torch.int64))
    torch.testing.assert_close(
        metadata.token_to_req_indices, torch.zeros(8, dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cache_slot_mapping, torch.full((8,), -1, dtype=torch.int64)
    )
    assert metadata.block_table_tensor.data_ptr() == block_tables[0].data_ptr()


def test_dsv4_pcp_rejects_mixed_prefill_decode_batch() -> None:
    manager = object.__new__(PCPManager)
    manager._req_states = object()
    manager._input_buffers = object()
    manager.requires_pure_prefill = True
    input_batch = type(
        "MixedBatch",
        (),
        {
            "num_draft_tokens": 0,
            "is_prefilling_np": np.array([True, False]),
        },
    )()

    with pytest.raises(NotImplementedError, match="pure prefill batches only"):
        manager.partition_batch(input_batch)


def _make_dsv4_pcp_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v4",
                index_topk=2048,
            ),
            use_mla=True,
            is_encoder_decoder=False,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_ds_mla",
            enable_prefix_caching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        lora_config=None,
        speculative_config=None,
    )


@pytest.mark.parametrize(
    ("config_path", "value", "match"),
    [
        ("parallel_config.decode_context_parallel_size", 2, "requires DCP=1"),
        ("parallel_config.data_parallel_size", 2, "requires DP=1"),
        ("parallel_config.pipeline_parallel_size", 2, "does not support PP"),
        ("cache_config.cache_dtype", "bfloat16", "fp8_ds_mla"),
        ("cache_config.enable_prefix_caching", True, "prefix caching off"),
        ("compilation_config.cudagraph_mode", CUDAGraphMode.FULL, "CUDA graphs"),
    ],
)
def test_dsv4_pcp_config_support_matrix_fails_closed(
    config_path: str,
    value,
    match: str,
) -> None:
    config = _make_dsv4_pcp_config()
    owner = config
    path = config_path.split(".")
    for name in path[:-1]:
        owner = getattr(owner, name)
    setattr(owner, path[-1], value)

    with pytest.raises(NotImplementedError, match=match):
        PCPManager.validate_config(config, supports_mm_inputs=False)


def test_dsv4_pcp_config_accepts_frozen_initial_support_matrix() -> None:
    PCPManager.validate_config(_make_dsv4_pcp_config(), supports_mm_inputs=False)


def test_build_attn_metadata_propagates_group_specific_pcp_cache_view() -> None:
    class RecordingBuilder:
        def build(self, common_prefix_len, common_attn_metadata):
            assert common_prefix_len == 0
            return common_attn_metadata

    class RecordingGroup:
        layer_names = ["layer"]

        def get_metadata_builder(self, virtual_engine):
            assert virtual_engine == 0
            return RecordingBuilder()

    pcp_metadata = _make_pcp_metadata()
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                ),
            )
        ],
    )

    metadata = build_attn_metadata(
        attn_groups=[[RecordingGroup()]],
        num_reqs=1,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        max_seq_len=2,
        block_tables=(torch.tensor([[3]], dtype=torch.int32),),
        slot_mappings=torch.tensor([[1, 2]], dtype=torch.int64),
        kv_cache_config=kv_cache_config,
        pcp_attn_metadata=(pcp_metadata,),
    )

    assert metadata["layer"].pcp_metadata is pcp_metadata
