# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.models.deepseek_v4.attention as attention_module
import vllm.models.deepseek_v4.nvidia.flashmla as flashmla_module
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.compressor import (
    CompressorBackend,
    CompressorMetadataBuilder,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.pcp_metadata import DeepseekV4PcpPrefillMetadata
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.worker.cp_utils import (
    PCPManager,
    build_pcp_interleave_request_views,
)


def test_flashmla_declares_pcp_support():
    assert DeepseekV4FlashMLAAttention.supports_pcp


def test_deepseek_v4_hybrid_backends_declare_pcp_support():
    assert CompressorBackend.supports_pcp()
    assert DeepseekV32IndexerBackend.supports_pcp()
    assert DeepseekSparseSWABackend.supports_pcp()


def test_common_attention_metadata_unpadded_preserves_pcp_batch_view():
    request_views = [object(), object(), object()]
    restore_idx = torch.tensor([0, 2, 3, 1], dtype=torch.int64)
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        seq_lens=torch.tensor([12, 24, 36], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=6,
        max_query_len=2,
        max_seq_len=36,
        block_table_tensor=torch.arange(12).reshape(3, 4),
        slot_mapping=torch.arange(6),
        pcp_allgather_restore_idx=restore_idx,
        pcp_full_seq_lens=torch.tensor([12, 24, 36], dtype=torch.int32),
        pcp_full_seq_lens_cpu=torch.tensor([12, 24, 36], dtype=torch.int32),
        pcp_request_views=request_views,
        positions=torch.arange(6),
    )

    unpadded = metadata.unpadded(num_actual_tokens=4, num_actual_reqs=2)

    assert unpadded.pcp_allgather_restore_idx is restore_idx
    torch.testing.assert_close(
        unpadded.pcp_full_seq_lens, torch.tensor([12, 24], dtype=torch.int32)
    )
    torch.testing.assert_close(
        unpadded.pcp_full_seq_lens_cpu,
        torch.tensor([12, 24], dtype=torch.int32),
    )
    assert unpadded.pcp_request_views == request_views[:2]
    torch.testing.assert_close(unpadded.positions, torch.arange(4))


def test_pcp_request_views_follow_dual_chunk_manager_layout():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )

    pcp_tokens, _ = manager.update_tokens_for_pcp(
        np.array([1, 5, 8], dtype=np.int32),
        np.arange(64, dtype=np.int32),
        num_reqs=3,
        reorder_batch_threshold=1,
    )

    assert pcp_tokens.tolist() == [1, 4, 4]
    views = manager.pcp_request_views
    assert [view.req_idx for view in views] == [0, 1, 2]
    assert [view.global_seq_len for view in views] == [1, 5, 8]
    assert [view.local_token_count for view in views] == [1, 2, 4]
    assert [(view.local_query_start, view.local_query_end) for view in views] == [
        (0, 1),
        (1, 3),
        (3, 7),
    ]
    torch.testing.assert_close(views[0].global_positions, torch.tensor([0]))
    torch.testing.assert_close(views[1].global_positions, torch.tensor([0, 1]))
    torch.testing.assert_close(views[2].global_positions, torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(views[1].global_slot_mapping, torch.tensor([1, 2]))
    torch.testing.assert_close(
        views[2].global_slot_mapping, torch.tensor([6, 7, 12, 13])
    )
    assert [view.restore_idx.numel() for view in views] == [2, 8, 8]
    assert [(view.local_kv_base, view.local_kv_len) for view in views] == [
        (0, 1),
        (1, 2),
        (3, 4),
    ]


def test_build_pcp_request_views_preserves_explicit_global_slot_identity():
    views = build_pcp_interleave_request_views(
        original_token_counts=torch.tensor([4, 4]),
        local_token_counts=torch.tensor([2, 2]),
        local_positions=torch.tensor([0, 3, 1, 2]),
        restore_idx=torch.tensor([0, 2, 3, 1, 4, 6, 7, 5]),
        pcp_world_size=2,
        global_slot_mapping=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
    )

    assert len(views) == 2
    torch.testing.assert_close(views[0].global_positions, torch.tensor([0, 3]))
    torch.testing.assert_close(views[0].global_slot_mapping, torch.tensor([10, 13]))
    torch.testing.assert_close(views[0].restore_idx, torch.tensor([0, 2, 3, 1]))
    torch.testing.assert_close(views[1].global_positions, torch.tensor([1, 2]))
    torch.testing.assert_close(views[1].global_slot_mapping, torch.tensor([21, 22]))
    torch.testing.assert_close(views[1].restore_idx, torch.tensor([4, 6, 7, 5]))


def test_pcp_manager_uses_distinct_slot_mapping_buffers_per_kv_group():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=16,
        max_num_reqs=4,
        device=torch.device("cpu"),
    )

    gid0_slots = manager.get_pcp_padded_slot_mapping(0)
    gid1_slots = manager.get_pcp_padded_slot_mapping(1)

    assert gid0_slots.data_ptr() != gid1_slots.data_ptr()
    gid0_slots[:4] = torch.tensor([0, 1, 2, 3])
    gid1_slots[:4] = torch.tensor([10, 11, 12, 13])
    torch.testing.assert_close(gid0_slots[:4], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(gid1_slots[:4], torch.tensor([10, 11, 12, 13]))
    assert manager.get_pcp_padded_slot_mapping(1).data_ptr() == gid1_slots.data_ptr()


def test_pcp_manager_builds_full_query_start_for_restored_tokens():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=8,
        device=torch.device("cpu"),
    )

    pcp_tokens, _ = manager.update_tokens_for_pcp(
        np.array([1, 5, 8], dtype=np.int32),
        np.arange(64, dtype=np.int32),
        num_reqs=3,
        reorder_batch_threshold=1,
    )

    assert pcp_tokens.tolist() == [1, 4, 4]
    torch.testing.assert_close(
        manager.pcp_padded_query_start_loc.cpu[:4],
        torch.tensor([0, 2, 10, 18], dtype=torch.int32),
    )


def test_pcp_194_local_tokens_restore_to_388_full_tokens():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=512,
        max_num_reqs=1,
        device=torch.device("cpu"),
    )

    pcp_tokens, pcp_positions = manager.update_tokens_for_pcp(
        np.array([388], dtype=np.int32),
        np.arange(512, dtype=np.int32),
        num_reqs=1,
        reorder_batch_threshold=1,
    )

    assert pcp_tokens.tolist() == [194]
    assert pcp_positions[:194].tolist() == list(range(97)) + list(range(291, 388))
    assert manager.pcp_allgather_restore_idx.cpu[:388].numel() == 388
    assert manager.pcp_local_unpad_mask_cpu_tensor[:194].all()

    views = manager.pcp_request_views
    assert len(views) == 1
    assert views[0].global_seq_len == 388
    assert views[0].local_token_count == 194
    assert views[0].local_query_start == 0
    assert views[0].local_query_end == 194
    torch.testing.assert_close(
        views[0].global_positions,
        torch.tensor(list(range(97)) + list(range(291, 388))),
    )


def test_sparse_swa_builder_creates_global_prefill_pcp_view():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=4,
        device=torch.device("cpu"),
    )
    local_counts, local_positions = manager.update_tokens_for_pcp(
        np.array([8], dtype=np.int32),
        np.arange(64, dtype=np.int32),
        num_reqs=1,
        reorder_batch_threshold=1,
    )
    assert local_counts.tolist() == [4]

    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder.window_size = 4
    builder.max_model_len = 128
    builder.max_num_batched_tokens = 64
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    slot_mapping = torch.tensor([10, 11, 16, 17], dtype=torch.int64)
    restore_idx = manager.pcp_allgather_restore_idx.cpu[:8]

    fields = builder._build_deepseek_v4_metadata(
        num_decodes=0,
        num_prefills=1,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        slot_mapping=slot_mapping,
        positions=torch.from_numpy(local_positions[:4].copy()),
        pcp_allgather_restore_idx=restore_idx,
        pcp_request_views=manager.pcp_request_views,
        pcp_enabled=True,
    )

    torch.testing.assert_close(
        fields["prefill_seq_lens"], torch.tensor([8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        fields["prefill_gather_lens"], torch.tensor([8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        fields["prefill_query_lens_cpu"], torch.tensor([4], dtype=torch.int32)
    )
    pcp_metadata = fields["pcp_prefill_metadata"]
    assert pcp_metadata.cp_size == 2
    assert pcp_metadata.cp_rank == 0
    torch.testing.assert_close(
        pcp_metadata.local_query_start_loc,
        torch.tensor([0, 4], dtype=torch.int32),
    )
    assert pcp_metadata.views is manager.pcp_request_views
    assert pcp_metadata.restore_idx is restore_idx
    assert pcp_metadata.global_slot_mapping is slot_mapping


def _make_pcp_common_metadata(
    manager: PCPManager,
) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=8,
        block_table_tensor=torch.tensor([[3, 4]], dtype=torch.int32),
        slot_mapping=torch.tensor([10, 11, 16, 17], dtype=torch.int64),
        pcp_allgather_restore_idx=manager.pcp_allgather_restore_idx.cpu[:8],
        pcp_full_seq_lens=torch.tensor([8], dtype=torch.int32),
        pcp_full_seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        pcp_request_views=manager.pcp_request_views,
    )


def test_compressor_builder_expands_request_indices_for_restored_pcp_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.compressor.np_to_pinned_tensor",
        torch.from_numpy,
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=64,
        max_num_reqs=4,
        device=torch.device("cpu"),
    )
    manager.update_tokens_for_pcp(
        np.array([8], dtype=np.int32),
        np.arange(64, dtype=np.int32),
        num_reqs=1,
        reorder_batch_threshold=1,
    )
    common = _make_pcp_common_metadata(manager)
    builder = object.__new__(CompressorMetadataBuilder)
    builder.pcp_world_size = 2
    builder.block_size = 4
    builder.token_to_req_indices = torch.full((64,), -1, dtype=torch.int32)

    metadata = builder.build(0, common)

    assert metadata.pcp_allgather_restore_idx is common.pcp_allgather_restore_idx
    assert metadata.pcp_request_views is common.pcp_request_views
    torch.testing.assert_close(
        metadata.token_to_req_indices,
        torch.zeros(8, dtype=torch.int32),
    )


class _TestDeepseekV4Attention(DeepseekV4Attention):
    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(self, q, kv, positions, out) -> None:
        raise NotImplementedError

    def _o_proj(self, o, positions):
        raise NotImplementedError


class _AttentionPCPGroup:
    world_size = 2
    rank_in_group = 0

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        if tensor.ndim == 3:
            other = torch.full_like(tensor, 0)
            other[0].fill_(11)
            other[1].fill_(12)
        elif tensor.ndim == 2:
            other = torch.full_like(tensor, 0)
            other[0].fill_(21)
            other[1].fill_(22)
        else:
            other = torch.tensor([1, 2], dtype=tensor.dtype)
        return torch.cat([tensor, other], dim=dim)


def test_attention_pcp_restores_global_swa_slots_and_returns_local_queries(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        max_buffer_num_tokens=16,
        max_num_reqs=1,
        device=torch.device("cpu"),
    )
    _, local_positions = manager.update_tokens_for_pcp(
        np.array([4], dtype=np.int32),
        np.arange(16, dtype=np.int32),
        num_reqs=1,
        reorder_batch_threshold=1,
    )
    restore_idx = manager.pcp_allgather_restore_idx.cpu[:4]
    pcp_metadata = DeepseekV4PcpPrefillMetadata(
        cp_size=2,
        cp_rank=0,
        strategy="dual_chunk",
        views=manager.pcp_request_views,
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        local_seq_lens=torch.tensor([2], dtype=torch.int32),
        local_swa_indices=None,
        local_swa_valid_lens=None,
        local_c4_indices=None,
        local_c128_indices=None,
        global_slot_mapping=torch.tensor([4, 7], dtype=torch.int64),
        compressor_write_locs_global=None,
        restore_idx=restore_idx,
        debug_global_positions=torch.from_numpy(local_positions[:2].copy()),
    )
    swa_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[1]], dtype=torch.int32),
        slot_mapping=torch.tensor([4, 7], dtype=torch.int64),
        block_size=4,
        pcp_allgather_restore_idx=restore_idx,
        pcp_prefill_metadata=pcp_metadata,
    )

    attention = object.__new__(_TestDeepseekV4Attention)
    object.__setattr__(attention, "pcp_world_size", 2)
    object.__setattr__(attention, "pcp_rank", 0)
    object.__setattr__(attention, "cp_kv_cache_interleave_size", 1)
    object.__setattr__(attention, "n_local_heads", 1)
    object.__setattr__(attention, "padded_heads", 1)
    object.__setattr__(attention, "head_dim", 512)
    object.__setattr__(attention, "eps", 1e-6)
    object.__setattr__(
        attention,
        "swa_cache_layer",
        SimpleNamespace(
            prefix="swa",
            block_size=4,
            kv_cache=torch.zeros((2, 4, 512), dtype=torch.bfloat16),
        ),
    )
    object.__setattr__(
        attention,
        "rotary_emb",
        SimpleNamespace(cos_sin_cache=torch.zeros((8, 64))),
    )

    captured = {}

    def fake_bf16_insert(q, kv, cache, slots, positions, *args):
        captured["q"] = q.clone()
        captured["kv"] = kv.clone()
        captured["slots"] = slots.clone()
        captured["positions"] = positions.clone()
        q.add_(100)

    monkeypatch.setattr(
        attention_module,
        "get_pcp_group",
        lambda: _AttentionPCPGroup(),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
        fake_bf16_insert,
        raising=False,
    )

    q = torch.empty((2, 1, 512), dtype=torch.bfloat16)
    q[0].fill_(10)
    q[1].fill_(13)
    kv = torch.empty((2, 512), dtype=torch.bfloat16)
    kv[0].fill_(20)
    kv[1].fill_(23)
    result = attention._fused_qnorm_rope_kv_insert(
        q,
        kv,
        torch.from_numpy(local_positions[:2].copy()).to(torch.int64),
        {"swa": swa_metadata},
    )

    torch.testing.assert_close(
        captured["q"][:, 0, 0],
        torch.tensor([10, 11, 12, 13], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        captured["kv"][:, 0],
        torch.tensor([20, 21, 22, 23], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(captured["positions"], torch.arange(4))
    torch.testing.assert_close(captured["slots"], torch.tensor([4, -1, 5, -1]))
    torch.testing.assert_close(
        result[:, 0, 0],
        torch.tensor([110, 113], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(pcp_metadata.restored_swa_positions, torch.arange(4))
    assert pcp_metadata.restored_swa_valid_mask is not None
    assert pcp_metadata.restored_swa_valid_mask.all()


class _TestWorkspaceManager:
    def get_simultaneous(self, *specs):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in specs]


def test_flashmla_prefill_compacts_queries_and_uses_global_kv_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    pcp_metadata = DeepseekV4PcpPrefillMetadata(
        cp_size=2,
        cp_rank=0,
        strategy="dual_chunk",
        views=[SimpleNamespace(restore_idx=torch.arange(4))],
        local_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        local_seq_lens=torch.tensor([2], dtype=torch.int32),
        local_swa_indices=None,
        local_swa_valid_lens=None,
        local_c4_indices=None,
        local_c128_indices=None,
        global_slot_mapping=torch.tensor([4, 7]),
        compressor_write_locs_global=None,
        restore_idx=torch.tensor([0, 2, 3, 1]),
        debug_global_positions=torch.tensor([0, 3]),
        restored_swa_kv=torch.tensor(
            [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0], [23.0, 23.0]],
            dtype=torch.bfloat16,
        ),
        restored_swa_positions=torch.arange(4),
        restored_swa_valid_mask=torch.ones(4, dtype=torch.bool),
    )
    swa_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 3]),
        block_size=4,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        num_prefills=1,
        num_prefill_tokens=2,
        prefill_seq_lens=torch.tensor([4], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        prefill_gather_lens=torch.tensor([4], dtype=torch.int32),
        prefill_gather_lens_cpu=torch.tensor([4], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([2], dtype=torch.int32),
        prefill_window_size=4,
        prefill_max_model_len=4,
        prefill_max_num_batched_tokens=4,
        pcp_allgather_restore_idx=torch.tensor([0, 2, 3, 1]),
        pcp_prefill_metadata=pcp_metadata,
    )
    attn_metadata = DeepseekV4FlashMLAMetadata(
        num_reqs=1,
        max_query_len=2,
        max_seq_len=4,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 0]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
        block_size=4,
        topk_tokens=1,
    )

    attention = object.__new__(DeepseekV4FlashMLAAttention)
    object.__setattr__(attention, "compress_ratio", 4)
    object.__setattr__(attention, "window_size", 4)
    object.__setattr__(attention, "PREFILL_CHUNK_SIZE", 4)
    object.__setattr__(attention, "scale", 1.0)
    object.__setattr__(attention, "attn_sink", None)
    object.__setattr__(
        attention,
        "topk_indices_buffer",
        torch.tensor([[0], [0]], dtype=torch.int32),
    )

    captured = {}

    def fake_gather(out, *args, offset, **kwargs):
        out.fill_(-1)
        captured.setdefault("gather_offsets", []).append(offset)

    def fake_combine(
        topk_indices,
        query_positions,
        query_start_loc,
        seq_lens,
        gather_lens,
        *args,
    ):
        captured["query_positions"] = query_positions.clone()
        captured["query_start_loc"] = query_start_loc.clone()
        return (
            torch.tensor([[0, -1], [3, -1]], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
        )

    def fake_sparse_fwd(*, q, kv, indices, topk_length, out, **kwargs):
        captured["kernel_q"] = q.clone()
        captured["kernel_kv"] = kv.clone()
        captured["kernel_indices"] = indices.clone()
        captured["kernel_lens"] = topk_length.clone()
        out.copy_(q)

    monkeypatch.setattr(
        flashmla_module, "current_workspace_manager", lambda: _TestWorkspaceManager()
    )
    monkeypatch.setattr(flashmla_module, "dequantize_and_gather_k_cache", fake_gather)
    monkeypatch.setattr(
        flashmla_module,
        "combine_topk_swa_indices_with_positions",
        fake_combine,
    )
    monkeypatch.setattr(flashmla_module, "flash_mla_sparse_fwd", fake_sparse_fwd)

    q = torch.tensor([[[10.0, 10.0]], [[13.0, 13.0]]])
    output = torch.empty_like(q)
    attention._forward_prefill(
        q=q,
        positions=torch.tensor([0, 3]),
        compressed_k_cache=torch.zeros((1, 1, 2), dtype=torch.uint8),
        swa_k_cache=torch.zeros((1, 4, 2), dtype=torch.uint8),
        output=output,
        attn_metadata=attn_metadata,
        swa_metadata=swa_metadata,
    )

    torch.testing.assert_close(captured["query_positions"], torch.tensor([0, 3]))
    torch.testing.assert_close(
        captured["query_start_loc"], torch.tensor([0, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["kernel_q"][:, 0, 0],
        torch.tensor([10.0, 13.0]),
    )
    torch.testing.assert_close(
        captured["kernel_indices"][:, 0, 0],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["kernel_lens"],
        torch.tensor([1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["kernel_kv"][:1, 0, 0],
        torch.tensor([-1.0], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        captured["kernel_kv"][1:5, 0, 0],
        torch.tensor([20.0, 21.0, 22.0, 23.0], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([10.0, 13.0]))
