# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.models.deepseek_v41.nvidia.flashmla as flashmla
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class _Workspace:
    def __init__(self) -> None:
        self.calls = []

    def get_simultaneous(self, *specs):
        self.calls.append(specs)
        return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)


def _metadata(query_lens, num_decodes, chunk_plan):
    query_start = torch.tensor(
        [0, *accumulate(query_lens)],
        dtype=torch.int32,
    )
    prefill_lens = query_lens[num_decodes:]
    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=sum(query_lens[:num_decodes]),
        num_prefill_tokens=sum(prefill_lens),
        prefill_seq_lens=torch.tensor(prefill_lens, dtype=torch.int32),
        prefill_gather_lens=torch.tensor(prefill_lens, dtype=torch.int32),
        query_start_loc_cpu=query_start,
        query_start_loc=query_start,
        block_table=torch.zeros(len(query_lens), 1, dtype=torch.int32),
        block_size=128,
    )
    metadata.get_prefill_chunk_plan = MagicMock(return_value=chunk_plan)
    return metadata


def _run_prefill(monkeypatch, query_lens, num_decodes, chunk_plan, compress_ratio):
    metadata = _metadata(query_lens, num_decodes, chunk_plan)
    num_tokens = metadata.num_decode_tokens + metadata.num_prefill_tokens
    attention = SimpleNamespace(
        compress_ratio=compress_ratio,
        topk_indices_buffer=torch.zeros(num_tokens, 1, dtype=torch.int32),
        PREFILL_CHUNK_SIZE=4,
        window_size=128,
        max_num_batched_tokens=2048,
        scale=1.0,
        attn_sink=None,
    )
    workspace = _Workspace()
    gather = MagicMock()
    sparse_fwd = MagicMock()
    monkeypatch.setattr(flashmla, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(flashmla, "dequantize_and_gather_k_cache", gather)
    monkeypatch.setattr(
        flashmla,
        "combine_topk_swa_indices",
        lambda *args, out, **kwargs: out,
    )
    monkeypatch.setattr(flashmla, "flash_mla_sparse_fwd", sparse_fwd)

    prefill_tokens = metadata.num_prefill_tokens
    DeepseekV4FlashMLAAttention._forward_prefill(
        attention,
        q=torch.empty(prefill_tokens, 1, 4),
        positions=torch.arange(prefill_tokens),
        compressed_k_cache=(
            None if compress_ratio == 0 else torch.empty(1, dtype=torch.uint8)
        ),
        swa_k_cache=torch.empty(1, dtype=torch.uint8),
        output=torch.empty(prefill_tokens, 1, 4),
        attn_metadata=(
            None
            if compress_ratio == 0
            else SimpleNamespace(
                block_table=metadata.block_table,
                block_size=128,
            )
        ),
        swa_metadata=metadata,
        cp_plan=None,
    )
    return workspace, gather, sparse_fwd


def test_padded_capture_shape_emits_zero_query_tail_chunk():
    query_lens = torch.tensor([128, *([0] * 79)], dtype=torch.int32)
    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0),
        slot_mapping=torch.empty(0),
        block_size=128,
        num_prefills=80,
        prefill_seq_lens_cpu=query_lens,
        prefill_query_lens_cpu=query_lens,
        prefill_window_size=128,
        prefill_max_model_len=2048,
        prefill_max_num_batched_tokens=2048,
    )

    assert metadata.get_prefill_chunk_plan(
        compress_ratio=0,
        prefill_chunk_size=4,
        has_compressed=False,
    ) == [(0, 68, 0, 128), (68, 80, 0, 0)]


@pytest.mark.parametrize("compress_ratio", [0, 1, 2])
def test_zero_query_tail_skips_all_prefill_work(monkeypatch, compress_ratio):
    workspace, gather, sparse_fwd = _run_prefill(
        monkeypatch,
        query_lens=[6] * 16 + [128] + [0] * 79,
        num_decodes=16,
        chunk_plan=[(0, 68, 0, 128), (68, 80, 0, 0)],
        compress_ratio=compress_ratio,
    )

    assert len(workspace.calls) == 1
    assert gather.call_count == (1 if compress_ratio == 0 else 2)
    assert [call.kwargs["q"].shape[0] for call in sparse_fwd.call_args_list] == [128]


def test_nonempty_chunks_are_all_executed(monkeypatch):
    workspace, gather, sparse_fwd = _run_prefill(
        monkeypatch,
        query_lens=[2, 3],
        num_decodes=0,
        chunk_plan=[(0, 1, 0, 2), (1, 2, 0, 3)],
        compress_ratio=0,
    )

    assert len(workspace.calls) == 2
    assert gather.call_count == 2
    assert [call.kwargs["q"].shape[0] for call in sparse_fwd.call_args_list] == [2, 3]


def test_cp_local_empty_chunk_still_skips_workspace(monkeypatch):
    metadata = _metadata([3], 0, [(0, 1, 0, 3)])
    attention = SimpleNamespace(
        compress_ratio=0,
        topk_indices_buffer=torch.zeros(3, 1, dtype=torch.int32),
        PREFILL_CHUNK_SIZE=4,
        window_size=128,
        max_num_batched_tokens=2048,
    )
    workspace = _Workspace()
    monkeypatch.setattr(flashmla, "current_workspace_manager", lambda: workspace)

    DeepseekV4FlashMLAAttention._forward_prefill(
        attention,
        q=torch.empty(3, 1, 4),
        positions=torch.arange(3),
        compressed_k_cache=None,
        swa_k_cache=torch.empty(1),
        output=torch.empty(3, 1, 4),
        attn_metadata=None,
        swa_metadata=metadata,
        cp_plan=SimpleNamespace(
            local_query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
            token_indices=torch.empty(0, dtype=torch.int64),
        ),
    )

    assert workspace.calls == []


def test_pure_decode_does_not_enter_prefill(monkeypatch):
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
    )
    attention = SimpleNamespace(
        compressed_cache_prefix=None,
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=torch.empty(1)),
        compress_ratio=0,
        _build_attention_cp_plan=MagicMock(return_value=None),
        _forward_prefill=MagicMock(),
        _forward_decode=MagicMock(),
    )
    monkeypatch.setattr(
        flashmla,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"swa": metadata}),
    )

    DeepseekV4FlashMLAAttention.forward_mqa(
        attention,
        q=torch.empty(1, 1, 4),
        kv=torch.empty(1),
        positions=torch.zeros(1, dtype=torch.int64),
        output=torch.empty(1, 1, 4),
    )

    attention._forward_prefill.assert_not_called()
    attention._forward_decode.assert_called_once()
