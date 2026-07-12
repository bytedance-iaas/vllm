# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.models.deepseek_v4 import online_c128
from vllm.models.deepseek_v4.compressor import CompressorMetadataBuilder
from vllm.models.deepseek_v4.online_c128 import (
    ONLINE_C128_ROW_MODE_PREFILL,
    ONLINE_C128_ROW_MODE_VERIFY,
    assert_online_c128_supported,
    online_c128_uses_mtp,
    plan_online_c128_segments,
    plan_online_c128_verify,
)
from vllm.v1.attention.backend import CommonAttentionMetadata


def test_plan_online_c128_segments_exact_boundary_emits_and_resets_bank0():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 128], dtype=np.int32),
        seq_lens_cpu=np.array([128], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert plan.emit_segments.cpu().tolist() == [[0, 128, -1, 127, -1]]
    assert plan.update_segments.shape == (0, 5)
    assert plan.reset_rows.cpu().tolist() == [3]
    assert plan.reset_rows_long.cpu().tolist() == [3]


def test_online_c128_uses_mtp_from_speculative_method(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    mtp_config = SimpleNamespace(speculative_config=SimpleNamespace(method="mtp"))
    eagle_config = SimpleNamespace(speculative_config=SimpleNamespace(method="eagle3"))

    assert online_c128_uses_mtp(mtp_config)
    assert not online_c128_uses_mtp(eagle_config)


def test_online_c128_rejects_non_mtp_speculative(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    monkeypatch.setattr(online_c128, "_is_sm90", lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(method="eagle3"),
    )

    with pytest.raises(ValueError, match="only supports the MTP speculative method"):
        assert_online_c128_supported(config, compress_ratio=128, head_dim=512)


def test_online_c128_rejects_dbo_ubatching(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    monkeypatch.setattr(online_c128, "_is_sm90", lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=True,
            ubatch_size=0,
            use_ubatching=True,
        ),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match="does not support DBO/u-batching"):
        assert_online_c128_supported(config, compress_ratio=128, head_dim=512)


def test_online_c128_allows_async_scheduling(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_ONLINE_C128_COMPRESS", True)
    monkeypatch.setattr(online_c128, "_is_sm90", lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
            use_ubatching=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=None,
    )

    assert_online_c128_supported(config, compress_ratio=128, head_dim=512)


def _make_compressor_metadata_builder(*, online_c128_enabled: bool):
    builder = CompressorMetadataBuilder.__new__(CompressorMetadataBuilder)
    builder._online_c128 = online_c128_enabled
    builder.device = "cpu"
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=None,
    )
    builder.block_size = 8
    if not online_c128_enabled:
        builder.token_to_req_indices = torch.zeros(8, dtype=torch.int32)
    else:
        builder.online_c128_row_modes = torch.full(
            (8,), ONLINE_C128_ROW_MODE_PREFILL, dtype=torch.int32
        )
    return builder


def _make_common_metadata(
    *,
    is_prefilling: torch.Tensor | None = None,
    num_draft_tokens_per_req_cpu: np.ndarray | None = None,
) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        seq_lens=torch.tensor([1, 3], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1, 3], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=3,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        req_state_indices=torch.tensor([4, 5], dtype=torch.int32),
        req_state_indices_cpu=np.array([4, 5], dtype=np.int32),
        is_prefilling=is_prefilling,
        num_draft_tokens_per_req_cpu=num_draft_tokens_per_req_cpu,
    )


def test_compressor_metadata_builder_keeps_offline_token_mapping(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    metadata = _make_compressor_metadata_builder(
        online_c128_enabled=False
    ).build(0, _make_common_metadata())

    assert metadata.token_to_req_indices is not None
    assert metadata.token_to_req_indices.cpu().tolist() == [0, 1, 1]
    assert metadata.req_state_indices is None
    assert metadata.online_c128_plan is None


def test_compressor_metadata_builder_skips_online_token_mapping(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    common_metadata = _make_common_metadata(
        is_prefilling=torch.tensor([True, False]),
        num_draft_tokens_per_req_cpu=np.array([0, 2], dtype=np.int32),
    )
    metadata = _make_compressor_metadata_builder(
        online_c128_enabled=True
    ).build(0, common_metadata)

    assert metadata.token_to_req_indices is None
    assert metadata.req_state_indices is not None
    assert metadata.req_state_indices.cpu().tolist() == [4, 5]
    assert metadata.query_start_loc is not None
    assert metadata.online_c128_row_modes is not None
    assert metadata.online_c128_row_modes.cpu().tolist() == [
        ONLINE_C128_ROW_MODE_PREFILL,
        ONLINE_C128_ROW_MODE_VERIFY,
    ]
    assert metadata.online_c128_has_decode_rows
    assert metadata.online_c128_plan is not None


def test_plan_online_c128_segments_partial_start_splits_emit_and_update():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 3], dtype=np.int32),
        seq_lens_cpu=np.array([130], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert plan.emit_segments.cpu().tolist() == [[0, 1, 3, 0, -1]]
    assert plan.update_segments.cpu().tolist() == [[1, 2, -1, -1, 3]]
    assert plan.reset_rows.cpu().tolist() == []
    assert plan.reset_rows_long.cpu().tolist() == []


def test_plan_online_c128_segments_req_mask_skips_unselected_reqs():
    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, 128, 256], dtype=np.int32),
        seq_lens_cpu=np.array([128, 256], dtype=np.int32),
        req_state_indices_cpu=np.array([3, 4], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
        req_mask=np.array([False, True]),
    )

    assert plan.emit_segments.cpu().tolist() == [[128, 128, -1, 255, -1]]
    assert plan.update_segments.shape == (0, 5)
    assert plan.reset_rows.cpu().tolist() == [4]


def test_plan_online_c128_verify_aligned_start_uses_identity():
    segments = plan_online_c128_verify(
        query_start_loc_cpu=np.array([0, 1], dtype=np.int32),
        seq_lens_cpu=np.array([129], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert segments[0].cpu().tolist() == [[0, 1, -1, -1, 11]]


def test_plan_online_c128_verify_partial_start_reads_bank0():
    segments = plan_online_c128_verify(
        query_start_loc_cpu=np.array([0, 1], dtype=np.int32),
        seq_lens_cpu=np.array([128], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert segments[0].cpu().tolist() == [[0, 1, 3, 0, 11]]


def test_plan_online_c128_verify_req_mask_limits_candidate_rows():
    segments = plan_online_c128_verify(
        query_start_loc_cpu=np.array([0, 2, 5], dtype=np.int32),
        seq_lens_cpu=np.array([130, 133], dtype=np.int32),
        req_state_indices_cpu=np.array([3, 4], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
        req_mask=np.array([True, False]),
        max_query_len=2,
    )

    assert len(segments) == 2
    assert all(seg.shape[0] == 1 for seg in segments)


def test_plan_online_c128_verify_boundary_step_resets_next_candidate_bank():
    segments = plan_online_c128_verify(
        query_start_loc_cpu=np.array([0, 3], dtype=np.int32),
        seq_lens_cpu=np.array([129], dtype=np.int32),
        req_state_indices_cpu=np.array([3], dtype=np.int32),
        max_num_reqs=8,
        device="cpu",
    )

    assert [seg.cpu().tolist() for seg in segments] == [
        [[0, 1, 3, -1, 11]],
        [[1, 1, 11, 1, 19]],
        [[2, 1, 19, -1, 27]],
    ]


def test_plan_online_c128_verify_rejects_query_longer_than_candidate_banks():
    with pytest.raises(ValueError, match="exceeds allocated candidate banks"):
        plan_online_c128_verify(
            query_start_loc_cpu=np.array([0, 3], dtype=np.int32),
            seq_lens_cpu=np.array([3], dtype=np.int32),
            req_state_indices_cpu=np.array([0], dtype=np.int32),
            max_num_reqs=8,
            device="cpu",
            max_query_len=2,
        )
