# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.models.deepseek_v4.online_c128 import (
    plan_online_c128_segments,
    plan_online_c128_verify,
)


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
