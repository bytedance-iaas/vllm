# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.models.deepseek_v4.online_c128 import plan_online_c128_verify


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
