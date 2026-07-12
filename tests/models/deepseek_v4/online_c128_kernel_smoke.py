# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H20/SM90 smoke test for DeepSeek-V4 online C128 CuTeDSL kernels.

Run inside the target vLLM pod after overlaying this checkout:

    python3 tests/models/deepseek_v4/online_c128_kernel_smoke.py --device cuda

The script intentionally launches the real kernels and checks them against a
small torch reference. It is not named test_*.py because it requires a CUDA SM90
runtime with CuTeDSL/quack available.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from vllm.models.deepseek_v4.nvidia.ops.online_c128_cutedsl import (
    OnlineC128DecodeKernel,
    OnlineC128MergeKernel,
    online_c128_decode,
    online_c128_merge,
)
from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
    compile_split_sparse_attn_cutedsl,
)
from vllm.models.deepseek_v4.online_c128 import (
    ONLINE_C128_COMPRESS_RATIO,
    ONLINE_C128_ROW_MODE_VERIFY,
    commit_online_c128_verify,
    plan_online_c128_segments,
)


def _empty_rows(num_rows: int, head_dim: int, device: torch.device) -> torch.Tensor:
    rows = torch.zeros(num_rows, 3 * head_dim, dtype=torch.float32, device=device)
    rows[:, :head_dim] = -float("inf")
    return rows


def _empty_row(head_dim: int, device: torch.device) -> torch.Tensor:
    return _empty_rows(1, head_dim, device)[0]


def _online_ref(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    *,
    init_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = kv.shape[1]
    if init_state is None:
        run_max = torch.full(
            (head_dim,), -float("inf"), dtype=torch.float32, device=kv.device
        )
        run_sum = torch.zeros(head_dim, dtype=torch.float32, device=kv.device)
        run_product = torch.zeros(head_dim, dtype=torch.float32, device=kv.device)
    else:
        run_max = init_state[:head_dim].clone()
        run_sum = init_state[head_dim : 2 * head_dim].clone()
        run_product = init_state[2 * head_dim :].clone()

    for row in range(kv.shape[0]):
        score_row = score[row].float() + ape[int(positions[row]) % ape.shape[0]]
        kv_row = kv[row].float()
        new_max = torch.maximum(run_max, score_row)
        old_scale = torch.exp(run_max - new_max)
        new_scale = torch.exp(score_row - new_max)
        run_sum = run_sum * old_scale + new_scale
        run_product = run_product * old_scale + kv_row * new_scale
        run_max = new_max

    state = torch.cat([run_max, run_sum, run_product])
    compressed = run_product / run_sum
    return state, compressed


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=2e-4,
        atol=2e-4,
        msg=f"{name} mismatch",
    )


def _compile_smoke(head_dim: int, max_num_reqs: int) -> None:
    OnlineC128MergeKernel.compile(
        head_size=head_dim,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    OnlineC128DecodeKernel.compile(
        head_size=head_dim,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
        max_num_reqs=max_num_reqs,
    )
    compile_split_sparse_attn_cutedsl(
        head_size=head_dim,
        state_width=head_dim,
        block_size=8,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        kv_cache_block_size=256,
        kv_block_stride=147456,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
        overlap=False,
        rms_norm_weight_dtype=torch.float32,
        store_full_kv=False,
    )


def _check_planned_prefill(
    device: torch.device,
    head_dim: int,
    max_num_reqs: int,
) -> None:
    num_tokens = ONLINE_C128_COMPRESS_RATIO + 2
    padded_tokens = num_tokens + 6
    torch.manual_seed(11)
    kv = torch.randn(padded_tokens, head_dim, device=device, dtype=torch.float32) * 0.2
    score = (
        torch.randn(padded_tokens, head_dim, device=device, dtype=torch.float32) * 0.1
    )
    ape = (
        torch.randn(
            ONLINE_C128_COMPRESS_RATIO, head_dim, device=device, dtype=torch.float32
        )
        * 0.01
    )
    positions = torch.arange(padded_tokens, device=device, dtype=torch.int64)
    run_state = _empty_rows(max_num_reqs, head_dim, device)
    compressed_kv = torch.zeros(
        num_tokens, head_dim, device=device, dtype=torch.float32
    )

    plan = plan_online_c128_segments(
        query_start_loc_cpu=np.array([0, num_tokens], dtype=np.int32),
        seq_lens_cpu=np.array([num_tokens], dtype=np.int32),
        req_state_indices_cpu=np.array([0], dtype=np.int32),
        max_num_reqs=max_num_reqs,
        device=device,
    )
    online_c128_merge(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        run_state=run_state,
        segments=plan.emit_segments,
        compressed_kv=compressed_kv,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    online_c128_merge(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        run_state=run_state,
        segments=plan.update_segments,
        compressed_kv=compressed_kv,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    torch.cuda.synchronize(device)

    _, expected_boundary = _online_ref(
        kv[:ONLINE_C128_COMPRESS_RATIO],
        score[:ONLINE_C128_COMPRESS_RATIO],
        ape,
        positions[:ONLINE_C128_COMPRESS_RATIO],
    )
    expected_tail, _ = _online_ref(
        kv[ONLINE_C128_COMPRESS_RATIO:num_tokens],
        score[ONLINE_C128_COMPRESS_RATIO:num_tokens],
        ape,
        positions[ONLINE_C128_COMPRESS_RATIO:num_tokens],
    )
    _assert_close(
        "planned_prefill boundary compressed_kv",
        compressed_kv[ONLINE_C128_COMPRESS_RATIO - 1],
        expected_boundary,
    )
    _assert_close("planned_prefill trailing bank0", run_state[0], expected_tail)


def _check_candidate_chain_and_commit(
    device: torch.device,
    head_dim: int,
    max_num_reqs: int,
) -> None:
    torch.manual_seed(23)
    prefix_len = ONLINE_C128_COMPRESS_RATIO - 2
    verify_len = 3
    padded_verify_len = verify_len + 2
    num_banks = verify_len + 1
    prefix_kv = torch.randn(prefix_len, head_dim, device=device) * 0.2
    prefix_score = torch.randn(prefix_len, head_dim, device=device) * 0.1
    query_kv = torch.randn(padded_verify_len, head_dim, device=device) * 0.2
    query_score = torch.randn(padded_verify_len, head_dim, device=device) * 0.1
    ape = torch.randn(ONLINE_C128_COMPRESS_RATIO, head_dim, device=device) * 0.01
    prefix_positions = torch.arange(prefix_len, device=device, dtype=torch.int64)
    query_positions = torch.arange(
        prefix_len,
        prefix_len + padded_verify_len,
        device=device,
        dtype=torch.int64,
    )

    prefix_state, _ = _online_ref(
        prefix_kv,
        prefix_score,
        ape,
        prefix_positions,
    )
    run_state = _empty_rows(num_banks * max_num_reqs, head_dim, device)
    run_state[0].copy_(prefix_state)
    compressed_kv = torch.zeros(
        verify_len, head_dim, device=device, dtype=torch.float32
    )
    query_start_loc = torch.tensor([0, verify_len], dtype=torch.int32, device=device)
    req_state_indices = torch.tensor([0], dtype=torch.int32, device=device)
    row_modes = torch.tensor(
        [ONLINE_C128_ROW_MODE_VERIFY], dtype=torch.int32, device=device
    )

    online_c128_decode(
        kv=query_kv,
        score=query_score,
        ape=ape,
        positions=query_positions,
        query_start_loc=query_start_loc,
        req_state_indices=req_state_indices,
        row_modes=row_modes,
        run_state=run_state,
        compressed_kv=compressed_kv,
        max_num_reqs=max_num_reqs,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    torch.cuda.synchronize(device)

    bank1, _ = _online_ref(
        query_kv[:1],
        query_score[:1],
        ape,
        query_positions[:1],
        init_state=prefix_state,
    )
    _, boundary_compressed = _online_ref(
        query_kv[:2],
        query_score[:2],
        ape,
        query_positions[:2],
        init_state=prefix_state,
    )
    bank3, _ = _online_ref(
        query_kv[2:3],
        query_score[2:3],
        ape,
        query_positions[2:3],
    )
    empty = _empty_row(head_dim, device)
    run_state_after_verify = run_state.clone()

    _assert_close("candidate bank0 unchanged", run_state_after_verify[0], prefix_state)
    _assert_close("candidate bank1", run_state_after_verify[max_num_reqs], bank1)
    _assert_close(
        "candidate boundary compressed_kv", compressed_kv[1], boundary_compressed
    )
    _assert_close(
        "candidate bank2 reset", run_state_after_verify[2 * max_num_reqs], empty
    )
    _assert_close("candidate bank3", run_state_after_verify[3 * max_num_reqs], bank3)

    accepted_len = torch.tensor([1], dtype=torch.int32, device=device)
    final_seq_len = torch.tensor([prefix_len + 1], dtype=torch.int32, device=device)
    run_state.copy_(run_state_after_verify)
    commit_online_c128_verify(
        run_state=run_state,
        req_state_indices=req_state_indices,
        accepted_len=accepted_len,
        final_seq_len=final_seq_len,
        max_num_reqs=max_num_reqs,
        head_dim=head_dim,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    torch.cuda.synchronize(device)
    _assert_close("commit accepted_len=1 copies bank1", run_state[0], bank1)

    accepted_len = torch.tensor([2], dtype=torch.int32, device=device)
    final_seq_len = torch.tensor(
        [ONLINE_C128_COMPRESS_RATIO], dtype=torch.int32, device=device
    )
    run_state.copy_(run_state_after_verify)
    commit_online_c128_verify(
        run_state=run_state,
        req_state_indices=req_state_indices,
        accepted_len=accepted_len,
        final_seq_len=final_seq_len,
        max_num_reqs=max_num_reqs,
        head_dim=head_dim,
        compress_ratio=ONLINE_C128_COMPRESS_RATIO,
    )
    torch.cuda.synchronize(device)
    _assert_close("commit boundary resets bank0", run_state[0], empty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--max-num-reqs", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for online C128 kernel smoke.")
    device = torch.device(args.device)
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 9:
            raise SystemExit(f"SM90/H20 class GPU is required, got {capability}.")

    _compile_smoke(args.head_dim, args.max_num_reqs)
    _check_planned_prefill(device, args.head_dim, args.max_num_reqs)
    _check_candidate_chain_and_commit(device, args.head_dim, args.max_num_reqs)
    print("online_c128 kernel smoke: OK")


if __name__ == "__main__":
    main()
