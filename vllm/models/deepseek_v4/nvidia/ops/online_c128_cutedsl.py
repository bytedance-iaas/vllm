# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels for DeepSeek-V4 online C128 running-state updates."""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from quack.compile_utils import make_fake_tensor

RCP_LN2 = 1.4426950408889634


class OnlineC128MergeKernel:
    """Merge this step's segment tokens into the per-request running state."""

    elems_per_lane = 2

    def __init__(self, head_size: int, compress_ratio: int):
        self.head_dim = head_size
        self.compress_ratio = compress_ratio
        self.tb_size = head_size // self.elems_per_lane

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [num_tokens, head_dim] fp32
        score: cute.Tensor,  # [num_tokens, head_dim] fp32
        ape: cute.Tensor,  # [compress_ratio, head_dim] fp32
        positions: cute.Tensor,  # [num_tokens] int64
        run_state: cute.Tensor,  # [num_rows, 3 * head_dim] fp32
        segments: cute.Tensor,  # [num_segments, 5] int32
        compressed_kv: cute.Tensor,  # [num_output_tokens, head_dim] fp32
        stream: CUstream,
    ):
        grid = (segments.shape[0], 1, 1)
        self.kernel(
            kv, score, ape, positions, run_state, segments, compressed_kv
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        kv: cute.Tensor,
        score: cute.Tensor,
        ape: cute.Tensor,
        positions: cute.Tensor,
        run_state: cute.Tensor,
        segments: cute.Tensor,
        compressed_kv: cute.Tensor,
    ):
        seg_id, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        col0 = tid * self.elems_per_lane

        row_base = segments[seg_id, 0]
        num_rows = segments[seg_id, 1]
        read_row = segments[seg_id, 2]
        emit_token = segments[seg_id, 3]
        write_row = segments[seg_id, 4]

        max_off = Int64(0)
        sum_off = Int64(self.head_dim)
        wsum_off = Int64(2 * self.head_dim)

        run_state_w = run_state.stride[0]
        kv_w = kv.stride[0]
        score_w = score.stride[0]
        ape_w = ape.stride[0]
        compressed_w = compressed_kv.stride[0]

        local_max = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
        local_sum = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
        local_product = cute.make_rmem_tensor((self.elems_per_lane,), Float32)

        # Seed from the committed carry row if requested, else identity.
        for e in cutlass.range_constexpr(self.elems_per_lane):
            local_max[e] = -Float32.inf
            local_sum[e] = Float32(0.0)
            local_product[e] = Float32(0.0)

        if read_row >= Int32(0):
            base = read_row.to(Int64) * run_state_w + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                local_max[e] = run_state.iterator[base + max_off + Int64(e)]
                local_sum[e] = run_state.iterator[base + sum_off + Int64(e)]
                local_product[e] = run_state.iterator[base + wsum_off + Int64(e)]

        # Sequentially merge each step row into the accumulator (online softmax).
        for r in cutlass.range(num_rows, unroll=1):
            token = row_base + r
            tok64 = token.to(Int64)
            position = positions[tok64]
            ape_row = (position % Int64(self.compress_ratio)) * ape_w
            kv_base = tok64 * kv_w + col0.to(Int64)
            score_base = tok64 * score_w + col0.to(Int64)
            ape_base = ape_row + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                kv_e = kv.iterator[kv_base + Int64(e)].to(Float32)
                score_e = score.iterator[score_base + Int64(e)].to(Float32)
                ape_e = ape.iterator[ape_base + Int64(e)]
                score_e = score_e + ape_e
                new_max = cute.arch.fmax(local_max[e], score_e)
                old_scale = cute.math.exp2(
                    (local_max[e] - new_max) * Float32(RCP_LN2), fastmath=True
                )
                new_scale = cute.math.exp2(
                    (score_e - new_max) * Float32(RCP_LN2), fastmath=True
                )
                local_sum[e] = local_sum[e] * old_scale + new_scale
                local_product[e] = local_product[e] * old_scale + kv_e * new_scale
                local_max[e] = new_max

        if emit_token >= Int32(0):
            ebase = emit_token.to(Int64) * compressed_w + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                compressed_kv.iterator[ebase + Int64(e)] = (
                    local_product[e] / local_sum[e]
                )

        if write_row >= Int32(0):
            wbase = write_row.to(Int64) * run_state_w + col0.to(Int64)
            if emit_token >= Int32(0):
                # Boundary in a candidate chain: the chunk closed at this token,
                # so the carry for the next chain position is the empty identity.
                for e in cutlass.range_constexpr(self.elems_per_lane):
                    run_state.iterator[wbase + max_off + Int64(e)] = -Float32.inf
                    run_state.iterator[wbase + sum_off + Int64(e)] = Float32(0.0)
                    run_state.iterator[wbase + wsum_off + Int64(e)] = Float32(0.0)
            else:
                for e in cutlass.range_constexpr(self.elems_per_lane):
                    run_state.iterator[wbase + max_off + Int64(e)] = local_max[e]
                    run_state.iterator[wbase + sum_off + Int64(e)] = local_sum[e]
                    run_state.iterator[wbase + wsum_off + Int64(e)] = local_product[e]

    @cache
    @staticmethod
    def compile(head_size: int = 512, compress_ratio: int = 128):
        if head_size % OnlineC128MergeKernel.elems_per_lane != 0:
            raise ValueError("head_size must be even.")
        num_tokens = cute.sym_int()
        num_rows = cute.sym_int()
        num_segments = cute.sym_int()
        num_output_tokens = cute.sym_int()

        kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        score = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        ape = cute.runtime.make_fake_tensor(
            Float32,
            (compress_ratio, head_size),
            stride=(head_size, 1),
            assumed_align=16,
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=8)
        run_state = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, 3 * head_size),
            stride=(cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        segments = make_fake_tensor(Int32, (num_segments, 5), divisibility=1)
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_output_tokens, head_size),
            stride=(head_size, 1),
            assumed_align=4,
        )
        kernel = OnlineC128MergeKernel(head_size, compress_ratio)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            kv,
            score,
            ape,
            positions,
            run_state,
            segments,
            compressed_kv,
            stream,
            options="--enable-tvm-ffi",
        )


def online_c128_merge(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    run_state: torch.Tensor,
    segments: torch.Tensor,
    compressed_kv: torch.Tensor,
    compress_ratio: int = 128,
) -> None:
    """Launch the online C128 merge kernel for one segment list."""
    if segments.numel() == 0:
        return
    head_size = compressed_kv.shape[-1]
    if kv.dtype != torch.float32 or score.dtype != torch.float32:
        raise ValueError(
            "online_c128_merge expects fp32 kv/score, got "
            f"{kv.dtype} / {score.dtype}."
        )
    compiled = OnlineC128MergeKernel.compile(
        head_size=head_size, compress_ratio=compress_ratio
    )
    compiled(kv, score, ape, positions, run_state, segments, compressed_kv)


class OnlineC128DecodeKernel:
    """Graph-safe per-request recurrence for decode / MTP verify."""

    elems_per_lane = 2

    def __init__(
        self,
        head_size: int,
        compress_ratio: int,
        max_num_reqs: int,
        candidate_chain: bool,
    ):
        self.head_dim = head_size
        self.compress_ratio = compress_ratio
        self.max_num_reqs = max_num_reqs
        self.candidate_chain = candidate_chain
        self.tb_size = head_size // self.elems_per_lane

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [num_tokens, head_dim] fp32
        score: cute.Tensor,  # [num_tokens, head_dim] fp32
        ape: cute.Tensor,  # [compress_ratio, head_dim] fp32
        positions: cute.Tensor,  # [num_tokens] int64
        query_start_loc: cute.Tensor,  # [num_reqs + 1] int32
        req_state_indices: cute.Tensor,  # [num_reqs] int32 (-1 pad)
        run_state: cute.Tensor,  # [num_banks * max_num_reqs, 3*head_dim] fp32
        compressed_kv: cute.Tensor,  # [num_output_tokens, head_dim] fp32
        stream: CUstream,
    ):
        grid = (req_state_indices.shape[0], 1, 1)
        self.kernel(
            kv,
            score,
            ape,
            positions,
            query_start_loc,
            req_state_indices,
            run_state,
            compressed_kv,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        kv: cute.Tensor,
        score: cute.Tensor,
        ape: cute.Tensor,
        positions: cute.Tensor,
        query_start_loc: cute.Tensor,
        req_state_indices: cute.Tensor,
        run_state: cute.Tensor,
        compressed_kv: cute.Tensor,
    ):
        req, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        col0 = tid * self.elems_per_lane

        rsi = req_state_indices[req]
        if rsi >= Int32(0):
            tok_start = query_start_loc[req]
            tok_end = query_start_loc[req + Int32(1)]
            query_len = tok_end - tok_start

            if query_len > Int32(0):
                max_off = Int64(0)
                sum_off = Int64(self.head_dim)
                wsum_off = Int64(2 * self.head_dim)

                run_state_w = run_state.stride[0]
                kv_w = kv.stride[0]
                score_w = score.stride[0]
                ape_w = ape.stride[0]
                compressed_w = compressed_kv.stride[0]
                max_num_reqs64 = Int64(self.max_num_reqs)

                local_max = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
                local_sum = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
                local_product = cute.make_rmem_tensor(
                    (self.elems_per_lane,), Float32
                )

                # Aligned starts seed from identity; bank0 may contain stale carry.
                first_pos = positions[tok_start.to(Int64)]
                carry_len = first_pos % Int64(self.compress_ratio)
                bank0_base = rsi.to(Int64) * run_state_w + col0.to(Int64)
                if carry_len != Int64(0):
                    for e in cutlass.range_constexpr(self.elems_per_lane):
                        local_max[e] = run_state.iterator[
                            bank0_base + max_off + Int64(e)
                        ]
                        local_sum[e] = run_state.iterator[
                            bank0_base + sum_off + Int64(e)
                        ]
                        local_product[e] = run_state.iterator[
                            bank0_base + wsum_off + Int64(e)
                        ]
                else:
                    for e in cutlass.range_constexpr(self.elems_per_lane):
                        local_max[e] = -Float32.inf
                        local_sum[e] = Float32(0.0)
                        local_product[e] = Float32(0.0)

                rsi64 = rsi.to(Int64)
                # Walk the request's query tokens sequentially.
                for j in cutlass.range(query_len, unroll=1):
                    token = tok_start + j
                    tok64 = token.to(Int64)
                    position = positions[tok64]
                    ape_row = (position % Int64(self.compress_ratio)) * ape_w
                    kv_base = tok64 * kv_w + col0.to(Int64)
                    score_base = tok64 * score_w + col0.to(Int64)
                    ape_base = ape_row + col0.to(Int64)
                    for e in cutlass.range_constexpr(self.elems_per_lane):
                        kv_e = kv.iterator[kv_base + Int64(e)].to(Float32)
                        score_e = score.iterator[score_base + Int64(e)].to(Float32)
                        ape_e = ape.iterator[ape_base + Int64(e)]
                        score_e = score_e + ape_e
                        new_max = cute.arch.fmax(local_max[e], score_e)
                        old_scale = cute.math.exp2(
                            (local_max[e] - new_max) * Float32(RCP_LN2),
                            fastmath=True,
                        )
                        new_scale = cute.math.exp2(
                            (score_e - new_max) * Float32(RCP_LN2), fastmath=True
                        )
                        local_sum[e] = local_sum[e] * old_scale + new_scale
                        local_product[e] = (
                            local_product[e] * old_scale + kv_e * new_scale
                        )
                        local_max[e] = new_max

                    position_i = position + Int64(1)
                    boundary = (position_i % Int64(self.compress_ratio)) == Int64(0)
                    if boundary:
                        ebase = tok64 * compressed_w + col0.to(Int64)
                        for e in cutlass.range_constexpr(self.elems_per_lane):
                            compressed_kv.iterator[ebase + Int64(e)] = (
                                local_product[e] / local_sum[e]
                            )

                    if cutlass.const_expr(self.candidate_chain):
                        # Verify writes candidate banks; commit advances bank0.
                        write_row = (
                            (j + Int32(1)).to(Int64) * max_num_reqs64 + rsi64
                        )
                        wbase = write_row * run_state_w + col0.to(Int64)
                        if boundary:
                            for e in cutlass.range_constexpr(self.elems_per_lane):
                                run_state.iterator[
                                    wbase + max_off + Int64(e)
                                ] = -Float32.inf
                                run_state.iterator[
                                    wbase + sum_off + Int64(e)
                                ] = Float32(0.0)
                                run_state.iterator[
                                    wbase + wsum_off + Int64(e)
                                ] = Float32(0.0)
                                local_max[e] = -Float32.inf
                                local_sum[e] = Float32(0.0)
                                local_product[e] = Float32(0.0)
                        else:
                            for e in cutlass.range_constexpr(self.elems_per_lane):
                                run_state.iterator[
                                    wbase + max_off + Int64(e)
                                ] = local_max[e]
                                run_state.iterator[
                                    wbase + sum_off + Int64(e)
                                ] = local_sum[e]
                                run_state.iterator[
                                    wbase + wsum_off + Int64(e)
                                ] = local_product[e]
                    else:
                        if boundary:
                            # Decode: chunk closed; restart from identity.
                            for e in cutlass.range_constexpr(self.elems_per_lane):
                                local_max[e] = -Float32.inf
                                local_sum[e] = Float32(0.0)
                                local_product[e] = Float32(0.0)

                if cutlass.const_expr(not self.candidate_chain):
                    # Decode: write trailing carry back to committed bank0.
                    for e in cutlass.range_constexpr(self.elems_per_lane):
                        run_state.iterator[
                            bank0_base + max_off + Int64(e)
                        ] = local_max[e]
                        run_state.iterator[
                            bank0_base + sum_off + Int64(e)
                        ] = local_sum[e]
                        run_state.iterator[
                            bank0_base + wsum_off + Int64(e)
                        ] = local_product[e]

    @cache
    @staticmethod
    def compile(
        head_size: int = 512,
        compress_ratio: int = 128,
        max_num_reqs: int = 1,
        candidate_chain: bool = False,
    ):
        if head_size % OnlineC128DecodeKernel.elems_per_lane != 0:
            raise ValueError("head_size must be even.")
        num_tokens = cute.sym_int()
        num_reqs = cute.sym_int()
        num_query_locs = cute.sym_int()
        num_rows = cute.sym_int()
        num_output_tokens = cute.sym_int()

        kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        score = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        ape = cute.runtime.make_fake_tensor(
            Float32,
            (compress_ratio, head_size),
            stride=(head_size, 1),
            assumed_align=16,
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=8)
        query_start_loc = make_fake_tensor(
            Int32, (num_query_locs,), divisibility=1
        )
        req_state_indices = make_fake_tensor(Int32, (num_reqs,), divisibility=1)
        run_state = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, 3 * head_size),
            stride=(cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_output_tokens, head_size),
            stride=(head_size, 1),
            assumed_align=4,
        )
        kernel = OnlineC128DecodeKernel(
            head_size, compress_ratio, max_num_reqs, candidate_chain
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            kv,
            score,
            ape,
            positions,
            query_start_loc,
            req_state_indices,
            run_state,
            compressed_kv,
            stream,
            options="--enable-tvm-ffi",
        )


def online_c128_decode(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_state_indices: torch.Tensor,
    run_state: torch.Tensor,
    compressed_kv: torch.Tensor,
    max_num_reqs: int,
    compress_ratio: int = 128,
    candidate_chain: bool = False,
) -> None:
    """Launch the fixed-address decode / MTP-verify recurrence."""
    if req_state_indices.numel() == 0:
        return
    head_size = compressed_kv.shape[-1]
    if kv.dtype != torch.float32 or score.dtype != torch.float32:
        raise ValueError(
            "online_c128_decode expects fp32 kv/score, got "
            f"{kv.dtype} / {score.dtype}."
        )
    compiled = OnlineC128DecodeKernel.compile(
        head_size=head_size,
        compress_ratio=compress_ratio,
        max_num_reqs=max_num_reqs,
        candidate_chain=candidate_chain,
    )
    compiled(
        kv,
        score,
        ape,
        positions,
        query_start_loc,
        req_state_indices,
        run_state,
        compressed_kv,
    )
