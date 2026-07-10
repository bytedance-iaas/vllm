# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 C128 online running state.

Rows are fp32 ``[run_max, run_sum, run_weighted_sum]``. Bank 0 is committed;
banks 1..N are MTP candidates.
"""

from __future__ import annotations

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Fixed support envelope for the C128 online path.
ONLINE_C128_COMPRESS_RATIO = 128
ONLINE_C128_HEAD_DIM = 512
# Number of running-state vectors per row: [run_max, run_sum, run_weighted_sum].
ONLINE_C128_NUM_STATE_VECTORS = 3
ONLINE_C128_STATE_DTYPE = torch.float32


def online_c128_compress_enabled() -> bool:
    """Whether the C128 online compress feature flag is set."""
    return bool(envs.VLLM_USE_ONLINE_C128_COMPRESS)


def online_c128_uses_mtp(vllm_config: VllmConfig) -> bool:
    """Whether online C128 is running with MTP speculative decoding."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return (
        online_c128_compress_enabled()
        and speculative_config is not None
        and getattr(speculative_config, "method", None) == "mtp"
    )


def _is_sm90() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability(90)


def assert_online_c128_supported(
    vllm_config: VllmConfig,
    *,
    compress_ratio: int,
    head_dim: int,
) -> None:
    """Fail-closed guard for the supported C128 online envelope.

    Only invoked when ``online_c128_compress_enabled()`` is True. Raises on any
    unsupported configuration rather than silently degrading.
    """
    if not _is_sm90():
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS is only supported on CUDA SM90 "
            "(Hopper)."
        )
    if compress_ratio != ONLINE_C128_COMPRESS_RATIO:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS requires compress_ratio == "
            f"{ONLINE_C128_COMPRESS_RATIO}, got {compress_ratio}."
        )
    if head_dim != ONLINE_C128_HEAD_DIM:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS requires head_dim == "
            f"{ONLINE_C128_HEAD_DIM}, got {head_dim}."
        )

    parallel_config = vllm_config.parallel_config
    if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support decode context "
            "parallelism (DCP)."
        )
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1:
        raise ValueError(
            "VLLM_USE_ONLINE_C128_COMPRESS does not support prefill context "
            "parallelism (PCP)."
        )

    # Any speculative decoding under online compress MUST be MTP. Other
    # speculative methods would merge rejected draft tokens into committed bank0
    # and pollute later decode.
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        method = getattr(speculative_config, "method", None)
        if method != "mtp":
            raise ValueError(
                "VLLM_USE_ONLINE_C128_COMPRESS with speculative decoding only "
                f"supports the MTP speculative method; got {method!r}."
            )


def online_c128_num_banks(vllm_config: VllmConfig) -> int:
    """Number of state banks: bank0 (committed) plus MTP candidate banks.

    For the non-MTP path this is 1. For MTP it is ``1 + verify_query_len`` where
    the verify query len upper bound is ``num_speculative_tokens + 1`` (the MTP
    draft tokens plus the bonus token verified by the target model).
    """
    if not online_c128_uses_mtp(vllm_config):
        return 1
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_spec = 0
    if speculative_config is not None:
        num_spec = getattr(speculative_config, "num_speculative_tokens", None) or 0
    # bank0 committed + (num_spec + 1) candidate banks for the verify chain.
    return 1 + (num_spec + 1)


class DeepseekOnlineC128State(torch.nn.Module):
    """Dense per-request running state for one C128 online compressor layer.

    The state is a single contiguous ``float32`` tensor of shape
    ``[num_banks * max_num_reqs, num_state_vectors * head_dim]`` allocated at
    model init. It is intentionally NOT an ``AttentionLayerBase`` and is NOT
    registered in ``static_forward_context`` as a KV-cache layer, so it is not
    enumerated into the KV cache spec and not carved from profiled KV memory.
    Because it is allocated before ``profile_run``, its bytes are captured in
    the post-init memory baseline and excluded from the available KV pool
    automatically (no double counting).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        head_dim: int,
        layer_index: int,
        device: torch.device | str,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.layer_index = layer_index
        self.num_state_vectors = ONLINE_C128_NUM_STATE_VECTORS
        self.row_width = self.num_state_vectors * head_dim
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.num_banks = online_c128_num_banks(vllm_config)

        # [num_banks * max_num_reqs, num_state_vectors * head_dim], fp32.
        # run_max initialized to -inf, run_sum / run_weighted_sum to 0 so an
        # "empty" row is a valid online-softmax identity.
        self.state = torch.empty(
            (self.num_banks * self.max_num_reqs, self.row_width),
            dtype=ONLINE_C128_STATE_DTYPE,
            device=device,
        )
        self.reset_all()

    @property
    def device(self) -> torch.device:
        return self.state.device

    def bank_row_slice(self, bank_id: int) -> torch.Tensor:
        """View of all request rows for a given bank."""
        start = bank_id * self.max_num_reqs
        return self.state[start : start + self.max_num_reqs]

    def reset_all(self) -> None:
        """Reset every row to the empty online-softmax identity."""
        max_view = self.state[:, : self.head_dim]
        rest_view = self.state[:, self.head_dim :]
        max_view.fill_(float("-inf"))
        rest_view.zero_()

    def reset_rows(self, bank_id: int, req_state_indices: torch.Tensor) -> None:
        """Reset specific request rows in a bank to the empty identity."""
        rows = bank_id * self.max_num_reqs + req_state_indices.to(self.state.device)
        self.state[rows, : self.head_dim] = float("-inf")
        self.state[rows, self.head_dim :] = 0.0


# Module-level registry of all live online-state instances (one per compressor
# layer). Populated at model init; lets the model runner drive the MTP
# begin_verify / commit transition across every layer without threading a layer
# list through the forward signature.
_ONLINE_C128_STATES: "list[DeepseekOnlineC128State]" = []

# Shared fixed-address scratch for the per-boundary compressed KV output. The
# FULL decode cudagraph captures the compressor inside the graph, so the
# ``compressed_kv`` the merge kernel writes (and the store kernel reads) must be
# a stable pointer rather than a fresh ``torch.empty`` each step. One buffer is
# shared across all layers because each layer fully produces+consumes it within
# its own forward (FULL cudagraph replay is serial on the model stream). The
# eager / PIECEWISE path keeps allocating per-step (concurrency-safe), so this
# is only used on the FULL path.
_ONLINE_C128_COMPRESSED_KV: "torch.Tensor | None" = None


def ensure_online_c128_compressed_kv(
    max_num_tokens: int, head_dim: int, device: torch.device | str
) -> None:
    """Allocate the shared fixed-address compressed-KV scratch at model init.

    Sized to the maximum batched token count so any FULL cudagraph capture size
    slices a fixed prefix (stable base pointer). Called from the compressor
    constructor before any capture happens.
    """
    global _ONLINE_C128_COMPRESSED_KV
    buf = _ONLINE_C128_COMPRESSED_KV
    if (
        buf is None
        or buf.shape[0] < max_num_tokens
        or buf.shape[1] != head_dim
        or str(buf.device) != str(device)
    ):
        _ONLINE_C128_COMPRESSED_KV = torch.empty(
            (max_num_tokens, head_dim),
            dtype=ONLINE_C128_STATE_DTYPE,
            device=device,
        )


def online_c128_compressed_kv(num_tokens: int) -> torch.Tensor:
    """Fixed-address compressed-KV scratch sliced to ``num_tokens`` (FULL path)."""
    assert _ONLINE_C128_COMPRESSED_KV is not None, (
        "online C128 compressed-kv scratch not allocated; "
        "ensure_online_c128_compressed_kv must run at model init."
    )
    return _ONLINE_C128_COMPRESSED_KV[:num_tokens]


# MTP verify mode flag. The model runner sets this for the target-verify forward
# so the compressor uses the transactional candidate-chain planner instead of
# the committed bank0 planner. Reset after commit.
_ONLINE_C128_VERIFY_ACTIVE = False


def register_online_c128_state(state: DeepseekOnlineC128State) -> None:
    _ONLINE_C128_STATES.append(state)


def get_online_c128_states() -> "list[DeepseekOnlineC128State]":
    return _ONLINE_C128_STATES


def clear_online_c128_states() -> None:
    global _ONLINE_C128_COMPRESSED_KV
    _ONLINE_C128_STATES.clear()
    _ONLINE_C128_COMPRESSED_KV = None


def begin_online_c128_verify() -> None:
    """Mark the upcoming target forward as an MTP verify (candidate-chain)."""
    global _ONLINE_C128_VERIFY_ACTIVE
    _ONLINE_C128_VERIFY_ACTIVE = True


def end_online_c128_verify() -> None:
    global _ONLINE_C128_VERIFY_ACTIVE
    _ONLINE_C128_VERIFY_ACTIVE = False


def online_c128_verify_active() -> bool:
    return _ONLINE_C128_VERIFY_ACTIVE


def commit_all_online_c128_verify(
    req_state_indices: torch.Tensor,
    accepted_len: torch.Tensor,
    final_seq_len: torch.Tensor,
    compress_ratio: int = ONLINE_C128_COMPRESS_RATIO,
) -> None:
    """Commit accepted MTP candidates into bank0 for every online layer."""
    for state in _ONLINE_C128_STATES:
        commit_online_c128_verify(
            run_state=state.state,
            req_state_indices=req_state_indices,
            accepted_len=accepted_len,
            final_seq_len=final_seq_len,
            max_num_reqs=state.max_num_reqs,
            head_dim=state.head_dim,
            compress_ratio=compress_ratio,
        )


# ---------------------------------------------------------------------------
# Segment planning (prefill / chunked prefill)
# ---------------------------------------------------------------------------
#
# Planned path splits emit and bank-update launches to avoid bank row races.

# Descriptor columns (int32):
#   0: row_base    - first step-row index for this segment
#   1: num_rows    - number of step rows in this segment (1..128)
#   2: read_row    - absolute state row to seed the accumulator from, or -1
#   3: emit_token  - step-row index to write compressed_kv to, or -1
#   4: write_row   - absolute state row to write the merged carry to, or -1
SEGMENT_NUM_COLS = 5


class OnlineC128Plan:
    """Host-built segment plan for one forward step (prefill / chunked)."""

    def __init__(
        self,
        emit_segments: torch.Tensor,
        update_segments: torch.Tensor,
        reset_rows: torch.Tensor,
    ):
        # [num_emit, 5], [num_update, 5], [num_reset]
        self.emit_segments = emit_segments
        self.update_segments = update_segments
        self.reset_rows = reset_rows

    @property
    def is_empty(self) -> bool:
        return (
            self.emit_segments.shape[0] == 0
            and self.update_segments.shape[0] == 0
            and self.reset_rows.shape[0] == 0
        )


def plan_online_c128_segments(
    query_start_loc_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    req_state_indices_cpu: np.ndarray,
    max_num_reqs: int,
    device: torch.device | str,
    bank_id: int = 0,
    compress_ratio: int = ONLINE_C128_COMPRESS_RATIO,
    req_mask: np.ndarray | None = None,
) -> OnlineC128Plan:
    """Build emit/update/reset segments from CPU batch metadata."""
    num_reqs = len(req_state_indices_cpu)
    bank_base = bank_id * max_num_reqs
    emit: list[list[int]] = []
    update: list[list[int]] = []
    reset: list[int] = []

    for req in range(num_reqs):
        if req_mask is not None and not bool(req_mask[req]):
            continue
        rsi = int(req_state_indices_cpu[req])
        if rsi < 0:
            continue
        bank_row = bank_base + rsi
        row_start = int(query_start_loc_cpu[req])
        row_end = int(query_start_loc_cpu[req + 1])
        query_len = row_end - row_start
        if query_len <= 0:
            continue
        seq_end = int(seq_lens_cpu[req])  # exclusive; last pos = seq_end - 1
        first_pos = seq_end - query_len  # absolute position of row_start
        # tokens already accumulated into bank0 for the current open chunk.
        carry_len = first_pos % compress_ratio

        cur_row = row_start
        cur_pos = first_pos
        seeded_from_bank = carry_len > 0
        # Distance (in tokens) to the next chunk boundary from cur_pos.
        to_boundary = compress_ratio - (cur_pos % compress_ratio)

        last_was_boundary = False
        while cur_row < row_end:
            seg_rows = min(to_boundary, row_end - cur_row)
            closes_chunk = seg_rows == to_boundary
            read_row = bank_row if seeded_from_bank else -1
            if closes_chunk:
                emit_token = cur_row + seg_rows - 1
                # emit phase: read carry (if any) read-only, write compressed_kv.
                emit.append([cur_row, seg_rows, read_row, emit_token, -1])
                last_was_boundary = True
            else:
                # trailing partial: write the new carry to bank0.
                update.append([cur_row, seg_rows, read_row, -1, bank_row])
                last_was_boundary = False
            cur_row += seg_rows
            cur_pos += seg_rows
            # Subsequent chunks within this step start fresh (no carry).
            seeded_from_bank = False
            to_boundary = compress_ratio

        if last_was_boundary:
            # Step ended exactly on a boundary: clear bank0 for the next step.
            reset.append(bank_row)

    emit_t = torch.tensor(
        emit if emit else [], dtype=torch.int32, device=device
    ).reshape(-1, SEGMENT_NUM_COLS)
    update_t = torch.tensor(
        update if update else [], dtype=torch.int32, device=device
    ).reshape(-1, SEGMENT_NUM_COLS)
    reset_t = torch.tensor(reset, dtype=torch.int32, device=device)
    return OnlineC128Plan(emit_t, update_t, reset_t)


# ---------------------------------------------------------------------------
# MTP transactional candidate banks
# ---------------------------------------------------------------------------
# Verify token j reads candidate bank j and writes bank j+1; launch by step.


def plan_online_c128_verify(
    query_start_loc_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    req_state_indices_cpu: np.ndarray,
    max_num_reqs: int,
    device: torch.device | str,
    compress_ratio: int = ONLINE_C128_COMPRESS_RATIO,
    req_mask: np.ndarray | None = None,
    max_query_len: int | None = None,
) -> list[torch.Tensor]:
    """Build MTP verify segment lists, one launch per candidate step."""
    num_reqs = len(req_state_indices_cpu)
    # segs_by_step[j] collects the j-th verify token across all requests.
    segs_by_step: list[list[list[int]]] = []
    for req in range(num_reqs):
        if req_mask is not None and not bool(req_mask[req]):
            continue
        rsi = int(req_state_indices_cpu[req])
        if rsi < 0:
            continue
        row_start = int(query_start_loc_cpu[req])
        row_end = int(query_start_loc_cpu[req + 1])
        query_len = row_end - row_start
        if query_len <= 0:
            continue
        if max_query_len is not None and query_len > max_query_len:
            raise ValueError(
                "C128 online MTP verify query length exceeds allocated "
                f"candidate banks: query_len={query_len}, max_query_len="
                f"{max_query_len}."
            )
        seq_end = int(seq_lens_cpu[req])
        first_pos = seq_end - query_len
        seed_from_bank0 = first_pos % compress_ratio != 0
        for j in range(query_len):
            token = row_start + j
            pos = first_pos + j
            # Aligned starts must not read stale bank0.
            read_row = (
                j * max_num_reqs + rsi
                if j > 0 or seed_from_bank0
                else -1
            )
            write_row = (j + 1) * max_num_reqs + rsi
            closes = (pos + 1) % compress_ratio == 0
            emit_token = token if closes else -1
            if j >= len(segs_by_step):
                segs_by_step.append([])
            # row_base=token, num_rows=1, read_row, emit_token, write_row
            segs_by_step[j].append([token, 1, read_row, emit_token, write_row])

    return [
        torch.tensor(segs, dtype=torch.int32, device=device).reshape(
            -1, SEGMENT_NUM_COLS
        )
        for segs in segs_by_step
    ]


@triton.jit
def _commit_verify_kernel(
    run_state_ptr,
    run_state_stride,
    req_state_indices_ptr,  # [num_reqs] persistent slot per batch req (-1 pad)
    accepted_len_ptr,  # [num_reqs] = query_len - num_rejected
    final_seq_len_ptr,  # [num_reqs] = base_seq_len + accepted_len
    max_num_reqs,
    row_width,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Commit the accepted MTP candidate bank into bank0 (one program / req)."""
    req = tl.program_id(0)
    rsi = tl.load(req_state_indices_ptr + req)
    if rsi < 0:
        return
    accepted = tl.load(accepted_len_ptr + req)
    final_seq_len = tl.load(final_seq_len_ptr + req)

    bank0_row = rsi
    dst_base = bank0_row * run_state_stride

    if final_seq_len % COMPRESS_RATIO == 0:
        # Chunk closed exactly on the accepted boundary: clear bank0.
        for off in tl.range(0, row_width, BLOCK_SIZE):
            block = off + tl.arange(0, BLOCK_SIZE)
            mask = block < row_width
            # run_max (first HEAD_DIM cols) -> -inf, rest -> 0.
            init = tl.where(block < HEAD_DIM, -float("inf"), 0.0)
            tl.store(run_state_ptr + dst_base + block, init, mask=mask)
        return

    # Copy candidate bank `accepted` -> bank0.
    src_row = accepted * max_num_reqs + rsi
    src_base = src_row * run_state_stride
    for off in tl.range(0, row_width, BLOCK_SIZE):
        block = off + tl.arange(0, BLOCK_SIZE)
        mask = block < row_width
        vals = tl.load(run_state_ptr + src_base + block, mask=mask)
        tl.store(run_state_ptr + dst_base + block, vals, mask=mask)


def commit_online_c128_verify(
    run_state: torch.Tensor,
    req_state_indices: torch.Tensor,
    accepted_len: torch.Tensor,
    final_seq_len: torch.Tensor,
    max_num_reqs: int,
    head_dim: int,
    compress_ratio: int = ONLINE_C128_COMPRESS_RATIO,
) -> None:
    """Commit accepted MTP candidate banks into bank0."""
    num_reqs = req_state_indices.shape[0]
    if num_reqs == 0:
        return
    row_width = run_state.shape[1]
    _commit_verify_kernel[(num_reqs,)](
        run_state,
        run_state.stride(0),
        req_state_indices,
        accepted_len,
        final_seq_len,
        max_num_reqs,
        row_width,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_SIZE=1024,
        HEAD_DIM=head_dim,
    )
