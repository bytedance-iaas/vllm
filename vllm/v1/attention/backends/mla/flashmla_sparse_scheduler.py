# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-request KV partitions for the SM90 sparse FlashMLA kernel."""

import torch


def single_split_flashmla_schedule(
    batch_size: int,
    query_tokens: int,
    query_heads: int,
    topk: int,
    sm_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every request's entire top-k to exactly one native partition.

    FlashMLA's SM90 API requires max(SMs // Q // (H // 64), 1) rows of
    32-byte DecodingSchedMeta. Keep that launch geometry, but never divide a
    request's KV reduction between partitions. Surplus partitions are idle;
    when requests outnumber partitions, each partition processes consecutive
    whole requests. A split prefix [0, 1, ..., B] makes the native combine
    kernel leave the directly written outputs untouched.
    """
    if batch_size <= 0 or query_tokens <= 0 or sm_count <= 0:
        raise ValueError("FlashMLA schedule dimensions must be positive")
    if query_heads not in (64, 128) or topk <= 0 or topk % 64:
        raise ValueError(
            "SM90 sparse FlashMLA requires 64/128 heads and top-k % 64 == 0"
        )

    partitions = max(sm_count // query_tokens // (query_heads // 64), 1)
    requests_per_partition = (batch_size + partitions - 1) // partitions
    start = (
        torch.arange(partitions, dtype=torch.int32, device=device)
        * requests_per_partition
    ).clamp_max(batch_size)
    end = (start + requests_per_partition).clamp_max(batch_size) - 1
    zero = torch.zeros_like(start)
    blocks = torch.full_like(start, topk // 64)
    # begin/end request, begin/end KV block, begin split, first/last split,
    # padding. begin_request == B is the native idle-partition sentinel.
    schedule = torch.stack((start, end, zero, blocks, zero, zero, zero, zero), dim=1)
    splits = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    return schedule, splits
