# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical context-parallel token and page ownership helpers."""

from collections.abc import Iterator

import torch


def _validate_cp(cp_world_size: int, interleave_size: int) -> None:
    if cp_world_size <= 0:
        raise ValueError("cp_world_size must be positive")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive")


def _validate_rank(cp_rank: int, cp_world_size: int) -> None:
    if not 0 <= cp_rank < cp_world_size:
        raise ValueError(f"cp_rank must be in [0, {cp_world_size}), got {cp_rank}")


def get_cp_token_owner(
    global_position: int,
    cp_world_size: int,
    interleave_size: int,
) -> int:
    """Return the context-parallel rank that owns a global token position."""
    _validate_cp(cp_world_size, interleave_size)
    if global_position < 0:
        raise ValueError("global_position must be non-negative")
    return (global_position // interleave_size) % cp_world_size


def get_cp_local_position(
    global_position: int,
    cp_world_size: int,
    interleave_size: int,
) -> int:
    """Map a global token position to its compact owner-local position."""
    _validate_cp(cp_world_size, interleave_size)
    if global_position < 0:
        raise ValueError("global_position must be non-negative")
    cycle_size = cp_world_size * interleave_size
    return (
        global_position // cycle_size * interleave_size
        + global_position % interleave_size
    )


def get_cp_global_position(
    cp_rank: int,
    local_position: int,
    cp_world_size: int,
    interleave_size: int,
) -> int:
    """Map an owner-local token position back to its global position."""
    _validate_cp(cp_world_size, interleave_size)
    _validate_rank(cp_rank, cp_world_size)
    if local_position < 0:
        raise ValueError("local_position must be non-negative")
    local_cycle, cycle_offset = divmod(local_position, interleave_size)
    return (
        local_cycle * cp_world_size * interleave_size
        + cp_rank * interleave_size
        + cycle_offset
    )


def map_cp_positions(
    global_positions: torch.Tensor,
    cp_world_size: int,
    interleave_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized global-position mapping for runtime metadata paths."""
    _validate_cp(cp_world_size, interleave_size)
    cycle_size = cp_world_size * interleave_size
    owners = global_positions.div(interleave_size, rounding_mode="floor").remainder(
        cp_world_size
    )
    local_positions = global_positions.div(
        cycle_size, rounding_mode="floor"
    ) * interleave_size + global_positions.remainder(interleave_size)
    return owners, local_positions


def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    cp_world_size: int = 1,
    cp_rank: int | None = None,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return compact sequence lengths for one or every CP rank."""
    _validate_cp(cp_world_size, interleave_size)
    if cp_rank is not None:
        _validate_rank(cp_rank, cp_world_size)

    seq_lens_i32 = seq_lens.to(torch.int32)
    if cp_rank is None:
        rank_offsets = torch.arange(
            cp_world_size,
            dtype=torch.int32,
            device=seq_lens.device,
        ).view(
            *((1,) * seq_lens_i32.dim()),
            cp_world_size,
        )
        seq_lens_tiled = seq_lens_i32.unsqueeze(-1)
    else:
        rank_offsets = torch.tensor(
            cp_rank,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        seq_lens_tiled = seq_lens_i32

    cycle_size = cp_world_size * interleave_size
    full_cycles, remainder = (
        torch.div(
            seq_lens_tiled,
            cycle_size,
            rounding_mode="floor",
        ),
        seq_lens_tiled.remainder(cycle_size),
    )
    rank_tail = torch.clamp(
        remainder - rank_offsets * interleave_size,
        0,
        interleave_size,
    )
    return full_cycles * interleave_size + rank_tail


def get_cp_local_seq_len(
    seq_len: int,
    cp_world_size: int,
    cp_rank: int,
    interleave_size: int,
) -> int:
    """Scalar equivalent of :func:`get_cp_local_seq_lens`."""
    _validate_cp(cp_world_size, interleave_size)
    _validate_rank(cp_rank, cp_world_size)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    cycle_size = cp_world_size * interleave_size
    full_cycles, remainder = divmod(seq_len, cycle_size)
    rank_tail = min(
        max(remainder - cp_rank * interleave_size, 0),
        interleave_size,
    )
    return full_cycles * interleave_size + rank_tail


def global_page_to_local_page(
    global_page: int,
    cp_world_size: int,
    page_size: int,
    interleave_size: int,
) -> tuple[int, int]:
    """Map a global page to ``(owner, compact local page)``.

    The production contract initially requires one ownership interleave per
    physical page.
    """
    _validate_cp(cp_world_size, interleave_size)
    if page_size != interleave_size:
        raise ValueError("CP page mapping requires page_size == interleave_size")
    if global_page < 0:
        raise ValueError("global_page must be non-negative")
    return global_page % cp_world_size, global_page // cp_world_size


def local_page_to_global_page(
    cp_rank: int,
    local_page: int,
    cp_world_size: int,
    page_size: int,
    interleave_size: int,
) -> int:
    """Map an owner-local physical page back to its global page."""
    _validate_cp(cp_world_size, interleave_size)
    _validate_rank(cp_rank, cp_world_size)
    if page_size != interleave_size:
        raise ValueError("CP page mapping requires page_size == interleave_size")
    if local_page < 0:
        raise ValueError("local_page must be non-negative")
    return local_page * cp_world_size + cp_rank


def get_suffix_global_page_range(
    total_tokens: int,
    external_start_token: int,
    page_size: int,
) -> range:
    """Return global pages touched by an exact suffix token range."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if total_tokens < 0 or not 0 <= external_start_token <= total_tokens:
        raise ValueError("suffix token range is invalid")
    if external_start_token == total_tokens:
        return range(0)
    first_page = external_start_token // page_size
    end_page = (total_tokens + page_size - 1) // page_size
    return range(first_page, end_page)


def iter_owned_suffix_pages(
    total_tokens: int,
    external_start_token: int,
    cp_world_size: int,
    cp_rank: int,
    page_size: int,
    interleave_size: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(global_page, local_page)`` pairs owned by one CP rank."""
    _validate_rank(cp_rank, cp_world_size)
    for global_page in get_suffix_global_page_range(
        total_tokens,
        external_start_token,
        page_size,
    ):
        owner, local_page = global_page_to_local_page(
            global_page,
            cp_world_size,
            page_size,
            interleave_size,
        )
        if owner == cp_rank:
            yield global_page, local_page
