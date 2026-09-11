# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.cp_mapping import (
    get_cp_global_position,
    get_cp_local_position,
    get_cp_local_seq_len,
    get_cp_local_seq_lens,
    get_cp_token_owner,
    get_suffix_global_page_range,
    global_page_to_local_page,
    iter_owned_suffix_pages,
    local_page_to_global_page,
    map_cp_positions,
)

BOUNDARY_LENGTHS = (0, 1, 127, 128, 129, 255, 256, 257, 3563, 3564)


@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
@pytest.mark.parametrize("interleave_size", (1, 128))
@pytest.mark.parametrize("seq_len", BOUNDARY_LENGTHS)
def test_token_mapping_is_bijective_and_complete(
    cp_world_size: int,
    interleave_size: int,
    seq_len: int,
) -> None:
    local_positions = [[] for _ in range(cp_world_size)]

    for global_position in range(seq_len):
        owner = get_cp_token_owner(
            global_position,
            cp_world_size,
            interleave_size,
        )
        local_position = get_cp_local_position(
            global_position,
            cp_world_size,
            interleave_size,
        )
        assert (
            get_cp_global_position(
                owner,
                local_position,
                cp_world_size,
                interleave_size,
            )
            == global_position
        )
        local_positions[owner].append(local_position)

    for cp_rank, positions in enumerate(local_positions):
        assert positions == list(range(len(positions)))
        assert len(positions) == get_cp_local_seq_len(
            seq_len,
            cp_world_size,
            cp_rank,
            interleave_size,
        )
    assert sum(map(len, local_positions)) == seq_len


@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
@pytest.mark.parametrize("interleave_size", (1, 128))
def test_tensor_mapping_matches_scalar_oracle(
    cp_world_size: int,
    interleave_size: int,
) -> None:
    positions = torch.tensor(BOUNDARY_LENGTHS[1:], dtype=torch.int64) - 1

    owners, local_positions = map_cp_positions(
        positions,
        cp_world_size,
        interleave_size,
    )

    assert owners.tolist() == [
        get_cp_token_owner(int(position), cp_world_size, interleave_size)
        for position in positions
    ]
    assert local_positions.tolist() == [
        get_cp_local_position(int(position), cp_world_size, interleave_size)
        for position in positions
    ]


@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
@pytest.mark.parametrize("interleave_size", (1, 128))
def test_tensor_local_lengths_match_scalar_oracle(
    cp_world_size: int,
    interleave_size: int,
) -> None:
    seq_lens = torch.tensor(BOUNDARY_LENGTHS, dtype=torch.int64)

    all_ranks = get_cp_local_seq_lens(
        seq_lens,
        cp_world_size=cp_world_size,
        interleave_size=interleave_size,
    )
    assert all_ranks.shape == (len(BOUNDARY_LENGTHS), cp_world_size)
    assert torch.equal(
        all_ranks.sum(dim=-1),
        seq_lens.to(torch.int32),
    )

    for cp_rank in range(cp_world_size):
        expected = [
            get_cp_local_seq_len(
                seq_len,
                cp_world_size,
                cp_rank,
                interleave_size,
            )
            for seq_len in BOUNDARY_LENGTHS
        ]
        assert all_ranks[:, cp_rank].tolist() == expected
        assert (
            get_cp_local_seq_lens(
                seq_lens,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                interleave_size=interleave_size,
            ).tolist()
            == expected
        )


@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
def test_page_mapping_is_bijective(cp_world_size: int) -> None:
    for global_page in range(32):
        owner, local_page = global_page_to_local_page(
            global_page,
            cp_world_size,
            page_size=128,
            interleave_size=128,
        )
        assert (
            local_page_to_global_page(
                owner,
                local_page,
                cp_world_size,
                page_size=128,
                interleave_size=128,
            )
            == global_page
        )


@pytest.mark.parametrize(
    ("total_tokens", "external_start_token", "expected"),
    [
        (0, 0, []),
        (1, 0, [0]),
        (128, 0, [0]),
        (129, 128, [1]),
        (257, 127, [0, 1, 2]),
        (3563, 256, list(range(2, 28))),
        (3564, 3563, [27]),
    ],
)
def test_suffix_global_page_range(
    total_tokens: int,
    external_start_token: int,
    expected: list[int],
) -> None:
    assert (
        list(
            get_suffix_global_page_range(
                total_tokens,
                external_start_token,
                page_size=128,
            )
        )
        == expected
    )


def test_owned_suffix_pages_partition_global_pages() -> None:
    expected = list(get_suffix_global_page_range(3564, 127, page_size=128))
    owned = [
        pair
        for rank in range(2)
        for pair in iter_owned_suffix_pages(
            total_tokens=3564,
            external_start_token=127,
            cp_world_size=2,
            cp_rank=rank,
            page_size=128,
            interleave_size=128,
        )
    ]

    assert sorted(global_page for global_page, _ in owned) == expected
    assert len({global_page for global_page, _ in owned}) == len(expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda: get_cp_token_owner(0, 0, 128),
        lambda: get_cp_local_position(-1, 2, 128),
        lambda: get_cp_global_position(2, 0, 2, 128),
        lambda: global_page_to_local_page(0, 2, 64, 128),
        lambda: get_suffix_global_page_range(1, 2, 128),
    ],
)
def test_invalid_mapping_contract_fails_closed(call) -> None:
    with pytest.raises(ValueError):
        call()
