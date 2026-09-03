# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    _process_group_spans_multiple_nodes,
)

pytestmark = pytest.mark.cpu_test


def test_process_group_uses_actual_same_node_membership():
    cpu_group = object()
    tp_group = SimpleNamespace(cpu_group=cpu_group, device_group=object())

    with (
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "get_node_count",
            return_value=2,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce.get_tp_group",
            return_value=tp_group,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "in_the_same_node_as",
            return_value=[True, True, True, True, False, False, False, False],
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "dist.get_process_group_ranks",
            return_value=list(range(8)),
        ),
    ):
        assert _process_group_spans_multiple_nodes(cpu_group)


def test_process_group_allows_node_local_tp_group_in_multi_node_world():
    cpu_group = object()
    tp_group = SimpleNamespace(cpu_group=cpu_group, device_group=object())

    with (
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "get_node_count",
            return_value=2,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce.get_tp_group",
            return_value=tp_group,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "in_the_same_node_as",
            return_value=[True] * 8,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "dist.get_process_group_ranks",
            return_value=list(range(8, 16)),
        ),
    ):
        assert not _process_group_spans_multiple_nodes(cpu_group)


def test_process_group_mismatch_fails_closed():
    tp_group = SimpleNamespace(cpu_group=object(), device_group=object())

    with (
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce."
            "get_node_count",
            return_value=2,
        ),
        patch(
            "vllm.distributed.device_communicators.flashinfer_all_reduce.get_tp_group",
            return_value=tp_group,
        ),
    ):
        assert _process_group_spans_multiple_nodes(object())
