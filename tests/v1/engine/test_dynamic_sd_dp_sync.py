# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import ParallelConfig
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore
from vllm.v1.engine.utils import (
    compute_dp_collective_config_hash,
    validate_dp_collective_config_hash,
)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


class _FakeScheduler:
    def __init__(self, pressures: list[int]) -> None:
        self.pressures = pressures
        self.overrides: list[int] = []

    def get_dynamic_sd_local_batch_pressure(self) -> int:
        assert self.pressures
        return self.pressures.pop(0)

    def set_dynamic_sd_batch_size_override(self, batch_size: int) -> None:
        self.overrides.append(batch_size)


def _make_core(
    *,
    pressures: list[int],
    interval: int = 8,
    global_max: bool = True,
) -> DPEngineCoreProc:
    core = DPEngineCoreProc.__new__(DPEngineCoreProc)
    core.scheduler = _FakeScheduler(pressures)
    core.dp_group = object()
    core.step_counter = 0
    core._dynamic_sd_cached_global_batch_pressure = None
    core.dynamic_sd_trace_path = None
    core.dynamic_sd_trace_fd = None
    core.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            uses_dynamic_sd_dp_global_max_policy=lambda: global_max,
            dynamic_sd_dp_sync_interval=interval,
        )
    )
    return core


def test_dynamic_sd_dp_pressure_sync_uses_cached_pressure(monkeypatch):
    core = _make_core(pressures=[0, 3, 9], interval=8)
    sync_results = [0, 24, 99]
    sync_calls: list[int] = []

    def fake_sync(dp_group, local_pressure):
        sync_calls.append(local_pressure)
        return sync_results.pop(0)

    monkeypatch.setattr(
        ParallelConfig,
        "sync_dynamic_sd_batch_pressure",
        staticmethod(fake_sync),
    )

    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)
    core.step_counter = 1
    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)
    core.step_counter = 2
    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)
    core.step_counter = 8
    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)

    assert sync_calls == [0, 3, 9]
    assert core.scheduler.overrides == [0, 24, 24, 99]


def test_dynamic_sd_dp_pressure_sync_interval_one_syncs_every_step(monkeypatch):
    core = _make_core(pressures=[2, 4, 6], interval=1)
    sync_calls: list[int] = []

    def fake_sync(dp_group, local_pressure):
        sync_calls.append(local_pressure)
        return local_pressure + 10

    monkeypatch.setattr(
        ParallelConfig,
        "sync_dynamic_sd_batch_pressure",
        staticmethod(fake_sync),
    )

    for step in range(3):
        core.step_counter = step
        DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)

    assert sync_calls == [2, 4, 6]
    assert core.scheduler.overrides == [12, 14, 16]


def test_dynamic_sd_dp_pressure_sync_noops_without_global_policy(monkeypatch):
    core = _make_core(pressures=[2], global_max=False)
    sync = Mock()
    monkeypatch.setattr(
        ParallelConfig,
        "sync_dynamic_sd_batch_pressure",
        staticmethod(sync),
    )

    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)

    sync.assert_not_called()
    assert core.scheduler.overrides == []


def test_dynamic_sd_dp_pressure_cache_reset_forces_next_sync(monkeypatch):
    core = _make_core(pressures=[2, 5], interval=8)
    sync_calls: list[int] = []

    def fake_sync(dp_group, local_pressure):
        sync_calls.append(local_pressure)
        return local_pressure + 20

    monkeypatch.setattr(
        ParallelConfig,
        "sync_dynamic_sd_batch_pressure",
        staticmethod(fake_sync),
    )

    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)
    core.step_counter = 1
    core._reset_dynamic_sd_batch_pressure_cache()
    DPEngineCoreProc._sync_dynamic_sd_batch_size_override(core)

    assert sync_calls == [2, 5]
    assert core.scheduler.overrides == [22, 25]


def test_feature_off_dummy_batch_preserves_legacy_no_arg_call():
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
    )
    core.scheduler = Mock()
    core.model_executor = Mock()

    EngineCore.execute_dummy_batch(core)

    core.scheduler.get_num_spec_tokens_to_schedule_for_dummy_batch.assert_not_called()
    core.model_executor.execute_dummy_batch.assert_called_once_with()


def test_dynamic_sd_dummy_batch_passes_runtime_k():
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: True
        ),
    )
    core.scheduler = Mock()
    core.scheduler.get_num_spec_tokens_to_schedule_for_dummy_batch.return_value = 3
    core.model_executor = Mock()

    EngineCore.execute_dummy_batch(core)

    core.model_executor.execute_dummy_batch.assert_called_once_with(3)


def test_executor_feature_off_dummy_rpc_has_no_positional_args():
    executor = Mock()

    Executor.execute_dummy_batch(executor)

    executor.collective_rpc.assert_called_once_with("execute_dummy_batch")


def test_multiproc_executor_feature_off_dummy_rpc_has_no_positional_args():
    executor = Mock(output_rank=2)

    MultiprocExecutor.execute_dummy_batch(executor)

    executor.collective_rpc.assert_called_once_with(
        "execute_dummy_batch",
        unique_reply_rank=2,
    )


def _dp_collective_hash(
    *,
    policy: str | None = "global_max",
    interval: int = 8,
    schedule: list[tuple[int, int, int]] | None = None,
) -> str:
    if schedule is None:
        schedule = [(1, 4, 3), (5, 16, 1)]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(compute_hash=lambda: "parallel"),
        speculative_config=SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: True,
            num_speculative_tokens_per_batch_size=schedule,
            dynamic_sd_dp_batch_policy=policy,
            dynamic_sd_dp_sync_interval=interval,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        num_speculative_tokens=3,
    )
    return compute_dp_collective_config_hash(config)


def test_dp_collective_hash_covers_dynamic_sd_collective_config():
    base_hash = _dp_collective_hash()

    assert _dp_collective_hash(policy=None) != base_hash
    assert _dp_collective_hash(interval=1) != base_hash
    assert _dp_collective_hash(schedule=[(1, 8, 3), (9, 16, 0)]) != base_hash


def test_dp_collective_hash_mismatch_fails_before_engine_ready():
    validate_dp_collective_config_hash("same", "same", 1)

    with pytest.raises(RuntimeError, match="Dynamic SD policy"):
        validate_dp_collective_config_hash("worker", "expected", 2)
