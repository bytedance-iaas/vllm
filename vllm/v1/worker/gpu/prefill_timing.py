# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded CUDA event samples for diagnostic Prefill cost calibration."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TimingStep:
    metadata: dict[str, Any]
    target_start: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(enable_timing=True)
    )
    target_end: torch.cuda.Event | None = None
    sample_start: torch.cuda.Event | None = None
    sample_end: torch.cuda.Event | None = None


class PrefillTimingRecorder:
    """No synchronization in the serving path; read only completed samples.

    Events measure one CUDA stream's elapsed time including its dependencies
    and host launch gaps. They are neither kernel sums nor removable latency.
    The caller owns target/sample pairing and the metadata's source identity.
    """

    def __init__(self):
        self.active = False
        self.limit = 0
        self.records: list[TimingStep] = []
        self.pending_sample: TimingStep | None = None

    def start(self, max_steps: int) -> None:
        if type(max_steps) is not int or not 1 <= max_steps <= 64:
            raise ValueError("max_steps must be between 1 and 64")
        if self.active:
            raise RuntimeError("A timing capture is already active")
        if any(
            r.target_end is None
            or not r.target_end.query()
            or (
                r.sample_start is not None
                and (r.sample_end is None or not r.sample_end.query())
            )
            for r in self.records
        ):
            raise RuntimeError("The previous timing capture has unsettled samples")
        self.limit = max_steps
        self.records = []
        self.pending_sample = None
        self.active = True

    def begin_target(self, metadata: dict[str, Any]) -> TimingStep | None:
        self.pending_sample = None
        if not self.active or len(self.records) >= self.limit:
            return None
        step = TimingStep(dict(metadata))
        step.target_start.record()
        self.records.append(step)
        self.pending_sample = step
        return step

    def end_target(self, step: TimingStep | None) -> None:
        if step is not None:
            if step.target_end is not None:
                raise RuntimeError("A target step cannot be ended twice")
            step.target_end = torch.cuda.Event(enable_timing=True)
            step.target_end.record()

    def begin_sample(self) -> TimingStep | None:
        step, self.pending_sample = self.pending_sample, None
        if step is not None:
            step.sample_start = torch.cuda.Event(enable_timing=True)
            step.sample_start.record()
        return step

    def end_sample(self, step: TimingStep | None) -> None:
        if step is not None:
            if step.sample_end is not None:
                raise RuntimeError("A sampling step cannot be ended twice")
            step.sample_end = torch.cuda.Event(enable_timing=True)
            step.sample_end.record()

    def read(self, stop: bool = True) -> dict[str, Any]:
        if stop:
            self.active = False
        rows, pending = [], 0
        for index, step in enumerate(self.records):
            if (
                step.target_end is None
                or not step.target_end.query()
                or (
                    step.sample_start is not None
                    and (step.sample_end is None or not step.sample_end.query())
                )
            ):
                pending += 1
                continue
            sample_ms = (
                step.sample_start.elapsed_time(step.sample_end)
                if step.sample_start is not None and step.sample_end is not None
                else None
            )
            rows.append(
                {
                    "local_sample": index,
                    **step.metadata,
                    "target_elapsed_ms": step.target_start.elapsed_time(
                        step.target_end
                    ),
                    "sample_and_draft_elapsed_ms": sample_ms,
                }
            )
        return {
            "active": self.active,
            "limit": self.limit,
            "pending": pending,
            "recorded": len(self.records),
            "samples": rows,
        }
