# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental, synchronous DP prefill grants for one shared EP domain."""

import hashlib
import json
import math
from dataclasses import dataclass, field

BUCKETS = ("short", "medium", "long", "very_long")


@dataclass(frozen=True)
class StageCost:
    private_ms_per_row: float
    scan_ms_per_pair: float
    ep_ms_per_token: float
    draft_ms_per_row: float
    fixed_ms: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in vars(self).values()):
            raise ValueError("Stage costs must be finite and nonnegative")
        if self.private_ms_per_row + self.ep_ms_per_token <= 0:
            raise ValueError("A stage must have a positive row cost")


@dataclass(frozen=True)
class SharedEPBudgetConfig:
    """Explicit calibrated profile, not a built-in latency promise.

    Coefficients describe private PCP rows, global history scans and the sum
    of actual EP rows across DPs. `draft_ms_per_row` includes sampling/draft.
    """

    stages: tuple[StageCost, ...]
    calibration_id: str
    dp_size: int
    pcp_size: int
    history_limit: int
    query_limit: int
    protect_ms: float
    throughput_ms: float
    quantum: int = 128
    aging_steps: int = 8
    error_margin: float = 1.25
    max_relative_error: float = 0.25
    observed_relative_error: float = 0.0

    def __post_init__(self):
        if not self.stages or not self.calibration_id:
            raise ValueError("A cost profile requires stages and calibration identity")
        for name in (
            "dp_size",
            "pcp_size",
            "history_limit",
            "query_limit",
            "quantum",
            "aging_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "protect_ms",
            "throughput_ms",
            "error_margin",
            "max_relative_error",
            "observed_relative_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.protect_ms <= self.throughput_ms or self.error_margin < 1:
            raise ValueError("Invalid time budget or error margin")
        if self.quantum % self.pcp_size:
            raise ValueError("The query quantum must align with PCP")

    @classmethod
    def from_dict(cls, values: dict) -> "SharedEPBudgetConfig":
        values = dict(values)
        values["stages"] = tuple(StageCost(**v) for v in values["stages"])
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        from dataclasses import asdict

        data = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()

    def cost_ms(
        self,
        tokens: list[list[int]],
        histories: list[list[int]],
        include_fixed: bool = True,
    ) -> float:
        if not any(any(row) for row in tokens):
            return 0.0
        costs = []
        for stage in self.stages:
            private = []
            for rank_tokens, rank_histories in zip(tokens, histories):
                total = 0.0
                for q, h in zip(rank_tokens, rank_histories):
                    rows = math.ceil(q / self.pcp_size)
                    total += (stage.private_ms_per_row + stage.draft_ms_per_row) * rows
                    # Max history plus triangular query scan is a conservative
                    # envelope when several requests share a bucket grant.
                    total += stage.scan_ms_per_pair * rows * (h + (q + 1) / 2)
                private.append(total)
            # Idle peers still execute one dummy row per PCP worker.
            ep_rows = sum(sum(row) or self.pcp_size for row in tokens)
            costs.append(
                max(private)
                + stage.ep_ms_per_token * ep_rows
                + (stage.fixed_ms if include_fixed else 0.0)
            )
        return max(costs) * self.error_margin


@dataclass(frozen=True)
class BucketDemand:
    tokens: int = 0
    history: int = 0


@dataclass(frozen=True)
class BudgetSnapshot:
    epoch: str
    wave: int
    step: int
    profile: str
    token_limit: int
    demands: tuple[BucketDemand, ...]
    charged_tokens: tuple[int, ...] = (0, 0, 0, 0)
    charged_histories: tuple[int, ...] = (0, 0, 0, 0)
    supported: bool = True


@dataclass
class BudgetDecision:
    grants: list[list[int]]
    order: tuple[int, ...]
    mode: str
    budget_ms: float
    predicted_ms: float
    fallback: str | None = None


@dataclass
class SharedEPBudget:
    config: SharedEPBudgetConfig
    deficits: list[float] = field(default_factory=lambda: [0.0] * 4)
    ages: list[int] = field(default_factory=lambda: [0] * 4)
    identity: tuple | None = None
    last_step: int = -1

    def decide(self, snapshots: list[BudgetSnapshot]) -> BudgetDecision:
        """All ranks evaluate the same bounded payload before the next step.

        Transport failure is fatal to the DP group, not a rank-local fallback.
        A model/identity mismatch visible to all ranks falls back collectively.
        A grant does not reserve KV, and denied allocations are not charged.
        """
        cfg = self.config
        empty = [[0] * 4 for _ in snapshots]

        def fallback(reason: str) -> BudgetDecision:
            self.identity = None
            self.last_step = -1
            self.deficits = [0.0] * 4
            self.ages = [0] * 4
            return BudgetDecision(empty, (0, 1, 2, 3), "fallback", 0, 0, reason)

        if len(snapshots) != cfg.dp_size:
            return fallback("dp_size")
        if any(s.profile != cfg.fingerprint for s in snapshots):
            return fallback("profile_mismatch")
        if len({(s.wave, s.step) for s in snapshots}) != 1:
            return fallback("wave_step_mismatch")
        identity = tuple(s.epoch for s in snapshots), snapshots[0].wave
        step = snapshots[0].step
        fresh_identity = identity != self.identity
        if fresh_identity:
            self.identity = identity
            self.last_step = -1
            self.deficits = [0.0] * 4
            self.ages = [0] * 4
        if step <= self.last_step:
            return fallback("stale_step")
        self.last_step = step
        if cfg.observed_relative_error > cfg.max_relative_error:
            return fallback("model_error")
        for s in snapshots:
            if not s.supported or s.token_limit <= 0:
                return fallback("unsupported_shape")
            if not (
                len(s.demands) == len(s.charged_tokens) == len(s.charged_histories) == 4
            ):
                return fallback("invalid_payload")
            if any(
                d.tokens < 0 or not 0 <= d.history <= cfg.history_limit
                for d in s.demands
            ):
                return fallback("history_out_of_range")
            if any(q < 0 or q > s.token_limit for q in s.charged_tokens):
                return fallback("invalid_charge")
            if sum(s.charged_tokens) > s.token_limit:
                return fallback("invalid_charge")
            if any(not 0 <= h <= cfg.history_limit for h in s.charged_histories):
                return fallback("invalid_charge_history")

        active = [i for i in range(4) if any(s.demands[i].tokens for s in snapshots)]
        for i in range(4):
            if i not in active:
                self.deficits[i] = 0
                self.ages[i] = 0
                continue
            charged = [
                [s.charged_tokens[j] if j == i else 0 for j in range(4)]
                for s in snapshots
            ]
            histories = [list(s.charged_histories) for s in snapshots]
            service = (
                0.0
                if fresh_identity
                else cfg.cost_ms(charged, histories, include_fixed=False)
            )
            self.deficits[i] = max(-cfg.throughput_ms, self.deficits[i] - service)
            self.ages[i] = (
                0
                if service or fresh_identity
                else min(self.ages[i] + 1, cfg.aging_steps * 2)
            )

        urgent = any(self.ages[i] >= cfg.aging_steps for i in active)
        mode = "protect" if 0 in active and not urgent else "throughput"
        budget = cfg.protect_ms if mode == "protect" else cfg.throughput_ms
        for i in active:
            self.deficits[i] = min(
                cfg.throughput_ms * cfg.aging_steps,
                self.deficits[i] + budget / len(active),
            )

        order = tuple(
            sorted(
                range(4),
                key=lambda i: (
                    -(self.ages[i] >= cfg.aging_steps),
                    -self.deficits[i],
                    -self.ages[i],
                    i,
                ),
            )
        )
        grants = [[0] * 4 for _ in snapshots]
        histories = [[d.history for d in s.demands] for s in snapshots]
        for rank, s in enumerate(snapshots):
            for i in active:
                minimum = min(
                    cfg.quantum, s.demands[i].tokens, s.token_limit, cfg.query_limit
                )
                probe = [[0] * 4 for _ in snapshots]
                probe[rank][i] = minimum
                if cfg.cost_ms(probe, histories) > budget:
                    return fallback("budget_below_quantum")
        virtual = list(self.deficits)
        while active:
            progressed = False
            for i in sorted(
                active,
                key=lambda i: (-(self.ages[i] >= cfg.aging_steps), -virtual[i], i),
            ):
                before = cfg.cost_ms(grants, histories)
                for offset in range(cfg.dp_size):
                    rank = (step + offset) % cfg.dp_size
                    s = snapshots[rank]
                    remaining = min(
                        s.demands[i].tokens - grants[rank][i],
                        s.token_limit - sum(grants[rank]),
                        cfg.query_limit - sum(grants[rank]),
                    )
                    delta = min(cfg.quantum, remaining)
                    if delta <= 0:
                        continue
                    grants[rank][i] += delta
                    if cfg.cost_ms(grants, histories) > budget:
                        grants[rank][i] -= delta
                    else:
                        progressed = True
                virtual[i] -= max(0, cfg.cost_ms(grants, histories) - before)
            if not progressed:
                break
        if active and not any(any(row) for row in grants):
            # A budget smaller than a legal chunk must not deadlock a request.
            return fallback("budget_below_quantum")
        return BudgetDecision(
            grants, order, mode, budget, cfg.cost_ms(grants, histories)
        )
