#!/usr/bin/env python3
"""Calibrate and validate typed DeepSeek V4.1 retained A-control evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA = 1
QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


@dataclass
class EvidenceRun:
    manifest_path: Path
    manifest: dict[str, Any]
    payload: dict[str, torch.Tensor]
    leaves: dict[str, dict[str, Any]]

    @property
    def rank(self) -> int:
        return int(self.manifest["dp_rank"])

    @property
    def ordinal(self) -> int:
        return int(self.manifest["ordinal"])

    @property
    def kind(self) -> str:
        return str(self.manifest["run_kind"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(path: Path) -> EvidenceRun:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError(f"{path}: incomplete or unsupported manifest")
    payload_path = path.parent / manifest["payload"]
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise ValueError(f"{path}: payload checksum mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    leaves = {}
    for leaf in manifest["leaves"]:
        key = f"{leaf['checkpoint']}:{leaf['path']}"
        if key in leaves:
            raise ValueError(f"{path}: duplicate leaf {key}")
        leaves[key] = leaf
    return EvidenceRun(path, manifest, payload, leaves)


def _load_directory(
    root: Path,
    *,
    allowed_kinds: set[str] | None = None,
    max_a_ordinal: int | None = None,
) -> dict[tuple[str, int, int], EvidenceRun]:
    runs = {}
    for path in sorted(root.glob("rank-*/*.json")):
        if path.name == "transaction-summary.json":
            continue
        try:
            kind, ordinal_text = path.stem.rsplit("-", 1)
            ordinal = int(ordinal_text)
        except ValueError as error:
            raise ValueError(f"{path}: invalid evidence filename") from error
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        if kind == "a" and max_a_ordinal is not None and ordinal > max_a_ordinal:
            continue
        run = _load_run(path)
        key = (run.kind, run.rank, run.ordinal)
        if key in runs:
            raise ValueError(f"duplicate run {key}")
        runs[key] = run
    if not runs:
        raise ValueError(f"no evidence manifests below {root}")
    return runs


def _load_summaries(root: Path) -> dict[int, dict[str, Any]]:
    summaries = {}
    for path in sorted(root.glob("rank-*/transaction-summary.json")):
        document = json.loads(path.read_text())
        if document.get("schema") != SCHEMA or not document.get("complete"):
            raise ValueError(f"{path}: incomplete transaction summary")
        rank = int(document["dp_rank"])
        if rank in summaries:
            raise ValueError(f"duplicate transaction summary for rank {rank}")
        summaries[rank] = document
    return summaries


def _validate_cohort(
    runs: dict[tuple[str, int, int], EvidenceRun],
    summaries: dict[int, dict[str, Any]],
    ranks: list[int],
) -> None:
    if set(summaries) != set(ranks):
        raise ValueError(
            f"transaction summaries do not match ranks: "
            f"summaries={sorted(summaries)} ranks={ranks}"
        )
    run_ids = {summary["run_id"] for summary in summaries.values()}
    if len(run_ids) != 1:
        raise ValueError(f"transaction run IDs differ: {sorted(run_ids)}")
    dp_sizes = {run.manifest["dp_size"] for run in runs.values()}
    if len(dp_sizes) != 1:
        raise ValueError("DP topology differs across evidence")
    dp_size = int(next(iter(dp_sizes)))
    if ranks != list(range(dp_size)):
        raise ValueError(f"expected ranks 0..{dp_size - 1}, found {ranks}")
    global_fields = ("run_id", "config_digest", "code_hashes", "model", "draft_model")
    for field in global_fields:
        values = {
            json.dumps(run.manifest.get(field), sort_keys=True) for run in runs.values()
        }
        if len(values) != 1:
            raise ValueError(f"evidence field differs across cohort: {field}")
    for rank in ranks:
        rank_runs = [run for run in runs.values() if run.rank == rank]
        for field in (
            "worker_incarnation",
            "fixture_id",
            "fixture_digest",
            "input_digest",
            "active_role",
            "s0_digest",
            "source_digest",
        ):
            values = {
                json.dumps(run.manifest.get(field), sort_keys=True) for run in rank_runs
            }
            if len(values) != 1:
                raise ValueError(f"rank {rank}: evidence field changed: {field}")
        graph_identities = set()
        for run in rank_runs:
            graph = run.manifest["graph"]
            if graph["callback_calls"] != 1 or graph["capture_replay_calls"] != 1:
                raise ValueError(f"{run.manifest_path}: graph was not replayed once")
            if graph["num_graphs"] < 1:
                raise ValueError(f"{run.manifest_path}: graph has no segments")
            graph_identities.add(
                (
                    graph["manager_id"],
                    graph["capture_id"],
                    graph["descriptor_rows"],
                    graph["num_graphs"],
                )
            )
        if len(graph_identities) != 1:
            raise ValueError(f"rank {rank}: selected graph identity changed")
        summary = summaries[rank]
        if not summary.get("live_restored"):
            raise ValueError(f"rank {rank}: LIVE restoration was not proven")
        if int(summary["completed_runs"]) < max(run.ordinal for run in rank_runs):
            raise ValueError(f"rank {rank}: transaction summary is incomplete")


def _leaf_schema(leaf: dict[str, Any]) -> dict[str, Any]:
    return {
        key: leaf.get(key)
        for key in (
            "checkpoint",
            "path",
            "family",
            "comparison",
            "dtype",
            "shape",
            "metadata",
        )
    }


def _envelope_key(leaf: dict[str, Any]) -> str:
    return json.dumps(
        {
            "checkpoint": leaf["checkpoint"],
            "family": leaf["family"],
            "dtype": leaf["dtype"],
        },
        sort_keys=True,
    )


def _exact_value(run: EvidenceRun, leaf: dict[str, Any]) -> Any:
    payload_key = leaf.get("payload_key")
    return leaf.get("value") if payload_key is None else run.payload[payload_key]


def _floating_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    denominator_floor: float,
) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("floating tensor shape or dtype mismatch")
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    left_finite = torch.isfinite(left64)
    right_finite = torch.isfinite(right64)
    if not bool(left_finite.all()) or not bool(right_finite.all()):
        raise ValueError("unexpected nonfinite floating evidence")
    difference = (left64 - right64).abs().reshape(-1)
    if difference.numel() == 0:
        return {
            "elements": 0,
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "abs_quantiles": {str(q): 0.0 for q in QUANTILES},
            "worst_flat_index": None,
            "worst_left": None,
            "worst_right": None,
        }
    max_abs, worst_index = difference.max(dim=0)
    denominator = max(
        float(torch.linalg.vector_norm(left64).item()),
        float(torch.linalg.vector_norm(right64).item()),
        denominator_floor,
    )
    relative_l2 = float(torch.linalg.vector_norm(difference).item()) / denominator
    quantiles = torch.quantile(
        difference,
        torch.tensor(QUANTILES, dtype=torch.float64),
    )
    index = int(worst_index.item())
    flat_left = left64.reshape(-1)
    flat_right = right64.reshape(-1)
    return {
        "elements": difference.numel(),
        "max_abs": float(max_abs.item()),
        "relative_l2": relative_l2,
        "abs_quantiles": {
            str(q): float(value) for q, value in zip(QUANTILES, quantiles.tolist())
        },
        "worst_flat_index": index,
        "worst_left": float(flat_left[index].item()),
        "worst_right": float(flat_right[index].item()),
    }


def _compare_runs(
    left: EvidenceRun,
    right: EvidenceRun,
    *,
    denominator_floor: float,
    envelope: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    if left.rank != right.rank:
        raise ValueError("comparisons must stay within one DP rank")
    if set(left.leaves) != set(right.leaves):
        missing_left = sorted(set(right.leaves) - set(left.leaves))
        missing_right = sorted(set(left.leaves) - set(right.leaves))
        raise ValueError(
            f"leaf schema differs: missing_left={missing_left[:4]} "
            f"missing_right={missing_right[:4]}"
        )
    floating = []
    exact_mismatches = []
    for key in sorted(left.leaves):
        left_leaf = left.leaves[key]
        right_leaf = right.leaves[key]
        if _leaf_schema(left_leaf) != _leaf_schema(right_leaf):
            raise ValueError(f"{key}: leaf schema differs")
        if left_leaf["comparison"] == "exact":
            left_value = _exact_value(left, left_leaf)
            right_value = _exact_value(right, right_leaf)
            equal = (
                torch.equal(left_value, right_value)
                if isinstance(left_value, torch.Tensor)
                and isinstance(right_value, torch.Tensor)
                else left_value == right_value
            )
            if not equal:
                exact_mismatches.append(key)
            continue
        if left_leaf["comparison"] != "floating":
            raise ValueError(f"{key}: invalid comparison class")
        left_tensor = left.payload[left_leaf["payload_key"]]
        right_tensor = right.payload[right_leaf["payload_key"]]
        metrics = _floating_metrics(
            left_tensor,
            right_tensor,
            denominator_floor=denominator_floor,
        )
        family = _envelope_key(left_leaf)
        passed = None
        if envelope is not None:
            threshold = envelope.get(family)
            if threshold is None:
                raise ValueError(f"{key}: frozen envelope is absent")
            passed = (
                metrics["max_abs"] <= threshold["max_abs"]
                and metrics["relative_l2"] <= threshold["relative_l2"]
            )
        floating.append(
            {
                "leaf": key,
                "envelope_key": family,
                "passed": passed,
                **metrics,
            }
        )
    return {
        "left": {
            "kind": left.kind,
            "rank": left.rank,
            "ordinal": left.ordinal,
        },
        "right": {
            "kind": right.kind,
            "rank": right.rank,
            "ordinal": right.ordinal,
        },
        "exact_mismatches": exact_mismatches,
        "floating": floating,
        "passed": not exact_mismatches
        and all(row["passed"] is not False for row in floating),
    }


def _require_matrix(
    runs: dict[tuple[str, int, int], EvidenceRun],
    *,
    a_runs: int,
) -> list[int]:
    ranks = sorted({rank for kind, rank, _ in runs if kind == "a"})
    if not ranks:
        raise ValueError("no A-control ranks found")
    expected_ordinals = set(range(1, a_runs + 1))
    for rank in ranks:
        found = {
            ordinal
            for kind, candidate_rank, ordinal in runs
            if kind == "a" and candidate_rank == rank
        }
        if found != expected_ordinals:
            raise ValueError(
                f"rank {rank}: expected A ordinals {sorted(expected_ordinals)}, "
                f"found {sorted(found)}"
            )
    return ranks


def calibrate(args: argparse.Namespace) -> None:
    runs = _load_directory(
        args.input_dir,
        allowed_kinds={"a"},
        max_a_ordinal=args.calibration_runs,
    )
    ranks = _require_matrix(runs, a_runs=args.calibration_runs)
    _validate_cohort(runs, _load_summaries(args.input_dir), ranks)
    envelope: dict[str, dict[str, float]] = {}
    pair_reports = []
    for rank in ranks:
        controls = [
            runs[("a", rank, ordinal)]
            for ordinal in range(1, args.calibration_runs + 1)
        ]
        for left, right in itertools.combinations(controls, 2):
            report = _compare_runs(
                left,
                right,
                denominator_floor=args.denominator_floor,
                envelope=None,
            )
            if report["exact_mismatches"]:
                raise ValueError(
                    f"calibration exact mismatch rank={rank}: "
                    f"{report['exact_mismatches'][:4]}"
                )
            pair_reports.append(report)
            for row in report["floating"]:
                threshold = envelope.setdefault(
                    row["envelope_key"],
                    {"max_abs": 0.0, "relative_l2": 0.0},
                )
                threshold["max_abs"] = max(threshold["max_abs"], row["max_abs"])
                threshold["relative_l2"] = max(
                    threshold["relative_l2"],
                    row["relative_l2"],
                )
    document = {
        "schema": SCHEMA,
        "calibration_runs": args.calibration_runs,
        "calibration_pairs_per_rank": (
            args.calibration_runs * (args.calibration_runs - 1) // 2
        ),
        "ranks": ranks,
        "denominator_floor": args.denominator_floor,
        "envelope": envelope,
        "pairs": pair_reports,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True).encode()
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": "calibrated",
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "families": len(envelope),
                "pairs": len(pair_reports),
            },
            sort_keys=True,
        )
    )


def validate(args: argparse.Namespace) -> None:
    runs = _load_directory(args.input_dir)
    envelope_document = json.loads(args.envelope.read_text())
    if envelope_document.get("schema") != SCHEMA:
        raise ValueError("unsupported envelope schema")
    calibration_runs = int(envelope_document["calibration_runs"])
    denominator_floor = float(envelope_document["denominator_floor"])
    envelope = envelope_document["envelope"]
    ranks = _require_matrix(runs, a_runs=args.a_runs)
    _validate_cohort(runs, _load_summaries(args.input_dir), ranks)
    reports = []
    failures = []
    for rank in ranks:
        controls = [runs[("a", rank, ordinal)] for ordinal in range(1, args.a_runs + 1)]
        production = runs.get(("production", rank, 0))
        if production is None:
            raise ValueError(f"rank {rank}: production evidence is absent")
        for left, right in itertools.combinations(controls, 2):
            report = _compare_runs(
                left,
                right,
                denominator_floor=denominator_floor,
                envelope=envelope,
            )
            reports.append(report)
            if not report["passed"]:
                failures.append(report)
        for control in controls:
            report = _compare_runs(
                production,
                control,
                denominator_floor=denominator_floor,
                envelope=envelope,
            )
            reports.append(report)
            if not report["passed"]:
                failures.append(report)
    expected_pairs = args.a_runs * (args.a_runs - 1) // 2
    document = {
        "schema": SCHEMA,
        "status": "passed" if not failures else "failed",
        "a_runs": args.a_runs,
        "calibration_runs": calibration_runs,
        "a_pairs_per_rank": expected_pairs,
        "production_comparisons_per_rank": args.a_runs,
        "ranks": ranks,
        "envelope_path": str(args.envelope),
        "envelope_sha256": _sha256(args.envelope),
        "failures": failures,
        "comparisons": reports,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(
            f"validation failed: {len(failures)} comparison(s); report={args.output}"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "ranks": ranks,
                "a_pairs_per_rank": expected_pairs,
                "production_comparisons_per_rank": args.a_runs,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--input-dir", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--calibration-runs", type=int, default=10)
    calibrate_parser.add_argument("--denominator-floor", type=float, default=1e-12)
    calibrate_parser.set_defaults(func=calibrate)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument("--envelope", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--a-runs", type=int, default=20)
    validate_parser.set_defaults(func=validate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
