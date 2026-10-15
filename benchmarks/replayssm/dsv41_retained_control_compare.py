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
REQUIRED_CHECKPOINTS = {"l20", "target", "post_sampling", "s1"}
REQUIRED_PATHS = {
    "l20": {
        "boundary/hidden_states",
        "boundary/positions",
        "boundary/input_ids",
        "boundary/pre_mix",
        "boundary/post_mix",
        "boundary/res_mix",
        "boundary/residual",
        "boundary/aux_hidden_by_layer/__keys__/__type__",
        "boundary/remote_aux/__type__",
        "boundary/full_num_tokens",
    },
    "target": {
        "target/hidden_states",
        "target/aux_hidden_states/__type__",
        "target/mtp_pre_collapse",
    },
    "post_sampling": {
        "decisions/skipped",
        "decisions/num_sampled",
        "decisions/num_rejected",
    },
    "s1": {
        "decisions/skipped",
        "decisions/num_sampled",
        "decisions/num_rejected",
    },
}
NUMERIC_REGISTRY_PREFIXES = (
    "registry/model:_mtp_hidden_buffer",
    "registry/model:candidate_compact_logits_buffer",
    "registry/replay_graph:input:",
    "registry/speculator:draft_token_confidence_probs",
    "registry/speculator:_draft_scatter_buf",
    "registry/speculator:hidden_states",
    "registry/speculator:draft_logits",
    "registry/speculator:model:_mtp_hidden_buffer",
    "registry/acceptance:",
    "registry/adaptive:confidence_probs",
    "registry/adaptive:stale_gpu:",
    "registry/adaptive:stale_cpu:",
)
EXACT_FLOAT_REGISTRY_PREFIXES = (
    "registry/sampler:",
    "registry/speculator:temperature",
)
TRIMMABLE_REGISTRY_NAMES = {
    "replay_graph:input:0",
    "replay_graph:input:1",
    "replay_graph:input:2",
    "replay_graph:input:3",
    "replay_graph:input:4",
    "replay_graph:input:5",
    "replay_graph:input:6",
    "speculator:draft_tokens",
    "speculator:draft_token_confidence_probs",
    "speculator:idx_mapping",
    "speculator:_draft_scatter_buf",
}


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


def _safe_path_component(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _validate_container_children(
    path: Path,
    leaves: dict[str, dict[str, Any]],
) -> None:
    for key, leaf in leaves.items():
        leaf_path = leaf["path"]
        if not leaf_path.endswith("/__len__"):
            continue
        length = leaf.get("value")
        if not isinstance(length, int) or length < 0:
            raise ValueError(f"{path}: invalid container length at {key}")
        if length == 0:
            continue
        checkpoint = leaf["checkpoint"]
        base = leaf_path[: -len("/__len__")]
        for index in range(length):
            child = f"{checkpoint}:{base}/{index}"
            if not any(
                candidate == child or candidate.startswith(f"{child}/")
                for candidate in leaves
            ):
                raise ValueError(f"{path}: container child is absent: {child}")

        if not base.endswith("/__keys__"):
            continue
        mapping_base = base[: -len("/__keys__")]
        children = {
            candidate.split(":", 1)[1][len(mapping_base) + 1 :].split("/", 1)[0]
            for candidate in leaves
            if candidate.startswith(f"{checkpoint}:{mapping_base}/")
            and not candidate.startswith(f"{checkpoint}:{mapping_base}/__")
        }
        if len(children) < length:
            raise ValueError(
                f"{path}: mapping declares {length} keys but has "
                f"{len(children)} children"
            )


def _validate_evidence_contract(
    path: Path,
    manifest: dict[str, Any],
    leaves: dict[str, dict[str, Any]],
) -> None:
    contract = manifest.get("evidence_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: evidence contract is absent")
    checkpoint_paths = contract.get("checkpoint_paths")
    comparison_classes = contract.get("comparison_classes")
    registry_entries = contract.get("registry_entries")
    meaningful_counts = contract.get("meaningful_counts")
    graph_input_count = contract.get("graph_input_count")
    if not (
        isinstance(checkpoint_paths, dict)
        and isinstance(comparison_classes, dict)
        and isinstance(registry_entries, list)
        and isinstance(meaningful_counts, dict)
        and isinstance(graph_input_count, int)
    ):
        raise ValueError(f"{path}: evidence contract is incomplete")

    for checkpoint in REQUIRED_CHECKPOINTS:
        actual = sorted(
            leaf["path"] for leaf in leaves.values() if leaf["checkpoint"] == checkpoint
        )
        if actual != checkpoint_paths.get(checkpoint):
            raise ValueError(f"{path}: {checkpoint} inventory differs from contract")
    actual_classes = {key: leaf["comparison"] for key, leaf in leaves.items()}
    if actual_classes != comparison_classes:
        raise ValueError(f"{path}: comparison classes differ from contract")

    for checkpoint in ("post_sampling", "s1"):
        checkpoint_leaf_paths = {
            leaf["path"] for leaf in leaves.values() if leaf["checkpoint"] == checkpoint
        }
        for name in registry_entries:
            root = f"registry/{_safe_path_component(name)}"
            if not any(
                candidate == root or candidate.startswith(f"{root}/")
                for candidate in checkpoint_leaf_paths
            ):
                raise ValueError(f"{path}: {checkpoint} misses registry entry {name}")

    graph_names = {
        name for name in registry_entries if name.startswith("replay_graph:input:")
    }
    if graph_input_count != 7:
        raise ValueError(f"{path}: expected seven replay graph inputs")
    expected_graph_names = {
        f"replay_graph:input:{index}" for index in range(graph_input_count)
    }
    if graph_names != expected_graph_names:
        raise ValueError(f"{path}: graph input inventory is incomplete")

    required_registry = {
        "model:_mtp_hidden_buffer",
        "request:last_sampled_tokens",
        "request:draft_tokens",
        "request:num_computed_tokens",
        "request:total_len",
        "request:next_prefill_tokens",
        "request:all_token_ids",
        "sampler:output_bin_counts",
        "sampler:temperature",
        "sampler:top_k",
        "sampler:top_p",
        "sampler:min_p",
        "sampler:seeds",
        "speculator:draft_tokens",
        "speculator:draft_token_confidence_probs",
        "speculator:hidden_states",
        "speculator:context_positions",
        "speculator:idx_mapping",
    }
    speculative = manifest.get("shared_config", {}).get("speculative_config", {})
    if speculative.get("draft_sample_method") == "probabilistic":
        required_registry.add("speculator:draft_logits")
    if speculative.get("enable_adaptive_verification") is True:
        required_registry.update(
            {
                "adaptive:confidence_probs",
                "adaptive:_batch_draft_capacity",
                "adaptive:_num_non_draft_tokens",
                "adaptive:cu_num_logits",
                "adaptive:_stale_idx",
                "adaptive:_pending_resets",
                "adaptive:_batch_budget",
            }
        )
        if not any(name.startswith("adaptive:stale_gpu:") for name in registry_entries):
            raise ValueError(f"{path}: adaptive GPU stale state is absent")
        if not any(name.startswith("adaptive:stale_cpu:") for name in registry_entries):
            raise ValueError(f"{path}: adaptive CPU stale state is absent")
    missing_configured = required_registry - set(registry_entries)
    if missing_configured:
        raise ValueError(
            f"{path}: configured registry entries are absent: "
            f"{sorted(missing_configured)}"
        )

    if not set(meaningful_counts).issubset(TRIMMABLE_REGISTRY_NAMES):
        raise ValueError(f"{path}: unsupported meaningful-row trimming family")
    if any(
        not isinstance(count, int) or count <= 0 for count in meaningful_counts.values()
    ):
        raise ValueError(f"{path}: meaningful-row counts must be positive")
    for checkpoint in ("post_sampling", "s1"):
        for name, expected_prefix in meaningful_counts.items():
            leaf = leaves[f"{checkpoint}:registry/{_safe_path_component(name)}"]
            actual_prefix = leaf.get("metadata", {}).get("compare_prefix")
            if actual_prefix != expected_prefix:
                raise ValueError(f"{path}: meaningful-row prefix differs for {name}")
    _validate_container_children(path, leaves)


def _load_run(path: Path) -> EvidenceRun:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError(f"{path}: incomplete or unsupported manifest")
    payload_path = path.parent / manifest["payload"]
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise ValueError(f"{path}: payload checksum mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError(f"{path}: payload is not tensor-only")
    if set(manifest.get("checkpoints", ())) != REQUIRED_CHECKPOINTS:
        raise ValueError(f"{path}: required checkpoints are absent")
    leaves = {}
    for leaf in manifest["leaves"]:
        key = f"{leaf['checkpoint']}:{leaf['path']}"
        if key in leaves:
            raise ValueError(f"{path}: duplicate leaf {key}")
        payload_key = leaf.get("payload_key")
        if payload_key is not None:
            tensor = payload.get(payload_key)
            if tensor is None:
                raise ValueError(f"{path}: missing payload tensor {payload_key}")
            if str(tensor.dtype) != leaf["dtype"]:
                raise ValueError(f"{path}: payload dtype differs for {key}")
            if list(tensor.shape) != leaf["shape"]:
                raise ValueError(f"{path}: payload shape differs for {key}")
        leaves[key] = leaf
    _validate_evidence_contract(path, manifest, leaves)
    for checkpoint, required_paths in REQUIRED_PATHS.items():
        actual_paths = {
            leaf["path"]
            for leaf in manifest["leaves"]
            if leaf["checkpoint"] == checkpoint
        }
        missing = required_paths - actual_paths
        if missing:
            raise ValueError(
                f"{path}: {checkpoint} misses required paths {sorted(missing)}"
            )
    for checkpoint in ("post_sampling", "s1"):
        paths = {
            leaf["path"]
            for leaf in manifest["leaves"]
            if leaf["checkpoint"] == checkpoint
        }
        if not any(path.startswith("registry/") for path in paths):
            raise ValueError(f"{path}: {checkpoint} registry evidence is absent")
        if not (
            "decisions/sampled_token_ids" in paths
            or "decisions/sampled_token_ids/__type__" in paths
        ):
            raise ValueError(f"{path}: {checkpoint} sampled decisions are absent")
        mandatory_registry = {
            "registry/replay_graph:input:0",
            "registry/model:_mtp_hidden_buffer",
            "registry/request:last_sampled_tokens",
            "registry/speculator:draft_tokens",
            "registry/speculator:draft_token_confidence_probs",
        }
        missing_registry = mandatory_registry - paths
        if missing_registry:
            raise ValueError(
                f"{path}: {checkpoint} misses registry families "
                f"{sorted(missing_registry)}"
            )
        if manifest.get("active_role") == "active" and not any(
            item.startswith("registry/kv-dst:") for item in paths
        ):
            raise ValueError(f"{path}: {checkpoint} KV destination evidence is absent")
    for key, leaf in leaves.items():
        comparison = leaf["comparison"]
        path_name = leaf["path"]
        if "compare_prefix" in leaf.get("metadata", {}) and not (
            leaf["checkpoint"] in ("post_sampling", "s1")
            and path_name.startswith("registry/")
        ):
            raise ValueError(f"{path}: compare_prefix is not allowed for {key}")
        dtype = str(leaf["dtype"])
        is_float = dtype.startswith("torch.float") or dtype.startswith("torch.bfloat")
        is_tensor = leaf.get("payload_key") is not None
        if is_tensor and not is_float and comparison != "exact":
            raise ValueError(f"{path}: non-floating tensor must be exact: {key}")
        if path_name.startswith("registry/kv-") and comparison != "exact":
            raise ValueError(f"{path}: KV bytes must remain exact: {key}")
        if path_name.startswith("decisions/") and comparison != "exact":
            raise ValueError(f"{path}: decisions must remain exact: {key}")
        if not (is_tensor and is_float):
            continue
        expected = None
        if leaf["checkpoint"] in ("l20", "target"):
            expected = "floating"
        elif path_name.startswith("registry/"):
            if any(
                path_name.startswith(prefix) for prefix in NUMERIC_REGISTRY_PREFIXES
            ):
                expected = "floating"
            elif any(
                path_name.startswith(prefix) for prefix in EXACT_FLOAT_REGISTRY_PREFIXES
            ):
                expected = "exact"
        if expected is None:
            raise ValueError(f"{path}: unclassified floating family: {key}")
        if comparison != expected:
            raise ValueError(f"{path}: expected {expected} comparison for {key}")
    target_hidden = leaves["target:target/hidden_states"]
    if not target_hidden["shape"] or target_hidden["shape"][0] < 1:
        raise ValueError(f"{path}: target meaningful rows are absent")
    expected_skipped = manifest.get("active_role") == "idle"
    for checkpoint in ("post_sampling", "s1"):
        skipped = leaves[f"{checkpoint}:decisions/skipped"]
        if skipped.get("value") is not expected_skipped:
            raise ValueError(f"{path}: {checkpoint} skipped role is inconsistent")
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
    *,
    require_complete_files: bool,
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
    global_fields = (
        "run_id",
        "config_digest",
        "shared_config",
        "code_hashes",
        "model",
        "draft_model",
    )
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
            "evidence_contract",
        ):
            values = {
                json.dumps(run.manifest.get(field), sort_keys=True) for run in rank_runs
            }
            if len(values) != 1:
                raise ValueError(f"rank {rank}: evidence field changed: {field}")
        if any(
            int(run.manifest["rank_config"]["data_parallel_rank"]) != rank
            for run in rank_runs
        ):
            raise ValueError(f"rank {rank}: rank-local configuration is invalid")
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
        representative = rank_runs[0].manifest
        for field in (
            "run_id",
            "worker_incarnation",
            "fixture_id",
            "fixture_digest",
            "source_digest",
        ):
            if summary.get(field) != representative.get(field):
                raise ValueError(f"rank {rank}: summary binding differs: {field}")
        if not summary.get("live_restored"):
            raise ValueError(f"rank {rank}: LIVE restoration was not proven")
        if int(summary["completed_runs"]) < max(run.ordinal for run in rank_runs):
            raise ValueError(f"rank {rank}: transaction summary is incomplete")
        summary_files = {
            (str(item["kind"]), int(item["ordinal"])): item
            for item in summary.get("files", ())
        }
        loaded_files = {(run.kind, run.ordinal): run for run in rank_runs}
        if require_complete_files and set(summary_files) != set(loaded_files):
            raise ValueError(f"rank {rank}: summary file set differs")
        for key, run in loaded_files.items():
            item = summary_files.get(key)
            if item is None:
                raise ValueError(f"rank {rank}: summary misses {key}")
            if Path(item["manifest_path"]).name != run.manifest_path.name:
                raise ValueError(
                    f"rank {rank}: summary manifest path differs for {key}"
                )
            if item["manifest_sha256"] != _sha256(run.manifest_path):
                raise ValueError(
                    f"rank {rank}: summary manifest hash differs for {key}"
                )
            payload_path = run.manifest_path.parent / run.manifest["payload"]
            if item["payload_sha256"] != _sha256(payload_path):
                raise ValueError(f"rank {rank}: summary payload hash differs for {key}")


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
    if payload_key is None:
        return leaf.get("value")
    return _meaningful_tensor(run, leaf)


def _meaningful_tensor(
    run: EvidenceRun,
    leaf: dict[str, Any],
) -> torch.Tensor:
    tensor = run.payload[leaf["payload_key"]]
    compare_prefix = leaf.get("metadata", {}).get("compare_prefix")
    if compare_prefix is not None:
        if tensor.ndim < 1 or not 0 < int(compare_prefix) <= tensor.shape[0]:
            raise ValueError(f"{leaf['path']}: invalid meaningful prefix")
        tensor = tensor[: int(compare_prefix)]
    return tensor


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
    if bool(torch.isnan(left64).any()) or bool(torch.isnan(right64).any()):
        raise ValueError("unexpected NaN in floating evidence")
    for predicate, name in (
        (torch.isposinf, "positive-infinity"),
        (torch.isneginf, "negative-infinity"),
    ):
        if not torch.equal(predicate(left64), predicate(right64)):
            raise ValueError(f"{name} masks differ")
    left_finite = torch.isfinite(left64)
    right_finite = torch.isfinite(right64)
    if not torch.equal(left_finite, right_finite):
        raise ValueError("finite masks differ")
    flat_left = left64.reshape(-1)
    flat_right = right64.reshape(-1)
    finite_indices = torch.nonzero(left_finite.reshape(-1), as_tuple=False).flatten()
    finite_left = flat_left.index_select(0, finite_indices)
    finite_right = flat_right.index_select(0, finite_indices)
    difference = (finite_left - finite_right).abs()
    if difference.numel() == 0:
        return {
            "elements": 0,
            "positive_infinities": int(torch.isposinf(left64).sum().item()),
            "negative_infinities": int(torch.isneginf(left64).sum().item()),
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "abs_quantiles": {str(q): 0.0 for q in QUANTILES},
            "worst_flat_index": None,
            "worst_left": None,
            "worst_right": None,
        }
    max_abs, worst_index = difference.max(dim=0)
    denominator = max(
        float(torch.linalg.vector_norm(finite_left).item()),
        float(torch.linalg.vector_norm(finite_right).item()),
        denominator_floor,
    )
    relative_l2 = float(torch.linalg.vector_norm(difference).item()) / denominator
    quantiles = torch.quantile(
        difference,
        torch.tensor(QUANTILES, dtype=torch.float64),
    )
    index = int(finite_indices[int(worst_index.item())].item())
    return {
        "elements": difference.numel(),
        "positive_infinities": int(torch.isposinf(left64).sum().item()),
        "negative_infinities": int(torch.isneginf(left64).sum().item()),
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
        left_tensor = _meaningful_tensor(left, left_leaf)
        right_tensor = _meaningful_tensor(right, right_leaf)
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


def _calibration_provenance(
    runs: dict[tuple[str, int, int], EvidenceRun],
    summaries: dict[int, dict[str, Any]],
    ranks: list[int],
) -> dict[str, Any]:
    representative = runs[("a", ranks[0], 1)].manifest
    return {
        "run_id": representative["run_id"],
        "config_digest": representative["config_digest"],
        "code_hashes": representative["code_hashes"],
        "model": representative["model"],
        "draft_model": representative["draft_model"],
        "dp_size": representative["dp_size"],
        "ranks": ranks,
        "manifests": [
            {
                "rank": run.rank,
                "ordinal": run.ordinal,
                "sha256": _sha256(run.manifest_path),
            }
            for run in sorted(
                runs.values(),
                key=lambda item: (item.rank, item.ordinal),
            )
        ],
        "summaries": [
            {
                "rank": rank,
                "run_id": summaries[rank]["run_id"],
                "fixture_digest": summaries[rank]["fixture_digest"],
                "worker_incarnation": summaries[rank]["worker_incarnation"],
            }
            for rank in ranks
        ],
    }


def calibrate(args: argparse.Namespace) -> None:
    if args.calibration_runs != 10:
        raise ValueError("calibration must use exactly A1-A10")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite frozen envelope: {args.output}")
    runs = _load_directory(
        args.input_dir,
        allowed_kinds={"a"},
        max_a_ordinal=args.calibration_runs,
    )
    ranks = _require_matrix(runs, a_runs=args.calibration_runs)
    summaries = _load_summaries(args.input_dir)
    _validate_cohort(
        runs,
        summaries,
        ranks,
        require_complete_files=False,
    )
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
        "provenance": _calibration_provenance(runs, summaries, ranks),
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
    if args.a_runs != 20:
        raise ValueError("validation must use exactly 20 A controls")
    envelope_document = json.loads(args.envelope.read_text())
    if envelope_document.get("schema") != SCHEMA:
        raise ValueError("unsupported envelope schema")
    calibration_runs = int(envelope_document["calibration_runs"])
    if calibration_runs != 10:
        raise ValueError("frozen envelope was not calibrated from A1-A10")
    denominator_floor = float(envelope_document["denominator_floor"])
    envelope = envelope_document["envelope"]
    provenance = envelope_document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("frozen envelope provenance is absent")
    runs = _load_directory(args.input_dir)
    ranks = _require_matrix(runs, a_runs=args.a_runs)
    summaries = _load_summaries(args.input_dir)
    _validate_cohort(
        runs,
        summaries,
        ranks,
        require_complete_files=True,
    )
    calibration_subset = {
        key: run
        for key, run in runs.items()
        if run.kind == "a" and 1 <= run.ordinal <= calibration_runs
    }
    current_provenance = _calibration_provenance(
        calibration_subset,
        summaries,
        ranks,
    )
    if current_provenance != provenance:
        raise ValueError("frozen envelope provenance does not match evidence")
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
