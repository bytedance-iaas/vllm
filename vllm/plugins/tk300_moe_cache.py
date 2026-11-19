# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load a verified native CUTLASS MoE cache after full-model startup warmup."""

import functools
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)
SCHEMA = "tk300-cutlass-moe-load-v1"
SHAPES = "((8192, 3072), (64, 4096, 384), (0,), (64, 6144, 128), (0,))"
EXTRAS = (
    "(torch.uint8, torch.int64, torch.bfloat16, 8, 1, 4, 1, "
    "False, False, False, False, False, False, True, 3, False, True)"
)


def read_manifest(path: Path, recipe: dict) -> tuple[bytes, dict]:
    """Validate the complete file before calling FlashInfer's mutable loader."""
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported MoE manifest schema")
    if manifest.get("recipe") != recipe:
        raise ValueError("MoE manifest recipe does not match this worker")
    if manifest.get("trained_rows") != [8192] or manifest.get("validated_rows") != [
        6528,
        8192,
    ]:
        raise ValueError("MoE cache needs trained and held-out real-input validation")
    if manifest.get("native_bucket_interval") != [4097, 8192]:
        raise ValueError("Unexpected native bucket mapping")
    counts = manifest.get("native_profile_counts")
    if counts != [16, 28]:
        raise ValueError("Unexpected pinned SM120 NVFP4 tactic counts")
    profiles = manifest.get("profile_ids")
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise ValueError("MoE manifest needs two profile IDs")
    if any(type(value) is not int for value in profiles):
        raise ValueError("MoE profile IDs must be integers")
    if not (0 <= profiles[0] < counts[0] and counts[0] <= profiles[1] < sum(counts)):
        raise ValueError("MoE profile ID is outside its GEMM stage")
    cache = (path.parent / manifest["cache_file"]).resolve()
    if not cache.is_relative_to(path.parent.resolve()) or cache.suffix != ".json":
        raise ValueError("MoE cache must be a JSON file within the manifest directory")
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get("cache_sha256"):
        raise ValueError("MoE cache differs from the validated artifact")
    entries = json.loads(raw)
    if not isinstance(entries.get("_metadata"), dict):
        raise TypeError("MoE cache has no native environment identity")
    keys = [key for key in entries if not key.startswith("_")]
    if sorted(keys) != sorted(manifest.get("native_file_keys", [])) or len(keys) != 2:
        raise ValueError("MoE cache must contain exactly the two verified GEMM entries")
    for stage, tactic in zip(("gemm1", "gemm2"), profiles, strict=True):
        expected_key = (
            f"('trtllm::fused_moe::{stage}', 'MoERunner', {SHAPES}, {EXTRAS})"
        )
        if expected_key not in keys:
            raise ValueError(
                "Native MoE shape or constructor signature is incompatible"
            )
        entry = entries[expected_key]
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or type(entry[1]) is not int
            or entry[:2] != ["MoERunner", tactic]
        ):
            raise ValueError("Native MoE entry does not match its validated profile")
        if entry[2] != ["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"]:
            raise ValueError(
                "MoE cache was not measured using the required eager policy"
            )
    return raw, manifest


def worker_recipe(worker) -> dict:
    config = worker.vllm_config
    parallel = config.parallel_config
    return {
        "model": Path(config.model_config.model).name,
        "main_layers": config.model_config.hf_config.num_hidden_layers,
        "quantization": config.model_config.quantization,
        "q": config.scheduler_config.max_num_batched_tokens,
        "slots": config.scheduler_config.max_num_seqs,
        "tp": parallel.tensor_parallel_size,
        "pp": parallel.pipeline_parallel_size,
        "pcp": parallel.prefill_context_parallel_size,
        "dp": parallel.data_parallel_size,
        "ep": parallel.enable_expert_parallel,
        "all2all": parallel.all2all_backend,
        "graph": config.compilation_config.cudagraph_mode.name,
        "autotune": config.kernel_config.enable_flashinfer_autotune,
        "block_size": config.cache_config.block_size,
        "kv_dtype": config.cache_config.cache_dtype,
        "mtp": config.speculative_config.num_speculative_tokens,
        "sampling_policy": getattr(
            config.kernel_config, "sparse_indexer_sampling_policy", "legacy"
        ),
        "mla_bmm": config.kernel_config.mla_bmm_backend,
        "partition": os.environ.get("VLLM_PP_LAYER_PARTITION"),
    }


class MoEAutotuneCacheWorker:
    """Startup-only extension; native cache hits never add collectives or searches."""

    vllm_config: "VllmConfig"
    rank: int
    compile_or_warm_up_model: Callable[..., Any]

    def init_worker_extension(self):
        self._tk300_moe_cache_status = {"loaded": False, "reason": "disabled"}
        configured = os.environ.get("TK300_MOE_AUTOTUNE_MANIFEST")
        if not configured:
            return
        original = self.compile_or_warm_up_model

        @functools.wraps(original)
        def warmup_then_load(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                recipe = worker_recipe(self)
                if recipe["autotune"] or recipe["q"] != 8192:
                    raise ValueError(
                        "MoE replay requires global autotune off and q8192"
                    )
                raw, manifest = read_manifest(Path(configured), recipe)
                from flashinfer.autotuner import AutoTuner
                from flashinfer.autotuner.autotuner import _collect_metadata

                tuner = AutoTuner.get()
                if tuner.is_tuning_mode:
                    raise ValueError("Cannot load MoE replay while profiling is active")
                if json.loads(raw)["_metadata"] != _collect_metadata():
                    raise ValueError(
                        "Native library/device environment is incompatible"
                    )
                with tempfile.NamedTemporaryFile(suffix=".json") as snapshot:
                    snapshot.write(raw)
                    snapshot.flush()
                    loaded = tuner.load_configs(snapshot.name)
                self._tk300_moe_cache_status = {
                    "loaded": loaded,
                    "reason": "loaded" if loaded else "native environment mismatch",
                    "cache_sha256": manifest["cache_sha256"],
                    "profile_ids": manifest["profile_ids"],
                    "validated_rows": manifest["validated_rows"],
                }
                logger.info(
                    "TK300 MoE load-only rank=%s status=%s",
                    self.rank,
                    self._tk300_moe_cache_status,
                )
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self._tk300_moe_cache_status = {"loaded": False, "reason": str(error)}
                logger.warning(
                    "TK300 MoE cache rejected on rank %s; using defaults: %s",
                    self.rank,
                    error,
                )
            return result

        self.compile_or_warm_up_model = warmup_then_load

    def moe_autotune_cache_status(self):
        return dict(self._tk300_moe_cache_status, rank=self.rank)
