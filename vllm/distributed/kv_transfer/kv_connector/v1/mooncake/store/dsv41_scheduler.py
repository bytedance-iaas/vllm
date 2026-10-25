# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler for the opt-in canonical Store; legacy Store scheduling is unchanged."""

import math
import time
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

from .data import MooncakeStoreConnectorMetadata, MooncakeStoreWorkerMetadata
from .dsv41_lookup import LookupClient


@dataclass(frozen=True)
class Job:
    request_id: str
    block_ids: tuple[list[int], ...]
    hashes: tuple[bytes, ...]
    start: int
    end: int
    save_id: int | None = None
    after_forward: bool = False


class Metadata(MooncakeStoreConnectorMetadata):
    def __init__(self):
        super().__init__(set(), set())
        self.saves: list[Job] = []
        self.loads: list[Job] = []
        self.resources: dict[str, int] = {}


class Dsv41StoreScheduler:
    def __init__(self, config, cache_config):
        extra = config.kv_transfer_config.kv_connector_extra_config
        self.prefill = extra["pd_role"] == "prefill"
        self.save_decode = not self.prefill and extra.get("save_decode_cache", False)
        self.capacity_only = not self.prefill and not self.save_decode
        self.pending_finish = []
        self.admitted = set()
        _, self.hash_tokens = resolve_kv_cache_block_sizes(cache_config, config)
        self.chunk_tokens = math.lcm(
            self.hash_tokens,
            *(g.kv_cache_spec.block_size for g in cache_config.prefix_cacheable_groups),
        )
        self.cache_config = cache_config
        self.workers = config.parallel_config.world_size
        self.pool = None
        self.max_jobs = int(extra.get("max_pending_jobs", 32))
        self.max_blocks = int(extra.get("max_pinned_blocks", 4096))
        self.max_bytes = int(extra.get("max_pinned_bytes", 512 * 1024**2))
        self.stall_seconds = float(extra.get("store_stall_timeout_s", 30))
        self.bytes_per_block = sum(
            g.kv_cache_spec.page_size_bytes * max(1, len(g.layer_names))
            for g in cache_config.kv_cache_groups
        )
        self.pin_times = {}
        self.resource_flush = False
        self.requests = {}
        self._pending_admissions = {}
        self.saved = {}
        self.pins = {}
        self.next_id = 0
        self.load_specs = {}
        self.bypass = set()
        self.pending_loads = []
        self.lookup_min_prompt_tokens = extra.get("lookup_min_prompt_tokens", 0)
        self.lookup_skip_short_prompt = 0
        self.lookup_skip_no_possible_increment = 0
        self.client = (
            LookupClient(config)
            if self.prefill and extra.get("enable_lookup", False)
            else None
        )

    def bind_gpu_block_pool(self, pool):
        self.pool = pool

    def has_pending_push_work(self):
        return bool(self.pins) or getattr(self, "resource_flush", False)

    @staticmethod
    def eligible(request):
        return not (
            request.mm_features
            or request.lora_request
            or request.prompt_embeds is not None
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if (
            self.client is None
            or not self.eligible(request)
            or request.request_id in self.bypass
        ):
            return 0, False
        if request.num_prompt_tokens < self.lookup_min_prompt_tokens:
            self.lookup_skip_short_prompt += 1
            return 0, False
        # A complete-page load must not overwrite a partly shared local page.
        if num_computed_tokens % self.chunk_tokens:
            return 0, False
        max_usable_hit = (
            min(request.num_tokens - 1, len(request.block_hashes) * self.hash_tokens)
            // self.chunk_tokens
            * self.chunk_tokens
        )
        if max_usable_hit <= max(num_computed_tokens, 128):
            self.lookup_skip_no_possible_increment += 1
            return 0, False
        result = self.client.lookup(
            request.request_id, request.num_tokens, request.block_hashes
        )
        if result is None:
            return None, False
        hit = result.hit_length
        if hit <= max(num_computed_tokens, 128):
            return 0, False
        if hit % self.chunk_tokens or hit >= request.num_tokens:
            return 0, False
        self.load_specs[request.request_id] = (num_computed_tokens, hit)
        return hit - num_computed_tokens, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self.capacity_only:
            assert num_external_tokens == 0
            return
        if self.eligible(request):
            self._pending_admissions[request.request_id] = request
        spec = self.load_specs.pop(request.request_id, None)
        if num_external_tokens:
            assert spec is not None and num_external_tokens == spec[1] - spec[0]
            self.pending_loads.append(
                Job(
                    request.request_id,
                    tuple(list(g) for g in blocks.get_block_ids()),
                    tuple(request.block_hashes),
                    *spec,
                )
            )

    def _save(self, request, blocks, end):
        if not self.eligible(request):
            return None
        end = min(end, len(request.block_hashes) * self.hash_tokens)
        end = end // self.chunk_tokens * self.chunk_tokens
        start = self.saved.get(request.request_id, 0)
        if end <= start:
            return None
        pinned_blocks = sum(len(value[0]) for value in self.pins.values())
        if len(self.pins) >= self.max_jobs or any(
            time.monotonic() - t > self.stall_seconds for t in self.pin_times.values()
        ):
            return None
        available = (
            min(self.max_blocks, self.max_bytes // self.bytes_per_block) - pinned_blocks
        )
        pages_per_chunk = sum(
            self.chunk_tokens
            // self.cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
            for gid in self.cache_config.prefix_cacheable_group_ids
        )
        end = min(end, start + available // pages_per_chunk * self.chunk_tokens)
        if end <= start:
            return None
        ids = []
        for gid in self.cache_config.prefix_cacheable_group_ids:
            span = self.cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
            pages = blocks[gid][start // span : end // span]
            if len(pages) != (end - start) // span or any(b == 0 for b in pages):
                return None
            ids.extend(pages)
        ids = list(dict.fromkeys(ids))
        if not ids:
            return None
        assert self.pool is not None
        job_id = self.next_id
        self.next_id += 1
        self.pool.touch([self.pool.blocks[b] for b in ids])
        self.pins[job_id] = (ids, self.workers)
        self.pin_times[job_id] = time.monotonic()
        self.saved[request.request_id] = end
        return Job(
            request.request_id,
            tuple(list(g) for g in blocks),
            tuple(request.block_hashes),
            start,
            end,
            job_id,
            self.prefill,
        )

    def build_connector_meta(self, output):
        meta = Metadata()
        if self.capacity_only:
            return self._with_resources(meta)
        meta.loads, self.pending_loads = self.pending_loads, []
        meta.saves, self.pending_finish = self.pending_finish, []
        for rid in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.requests.pop(rid, None)
            self.saved.pop(rid, None)
            self.load_specs.pop(rid, None)
            self.admitted.discard(rid)
            if rid in output.finished_req_ids:
                self.bypass.discard(rid)
            if self.client:
                self.client.discard(rid)
        # Allocation can re-admit a preempted request in this same step.
        self.requests.update(
            (rid, request)
            for rid, request in self._pending_admissions.items()
            if rid not in output.finished_req_ids
        )
        self._pending_admissions.clear()
        if not self.prefill and not self.save_decode:
            return self._with_resources(meta)
        state = output.kv_connector_block_state
        for rid, scheduled in output.num_scheduled_tokens.items():
            request = self.requests.get(rid)
            if request is None:
                continue
            blocks = state.get_block_ids(rid) if state else None
            if blocks is None:
                continue
            self.admitted.add(rid)
            end = (
                min(request.num_computed_tokens + scheduled, request.num_prompt_tokens)
                if self.prefill
                else self.stable_end(request)
            )
            if job := self._save(request, blocks, end):
                meta.saves.append(job)
        return self._with_resources(meta)

    def _with_resources(self, meta):
        self.resource_flush = False
        blocks = sum(len(value[0]) for value in self.pins.values())
        meta.resources = {
            "pinned_jobs": len(self.pins),
            "pinned_blocks": blocks,
            "pinned_bytes": blocks * self.bytes_per_block,
        }
        return meta

    @staticmethod
    def stable_end(request):
        # Computed counters include scheduled drafts until their output is accepted.
        if request.num_stale_output_tokens:
            return 0
        return max(
            0,
            min(
                request.num_computed_tokens - request.num_in_flight_tokens,
                request.num_tokens,
            ),
        )

    def request_finished(self, request, blocks):
        if not (
            self.save_decode
            and request.request_id in self.admitted
            and self.eligible(request)
        ):
            return
        if job := self._save(request, blocks, self.stable_end(request)):
            self.pending_finish.append(job)

    def update_connector_output(self, output):
        if self.capacity_only:
            return
        self.bypass.update(
            rid for rid in getattr(output, "failed_recving", ()) if rid in self.requests
        )
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, MooncakeStoreWorkerMetadata):
            return
        for job_id, count in metadata.completed_saves.items():
            if job_id not in self.pins:
                continue
            ids, remaining = self.pins[job_id]
            remaining -= count
            if remaining <= 0:
                assert remaining == 0, (
                    "Canonical save received duplicate acknowledgments"
                )
                assert self.pool is not None
                self.pool.free_blocks([self.pool.blocks[b] for b in reversed(ids)])
                del self.pins[job_id]
                del self.pin_times[job_id]
                self.resource_flush = True
            else:
                self.pins[job_id] = ids, remaining

    def register_finished_partial_tail(self, *args):
        return False

    def reset_store(self):
        # A shared namespace must never use the legacy global remove_all RPC.
        return False

    def close(self):
        if self.client is not None:
            self.client.close()
