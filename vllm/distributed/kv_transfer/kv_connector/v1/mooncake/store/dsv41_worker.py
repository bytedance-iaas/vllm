# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GPU-page transfers to Mooncake's DRAM tier."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.distributed import get_pcp_group, get_pp_group, get_world_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

from .data import MooncakeLookupResult, MooncakeStoreWorkerMetadata
from .dsv41_layout import Manifest, build_bindings, merge_components
from .dsv41_lookup import LookupServer
from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)


class Dsv41StoreWorker:
    enable_kv_events = False
    kv_send_thread = None

    def __init__(self, config, cache_config):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        self.config = config
        self.cache_config = cache_config
        self.extra = config.kv_transfer_config.kv_connector_extra_config
        self.prefill = self.extra["pd_role"] == "prefill"
        self.capacity_only = not self.prefill and not self.extra.get(
            "save_decode_cache", False
        )
        store_config = MooncakeStoreConfig.load_from_config()
        if store_config.enable_offload:
            raise ValueError("Canonical L2 profile requires a DRAM-only Mooncake Store")
        self.device = torch.cuda.current_device()
        try:
            self.store = MooncakeDistributedStore()
            setup_mooncake_store(
                self.store,
                store_config,
                rdma_utils.get_requester_local_hostname(get_ip()),
            )
        finally:
            # Native transport initialization may change the thread's device.
            torch.cuda.set_device(self.device)
        self.replication = ReplicateConfig()
        preferred = rdma_utils.get_configured_preferred_segment(self.extra)
        if preferred is not None:
            self.replication.preferred_segment = preferred
        self.writer = not self.prefill or get_pcp_group().rank_in_group == 0
        self.put_executor = (
            None
            if self.capacity_only
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="CanonicalPut")
        )
        self.get_executor = (
            None
            if self.capacity_only
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="CanonicalGet")
        )
        self.lock = threading.Lock()
        self.completed = {}
        self.bindings = []
        self.manifest = None
        self.server = None
        self.report_resources = (
            get_pp_group().rank_in_group == 0 and get_pcp_group().rank_in_group == 0
        )
        self.scheduler_resources = (
            {"pinned_jobs": 0, "pinned_blocks": 0, "pinned_bytes": 0}
            if self.capacity_only
            else {}
        )
        self.received = set()
        self.failed = set()
        self.max_jobs = int(self.extra.get("max_pending_jobs", 32))
        self.max_bytes = int(self.extra.get("max_inflight_bytes", 256 * 1024**2))
        self.stall_seconds = float(self.extra.get("store_stall_timeout_s", 30))
        self.cooldown = float(self.extra.get("failure_cooldown_s", 5))
        self.pending = {}
        self.unhealthy_until = 0.0
        self.stats = MooncakeStoreConnectorStats()
        self.next_capacity_stats = 0.0

    def register_kv_caches(self, caches):
        if self.capacity_only:
            logger.info(
                "Canonical Store capacity-only: role=decode device=%s "
                "GPU KV registration disabled module=%s",
                self.device,
                __file__,
            )
            return
        # All workers participate even if a local layout is invalid.
        error = None
        try:
            self.bindings = build_bindings(
                self.cache_config, caches, prefill=self.prefill
            )
            devices = {binding.tensor.device for binding in self.bindings}
            if len(devices) != 1:
                raise ValueError("Canonical KV tensors must share one CUDA device")
            device = devices.pop()
            if device.type != "cuda" or device.index is None:
                raise ValueError("Canonical KV tensors must reside on CUDA")
            self.device = device.index
        except Exception as exc:
            error = str(exc)
        _, hash_tokens = resolve_kv_cache_block_sizes(self.cache_config, self.config)
        group = get_world_group()
        descriptors = [None] * group.world_size
        torch.distributed.all_gather_object(
            descriptors,
            (error, [b.component for b in self.bindings], hash_tokens),
            group=group.cpu_group,
        )
        errors = [item[0] for item in descriptors if item[0]]
        if errors:
            raise ValueError(f"Canonical worker layout validation failed: {errors}")
        components = merge_components([item[1] for item in descriptors])
        if len({item[2] for item in descriptors}) != 1:
            raise ValueError("Canonical workers must share the hash granularity")
        self.manifest = Manifest(
            self.extra["cache_prefix"],
            self.extra["model_revision"],
            components,
            hash_tokens,
        )
        storages = {
            b.tensor.untyped_storage().data_ptr(): b.tensor.untyped_storage()
            for b in self.bindings
        }
        for address, storage in storages.items():
            with torch.cuda.device(self.device):
                result = self.store.register_buffer(address, storage.nbytes())
            if result != 0:
                raise RuntimeError(
                    f"Canonical GPU buffer registration failed: {result}"
                )
        logger.info(
            "Canonical Store registered: role=%s device=%s writer=%s "
            "chunk_tokens=%s manifest=%s local_components=%s module=%s",
            self.extra["pd_role"],
            self.device,
            self.writer,
            self.manifest.chunk_tokens,
            self.manifest.prefix,
            [binding.component.name for binding in self.bindings],
            __file__,
        )

        if (
            self.prefill
            and self.extra.get("enable_lookup", False)
            and get_pp_group().rank_in_group == 0
            and get_pcp_group().rank_in_group == 0
        ):
            self.server = LookupServer(self, self.config)

    def lookup(self, num_tokens, hashes):
        manifest = self.manifest
        assert manifest is not None
        if not self._healthy():
            return MooncakeLookupResult(0)
        bytes_per_chunk = sum(
            c.page_bytes * (manifest.chunk_tokens // c.block_tokens)
            for c in manifest.components
        )
        limit = min(
            num_tokens - 1,
            len(hashes) * manifest.hash_tokens,
            self.max_bytes // bytes_per_chunk * manifest.chunk_tokens,
        )
        hit = 0
        deadline = (
            time.monotonic()
            + int(self.extra.get("lookup_timeout_ms", 1000)) / 1000 * 0.8
        )
        span = manifest.chunk_tokens
        for begin in range(span, limit + 1, span * 32):
            ends = list(range(begin, min(limit, begin + span * 31) + 1, span))
            keys = [
                manifest.key(component, hashes[end // manifest.hash_tokens - 1])
                for end in ends
                for component in manifest.components
            ]
            started = time.monotonic()
            exists = self.store.batch_is_exist(keys)
            self._record("lookup_exists", started, len(keys))
            if len(exists) != len(keys):
                raise RuntimeError("Invalid canonical existence response")
            width = len(manifest.components)
            for i, end in enumerate(ends):
                if any(result != 1 for result in exists[i * width : (i + 1) * width]):
                    return MooncakeLookupResult(hit)
                hit = end
            if time.monotonic() >= deadline:
                break
        return MooncakeLookupResult(hit)

    def record_lookup_error(self):
        self._trip()
        self._record("lookup_exists", time.monotonic(), 0, status="error")
        logger.warning("Canonical Store lookup failed; treating it as a cache miss")

    def _batch(self, job, *, saving):
        manifest = self.manifest
        assert manifest is not None
        if job.start % manifest.chunk_tokens or job.end % manifest.chunk_tokens:
            raise ValueError("Scheduler/worker canonical chunk alignment mismatch")
        keys, addresses, sizes = [], [], []
        for start in range(job.start, job.end, manifest.chunk_tokens):
            end = start + manifest.chunk_tokens
            block_hash = job.hashes[end // manifest.hash_tokens - 1]
            for binding in self.bindings:
                if saving and (not self.writer or not binding.writer):
                    continue
                regions = binding.regions(start, end, job.block_ids[binding.group_id])
                keys.append(manifest.key(binding.component, block_hash))
                addresses.append([addr for addr, _ in regions])
                sizes.append([size for _, size in regions])
        return keys, addresses, sizes

    def _healthy(self):
        with self.lock:
            return time.monotonic() >= self.unhealthy_until and all(
                time.monotonic() - started < self.stall_seconds
                for _, started in self.pending.values()
            )

    def _trip(self):
        with self.lock:
            self.unhealthy_until = time.monotonic() + self.cooldown

    def _record(self, operation, started, keys, *, size=0, status="ok"):
        with self.lock:
            records = self.stats.data.get(operation, [])
            if len(records) >= 512:
                del records[0]
            self.stats.record_operation(
                operation,
                time.monotonic() - started,
                keys,
                num_bytes=size,
                status=status,
                num_failed_keys=keys if status == "error" else 0,
            )

    def _finish(self, job, saving, failed):
        with self.lock:
            self.pending.pop((saving, job.save_id if saving else job.request_id), None)
            if saving:
                self.completed[job.save_id] = 1
            else:
                self.received.add(job.request_id)
                if failed:
                    self.failed.add(job.request_id)

    def _transfer(self, job, ready, saving):
        failed = False
        started = time.monotonic()
        total_keys = total_bytes = 0
        try:
            with torch.cuda.device(self.device):
                if ready is not None:
                    ready.synchronize()
                # Limit native RPC descriptor size without changing object identity.
                from dataclasses import replace

                chunk = self.manifest.chunk_tokens
                for offset in range(job.start, job.end, chunk * 32):
                    part = replace(
                        job, start=offset, end=min(job.end, offset + chunk * 32)
                    )
                    keys, addresses, sizes = self._batch(part, saving=saving)
                    if not keys:
                        continue
                    if saving:
                        exists_started = time.monotonic()
                        exists = self.store.batch_is_exist(keys)
                        self._record("save_exists", exists_started, len(keys))
                        if len(exists) != len(keys) or any(r < 0 for r in exists):
                            raise RuntimeError("Canonical Store existence check failed")
                        missing = [i for i, value in enumerate(exists) if value != 1]
                        keys = [keys[i] for i in missing]
                        addresses = [addresses[i] for i in missing]
                        sizes = [sizes[i] for i in missing]
                        if not keys:
                            continue
                    total_keys += len(keys)
                    total_bytes += sum(sum(row) for row in sizes)
                    result = (
                        self.store.batch_put_from_multi_buffers(
                            keys, addresses, sizes, self.replication
                        )
                        if saving
                        else self.store.batch_get_into_multi_buffers(
                            keys, addresses, sizes
                        )
                    )
                    if len(result) != len(keys):
                        raise RuntimeError("Invalid canonical Store response length")
                    if any(r < 0 for r in result):
                        # A concurrent writer may have won the PUT race.
                        exists = self.store.batch_is_exist(keys) if saving else []
                        if len(exists) != len(keys) or any(r != 1 for r in exists):
                            raise RuntimeError("Canonical Store transfer failed")
        except Exception:
            failed = True
            self._trip()
            logger.exception("Canonical Store %s failed", "PUT" if saving else "GET")
        finally:
            self._record(
                "save_put" if saving else "load_get",
                started,
                total_keys,
                size=total_bytes,
                status="error" if failed else "ok",
            )
            # The synchronous native call has returned; its buffers may now retire.
            self._finish(job, saving, failed)

    def _submit(self, job, *, saving):
        assert not self.capacity_only
        size = sum(
            (job.end - job.start) // b.component.block_tokens * b.component.page_bytes
            for b in self.bindings
            if not saving or (self.writer and b.writer)
        )
        key = (saving, job.save_id if saving else job.request_id)
        admitted = False
        if self._healthy():
            with self.lock:
                if (
                    len(self.pending) < self.max_jobs
                    and sum(n for n, _ in self.pending.values()) + size
                    <= self.max_bytes
                ):
                    self.pending[key] = size, time.monotonic()
                    admitted = True
        if not admitted:
            self._record(
                "save_admission" if saving else "load_admission",
                time.monotonic(),
                0,
                size=size,
                status="skipped",
            )
            self._finish(job, saving, not saving)
            return
        try:
            ready = None
            if saving:
                with torch.cuda.device(self.device):
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(self.device))
            executor = self.put_executor if saving else self.get_executor
            executor.submit(self._transfer, job, ready, saving)
        except Exception:
            self._trip()
            self._finish(job, saving, not saving)
            logger.exception("Canonical transfer submission failed")

    def start_load_kv(self, metadata):
        if self.capacity_only:
            assert not metadata.loads and not metadata.saves
            return
        if self.report_resources:
            with self.lock:
                self.scheduler_resources = metadata.resources
        for job in metadata.loads:
            self._submit(job, saving=False)
        # MRV2 skips wait_for_save on zero-token polling steps.
        for job in metadata.saves:
            if not job.after_forward:
                self._submit(job, saving=True)

    def wait_for_save(self, metadata):
        for job in metadata.saves:
            if job.after_forward:
                self._submit(job, saving=True)

    def get_finished(self, finished_req_ids, metadata):
        results = self.get_transfer_results(finished_req_ids, metadata)
        return results.finished_sending or set(), results.finished_recving or set()

    def get_transfer_results(self, finished_req_ids, metadata):
        if self.capacity_only:
            return KVConnectorTransferResults()
        with self.lock:
            received, self.received = self.received, set()
            failed, self.failed = self.failed, set()
        return KVConnectorTransferResults(
            finished_recving=received, failed_recving=failed
        )

    def get_block_ids_with_load_errors(self):
        # HMA uses group-local block IDs. Report request failures only, after
        # receive quorum, so the scheduler can safely recompute the entire prefix.
        return set()

    def build_connector_worker_meta(self):
        if self.capacity_only:
            return None
        with self.lock:
            completed, self.completed = self.completed, {}
        return (
            MooncakeStoreWorkerMetadata(completed_saves=completed)
            if completed
            else None
        )

    def get_kv_connector_stats(self):
        if self.capacity_only:
            now = time.monotonic()
            if now < self.next_capacity_stats:
                return None
            self.next_capacity_stats = now + 1.0
            return MooncakeStoreConnectorStats(
                data={
                    "resources": [
                        {
                            "pending_jobs": 0,
                            "pending_bytes": 0,
                            "unhealthy_workers": 0,
                            **self.scheduler_resources,
                        }
                    ]
                }
            )
        with self.lock:
            self.stats.data["resources"] = [
                {
                    "pending_jobs": len(self.pending),
                    "pending_bytes": sum(n for n, _ in self.pending.values()),
                    "unhealthy_workers": int(
                        time.monotonic() < self.unhealthy_until
                        or any(
                            time.monotonic() - t >= self.stall_seconds
                            for _, t in self.pending.values()
                        )
                    ),
                    **self.scheduler_resources,
                }
            ]
            stats, self.stats = self.stats, MooncakeStoreConnectorStats()
        return None if stats.is_empty() else stats

    def close(self):
        server = getattr(self, "server", None)
        if server is not None:
            server.close()
            self.server = None
        for name in ("put_executor", "get_executor"):
            executor = getattr(self, name, None)
            if executor is not None:
                # Never release registered GPU buffers on a wall-clock timeout.
                executor.shutdown(wait=True)
                setattr(self, name, None)
        store = getattr(self, "store", None)
        if store is not None:
            store.close()
            self.store = None
