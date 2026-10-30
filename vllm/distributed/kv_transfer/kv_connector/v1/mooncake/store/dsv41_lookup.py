# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical lookup channel. It deliberately exposes no shared-store reset RPC."""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import zmq

from .data import BlobBlockHashes, MooncakeLookupResult
from .protocol import LOOKUP_MSG, decode_lookup_response, encode_lookup_response
from .worker import get_zmq_rpc_path_lookup


def _path(config):
    engine = config.kv_transfer_config.engine_id.encode()
    suffix = hashlib.sha256(engine).hexdigest()[:12]
    return f"{get_zmq_rpc_path_lookup(config)}_canonical_{suffix}"


class LookupClient:
    def __init__(self, config):
        self.path = _path(config)
        self.timeout_ms = int(
            config.kv_transfer_config.kv_connector_extra_config.get(
                "lookup_timeout_ms", 1000
            )
        )
        if self.timeout_ms <= 0:
            raise ValueError("lookup_timeout_ms must be positive")
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="CanonicalLookup"
        )
        extra = config.kv_transfer_config.kv_connector_extra_config
        self.max_pending = int(extra.get("max_pending_lookups", 32))
        self.cooldown = float(extra.get("failure_cooldown_s", 5))
        self.unhealthy_until = 0.0
        self.futures = {}
        self.started = {}

    def _query(self, num_tokens, hashes):
        # Each RPC owns its socket; a timeout cannot poison the next REQ/REP exchange.
        if time.monotonic() < self.unhealthy_until:
            return MooncakeLookupResult(0)
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        try:
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.connect(self.path)
            sock.send_multipart(
                [
                    LOOKUP_MSG,
                    num_tokens.to_bytes(4, "big"),
                    (len(hashes[0]) if hashes else 0).to_bytes(2, "big"),
                    b"".join(hashes),
                ]
            )
            return decode_lookup_response(sock.recv())
        except (zmq.ZMQError, ValueError):
            self.unhealthy_until = time.monotonic() + self.cooldown
            return MooncakeLookupResult(0)
        finally:
            sock.close()
            context.term()

    def lookup(self, request_id, num_tokens, hashes):
        future = self.futures.get(request_id)
        if future is None:
            if (
                len(self.futures) >= self.max_pending
                or time.monotonic() < self.unhealthy_until
            ):
                return MooncakeLookupResult(0)
            self.started[request_id] = time.monotonic()
            future = self.executor.submit(self._query, num_tokens, tuple(hashes))
            self.futures[request_id] = future
        if not future.done():
            if time.monotonic() - self.started[request_id] > 2 * self.timeout_ms / 1000:
                self.discard(request_id)
                return MooncakeLookupResult(0)
            return None
        del self.futures[request_id]
        self.started.pop(request_id, None)
        return future.result()

    def discard(self, request_id):
        self.started.pop(request_id, None)
        future = self.futures.pop(request_id, None)
        if future:
            future.cancel()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class LookupServer:
    def __init__(self, worker, config):
        self.path = _path(config)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(
            target=self._run, args=(worker,), daemon=True, name="CanonicalLookupServer"
        )
        self.thread.start()
        if not self.ready.wait(timeout=5) or self.error:
            raise RuntimeError(f"Canonical lookup bind failed: {self.error}")

    def _run(self, worker):
        context = zmq.Context()
        sock = context.socket(zmq.REP)
        try:
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 100)
            sock.setsockopt(zmq.SNDTIMEO, 1000)
            sock.bind(self.path)
            self.ready.set()
            while not self.stop.is_set():
                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    continue
                result = MooncakeLookupResult(0)
                try:
                    if len(frames) == 4 and frames[0] == LOOKUP_MSG:
                        result = worker.lookup(
                            int.from_bytes(frames[1], "big"),
                            BlobBlockHashes(
                                memoryview(frames[3]), int.from_bytes(frames[2], "big")
                            ),
                        )
                except Exception:
                    worker.record_lookup_error()
                sock.send(encode_lookup_response(result))
        except Exception as exc:
            self.error = exc
            self.ready.set()
        finally:
            sock.close()
            context.term()

    def close(self):
        self.stop.set()
        self.thread.join()
