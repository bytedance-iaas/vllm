# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical lookup channel. It deliberately exposes no shared-store reset RPC."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import zmq

from .data import BlobBlockHashes, MooncakeLookupResult
from .protocol import LOOKUP_MSG, decode_lookup_response, encode_lookup_response
from .worker import get_zmq_rpc_path_lookup

# Canonical-only versioned request: u32 num_tokens, u32 start_tokens,
# u16 hash width, concatenated hashes. Legacy LOOKUP_MSG means start=0.
# Old servers reject the new tag as a miss, never reinterpret its offset.
CANONICAL_LOOKUP_V2 = b"canonical_lookup_v2"


def _decode_request(frames):
    if len(frames) == 4 and frames[0] == LOOKUP_MSG:
        _, count, width, blob = frames
        start = b"\x00\x00\x00\x00"
    elif len(frames) == 5 and frames[0] == CANONICAL_LOOKUP_V2:
        _, count, start, width, blob = frames
    else:
        raise ValueError("Invalid canonical lookup message")
    if len(count) != 4 or len(start) != 4 or len(width) != 2:
        raise ValueError("Invalid canonical lookup frame width")
    hash_len = int.from_bytes(width, "big")
    if (not hash_len and blob) or (hash_len and (not blob or len(blob) % hash_len)):
        raise ValueError("Invalid canonical hash payload")
    num_tokens, start_tokens = (
        int.from_bytes(count, "big"),
        int.from_bytes(start, "big"),
    )
    if not num_tokens or start_tokens >= num_tokens:
        raise ValueError("Invalid canonical lookup range")
    return num_tokens, BlobBlockHashes(memoryview(blob), hash_len), start_tokens


def _path(config):
    address = get_zmq_rpc_path_lookup(config)
    identity = json.dumps([address, config.kv_transfer_config.engine_id]).encode()
    suffix = hashlib.sha256(identity).hexdigest()[:32]
    directory = address.rsplit("/", 1)[0]
    return f"{directory}/canonical_lookup_{suffix}"


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
        self.contexts = {}
        # Includes detached running queries: cancellation cannot interrupt ZMQ.
        self.inflight = set()
        self._context = None
        self._socket = None
        self._closed = False

    def _get_socket(self):
        if self._context is None:
            self._context = zmq.Context()
        if self._socket is None:
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.connect(self.path)
        return self._socket

    def _reset_socket(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _close_transport(self):
        self._reset_socket()
        if self._context is not None:
            self._context.term()
            self._context = None

    def _query(self, num_tokens, hashes, start_tokens):
        if time.monotonic() < self.unhealthy_until:
            return MooncakeLookupResult(0)
        try:
            sock = self._get_socket()
            sock.send_multipart(
                [
                    CANONICAL_LOOKUP_V2,
                    num_tokens.to_bytes(4, "big"),
                    start_tokens.to_bytes(4, "big"),
                    (len(hashes[0]) if hashes else 0).to_bytes(2, "big"),
                    b"".join(hashes),
                ]
            )
            return decode_lookup_response(sock.recv())
        except (zmq.ZMQError, ValueError):
            # Failed REQ exchanges must not consume a subsequent request's reply.
            self._reset_socket()
            self.unhealthy_until = time.monotonic() + self.cooldown
            return MooncakeLookupResult(0)

    def lookup(self, request_id, num_tokens, hashes, start_tokens=0):
        if self._closed:
            return MooncakeLookupResult(0)
        # Own immutable hashes while the executor/IPC thread uses them. A request
        # id alone is insufficient after cache eviction, preemption or reentry.
        if (
            type(num_tokens) is not int
            or not 0 < num_tokens < 2**32
            or type(start_tokens) is not int
            or not 0 <= start_tokens < num_tokens
        ):
            self.discard(request_id)
            return MooncakeLookupResult(0)
        hashes = tuple(bytes(h) for h in hashes)
        if hashes and (
            not 0 < len(hashes[0]) < 2**16
            or any(len(h) != len(hashes[0]) for h in hashes)
        ):
            self.discard(request_id)
            return MooncakeLookupResult(0)
        context = (num_tokens, hashes, start_tokens)
        if request_id in self.contexts and self.contexts[request_id] != context:
            self.discard(request_id)
        future = self.futures.get(request_id)
        if future is None:
            self.inflight = {f for f in self.inflight if not f.done()}
            if (
                len(self.futures) >= self.max_pending
                or len(self.inflight) >= self.max_pending
                or time.monotonic() < self.unhealthy_until
            ):
                return MooncakeLookupResult(0)
            self.started[request_id] = time.monotonic()
            future = self.executor.submit(self._query, num_tokens, hashes, start_tokens)
            self.futures[request_id] = future
            self.contexts[request_id] = context
            self.inflight.add(future)
        if not future.done():
            if time.monotonic() - self.started[request_id] > 2 * self.timeout_ms / 1000:
                self.discard(request_id)
                return MooncakeLookupResult(0)
            return None
        del self.futures[request_id]
        self.started.pop(request_id, None)
        self.contexts.pop(request_id, None)
        self.inflight.discard(future)
        return future.result()

    def discard(self, request_id):
        self.started.pop(request_id, None)
        self.contexts.pop(request_id, None)
        future = self.futures.pop(request_id, None)
        if future:
            future.cancel()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for request_id in list(self.futures):
            self.discard(request_id)
        try:
            # Cleanup runs on the socket owner after the active query completes.
            self.executor.submit(self._close_transport).result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.inflight.clear()


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
                    num_tokens, hashes, start_tokens = _decode_request(frames)
                    result = worker.lookup(
                        num_tokens, hashes, start_tokens=start_tokens
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
