# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build diagnostic source variants from exact captured P/H20 files.

This creates no containers. Probes preserve the four-stage barrier and original
execution calls; model dispatch timestamps describe CPU entry, not GPU clocks.
"""

import argparse
import ast
import json
from hashlib import sha256
from pathlib import Path

NIXL = "vllm/distributed/kv_transfer/kv_connector/v1/nixl/"
CORE = "vllm/v1/engine/core.py"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"

HELPER = """
import os as _tk300_trace_os
_HANDOFF_DIAGNOSTIC_ENABLED = _tk300_trace_os.environ.get("TK300_HANDOFF_TRACE") == "1"
_HANDOFF_LOGGED_ONCE = set()
_HANDOFF_CHUNK_WAIT_ONCE = set()

def _vrdma_block_count(blocks):
    try:
        if not blocks:
            return 0
        first = blocks[0]
        return len(first) if hasattr(first, "__len__") else len(blocks)
    except Exception:
        return None

def _vrdma_notif_request(msg):
    try:
        return msg.decode("utf-8").rsplit(":", 1)[0]
    except Exception:
        return None

def _vrdma_chunk_wait(stage, rid, **values):
    if not _HANDOFF_DIAGNOSTIC_ENABLED or values.get("total_blocks", 0) < 4096:
        return
    key = (stage, rid, values.get("start_block"), values.get("end_block"),
           values.get("pending_jobs"), values.get("reason"))
    if key in _HANDOFF_CHUNK_WAIT_ONCE or len(_HANDOFF_CHUNK_WAIT_ONCE) >= 65536:
        return
    _HANDOFF_CHUNK_WAIT_ONCE.add(key)
    _vrdma_e2e(stage, rid, **values)

def _vrdma_e2e(stage, rid, **values):
    # Diagnostic logging must never change request execution or notification flow.
    try:
        if isinstance(rid, bytes):
            if rid.startswith(b"HB:"):
                return
            rid = rid.decode("utf-8").rsplit(":", 1)[0]
        if not _HANDOFF_DIAGNOSTIC_ENABLED and "vrdmae2e-" not in rid:
            return
        if stage in {"model_execute_first_dispatch", "model_output_first_processed",
                     "producer_scheduler_metadata"}:
            key = (stage, rid, values.get("engine_epoch"))
            if key in _HANDOFF_LOGGED_ONCE or len(_HANDOFF_LOGGED_ONCE) >= 65536:
                return
            _HANDOFF_LOGGED_ONCE.add(key)
        import json
        import socket
        import time
        if "pcp" in values:
            try:
                from vllm.distributed import get_pp_group
                values["pp"] = get_pp_group().rank_in_group
            except Exception:
                values["pp"] = None
        wall_ns = time.time_ns()
        logger.info("VRDMA_E2E %s", json.dumps(dict(
            schema=2, stage=stage, rid=rid, epoch=wall_ns/1e9,
            epoch_ns=wall_ns, monotonic_ns=time.monotonic_ns(),
            pid=_tk300_trace_os.getpid(), host=socket.gethostname(),
            run_id=_tk300_trace_os.environ.get("TK300_HANDOFF_RUN_ID", ""),
            **values), separators=(",", ":"), default=str))
    except Exception:
        return
"""


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def add_helper(text):
    tree = ast.parse(text)
    existing = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_vrdma_e2e"
    ]
    if existing:
        assert len(existing) == 1
        node = existing[0]
        lines = text.splitlines(keepends=True)
        text = "".join(lines[: node.lineno - 1] + lines[node.end_lineno :])
    assert "_HANDOFF_DIAGNOSTIC_ENABLED =" not in text
    return text.rstrip() + "\n\n" + HELPER


def instrument(name, text):
    assert "_HANDOFF_DIAGNOSTIC_ENABLED" not in text
    if name.endswith("push_worker.py"):
        if "def finish_chunk_step(" in text:
            text = instrument_chunks(text)
        old = "                self._finished_blocks_inbox.put((req_id, block_ids))\n"
        text = replace(
            text,
            old,
            old
            + (
                "                if _HANDOFF_DIAGNOSTIC_ENABLED:\n         "
                '           _vrdma_e2e("producer_worker_metadata_queued", r'
                "eq_id,\n                               engine_id=self.engi"
                "ne_id, tp=self.tp_rank, pcp=self.pcp_rank)\n"
            ),
        )
        text = replace(
            text,
            (
                "                for notifs in self.nixl_wrapper."
                "get_new_notifs().values():"
            ),
            (
                "                for _handoff_agent, notifs in self.nixl_wr"
                "apper.get_new_notifs().items():"
            ),
        )
        old = "                            self._pending_completion_notifs.put(notif)\n"
        text = replace(
            text,
            old,
            old
            + (
                "                            if _HANDOFF_DIAGNOSTIC_ENABLED"
                ':\n                                _vrdma_e2e("decoder_not'
                'ification_arrival", notif,\n                              '
                "             engine_id=self.engine_id, source_agent=_hando"
                "ff_agent)\n"
            ),
        )
        old = (
            "                self.nixl_wrapper.send_notif(age"
            "nt_name, notif_msg=notif_msg)\n"
        )
        text = replace(
            text,
            old,
            old
            + """                if _HANDOFF_DIAGNOSTIC_ENABLED:
                    _vrdma_e2e("decoder_registration_sent", req_id,
                               producer_engine=engine_id, producer_agent=agent_name,
                               producer_rank=rank)
""",
        )
        old = "            self.nixl_wrapper.transfer(handle)\n"
        text = replace(
            text,
            old,
            old
            + (
                "            if _HANDOFF_DIAGNOSTIC_ENABLED:\n             "
                '   _vrdma_e2e("producer_write_submitted", request_id,\n   '
                "                        decoder_request_id=remote_request_"
                "id, decoder_engine=dst_engine_id,\n                       "
                "    handle=handle, descriptors=len(remote_block_descs_ids)"
                ",\n                           engine_id=self.engine_id, tp"
                "=self.tp_rank, pcp=self.pcp_rank)\n"
            ),
        )
        old = "            expected_notifs = meta.pp_size * producers_per_consumer\n"
        text = replace(
            text,
            old,
            old
            + """            if _HANDOFF_DIAGNOSTIC_ENABLED:
                _vrdma_e2e("decoder_notification_consumed", req_id,
                           engine_id=self.engine_id, expected=expected_notifs,
                           count=self.consumer_notification_counts_by_req[req_id]+1)
""",
        )
        old = "            self._recving_transfers.setdefault(req_id, [])\n"
        text = replace(
            text,
            old,
            old
            + """            if _HANDOFF_DIAGNOSTIC_ENABLED:
                _vrdma_e2e("decoder_producer_barrier_reported", req_id,
                           engine_id=self.engine_id, expected=expected_notifs)
""",
        )
        old = "            done_sending.add(req_id)\n"
        text = replace(
            text,
            old,
            old
            + """            if _HANDOFF_DIAGNOSTIC_ENABLED:
                _vrdma_e2e("producer_write_handles_retired", req_id,
                           engine_id=self.engine_id, tp=self.tp_rank, pcp=self.pcp_rank)
""",
        )
    elif name.endswith("base_scheduler.py"):
        old = "        meta.reqs_to_send = self._reqs_need_send\n"
        text = replace(
            text,
            old,
            old
            + """        if _HANDOFF_DIAGNOSTIC_ENABLED:
            for _handoff_rid in self._reqs_need_send:
                _vrdma_e2e("producer_scheduler_metadata", _handoff_rid,
                           engine_id=self.engine_id)
""",
        )
    elif name == SCHEDULER:
        old = "        self.finished_recving_kv_req_ids.remove(request.request_id)\n"
        text = replace(
            text,
            old,
            old
            + (
                "        if _HANDOFF_DIAGNOSTIC_ENABLED:\n            _vrdm"
                'a_e2e("decoder_scheduler_kv_ready", request.request_id,\n '
                "                      engine_epoch=self.prefill_progress_e"
                "poch,\n                       dp_rank=self.parallel_config"
                ".data_parallel_rank, load_failed=load_failed)\n"
            ),
        )
        methods = [
            node
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.FunctionDef) and node.name == "get_decode_capacity"
        ]
        assert len(methods) == 1
        final_return = methods[0].body[-1]
        assert isinstance(final_return, ast.Return)
        assert isinstance(final_return.value, ast.Name)
        assert final_return.value.id == "snapshot"
        lines = text.splitlines(keepends=True)
        clock = (
            "        if _HANDOFF_DIAGNOSTIC_ENABLED:\n"
            "            _clock_before = time.time_ns()\n"
            "            _clock_mono = time.monotonic_ns()\n"
            "            _clock_after = time.time_ns()\n"
            '            snapshot["handoff_clock"] = {\n'
            '                "epoch_ns": (_clock_before + _clock_after) // 2,\n'
            '                "monotonic_ns": _clock_mono,\n'
            '                "anchor_bracket_ns": _clock_after - _clock_before,\n'
            '                "run_id": _tk300_trace_os.environ.get(\n'
            '                    "TK300_HANDOFF_RUN_ID", ""),\n'
            "            }\n"
        )
        text = "".join(lines[: final_return.lineno - 1]) + clock
        text += "".join(lines[final_return.lineno - 1 :])
    elif name == CORE:
        old = (
            "        future = self.model_executor.execute_mod"
            "el(scheduler_output, non_block=True)\n"
        )
        text = replace(text, old, step_events("model_execute_first_dispatch", 8) + old)
        old = "                exec_future = self.model_executor.execute_model(\n"
        text = replace(text, old, step_events("model_execute_first_dispatch", 16) + old)
        old = (
            "        self._attach_iteration_details(engine_co"
            "re_outputs, iteration_details)\n"
        )
        # There is one unrelated attach call earlier; target update_from_output blocks.
        prefix = """        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
"""
        text = replace(
            text,
            prefix,
            prefix + step_events("model_output_first_processed", 8),
            count=2,
        )
    else:
        raise ValueError(name)
    return add_helper(text)


def instrument_chunks(text):
    """Record long-request job state without changing calls or branch outcomes."""
    old = "            self._chunk_job_inbox.put((job, event))\n"
    text = replace(
        text,
        old,
        old
        + """            if _HANDOFF_DIAGNOSTIC_ENABLED and job.total_blocks >= 4096:
                _vrdma_e2e("producer_chunk_published", job.request_id,
                           engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                           start_block=job.start_block, end_block=job.end_block,
                           total_blocks=job.total_blocks,
                           block_count=len(job.block_ids))
""",
    )
    old = "            self._chunk_deferred.append(_ChunkTask(job, event, state))\n"
    text = replace(
        text,
        old,
        old
        + """            if _HANDOFF_DIAGNOSTIC_ENABLED and job.total_blocks >= 4096:
                _vrdma_e2e("producer_chunk_received", job.request_id,
                           engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                           start_block=job.start_block, end_block=job.end_block,
                           total_blocks=job.total_blocks,
                           block_count=len(job.block_ids),
                           pending_jobs=state.pending_jobs)
""",
    )
    old = "            if last and state.pending_jobs > 1:\n"
    text = replace(
        text,
        old,
        old
        + """                _vrdma_chunk_wait("producer_chunk_wait", job.request_id,
                                  engine_id=self.engine_id, pcp=self.pcp_rank,
                                  start_block=job.start_block, end_block=job.end_block,
                                  total_blocks=job.total_blocks,
                                  pending_jobs=state.pending_jobs,
                                  reason="prior_WRITEs")
""",
    )
    old = (
        "            if reg_data is None:\n"
        "                self._chunk_deferred.append(task)\n"
    )
    text = replace(
        text,
        old,
        """            if reg_data is None:
                _vrdma_chunk_wait("producer_chunk_wait", job.request_id,
                                  engine_id=self.engine_id, pcp=self.pcp_rank,
                                  start_block=job.start_block, end_block=job.end_block,
                                  total_blocks=job.total_blocks,
                                  pending_jobs=state.pending_jobs,
                                  reason="registration")
                self._chunk_deferred.append(task)
""",
    )
    old = "            if not self._chunk_path_supported(job, reg_data):\n"
    text = replace(
        text,
        old,
        """            _handoff_supported = self._chunk_path_supported(job, reg_data)
            if _HANDOFF_DIAGNOSTIC_ENABLED and job.total_blocks >= 4096:
                _vrdma_e2e("producer_chunk_path", job.request_id,
                           engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                           start_block=job.start_block, end_block=job.end_block,
                           total_blocks=job.total_blocks,
                                  pending_jobs=state.pending_jobs,
                           supported=_handoff_supported,
                           remote_blocks=_vrdma_block_count(
                               reg_data.get("local_block_ids")))
            if not _handoff_supported:
""",
    )
    old = "        task.state.failed |= failed\n"
    text = replace(
        text,
        old,
        old
        + """        if _HANDOFF_DIAGNOSTIC_ENABLED and task.job.total_blocks >= 4096:
            _vrdma_e2e("producer_chunk_finished", task.job.request_id,
                       engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                       start_block=task.job.start_block, end_block=task.job.end_block,
                       total_blocks=task.job.total_blocks,
                       pending_jobs=task.state.pending_jobs,
                       failed=task.state.failed)
""",
    )
    old = "        state.finished_blocks = block_ids\n"
    text = replace(
        text,
        old,
        old
        + """        if _HANDOFF_DIAGNOSTIC_ENABLED:
            _vrdma_e2e("producer_terminal_chunk_state", req_id,
                       engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                       pending_jobs=state.pending_jobs,
                       has_registration=state.reg_data is not None,
                       notified=state.notified, failed=state.failed)
""",
    )
    old = (
        "            self.nixl_wrapper.send_notif("
        "ready.agent, notif_msg=ready.key[2])\n"
    )
    text = replace(
        text,
        old,
        old
        + """            if _HANDOFF_DIAGNOSTIC_ENABLED:
                _vrdma_e2e("producer_completion_notified", request_id,
                           engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                           decoder_request_id=_vrdma_notif_request(ready.key[2]),
                           remote_rank=ready.key[1], agent=ready.agent)
""",
    )
    old = "        match = self._pop_matching_finished_blocks(rid)\n"
    text = replace(
        text,
        old,
        """        if _HANDOFF_DIAGNOSTIC_ENABLED:
            _vrdma_e2e("producer_registration_received", rid,
                       engine_id=self.engine_id, tp=self.tp_rank,
                           pcp=self.pcp_rank,
                       decode_engine=reg_data.get("decode_engine_id"),
                       blocks=_vrdma_block_count(reg_data.get("local_block_ids")))
"""
        + old,
    )
    return text


def step_events(stage, indent):
    pad = " " * indent
    return (
        "\n".join(
            pad + line
            for line in [
                "if _HANDOFF_DIAGNOSTIC_ENABLED:",
                (
                    "    for _handoff_rid, _handoff_tokens in scheduler_output."
                    "num_scheduled_tokens.items():"
                ),
                "        if _handoff_tokens > 0:",
                f"            _vrdma_e2e({stage!r}, _handoff_rid,",
                (
                    "                       dp_rank=self.scheduler.pa"
                    "rallel_config.data_parallel_rank,"
                ),
                (
                    "                       engine_epoch=self.scheduler"
                    ".prefill_progress_epoch,"
                ),
                "                       scheduled_tokens=_handoff_tokens)",
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-source-root",
        type=Path,
        required=True,
        help="Captured P0/D0 NIXL source tree and manifest",
    )
    parser.add_argument(
        "--delivery-root",
        type=Path,
        required=True,
        help="Fixed P/D API source package and manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New diagnostic-only source package directory",
    )
    args = parser.parse_args()
    capture = args.runtime_source_root
    delivery_root = args.delivery_root
    out = args.output_root
    manifest = json.loads((capture / "manifest.json").read_text())
    delivery = json.loads((delivery_root / "manifest.json").read_text())
    receipt = {
        "diagnostic_only": True,
        "deployed": False,
        "source_capture": str(capture / "manifest.json"),
        "page_delivery": str(delivery_root / "manifest.json"),
        "flag": "TK300_HANDOFF_TRACE=1",
        "run_id_flag": "TK300_HANDOFF_RUN_ID",
        "base_workers_preserved": {},
        "roles": {},
    }
    for group, role in [("P", "P0"), ("D", "D0")]:
        spec = next(x for x in manifest["roles"] if x["role"] == role)
        receipt["base_workers_preserved"][group] = spec["files"][
            NIXL + "base_worker.py"
        ]["sha256"]
        receipt["roles"][group] = {}
        for name in (
            NIXL + "push_worker.py",
            NIXL + "base_scheduler.py",
            CORE,
            SCHEDULER,
        ):
            if name.startswith(NIXL):
                source = capture / role / name
                expected = spec["files"][name]["sha256"]
            else:
                source = delivery_root / group / name
                expected = delivery["roles"][group][name]
            raw = source.read_bytes()
            assert sha256(raw).hexdigest() == expected
            patched = instrument(name, raw.decode()).encode()
            compile(patched, name, "exec")
            destination = out / group / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert destination.read_bytes() == patched, (
                    "Existing probe differs; inspect before replacing"
                )
            else:
                destination.write_bytes(patched)
            receipt["roles"][group][name] = {
                "original_sha256": expected,
                "probe_sha256": sha256(patched).hexdigest(),
                "added_lines": len(patched.splitlines()) - len(raw.splitlines()),
            }
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "generated_roles": {k: len(v) for k, v in receipt["roles"].items()},
                "base_workers_preserved": receipt["base_workers_preserved"],
                ("validation"): (
                    "source guards and Python syntax only; target rep"
                    "lay remains required"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
