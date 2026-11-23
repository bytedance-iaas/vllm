# TK300 role compatibility and diagnostic tools

`apply_page_admission_patch.py` applies the minimal API backport to the exact
older Decode source version recorded in `h20-page-admission-manifest.json`.
Keep the role's compute files and customized NIXL `base_worker.py` intact.

`build_timely_push_package.py` packages the five fixed producer modules for
step-wise push. It requires the exact bound runtime metadata hash and preserves
only its already-deployed compatibility hash function after verifying that the
rest matches the fixed baseline. It never copies producer modules to H20. The
manifest records canonical and delivered hashes; GPU serving remains a separate
validation gate.

`generate_handoff_probe.py` generates diagnostic logs from captured Prefill and
Decode NIXL sources and a fixed API delivery package. It checks original file
hashes, reuses existing request wait probes when present, and changes only
logging in `push_worker.py`, `base_scheduler.py`, `scheduler.py`, and `core.py`.
It creates no serving resources and makes no performance or correctness claim.

For step-wise producers, long-request logs include each published block range,
registration matching, waits for earlier WRITEs, job completion, and the final
notification. Wait records are deduplicated by job state and restricted to
requests with at least 4,096 blocks. Opaque or malformed optional diagnostic
fields cannot interrupt execution. These logs describe reported transfer state;
they do not independently establish GPU kernel or DMA timing bounds.

```bash
.venv/bin/python tools/tk300_compat/generate_handoff_probe.py \
  --runtime-source-root /artifacts/handoff-runtime-source \
  --delivery-root /artifacts/delivery-page-r2 \
  --output-root /artifacts/handoff-probe
```

The captured source manifest contains `roles` entries for `P0` and `D0`, each
with per-file SHA256 values. The API delivery manifest records Prefill and Decode
files under `roles.P` and `roles.D`. The output includes original and generated
hashes; it does not overwrite a different existing probe. Keep captured runtime
source, models, request logs, credentials, and generated deployment artifacts in
the task's evidence directory rather than committing them with this tool.

Enable logs only in a separate diagnostic runtime with
`TK300_HANDOFF_TRACE=1` and a unique `TK300_HANDOFF_RUN_ID`. Legacy
`vrdmae2e-` request tags remain supported. Logs record registration, scheduler and
writer metadata, write submission and handle retirement, notification source
agent and arrival, engine consumption, the existing producer barrier report,
KV-ready scheduling, and first model-executor dispatch and result processing.
They retain wall and monotonic clock anchors without logging prompts or tokens.
Opaque native handles and byte-valued agents are serialized as strings so they
cannot silently drop a whole event. Producer records include the initialized
pipeline rank when available; registration records retain the peer rank tuple.
With the diagnostic flag enabled, each Decode-capacity snapshot also returns
`handoff_clock` with integer wall/monotonic nanoseconds and the local anchor
bracket. A target-runtime client can bound clock offset using the request's
send/receive interval. This is a read-only clock echo, not a clock adjustment;
the field is absent when probes are disabled.

Model dispatch timestamps describe the real CPU entry into the executor, not a
GPU kernel timestamp. Retired handles do not establish the exact instant when
remote data became readable. A notification count alone does not establish four
independent producer stages: join source agents, request IDs, producer logs and
engine generations. The generator adds no polling, wakeup, GPU synchronization
or completion notification; the existing transfer and scheduling rules remain.

Validate the generated sources in each exact role image, then verify real
request output and complete event coverage before using the timeline. Record
cross-host clock error before subtracting wall timestamps. Disable probes for
formal performance windows and restore the fixed delivery package afterward.
The upstream completion-order design at
[vLLM PR 68686](https://github.com/vllm-project/vllm/pull/68686) is a correctness
reference; this tool neither applies that fix nor proves its failure mechanism
occurs in a particular deployment.
