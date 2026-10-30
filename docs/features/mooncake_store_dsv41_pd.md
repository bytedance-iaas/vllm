# DSv4.1 encoder-only PD with a canonical DRAM Store

This opt-in profile composes direct `MooncakeConnector` P→D transfer with
`MooncakeStoreConnector` prefix persistence in the DSv4.1-Flash fork. Its canonical
source includes the validated Store changes through `62f9fa293d`. This integration
preserves the generic Store no-increment lookup path and uses the repository's
`regex` and `torch.accelerator` APIs.

The initial boundary is P TP1/PP2/PCP4 with the `10,30` PP split, and D
TP1/PP1/PCP1 with one scheduler per DP engine (including DP8/EP8). Keep the
existing MRV2, FP8 DS MLA, 128-token SWA bounded replay and DSpark configuration.
Existing encoder-only checks still apply. The original direct-only and legacy
Store configurations use their existing paths.

## Module responsibilities

All profile-specific modules live under
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/`.

| Module | Responsibility |
| --- | --- |
| `dsv41_profile.py` | Validate the role, topology, canonical format and resource limits. |
| `dsv41_layout.py` | Build rank-independent keys and rank-local KV component views. |
| `dsv41_lookup.py` | Coordinate asynchronous existence checks with reusable IPC and byte limits. |
| `dsv41_scheduler.py` | Select contiguous hits, schedule save/load ranges and retain block references. |
| `dsv41_worker.py` | Register buffers, execute native GET/PUT, track completion and expose telemetry. |
| `connector.py` | Select canonical or legacy scheduler/worker implementations. |

Directional completion contracts in `base.py`, `utils.py` and
`multi_connector.py` distinguish direct-transfer quorums from Store quorums.
The legacy `scheduler.py` and `worker.py` retain their existing data path.

## Data flow

P writes complete global-cache chunks and optionally reloads historical prefixes.
D receives this request's KV from direct Mooncake, then optionally writes its
confirmed prompt and generated KV. D never looks up the Store. The next turn's P
can therefore reuse full chunks produced by D, independent of the selected DP
engine. GPU KV still has to be resident during attention; this is a shared prefix
cache rather than active-attention execution from host memory.

Each chunk contains the Main-KV and Indexer-K components from source layers
2, 8, 14 and 20. Keys include the explicit namespace/revision, payload ABI and
chained token hash, without PP/PCP/DP rank IDs. Only PCP rank 0 writes a P stage's
owned components. Every P worker restores its own views. The stage-1 layer-10
Main-KV relay uses the layer-8 key with its own destination block table. SWA,
compressor continuation, Engram lookback and draft state follow existing replay
and reconstruction paths.

A hit requires all eight components at every preceding chunk. Partial pages and
incomplete tails are not persisted. P restores eight workers before resuming;
direct P sends retain their two-worker completion quorum. A complete local hit
at an unaligned canonical boundary takes the ordinary compute path rather than
overwriting a partially shared page.

## Connector configuration

Keep the current direct Mooncake child's extra configuration (RDMA devices,
bootstrap settings and other existing options). The following P JSON shows only
the connector changes; it does not replace the model or parallelism arguments.
Set `model_revision` to an immutable identifier for weights and compatible model
code. Both roles must have identical namespace, revision, KV geometry and hash
granularity.

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "dsv41_encoder_only_prefill": true,
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "pd_store_profile": "dsv41_flash_v1",
    "connectors": [
      {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_buffer_device": "cuda",
        "dsv41_encoder_only_prefill": true,
        "kv_connector_extra_config": {}
      },
      {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_buffer_device": "cuda",
        "dsv41_encoder_only_prefill": true,
        "kv_connector_extra_config": {
          "store_format": "dsv41_canonical_v1",
          "pd_role": "prefill",
          "cache_prefix": "dsv41-flash-prod-v1",
          "model_revision": "replace-with-immutable-model-and-code-revision",
          "enable_lookup": true,
          "lookup_min_prompt_tokens": 2048,
          "load_async": true,
          "save_decode_cache": false,
          "max_pending_jobs": 32,
          "max_pending_lookups": 32,
          "max_pinned_blocks": 4096,
          "max_pinned_bytes": 536870912,
          "max_inflight_bytes": 268435456,
          "lookup_timeout_ms": 1000,
          "store_stall_timeout_s": 30,
          "failure_cooldown_s": 5
        }
      }
    ]
  }
}
```

For D, make these changes to the same structure:

- Parent and direct Mooncake child: `kv_role="kv_consumer"`.
- Parent: `kv_load_failure_policy="fail"`, preserving reliable direct PD delivery.
- Store child: keep `kv_role="kv_both"` and set `pd_role="decode"`,
  `enable_lookup=false`, `save_decode_cache=true`.

For shadow writes, set P `enable_lookup=false`; initially leave D
`save_decode_cache=false`, then enable it after P writes and reads are validated.
Keeping D saves disabled makes it a capacity-only owner: its Store still
registers a DRAM segment, while KV buffer descriptors and transfer executors
are omitted. P prompts below `lookup_min_prompt_tokens` skip lookup but can
still save complete chunks. Zero allows lookup for all eligible prompts.
Do not add `--kv-offloading-size`. The new profile requires explicit
`--kv-cache-dtype fp8_ds_mla` and the existing encoder-only parallelism settings.
Custom connector modules and external-launcher DP execution are rejected.
Multimodal, LoRA and prompt-embedding requests bypass the Store subpath.

## DRAM setup and budgets

Install the same Mooncake native package used by the fork's Store connector. It
must provide `register_buffer`, `batch_is_exist`,
`batch_put_from_multi_buffers`, `batch_get_into_multi_buffers`, and synchronous
completion of native buffer accesses when transfer calls return. Configure
`MOONCAKE_CONFIG_PATH` on all vLLM workers as described in the
[Store usage guide](mooncake_store_connector_usage.md). `enable_offload` must be
false. The master and owners must also be configured for DRAM only; the vLLM
client flag cannot verify or disable a remote owner's SSD tier.

Embedded clients contribute `global_segment_size` per worker. For externally
managed DRAM owners, use `mode="standalone-store"` with
`global_segment_size=0`. Requester buffers still consume `local_buffer_size`
per worker. Budget P's eight clients and D's eight clients separately, including
host memory and registered buffers. Shared keys may be served by remote owners;
CPU DRAM L2 does not imply NUMA-local placement. Existing Store
`preferred_segment` configuration is passed to Mooncake.

`max_pending_jobs` and `max_inflight_bytes` bound local queued plus executing
transfers. GET and PUT use separate single-thread executors. The scheduler also
bounds save jobs, referenced blocks and a conservative byte estimate based on
cache-group page sizes. These are block-reference budgets, separate from the
Store's DRAM capacity. Save pins cover only complete pages in that job's range.
Objects are sent in batches of at most 32 chunks. Scheduler save ranges can be
shortened to fit the pin budget; worker admission can skip oversized saves.
Lookup length is capped by the in-flight byte budget.

Decode's publication boundary is capped by both computed tokens minus in-flight
tokens and the accepted token history. Stale output cannot advance it. Finish-time
jobs freeze and pin block tables before normal request cleanup, and are submitted
on zero-forward polling steps too. The final sampled token has no KV until it has
actually been computed; it is not published merely because sampling accepted it.

## Failures, reset and telemetry

Lookup timeout/unavailability becomes a miss. Failed GETs report request-level
failure after all workers finish native I/O, then recompute; that request bypasses
Store lookup on retry. No group-local block IDs are exposed as HMA-wide invalid
IDs. Save failure/full queues become future misses. Existing immutable objects
are skipped on PUT, including a concurrent writer winning the same key.
Publication is best effort: a failed or skipped chunk may leave a hole, so lookup
always checks continuity and never assumes later chunks imply earlier ones exist.

A stalled native call stops new admissions and retains its GPU references. An
elapsed timeout does not establish DMA quiescence. Shutdown waits for native calls
to return before closing the Store or releasing registered storage; a permanently
hung native call may therefore require terminating and restarting the worker.
There is no hard-cancel promise from this adapter.

Prefix-cache reset returns false for this shared profile; it never invokes global
`remove_all`. Drain/restart both roles with a new namespace/revision for weight or
ABI changes. Reset is not a live namespace-rotation API.

Existing operation metrics cover `lookup_exists`, `save_exists`, `save_put` and
`load_get`; `save_admission`/`load_admission` with `status="skipped"` expose
resource or health rejection. `vllm:mooncake_store_resources{resource=...}` exposes
pending jobs/bytes, unhealthy workers and scheduler pinned jobs/blocks/estimated
bytes, summed per engine. Pin snapshots are reported by one worker per engine.
No request IDs or hashes become metric labels. Operation samples are bounded;
under sustained overload the oldest unreported samples can be discarded.
GPU/Store hit tokens remain visible through existing prefix-cache statistics.

## Validation scope

The existing 8P+8D deployment has exercised P reload after GPU-prefix eviction,
cross-turn D-generated KV reload, and sustained mixed-length traffic. Its current
encoder-only configuration uses P lookup threshold 2048 and capacity-only D.
The two-hour Bailian Trace A comparison measured Prefill duration -15.92%,
Mean TTFT -5.36%, weighted TPOT +1.91%, and Mean TPOT +2.13% versus L1-only.
These workload-specific results do not establish a GET-saturation limit,
bounded master memory growth, or every injected-failure case below.

This source migration used syntax, style and source-equivalence checks; it did
not rerun local unit tests, model evaluation or GPU performance A/B. The serving
deployment still loads prepared role-local runtimes; migrating source does not
update a serving image or those runtimes.

The following commands are targeted validation entry points. Install the
repository's test dependencies
in the target GPU environment first.

```bash
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_output_aggregator.py tests/v1/kv_connector/unit/test_multi_connector.py tests/v1/kv_connector/unit/test_mooncake_store_layout.py tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py tests/v1/kv_connector/unit/test_mooncake_store_worker.py tests/v1/kv_connector/unit/test_mooncake_store_connector.py -v
.venv/bin/python -m pytest tests/v1/kv_connector/mooncake_integration/test_store_canonical_gpu.py -k descriptor -v
VLLM_RUN_CANONICAL_STORE_TEST=1 MOONCAKE_CONFIG_PATH=/path/to/isolated-dram-store.json .venv/bin/python -m pytest tests/v1/kv_connector/mooncake_integration/test_store_canonical_gpu.py -k real_store -v
```

The descriptor probe checks packed strides and surrounding sentinel bytes. The
real-Store probe uses one GPU, unique synthetic keys and an isolated DRAM Store;
it leaves keys for that test Store's eviction policy. These probes do not validate
the model's distributed lifecycle.

Then validate the current 8P+8D deployment in this order:

1. Direct-only baseline, including cancellation and sender-slot recovery.
2. P shadow writes with real Store objects, correct output and pin drain.
3. P save → GPU-prefix eviction → Store reload, including stage-1 relay bytes.
   Exercise 31/32/33, 63/64/65, 127/128/129 and long-prefix boundaries.
4. Enable D saves under DSpark and verify the next turn's P loads complete chunks
   of previously generated output, including DP route changes and EOS.
5. Inject lookup timeout, eviction between lookup/GET, partial GET/PUT failure,
   cancellation, preemption and master/owner restart. Require no premature block
   release, late DMA corruption, stuck requests or sender-slot leaks.
6. Compare feature-off, cold-write, P-hit and generated-KV-hit cases with identical
   workloads. Measure quality, TTFT, p99 TPOT, throughput, transfer bytes, host
   memory and pin/queue peaks. Do not infer performance from static checks.

The matching Helm chart is `vllm/deepseek/deepseek-v4.1-flash-offload` in
ServingKit. Image publication remains a separate stage. Roll back by draining
and restoring the direct-only connector
configuration; shared DRAM objects may expire independently.
