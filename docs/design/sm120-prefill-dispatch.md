# Experimental SM120 Prefill dispatch

These opt-in paths retain the existing defaults. They require runtime validation
with the chosen model, package versions, hardware and parallel configuration.

`--kernel-config '{"sparse_indexer_topk_backend":"flashinfer_sm120"}'` uses the
FlashInfer experimental SM120 exact scorer/TopK backend for FP8 indexer chunks
whose exact metadata certifies more than 48K visible keys in every nonempty
segment. The integration requires the backported `min_seqlen_k` API. Explicit
causal offsets remain authoritative; this bound only controls scratch routing.
Long chunks bypass the dense logits byte cap. Other chunks retain the 512 MiB
cap (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) and native scoring/selection. Decode
retains automatic TopK dispatch. The opt-in requires DCP=1, uncompressed FP8
indexer KV, head dimension 128, and no TP row sharding or candidate blocks.
PDL is disabled. Startup reserves the greater of the fused scratch bound and
the dense fallback budget; fewer tensor elements are not a promised KV saving.

`mla_bmm_backend="flashinfer_cudnn"` tunes BF16 query-absorb and value-up BMMs
during weight post-processing. The finite row set contains the scheduler token
budget divided by PCP size, cache-block-aligned tails up to that local limit,
and the existing small tail shapes within the limit. For block size 64 and
PCP4, budgets 2048/4096/8192 give local full rows 512/1024/2048 and a tail
quantum of 16. The policy is shared across requests and workload points.
Query tuning covers both compact input and the MRV2 slice whose head stride
includes RoPE columns. Runtime dispatch requires matching device, dtypes,
shapes and strides, including the output view. Other layouts use `torch.bmm`.
Tuning is cached with geometry, device capability and FlashInfer/cuDNN version
identity and does not enable full-model FlashInfer autotuning. Account for
tuning time, transient tensors and cuDNN workspace in the actual memory profile.

The kernel regression covers both Q/V dimensions 128/128 and GLM-5.3's
192/256, with 64 heads, KV rank 512, RoPE width 64, and the linear weight's
transpose/view layout. These numerical/layout checks do not establish serving
throughput, startup cost or KV capacity.

`enable_sparse_indexer_remap_reuse=true` reuses physical row indices inside
one eager forward and one indexer generation. Every producer invalidates the
map. The page-table identity/layout, request-row mapping, query/TopK shape,
KV row layout and CUDA stream must match. Page tables are immutable for that
forward in the ordinary backend. HiSparse and captured execution use the
existing path. No map or KV tensor persists between forwards.

## Internal progress snapshots

The optional shared-EP stage profile also accepts `key_ms_per_token` (default0).
It models an independent visible-key preparation envelope in addition to the
query/history scan product, not a causal kernel timing. A bucket grant uses its
maximum history and real candidate count to bound per-request key preparation;
actual charges retain the number of requests really scheduled. Idle buckets do
not pay key preparation. Missing/invalid actual counts fail closed for a nonzero
key coefficient. All coefficients, query limits and held-out error must match
the measured topology and dispatcher. Legacy profiles retain their zero-key cost.

The query floor is checked per request as well as per bucket. A small known tail
cannot become qualified by another request's larger bucket total. Actual charge
qualification also retains per-request floor violations for the next common
decision. On single-DP, a newly confirmed prefix may reveal an unqualified tiny
query after planning; that call abandons the research quota without rewinding
visited requests or KV. The next decision restores ordinary fallback ordering.
Multi-DP does not make a rank-local transport decision or skip a collective.
For a KV-denied local refit, the progress observation retains the unconsumed
grants and actual charges immediately before refitting. Compare these with the
resulting grants instead of assuming any query above the quantum was borrowed.
Progress also includes bounded reason-keyed `shared_ep_fallbacks` counters for
the current engine epoch. A fast fallback can disappear from the last-step
observation before a poll arrives; compare counter deltas rather than changing
the query workload or assuming an unseen event did not happen. Counters do not
change allocation, transport or fallback decisions.

`VLLM_ENABLE_PREFILL_PROGRESS=1` enables `POST /internal/prefill/progress` with
`{"request_ids":["external-request-id"]}`. It is disabled by default and intended
for a trusted internal router. Each response contains bounded snapshots from
all managed engines, each with a distinct epoch and monotonic sequence. IDs
are the engine's external IDs (for OpenAI requests, the response ID with the
completion prompt suffix), and each snapshot also identifies the internal ID.
Query at most 128 IDs at a time.

`seq` orders queries, while `scheduler_step`, `scheduled_step_seq`, and
`processed_step_seq` identify scheduler calls, nonempty submitted steps, and
settled outputs respectively. PP can have several submitted steps in flight.
The progress counters advance for every nonempty step while progress is enabled;
they are distinct from connector-specific deferred-free fences, which may remain
zero on a producer. They do not change KV lifetime or release ordering.
Poll counts and CPU enqueue spans must not be used as completed-wave counts.
The engine adds `dp_wave` and `dp_step` when it participates in DP coordination,
or null when there is no shared counter. A DP wave spans a collective
running-to-idle epoch; it is not one GPU forward. These observations are not
quota grants and cannot authorize skipping a peer or changing an in-flight step.

Confirmed tokens exclude in-flight execution and speculative placeholders.
The full input length determines the context bucket: below 16K, 16–64K,
64–256K, or at least 256K. A cached prefix does not change that bucket.
Cache counts remain unknown until engine admission; missing requests are
explicitly `UNKNOWN`. Terminal summaries are bounded to 4096 entries. HTTP
completion and KV/GPU drain are distinct: deferred transfers and frees must
settle before `drained=true`. Consumers must reject stale sequences and must
not settle an old execution from a new epoch or an unknown request.

`free_kv_blocks` is an observation of the real pool, not a reservation or a
guarantee of request capacity. Actual KV admission remains in the engine.
The queue counts alone are views, not shared-EP grants.

## Experimental shared-EP budgets

An explicit `additional_config.shared_ep_budget` cost profile enables a
synchronous engine-coordinator budget before each DP step. This is disabled
by default; there are no baked-in TK300 latency coefficients. All DP engine
processes must use the same source, profile and flag. The current opt-in requires
EP, at least two DPs, matching PCP/PP geometry, FCFS and chunked prefill. Changing
the DP topology requires a new calibration; elastic resizing is not supported.

The profile has `calibration_id`, `dp_size`, `pcp_size`, `history_limit`,
`query_limit`, `protect_ms`, `throughput_ms`, and one entry per PP stage in
`stages`. Stage entries contain `private_ms_per_row`, `scan_ms_per_pair`,
`ep_ms_per_token`, `draft_ms_per_row` (sampling/draft combined), and `fixed_ms`.
Optional controls are `quantum` (128 tokens, PCP-aligned), `aging_steps` (8),
`error_margin` (1.25), `max_relative_error` (0.25), and
`observed_relative_error` (0). These defaults are research controls, not
production weights or calibrated performance promises. Record calibration
coverage and measured held-out error before enabling the profile. Cumulative
kernel time alone is not a wall-time calibration.

Each rank sends four bounded demand envelopes and the preceding step's actual
scheduled charge through the existing CPU DP process group. The model uses
the maximum private stage cost plus shared EP cost from the **sum** of real
per-DP rows, with dummy peer rows included. Full prompt length determines the
bucket; unadmitted cache hits remain unknown and use a conservative history
envelope. Scheduling caps actual tokens only after the ordinary engine prefix
lookup. Equal active-bucket service credit, bounded deficits, aging and rank
rotation allow unused shares to be borrowed without strict short-first policy.
Short demand chooses the protection budget unless an aged bucket calls for
a throughput step. These are estimated service shares, not hard isolation.

Grants are consumed once by the real scheduler; block holders keep their
separate drain-first queue, and existing PP/MTP alignment and release fences
still govern execution. Actual scheduled work is charged on the next DP
step, not requests that failed KV allocation. Confirmed/settled progress is
reported separately: submitted work is not GPU completion. Atomic KV admission
checks the full input against the real deduplicated allocator after subtracting
future blocks owed to other resident requests plus speculative step headroom.
Already-held transfer leases are unavailable in the real free pool. No Router
snapshot can grant KV, and new requests cannot repeatedly pass a first-chunk
check against an overcommitted full-input pool.

A cancelled request's terminal progress is retained until its submitted GPU
outputs settle. `drained=true` requires both that completion fence and any
existing deferred KV release fence; it then reports zero in-flight tokens.
The completion observation does not change the allocator's release ordering.

`dummy_tokens_per_pcp` records the actual idle target rows per PCP worker.
For MTP5 in this eager recipe it is six, so an idle PCP4 peer contributes 24
EP rows. The scheduler rejects a profile whose dummy query differs from the
model's speculative plus sampled-token width; private prefill rows remain
partitioned by PCP. Do not infer idle geometry from the prefill quantum.

`query_minimum` and the query/history limits bound the calibrated shape range.
The quantum must be at least that query floor. Smaller demands or actual charges
fall back collectively; a short tail is not extrapolated from large-query data.

Visible profile/epoch/wave mismatch, stale steps, out-of-range history, excessive
declared calibration error, unsupported multimodal/streaming/decode traffic or
a budget smaller than one quantum cause a **collective** fallback to the original
FCFS/token strategy. The opt-in full-input KV qualification remains active.
Communication failure follows the existing fatal DP error path, never a
rank-local skip or reuse of stale grants. Progress responses include
`shared_ep_budget` with wave/step, mode, grants, estimated cost, deficits, ages
and fallback reason. Real multi-DP/PP service, cancellation, restart and mixed
load calibration are required before adoption; CPU state tests are not that
validation.

Budget collectives follow a running DP wave or real unfinished request work.
An idle rank flushing finished notifications or connector cleanup does not join
them: those local scheduler steps can run after its peers have stopped.

With a single DP replica, the same planner runs at the scheduler boundary without
a transport collective. It retains bucket grants, aging, actual charges,
calibration fallback and atomic KV admission; the step identity is the local
scheduler sequence. Profiles must match the actual PCP/PP topology and query
budget. A profile calibrated for DP2/PCP4 is not valid for DP1/PCP8.

After a real KV allocation denial, the single-DP path refits the unconsumed
grant among the remaining, unvisited waiting candidates. It counts actual
already-scheduled q/history against the same step budget and token limit; no
second credit or age update is issued. Denied and already-visited requests do
not receive that loan. Prefix lookup and final KV eligibility remain owned by
the ordinary allocator. An actual query below the calibration floor prevents
this local refit. Multi-DP grants still require their existing joint decision.

For diagnostic Torch captures, `profiler_config.detailed_trace_annotation=true`
adds scheduled query/history geometry using the existing worker annotations.
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` additionally marks `gpu_worker: dp_dummy`
and `gpu_worker: sample_and_draft`, including V2 runner calls. Sampling and draft
remain one combined scope, and GPU work is attributed through launch correlation;
scope CPU duration is not a GPU completion measurement. Use a fresh worker process
for each capture session on the TK300 Torch/Kineto package after the observed
repeated-capture destructor crash. Diagnostic request latency is excluded from
performance acceptance.

For a lightweight completion-timing window, Model Runner V2 workers provide
`start_prefill_timing(max_steps="32")` and `read_prefill_timing(stop="true")`
through the existing development `collective_rpc` endpoint. This endpoint must
be used only on an explicitly task-owned development service; do not enable it
on production or expose it as a public API. The recorder is inactive by default,
holds at most64 local records, and rejects restarting over unfinished CUDA events.
It does not synchronize the GPU in the serving path. Before worker control RPCs,
wait for both real requests and the DP engine wave to become idle: zero request
counts alone can leave dummy EP steps running. Read every relevant worker and
reject incomplete samples or missing ranks.

The target interval begins after input/DP/PP preparation and surrounds the actual
model forward and its ordinary joins on the current stream. The separate worker
sampling interval includes sampling, draft, and PP sampled-token dependencies;
non-last PP intervals can therefore contain broadcast waits. The metadata records
scheduled q/history, actual local/padded rows, and the agreed per-DP local row
counts. Dummy model forwards are not represented as real request throughput.
CUDA event elapsed time includes stream dependencies and host launch gaps; it
is not private kernel-only time, removable latency, or full-request TTFT.
Use source/shape-matched stage measurements and existing trace attribution to
test a cost model; do not identify cross-host timestamps or sum stage waits as
an additive speedup. Repeated event windows do not start a Kineto profiler.

## Idle DP drafting

`enable_idle_dp_draft_skip=true` is an experimental CUDA/V2, single-module MTP,
static EP/DP, eager-only option. A runtime idle DP worker votes to allow the
existing AND-reduced unused-draft skip. Every real rank must still allow it;
if a real request needs drafts, all peers keep the original proposal count.
The idle proposal uses the agreed zero count too, preventing divergent draft
collectives. The ordinary zero-count draft prefill pass remains and maintains
draft KV; only unused draft sampling/decode work is skipped. Profiling and graph
warmup calls keep their original dummy behavior. The option defaults off and
requires real single-/mixed-DP, prefix and PD regression before adoption.

## Cross-stage TopK

`enable_pp_indexer_handoff=true` adds logical `int32` TopK to the existing
hidden/residual intermediate tensors in the NVIDIA DSA model. It deliberately
preserves both hidden and residual tensors from the local model version.
Each outgoing microbatch clones its TopK slice so a later forward or MTP
write cannot overwrite an asynchronous send. The receiver copies the logical
indices into its own buffer before shared-index consumers execute; physical
page mapping stays in that stage. Receive-buffer allocation preserves TopK
dtype and geometry. This option adds a transfer at each PP boundary and does
not select a layer partition automatically. Validate PP/PCP, multiple in-flight
microbatches, empty/dummy batches and NIXL layer ranges before adopting a new
partition.
