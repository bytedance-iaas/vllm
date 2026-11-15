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

`mla_bmm_backend="flashinfer_cudnn"` tunes the two BF16 MLA BMM shapes with
1024 rows during weight post-processing. Runtime dispatch requires matching
device, dtypes, shapes and strides, including the output view. Other shapes
use `torch.bmm`. Tuning is cached with FlashInfer/cuDNN version identity and
does not enable full-model FlashInfer autotuning. Account for cuDNN workspace
in the actual memory profile.

`enable_sparse_indexer_remap_reuse=true` reuses physical row indices inside
one eager forward and one indexer generation. Every producer invalidates the
map. The page-table identity/layout, request-row mapping, query/TopK shape,
KV row layout and CUDA stream must match. Page tables are immutable for that
forward in the ordinary backend. HiSparse and captured execution use the
existing path. No map or KV tensor persists between forwards.

## Internal progress snapshots

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
The queue counts are views; they do not yet implement shared-EP time quotas.

For diagnostic Torch captures, `profiler_config.detailed_trace_annotation=true`
adds scheduled query/history geometry using the existing worker annotations.
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` additionally marks `gpu_worker: dp_dummy`
and `gpu_worker: sample_and_draft`, including V2 runner calls. Sampling and draft
remain one combined scope, and GPU work is attributed through launch correlation;
scope CPU duration is not a GPU completion measurement. Use a fresh worker process
for each capture session on the TK300 Torch/Kineto package after the observed
repeated-capture destructor crash. Diagnostic request latency is excluded from
performance acceptance.

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
