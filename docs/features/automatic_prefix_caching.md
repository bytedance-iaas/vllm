# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Hybrid Mamba models

With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- prefix caching enabled
- EAGLE/MTP speculative decoding on the Mamba group
- `--prefix-match-unit` smaller than the Mamba block size
- the model does not use multi-module MTP

```bash
vllm serve <hybrid-model> \
    --prefix-match-unit 64 \
    --enable-mamba-shared-prefix-checkpoint
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. For hybrid Mamba models that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, and that is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates both at startup and names the offending sizes in the error. Read the served block size from the startup log. 64 is a reasonable starting point.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).

## Internal request capacity snapshots

With `VLLM_ENABLE_PREFILL_PROGRESS=1`, the Python server exposes
`POST /internal/decode/capacity`. The request groups external request IDs by
global data-parallel rank, with at most 64 ranks and 128 IDs per rank:

```json
{"request_ids_by_rank": {"0": ["chatcmpl-request-a"], "1": ["chatcmpl-request-b"]}}
```

Each engine reads its own request group in one scheduler utility call. Its
snapshot contains the scheduler `epoch`, observation `seq`, `dp_rank`, physical
`free_kv_blocks`, `total_kv_blocks`, cache `group_block_sizes`, and the existing
request progress and release fences. `allocated_pages` counts distinct physical
blocks with active references held by that request, excluding the null block.
An unknown request has `allocated_pages: null`. `unmanaged_requests` counts
live requests omitted from that rank's query.

Free capacity includes cached blocks with no active reference. Cache residency
alone therefore does not justify subtracting a request's prefix from its page
obligation. A terminal request can still have an outstanding execution fence;
callers must inspect `drained` rather than infer completion from a zero page
count. A new scheduler has a new epoch, so old request observations cannot be
used to release reservations in the replacement scheduler.

The default request observes capacity without reserving or canceling requests.
A controller can additionally supply `close_request_ids_by_rank` and
`expected_epochs_by_rank`. Only the engine with the matching epoch acknowledges
closure. It aborts matching active requests and retains a closure record so a
late registration of the same external ID cannot allocate pages. This also
protects requests canceled before they appeared in a progress snapshot.

Managed Decode requests carry `router_admission_epoch` in `kv_transfer_params`.
The scheduler aborts registrations from a different epoch and duplicate managed
registrations before scheduling them. A stale close operation cannot affect a
new engine. Closure records are capped at 4,096 per rank and scheduler epoch;
new IDs are refused once the cap is reached, rather than evicting records that
could permit late work. Controllers must keep refused or uncertain closures
charged until authoritative evidence permits release.

`closed_request_ids` acknowledges that no future registration can proceed. It
does not bypass an outstanding execution fence: a known terminal request must
also have `drained: true` before its reservation is retired. An acknowledged
unknown request has no prior execution and is protected against late arrival.
A placement controller must reconcile unpublished and future page obligations
atomically with selection and preserve request generations on every release.
