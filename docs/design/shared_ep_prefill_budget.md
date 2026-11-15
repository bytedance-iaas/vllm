# Experimental shared-EP Prefill time budget

`additional_config.shared_ep_budget` is an explicit, default-off research profile
for four Prefill context buckets that share one expert-parallel domain. The
profile must match DP, PCP, stage costs, model history, query range and measured
error. It does not replace full-input KV admission, cancellation, transfer
completion or the normal fallback scheduler.

The optional `query_alignment` defaults to `1`, preserving the existing token
granularity. Set it to a measured query multiple when the calibration does not
cover arbitrary tails. `query_minimum`, `quantum` and `query_limit` must be
positive multiples of this alignment. A modeled per-request grant is rounded
down to the alignment. The final remainder below `query_minimum` completes on
the native fallback path; it is not dropped or reported as a calibrated step.

For example, a profile with a 1024-token floor, quantum and alignment schedules
a remaining 1728-token query as 1024 modeled tokens followed by a 704-token
native tail. The request's full prompt still participates in KV admission.
This can add a scheduling step and is a workload-dependent performance tradeoff,
not a latency guarantee. Compare complete requests, including tails, against
the original profile before adoption.

Actual charges are checked independently of planned grants. A nonaligned or
below-floor charge invalidates the next model-based decision. Single-DP KV
denial may refit unused grants without issuing new credit; this refit uses the
same alignment and includes already scheduled work. Peer epochs, errors and
unsupported features retain the existing conservative fallback semantics.

The progress endpoint's epoch-scoped fallback counters are diagnostic evidence.
They do not establish model accuracy or serving performance. Preserve invalid
calibration samples alongside the validated range and report native-tail cost
separately from model-qualified work.
