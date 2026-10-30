# DeepSeek V4.1 encoder-only Prefill weight pruning

`KVTransferConfig.dsv41_encoder_only_prune_weights` avoids constructing unused
backbone layers on a dedicated encoder-only Prefill producer. It defaults to
`false` and requires `dsv41_encoder_only_prefill=true` and
`kv_role="kv_producer"`.

For the supported 40-layer model, Prefill executes L0–L19 and uses L20 to publish
Main KV and Indexer K. Pruning replaces L21–L39 with `PPMissingLayer`; L20 remains
complete. The logical layer count, PP partition, cache source names and relay
identities remain intact. Consumers retain their full weights and speculator.

Add the flag to the producer's existing transfer configuration:

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_buffer_device": "cuda",
  "dsv41_encoder_only_prefill": true,
  "dsv41_encoder_only_prune_weights": true
}
```

Preserve the existing connector options. Do not enable pruning on Decode or
share the same pruning-enabled configuration between the roles.

The existing encoder-only topology and model validation still apply. Pruning
does not introduce additional PP/PCP layouts. Keep `num_hidden_layers=40`, the
validated PP partition and SWA bounded replay. The producer must not configure a
speculator.

The normal weight loader skips missing layers. Release MoE finalization, mHC
successor setup and Engram huge-page handling visit retained decoder layers.
Pruning reduces GPU module residency; it does not filter checkpoint files before
reading them, so reduced checkpoint I/O is not guaranteed.

Validate model memory, available KV memory, cache groups, block count and actual
prefix reuse per rank. With an uneven PP partition, savings can occur only on
one stage; uniform block allocation can still be limited by another stage.
Measure cold and warm throughput separately, because encoder-only Prefill
already skips full tail computation before pruning.

After replacing a producer process, refresh routers that cache its engine
identity and confirm that requests use its new bootstrap identity.

To restore full producer weights, remove the flag or set it to `false`, restart
Prefill, refresh the router and repeat smoke validation.
