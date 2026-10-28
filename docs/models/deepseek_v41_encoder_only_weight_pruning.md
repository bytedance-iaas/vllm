# DeepSeek V4.1 encoder-only Prefill weight pruning

`KVTransferConfig.dsv41_encoder_only_prune_weights` avoids constructing unused
backbone layers on a dedicated encoder-only Prefill producer. It defaults to
`false` and requires `dsv41_encoder_only_prefill=true` and
`kv_role="kv_producer"`.

For the supported 40-layer model, Prefill executes L0–L19 and uses L20 to publish
Main KV and Indexer K. Pruning replaces L21–L39 with `PPMissingLayer`; L20 remains
complete. The logical layer count, PP partition, cache source names and relay
identities remain intact. Consumer weights and speculative decoding are
unaffected.

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

Preserve the deployment's existing connector options. With a supported
`MultiConnector` composition, enable pruning on the outer producer
configuration. A Store child with `kv_role="kv_both"` must keep pruning disabled.
Do not enable the flag on Decode or put it in a transfer configuration shared
unchanged by both roles.

The existing encoder-only topology and model validation still apply. This
change does not introduce support for additional PP/PCP layouts. Keep
`num_hidden_layers=40`, the validated PP partition and SWA bounded replay.
Dedicated Prefill already rejects a producer speculator; pruning does not remove
Decode DSpark.

The normal weight loader skips missing layers, and post-load MoE/attention
finalization visits retained decoder layers. This change reduces GPU module
residency. It does not filter checkpoint files before reading them, so reduced
checkpoint I/O is not guaranteed.

Validate the effect per rank: model memory, available KV memory, cache groups,
block count and actual prefix reuse. With an uneven PP partition, savings may
occur only on one stage; uniform block allocation can still be limited by that
stage or another stage. Measure cold and warm throughput separately because
encoder-only Prefill already skips full tail computation before pruning.

After replacing a producer process, refresh routers that cache its engine
identity before testing P/D transfer. Confirm that the router uses the new
producer's bootstrap identity.

To restore full producer weights, remove the pruning flag or set it to `false`,
restart Prefill, refresh the router and repeat smoke validation.
