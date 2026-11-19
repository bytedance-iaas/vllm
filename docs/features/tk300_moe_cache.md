# Verified TK300 MoE cache loading

The optional worker extension
`vllm.plugins.tk300_moe_cache.MoEAutotuneCacheWorker` loads a verified
FlashInfer CUTLASS MoE cache after complete model startup warmup. Global
FlashInfer autotuning remains disabled. This avoids profiling or applying the
serving bucket to the larger startup dummy calls.

Add the following to the existing qualified deployment command:

```bash
export TK300_MOE_AUTOTUNE_MANIFEST=/path/to/manifest.json
# Additional vLLM argument:
# --worker-extension-cls vllm.plugins.tk300_moe_cache.MoEAutotuneCacheWorker
```

The current manifest is qualified for full78-layer GLM-5.3-NVFP4, MTP5,
TP1/PP4/PCP4/DP1/EP4, AG/RS, PP18/20/20/20, q8192/slots24, FP8 ds-MLA
KV/block64 and Graph NONE. Its native8192 bucket covers rows4097–8192;
8192 and6528 were numerically checked across PP/EP ranks. Other shapes,
constructor signatures and MTP retain native defaults.

Manifest and native cache belong to the deployment configuration repository.
The loader checks recipe, checksum, exact native keys/GEMM stages and
FlashInfer library/device metadata before loading an immutable private
snapshot. Missing or incompatible artifacts preserve defaults and the
existing BMM cache. The baseline without a sampling-policy configuration
field has the existing legacy policy; an explicit policy still must match
the qualified recipe.

The original `0.7.2.post1+tk3005.1de96ee5` cache declares16/28 native
GEMM1/GEMM2 tactics. The qualified complete-source `0.7.2.post1` build
declares16/32; its real-input replay still selects10/41. The manifest count
must match its qualified library identity. Regenerate the native artifact and
verify its numeric/hit/fallback behavior when switching between these builds.

Leave `TK300_MOE_AUTOTUNE_MANIFEST` unset to disable loading. Changing the
software, checkpoint, device or parallel recipe requires requalification;
TK300 tactics must not be reused on H20 solely because the code shares an
integration branch. Loading does not add GPU workspaces, runtime profiling
or collectives. Filesystem tests validate rejection, not model performance;
use the full-model diagnostic and real serving path to verify actual hits.
