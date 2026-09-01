# IAAS `v0.27.0` Migration Plan and Status

## 1. Scope and Baseline

This document defines and tracks the selective migration of `iaas_main` onto
upstream `v0.27.0`.

| Item | Revision |
| --- | --- |
| Source branch | `iaas_main` |
| Source head | `8da5f70fca26827d8542722f6c8663f11bf2b32c` |
| Target tag | `v0.27.0` |
| Target commit | `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf` |
| Work branch | `wyc/iaas-v0.27.0` |
| Current implementation head | `f1d5ef2237` |
| Merge base | `dcfebf93f4eccf30f71872283331eee757915daf` |

The raw range `v0.27.0..iaas_main` contains 86 commits because the release
branches diverged before `v0.26.0`. The meaningful IAAS stack starts after
`v0.26.0` (`568afb3a13806beb53bb2e6bd518269357b237c0`) and contains:

- 66 non-merge commits.
- 8 merge commits.
- 12 upstream `v0.26.0` release commits before the IAAS stack.

The migration should therefore be a selective reconstruction on
`wyc/iaas-v0.27.0`, not a blind `git rebase --onto` of all 86 commits.

## 2. Analysis Method

The classifications below use four checks:

1. `git cherry -v v0.27.0 iaas_main` for patch equivalence.
2. Commit-message source references and upstream PR numbers.
3. Symbol and behavior comparison between the final `iaas_main` tree and
   `v0.27.0`.
4. `git apply --check` for each non-merge commit against the clean target.

`git apply --check` currently reports only 11 commits as mechanically clean.
That result is not sufficient to prove semantic compatibility. In particular,
Mooncake, CUDA graph, MoE, and cache-layout patches still require focused tests.

Disposition terms:

- **DROP**: already present upstream, superseded, merge-only, or formatting-only.
- **PORT**: behavior is absent from `v0.27.0`; migrate it.
- **SPLIT**: only part of the old commit is still needed.
- **SQUASH**: retain the behavior/test but fold it into a new feature commit.
- **HOLD**: do not migrate by default until correctness evidence is available.

## 3. Executive Summary

| Area | Upstream `v0.27.0` state | IAAS work to retain | Recommended approach |
| --- | --- | --- | --- |
| ByteIAAS build/release | No ByteIAAS workflows | Image/wheel workflows, tags, zstd/nydus, mirrors, HPC/Humming installation | Rebuild as 2-3 target-native commits |
| DSV4 correctness | Several fixes landed, but not all IAAS fixes | MTP RoPE, FP8 metadata inference, BF16 O-proj fallback, metadata guards | Split `3c7ed3742c`; drop only covered hunks |
| DSV4 IndexCache | Upstream has metadata caching and packed overlays, not IAAS layer-skipping IndexCache | Layer skip policy and memory accounting | Manual redesign on the new cache layout |
| SM90 MegaMoE | Upstream target remains tied to its own DeepGEMM integration | SM90 FP4/FP8 dispatch, empty-rank collectives, EPLB tensors | Port code and dependency pin together |
| Dynamic SD | Basic schedule and MRV2 full-CUDA-graph support exist; DP is explicitly disabled | Budget reclaim, runtime K, DP-global K, DSpark graph families | Build on upstream implementation; do not replace it |
| DFlash/DSpark correctness | Core implementations exist | Async bounds, draft cache ownership, producer-only mode, prefix masking, warmup fixes | Port as focused correctness commits |
| Mooncake | PP and basic heterogeneous TP exist | Shared aliases, GQA replica mapping, stale completions, long-request gate, fail-closed topology checks | Extend current connector incrementally |
| PCP | Upstream MRV2 virtual-batch PCP exists | Only DSV4 hybrid-backend and direct Mooncake gaps | Do not port legacy PCP implementation wholesale |
| MiniMax HPC | Generic HPC backend exists | MiniMax-M3 MXFP8 K32 path, BF16 fused path, workspaces | Port as one backend stack |
| MiniMax W4A8 | Basic CUTLASS W4A8 exists | SwiGLU parameters, DeepEP LL/HT, scheduling, Humming W4A8 | Port in dependency order |
| Prefill token buckets | Absent | Opt-in prompt-length buckets with a whole-step token cap | Reimplement on the v0.27 scheduler; omit metrics and tracing |

### 3.1 Current Execution Status

The CPU/static migration is implemented through `f1d5ef2237`. That revision has
56 target-native commits after `v0.27.0`, including this document's initial
planning commit.

| Area | Status | Target commits |
| --- | --- | --- |
| DFlash/DSpark foundation | Complete | `aa5a75d799`, `86ec283e47`, `0c142e5b8e`, `66bf65058d` |
| DSV4 correctness and IndexCache | Complete | `549942052e`, `effe7093cf` |
| SM90 MegaMoE | Core migrated; Kimi DeepGEMM ABI fix and GPU validation pending | `78dea35eb9`, `b84298dbe1` |
| Dynamic SD | Core migrated; DSpark reduced-K and custom-scheduler fixes pending | `9e006972e2` through `d00896ae74` |
| Mooncake | Complete except direct PCP | `cca9a49550`, `5b701d086f`, `f00de7548e`, `4324aa0125`, `d5115b2da8` |
| MiniMax HPC | Complete, image/GPU pending | `6120c51141`, `deb0b32e0d` |
| CUTLASS W4A8 and DeepEP | Complete, GPU/multi-rank pending | `099caa830f` through `90b13c90e1` |
| Humming W4A8 | Complete, GPU pending | `19d96bcbfa` |
| ByteIAAS build/release | Implemented; DeepGEMM pin fix and CI execution pending | `f8040e5a16` through `3b27f22d7c` |
| Step-level prefill token buckets | Complete, opt-in | `a8fd87c724` through `f1d5ef2237` |
| MRV2 direct-Mooncake PCP | Hold | No safe target-native implementation yet |
| DSpark K below block size | Hold | `0d6fd1c83c` intentionally not migrated |
| MiniMax indexer E5M2 removal | Hold | Existing top-k layout is upstream; dtype removal lacks evidence |

## 4. Changes Already Covered Upstream

### 4.1 Upstream release commits

The following commits are not IAAS features and must not be replayed:

```text
a54c93a146  WSL pin_memory circular-import fix
2dd1e7cd3b  BGE-M3 token expectation update
9d37a50c80  Empty MLA context merge fix
8b30569e83  DeepGEMM warmup fix
e5949f1000  Grammar compilation failure handling
ba694b86f2  Buildkite timeout
091db8b58f  Buildkite timeout
bb26ce8e93  Buildkite timeout
ffd6ee4bcc  Buildkite timeout
ffd46bfab2  AXK1 registration
f2654939e6  ROCm release pipeline fix
568afb3a13  macOS wheel tag refresh (`v0.26.0`)
```

Eleven are patch-equivalent to commits already reachable from `v0.27.0`.
`e5949f1000` is semantically covered by upstream `12213c6795` for PR `#47312`,
despite having a different patch ID.

### 4.2 Partial upstream equivalents

These are the main cases where an IAAS commit cannot be handled as one unit:

- `3c7ed3742c`: its per-KV-group concurrency calculation is already upstream as
  `2e2e626b40` (`#48317`). Its other DSV4 correctness changes are not covered.
- `8b291ae6f3`: token-major MiniMax top-k buffer handling is already upstream as
  `d1a8ba63d9` (`#49149`). The IAAS removal of unsupported `fp8_e5m2` from the
  indexer capability list is still absent.
- `87ec6ecd11`: upstream `b6ff8a2f50` (`#46570`) and `60417b4b74` (`#50034`)
  replace the old PCP execution model with MRV2 virtual-batch PCP. The legacy
  IAAS PCP implementation must not overwrite it.
- `f4a2217461` and `20bce687d4`: upstream already declares PCP support for the
  generic MLA indexer, but not all DSV4 hybrid cache backends. Port only the
  declarations and adapter behavior that remain necessary after MRV2 tests.
- `d623aef9c0`: upstream `0d12618e` and `ecf4aa5c` cover the non-causal
  FlashInfer graph fixes intentionally excluded by the IAAS commit. The
  remaining correctness fixes are still candidates.

## 5. Feature Migration Design

### 5.1 ByteIAAS Build and Release

Retain the private workflows and release behavior, but rebase them onto the
current `v0.27.0` Docker and DeepGEMM build layout.

Migration rules:

- Add the ByteIAAS workflow files and CI helper scripts without restoring
  deleted or renamed upstream workflow infrastructure.
- Rework `docker/Dockerfile` by stage and argument rather than replacing it
  with the `v0.26.0` file.
- Preserve the `v0.27.0` vendored DeepGEMM build model and use its existing
  exact source/ref and required-symbol validation. Do not install the old
  external CPython-specific DeepGEMM wheel.
- Build HPC-Ops from its exact commit in an isolated stage, install it with
  `--no-deps`, and statically verify the ABI3 binary and required APIs.
- Reuse the `humming-kernels[cu13]==0.1.10` target dependency; do not reinstall
  Humming in the devel image.
- Resolve requested refs to one immutable SHA in `iaas_main` history before
  launching wheel and image jobs. For pre-merge validation, permit only a
  branch-based manual dispatch to build the exact `github.sha` carrying that
  workflow; tag dispatches remain subject to the ancestry check. Require the
  versioned ByteIAAS build contract and limit registry credentials to the
  login step.
- Build CUDA 13.0.3 OpenAI/devel images by digest. Build zstd devel from the
  zstd OpenAI digest, and verify all relevant child manifests before final
  tags are created.
- Convert nydus images through unique staging tags, validate them, and publish
  the final tags from the verified immutable digests.
- Fold the apt-mirror experiments into the final Volces mirror state.

### 5.2 DSV4 Correctness and IndexCache

Split `3c7ed3742c` into target-native patches:

1. Port compressed slot-mapping bounds checks and padded-row protection.
2. Port MTP-layer `compress_ratios` handling and unscaled RoPE selection.
3. Port FP8 expert dtype inference from the selected safetensors index.
4. Port the BF16 `wo_a` fallback and robust `weight_scale` selection.
5. Drop the already-upstream KV-cache concurrency hunk.

The IndexCache stack from `39d86ed8c7` is not present upstream. It should be
ported manually because `v0.27.0` introduced compact MXFP4 indexer caches and
packed group overlays in `f3a920a076`. Required invariants:

- Skip decisions apply only to intended C4 layers.
- PP stages compute local producer/consumer ownership correctly.
- Omitted indexer/compressor cache specs do not corrupt group accounting.
- Alias metadata remains precise enough for Mooncake transfer matching.
- Unsupported platform and ubatching combinations fail closed.

The indexer safety chain `657132e0aa` -> `3bc91bb763` -> `d62244f050` ->
`6ca83dd045` should be ported as one logical series. It protects dummy decode
rows, stable CUDA graph addresses, capture metadata, and eager full-graph
warmup.

### 5.3 SM90 MegaMoE

`v0.27.0` has DeepSeek V4 MegaMoE but does not contain the IAAS SM90 FP4/FP8
path or its fork-only DeepGEMM APIs.

Port `3e60ad63d3` as two commits:

1. Runtime support: SM90 weight transforms, staging, dispatch, DP-wide
   capacity, empty-rank collective participation, and EPLB tensor exposure.
2. Build contract: configurable DeepGEMM repository/ref, exact checkout
   validation, required-symbol validation, and matching image arguments.

Do not replace the target's DeepGEMM CMake file. The target vendors per-Python
bindings and installs the `mega/` package; the private override must preserve
that behavior.

### 5.4 Dynamic Speculative Decoding and DSpark/DFlash

Use upstream Dynamic SD as the base:

- `4ef4492e9b`: base Dynamic SD.
- `07516fda67`: MRV2 full-CUDA-graph support.
- `93e2ab7111`: fail-closed disabling under DP.

Then port the IAAS extensions in this order:

1. `d623aef9c0` residual correctness fixes.
2. `83c16e1cb6` early K selection, padding policy, and token-budget reclaim.
3. `bb946a7ce7` runtime K propagation through executor, dummy runs, target, and
   proposer.
4. `67a597c869` opt-in DP-global pressure synchronization. This must replace,
   not bypass accidentally, upstream `93e2ab7111`.
5. `7a6f9efb6f` CUDA graph family dispatch by query length.
6. `09cede6391` and `77ed1eece5` full-K warmup and cleanup.
7. `aa8e0a91b8` plus `9016ee5bb6` dummy-batch K contract.

Port the remaining correctness patches around this stack:

- Async draft-group bounds and invalid-token trimming.
- Draft KV dtype ownership, including dense-draft fallback from
  `fp8_ds_mla`.
- Producer-only P/D workers must not instantiate or execute a drafter.
- DFlash prefix-cache masking for target-only restored prefixes.
- DFlash/DSpark draft MoE backend override.
- Idle-DP padded draft rows must remain padding through MegaMoE.
- DeepGEMM TMA-aligned activation scales.

`0d6fd1c83c` is a deliberate reversal of an upstream safety check:
`v0.27.0` rejects `DSpark K < dspark_block_size`, while IAAS removes the
rejection. Keep this commit on hold until GPU correctness and acceptance-rate
tests prove that every supported DSpark checkpoint handles reduced K.

### 5.5 Mooncake and KV Transfer

Build on upstream PP support `d53f4593ce` and heterogeneous-TP support
`b745e8b5d3`; neither fully covers the IAAS extensions.

Port in this order:

1. Shared alias/group transfer identity (`cfcba9469a`, `95188da984`).
2. Standard-cache backend selection (`9c8a93450d`).
3. Heterogeneous TP extension and stale-completion handling (`88e2f20ed2`).
4. Unsupported PP fanout rejection (`07fbbb0bdb`).
5. Optional node-shared long-context send gate and diagnostics
   (`f20c7435de`).

The `88e2f20ed2` commit must be split. Upstream already handles simple TP-ratio
transfers, but still lacks IAAS semantics for:

- Replicated GQA heads when `TP > Hkv`.
- Per-region KV-head inference.
- Fully replicated regions.
- Stale send/receive completions after request abort.

The PCP-related portions of `87ec6ecd11` need a separate decision. Upstream
Mooncake Store has PCP-aware namespacing, but the direct `MooncakeConnector`
does not expose the same producer PCP fan-in used by IAAS. Port that part only
if the production topology uses direct Mooncake P/D with PCP.

### 5.6 MiniMax-M3 MoE

Upstream `v0.27.0` already has:

- MiniMax-M3 model support.
- Generic HPC FP8 MoE support.
- Humming support in several other MoE oracles.
- The token-major top-k fix `d1a8ba63d9`.

It does not have the IAAS MiniMax-specific HPC or W4A8 stacks.

Recommended sub-stacks:

1. HPC activation clamp: `f37b865341`.
2. MiniMax HPC MXFP8 K32: `57fed50368`, `25e140853f`,
   `4799bb3106`, `55b2e207d1`, `4b730aef58`.
3. CUTLASS W4A8 SwiGLU: `8bef682e2d`, `5188517c82`,
   `df0273759e`.
4. DeepEP LL/HT: `b283f4b26a`, `2bbc5b51a9`, `6226ac4e52`,
   `0f233eeec3`.
5. Humming W4A8: `031fc9a691`; drop `2bee42cda6` because `v0.27.0` already
   pins Humming 0.1.10 in `requirements/cuda.txt`.

For `8b291ae6f3`, drop the already-upstream top-k layout changes and separately
decide whether to retain the removal of `fp8_e5m2` from the advertised indexer
cache dtypes.

### 5.7 Small Independent Fixes

- `027887b86b`: port the `VLLM_USE_CUTEDSL_LL_BF16` kill switch. Upstream
  `96fa3f42c9` only avoids warming the kernel for non-MoE models; it does not
  provide an import/dispatch kill switch.
- `f56547192d`: port. PR `#48303` is not present in `v0.27.0`; the target still
  hardcodes GPT-OSS SwiGLU parameters in `FlashInferExperts` and lacks the
  DeepSeek/GLM/MiMo weight conversion branch.

## 6. Per-Commit Disposition

### 6.1 Build and Release

| Old commit | Disposition | Result |
| --- | --- | --- |
| `a9f1617cc7` | SPLIT | Helpers/workflows/Docker rebuilt as `f8040e5a16`, `7899a9fa28`, and `dfd6f57723` |
| `4edbfab0a9` | SQUASH | Runner metadata folded into `dfd6f57723` |
| `23dfea6fca` | SPLIT | HPC checks in `7899a9fa28`; rustup retry in `2fd47e7683` |
| `ec58aaf9f3` | SQUASH | Final HPC source ref folded into `7899a9fa28` |
| `30cf9d9bad` | SQUASH | Final HPC API validation folded into `7899a9fa28` |
| `9493f04276` | SPLIT | Tag helpers in `f8040e5a16`; publishing in `9d72a701d3` |
| `aa4fd55c54` | SQUASH | Reworked in `f8040e5a16`, `16d786bc7b`, and `9d72a701d3` |
| `348c70ca0b` | DROP | Intermediate mirror state omitted |
| `0b37e9134c` | DROP | Intermediate DNS workaround omitted |
| `09aa3c1bde` | DROP | Intermediate host-mapping workaround omitted |
| `18f1129660` | DROP | Intermediate mirror workaround omitted |
| `eb454b2d43` | PORT | Final Volces mirror state in `7899a9fa28` |
| `2bee42cda6` | DROP | Humming 0.1.10 is already pinned by upstream `v0.27.0` |

### 6.2 DSV4

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `3c7ed3742c` | SPLIT | Drop concurrency hunk; port RoPE, FP8 metadata, BF16 O-proj, and metadata bounds |
| `39d86ed8c7` | PORT | Manual IndexCache port onto compact/packed `v0.27.0` cache layout |
| `3e60ad63d3` | SPLIT | Separate runtime SM90 MegaMoE from DeepGEMM build contract |
| `657132e0aa` | PORT | Port dummy decode row guard |
| `3bc91bb763` | SQUASH | Fold stable seq-lens buffer fix into indexer safety commit |
| `d62244f050` | SQUASH | Fold capture-specific seq-lens handling into indexer safety commit |
| `6ca83dd045` | PORT | Port capture metadata use for eager full warmup |
| `87ec6ecd11` | SPLIT | Drop legacy PCP core; retain only proven DSV4/Mooncake gaps |
| `f4a2217461` | SPLIT | Re-evaluate hybrid backend declarations against MRV2 |
| `20bce687d4` | SPLIT | Re-evaluate sparse MLA declaration against MRV2 |

### 6.3 Dynamic SD and Drafter Correctness

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `d623aef9c0` | SPLIT | Port residual correctness fixes; do not duplicate upstream non-causal fixes |
| `83c16e1cb6` | PORT | Rebase scheduler policy and budget reclaim onto current scheduler |
| `bb946a7ce7` | PORT | Propagate runtime K through MRV2 executor, dummy, target, and proposer paths |
| `67a597c869` | PORT | Add opt-in DP-global K policy and preserve fail-closed default |
| `7a6f9efb6f` | PORT | Current patch applies cleanly; validate graph families after runtime-K port |
| `09cede6391` | PORT | Add full-K DFlash/DSpark proposer warmup |
| `77ed1eece5` | SQUASH | Keep warmup cleanup/reset with the preceding commit |
| `23b6150467` | PORT | Preserve padding masks for idle DP ranks and MegaMoE collectives |
| `da0e8131f2` | SQUASH | Fold DFlash MoE override into a shared draft-kernel-config patch |
| `129e53cf71` | DROP | Style-only test normalization |
| `1f00d2a3bf` | SQUASH | Keep passed draft config behavior with the backend override |
| `aa8e0a91b8` | PORT | Always pass scheduler-selected K to dummy execution |
| `9016ee5bb6` | SQUASH | Keep as regression coverage for `aa8e0a91b8` |
| `55bfba2920` | PORT | Current patch applies cleanly; preserve TMA-aligned scales |
| `0d6fd1c83c` | HOLD | Conflicts with upstream correctness guard; require model evidence |
| `9dd89bf957` | SQUASH | Finalize shared DFlash/DSpark backend override implementation |

### 6.4 Mooncake and KV Transfer

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `cfcba9469a` | PORT | Current patch applies cleanly; add shared alias/group identity |
| `95188da984` | SQUASH | Keep regression coverage with alias support |
| `f20c7435de` | PORT | Add opt-in node-shared long-request gate and trace lifecycle |
| `88e2f20ed2` | SPLIT | Port stale completions and GQA/replicated-region extensions only |
| `9c8a93450d` | PORT | Current patch applies cleanly; skip nonstandard indexer backend |
| `07fbbb0bdb` | PORT | Add fail-closed producer/consumer PP fanout validation |

### 6.5 MoE and MiniMax

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `f37b865341` | PORT | Current patch applies cleanly; add HPC blockwise clamp contract |
| `f56547192d` | PORT | Current patch applies cleanly; missing upstream PR `#48303` behavior |
| `57fed50368` | PORT | Current patch applies cleanly; base MiniMax HPC MXFP8 backend |
| `25e140853f` | SQUASH | Keep strict capability/shape gating with the HPC backend |
| `4799bb3106` | SQUASH | Keep fused BF16 candidate path |
| `55b2e207d1` | SQUASH | Keep preallocated workspaces and output alias contract |
| `4b730aef58` | SQUASH | Disable unsupported chunking in the MiniMax HPC backend |
| `8b291ae6f3` | SPLIT | Drop upstream top-k fix; decide separately on `fp8_e5m2` |
| `8bef682e2d` | PORT | Current patch applies cleanly; propagate W4A8 SwiGLU parameters |
| `5188517c82` | PORT | Current patch applies cleanly; add MiniMax CUTLASS W4A8 activation |
| `df0273759e` | SQUASH | Keep tests with the preceding W4A8 commits |
| `b283f4b26a` | PORT | Manually adapt batched W4A8 to current DeepEP LL interfaces |
| `2bbc5b51a9` | PORT | Retain useful-token and DP-total scheduling |
| `6226ac4e52` | PORT | Manually adapt W4A8 to current DeepEP HT and graph rules |
| `176c597e1e` | DROP | Formatting-only; apply formatting to the rewritten code |
| `0f233eeec3` | PORT | Retain guarded MiniMax long-prefill schedule |
| `031fc9a691` | PORT | Add W4A8 to the target's standardized Humming integration |

### 6.6 Runtime Switch

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `027887b86b` | PORT | Add the CuTe DSL LL BF16 kill switch and tests |

### 6.7 Step-Level Prefill Token Buckets

| Old commit | Disposition | Migration note |
| --- | --- | --- |
| `7f716ff4c5` | SPLIT | Port config, CLI, scheduler behavior, and focused tests only |
| `efbe85cff4` | DROP | Metrics-only gauge fix; observability is outside the approved scope |
| `494b0ecf05` | PORT | Reimplement final whole-step cap and same-bucket semantics |

The target-native implementation also preserves dynamic-SD budgets, DP
prefill cadence, priority-preemption rollback, async KV resume, and Mamba
block-aligned progress. Prometheus metrics, PP queue telemetry, step latency,
NVTX, trace, and debug helpers are intentionally not migrated.

### 6.8 Merge Commits

Drop these merge commits. Their non-merge children are classified above:

```text
d86d2d03cd  Merge MiniMax-M3 HPC
f292c28cb6  Merge DSV4 PCP refresh
71708e0b1c  Merge MiniMax W4A8
0bba7b0a64  Merge zstd/nydus
e947b67b83  Merge W4A8 optimization
ad70d89158  Merge apt mirror fixes
de55af9138  Merge Humming W4A8
8da5f70fca  Merge step-level prefill token buckets
```

## 7. Implemented Source-to-Target Mapping

The migration uses a linear target-native history. Multiple source commits are
intentionally folded where later source commits only harden or test the same
behavior.

| Source commit(s) | Target commit(s) | State |
| --- | --- | --- |
| `55bfba2920` | `aa5a75d799` | Ported |
| `9c8a93450d` | `cca9a49550` | Ported |
| `f37b865341` | `6120c51141` | Ported |
| `027887b86b` | `f24fd3012b` | Ported |
| `f56547192d` | `5a20fd1389` | Ported |
| `3c7ed3742c`, `657132e0aa`, `3bc91bb763`, `d62244f050`, `6ca83dd045` | `549942052e` | Ported/squashed |
| `39d86ed8c7` | `effe7093cf` | Ported |
| `3e60ad63d3` | `78dea35eb9`, `b84298dbe1` | Split into runtime/build |
| `d623aef9c0` | `86ec283e47` | Ported without upstream-covered hunks |
| `83c16e1cb6` | `9e006972e2` | Ported |
| `bb946a7ce7` | `33176a5408` | Ported |
| `67a597c869`, `aa8e0a91b8`, `9016ee5bb6` | `aecdda1e0d`, `b5c95a75f7` | Ported and hardened |
| `7a6f9efb6f` | `d709a41c46` | Ported |
| `09cede6391`, `77ed1eece5` | `d00896ae74` | Squashed |
| `23b6150467` | `0c142e5b8e` | Ported |
| `da0e8131f2`, `1f00d2a3bf`, `9dd89bf957` | `66bf65058d` | Squashed |
| `129e53cf71` | None | Dropped as formatting-only |
| `0d6fd1c83c` | None | Hold pending reduced-K GPU evidence |
| `cfcba9469a`, `95188da984` | `5b701d086f` | Squashed |
| `88e2f20ed2` | `f00de7548e` | Partial port |
| `07fbbb0bdb` | `4324aa0125` | Ported |
| `f20c7435de` | `d5115b2da8` | Ported |
| `57fed50368`, `25e140853f`, `4799bb3106`, `55b2e207d1`, `4b730aef58` | `deb0b32e0d` | Squashed |
| `8b291ae6f3` | Upstream `d1a8ba63d9`; no target commit | Top-k covered; E5M2 removal held |
| `8bef682e2d`, `5188517c82`, `df0273759e` | `099caa830f` | Squashed |
| `b283f4b26a` | `29a60ddd0a` | Ported |
| `2bbc5b51a9` | `4779c1117c` | Ported |
| `6226ac4e52` | `d39f8de0c0` | Ported |
| `0f233eeec3` | `7d07b7b5b3` | Ported |
| `176c597e1e` | None | Dropped as formatting-only |
| `031fc9a691` | `19d96bcbfa` | Ported and hardened |
| `87ec6ecd11` | None | Legacy PCP core dropped; direct Mooncake PCP held |
| `f4a2217461`, `20bce687d4` | None | Hold pending MRV2 adapter and parity work |
| `a9f1617cc7`, `4edbfab0a9`, `23dfea6fca`, `ec58aaf9f3`, `30cf9d9bad`, `9493f04276`, `aa4fd55c54`, `eb454b2d43` | `f8040e5a16` through `f798a87ca3`, plus `2fd47e7683` | Rebuilt on v0.27.0 |
| `348c70ca0b`, `0b37e9134c`, `09aa3c1bde`, `18f1129660` | None | Intermediate mirror workarounds dropped |
| `2bee42cda6` | Upstream `requirements/cuda.txt` | Already covered |
| `7f716ff4c5`, `494b0ecf05` | `a8fd87c724` through `f1d5ef2237` | Core scheduler semantics ported and hardened |
| `efbe85cff4` | None | Dropped as metrics-only by scope |
| `8da5f70fca` | None | Dropped as merge-only |

Target-only hardening commits such as `e3f14d2e48`, `90b13c90e1`,
`16d786bc7b`, `7bac341c97`, `ac36a3ee3b`, `f798a87ca3`, `90aecd5ad3`, and
`4a1bd81125`, `138e061e38`, and `f911556cb6` close review findings discovered
during the migration rather than correspond to one source commit.

## 8. Validation Matrix

### Static and CPU tests

```bash
python -m pytest -q \
  tests/config/test_dspark_dflash_validation.py \
  tests/v1/spec_decode/test_dynamic_sd.py \
  tests/v1/spec_decode/test_dynamic_sd_cug.py \
  tests/v1/engine/test_dynamic_sd_dp_sync.py \
  tests/v1/kv_connector/unit/test_mooncake_connector.py \
  tests/v1/kv_connector/unit/test_transfer_topology_sharded.py

python -m pytest -q \
  tests/models/test_deepseek_v4_index_cache.py \
  tests/models/deepseek_v4/test_nvidia_o_proj.py \
  tests/models/deepseek_v4/test_rope.py \
  tests/transformers_utils/test_dsv4_config.py \
  tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py

python -m pytest -q \
  tests/kernels/moe/test_hpc_fp8_backend.py \
  tests/kernels/moe/test_hpc_moe.py \
  tests/kernels/moe/test_cutlass_moe.py \
  tests/kernels/moe/test_w4a8_humming.py \
  tests/kernels/quantization/test_cutlass_w4a8_moe.py

python3 -m unittest \
  scripts.ci.test_get_byteiaas_image_tag \
  scripts.ci.test_verify_byteiaas_image_format

actionlint \
  .github/workflows/_byteiaas-build-wheel.yml \
  .github/workflows/_byteiaas-build-and-publish-image.yml \
  .github/workflows/byteiaas-release-dev.yml \
  .github/workflows/byteiaas-release.yml

python tools/generate_versions_json.py --check
bash -n build_rust.sh
```

Run `ruff` and `mypy` for touched runtime modules before GPU validation.

### GPU and distributed validation

| Area | Required coverage |
| --- | --- |
| DSV4 correctness | BF16/FP8 experts, C4 compressed layers, zero/short lengths, eager and full graph |
| IndexCache | enabled/disabled, frequency/pattern modes, PP-local layers, memory accounting |
| SM90 MegaMoE | FP4 and FP8, TP/EP, idle rank, EPLB, exact fork API validation |
| Dynamic SD | K=`0/3/5/7`, C1/C2/C4/saturation, async on/off, eager/piecewise/full graph |
| DP Dynamic SD | DP2 and DP4, uneven local load, sync interval 1 and 8, no collective divergence |
| DSpark reduced K | bitwise or tolerance-matched output, acceptance rate, no garbled tokens |
| Mooncake | TP1->8, TP8->1, TP4->2, replicated `Hkv<TP`, PP mismatch rejection, abort race |
| PCP | PCP2/4 on MRV2, hybrid SWA+C4, prefix cache, direct Mooncake P/D if used |
| MiniMax HPC | H20/SM90 MXFP8 K32, workspace reuse, no chunking, numerical reference |
| MiniMax W4A8 | CUTLASS standard, DeepEP LL, DeepEP HT, Humming, graph replay |
| Images | wheel import, HPC/Humming symbol probes, zstd and nydus child manifests |

For performance-sensitive paths, compare against matched `v0.27.0` controls:

- TTFT mean at C1/C2/C4/saturation.
- TPOT and end-to-end throughput.
- Dynamic SD acceptance rate and selected K per step.
- NVTX stage overlap and step latency for communication-heavy topologies.

### Validation Completed During Migration

- Focused CPU tests passed for DSV4, Dynamic SD, Mooncake, HPC, CUTLASS W4A8,
  DeepEP LL/HT, and Humming selection/shape contracts.
- The final Humming W4A8 review approved `19d96bcbfa`.
- The final CUTLASS W4A8/DeepEP review approved
  `099caa830f^..90b13c90e1`.
- The final ByteIAAS source review approved `f8040e5a16^..f911556cb6`.
- The branch self-dispatch hardening through `3b27f22d7c` passed independent
  review with no P0-P2 findings.
- ByteIAAS helper tests pass with 18 cases, including mixed child/layer,
  attestation, shared-DAG, cycle, tag-length, and ASCII validation.
- Actionlint 1.7.7 passes all ByteIAAS workflows.
- The source-selection matrix accepts an exact branch self-dispatch and rejects
  tag self-dispatch, a different feature-branch SHA, non-dispatch feature
  sources, and sources without build contract `1`.
- Both Dockerfiles parse and `docker/versions.json` matches
  `docker/Dockerfile`.
- Buildx 0.36.1 frontend validation reports zero warnings for the ByteIAAS
  devel Dockerfile. The main Dockerfile reports only the same two
  `SCCACHE_S3_NO_CREDENTIALS` false positives and legacy `ENV` warning present
  before the ByteIAAS changes.
- The existing v0.26 MiniMax-M3 Humming P/D control deployment is healthy on
  two 8xH20 nodes and passes a minimal completion request with zero pod
  restarts. This establishes the environment baseline but does not validate
  the v0.27 branch.
- A complete v0.27 wheel/image build and registry publication have not yet
  been triggered.
- Step-level prefill token buckets passed 21 focused scheduler cases, one CLI
  parsing case, two Mamba budget/alignment regressions, two remote-KV cadence
  cases, Ruff 0.15.12, `compileall`, and `git diff --check`.
- The complete source disposition is in
  `docs/contributing/iaas_v0_27_0_source_commit_disposition.md`.
- The post-migration audit is in
  `docs/contributing/iaas_v0_27_0_target_commit_audit.md`. It found two P1
  issues (DSpark Dynamic SD reduced-K and ByteIAAS DeepGEMM/Kimi ABI) plus one
  P2 custom-scheduler compatibility issue.

## 9. High-Risk Decisions Before Execution

1. **DSpark K below block size**: `0d6fd1c83c` remains intentionally absent,
   but Dynamic SD schedule values currently bypass the equivalent static guard.
   Reject every non-zero scheduled K below `dspark_block_size`.
2. **PCP implementation**: use upstream MRV2 PCP. Porting
   `vllm/models/deepseek_v4/pcp_metadata.py` wholesale would regress the target
   architecture.
3. **Mooncake PCP path**: confirm whether production uses direct
   `MooncakeConnector` or `MooncakeStoreConnector`; their PCP support differs.
4. **DeepGEMM source**: pin the exact fork commit and validate required symbols
   in both wheel and image builds.
5. **MiniMax indexer FP8 E5M2**: decide whether it is unsupported by contract or
   merely untested before carrying the capability removal.

## 10. Completion Criteria

| Criterion | Status |
| --- | --- |
| Every one of the 66 non-merge IAAS commits has a disposition | Complete |
| No upstream-equivalent or merge-only commit is replayed | Complete |
| Retained code is rebased onto `v0.27.0` APIs | Complete |
| Focused CPU/static tests pass | Complete for implemented stacks |
| Final source-to-target review has no unexplained behavior loss | Request changes: 2 P1, 1 P2 |
| GPU/distributed validation covers affected topology | Pending |
| Built images prove DeepGEMM, HPC-Ops, Humming, zstd, and nydus contracts | Pending |

The branch is not yet code-complete after the post-migration audit. In
addition to the pending GPU, multi-rank, RDMA, image publication checks, the
two P1 and one P2 findings above must be resolved. The intentionally absent
behaviors remain:

- Direct support for DSpark speculative K below its block size.
- Direct Mooncake P/D with MRV2 PCP fan-in.
- MiniMax indexer `fp8_e5m2` capability removal.
