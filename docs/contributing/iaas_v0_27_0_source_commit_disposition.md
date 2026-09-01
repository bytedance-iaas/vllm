# IAAS v0.27.0 Source Commit Disposition

## 1. Purpose and Scope

This document accounts for every IAAS commit after upstream `v0.26.0` and
explains how it was handled while reconstructing `iaas_main` on top of
upstream `v0.27.0`.

| Item | Revision |
| --- | --- |
| Source range | `568afb3a13..origin/iaas_main` |
| Source head | `8da5f70fca26827d8542722f6c8663f11bf2b32c` |
| Target baseline | upstream `v0.27.0` (`4bdc8a788d`) |
| Target branch | `wyc/iaas-v0.27.0` |
| Target implementation head | `f1d5ef2237` |

The source range contains 66 non-merge commits and 8 merge commits. All 74
commits are listed below.

## 2. Disposition Terms

- **PATCH-EQUIVALENT**: the source patch is present unchanged by patch-id,
  although the target commit hash differs.
- **MODIFIED**: required behavior was retained through a target-native port,
  split, squash, or reimplementation against the v0.27 architecture.
- **DROP**: the commit is upstream-covered, superseded, intermediate,
  formatting-only, metrics-only outside the approved scope, or merge-only.
- **HOLD**: intentionally not migrated because correctness or architecture
  evidence is insufficient.
- **DROP / HOLD**: upstream-covered or obsolete portions were dropped while a
  remaining unproven semantic was held.

`git cherry HEAD origin/iaas_main` identifies only `55bfba2920` and
`0f233eeec3` as patch-equivalent among the IAAS feature commits. Other retained
features were adapted and must not be described as simple cherry-picks.

## 3. Summary

| Disposition | Non-merge commits |
| --- | ---: |
| PATCH-EQUIVALENT | 2 |
| MODIFIED | 51 |
| DROP | 8 |
| HOLD | 3 |
| DROP / HOLD | 2 |
| **Total** | **66** |

All 8 merge commits are dropped because their non-merge children are accounted
for separately.

## 4. Build and Release

The old v0.26 workflows and Docker layout were not replayed directly. They
were rebuilt around the v0.27 Docker stages, vendored DeepGEMM layout,
HPC-Ops contract, immutable publication inputs, and scoped credentials.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `a9f1617cc7` | MODIFIED | Split across `f8040e5a16`, `7899a9fa28`, and `dfd6f57723`; helpers, Docker stages, and workflows were rebuilt for v0.27. |
| `4edbfab0a9` | MODIFIED | Actionlint/runner metadata folded into `dfd6f57723`. |
| `23dfea6fca` | MODIFIED | HPC image checks moved to `7899a9fa28`; rustup retry became `2fd47e7683`. |
| `ec58aaf9f3` | MODIFIED | Final HPC-Ops source and image integration folded into `7899a9fa28`. |
| `30cf9d9bad` | MODIFIED | HPC-Ops wheel/API validation folded into `7899a9fa28`. |
| `9493f04276` | MODIFIED | Tag logic moved to `f8040e5a16`; verified zstd/nydus publication became `9d72a701d3`. |
| `aa4fd55c54` | MODIFIED | Manifest inspection was redesigned in `f8040e5a16`, `16d786bc7b`, and `9d72a701d3`. |
| `348c70ca0b` | DROP | Intermediate apt-mirror state; superseded by the final Volces mirror configuration. |
| `0b37e9134c` | DROP | Intermediate DNS workaround; not part of the final desired state. |
| `09aa3c1bde` | DROP | Intermediate host-mapping workaround; superseded. |
| `18f1129660` | DROP | Intermediate mirror-access workaround; superseded. |
| `eb454b2d43` | MODIFIED | Final Volces Ubuntu mirror state retained in `7899a9fa28`. |
| `2bee42cda6` | DROP | `humming-kernels[cu13]==0.1.10` is already pinned by upstream v0.27. |

Target-only hardening was added after the source behavior was reconstructed:
`7bac341c97`, `ac36a3ee3b`, `f798a87ca3`, `90aecd5ad3`,
`4a1bd81125`, `138e061e38`, `f911556cb6`, `445cbf2b2c`, and
`3b27f22d7c`.

## 5. DSV4, IndexCache, and SM90 MegaMoE

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `3c7ed3742c` | MODIFIED | Missing RoPE, FP8 metadata, BF16 O-proj, and metadata-bound semantics moved to `549942052e`; the already-upstream concurrency hunk was dropped. |
| `39d86ed8c7` | MODIFIED | IndexCache was manually redesigned for compact/packed v0.27 cache layouts in `effe7093cf`. |
| `3e60ad63d3` | MODIFIED | Split into runtime `78dea35eb9` and validated DeepGEMM build contract `b84298dbe1`. |
| `657132e0aa` | MODIFIED | Dummy decode row guard folded into `549942052e`. |
| `3bc91bb763` | MODIFIED | Stable paged-MQA sequence-length storage folded into `549942052e`. |
| `d62244f050` | MODIFIED | CUDA-graph capture sequence-length semantics folded into `549942052e`. |
| `6ca83dd045` | MODIFIED | Capture metadata/eager warmup compatibility folded into `549942052e`. |
| `87ec6ecd11` | DROP / HOLD | Legacy PCP core was dropped because v0.27 uses MRV2 virtual-batch PCP; direct Mooncake PCP gaps remain held. |
| `f4a2217461` | HOLD | Hybrid-backend PCP declarations await target-native adapters and GPU parity. |
| `20bce687d4` | HOLD | Sparse-MLA PCP declaration awaits MRV2 adapter and parity evidence. |

## 6. Dynamic SD and Draft Correctness

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `d623aef9c0` | MODIFIED | Residual DFlash/DSpark correctness semantics became `86ec283e47`; upstream-covered non-causal fixes were omitted. |
| `83c16e1cb6` | MODIFIED | Scheduler policy and budget reclaim were rebuilt on the v0.27 scheduler as `9e006972e2`. |
| `bb946a7ce7` | MODIFIED | Runtime-K propagation across MRV2 executor, target, proposer, and dummy paths became `33176a5408`. |
| `67a597c869` | MODIFIED | Opt-in DP-global pressure became `aecdda1e0d` and was hardened by `b5c95a75f7`. |
| `7a6f9efb6f` | MODIFIED | Query-length CUDA-graph family dispatch became `d709a41c46`. |
| `09cede6391` | MODIFIED | Full-K proposer warmup folded into `d00896ae74`. |
| `77ed1eece5` | MODIFIED | Warmup cleanup/reset folded into `d00896ae74`. |
| `23b6150467` | MODIFIED | Idle-DP draft padding semantics became `0c142e5b8e`. |
| `da0e8131f2` | MODIFIED | DFlash backend override folded into `66bf65058d`. |
| `129e53cf71` | DROP | Test formatting/style only. |
| `1f00d2a3bf` | MODIFIED | Passed draft config behavior folded into `66bf65058d`. |
| `aa8e0a91b8` | MODIFIED | Explicit scheduler-selected K folded into `aecdda1e0d`/`b5c95a75f7`. |
| `9016ee5bb6` | MODIFIED | K=0 regression coverage folded into `aecdda1e0d`/`b5c95a75f7`. |
| `55bfba2920` | PATCH-EQUIVALENT | Retained as `aa5a75d799`; DeepGEMM TMA-aligned activation-scale behavior is patch-equivalent. |
| `0d6fd1c83c` | HOLD | Removing the upstream `K < block_size` guard lacks model-specific correctness and acceptance-rate evidence. |
| `9dd89bf957` | MODIFIED | Final shared DFlash/DSpark backend override folded into `66bf65058d`. |

## 7. Mooncake and KV Transfer

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `cfcba9469a` | MODIFIED | Shared alias/group transfer identity was rebuilt as `5b701d086f`. |
| `95188da984` | MODIFIED | Alias regression coverage folded into `5b701d086f`. |
| `f20c7435de` | MODIFIED | Node-shared long-context admission and generation-scoped lifecycle became `d5115b2da8`. |
| `88e2f20ed2` | MODIFIED | Only stale completion, GQA replica, and fully replicated-region semantics were ported in `f00de7548e`; upstream simple TP-ratio logic was retained. |
| `9c8a93450d` | MODIFIED | Standard-cache backend selection became `cca9a49550`; nonstandard indexer selection was intentionally omitted. |
| `07fbbb0bdb` | MODIFIED | Unsupported producer/consumer PP fanout now fails closed in `4324aa0125`. |

## 8. MoE, MiniMax, and Humming

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `f37b865341` | MODIFIED | HPC blockwise activation-clamp contract became `6120c51141`. |
| `f56547192d` | MODIFIED | Missing FlashInfer CUTLASS MXFP4 wiring was adapted as `5a20fd1389`. |
| `57fed50368` | MODIFIED | Base MiniMax-M3 HPC MXFP8 path folded into `deb0b32e0d`. |
| `25e140853f` | MODIFIED | Capability/shape gates folded into `deb0b32e0d`. |
| `4799bb3106` | MODIFIED | Fused BF16 candidate path folded into `deb0b32e0d`. |
| `55b2e207d1` | MODIFIED | Workspace reuse and output alias contract folded into `deb0b32e0d`. |
| `4b730aef58` | MODIFIED | Unsupported MiniMax HPC chunking is disabled in `deb0b32e0d`. |
| `8b291ae6f3` | DROP / HOLD | Token-major top-k layout is upstream as `d1a8ba63d9`; removing `fp8_e5m2` remains held for evidence. |
| `8bef682e2d` | MODIFIED | W4A8 SwiGLU parameter propagation folded into `099caa830f`. |
| `5188517c82` | MODIFIED | MiniMax CUTLASS W4A8 activation folded into `099caa830f`. |
| `df0273759e` | MODIFIED | CUTLASS W4A8 contract tests folded into `099caa830f`. |
| `b283f4b26a` | MODIFIED | Batched W4A8 was adapted to current DeepEP LL and stable ABI in `29a60ddd0a`. |
| `2bbc5b51a9` | MODIFIED | Useful-row/compact scheduling became `4779c1117c`. |
| `6226ac4e52` | MODIFIED | W4A8 was adapted to current DeepEP HT and graph policy in `d39f8de0c0`. |
| `176c597e1e` | DROP | Formatting-only; formatting was applied to rewritten code. |
| `0f233eeec3` | PATCH-EQUIVALENT | Retained as `7d07b7b5b3`; guarded MiniMax prefill schedule is patch-equivalent. |
| `031fc9a691` | MODIFIED | Humming W4A8 was integrated with v0.27 standardized Humming interfaces in `19d96bcbfa`, then hardened by `90b13c90e1`. |

`e3f14d2e48` and `90b13c90e1` are target-only fail-closed hardening commits,
not direct source replays.

## 9. Runtime Switch

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `027887b86b` | MODIFIED | CuTe DSL LL BF16 runtime gate was adapted to v0.27 env/config conventions as `f24fd3012b`. |

## 10. Step-Level Prefill Token Buckets

The approved scope retained only functional scheduling behavior. Metrics,
NVTX, trace, and debug code were explicitly excluded.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `7f716ff4c5` | MODIFIED | Config, CLI, parser, scheduler behavior, and focused tests were rebuilt in `a8fd87c724`. Superseded per-chunk logic and observability were omitted. |
| `efbe85cff4` | DROP | Metrics-only gauge collision fix; metrics are outside the approved scope. |
| `494b0ecf05` | MODIFIED | Final step-level semantics were reimplemented and hardened through `14e2f52317`, `c38ceb7a3b`, `b691276f62`, and `f1d5ef2237`. |

## 11. Merge Commits

All merge commits are dropped as topology-only commits. Their children are
listed above.

| Merge commit | Disposition | Covered feature |
| --- | --- | --- |
| `d86d2d03cd` | DROP | MiniMax-M3 HPC |
| `f292c28cb6` | DROP | DSV4 PCP refresh |
| `71708e0b1c` | DROP | MiniMax W4A8 |
| `0bba7b0a64` | DROP | zstd/nydus publication |
| `e947b67b83` | DROP | W4A8 optimization |
| `ad70d89158` | DROP | apt mirror fixes |
| `de55af9138` | DROP | Humming W4A8 |
| `8da5f70fca` | DROP | Step-level prefill token buckets |

## 12. Intentionally Absent Semantics

The following omissions are deliberate, not migration misses:

- DSpark speculative K below its block size.
- Legacy PCP implementation and direct Mooncake PCP fan-in.
- DSV4 hybrid/sparse PCP declarations without target-native adapters.
- MiniMax indexer `fp8_e5m2` capability removal.
- Prefill token bucket metrics, PP queue telemetry, step latency, NVTX, trace,
  and debug helpers.
