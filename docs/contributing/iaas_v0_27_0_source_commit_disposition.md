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
| Target implementation head | `944ecfdb054c7cd9d817c1901a2059775798018d` |

The source range contains 66 non-merge commits and 8 merge commits. All 74
commits are listed below. The second-pass audit also enumerated all 1,266
non-merge source hunks and checked every merge with a combined diff. The
function-level and direct-caller checks compare source behavior, upstream
v0.27 behavior, each mapped target commit, and the final target tree.

## 2. Disposition Terms

- **PATCH-EQUIVALENT**: the source patch is present unchanged by patch-id,
  although the target commit hash differs.
- **REIMPLEMENTED**: required behavior was retained through a target-native port,
  split, squash, or reimplementation against the v0.27 architecture.
- **UPSTREAM-COVERED**: upstream v0.27 independently supplies the behavior.
- **SUPERSEDED**: a later source state replaces this intermediate behavior.
- **DROP**: formatting, metrics, debug, or other explicitly excluded behavior.
- **HOLD**: intentionally not migrated because correctness or architecture
  evidence is insufficient.
- **WRONG**: the target contains behavior that conflicts with the approved
  disposition or source invariant.
- A slash-separated result means different hunks in one commit have different
  dispositions. The behavior-level ledger below explains each mixed result.

`git cherry 944ecfdb05 8da5f70fca` identifies only `55bfba2920` and
`0f233eeec3` as patch-equivalent among the IAAS feature commits after excluding
upstream-history matches. Other retained features were adapted and must not be
described as simple cherry-picks.

## 3. Summary

| Dominant disposition | Non-merge commits |
| --- | ---: |
| PATCH-EQUIVALENT | 2 |
| REIMPLEMENTED | 48 |
| DROP | 4 |
| SUPERSEDED | 3 |
| UPSTREAM-COVERED | 1 |
| HOLD | 2 |
| WRONG | 1 |
| MIXED | 5 |
| **Total** | **66** |

All 8 merge commits are topology-only: `git diff-tree -c --cc` reports zero
paths with merge-only conflict resolution. Their non-merge children are
accounted for separately.

### 3.1 Second-Pass Result

| Class | Result |
| --- | --- |
| Missing migration | The previously missed Python/uv bootstrap portion of `a9f1617cc7` is now restored by `944ecfdb05`; no additional accepted source behavior was found missing. |
| Wrong migration | `33176a5408` permits Dynamic SD to select `0 < K < dspark_block_size`, reintroducing the unsafe behavior intentionally held from `0d6fd1c83c`. |
| Extra migration | No target behavior was found without a source, upstream-adaptation, or documented hardening rationale. Target-only build security and fail-closed checks remain justified, although several are fork-only. |
| Other defect | `b691276f62` changed the Mamba helper call without updating a Dynamic SD test monkeypatch; the focused suite fails before exercising its assertion. |

### 3.2 Mixed-Behavior Ledger

| Source | Behavior-level disposition |
| --- | --- |
| `3c7ed3742c` | DSV4 RoPE, FP8 metadata, BF16 O-proj, and compressed-slot bounds are **REIMPLEMENTED** in `549942052e`; concurrency accounting is **UPSTREAM-COVERED** by v0.27. |
| `87ec6ecd11` | MRV2 PCP core is **UPSTREAM-COVERED**; the legacy runner/metadata implementation is **DROP** because it conflicts with v0.27; direct Mooncake PCP fan-in remains **HOLD**. |
| `8b291ae6f3` | Token-major top-k layout is **UPSTREAM-COVERED** by `d1a8ba63d9`; removing `fp8_e5m2` remains **HOLD**. |
| `7f716ff4c5` | Config/CLI and scheduling behavior are **REIMPLEMENTED**; per-chunk behavior is **SUPERSEDED** by step-level scheduling; metrics, trace, and debug hunks are **DROP**. |
| `18f1129660` | The old-mirror avoidance is **SUPERSEDED** by the Volces mirror; apt retry behavior is **REIMPLEMENTED** in the final Docker/workflow path. |
| `55b2e207d1` | HPC workspace reuse and the `_out` operator contract are **REIMPLEMENTED** in `deb0b32e0d`; generic modular-kernel output aliasing is **UPSTREAM-COVERED**. |
| `0d6fd1c83c` | Direct guard removal remains intentionally absent, but Dynamic SD reaches the same unsupported reduced-K behavior through `33176a5408`; final disposition is **WRONG**. |

## 4. Build and Release

The old v0.26 workflows and Docker layout were not replayed directly. They
were rebuilt around the v0.27 Docker stages, vendored DeepGEMM layout,
HPC-Ops contract, immutable publication inputs, and scoped credentials. All 85
source hunks in this group were checked.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `a9f1617cc7` | REIMPLEMENTED | Helpers, Docker stages, and workflows were rebuilt by `f8040e5a16`, `7899a9fa28`, and `dfd6f57723`; branch dispatch and mirror behavior were restored by `ec31fc96bc`; Python/uv bootstrap, local multi-Python DeepGEMM setup, bounded retries, and ByteIAAS `RUN_WHEEL_CHECK=false` were restored by `944ecfdb05`. |
| `4edbfab0a9` | DROP | Only updates the local actionlint hook from 1.7.7 to 1.7.12. The target keeps v0.27's pin, and actionlint passes on all four workflows; no runtime or workflow behavior is missing. |
| `23dfea6fca` | REIMPLEMENTED | HPC image checks moved to `7899a9fa28`; rustup retry became `2fd47e7683`. |
| `ec58aaf9f3` | REIMPLEMENTED | Final HPC-Ops source and image integration folded into `7899a9fa28`. |
| `30cf9d9bad` | REIMPLEMENTED | HPC-Ops wheel/API validation folded into `7899a9fa28`. |
| `9493f04276` | REIMPLEMENTED | Tag logic moved to `f8040e5a16`; verified zstd/nydus publication became `9d72a701d3`. |
| `aa4fd55c54` | REIMPLEMENTED | Manifest inspection was redesigned in `f8040e5a16`, `16d786bc7b`, and `9d72a701d3`. |
| `348c70ca0b` | SUPERSEDED | Intermediate ByteDance mirror state; replaced by the final Volces mirror. |
| `0b37e9134c` | SUPERSEDED | Intermediate fixed `/etc/hosts` workaround; removed by later source commits. |
| `09aa3c1bde` | SUPERSEDED | Intermediate BuildKit host mapping; removed by `18f1129660`. |
| `18f1129660` | SUPERSEDED / REIMPLEMENTED | Direct old-mirror access was superseded; apt retry remains in the target Docker path. |
| `eb454b2d43` | REIMPLEMENTED | Final Volces Ubuntu mirror state is retained and expanded to base, runtime, and devel builds by `7899a9fa28` and `ec31fc96bc`. |
| `2bee42cda6` | UPSTREAM-COVERED | `humming-kernels[cu13]==0.1.10` is already pinned by upstream v0.27. |

Target-only hardening was added after the source behavior was reconstructed:
`7bac341c97`, `ac36a3ee3b`, `f798a87ca3`, `90aecd5ad3`,
`4a1bd81125`, `138e061e38`, `f911556cb6`, `445cbf2b2c`, and
`3b27f22d7c`. `ec31fc96bc` and `944ecfdb05` are later source-equivalence
repairs, not target-only features.

## 5. DSV4, IndexCache, and SM90 MegaMoE

This group accounts for 454 source hunks. The second pass confirmed that PCP
omissions are architecture decisions rather than path-level misses.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `3c7ed3742c` | REIMPLEMENTED | Missing RoPE, FP8 metadata, BF16 O-proj, and metadata-bound semantics moved to `549942052e`; the already-upstream concurrency hunk was dropped. |
| `39d86ed8c7` | REIMPLEMENTED | IndexCache was manually redesigned for compact/packed v0.27 cache layouts in `effe7093cf`. |
| `3e60ad63d3` | REIMPLEMENTED | Split into runtime `78dea35eb9` and validated DeepGEMM build contract `b84298dbe158a454e91afbae6d2a25a3d63d2d20`. |
| `657132e0aa` | REIMPLEMENTED | Dummy decode row guard folded into `549942052e`. |
| `3bc91bb763` | REIMPLEMENTED | Stable paged-MQA sequence-length storage folded into `549942052e`. |
| `d62244f050` | REIMPLEMENTED | CUDA-graph capture sequence-length semantics folded into `549942052e`. |
| `6ca83dd045` | REIMPLEMENTED | Capture metadata/eager warmup compatibility folded into `549942052e`. |
| `87ec6ecd11` | DROP / HOLD | Legacy PCP core was dropped because v0.27 uses MRV2 virtual-batch PCP; direct Mooncake PCP gaps remain held. |
| `f4a2217461` | HOLD | Hybrid-backend PCP declarations await target-native adapters and GPU parity. |
| `20bce687d4` | HOLD | Sparse-MLA PCP declaration awaits MRV2 adapter and parity evidence. |

## 6. Dynamic SD and Draft Correctness

This group accounts for 322 source hunks. The retained paths are complete, but
one HOLD invariant is violated by the target runtime-K implementation.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `d623aef9c0` | REIMPLEMENTED | Residual DFlash/DSpark correctness semantics became `86ec283e47`; upstream-covered non-causal fixes were omitted. |
| `83c16e1cb6` | REIMPLEMENTED | Scheduler policy and budget reclaim were rebuilt on the v0.27 scheduler as `9e006972e2`. |
| `bb946a7ce7` | REIMPLEMENTED | Runtime-K propagation across MRV2 executor, target, proposer, and dummy paths became `33176a5408`. |
| `67a597c869` | REIMPLEMENTED | Opt-in DP-global pressure became `aecdda1e0d` and was hardened by `b5c95a75f7`. |
| `7a6f9efb6f` | REIMPLEMENTED | Query-length CUDA-graph family dispatch became `d709a41c46`. |
| `09cede6391` | REIMPLEMENTED | Full-K proposer warmup folded into `d00896ae74`. |
| `77ed1eece5` | REIMPLEMENTED | Warmup cleanup/reset folded into `d00896ae74`. |
| `23b6150467` | REIMPLEMENTED | Idle-DP draft padding semantics became `0c142e5b8e`. |
| `da0e8131f2` | REIMPLEMENTED | DFlash backend override folded into `66bf65058d`. |
| `129e53cf71` | DROP | Test formatting/style only. |
| `1f00d2a3bf` | REIMPLEMENTED | Passed draft config behavior folded into `66bf65058d`. |
| `aa8e0a91b8` | REIMPLEMENTED | Explicit scheduler-selected K folded into `aecdda1e0d`/`b5c95a75f7`. |
| `9016ee5bb6` | REIMPLEMENTED | K=0 regression coverage folded into `aecdda1e0d`/`b5c95a75f7`. |
| `55bfba2920` | PATCH-EQUIVALENT | Retained as `aa5a75d799`; DeepGEMM TMA-aligned activation-scale behavior is patch-equivalent. |
| `0d6fd1c83c` | WRONG | The direct guard removal was not cherry-picked, but `33176a5408` accepts every runtime K in `[0,Kmax]`; Dynamic SD can therefore select the same unsupported `0 < K < dspark_block_size` behavior. |
| `9dd89bf957` | REIMPLEMENTED | Final shared DFlash/DSpark backend override folded into `66bf65058d`. |

## 7. Mooncake and KV Transfer

This group accounts for 110 source hunks. PCP-related hunks were additionally
cross-checked with the 288 PCP hunks in the DSV4 group.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `cfcba9469a` | REIMPLEMENTED | Shared alias/group transfer identity was rebuilt as `5b701d086f`. |
| `95188da984` | REIMPLEMENTED | Alias regression coverage folded into `5b701d086f`. |
| `f20c7435de` | REIMPLEMENTED | Node-shared long-context admission and generation-scoped lifecycle became `d5115b2da8`. |
| `88e2f20ed2` | REIMPLEMENTED | Only stale completion, GQA replica, and fully replicated-region semantics were ported in `f00de7548e`; upstream simple TP-ratio logic was retained. |
| `9c8a93450d` | REIMPLEMENTED | Standard-cache backend selection became `cca9a49550`; nonstandard indexer selection was intentionally omitted. |
| `07fbbb0bdb` | REIMPLEMENTED | Unsupported producer/consumer PP fanout now fails closed in `4324aa0125`. |

## 8. MoE, MiniMax, and Humming

This group accounts for 225 source hunks.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `f37b865341` | REIMPLEMENTED | HPC blockwise activation-clamp contract became `6120c51141`. |
| `f56547192d` | REIMPLEMENTED | Missing FlashInfer CUTLASS MXFP4 wiring was adapted as `5a20fd1389`. |
| `57fed50368` | REIMPLEMENTED | Base MiniMax-M3 HPC MXFP8 path folded into `deb0b32e0d`. |
| `25e140853f` | REIMPLEMENTED | Capability/shape gates folded into `deb0b32e0d`. |
| `4799bb3106` | REIMPLEMENTED | Fused BF16 candidate path folded into `deb0b32e0d`. |
| `55b2e207d1` | REIMPLEMENTED / UPSTREAM-COVERED | HPC workspace reuse and the `_out` operator contract are rebuilt in `deb0b32e0d`; generic modular-kernel output aliasing is already present in upstream v0.27. |
| `4b730aef58` | REIMPLEMENTED | Unsupported MiniMax HPC chunking is disabled in `deb0b32e0d`. |
| `8b291ae6f3` | UPSTREAM-COVERED / HOLD | Token-major top-k layout is upstream as `d1a8ba63d9`; removing `fp8_e5m2` remains held for evidence. |
| `8bef682e2d` | REIMPLEMENTED | W4A8 SwiGLU parameter propagation folded into `099caa830f`. |
| `5188517c82` | REIMPLEMENTED | MiniMax CUTLASS W4A8 activation folded into `099caa830f`. |
| `df0273759e` | REIMPLEMENTED | CUTLASS W4A8 contract tests folded into `099caa830f`. |
| `b283f4b26a` | REIMPLEMENTED | Batched W4A8 was adapted to current DeepEP LL and stable ABI in `29a60ddd0a`. |
| `2bbc5b51a9` | REIMPLEMENTED | Useful-row/compact scheduling became `4779c1117c`. |
| `6226ac4e52` | REIMPLEMENTED | W4A8 was adapted to current DeepEP HT and graph policy in `d39f8de0c0`. |
| `176c597e1e` | DROP | Formatting-only; formatting was applied to rewritten code. |
| `0f233eeec3` | PATCH-EQUIVALENT | Retained as `7d07b7b5b3`; guarded MiniMax prefill schedule is patch-equivalent. |
| `031fc9a691` | REIMPLEMENTED | Humming W4A8 was integrated with v0.27 standardized Humming interfaces in `19d96bcbfa`, then hardened by `90b13c90e1`. |

`e3f14d2e48` and `90b13c90e1` are target-only fail-closed hardening commits,
not direct source replays.

## 9. Runtime Switch

All 6 source hunks were checked.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `027887b86b` | REIMPLEMENTED | CuTe DSL LL BF16 runtime gate was adapted to v0.27 env/config conventions as `f24fd3012b`. |

## 10. Step-Level Prefill Token Buckets

The approved scope retained only functional scheduling behavior. Metrics,
NVTX, trace, and debug code were explicitly excluded. All 64 source hunks were
checked.

| Source | Disposition | Target / reason |
| --- | --- | --- |
| `7f716ff4c5` | REIMPLEMENTED / SUPERSEDED / DROP | Config, CLI, parser, scheduler behavior, and focused tests were rebuilt in `a8fd87c724`; per-chunk logic was superseded by step-level scheduling; observability was deliberately omitted. |
| `efbe85cff4` | DROP | Metrics-only gauge collision fix; metrics are outside the approved scope. |
| `494b0ecf05` | REIMPLEMENTED | Final step-level semantics were reimplemented and hardened through `14e2f52317`, `c38ceb7a3b`, `b691276f62`, and `f1d5ef2237`. |

## 11. Merge Commits

All merge commits are dropped as topology-only commits. Their children are
listed above, and the combined diff for every merge has zero unique paths.

| Merge commit | Unique combined-diff paths | Disposition | Covered feature |
| --- | ---: | --- | --- |
| `d86d2d03cd` | 0 | DROP | MiniMax-M3 HPC |
| `f292c28cb6` | 0 | DROP | DSV4 PCP refresh |
| `71708e0b1c` | 0 | DROP | MiniMax W4A8 |
| `0bba7b0a64` | 0 | DROP | zstd/nydus publication |
| `e947b67b83` | 0 | DROP | W4A8 optimization |
| `ad70d89158` | 0 | DROP | apt mirror fixes |
| `de55af9138` | 0 | DROP | Humming W4A8 |
| `8da5f70fca` | 0 | DROP | Step-level prefill token buckets |

## 12. Intentionally Absent Semantics

The following omissions are deliberate, not migration misses:

- Legacy PCP implementation and direct Mooncake PCP fan-in.
- DSV4 hybrid/sparse PCP declarations without target-native adapters.
- MiniMax indexer `fp8_e5m2` capability removal.
- Prefill token bucket metrics, PP queue telemetry, step latency, NVTX, trace,
  and debug helpers.

DSpark speculative K below its block size is still an intended HOLD, but it is
not fully absent: the Dynamic SD runtime path currently reintroduces it. This is
the confirmed wrong-migration finding associated with `0d6fd1c83c` and
`33176a5408`.

## 13. Evidence Boundaries

Static/source-level evidence includes commit, path, hunk, patch-id, blame,
function/caller, merge combined-diff, focused CPU tests, and build-script
validation. It does not claim completion of GPU numeric parity, CUDA Graph
replay, multi-rank DeepEP/DP/PCP, RDMA Mooncake transfer, full Docker/Buildx, or
registry publication. Those remain runtime validation requirements.
