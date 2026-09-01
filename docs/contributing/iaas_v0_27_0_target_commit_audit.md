# IAAS v0.27.0 Target Commit Audit

## 1. Audit Scope

This document audits every commit added on top of upstream `v0.27.0` through
`37eb9535b5`.

The audit answers two questions:

1. Does the migrated implementation preserve the accepted behavior from
   `iaas_main`?
2. Is the implementation minimal, correctly placed, and suitable for an
   upstream/community contribution?

The range contains 57 non-merge commits:

- 51 implementation or hardening commits.
- 6 migration/status documentation commits.

The `bits-code-guard` pass inspected 90 production/config/build files covering
9,636 changed lines after filtering generated, binary, and test-only files.
Test changes were still reviewed per commit for behavioral coverage, together
with the validation and independent-review records from migration rounds
1-54. Cross-group checks covered build-to-runtime ABI, config-to-runtime
constraints, connector metadata, and public interface changes.

This is a static and evidence audit. Existing focused CPU tests and prior
independent reviews are considered, but this document does not treat missing
GPU, multi-rank, CUDA Graph, RDMA, image-build, or registry-publication runs as
completed evidence.

No implementation code was changed as part of this audit.

The complete source-commit disposition is recorded separately in
`docs/contributing/iaas_v0_27_0_source_commit_disposition.md`.

## 2. Rating Terms

### Semantic equivalence

- **Equivalent**: preserves the accepted source behavior.
- **Equivalent + hardened**: preserves source behavior and adds target-native
  validation or fail-closed behavior.
- **Series-only equivalent**: correct in the final tree, but this individual
  commit is incomplete or unsafe without a later hardening commit.
- **Intentional divergence**: differs from raw `iaas_main` by an explicit
  migration decision.
- **N/A**: documentation or target-only hardening with no direct source
  semantic.

### Community suitability

- **A - upstream-ready shape**: appropriately scoped and placed; normal review
  and runtime evidence are still required.
- **B - refactor/split first**: useful community feature, but the commit should
  be split, generalized, or moved behind a cleaner extension boundary.
- **C - fork-only**: organization-specific workflow, private dependency, or
  workload-specific tuning that should not be proposed upstream as-is.

## 3. Executive Verdict

### 3.1 Correctness

The final tree is source-equivalent for the **accepted migration scope**, not
for every raw behavior in `iaas_main`.

The following differences are deliberate:

- DSpark `K < block_size` remains rejected.
- Legacy PCP and direct Mooncake PCP are not migrated.
- DSV4 hybrid/sparse PCP declarations remain held.
- MiniMax indexer `fp8_e5m2` removal remains held.
- Prefill token bucket metrics, PP queue telemetry, step latency, NVTX, trace,
  and debug helpers are omitted.

For retained behavior, the target generally improves failure behavior:
unsupported topologies, transports, graph modes, stale transfer generations,
and incomplete schedules fail closed rather than silently selecting an unsafe
path.

The static audit found two open correctness gaps:

- Dynamic SD can select a non-zero DSpark K smaller than the checkpoint's
  `dspark_block_size`, bypassing the static guard and entering a layout known
  to produce garbled output.
- The DeepGEMM commit pinned by ByteIAAS workflows does not accept the Kimi-K3
  MegaMoE keyword arguments used by the target runtime.

The final tree therefore does not yet satisfy full source-equivalence for
every accepted Dynamic SD configuration or the packaged Kimi-K3 MegaMoE path.

However, runtime equivalence is still conditional for GPU-sensitive areas:

- SM90 MegaMoE and private DeepGEMM APIs.
- MiniMax HPC MXFP8 K32.
- CUTLASS W4A8 with DeepEP LL/HT and Humming.
- Dynamic SD runtime-K transitions and CUDA Graph replay.
- Mooncake heterogeneous TP and RDMA transfer.
- Full ByteIAAS image build and publication.

### 3.2 Minimality and Commit Quality

The final implementation is substantially better aligned with v0.27 than a
mechanical cherry-pick, but the history is not consistently suitable as an
upstream commit series:

- `78dea35eb9` depends on the build contract added by `b84298dbe1`.
- `aecdda1e0d` is not safe without `b5c95a75f7`.
- the initial W4A8 support requires later transport/device hardening.
- `a8fd87c724` requires `14e2f52317`, `b691276f62`, and `f1d5ef2237`.
- ByteIAAS workflow commits are intentionally iterative and should be
  squashed before external review.

For community submission, each of these sequences should be rebased into
independently correct commits.

### 3.3 Architecture and Upstream Fit

The main upstream-fit concerns are:

1. **Fork-specific build infrastructure**: ByteIAAS workflows, internal
   runners, Volcengine registry details, private mirror settings, and release
   policy are fork-only.
2. **Private dependency contracts**: SM90 MegaMoE and MiniMax HPC depend on
   non-upstream DeepGEMM/HPC-Ops APIs. The generic source override machinery is
   reusable, but the private API contract should not be embedded in an
   upstream default path.
3. **Model/workload tuning in generic code**:
   `7d07b7b5b3` places a MiniMax-M3 8192-token shape-specific schedule in
   generic `cutlass_moe.py`. It is correctly gated, but a backend tuning table
   or model-owned policy would be a cleaner community design.
4. **Workload policy in core scheduler config**: token buckets are generic,
   but the default `4095:8192,16383:4096,-1:8192` policy is workload-derived.
   Upstream should prefer an explicitly supplied policy or a pluggable
   scheduler policy without a deployment-specific default.
5. **Large cross-cutting commits**: DSV4 correctness, Dynamic SD, Mooncake
   shared regions, and DeepEP W4A8 each touch shared contracts and should be
   split into protocol/schema, implementation, and tests for upstream review.

### 3.4 Confirmed Static Findings

| Audit group | P0 | P1 | P2 | Result |
| --- | ---: | ---: | ---: | --- |
| ByteIAAS build/release | 0 | 1 | 0 | Request changes |
| Dynamic SD/scheduler/draft | 0 | 1 | 1 | Request changes |
| Mooncake/KV transfer | 0 | 0 | 0 | No P0-P2 |
| DSV4/SM90/IndexCache | 0 | 0 | 0 | No P0-P2 |
| MoE/HPC/W4A8/Humming | 0 | 0 | 0 | No P0-P2; runtime gates remain |
| Cross-module contracts | 0 | 0 | 0 | No additional P0-P2 |

#### P1: The pinned DeepGEMM source does not satisfy Kimi-K3 MegaMoE

- Location:
  `.github/workflows/_byteiaas-build-and-publish-image.yml:61-62` and
  `.github/workflows/_byteiaas-build-wheel.yml:38-39`
- Related checker: `tools/check_deepgemm_source.py:21-36`
- Trigger: build the ByteIAAS image/wheel with its pinned
  `wangyicong52/DeepGEMM@babdbf01...`, then enable Kimi-K3 MegaMoE.
- Impact: the pinned source's `transform_weights_for_mega_moe` accepts only
  two positional arguments and `fp8_fp4_mega_moe` does not accept the Kimi
  beta keywords, while `vllm/models/kimi_k3/nvidia/model.py` passes them.
  The path deterministically raises `TypeError`.
- Required resolution: pin a commit that jointly implements the SM90 and Kimi
  SiTU contracts, align argument names with that API, and extend the source
  checker to validate every keyword used by Kimi.

#### P1: Dynamic SD can select an invalid DSpark reduced K

- Location: `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-138`
- Introduced by: `33176a5408`
- Trigger: a DSpark checkpoint with `dspark_block_size=5`, static
  `num_speculative_tokens=7`, and a Dynamic SD schedule containing non-zero
  `K=3`.
- Impact: the runtime helper accepts every value in `[0, Kmax]`, while the
  static DSpark validation explicitly states that `0 < K < block_size`
  produces incorrect output.
- Required resolution: validate every DSpark schedule entry as
  `K == 0 or K >= dspark_block_size`, and retain a runtime defensive check.

#### P2: Dynamic SD hooks break older custom schedulers

- Location: `vllm/v1/engine/core.py:929-933` and
  `vllm/v1/core/sched/interface.py:85-92`
- Introduced by: `aecdda1e0d`
- Trigger: a custom scheduler that implements the v0.27 interface without
  inheriting the built-in scheduler, followed by an offline or idle-rank dummy
  step while Dynamic SD is disabled.
- Impact: `EngineCore.execute_dummy_batch()` unconditionally calls a newly
  added interface method whose default implementation raises
  `NotImplementedError`.
- Required resolution: provide a no-Dynamic-SD default and require the new
  hooks only when the feature is enabled.

## 4. Commit-by-Commit Audit

### 4.1 Initial Planning and Documentation

| Commit | Semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `9a20ec818a` | N/A; planning only. | One-file change, but contains fork migration state rather than community feature documentation. | Internal-only, correct. |
| `7c3c42f134` | N/A; status update only. | Minimal ledger update; not part of an upstream feature PR. | Internal-only, correct. |
| `2a34eb5428` | N/A; ledger finalization only. | Minimal and useful for traceability; fork-specific. | Internal-only, correct. |
| `df114553d9` | N/A; Buildx/H20 evidence only. | References internal validation environment. | Internal-only, correct. |
| `338066b626` | N/A; secure-branch evidence only. | Specific to ByteIAAS workflow policy. | Internal-only, correct. |
| `37eb9535b5` | N/A; token-bucket ledger update only. | Minimal. A community PR should carry feature documentation, not this migration ledger. | Internal-only, correct. |

### 4.2 DFlash, DSV4, and SM90 Foundation

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `aa5a75d799` | Patch-equivalent to `55bfba2920`; preserves TMA-aligned scale propagation. | Narrow kernel adapter plus focused tests. | Equivalent; **A**. |
| `cca9a49550` | Adapts `9c8a93450d`; retains standard-cache selection while omitting the obsolete nonstandard indexer route. | Generic topology helper, narrowly scoped. | Equivalent + hardened; **A**. |
| `6120c51141` | Adapts `f37b865341`; passes the blockwise activation-clamp contract with fail-closed API detection. | Generic HPC bridge, but usefulness depends on an external HPC-Ops API. | Equivalent + hardened; **B**. |
| `f24fd3012b` | Adapts `027887b86b`; preserves the LL BF16 runtime kill switch. | Small opt-in environment gate in the owning backend. | Equivalent; **A**. |
| `5a20fd1389` | Adapts `f56547192d`; preserves missing FlashInfer CUTLASS MXFP4 conversion and SwiGLU parameter behavior. | Generic backend change with explicit contracts; should be tied to a public dependency version. | Equivalent + hardened; **A/B**. |
| `549942052e` | Squashes `3c7ed3742c`, `657132e0aa`, `3bc91bb763`, `d62244f050`, and `6ca83dd045`, excluding upstream-covered concurrency logic. | Correct target-native split by behavior would be easier to review than this 12-file commit. Generic config/indexer changes are gated by DSV4 metadata. | Equivalent + hardened; **B**. |
| `effe7093cf` | Reimplements `39d86ed8c7` on packed cache layouts. | Most model policy remains in DSV4 files; the core cache utility extension should stay metadata-driven. | Equivalent + hardened; **B**. |
| `78dea35eb9` | Preserves the runtime half of `3e60ad63d3`, including sequence-parallel and Kimi compatibility adaptations. Its Kimi call contract is not met by the ByteIAAS-pinned dependency. | Model-owned runtime placement is mostly appropriate, but it is not independently buildable without a compatible fork API. | Series-only; packaged Kimi path has an open P1; **B/C**. |
| `b84298dbe1` | Preserves the build/dependency half of `3e60ad63d3` with exact-source and symbol validation, but the checker omits Kimi's `activation` and beta keyword contract. | Generic source override logic is reusable; the SM90/Kimi private API contract is incomplete and not upstream-ready. | `REQUEST CHANGES`; **C** until dependency APIs are complete and public. |

### 4.3 Dynamic SD and Draft Execution

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `86ec283e47` | Adapts residual behavior from `d623aef9c0`; preserves async bounds, draft KV ownership, producer-only paths, and prefix masking. | An 18-file correctness bundle is difficult to review; split by independent failure mode upstream. | Equivalent + hardened; **B**. |
| `9e006972e2` | Rebuilds `83c16e1cb6` on the current scheduler. | Scheduler policy is generic, but config/state logic should remain in a dedicated Dynamic SD module where possible. | Equivalent + hardened; **B**. |
| `33176a5408` | Preserves `bb946a7ce7` runtime-K propagation across MRV2, but accepts every K in `[0,Kmax]` and bypasses DSpark's block-size correctness constraint. | Cross-layer propagation is necessary, but the missing method-specific invariant leaves an open P1. | `REQUEST CHANGES`; **B**. |
| `aecdda1e0d` | Implements `67a597c869` plus explicit dummy-K behavior from `aa8e0a91b8`/`9016ee5bb6`. Initial MRV1, pressure, and empty-schedule defects were fixed later, but the unconditional custom-scheduler hook remains an open P2. | Not acceptable as a standalone commit; optional capabilities should not be mandatory interface calls when disabled. | `REQUEST CHANGES`; **B**. |
| `b5c95a75f7` | Target-only hardening that rejects MRV1 DP-global mode, fixes admission pressure, and validates schedules early. It does not validate DSpark schedule K values or restore custom-scheduler defaults. | Correctly scoped follow-up, but should be squashed into the feature commit upstream. | Correct hardening with two remaining cross-commit findings; **A/B**. |
| `d709a41c46` | Adapts `7a6f9efb6f`; preserves query-length graph families and tail/DP redispatch. | Cross-cuts graph manager and DFlash by necessity; should be submitted with explicit graph-family contract docs. | Equivalent + hardened; **B**. |
| `d00896ae74` | Squashes `09cede6391` and `77ed1eece5`; preserves full-K warmup and cleanup/state restoration. | Large single-function warmup change; factor proposer-specific warmup helpers before upstreaming. | Equivalent + hardened; **B**. |
| `0c142e5b8e` | Adapts `23b6150467`; preserves padded-row masks through MegaMoE. | DSpark/DFlash-owned logic with focused tests. | Equivalent; **A/B**. |
| `66bf65058d` | Squashes `da0e8131f2`, `1f00d2a3bf`, and `9dd89bf957`; preserves independent draft backend/config selection. | Shared configuration is appropriate, though DFlash/DSpark branches should remain isolated behind method interfaces. | Equivalent + hardened; **A/B**. |

### 4.4 Mooncake and KV Transfer

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `5b701d086f` | Squashes `cfcba9469a`/`95188da984`; preserves alias/group identity, PP intersections, per-region selection, and recovery validation. | Correctly concentrated in the Mooncake connector, but the 2.5k-line commit should be split into schema, matching, recovery, and tests. | Equivalent + hardened; **B**. |
| `f00de7548e` | Partially ports `88e2f20ed2`; retains GQA replicas, replicated regions, per-layer KV heads, and stale completion safety while keeping upstream TP-ratio logic. | Connector/base metadata changes are general; model metadata additions are narrowly gated. | Equivalent for accepted subset + hardened; **B**. |
| `4324aa0125` | Adapts `07fbbb0bdb`; rejects unsupported PP fanout. | Small connector-owned fail-closed validation. | Equivalent + hardened; **A**. |
| `d5115b2da8` | Adapts `f20c7435de`; preserves opt-in node-shared long-request admission and generation-scoped lifecycle. | Operational policy is correctly connector-local. The `VLLM_MOONCAKE_PD_TRACE` diagnostics should be split or omitted from a strict core-only community series. | Equivalent + hardened; **B/C**. |

### 4.5 MiniMax HPC and W4A8

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `deb0b32e0d` | Squashes `57fed50368`, `25e140853f`, `4799bb3106`, `55b2e207d1`, and `4b730aef58`; retains explicit TP16/SM90 MXFP8 K32 behavior. | Model-specific class is isolated, but private HPC-Ops and a narrow topology make this fork-only until a public plugin/dependency contract exists. | Equivalent + hardened; **C**. |
| `099caa830f` | Squashes `8bef682e2d`, `5188517c82`, and `df0273759e`; preserves SwiGLU parameters and standard CUTLASS W4A8 support. | Generic activation/quant contracts are appropriate; the initial commit relies on later transport/device hardening. | Series-only equivalent; **B**. |
| `29a60ddd0a` | Adapts `b283f4b26a` to current DeepEP LL and stable op ABI. | ABI and backend changes are broad but belong to the CUTLASS/DeepEP integration. | Equivalent + hardened; **B**. |
| `4779c1117c` | Adapts `2bbc5b51a9`; preserves useful-row and compact masked quantization scheduling. | Narrow performance change with contract coverage. | Equivalent; **A/B**. |
| `d39f8de0c0` | Adapts `6226ac4e52`; preserves per-token scales, scratch sizing, and graph topology guards for DeepEP HT. | Correct backend placement; upstream requires real multi-rank graph evidence. | Equivalent + hardened; **B**. |
| `7d07b7b5b3` | Patch-equivalent to `0f233eeec3`. | Exact MiniMax-M3 shape and 8192-token policy live in generic `cutlass_moe.py`; move to a tuning registry or model/backend policy before upstreaming. | Equivalent; **C** as written. |
| `e3f14d2e48` | Target-only hardening; rejects transports that cannot preserve W4A8 per-token scale semantics. | Small fail-closed backend gate; should be folded into initial support. | Correct hardening; **A**. |
| `19d96bcbfa` | Adapts `031fc9a691` to the standardized Humming integration. | Backend-specific implementation is reasonably isolated; upstream suitability depends on a public, supportable Humming dependency. | Equivalent + hardened; **B**. |
| `90b13c90e1` | Target-only hardening; restricts W4A8 to SM90 and applies graph policy to PCP all-to-all. | Appropriate capability gate, but should be part of the original backend commits. | Correct hardening; **A/B**. |

### 4.6 ByteIAAS Build and Release

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `f8040e5a16` | Rebuilds image-tag and manifest helpers from several source CI commits. Initial manifest traversal and tag-bound checks required `16d786bc7b`. | Helper logic is testable, but naming/tag policy is ByteIAAS-specific. | Series-only; **C**. |
| `16d786bc7b` | Target-only hardening of manifest DAG and format verification. | Good defensive follow-up; should be squashed with helper introduction. | Correct hardening; **C**. |
| `7899a9fa28` | Rebuilds HPC-Ops/devel-image portions of the source Docker changes. Its initial Ninja/CUDA package assumptions required `7bac341c97`. | Main `docker/Dockerfile` now contains private HPC-Ops defaults and ByteIAAS stages; isolate these in a fork Dockerfile or generic extension interface for upstream. | Series-only; **C**. |
| `dfd6f57723` | Rebuilds ByteIAAS CUDA 13.0.3 wheel/image workflows, but pins the incompatible DeepGEMM SHA used by the open Kimi P1. | Internal runners, mirrors, registry, and repository guards are organization-specific. | `REQUEST CHANGES`; **C**. |
| `9d72a701d3` | Rebuilds zstd/nydus publication. Immutable zstd inputs and nydus metadata isolation were completed by later hardening commits. | Verification techniques are reusable, but workflow and registry policy are fork-only. | Series-only; **C**. |
| `7bac341c97` | Target-only Docker build-contract hardening. | Correct fail-closed checks; still part of the private image pipeline. | Correct hardening; **C**. |
| `ac36a3ee3b` | Target-only source ancestry/contract pinning. Later commits complete secret, action, and checkout-token isolation. | Security-positive but tied to `iaas_main` and private release policy. | Series-only hardening; **C**. |
| `2fd47e7683` | Retains rustup retry behavior from `23dfea6fca`. | Generic and small; could be upstreamed independently if still needed. | Equivalent + hardened; **A**. |
| `f798a87ca3` | Target-only registry credential scoping. | Correct least-privilege hardening; workflow remains fork-only. | Correct hardening; **C**. |
| `90aecd5ad3` | Target-only immutable source/digest validation. | Correct release hardening; fork policy. | Correct hardening; **C**. |
| `4a1bd81125` | Target-only Docker credential and nydus metadata isolation; persistent-runner setup/cleanup is completed by `138e061e38`. | Correct hardening; implementation is specific to the private workflow. | Series-only hardening; **C**. |
| `138e061e38` | Target-only persistent-runner cleanup. | Required for private self-hosted runners; not an upstream concern. | Correct hardening; **C**. |
| `f911556cb6` | Target-only checkout credential isolation. | Correct least-privilege hardening; fork-only workflow. | Correct hardening; **C**. |
| `445cbf2b2c` | Allows exact self-dispatched branch validation. | Necessary internal pre-merge path, but unsafe without the following bypass fix. | Series-only; **C**. |
| `3b27f22d7c` | Rejects tag/unrelated-SHA bypass in self-dispatch. | Correct hardening; should be squashed with `445cbf2b2c`. | Final fork behavior correct; **C**. |

### 4.7 Step-Level Prefill Token Buckets

| Commit | Source relation and semantic audit | Minimality / community audit | Verdict |
| --- | --- | --- | --- |
| `a8fd87c724` | Reimplements functional parts of `7f716ff4c5` and `494b0ecf05`; omits metrics/debug scope. Initial version did not correctly cap decode-after-prefill and mishandled Mamba/async-KV edges. | Generic feature, but not independently correct and default policy is workload-derived. | Series-only; requires later fixes; **B**. |
| `14e2f52317` | Applies the active bucket cap to decode scheduled after prefill. | Necessary correctness fix; should be squashed into `a8fd87c724`. | Series hardening; **B**. |
| `c38ceb7a3b` | Preserves cross-bucket running-prefill pressure for DP cadence. | One-line targeted correction; should be squashed. | Series hardening; **B**. |
| `b691276f62` | Separates fixed effective Mamba cap from residual budget and allows pure async KV load after compute cap. | Correct cross-feature integration; should be part of the initial scheduler change. | Series hardening; **B**. |
| `f1d5ef2237` | Scans past capped compute requests so later async KV loads can overlap. | Completes the accepted semantics. Before upstreaming, remove the workload-specific default or move the policy behind a scheduler extension. | Equivalent + hardened final state; **B**. |

## 5. Recommended Upstream Decomposition

Do not propose `v0.27.0..HEAD` as one community change. A practical upstream
series would be:

1. Generic correctness fixes that do not require private dependencies:
   TMA scale propagation, standard cache selection, LL BF16 gate, residual
   DFlash/DSpark fixes.
2. DSV4 correctness and IndexCache as separate PRs, with model-owned policy
   and generic cache APIs separated.
3. Dynamic SD scheduler policy, runtime-K interfaces, DP synchronization, graph
   families, and warmup as distinct but ordered PRs.
4. Mooncake schema/identity, heterogeneous TP, stale completion, and admission
   policy as separate connector PRs.
5. Generic CUTLASS W4A8 activation/ABI changes, then DeepEP LL, scheduling,
   DeepEP HT, and Humming backends.
6. Model/workload tuning through a tuning registry rather than conditionals in
   generic kernels.
7. Token bucket scheduling with explicit user policy and no deployment-derived
   default.

ByteIAAS workflows, private mirrors, registry policy, internal runner cleanup,
private HPC-Ops defaults, and migration ledger commits should stay in the fork.

## 6. Remaining Evidence Required

Before claiming full production equivalence:

- Build the exact v0.27 image and verify packaged DeepGEMM/HPC-Ops/Humming ABIs.
- Run SM90/H20 numeric tests for HPC MXFP8 and CUTLASS/Humming W4A8.
- Run multi-rank DeepEP LL/HT and PCP2 graph replay.
- Run Dynamic SD eager and CUDA Graph transitions for K=`0/3/5/7`.
- Run Mooncake TP/PP/GQA matrices with RDMA and request-ID reuse.
- Run the token-bucket policy under Dynamic SD, async KV load, Mamba, priority
  preemption, and DP cadence in an integrated engine workload.

Until those checks complete, the correct conclusion is:

- **Static/source-level migration**: `REQUEST CHANGES` because of the two open
  P1 findings (DeepGEMM/Kimi ABI and DSpark reduced-K) and the custom-scheduler
  P2.
- **Production runtime equivalence**: not fully proven.
- **Community upstream readiness**: mixed; generic pieces are upstreamable
  after decomposition, while ByteIAAS/private-dependency pieces are fork-only.
