# vLLM DSV4 Flash P/D Fork-Base Build Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:subagent-driven-development` only if one worker owns GitHub branch/backup operations and another worker owns ByteIAAS build-file restoration. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe` the new source base for `bytedance-iaas/vllm:iaas_main`, preserve only the necessary ByteIAAS build/publish capabilities, build a runnable image, deploy it on `dev-cluster` without runtime hotfix/install, and benchmark the requested 64k/1 TTFT and cache-hit decode BS512 throughput cases.

**Architecture:** First create a remote GitHub backup branch named `backup/iaas_main-YYYYMMDD` pointing at the original `origin/iaas_main` SHA, then add a GitHub ruleset protecting `backup/iaas_main-*`. Then create a new integration branch from `wangyicong52/vllm.git` and restore only ByteIAAS build-related files from the backed-up old `iaas_main`; do not port old vLLM model/runtime/source logic. After the image is built, add a vLLM-owned deployment example based on the servingkit Helm chart, but remove runtime hotfixes, `git clone`, `pip install`, wheel download/install, and router package install paths because the image must contain all required code and libraries. The newly built vLLM image must also contain `oniond`; if the base image lacks it, install `onion-ai-data` during image build using the `onion-ai-data` skill's Volcengine apt source flow. The deployment template should use the same new vLLM image for Onion model preparation and serving, relying on Onion's idempotent skip behavior when the target model already exists. Deploy on `dev-cluster` with workspace-env GPU permits, then use evalscope for the requested benchmark runs and preserve raw artifacts. Only after the image, deployment, router smoke, and required evalscope measured runs complete or have explicitly accepted blockers should the remote `iaas_main` be updated.

**Tech Stack:** `bytedance-iaas/vllm`, `wangyicong52/vllm`, GitHub remote branches, GitHub Actions, Docker Buildx, CUDA 13.0.2, `uv`, ByteIAAS Volcengine CR publish workflow, Mooncake, DeepEP, DeepGEMM, `vllm-router`.

## Global Constraints

- 计划、命令引用、进展日志、目标、背景、假设、约束、非目标、里程碑、风险、审批说明、验证说明、进展摘要和最终摘要使用中文。
- 命令、文件路径、代码标识、错误信息、API 名称和专有名词保持原文。
- 主计划路径：`docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.md`
- 命令引用路径：`docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.commands.md`
- 进展日志路径：`docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.progress.md`
- 不创建 `.codex/plans`。
- 执行前必须在 GitHub 远端创建原始 `iaas_main` 备份分支；本地备份不满足要求。
- 新 `iaas_main` 起点必须是 `https://github.com/wangyicong52/vllm.git` 的 `dev/dsv4-mooncake-pp-megamoe`。
- 只从旧 `iaas_main` 保留构建相关能力；旧 `iaas_main` 的模型、运行时、调度、kernel、connector、API 逻辑不保留。
- Mooncake 保持当前 vLLM Dockerfile 中已有 KV connector/Mooncake 处理方式，不强制改为 Helm chart 中的 `mooncake-transfer-engine-cuda13==0.3.11`。
- servingkit Helm chart 不直接迁入本仓库；本任务在 vLLM 仓库新建一份参考其 P/D 形态的部署模板，但必须移除 runtime hotfix 和运行时安装逻辑。模型准备例外：部署模板应通过同一个新构建 vLLM 镜像中的 `oniond download model ... --turbo --dir ...` 做幂等模型下载，不能在 Pod 启动时安装 Onion 或其他代码库。
- 不新增、不恢复 `scripts/ci/check_byteiaas_dsv4_runtime.py`，也不在镜像构建流程补充 import/CLI smoke；构建后只做镜像构建结果、部署渲染、真实服务路径和 benchmark 验证。
- vLLM runtime 源码逻辑保持 fork 基线；本任务不得用 `vllm/` Python runtime fallback、模型逻辑、算子调用逻辑或调度逻辑修改来绕过部署失败。明确禁止在 `vllm/_custom_ops.py` 或其它 `vllm/**` runtime Python 文件中加入 `_moe_C` 到 `_moe_C_stable_libtorch` 的 `ImportError` fallback。vLLM 源码相关改动只允许解决构建过程暴露的问题，例如 Dockerfile、workflow、`setup.py`/CMake/package-data、wheel extraction 或扩展产物命名/打包问题；如需触碰 `csrc`，范围只能是扩展构建入口、目标导出或 package artifact 生成，不能改变算子语义。例外：用户已在 2026-06-30 明确批准按 upstream main 对齐 `vllm/_custom_ops.py::topk_hash_softplus_sqrt`，仅删除 `wangyicong52` fork 提交 `f7c4c621d` 引入的 `import vllm._moe_C` hard import；不得扩展成其它 runtime 逻辑改动。
- Kubernetes 目标环境固定为 `dev-cluster`，通过 `/data00/home/hanhan.hank/workspace/env/bin/envctl` 访问；当前只读验证 `envctl validate dev-cluster` 通过。
- dev-cluster GPU 工作必须先通过 workspace-env GPU Permit Queue 获取 permit；本计划默认 namespace 为 `vllm-dsv4-flash-pd`，release 为 `dsv4-flash-pd`。servingkit 模板中 `global.gpuCount=8` 会同时用于 prefill 和 decode 的 `nvidia.com/gpu` request/limit，因此本计划 GPU 总量默认 `16`，且必须选择两台不同的 8-GPU 节点分别承载 prefill 和 decode。
- benchmark 默认使用 evalscope；若 evalscope 不可安装或不能满足该请求，执行者必须停止并报告 blocker，不得擅自切换自定义 harness。
- 用户已在 2026-06-29 本线程同意本计划内的远端备份、备份分支保护规则、远端 `iaas_main` 更新、workflow 发布镜像、以及 chart 已出现外部 wheel URL 使用；执行阶段若命令、目标仓库、分支名、外部 URL 或发布范围偏离本计划，必须重新确认。

---

## Context Summary

- servingkit Helm chart 使用 `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe` 作为 runtime overlay，因此新的源码基线应直接使用该 fork，而不是把 fork 源码拆 port 到旧 `iaas_main`。
- servingkit `vllm/deepseek/deepseek-v4-flash-pd` 当前模板中 `global.gpuCount: 8`，prefill 和 decode 均通过 `nvidia.com/gpu: {{ .Values.global.gpuCount }}` 请求 8 张 GPU；现有 values 中 prefill/decode 都是 `replicas: 1`，prefill 使用 `dataParallelSize: 8`，decode 使用 `dataParallelSize: 8`，因此本迁移部署的 `1P1D` 必须跨两个不同节点，而不是同节点尝试调度 16 卡。
- 旧 `origin/iaas_main` 当前已知 SHA：`1ad5c27d4`，包含 ByteIAAS 日构建 workflow、Volcengine CR publish workflow、CUDA 13.0.2 image build、`INSTALL_KV_CONNECTORS=true`、`docker/byteiaas-openai-devel.Dockerfile` 和 image tag 脚本。
- fork 当前已知 SHA：`cde7799cc`，包含 DSV4 Flash/Mooncake/PP/MegaMoE 相关源码变化；它缺少 ByteIAAS release workflow 和 `docker/byteiaas-openai-devel.Dockerfile`。
- 关键迁移方向已经变更为：fork 源码是主线，旧 `iaas_main` 只作为构建能力来源。

## Owner And Write Scope

- Owner：当前执行 agent，执行时需在 `bytedance-iaas/vllm` 的干净工作区操作。
- Claimed write scope：
  - `.github/workflows/byteiaas-release-dev.yml`
  - `.github/workflows/byteiaas-release.yml`
  - `.github/workflows/_byteiaas-build-and-publish-image.yml`
  - `.github/workflows/_byteiaas-build-wheel.yml`
  - `scripts/ci/get_byteiaas_image_tag.py`
  - `docker/byteiaas-openai-devel.Dockerfile`
  - `docker/Dockerfile` 中 ByteIAAS image build 必需的最小构建参数
  - `setup.py`、`CMakeLists.txt`、`cmake/**` 中与 wheel/build/package artifact 直接相关的最小修正
  - `examples/deployment/deepseek-v4-flash-pd/**`
  - `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build*.md`
- Explicitly out of write scope：
  - vLLM 旧 `iaas_main` 业务逻辑回拷
  - `vllm/**` Python runtime 逻辑修改，例如 import fallback、模型/调度/算子调用路径修改
  - `infcp/servingkit` 仓库中的 Helm chart
  - production deployment config
  - unrelated GitHub Actions and Buildkite cleanup

## Milestones

### M1: 远端备份原始 `iaas_main`

- [x] 按命令引用 `C1` fetch 最新 `origin/iaas_main` 和 fork ref。
- [x] 按命令引用 `C2` 创建远端 GitHub 备份分支 `backup/iaas_main-YYYYMMDD`，本次默认 `backup/iaas_main-20260629`。
- [x] 按命令引用 `C2A` 为 `backup/iaas_main-*` 添加 GitHub 保护规则。
- [x] 在进展日志记录旧 `iaas_main` SHA、远端备份分支、`git ls-remote` 证据。
- Acceptance: `git ls-remote origin refs/heads/backup/iaas_main-20260629` 返回的 SHA 等于执行时的 `origin/iaas_main` SHA；GitHub ruleset 覆盖 `refs/heads/backup/iaas_main-*`，至少禁止 deletion 和 non-fast-forward。

### M2: 以 fork 创建新集成分支

- [ ] 按命令引用 `C3` 从 `wangyicong52/dev/dsv4-mooncake-pp-megamoe` 创建 `codex/vllm-dsv4-fork-base-byteiaas-build`。
- [ ] 确认工作区源码来自 fork SHA，而不是旧 `iaas_main`。
- [ ] 在进展日志记录 fork SHA、分支名、非目标说明。
- Acceptance: `git merge-base --is-ancestor <fork_sha> HEAD` 成功，且 `git diff --name-only <fork_sha>...HEAD` 仅出现计划内 ByteIAAS 构建文件。

### M3: 从旧 `iaas_main` 回拷 ByteIAAS 构建能力

- [ ] 按命令引用 `C4` 回拷 ByteIAAS workflow、tag 脚本和 `byteiaas-openai-devel` Dockerfile。
- [ ] 只对 `docker/Dockerfile` 做最小构建兼容 edits；不要整文件恢复旧 `iaas_main` Dockerfile。
- [ ] 保持 fork 源码逻辑；不回拷 `vllm/`、`csrc/`、`cmake/`、`vllm/models/` 等旧源码逻辑。
- [ ] 在进展日志记录每个被回拷文件的理由。
- Acceptance: `git diff --name-status <fork_sha>...HEAD` 中源码逻辑文件没有旧 `iaas_main` 回拷痕迹。

### M4: 补齐 fork-base image 的最小依赖能力

- [ ] Mooncake：保持 fork/current Dockerfile 的 `INSTALL_KV_CONNECTORS` 和 optional `MOONCAKE_WHEEL_*` 处理，不强制 pin chart 版本。
- [ ] DeepEP：复用 fork/current Dockerfile 中社区 DeepEP build/install stage；如 workflow build 未启用，补 build args，不 fork DeepEP。
- [ ] DeepGEMM：保留 fork 中的 DeepGEMM/MegaMoE 源码路径；如 image runtime 仍需要 `wangyicong52/DeepGEMM` wheel，按命令引用 `C6` 添加构建或安装路径。
- [ ] `vllm-router`：按社区包安装，不 fork。
- [ ] Onion：同一个新构建 vLLM 镜像必须提供 `oniond`；若 base image 中不存在，按 `onion-ai-data` skill 在 Dockerfile 中添加 Volcengine extra-tools apt source 并安装 `onion-ai-data`，然后 build-time `command -v oniond` 校验。
- [ ] 不新增、不恢复 `scripts/ci/check_byteiaas_dsv4_runtime.py`；不在构建流程补充 import/CLI smoke。
- Acceptance: Dockerfile/workflow 从结构上把 DeepEP、DeepGEMM、Mooncake/KV connector、`vllm-router` 和 Onion CLI 能力放进同一个新构建 vLLM 镜像；不依赖部署时 hotfix 或运行时安装。

### M5: 验证 workflow 并构建镜像

- [x] 按命令引用 `C9A` 撤销 runtime 源码 fallback 路线：取消或忽略 run `28414886195`，并从待发布分支中移除 `vllm/_custom_ops.py` fallback 修改；后续只允许 build/package 层修复 `_moe_C` artifact 问题。
- [x] 按命令引用 `C7` 做 YAML/actionlint 和 tag script 验证。
- [ ] 按命令引用 `C8` 做本地或 build-node Docker build；不做镜像内 import/CLI smoke。
- [ ] 按命令引用 `C9` 或 ByteIAAS workflow 构建并发布 openai/openai-devel 镜像，记录实际 image tag/digest。
- [ ] 更新主计划 current status 和进展日志。
- Acceptance: 待发布分支不包含 `vllm/_custom_ops.py` 或其它 vLLM runtime Python fallback 修改；至少产出一个可用于 deployment values 的 image tag/digest；若本地 Docker 不可用，则 ByteIAAS workflow 成功并输出 tag/digest。

### M7: 编写无 runtime hotfix/install 的 DSV4 P/D 部署模板

- [ ] 在 `examples/deployment/deepseek-v4-flash-pd/` 创建 vLLM-owned Helm/example deployment。
- [ ] 参考 servingkit `vllm/deepseek/deepseek-v4-flash-pd` 的 P/D 形态、router 参数、StormService/Service/ConfigMap 结构和 values 命名。
- [ ] 继续使用 servingkit 现有实现中的 `StormService` 作为 prefill/decode workload 形态；不改写为 StatefulSet、Deployment 或自定义控制逻辑。
- [ ] 删除或不引入所有 runtime hotfix/install 路径：`runtimePatch`、`git clone`、`pip install`、`install_deepgemm_wheel`、`ensure_pip_package`、wheel download、runtime DeepEP build、runtime Mooncake install、runtime `vllm-router` install。
- [ ] values 只接受 image tag/digest、model path、参数化 node placement、P/D/router shape、env、resources、ports、hostPath/hostNetwork 等部署参数；执行部署时再填写实际节点，`prefill.hostNetwork`、`decode.hostNetwork`、`router.hostNetwork` 保留 servingkit 现状并默认开启。
- [ ] 部署形态固定为 `1P1D`：`stormService.replicas=1`、`prefill.replicas=1`、`decode.replicas=1`、`router.replicas=1`，且 `global.gpuCount=8`。prefill 与 decode 各自请求 8 张 GPU，执行时必须填写非空且不同的 `prefill.nodeAffinity.values[0]` 和 `decode.nodeAffinity.values[0]`；router 默认跟随 decode 节点，除非执行时明确给出第三个节点。
- [ ] `C13` render 和 `C15` deploy 必须在 `PREFILL_NODE` 或 `DECODE_NODE` 缺失、两者相同、或 `GLOBAL_GPU_COUNT` 不是 `8` 时失败；`C14` 必须在部署前检查这两个节点均存在且 allocatable `nvidia.com/gpu` 至少为 8。
- [ ] BS512/1.5k output throughput 是该 `1P1D` router-path 结果，全部 output 来自单个 decode 节点，不是多 decode 聚合吞吐。
- [ ] values 增加 Onion 模型准备参数：`onion.enabled=true`、`onion.model=DeepSeek-V4-Flash`、`onion.dir=/data01`；模板用 initContainer 或等价启动前步骤执行 `oniond download model "${onion.model}" --turbo --dir "${onion.dir}"`，initContainer image 必须是同一个 `global.image`，已有模型时由 Onion 幂等跳过。
- [ ] 按命令引用 `C13` render 并 grep forbidden patterns。
- Acceptance: rendered manifest 使用新构建 image，prefill/decode 由 `StormService` 管理，`1P1D` replica 形态明确，prefill/decode 均渲染为 `nvidia.com/gpu: 8`，并带有不同节点的 required nodeAffinity；router 默认渲染到 decode 节点；模型数据由 Onion 准备；prefill/decode/router 命令没有运行时安装和 hotfix，P/D 形态与 servingkit chart 等价。

### M8: 在 `dev-cluster` 部署

- [ ] 按命令引用 `C14` 解析 env root、验证 `dev-cluster`、注册 workspace-env session、申请 16 GPU permit，并确认执行者选定的 `PREFILL_NODE` 与 `DECODE_NODE` 是两个不同且各自 allocatable GPU 至少为 8 的节点。
- [ ] 在 `C14` 中确认 `dev-cluster` 已存在 `stormservices.orchestration.aibrix.ai` CRD；如不存在，停止并记录 blocker，不在本任务中安装 Aibrix/StormService 控制面。
- [ ] 按命令引用 `C15` 创建 namespace、注册 Helm release、`helm upgrade --install`。
- [ ] 按命令引用 `C16` 收集 evidence ladder：render、Onion 模型准备日志、模型完整性检查、实际 argv/env、pod-local package/source evidence、稳定 readiness、router `/v1/models` 和一次真实 completion。
- [ ] 如果资源 Pending 或 readiness/real request 失败，按 workspace-env 规则清理本任务创建的资源，记录 blocker。
- Acceptance: Onion 模型准备成功或幂等跳过且模型完整性检查通过；prefill pod 和 decode pod 分别落在 `PREFILL_NODE` 与 `DECODE_NODE` 两个不同节点，且各自请求 8 张 GPU；prefill/decode/router 通过真实 router path 可生成非空输出，且没有 runtime hotfix/install 证据。

### M9: 使用 evalscope benchmark 64k/1 TTFT 和 cache-hit decode BS512/1.5k output

- [ ] 按命令引用 `C17` 准备 evalscope 环境；如果 evalscope 不可用或安装受阻，停止并记录 blocker。
- [ ] 按命令引用 `C18` 跳过 Prometheus 部署，仅采集 pod-local `/metrics` 头部、日志和 skipped note；最终 summary 必须明确 Prometheus skipped by user。
- [ ] 按命令引用 `C19` 先运行真实服务路径 smoke 和 warmup。
- [ ] 按命令引用 `C20` 运行 64k input / 1 output TTFT measured run。
- [ ] 按命令引用 `C21` 用固定 `--seed 42` 预热 prefix cache，再运行全命中 cache 的 decode BS512 output throughput measured run；固定 output length 为 1.5k，即 1536 tokens。
- [ ] 保存 raw evalscope output、timestamps、serving logs、pod-local metrics heads、rendered manifests 和 Markdown summary 到 artifact path。
- Acceptance: 两个 measured run 均有开始/结束时间、exit code、raw output、artifact 路径、TTFT/throughput 摘要；性能 gate 看 Avg，不看 P50/P95/P99；64k/1 Avg TTFT 必须小于 10s；BS512/1.5k evalscope overall output token throughput 必须达到 14000 tokens/s 以上；该 throughput 口径为 `1P1D` router-path，总 output 来自单个 decode 节点；summary 明确 Prometheus skipped by user，不能声称有完整服务侧 monitoring 诊断；若 run invalid 或性能未达阈值，必须明确 invalid/blocker 原因。

### M10: 使用已授权审批更新远端 `iaas_main`

- [ ] 确认 M5/M7/M8/M9 已完成，或任何未完成项都已在进展日志中记录为明确可接受 blocker。
- [ ] 可接受 blocker 仅限外部环境或资源问题，例如 GPU permit 长时间排队、`dev-cluster` 临时资源不足、CR/image pull 临时失败、Onion 模型源临时不可用；这些 blocker 只能进入人工发布决策，不能自动视为通过。
- [ ] 不可接受 blocker 包括 render/config 表达错误、镜像缺依赖、Onion init 或模型完整性失败、vLLM 启动失败、router real request 失败、KV transfer 错误、DeepGEMM/DeepEP/Mooncake import 或 runtime 错误；出现这些问题时不得更新远端 `iaas_main`。
- [ ] benchmark 跑通但性能未达阈值时也不得更新远端 `iaas_main`；本计划性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，`1P1D` router-path BS512/1.5k evalscope overall output token throughput >= 14000 tokens/s。
- [ ] 确认远端备份分支仍指向原始 `iaas_main` SHA。
- [ ] 使用本线程已有授权更新远端 `iaas_main`；无需再次停下来请求审批，除非目标仓库、备份分支、源 SHA、验证门槛或更新方式偏离本计划。
- [ ] 按命令引用 `C10` 使用 `--force-with-lease` 或仓库允许的受保护分支流程更新 `iaas_main`。
- [ ] 若 branch protection 拒绝直接更新，按命令引用 `C11` 创建 PR 或临时迁移分支，并在进展日志记录 blocker。
- Acceptance: `origin/iaas_main` 指向通过构建、部署和 benchmark gate 的 fork-base integration SHA，且远端备份分支仍可追溯原始 SHA。

## Acceptance Criteria

- GitHub 远端存在原始 `iaas_main` 备份分支，且 SHA 与替换前 `origin/iaas_main` 一致。
- 新 `iaas_main` 基线来自 `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe`。
- 旧 `iaas_main` 只回拷 ByteIAAS 构建、日构建、镜像发布、tag 生成相关文件。
- 旧 `iaas_main` 的 vLLM 业务源码逻辑没有被回拷。
- Mooncake 仍使用当前 Dockerfile 中已有处理方式，不被强制改成 Helm chart 的版本。
- ByteIAAS dev image workflow 能解析 tag、构建 openai/openai-devel image，并保留 `INSTALL_KV_CONNECTORS=true`。
- vLLM 仓库包含一份无 runtime hotfix/install 的 DSV4 Flash P/D 部署模板。
- 部署模板固定表达 `1P1D`、`global.gpuCount=8`、prefill/decode 各 8 GPU，且强制 P/D 节点参数非空并不同；router 默认跟 decode 节点。
- 部署模板包含 Onion 模型准备路径，已有模型时幂等跳过，模型不完整时不得继续宣称服务验证通过。
- `dev-cluster` 部署使用新镜像，真实 router path 可生成非空输出。
- benchmark 完成 64k input / 1 output TTFT 和 cache-hit decode BS512 / 1.5k output throughput 两个 measured run，并保存 raw artifacts、pod-local metrics heads 与 Markdown summary。

## Approval Forecast

- 已授权：用户已在 2026-06-29 本线程明确同意本计划内全部审批项。
- 可直接执行的已授权动作：
  - 在 `bytedance-iaas/vllm` 远端创建 `backup/iaas_main-20260629`，指向执行时当前 `origin/iaas_main` SHA。
  - 为 `backup/iaas_main-*` 添加 GitHub 保护规则。
  - 在 image、deployment、router smoke 和 benchmark gate 之后，将远端 `iaas_main` 更新为 fork-base integration 分支。
  - 触发 ByteIAAS dev image workflow 并发布 openai/openai-devel 镜像到 Volcengine CR。
  - 使用 Helm chart 中已出现的 `wangyicong52/DeepGEMM` release wheel URL。
  - 在 `dev-cluster` 创建 namespace、Helm release、GPU workload、port-forward/evalscope benchmark 资源，并在完成后清理。
- 仍需重新确认的情况：目标仓库不是 `bytedance-iaas/vllm`、备份分支名不是 `backup/iaas_main-YYYYMMDD`、外部 URL 不是 Helm chart 已出现 URL、需要生产部署、或需要本计划以外的 force push。

## Risks And Fallbacks

- Branch protection 可能拒绝直接替换 `iaas_main`：fallback 是保留远端备份，推送 fork-base integration 分支，并通过 PR 或管理员流程迁移。
- fork Dockerfile 与旧 ByteIAAS workflow 参数不完全兼容：fallback 是只补 workflow 需要的 build args，不整文件恢复旧 Dockerfile。
- DeepGEMM fork wheel 无可复现源码 ref：fallback 是使用 chart 中明确出现且本线程已授权的 release wheel URL；不退回社区 DeepGEMM。
- Mooncake/KV connector 在服务路径失败：先记录 render、argv/env、package/source 和日志证据；不直接 pin chart 版本，除非用户再次确认改变策略。
- `_moe_C` import 类问题不得通过 `vllm/` runtime Python fallback 解决；后续只能从 build/package artifact 角度处理，例如确认 wheel 应包含哪个扩展模块、CMake/setup 是否生成/抽取正确 artifact、Dockerfile 是否安装了正确 wheel。若 build/package 层无法解决，应停止并报告 blocker，而不是改 vLLM runtime 逻辑。
- Docker/GPU 环境不可用：本地完成静态和 workflow 验证，将 image build 转到 ByteIAAS workflow；GPU deployment/benchmark 等待 workspace-env permit 或报告 scheduler-visible blocker。
- 本轮按用户要求跳过 Prometheus：最终 summary 只能报告 evalscope 结果、pod-local metrics heads 和日志证据，不能声称有完整 Prometheus measured-window 诊断。
- cache-hit decode BS512 结果可能被 router dispatch、cache lookup、KV/bootstrap、decode admission 或 scheduler queue 限制；最终摘要必须区分 TTFT、TPOT/ITL、output throughput 和 queue/admission 现象。
- final gate 允许人工接受的 blocker 必须是外部环境或资源类；代码正确性、镜像内容、部署模板表达、真实请求路径、KV/Mooncake/DeepEP/DeepGEMM runtime 失败都必须阻止远端 `iaas_main` 更新。
- benchmark 性能未达阈值也必须阻止远端 `iaas_main` 更新；本计划性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，`1P1D` router-path BS512/1.5k evalscope overall output token throughput >= 14000 tokens/s。

## Current Status

- 当前分支：`codex/vllm-dsv4-fork-base-byteiaas-build`，基于 fork SHA `cde7799cc66c5a4cb349156a3ca3228f9798dbc9`。
- M1-M4 已完成：远端备份分支 `backup/iaas_main-20260629` 指向原 `origin/iaas_main` SHA `1ad5c27d41aa2b04d61a13c2adfe8d3db6ae2b16`，GitHub ruleset `Protect backup iaas_main branches` 已 active；ByteIAAS workflow、tag 脚本、`byteiaas-openai-devel` Dockerfile、Dockerfile 中 `vllm-router` 与 Onion CLI 构建能力已落到 fork-base 分支。
- M5 需重新打开：本地 Docker daemon 不可访问，已改用 ByteIAAS workflow；第三次 run `28389076984` 成功产出镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606300110-cu130`，digest `sha256:574c3dc2023be9300df8e699994798f76e3f048bff81f3e6719e8726197de113`，但该镜像在 M8 启动失败于缺少 `vllm._moe_C`，因此不能进入 M9/M10。后续只允许用 build/package artifact 层修复该问题，并重新构建新镜像。
- M7 部署模板已创建：`examples/deployment/deepseek-v4-flash-pd/` 使用 `StormService` 表达 `1P1D`，同一新镜像负责 vLLM、`vllm-router` 和 Onion 模型准备；chart 新增 Helm validation，强制 `global.gpuCount=8`、prefill/decode nodeAffinity 非空且 disjoint。
- M8 首轮部署已清理：`dev-cluster` preflight 通过，StormService CRD 存在；首轮 16 GPU permit `3813ef32-5e13-42b7-9a1c-a36aa463dd5b` 已释放。首轮部署确认 prefill 落到 `192.168.1.148` 且请求 8 GPU，decode 落到 `192.168.1.186` 且请求 8 GPU，router 跟随 decode 节点且不请求 GPU；P/D 不同节点约束成立。该镜像启动失败于 `No module named 'vllm._moe_C'`，不得进入 M9 或 M10。
- 用户在 2026-06-30 明确收窄范围：不接受 `vllm/_custom_ops.py` 中 `_moe_C` 到 `_moe_C_stable_libtorch` 的 runtime fallback；vLLM 源码修改只允许构建过程中遇到的问题。此前提交 `51b135cef854e6d72cb704068644c52d047706e5` 和 workflow run `28414886195` 被标记为无效路线，不得作为后续部署/benchmark/更新 `iaas_main` 的依据。详见进展日志 `P26`。
- C9A 已完成：workflow run `28414886195` 已取消，`vllm/_custom_ops.py` 已恢复为 fork baseline hard import `vllm._moe_C`，且 `uv run --no-project python -m py_compile vllm/_custom_ops.py` 通过。详见进展日志 `P28`。
- C9B 已完成：用户指出 build-side `_moe_C` rename 路线异常后，已取消 workflow run `28418542564`，并将 `CMakeLists.txt`、`setup.py`、`csrc/libtorch_stable/moe/torch_bindings.cpp` 恢复为 upstream main/fork baseline 的 `_moe_C_stable_libtorch` build artifact 命名。上游对比证明 upstream main 的自洽路径是：构建/import `vllm._moe_C_stable_libtorch`，但注册 `torch.ops._moe_C` namespace，且 `topk_hash_softplus_sqrt` 不再 hard import `vllm._moe_C`。详见进展日志 `P30`。
- 用户已批准按 upstream main 对齐 `topk_hash_softplus_sqrt`：删除 `f7c4c621d` 引入的 hard import，不做 fallback、不改 build artifact。详见进展日志 `P31`。
- M9/M10 尚未执行；性能不达标或服务路径未跑通时不得更新 `iaas_main`。

## Next Action

按 upstream main 对齐 `vllm/_custom_ops.py::topk_hash_softplus_sqrt`，删除 fork 残留的 `import vllm._moe_C` hard import；静态验证、提交推送后，按命令引用 `C9` 重启 ByteIAAS workflow 构建新镜像。
