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
- servingkit Helm chart 不直接迁入本仓库；本任务在 vLLM 仓库新建一份参考其 P/D 形态的部署模板，但必须移除 runtime hotfix 和运行时安装逻辑。部署语义必须以当前已 fetch 的 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 为基准，覆盖 P/D role、router 参数、端口、TP/PP/DP、KV transfer role、DeepEP/MegaMoE/MTP 参数、Service/StormService/ConfigMap 结构和 values 命名；不得因为调试失败随意发散。模型准备例外：部署模板应通过同一个新构建 vLLM 镜像中的 `oniond download model ... --turbo --dir ...` 做幂等模型下载，不能在 Pod 启动时安装 Onion 或其他代码库。
- 不新增、不恢复 `scripts/ci/check_byteiaas_dsv4_runtime.py`，也不在镜像构建流程补充 import/CLI smoke；构建后只做镜像构建结果、部署渲染、真实服务路径和 benchmark 验证。
- vLLM runtime 源码逻辑保持 fork 基线；本任务不得用 `vllm/` Python runtime fallback、模型逻辑、算子调用逻辑或调度逻辑修改来绕过部署失败。明确禁止在 `vllm/_custom_ops.py` 或其它 `vllm/**` runtime Python 文件中加入 `_moe_C` 到 `_moe_C_stable_libtorch` 的 `ImportError` fallback。vLLM 源码相关改动只允许解决构建过程暴露的问题，例如 Dockerfile、workflow、`setup.py`/CMake/package-data、wheel extraction 或扩展产物命名/打包问题；如需触碰 `csrc`，范围只能是扩展构建入口、目标导出或 package artifact 生成，不能改变算子语义。例外：用户已在 2026-06-30 明确批准按 upstream main 对齐 `vllm/_custom_ops.py::topk_hash_softplus_sqrt`，仅删除 `wangyicong52` fork 提交 `f7c4c621d` 引入的 `import vllm._moe_C` hard import；不得扩展成其它 runtime 逻辑改动。
- Kubernetes 目标环境固定为 `dev-cluster`，通过 `/data00/home/hanhan.hank/workspace/env/bin/envctl` 访问；当前只读验证 `envctl validate dev-cluster` 通过。
- dev-cluster GPU 工作必须先通过 workspace-env GPU Permit Queue 获取 permit；本计划默认 namespace 为 `vllm-dsv4-flash-pd`，release 为 `dsv4-flash-pd`。servingkit 模板中 `global.gpuCount=8` 会同时用于 prefill 和 decode 的 `nvidia.com/gpu` request/limit，因此本计划 GPU 总量默认 `16`，且必须选择两台不同的 8-GPU 节点分别承载 prefill 和 decode。
- benchmark 默认使用 evalscope；需要按命令引用额外对比 vLLM 自带 `vllm bench serve` / `vllm.benchmarks.serve` 与 evalscope 的口径差异，但在对比证明等价前不得擅自切换 gate harness。vLLM 自带压测必须在同一个新构建 vLLM 镜像启动的容器内执行，不能用本地工作站 Python 环境；该容器不请求 GPU，不做运行时安装、代码 clone 或 hotfix，通过集群内 router Service 发请求，结束后删除。
- evalscope 与 vLLM bench 结果出现显著差异时，必须先按命令引用 `C21W` 做差异分析；分析对象包括请求构造、prefix cache 命中语义、client 网络路径、并发/限流模型、超时与失败处理、统计口径、Prometheus running/waiting/output TPS、Mooncake/KV 错误窗口和服务状态延续。`C21W` 未完成或未能解释差异前，不得把 vLLM bench 的 BS512 成功结果作为替代 gate；如离线 artifacts 不能解释原因，才执行 `C21X` 重新部署做最小配对复现实验。
- `C21W` 若解释了 evalscope 与 vLLM bench 的差异但根因落在 evalscope BS512 触发的 Mooncake/KV transfer failure，则必须按命令引用 `C23` 做离线根因初筛；`C23` 只分析已有日志、render、servingkit 对齐状态和 vLLM/MooncakeConnector 代码路径，不修改源码、不修改部署语义、不创建 GPU workload。只有 `C23` 证据不足时，才规划后续 live diagnostic/repro。
- 2026-07-01 用户批准继续部署并分析根因，且允许为诊断修改调试代码并推送。新增 live 诊断必须先按 `C24` 使用当前成功构建镜像做无代码改动复现和偶现性判断：同一部署语义、同一节点约束、同一 tokenizer/prefix/output，至少覆盖 evalscope repeated run 与 vLLM bench cross-check，并在每个失败候选后重启 P/D/router，避免坏状态污染下一轮。如果 `C24` 不能定位根因，才进入 `C25` 添加最小调试代码、推送诊断分支、触发 dev image 构建并用调试镜像复测。调试代码必须只增加可控日志/计数/上下文证据，不改变调度、KV transfer、Mooncake 参数、runtime fallback 或性能语义；调试分支和镜像不得用于更新 `iaas_main`。
- C24 之后如继续分析 evalscope 与 vLLM bench 的性能差异，必须按命令引用 `C26` 做更细粒度 benchmark deep dive：对比 TTFT、TPOT/ITL、request start span、completion span、prompt payload/client path、Prometheus running/waiting/output TPS，并重点检查 speculative decoding 接受率或可用代理证据。当前离线 C26 结论是 `client-path + TTFT/admission dominated; speculative-acceptance-not-primary`：evalscope attempt 2 的 `Spec. Accept Rate=0.7707`、`Decoded Tok/Iter=4.3615`，vLLM bench JSON 未保存 spec accept/draft 字段，现有 chunk/ITL proxy 不支持 evalscope 接受率更差的判断。若后续要给出 definitive 结论，必须在同一 live 窗口采集 `vllm:spec_decode_*` Prometheus 指标，并优先运行 in-cluster evalscope no-GPU Pod 直连 router Service，隔离本地 port-forward 和大 prompt 上传路径。
- benchmark 必须部署 servingkit `origin/hanhan_dev:llm-serving-monitoring` 下 `llm-serving-monitoring` Helm chart 的最小 Prometheus 监控；使用本任务专属 namespace/release `vllm-dsv4-flash-pd-monitoring` / `dsv4-flash-pd-monitoring`，不得接管或清理已有共享 `vllm-monitoring`。C21/C21A 前后必须用 decode worker 的 `vllm:num_requests_running` 和 `vllm:num_requests_waiting` 检查实际 running BS。router `/metrics` 当前对 Prometheus GET 返回 `405 Method Not Allowed`，不作为 monitoring gate；router 可用性仍由 C16 `/health`、`/v1/models` 和真实 completion smoke 证明。若监控显示服务端达不到目标 running BS，则不得继续测试高于实际 running capacity 的候选。
- 用户已在 2026-06-29 本线程同意本计划内的远端备份、备份分支保护规则、远端 `iaas_main` 更新、workflow 发布镜像、以及 chart 已出现外部 wheel URL 使用；执行阶段若命令、目标仓库、分支名、外部 URL 或发布范围偏离本计划，必须重新确认。

---

## Context Summary

- servingkit Helm chart 使用 `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe` 作为 runtime overlay，因此新的源码基线应直接使用该 fork，而不是把 fork 源码拆 port 到旧 `iaas_main`。
- servingkit `vllm/deepseek/deepseek-v4-flash-pd` 当前参考 SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 中 `global.gpuCount: 8`，prefill 和 decode 均通过 `nvidia.com/gpu: {{ .Values.global.gpuCount }}` 请求 8 张 GPU；现有 values 中 prefill/decode 都是 `replicas: 1`，prefill 语义为 `kv_producer`、`port: 8000`、`dataParallelSize: 1`、`tensorParallelSize: 4`、`pipelineParallelSize: 2`、不启用 expert parallel，decode 语义为 `kv_consumer`、`port: 8001`、`dataParallelSize: 8`、`cpKvCacheInterleaveSize: 256`、`moeBackend: deep_gemm_mega_moe`、`enablePrefixCaching: true`、MTP speculative config，router 默认关闭 service discovery 并用静态 `--prefill/--decode` 指向 P/D 节点。因此本迁移部署的 `1P1D` 必须跨两个不同节点，而不是同节点尝试调度 16 卡。
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
- 非诊断目的的 `vllm/**` runtime 逻辑修改；C25 例外仅允许添加可开关的临时调试日志/计数，不允许改变行为
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
- [x] 按命令引用 `C9` 或 ByteIAAS workflow 构建并发布 openai/openai-devel 镜像，记录实际 image tag/digest；本次由用户在 run `28442949331` 完成，headSha `7186cf328963d12daabe8ee47087a29111c0cb75`。
- [x] 按命令引用 `C9D` 检查候选 `openai-devel` 镜像 manifest：`iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130`。
- [x] 更新主计划 current status 和进展日志。
- Acceptance: 待发布分支不包含 `vllm/_custom_ops.py` 或其它 vLLM runtime Python fallback 修改；至少产出一个可用于 deployment values 的 image tag/digest；若本地 Docker 不可用，则 ByteIAAS workflow 成功并输出 tag/digest。

### M7: 编写无 runtime hotfix/install 的 DSV4 P/D 部署模板

- [ ] 在 `examples/deployment/deepseek-v4-flash-pd/` 创建 vLLM-owned Helm/example deployment。
- [ ] 参考 servingkit `vllm/deepseek/deepseek-v4-flash-pd` 当前 SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 的 P/D 形态、router 参数、StormService/Service/ConfigMap 结构和 values 命名。
- [ ] 继续使用 servingkit 现有实现中的 `StormService` 作为 prefill/decode workload 形态；不改写为 StatefulSet、Deployment 或自定义控制逻辑。
- [ ] 删除或不引入所有 runtime hotfix/install 路径：`runtimePatch`、`git clone`、`pip install`、`install_deepgemm_wheel`、`ensure_pip_package`、wheel download、runtime DeepEP build、runtime Mooncake install、runtime `vllm-router` install。
- [ ] values 只接受 image tag/digest、model path、参数化 node placement、P/D/router shape、env、resources、ports、hostPath/hostNetwork 等部署参数；执行部署时再填写实际节点，`prefill.hostNetwork`、`decode.hostNetwork`、`router.hostNetwork` 保留 servingkit 现状并默认开启。
- [ ] 部署形态固定为 `1P1D`：`stormService.replicas=1`、`prefill.replicas=1`、`decode.replicas=1`、`router.replicas=1`，且 `global.gpuCount=8`。prefill 与 decode 各自请求 8 张 GPU，执行时必须填写非空且不同的 `prefill.nodeAffinity.values[0]` 和 `decode.nodeAffinity.values[0]`；router 默认跟随 decode 节点，除非执行时明确给出第三个节点。
- [ ] chart 默认值必须与 servingkit 当前 SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 的 P/D/router 语义保持一致；只允许这些有意差异：`global.image` 留空并由执行时填入新 ByteIAAS 镜像；TOS 下载替换为 Onion initContainer；节点 IP 改为参数化；runtime hotfix/install 删除。`prefill.args.noAsyncScheduling` 必须保持 `false`；`decode.args.maxNumSeqs` 是每个 worker 的 admission 值，必须保持 servingkit 当前语义 `96`，不得为了 BS512/1.5k benchmark 覆盖为 `512`。不得显式设置 `vllm.maxModelLen=66000`，应沿用 servingkit 当前 `maxModelLen: null`，rendered command 中不出现 `--max-model-len`。
- [ ] `C13` render 和 `C15` deploy 必须在 `PREFILL_NODE` 或 `DECODE_NODE` 缺失、两者相同、或 `GLOBAL_GPU_COUNT` 不是 `8` 时失败；`C14` 必须在部署前检查这两个节点均存在且 allocatable `nvidia.com/gpu` 至少为 8。
- [ ] BS512/1.5k output throughput 是该 `1P1D` router-path 结果，全部 output 来自单个 decode 节点，不是多 decode 聚合吞吐；标准 BS512 压测请求数必须为 `4 * BS = 2048`，以更充分压出 decode 端吞吐。
- [ ] values 增加 Onion 模型准备参数：`onion.enabled=true`、`onion.model=DeepSeek-V4-Flash`、`onion.dir=/data01`；模板用 initContainer 或等价启动前步骤执行 `oniond download model "${onion.model}" --turbo --dir "${onion.dir}"`，initContainer image 必须是同一个 `global.image`，已有模型时由 Onion 幂等跳过。
- [ ] 按命令引用 `C13` render 并 grep forbidden patterns。
- Acceptance: rendered manifest 使用新构建 image，prefill/decode 由 `StormService` 管理，`1P1D` replica 形态明确，prefill/decode 均渲染为 `nvidia.com/gpu: 8`，并带有不同节点的 required nodeAffinity；router 默认渲染到 decode 节点；模型数据由 Onion 准备；prefill/decode/router 命令没有运行时安装和 hotfix；除已列出的 image、Onion、节点参数化、删除 runtime install/hotfix 外，P/D/router 运行参数不得与 servingkit SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 严重发散。

### M8: 在 `dev-cluster` 部署

- [x] 按命令引用 `C14` 解析 env root、验证 `dev-cluster`、注册 workspace-env session、申请 16 GPU permit，并确认执行者选定的 `PREFILL_NODE` 与 `DECODE_NODE` 是两个不同且各自 allocatable GPU 至少为 8 的节点。
- [x] 在 `C14` 中确认 `dev-cluster` 已存在 `stormservices.orchestration.aibrix.ai` CRD；如不存在，停止并记录 blocker，不在本任务中安装 Aibrix/StormService 控制面。
- [x] 按命令引用 `C15` 创建 namespace、注册 Helm release、`helm upgrade --install`。
- [x] 按命令引用 `C16` 收集 evidence ladder：render、Onion 模型准备日志、模型完整性检查、实际 argv/env、pod-local package/source evidence、稳定 readiness、router `/v1/models` 和一次真实 completion。
- [x] 如果资源 Pending 或 readiness/real request 失败，按 workspace-env 规则清理本任务创建的资源，记录 blocker。
- Acceptance: Onion 模型准备成功或幂等跳过且模型完整性检查通过；prefill pod 和 decode pod 分别落在 `PREFILL_NODE` 与 `DECODE_NODE` 两个不同节点，且各自请求 8 张 GPU；prefill/decode/router 通过真实 router path 可生成非空输出，且没有 runtime hotfix/install 证据。

### M9: 使用 evalscope benchmark 64k/1 TTFT 和 cache-hit decode BS512/1.5k output

- [x] 按命令引用 `C17` 准备 evalscope 环境；如果 evalscope 不可用或安装受阻，停止并记录 blocker。
- [x] 按命令引用 `C18` 从 servingkit `origin/hanhan_dev:llm-serving-monitoring` 部署最小 Prometheus 监控 chart 到本任务专属 namespace，确认 prefill/decode 两个 vLLM worker scrape target `up == 1`；router `/metrics` 不可 scrape 时记录证据但不阻塞 running BS gate。
- [x] 按命令引用 `C19` 先运行真实服务路径 smoke 和 warmup。
- [x] 按命令引用 `C20` 运行 64k input / 1 output TTFT measured run。
- [x] 按命令引用 `C21` 用固定 `--seed 42` 预热 prefix cache，再运行全命中 cache 的 decode BS512 output throughput measured run；固定 output length 为 1.5k，即 1536 tokens；请求数必须为 `2048`，即 `4 * BS`。
- [x] 每次 C21/C21A 候选结束、失败或被中断后，按命令引用 `C21M` 查询 Prometheus running BS、waiting 和 output TPS；若实际 decode running BS 明显低于候选 BS，不得继续测试高于 observed capacity 的候选。
- [x] 若 BS512/2048 请求 run 跑不过、出现 KV/Mooncake runtime 错误、或没有有效 Avg/Overall throughput 结果，先按命令引用 `C21R` 重启 P/D/router 服务，再按 `C21A` 在 128-512 之间降档寻找能够完整通过压测的最大已测 BS；每个候选同样使用 `number = 4 * BS`。
- [x] 按命令引用 `C21V` 补充 vLLM 自带 `vllm bench serve` 与 evalscope 的压测口径对比；该步骤必须在同一个新构建 vLLM 镜像启动的容器内执行，至少保存容器内 help、对比说明、一个不高于 observed running BS 的对照运行或容器内无法运行原因。
- [x] 按用户要求补充 vLLM 自带 `vllm bench serve` 的 BS256/BS400/BS512 结果；该补充 sweep 使用同一个新构建 vLLM 镜像、同一个 router path、`number = 4 * BS`，并保存 Prometheus running BS 证据。
- [x] 按命令引用 `C21W` 分析 evalscope 与 vLLM bench 结果差异为什么这么大；先从现有 artifacts 离线分析，不创建 GPU workload，不更新 gate。
- [x] `C21W` 已能解释差异，当前不进入 `C21X`；只有用户明确要切换 gate 或要求同一服务状态配对复现实验时，再按命令引用 `C21X` 重新部署相同 `1P1D` 服务。
- [x] 按命令引用 `C23` 对 evalscope BS512 触发的 Mooncake/KV transfer failure 做离线根因初筛，判断是否属于部署语义发散、metadata/握手不一致、Mooncake/RDMA transfer timeout/descriptor pressure、client burst/服务状态污染，还是证据不足。
- [x] 保存新 C21/C21A/C21M/C21V raw output、timestamps、serving logs、Prometheus query results、rendered manifests 和 Markdown summary 到 artifact path；其中 C21V 必须来自 vLLM 镜像容器内执行结果。
- Acceptance: 两个 measured run 均有开始/结束时间、exit code、raw output、artifact 路径、TTFT/throughput 摘要；性能 gate 看 Avg，不看 P50/P95/P99；64k/1 Avg TTFT 必须小于 10s；BS512/1.5k evalscope overall output token throughput 必须达到 14000 tokens/s 以上；BS512 run 必须使用 2048 总请求；该 throughput 口径为 `1P1D` router-path，总 output 来自单个 decode 节点；summary 必须包含 servingkit monitoring chart 部署、Prometheus running BS evidence、vLLM benchmark script 对比结论；若 BS512 run invalid，必须先重启服务再执行 C21A 记录 128-512 降档 BS 探测结果；若 run invalid、实际 running BS 达不到目标、或性能未达阈值，必须明确 invalid/blocker 原因。新增差异分析 gate：`C21W` 必须产出 `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/summary.md`，明确指出差异主因属于 workload 不等价、client/network 差异、harness 统计口径、服务状态/失败窗口、Mooncake/KV transfer、还是证据不足；证据不足时必须执行或规划 `C21X`。新增 Mooncake/KV failure 初筛 gate：`C23` 必须产出 `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/summary.md`，明确记录失败日志统计、MooncakeConnector 失败代码路径、servingkit SHA 对齐/差异、节点与 rendered command 对比、当前根因假设和是否需要 live diagnostic。

### M10: 使用已授权审批更新远端 `iaas_main`

- [ ] 确认 M5/M7/M8/M9 已完成，或任何未完成项都已在进展日志中记录为明确可接受 blocker。
- [ ] 可接受 blocker 仅限外部环境或资源问题，例如 GPU permit 长时间排队、`dev-cluster` 临时资源不足、CR/image pull 临时失败、Onion 模型源临时不可用；这些 blocker 只能进入人工发布决策，不能自动视为通过。
- [ ] 不可接受 blocker 包括 render/config 表达错误、镜像缺依赖、Onion init 或模型完整性失败、vLLM 启动失败、router real request 失败、KV transfer 错误、DeepGEMM/DeepEP/Mooncake import 或 runtime 错误；出现这些问题时不得更新远端 `iaas_main`。
- [ ] benchmark 跑通但性能未达阈值时也不得更新远端 `iaas_main`；本计划性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，`1P1D` router-path BS512/1.5k evalscope overall output token throughput >= 14000 tokens/s；BS512 gate 必须来自 `number=2048` 且 Prometheus 证明实际 running BS 达到目标的结果，降档 BS 通过只能作为容量诊断，不能替代 BS512 gate。
- [ ] 确认远端备份分支仍指向原始 `iaas_main` SHA。
- [ ] 使用本线程已有授权更新远端 `iaas_main`；无需再次停下来请求审批，除非目标仓库、备份分支、源 SHA、验证门槛或更新方式偏离本计划。
- [ ] 按命令引用 `C10` 使用 `--force-with-lease` 或仓库允许的受保护分支流程更新 `iaas_main`。
- [ ] 若 branch protection 拒绝直接更新，按命令引用 `C11` 创建 PR 或临时迁移分支，并在进展日志记录 blocker。
- Acceptance: `origin/iaas_main` 指向通过构建、部署和 benchmark gate 的 fork-base integration SHA，且远端备份分支仍可追溯原始 SHA。

### M11: Live 复现与 Mooncake/KV 根因诊断

- [x] 按命令引用 `C24` 重新部署当前已成功构建镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，保持 servingkit 对齐语义：`prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`、`1P1D`、P/D 不同 8-GPU 节点。
- [x] `C24` 必须复用 monitoring，先做 C16 real router smoke，再做最小复现矩阵：evalscope BS512/2048 至少两次独立尝试，尝试之间必须重启 P/D/router；如果第一次成功，仍再跑一次 evalscope BS512 判断是否偶现；如果第一次失败，重启后再跑一次判断是否稳定复现；同时跑一个 vLLM bench BS512 cross-check，验证同一 live 服务状态下是否仍和 evalscope 分化。
- [x] `C24` 每次 run 必须保存 timestamps、exit code、raw output、request DB summary、Prometheus running/waiting/output TPS 窗口、P/D/router logs、Mooncake bad-log scan、pod restarts、node/events。
- [ ] 如果 `C24` 显示 evalscope BS512 连续成功且满足 throughput gate，则将此前 failure 标记为偶现候选，但仍需解释两次连续成功的服务状态与历史失败差异；不得直接进入 M10，除非 M9/M11 summary 同时满足 gate 且无新坏日志。
- [x] 如果 `C24` 显示 evalscope BS512 连续失败或一次失败一次成功，先判断是否与节点、服务启动年龄、warmup/cache seed、client 路径、request burst、decode running/waiting 或 Mooncake transfer batch 大小相关；不能直接归因于压测工具。
- [ ] 如果 `C24` 证据不足，按命令引用 `C25` 添加最小调试代码并推送诊断分支；调试代码只允许在 MooncakeConnector producer/consumer transfer 路径记录 batch descriptors、bytes、request count、remote session、role/rank、elapsed、ret、pending/expired request count 和可用 transfer config，不改变 runtime 语义。调试代码必须默认可控，例如通过 `VLLM_DSV4_MOONCAKE_DIAG=1` 或等价 env gate 开启。
- [ ] `C25` 构建调试镜像后复测时，部署模板只允许增加诊断 env，不能更改 Mooncake 版本、IB 设备、TP/PP/DP/EP、maxNumSeqs、maxNumBatchedTokens、maxModelLen、MTP、DeepEP 或 router 语义。
- [x] 诊断结束后按 workspace-env 规则清理 serving/monitoring/benchmark Pod/port-forward，并释放 permit；如保留资源，必须在进展日志和最终摘要中写明 cleanup 命令。
- Acceptance: 产出 `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/summary.md`；summary 明确分类为 `stable-repro`、`intermittent`、`harness-specific`、`node/environment-specific`、`instrumentation-needed` 或 `resolved-by-successful-rerun`，并说明是否需要 C25 调试镜像、是否仍阻止 M10。

### M12: evalscope 与 vLLM bench 差异深挖

- [x] 按命令引用 `C26` 离线分析 C24 evalscope attempt 2 与 vLLM bench cross-check 的详细差异，不创建 GPU workload、不修改源码、不更新 gate。
- [x] 对比 evalscope 与 vLLM bench 的 success/failure、duration、output throughput、Avg TTFT、TPOT/ITL、first/last 512 请求、Prometheus running/waiting/output TPS、prompt payload、client path 和 prefix/cache 证据。
- [x] 重点检查 speculative decoding：记录 evalscope `Spec. Accept Rate=0.7707`、`Decoded Tok/Iter=4.3615`；确认 vLLM bench JSON 未保存 spec/draft/accept 字段；用 response chunk/ITL 粒度作为弱代理，确认当前证据不支持 evalscope 接受率更差。
- [x] 保存分析到 `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/summary.md`，并在进展日志记录 P76。
- [ ] 若后续需要 definitive live 结论，按 `C26B` 重新部署相同镜像/语义，运行 local port-forward evalscope、in-cluster evalscope 和 in-cluster vLLM bench 配对实验，并在每个 measured window 抓取 `vllm:spec_decode_*` Prometheus 指标；该步骤会创建 GPU workload，必须重新获取 workspace-env permit。
- Acceptance: C26 summary 必须明确 evalscope 未达标的主因分类、speculative decoding 证据状态、是否需要 C25 调试代码、是否改变 M10 gate。当前 C26 不改变 gate：evalscope Avg gate 仍未通过，vLLM bench 通过仍只作为 harness cross-check。

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
- benchmark 完成 64k input / 1 output TTFT 和 cache-hit decode BS512 / 1.5k output throughput 两个 measured run，并保存 raw artifacts、Prometheus running BS evidence、vLLM benchmark script 对比、serving logs 与 Markdown summary；BS512 run 使用 2048 总请求，若无效则重启服务后保存 128-512 降档 BS sweep 结果。
- evalscope 与 vLLM bench 的显著差异已完成 `C21W` 分析，并保存可审计结论；若不能解释差异，则 `C21X` 配对复现实验已完成或被明确阻塞。差异原因未解释前，vLLM bench BS512 成功结果不能替代 evalscope BS512 gate。
- evalscope BS512 Mooncake/KV failure 已完成 C23 离线初筛；后续必须通过 M11/C24 判断是否稳定复现、偶现、harness-specific 或需要调试镜像，才能继续推进发布判断。
- evalscope 与 vLLM bench 的 live 成功结果差异已完成 C26 深挖；若怀疑 speculative decoding 接受率差异，必须保存 server-side `vllm:spec_decode_*` 指标或明确说明现有 artifacts 无法比较。当前不得仅凭 vLLM bench JSON 推断接受率。

## Approval Forecast

- 已授权：用户已在 2026-06-29 本线程明确同意本计划内全部审批项。
- 可直接执行的已授权动作：
  - 在 `bytedance-iaas/vllm` 远端创建 `backup/iaas_main-20260629`，指向执行时当前 `origin/iaas_main` SHA。
  - 为 `backup/iaas_main-*` 添加 GitHub 保护规则。
  - 在 image、deployment、router smoke 和 benchmark gate 之后，将远端 `iaas_main` 更新为 fork-base integration 分支。
  - 触发 ByteIAAS dev image workflow 并发布 openai/openai-devel 镜像到 Volcengine CR。
  - 使用 Helm chart 中已出现的 `wangyicong52/DeepGEMM` release wheel URL。
  - 在 `dev-cluster` 创建 namespace、Helm release、GPU workload、port-forward/evalscope benchmark 资源，并在完成后清理。
  - 2026-07-01 用户已追加授权：为根因诊断修改最小调试代码、推送诊断分支、触发 dev image 构建，并用调试镜像复测。该授权不包含更新远端 `iaas_main`，除非所有原 gate 重新满足。
- 仍需重新确认的情况：目标仓库不是 `bytedance-iaas/vllm`、备份分支名不是 `backup/iaas_main-YYYYMMDD`、外部 URL 不是 Helm chart 已出现 URL、需要生产部署、或需要本计划以外的 force push。

## Risks And Fallbacks

- Branch protection 可能拒绝直接替换 `iaas_main`：fallback 是保留远端备份，推送 fork-base integration 分支，并通过 PR 或管理员流程迁移。
- fork Dockerfile 与旧 ByteIAAS workflow 参数不完全兼容：fallback 是只补 workflow 需要的 build args，不整文件恢复旧 Dockerfile。
- DeepGEMM fork wheel 无可复现源码 ref：fallback 是使用 chart 中明确出现且本线程已授权的 release wheel URL；不退回社区 DeepGEMM。
- Mooncake/KV connector 在服务路径失败：先记录 render、argv/env、package/source 和日志证据；不直接 pin chart 版本，除非用户再次确认改变策略。
- `_moe_C` import 类问题不得通过 `vllm/` runtime Python fallback 解决；后续只能从 build/package artifact 角度处理，例如确认 wheel 应包含哪个扩展模块、CMake/setup 是否生成/抽取正确 artifact、Dockerfile 是否安装了正确 wheel。若 build/package 层无法解决，应停止并报告 blocker，而不是改 vLLM runtime 逻辑。
- Docker/GPU 环境不可用：本地完成静态和 workflow 验证，将 image build 转到 ByteIAAS workflow；GPU deployment/benchmark 等待 workspace-env permit 或报告 scheduler-visible blocker。
- servingkit monitoring chart 可能因为镜像拉取、Prometheus scrape、Service DNS 或权限失败而不可用：若 C18 不能让 prefill/decode/router `up == 1`，不得继续高 BS gate，应先修复监控部署或记录外部 blocker。
- cache-hit decode BS512 结果可能被 router dispatch、cache lookup、KV/bootstrap、decode admission 或 scheduler queue 限制；最终摘要必须区分 TTFT、TPOT/ITL、output throughput、running BS、waiting queue 和 admission 现象。BS512 无效时，需要先重启服务，再在 128-512 之间降档寻找可完整通过的最大已测 BS，帮助判断是容量边界还是固定配置/节点问题。
- 每次 benchmark 失败或中断后必须重启 P/D/router 服务，不能把 Mooncake/KV、router、queue 或 worker 的坏状态带入下一轮候选。
- vLLM 自带 `vllm bench serve` 支持 `--request-rate inf`、`--max-concurrency`、`--num-prompts`、`--random-prefix-len`、`--random-output-len` 和 `--ignore-eos`，但其统计口径、prefix 构造、tokenizer 对齐和并发限流可能与 evalscope 不同；C21V 只作为对照，除非结果证明等价，否则 final gate 仍使用 evalscope。C21V 必须在同一个新构建 vLLM 镜像启动的临时 benchmark 容器内运行；本地 Python 缺依赖不再作为 C21V 的充分验证结果。
- 当前 evalscope 和 vLLM bench 结果差异很大：evalscope BS512/2048 invalid 且有 Mooncake/KV transfer failure，vLLM bench BS512/2048 成功且 throughput `15281.44 tok/s`。这可能来自请求构造/缓存命中语义差异、client 网络路径差异、并发调度/超时策略差异、统计窗口差异，或前一轮失败服务状态污染；必须通过 C21W/C21X 归因，不能直接选择更好的 harness 结果。C21W 已解释差异后，仍需通过 C23 分析 evalscope BS512 的 Mooncake/KV failure，避免在未定位 transfer failure 的情况下把 harness 差异当作发布理由。
- C24 需要显式考虑偶现：若 evalscope BS512 重跑成功，不能默认说明压测方式无关或问题已解决；必须对比历史失败与成功 run 的节点、服务启动年龄、缓存预热、running/waiting、Mooncake bad logs、transfer 批量大小和 client path。若失败只出现一次或按概率出现，应归类为 `intermittent`，后续发布判断需要至少两次连续满足 gate 且坏日志清洁，或更具体的环境/代码根因。
- speculative decoding 接受率分析存在 artifact 缺口：evalscope summary 有 `Spec. Accept Rate`，但 vLLM bench JSON 没有 spec accept/draft 字段；C24 Prometheus 也未抓 `vllm:spec_decode_*`。因此当前只能判断“没有证据表明 evalscope 接受率更差”，不能宣称两者接受率完全相同。若该方向成为发布阻塞点，必须重跑 C26B 采集 server-side 指标。
- final gate 允许人工接受的 blocker 必须是外部环境或资源类；代码正确性、镜像内容、部署模板表达、真实请求路径、KV/Mooncake/DeepEP/DeepGEMM runtime 失败都必须阻止远端 `iaas_main` 更新。
- benchmark 性能未达阈值也必须阻止远端 `iaas_main` 更新；本计划性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，`1P1D` router-path BS512/1.5k evalscope overall output token throughput >= 14000 tokens/s，且 BS512 measured run 必须使用 `number=2048`。

## Current Status

- 当前分支：`codex/vllm-dsv4-fork-base-byteiaas-build`，基于 fork SHA `cde7799cc66c5a4cb349156a3ca3228f9798dbc9`。
- M1-M4 已完成：远端备份分支 `backup/iaas_main-20260629` 指向原 `origin/iaas_main` SHA `1ad5c27d41aa2b04d61a13c2adfe8d3db6ae2b16`，GitHub ruleset `Protect backup iaas_main branches` 已 active；ByteIAAS workflow、tag 脚本、`byteiaas-openai-devel` Dockerfile、Dockerfile 中 `vllm-router` 与 Onion CLI 构建能力已落到 fork-base 分支。
- M5 已第三次重新构建成功：旧候选镜像 `v0.10.0.iaas.dev.202606302152-openai-devel-cu130` 在 M8 第三轮部署中推进到 DeepGEMM 初始化后，decode 失败于 vendored DeepGEMM 缺少 MegaMoE SM90 FP4 API `transform_weights_for_mega_moe_sm90_fp4`。已提交 `d6fe62d15643d5619e6c5ac95201a060938a839f`，在 ByteIAAS image build 中安装 Helm chart 已出现的 `wangyicong52/DeepGEMM` release wheel，并做 build-time MegaMoE 符号检查。ByteIAAS run `28452612809` 成功，发布新候选镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，digest `sha256:57cb7a44de57b09bb8a45d214210dc8c4e76cd601c0ea0a8c78fc81f05e2d32a`，包含 `linux/amd64` manifest；后续 C13-C16 必须使用该新镜像。详见进展日志 `P38`。
- M7 部署模板已创建并在 2026-06-30 重新收敛到 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 的 P/D/router 语义：`examples/deployment/deepseek-v4-flash-pd/` 使用 `StormService` 表达 `1P1D`，同一新镜像负责 vLLM、`vllm-router` 和 Onion 模型准备；chart 新增 Helm validation，强制 `global.gpuCount=8`、prefill/decode nodeAffinity 非空且 disjoint。默认 values 不再保留调试残留 `NVSHMEM_QP_DEPTH=2048` 或 prefill `maxNumBatchedTokens=2048`，且不使用 `vllm.maxModelLen=66000`；`prefill.args.noAsyncScheduling=false`，`decode.args.maxNumSeqs=96`，其中 `maxNumSeqs` 按每个 worker admission 值理解，不因 BS512 benchmark 覆盖为 `512`。
- M8 第四轮部署已完成并清理：使用新镜像 `v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，16 GPU permit `7db945db-d65b-4d55-8f10-7c1ea453dfdd` 已释放；prefill 落到 `192.168.1.148` 且请求 8 GPU，decode 落到 `192.168.1.186` 且请求 8 GPU，router 跟随 decode 节点且不请求 GPU；Onion init 幂等跳过，模型完整性检查通过，真实 router `/v1/models` 与 `/v1/completions` 均成功；bad-log scan 和 runtime install/hotfix scan 均为 0。详见进展日志 `P39`。
- 用户在 2026-06-30 明确收窄范围：不接受 `vllm/_custom_ops.py` 中 `_moe_C` 到 `_moe_C_stable_libtorch` 的 runtime fallback；vLLM 源码修改只允许构建过程中遇到的问题。此前提交 `51b135cef854e6d72cb704068644c52d047706e5` 和 workflow run `28414886195` 被标记为无效路线，不得作为后续部署/benchmark/更新 `iaas_main` 的依据。详见进展日志 `P26`。
- C9A 已完成：workflow run `28414886195` 已取消，`vllm/_custom_ops.py` 已恢复为 fork baseline hard import `vllm._moe_C`，且 `uv run --no-project python -m py_compile vllm/_custom_ops.py` 通过。详见进展日志 `P28`。
- C9B 已完成：用户指出 build-side `_moe_C` rename 路线异常后，已取消 workflow run `28418542564`，并将 `CMakeLists.txt`、`setup.py`、`csrc/libtorch_stable/moe/torch_bindings.cpp` 恢复为 upstream main/fork baseline 的 `_moe_C_stable_libtorch` build artifact 命名。上游对比证明 upstream main 的自洽路径是：构建/import `vllm._moe_C_stable_libtorch`，但注册 `torch.ops._moe_C` namespace，且 `topk_hash_softplus_sqrt` 不再 hard import `vllm._moe_C`。详见进展日志 `P30`。
- 用户已批准按 upstream main 对齐 `topk_hash_softplus_sqrt`：删除 `f7c4c621d` 引入的 hard import，不做 fallback、不改 build artifact。详见进展日志 `P31`。
- C9D 已完成：run `28442949331` 和候选 `openai-devel` 镜像静态检查通过；当前不需要再触发一次 C9 重构建。详见进展日志 `P32`。
- P40 记录的是已废弃的历史 benchmark：该 run 使用过 `decode.args.maxNumSeqs=512`，已被用户判定不应作为当前部署语义，也不得作为 M9 gate 依据。当前有效部署语义必须始终是 `decode.args.maxNumSeqs=96`、`prefill.args.noAsyncScheduling=false`；后续 gate 只看该语义下的 C20/C21 结果。历史资源已清理。详见进展日志 `P40`、`P41` 与 `P42`。
- M10 不执行：性能未达标，按用户要求不得更新远端 `iaas_main`。
- 2026-06-30 本轮继续执行决策：不启用 `subagent-driven-development`；当前剩余工作是同一 release/image/namespace 的串行 render、permit、deploy、smoke、benchmark、cleanup，写入面集中在计划文件和同一 artifact 目录，分派并行子代理会增加状态冲突风险。主线程负责集群 mutation、registry/permit、artifact 和最终 gate 判断。详见进展日志 `P42`。
- C13 已按新语义重新通过：rendered manifest 使用 `v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，prefill 没有 `--no-async-scheduling`，decode 渲染 `--max-num-seqs "96"`，没有 `--max-model-len`、`66000` 或 runtime hotfix/install，同节点负例被 Helm validation 拒绝。详见进展日志 `P43`。
- C14 已获取新 GPU permit：session `codex-vllm-dsv4-flash-pd-20260630-234927-790782`，permit `585fd741-c3a7-490b-935f-cb761b5652fc`，状态 `granted`，P/D 节点仍为 `192.168.1.148` 和 `192.168.1.186`。命令尾部曾因本地状态文件写入 bug 失败，已修正且未创建 GPU workload。详见进展日志 `P44`。
- C15 已完成部署：旧节点 `192.168.1.148/192.168.1.186` 因已有 hostNetwork P/D 服务占用 8000/8001 端口导致 Pending，已卸载本任务失败 release 并改选空闲节点。当前 release `dsv4-flash-pd` 已 deployed，prefill 在 `192.168.1.149`，decode/router 在 `192.168.1.154`，三类 pod 均 Ready；permit 状态已标记 `running`。详见进展日志 `P45`。
- C16 已通过：Onion init 幂等跳过，P/D/router 均 Ready，router `/v1/models` 返回 `max_model_len=1048576`，真实 `/v1/completions` 返回非空 `","`，completion id 显示 route 到 prefill `192.168.1.149:8000` 和 decode `192.168.1.154:8001`；bad-log scan 和 runtime install scan 无命中。详见进展日志 `P46`。
- C17/C17A/C18 已复用已准备好的 evalscope、tokenizer 和轻量 metrics 证据。C19/C20 已在 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96` 部署上重新执行：C19 warmup 和 64k cache seed 成功；C20 64k/1 Avg TTFT 为 `6633.05 ms`，通过 `<10s` gate；该 C20 是计算结果，不是 cache 命中结果，因为 `prefix_length=0`、`Cached Prompt tok/s=0.00`。C21 旧形态 BS512/512 请求 run 未形成有效 throughput 结果：decode 日志出现大量 `Mooncake transfer engine returned -1`，prefill 日志出现 producer-side `timed out after 480 seconds without being sent`，evalscope 被中断并记录 exit `130`。所有临时 Helm release、namespace、port-forward 和本任务 GPU permit `585fd741-c3a7-490b-935f-cb761b5652fc` 已清理/释放。详见进展日志 `P47`。
- 2026-07-01 反向节点重试也未形成有效 BS512 throughput：P/D 对调为 prefill `192.168.1.154`、decode/router `192.168.1.149` 后，C20 64k/1 Avg TTFT `6636.64 ms` 通过且仍是计算结果；C21 旧形态 BS512/512 请求 run 处理到约 `487/512` 后静默，prefill 日志出现 `Sending to 192.168.1.149 ... failed (ret=-1)`，decode 日志出现 `Mooncake transfer engine returned -1`，evalscope 被中断并记录 exit `120`。反向节点 release、namespace、本地 port-forward/evalscope 和 permit `aaa9b695-2390-4e2f-ad21-2b52d0a27fd0` 已清理/释放。详见进展日志 `P49`。
- 用户在 2026-07-01 修改 benchmark 策略：C21 BS512 标准吞吐 run 请求数必须为 `4 * BS = 2048`；如果 BS512 跑不过，必须重启服务后按 C21A 在 128-512 之间降档寻找能完整通过压测的最大已测 BS；同时必须部署 servingkit `llm-serving-monitoring` 监控并用 `vllm:num_requests_running` 检查实际 running BS，实际 running BS 达不到目标时不继续测试更高 BS；还需补充 vLLM 自带压测脚本与 evalscope 的口径对比。详见进展日志 `P50`、`P51`。
- 2026-07-01 最新有效 M9 结果：C18 monitoring 通过 prefill/decode `up=1`，router metrics 因 `/metrics` 返回 `405` 被禁用但不影响 C16 router smoke；C20 64k/1 Avg TTFT `6775.64 ms`，通过 `<10s` gate，且 `prefix_length=0`、`Cached Prompt tok/s=0.00` 证明这是计算结果。C21 BS512/2048 请求 run exit `120`，出现 Mooncake/KV transfer failure，无有效 Avg/Overall throughput；C21M 显示 `max_decode_running=231.0`、`max_decode_waiting=512.0`、`max_decode_output_tps_30s=8578.58620689655`。按要求执行 C21R 重启后，C21A 只测试不高于 observed capacity 的 BS192：768/768 成功、Avg Output Tokens `1536.00`、evalscope Output Throughput `6354.03 tok/s`、Prometheus `max_decode_running=192.0`、bad-log scan clean，但该结果低于 `14000 tok/s` 且不能替代 BS512 gate。C21V 已在同一新构建 vLLM 镜像的临时 benchmark Pod 中完成：`python3 -m vllm.benchmarks.serve` 在镜像内是空入口，不可作为 CLI；实际 `vllm bench serve` 可运行，BS128/512 请求、1536 output、0 失败，但 output throughput 只有 `5018.14 tok/s`，Mean TTFT `18268.25 ms`，Prometheus 窗口显示 decode running max `128`、avg `31.97`、decode 30s output TPS max `8772.66`、avg `2531.23`。C21V 是口径对照，不能替代 evalscope gate。详见进展日志 `P55`-`P62` 与 artifact `summary.md`。
- 2026-07-01 按用户要求补充 vLLM 自带压测 BS sweep：在同一新构建 vLLM 镜像的 no-GPU benchmark Pod 内，通过 in-cluster router Service 执行 `vllm bench serve`，固定 64K prefix、1536 output、`number = 4 * BS`，覆盖 BS256/400/512。三组均 exit `0` 且 bad-log scan clean：BS256 `1024/1024` 成功，output throughput `8052.52 tok/s`，Mean TTFT `23914.05 ms`，Prometheus max decode running `256.0`；BS400 `1600/1600` 成功，output throughput `11091.25 tok/s`，Mean TTFT `10276.96 ms`，Prometheus max decode running `400.0`；BS512 `2048/2048` 成功，output throughput `15281.44 tok/s`，Mean TTFT `6876.14 ms`，Prometheus max decode running `510.0`。该补充结果证明 vLLM bench harness 可把 decode running 推近目标 BS，且 BS512 在 vLLM bench 口径超过 `14000 tok/s`；但它仍是 harness comparison，不替代已 invalid 的 evalscope BS512 gate，也不改变 M10 不更新远端 `iaas_main` 的决策。详见进展日志 `P64`、`P65` 与 artifact `vllm-bench-bs-sweep-20260701/summary.md`。
- C21W 离线差异分析已完成：结论分类为 `mixed`，主因是 `service-state/kv-transfer` 与 `harness-timeout/statistics` 叠加，`client-path` 是可信放大因素；evalscope BS512 是 invalid run，request DB `result_count=0` 且伴随 Mooncake/KV timeout，Prometheus 显示 decode running 最高只有 `231`、waiting 最高 `512`；vLLM bench BS512 是后续健康服务状态下的完整 run，`2048/2048` 成功、Prometheus decode running 最高 `510`、bad-log clean。该结论解释了差异，因此当前不进入 `C21X`，但也不把 vLLM bench 替代 evalscope gate。详见进展日志 `P68` 与 artifact `harness-diff-analysis-20260701/summary.md`。
- C23 离线根因初筛已完成：evalscope BS512 日志中 `sync_timeout=62`、`ret_failed=62`、`xfer_returned=93`、`producer_timeout=535`，而 `KV group count mismatch`、`handshake compatibility failure`、`Mooncake found no common KV transfer regions` 均为 0；失败 transfer 的 p50 约 `31.44s`、`82033` descriptors、`1.43GB`，最大接近 `2GB`。当前首要假设是 evalscope BS512 的请求突发/客户端路径触发 Mooncake/RDMA transfer timeout 或 descriptor pressure，而不是 metadata/握手不一致或明显部署语义发散。详见进展日志 `P70` 与 artifact `mooncake-failure-diagnosis-20260701/summary.md`。
- 用户已在 2026-07-01 追加授权继续部署分析根因，并允许修改调试代码和推送。计划已新增 M11/C24/C25：先用当前成功镜像 live 复现并判断是否偶现；若证据不足，再添加最小可开关调试日志、推送诊断分支并构建调试镜像复测。详见进展日志 `P71`。
- C24 live 诊断已完成并清理：使用当前成功镜像重新部署到 prefill `192.168.1.148`、decode/router `192.168.1.154`，保持 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`。第一次有效 evalscope BS512/2048 run 在 1762/2048 后停滞并出现 producer-side `timed out after 480 seconds without being sent`，中断为 exit `130`；重启 P/D/router 后第二次 evalscope BS512/2048 完整成功但 Avg output throughput `12291.66 tok/s`、Avg TTFT `20907.75 ms`，未达 gate；同一 live 服务上的 vLLM bench BS512/2048 cross-check 完整成功，output throughput `15152.89 tok/s`、Mean TTFT `7964.35 ms`，满足用户指定的两个 Avg 门槛。精确 Prometheus 窗口显示 evalscope decode running avg/max `294.54/512`、decode waiting avg/max `18.44/187`；vLLM bench decode running avg/max `391.93/512`、decode waiting avg/max `4.31/16`。C24 summary 分类为 `intermittent + harness-specific`，详见进展日志 `P72`-`P75` 与 artifact `live-mooncake-diagnosis-20260701/summary.md`。
- C26 benchmark deep dive 已完成：evalscope attempt 2 与 vLLM bench 都完成 `2048/2048`、总 output 都是 `3,145,728` tokens，但 evalscope duration `255.92s`、Avg TTFT `20907.75 ms`、output throughput `12291.66 tok/s`，vLLM bench duration `207.60s`、Mean TTFT `7964.35 ms`、output throughput `15152.89 tok/s`。evalscope TPOT/ITL 并不更差，且报告 `Spec. Accept Rate=0.7707`、`Decoded Tok/Iter=4.3615`；vLLM bench JSON 没有 spec/draft/accept 字段，C24 未抓 `vllm:spec_decode_*`，因此不能离线比较真实接受率。chunk/ITL 代理显示 evalscope 约 `4.261` output tokens/ITL event、vLLM bench 约 `4.070`，不支持 evalscope 接受率更差。当前主因分类为 `client-path + TTFT/admission dominated; speculative-acceptance-not-primary`，详见进展日志 `P76` 与 artifact `benchmark-diff-deep-dive-20260701/summary.md`。
- C26B live follow-up 已部分执行并清理：重新使用镜像 `v0.10.0.iaas.dev.202606302238-openai-devel-cu130` 部署 prefill `192.168.1.148`、decode/router `192.168.1.154`，smoke 成功。local port-forward evalscope BS512/2048 完成 `2048/2048`，但 request generation 耗时 `18:44`，Avg output throughput `11875.18 tok/s`、Avg TTFT `28649.67 ms`；processing window decode running avg/max `262.02/512`，weighted spec accept rate `0.8263`。in-cluster evalscope 首次被 client 环境阻塞：Pod 内无代理 `pip install evalscope==1.8.1` 卡依赖下载，本地 `.venv-evalscope` hostPath 不能跨到 Kubernetes 节点；随后已用临时 no-GPU Pod 验证代理安装 `evalscope[perf]==1.8.1` 可行，并把 C26B scaffold 改为显式使用 `http://100.68.170.29:3128`。in-cluster `vllm bench serve` no-GPU Pod 直连 router Service 完成 `2048/2048` 且 bad-log clean，但 output throughput `10688.35 tok/s`、Mean TTFT `27152.21 ms`，processing window decode running avg/max `305.34/512`、weighted spec accept rate `0.7234`。因此 evalscope 不达标不能归因成纯本地 port-forward；更准确分类是 client/request construction、admission/waiting、Mooncake/RDMA 偶发 timeout 与服务端 running 形态共同影响。详见进展日志 `P77`、`P78` 与 artifact `live-spec-paired-20260701/runs/incluster-vllm-bench-bs512-spec/summary.md`、`evalscope-proxy-install-20260701/pod.log`。
- C26B proxy rerun 首次尝试已清理：permit `f50acdc0-e818-43d2-bfa9-c88e5dab11a8` 获取成功，使用同镜像同语义重新部署 prefill `192.168.1.148`、decode/router `192.168.1.154`；decode 到达 Ready，但 prefill 所在节点 `192.168.1.148` 在启动中变为 `NotReady`/`Ready=Unknown`，原因 `Kubelet stopped posting node status`，并被加 `node.kubernetes.io/unreachable` NoSchedule/NoExecute taint，prefill pod 被 `TaintManagerEviction` 标记删除。因此该尝试没有进入 router smoke、monitoring 或 benchmark。release/namespace 已删除，卡住的本任务 prefill pod 已 force delete，permit 已释放；后续重试不得使用 `192.168.1.148`，除非节点恢复 Ready。详见进展日志 `P79` 与 artifact `live-spec-paired-20260701/proxy-rerun/`。
- C26B proxy rerun 第二次尝试已完成 benchmark：重新选择 prefill `192.168.1.186`、decode/router `192.168.1.154`，同镜像同语义 `1P1D` 部署、router smoke 和 monitoring 均成功；in-cluster no-GPU Pod 使用代理 `http://100.68.170.29:3128` 安装 `evalscope[perf]==1.8.1`，直连 router Service 执行 evalscope BS512/2048、64k prefix、1.5k output。Formal run 完成 `2048 / 2047 / 1`，Avg TTFT `20492.39 ms`、Overall Avg output throughput `12613.28 tok/s`，均未达用户 gate；processing-window Prometheus decode running avg/max `300.68/512.0`、waiting avg/max `45.30/398.0`、generation tps 30s avg/max `12047.57/18842.41`、rate-derived spec acceptance `0.7867`。因此“不达标只是本地 port-forward/client path 导致”被排除；更准确结论是 in-cluster evalscope 仍无法长期维持 512 running，且有 1 次 router prefill connection closed 失败请求。详见进展日志 `P80` 与 artifact `live-spec-paired-20260701/proxy-rerun-186-154/runs/incluster-evalscope-bs512-proxy/summary.md`。
- C26C evalscope proxy BS 降档 sweep 已完成并清理：继续使用同一镜像和同一 `1P1D` 语义，所有 no-GPU benchmark Pods 均通过代理命令 `python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'` 安装并确认 `evalscope 1.8.1`。BS400/256/128 均按 `number = 4 * BS` 完整完成，但 Avg output throughput 分别为 `10525.87`、`8042.26`、`4920.15 tok/s`，均未达到 `>= 14000 tok/s`；Avg TTFT 分别为 `28498.44`、`24342.75`、`21165.30 ms`。Prometheus 显示对应 decode running max 分别为 `400/256/128`，但 avg running 只有 `185.50/121.80/55.91`。BS128 日志只发现 startup 阶段 8 条 `Failed to open device mlx5_7 ... GID 3` RDMA 初始化错误，未见本轮请求处理期间 Mooncake KV pull/producer timeout 类错误。BS400/256/128 均未通过 Avg gate，且 BS128 已是用户指定降档范围下界；本轮 serving、monitoring、benchmark Pod、port-forward 和 permit `e927012a-ff9e-4626-a769-d80bc8cac77f` 均已清理/释放。详见进展日志 `P81` 与 artifact `evalscope-bs-downgrade-sweep-20260701/`。
- C25 已开始：为定位历史 attempt 1 producer timeout / Mooncake descriptor / RDMA 细节，已在诊断分支 `codex/vllm-dsv4-mooncake-transfer-diagnostics` 添加默认关闭的 `VLLM_DSV4_MOONCAKE_DIAG` 日志开关，仅记录 producer/consumer transfer 诊断信息，不改变 transfer、timeout、调度、retry、fallback import 或部署语义；`py_compile` 与 `git diff --check` 已通过。详见进展日志 `P82`，代码 diff artifact `live-mooncake-diagnosis-20260701/c25-debug-code.diff`。
- M10 不执行：evalscope BS512 gate 仍未满足。C24 attempt 1 有间歇性 KV/Mooncake producer timeout/stall；C24 attempt 2 虽完成但 Avg output throughput 低于 `14000 tokens/s` 且 Avg TTFT 高于 `10s`；C26B in-cluster evalscope BS512 仍未达标；C26C 在 BS400/256/128 的降档 sweep 中也没有找到通过 Avg output throughput gate 的候选。除非用户明确修改 gate，否则不得更新远端 `iaas_main`。

## Next Action

当前 C26C 已完成 128-512 范围内的 evalscope proxy 降档验证：BS400/256/128 均完整完成但 Avg output throughput 均低于 `14000 tok/s`，因此没有找到可通过 gate 的候选。所有本轮 serving release、monitoring release、benchmark Pod、port-forward 和 permit `e927012a-ff9e-4626-a769-d80bc8cac77f` 已清理/释放；两个任务 namespace 查询为空。M10 仍阻止更新远端 `iaas_main`。当前正在执行 C25：提交并推送诊断分支，触发 ByteIAAS dev image workflow，等待调试镜像；镜像成功后才能用同一部署语义、仅增加 `VLLM_DSV4_MOONCAKE_DIAG=1` 复测 attempt 1 类 workload。
