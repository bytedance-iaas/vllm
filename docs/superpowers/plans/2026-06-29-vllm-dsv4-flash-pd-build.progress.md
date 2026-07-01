# vLLM DSV4 Flash P/D Fork-Base Build Migration Progress Log

本文档记录详细进展、证据摘录、失败尝试、issue log、subagent 结果和最终摘要。主计划只保留 compact status 和指针。

## 2026-06-29 Initial Planning State

### 用户最新决策

- 原 `iaas_main` 备份必须在 GitHub 远端仓库进行，本地备份不满足要求。
- 远端备份分支格式改为 `backup/iaas_main-日期`，本计划使用 `backup/iaas_main-20260629`。
- 需要为 `backup/iaas_main-*` 添加保护规则。
- 新 `iaas_main` 应以 `https://github.com/wangyicong52/vllm.git` 的代码作为起点。
- 旧 `iaas_main` 只保留必要的构建相关能力。
- 旧 `iaas_main` 的原代码逻辑相关内容不保留。
- Mooncake 保持当前 vLLM Dockerfile 的处理方式即可，不需要强制使用 Helm chart 中的版本。
- 不新增、不恢复 `scripts/ci/check_byteiaas_dsv4_runtime.py`；不做镜像内 import/CLI smoke，不在构建流程补充 smoke。
- 远端 `iaas_main` 更新已在本线程获得授权，但必须作为最后发布门执行：image、deployment、router smoke、benchmark gate 完成或 blocker 明确可接受后，才运行 `C10`。
- Approval Forecast 中列出的计划内审批项均已同意；只有偏离本计划时才重新确认。
- 用户进一步明确：不需要镜像内 import/CLI smoke，也不需要在构建流程补充 smoke；完成代码后构建镜像，使用该镜像编写无运行时 hotfix/安装的部署并部署到 `dev-cluster`。
- benchmark 要求：参考 `llm-serving-benchmark`，进行 `64k input / 1 output` TTFT 测试，以及全命中 cache 场景下 decode 端 `bs512 / 1.5k output` throughput 测试。
- workspace-env 只读发现：有效 env root 是 `/data00/home/hanhan.hank/workspace/env`；`envctl info dev-cluster` 显示类型为 Kubernetes；`envctl validate dev-cluster` 已返回 `OK dev-cluster`。
- evalscope 命令参考官方 stress-test 文档；本计划使用 `evalscope perf`、`random` dataset、`--prefix-length`、`--min/max-prompt-length`、`--min/max-tokens`、`--seed 42`、`--outputs-dir` 和 `--extra-args '{"temperature":0,"ignore_eos":true}'`。

### 只读证据

- 当前本地 `origin/iaas_main` 已知 SHA：`1ad5c27d4`
- 当前本地 fork ref 已知 SHA：`cde7799cc`
- `origin/iaas_main` 最近提交：
  - `1ad5c27d4 Merge pull request #143 from bytedance-iaas/codex/vllm-byteiaas-daily-build`
- fork ref 最近提交：
  - `cde7799cc Route SM90 MegaMoE predispatch through Triton`
- fork tree 中未发现 ByteIAAS release workflows：
  - `.github/workflows/byteiaas-release-dev.yml`
  - `.github/workflows/byteiaas-release.yml`
  - `.github/workflows/_byteiaas-build-and-publish-image.yml`
  - `.github/workflows/_byteiaas-build-wheel.yml`
- fork tree 中未发现：
  - `docker/byteiaas-openai-devel.Dockerfile`
- fork `docker/Dockerfile` 中已存在：
  - `ARG INSTALL_KV_CONNECTORS=false`
  - `ARG MOONCAKE_WHEEL_AARCH64`
  - `ARG MOONCAKE_WHEEL_X86_64`
  - DeepEP build/install stage
  - DeepGEMM Python interpreter/vendoring build path

### 依赖处理分类

| 组件 | 当前来源 | 新计划处理 |
| --- | --- | --- |
| vLLM 源码 | `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe` | 作为新 `iaas_main` 起点 |
| ByteIAAS workflows | 旧 `origin/iaas_main` | 回拷保留 |
| ByteIAAS image tag script | 旧 `origin/iaas_main` | 回拷保留 |
| `docker/byteiaas-openai-devel.Dockerfile` | 旧 `origin/iaas_main` | 回拷保留 |
| `scripts/ci/check_byteiaas_dsv4_runtime.py` | 旧计划曾建议新增/恢复 | 不需要；不创建 |
| DSV4 P/D deployment | vLLM repo new example based on servingkit chart | 创建 `examples/deployment/deepseek-v4-flash-pd/**`，无 runtime hotfix/install |
| benchmark artifacts | local repo artifacts | 写入 `artifacts/2026-06-29-vllm-dsv4-flash-pd/` |
| old vLLM runtime/model/kernel logic | 旧 `origin/iaas_main` | 不保留 |
| Mooncake | 当前 Dockerfile KV connector handling | 保持现状，不强制 chart pin |
| DeepEP | fork/current Dockerfile community build stage | 保持社区来源 |
| DeepGEMM | fork 源码 + chart 中 `wangyicong52/DeepGEMM` 定制 wheel | 保留 fork 路线 |
| `vllm-router` | 社区包 | 社区安装，不 fork |

### 当前状态

- 已创建/更新计划文件：
  - `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.md`
  - `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.commands.md`
  - `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.progress.md`
- 尚未执行：
  - GitHub 远端备份分支 push
  - fork-base integration branch 创建
  - ByteIAAS 构建文件回拷
  - Dockerfile edits
  - workflow dispatch
  - 远端 `iaas_main` 更新，但只能在 deployment 和 benchmark gate 之后执行

## Issue Log

- `I0`: 执行 preflight 发现当前工作区不是 linked worktree，且只有本计划三份文档位于 `docs/superpowers/` 未跟踪；这些文件在主计划 write scope 内。为避免 C1 被计划文档本身阻塞，命令引用已把 clean check 收窄为“除本计划文档外无其它未提交改动”。
- `I1`: 当前本地 clone 位于旧工作分支，不应直接在该分支执行源码迁移。执行阶段必须从干净 `origin/iaas_main` 和 fork ref 创建新 integration branch。
- `I2`: 替换远端 `iaas_main` 可能触发 branch protection；需要准备 PR/admin fallback。
- `I3`: 如果 DeepGEMM fork release 没有可 checkout tag/ref，使用本线程已授权的 chart 中已经出现的 wheel URL。
- `I4`: GitHub rulesets API 可能因权限或 GitHub Enterprise 版本不可用而失败；失败时记录 exact error，并要求仓库管理员在 UI 上为 `backup/iaas_main-*` 添加 deletion/non-fast-forward 保护。
- `I5`: `dev-cluster` GPU workload 需要 workspace-env GPU permit；如果 permit queued/denied/blocked，不得创建 Pending workload 占位。
- `I6`: 若 evalscope 不可安装或无法表达 full-cache decode BS512 场景，必须停止并记录 blocker，不得擅自改用自定义 harness。
- `I7`: P7 的 Prometheus skipped 决策已被 P51 supersede；后续必须部署 servingkit `origin/hanhan_dev:llm-serving-monitoring` 的最小 Prometheus 监控，并用 `vllm:num_requests_running` / `vllm:num_requests_waiting` 判断 running BS 和排队状态。

## Progress Entries

### P13: Execution preflight and subagent decision

- Summary: 读取 `superpowers:executing-plans`、`superpowers:subagent-driven-development` 和 `superpowers:using-git-worktrees`；开始执行计划。当前不启用 subagent-driven-development 做并行写入，主线程串行执行 M1-M5，后续可将只读 review/final review 分给子代理。
- Evidence:
  - `git rev-parse --git-dir` 与 `git rev-parse --git-common-dir` 均为 `.git`，当前不是 linked worktree。
  - 当前 `git status --short` 仅显示 `?? docs/superpowers/`，对应本任务计划文件，在主计划 write scope 内。
  - 主计划 `Current Status` 已记录 subagent 决策：M1-M5 同时触及远端分支、同一 integration branch、workflow 和 Dockerfile，顺序依赖强；不做并行写入。
  - 命令引用 `C1` 已增加排除本计划三份文档后的 clean check。

### P14: M1 remote backup completed

- Summary: 完成 C1、C2、C2A。`origin/iaas_main` 已备份到 GitHub 远端 `backup/iaas_main-20260629`，并添加 `backup/iaas_main-*` family ruleset。
- Evidence:
  - C1 输出 `ORIGINAL_IAAS_MAIN_SHA=1ad5c27d41aa2b04d61a13c2adfe8d3db6ae2b16`。
  - C1 输出 `FORK_BASE_SHA=cde7799cc66c5a4cb349156a3ca3228f9798dbc9`。
  - C2 `git ls-remote origin refs/heads/backup/iaas_main-20260629` 返回 `1ad5c27d41aa2b04d61a13c2adfe8d3db6ae2b16`。
  - C2A 创建 GitHub ruleset `Protect backup iaas_main branches`，id `18250606`，`target=branch`，`enforcement=active`，include `refs/heads/backup/iaas_main-*`，rules 包含 `deletion` 和 `non_fast_forward`。

### P1: Plan rewritten for fork-base strategy

- Summary: 将计划从“把 fork 差异 port 到旧 `iaas_main`”改为“以 fork 为新源码基线，只从旧 `iaas_main` 回拷 ByteIAAS 构建能力”。
- Evidence:
  - 主计划 `Architecture` 已明确先远端备份，再从 fork 创建 integration branch。
  - M3 明确禁止回拷 `vllm/`、`csrc/`、`cmake/`、`vllm/models/` 等旧源码逻辑。
  - M4 明确 Mooncake 保持当前 Dockerfile 处理方式，不强制 pin chart 版本。

### P2: User-approved backup format, protection, and approvals

- Summary: 将备份分支改为 `backup/iaas_main-20260629`，新增 `backup/iaas_main-*` GitHub ruleset 步骤，移除 `scripts/ci/check_byteiaas_dsv4_runtime.py`，并标记计划内审批均已同意。
- Evidence:
  - 命令引用 `C2` 使用 `BACKUP_BRANCH="backup/iaas_main-20260629"`。
  - 命令引用新增 `C2A`，通过 GitHub rulesets 保护 `refs/heads/backup/iaas_main-*`。
  - 后续 P3 已移除构建流程 inline runtime smoke；`C9` 现用于 ByteIAAS workflow 构建并发布镜像。
  - 主计划 `Approval Forecast` 改为“已授权”。

### P3: Added dev-cluster deployment and benchmark scope

- Summary: 移除构建流程中的镜像内 import/CLI smoke；新增镜像构建后基于新镜像的 DSV4 P/D 部署模板、`dev-cluster` 部署、evalscope benchmark 和 artifacts 要求。
- Evidence:
  - 主计划新增 M7/M8/M9：部署模板、dev-cluster 部署、evalscope benchmark。
  - 命令引用 `C13` render 并检查 forbidden runtime hotfix/install patterns。
  - 命令引用 `C14` 使用 workspace-env registry 和 GPU permit。
  - 命令引用 `C20` 定义 64k input / 1 output TTFT measured run。
  - 命令引用 `C21` 定义 prefix-cache seed 后的 decode BS512 / 1.5k output throughput measured run。

### P6: Benchmark output length decision

- Summary: 用户显式要求继续按 `llm-serving-benchmark` skill，decode cache-hit output throughput 固定 output 为 1.5k。
- Evidence:
  - 主计划 `M9` 将 decode BS512 output length 从 128 改为 1.5k，即 1536 tokens。
  - 命令引用 `C21` 增加 `DECODE_OUTPUT_TOKENS=1536`，并把 `--min-tokens`、`--max-tokens` 都绑定到该值。
  - 命令引用 `C21` 保留 `--extra-args '{"temperature":0,"ignore_eos":true}'`，符合 `llm-serving-benchmark` 对固定输出长度的要求。
  - 命令引用 `C21` 将有效性阈值更新为 `1536 * 95% = 1459.2`，低于该平均输出 token 数时不得作为 throughput capacity 结果。

### P7: Prometheus skipped decision

- Summary: 用户变更方案：此处先跳过 Prometheus，不自动部署临时 Prometheus。
- Evidence:
  - 主计划 `M9` 改为 C18 仅采集 pod-local `/metrics` head、日志和 skipped note。
  - 命令引用 `C18` 写入 `metrics-lightweight/prometheus-skipped.md`，并保存 existing monitoring resource list、pod logs tail、pod-local `/metrics` head。
  - 命令引用 `C20` 和 `C21` 移除 Prometheus `query_range` measured-window 抓取。
  - 命令引用 `C22` 移除 Prometheus namespace/port-forward 清理逻辑，summary 改为标记 Prometheus skipped by user。

### P8: Remote iaas_main update moved to final gate

- Summary: 用户同意将远端 `iaas_main` 更新移动到最后，在 integration branch 完成 image、deployment、router smoke 和 benchmark gate 后再执行。
- Evidence:
  - 主计划将原 `M6` 改为 `M10`，并放在 M7/M8/M9 之后。
  - 主计划 `M10` 要求 M5/M7/M8/M9 已完成，或未完成项已作为明确可接受 blocker 记录，才能更新远端 `iaas_main`。
  - 命令引用 `C9` 明确必须在更新远端 `iaas_main` 前使用 integration branch `checkout_ref` 构建镜像。
  - 命令引用 `C10` 的 When 改为依赖 C9、C13、C16、C20/C21 和 C22 summary gate。

### P9: Final gate acceptable blocker boundary

- Summary: 用户同意 final gate 中只有外部环境或资源类 blocker 可进入人工发布决策；代码、镜像、部署模板和 runtime 路径失败不得更新远端 `iaas_main`。
- Evidence:
  - 主计划 `M10` 明确可接受 blocker 仅限 GPU permit 长时间排队、`dev-cluster` 临时资源不足、CR/image pull 临时失败、Onion 模型源临时不可用等外部环境或资源问题。
  - 主计划 `M10` 明确 render/config、镜像缺依赖、Onion init 或模型完整性失败、vLLM 启动失败、router real request 失败、KV transfer 错误、DeepGEMM/DeepEP/Mooncake import 或 runtime 错误不得进入远端 `iaas_main` 更新。
  - 命令引用 `C10` 的 When 条件同步加入该 blocker 分类，避免执行者只看命令文件时误判。

### P10: Performance gate blocks iaas_main update

- Summary: 用户明确性能 gate 看 Avg，性能未达阈值时不更新远端 `iaas_main`；阈值为 64k/1 Avg TTFT < 10s，BS512/1.5k evalscope overall output throughput >= 14000 tokens/s。
- Evidence:
  - 主计划 `M9` 要求性能 gate 看 Avg，不看 P50/P95/P99。
  - 主计划 `M9` 要求 64k/1 Avg TTFT 小于 10s，BS512/1.5k evalscope overall output throughput 达到 14000 tokens/s 以上。
  - 主计划 `M10` 明确 benchmark 跑通但性能未达上述阈值也不得更新远端 `iaas_main`。
  - 主计划 Risks 明确性能未达阈值必须阻止远端 `iaas_main` 更新。
  - 命令引用 `C10`、`C20`、`C21` 和 `C22` summary 模板同步加入上述性能阈值。

### P11: 1P1D throughput interpretation

- Summary: 用户明确部署应为 `1P1D`；此时整个集群的 output 来自单台 decode 机器。
- Evidence:
  - 主计划 `M7` 明确部署形态固定为 `stormService.replicas=1`、`prefill.replicas=1`、`decode.replicas=1`、`router.replicas=1`。
  - 主计划 `M9` 明确 BS512/1.5k output throughput 是 `1P1D` router-path 结果，总 output 来自单个 decode 节点，不是多 decode 聚合吞吐。
  - 命令引用 `C13` 和 `C15` 默认传入 `STORM_REPLICAS=1`、`PREFILL_REPLICAS=1`、`DECODE_REPLICAS=1`、`ROUTER_REPLICAS=1`。
  - 命令引用 `C21` 和 `C22` summary 模板同步标注该 throughput 口径。

### P12: P/D must use different 8-GPU nodes

- Summary: 用户指出当前给出的 servingkit 部署模板中 prefill 和 decode 都使用 8 卡，因此在集群中必须分布到不同节点；计划已把这点从资源总量说明升级为 render/deploy/preflight 的硬约束。
- Evidence:
  - 只读检查 `/data00/home/hanhan.hank/workspace/servingkit/vllm/deepseek/deepseek-v4-flash-pd/values.yaml`：`global.gpuCount: 8`，`stormService.replicas: 1`，`prefill.replicas: 1`，`decode.replicas: 1`，prefill/decode 均有 nodeAffinity。
  - 只读检查 `/data00/home/hanhan.hank/workspace/servingkit/vllm/deepseek/deepseek-v4-flash-pd/templates/stormservice.yaml`：prefill 和 decode 均将 `.Values.global.gpuCount` 渲染为 `nvidia.com/gpu` limits/requests。
  - 主计划 `M7` 已要求 `global.gpuCount=8`，`PREFILL_NODE` 与 `DECODE_NODE` 非空且不同，router 默认跟随 decode 节点。
  - 命令引用 `C13` 和 `C15` 在 `GLOBAL_GPU_COUNT != 8`、节点缺失或 P/D 节点相同时直接失败，并把 prefill/decode/router nodeAffinity 显式传入 Helm。
  - 命令引用 `C14` 在申请 GPU permit 前检查两个所选节点均存在、可调度、allocatable GPU 至少为 8。

### P4: Grill decisions for deployment shape

- Summary: 用户明确部署形态继续使用 `StormService`，并参考 servingkit 已存在实现；节点 placement 采用参数化方式，执行时填写实际节点；`hostNetwork` 保留 servingkit 现状并默认开启。
- Evidence:
  - 主计划 `M7` 明确 prefill/decode workload 继续使用 `StormService`，不改写为 StatefulSet、Deployment 或自定义控制逻辑。
  - 主计划 `M7` 明确 values 中 node placement 参数化，实际部署节点由执行时根据 `dev-cluster` permit/容量填写。
  - 主计划 `M7` 明确 `prefill.hostNetwork`、`decode.hostNetwork`、`router.hostNetwork` 默认开启。
  - 主计划 `M8` 和命令引用 `C14` 新增 `stormservices.orchestration.aibrix.ai` CRD preflight；CRD 不存在时停止并记录 blocker，不在本任务安装 Aibrix/StormService 控制面。
  - 命令引用 `C13` render evidence 增加 `kind: StormService` 和 `hostNetwork: true` 检查；`C15` deploy 默认传入 `HOST_NETWORK=true`。

### P5: Onion model preparation decision

- Summary: 用户明确部署模板应使用 Onion 进行模型下载，已有模型时由 Onion 自己跳过，这样验证才是端到端。
- Evidence:
  - 主计划 Architecture 和约束区明确模型准备由 `oniond download model ... --turbo --dir ...` 完成；这属于模型数据准备，不允许扩展成 Pod 启动时安装 Onion、pip 包或 runtime hotfix。
  - 主计划 `M4` 增加 Onion CLI 能力：同一个新构建 vLLM 镜像必须提供 `oniond`；若 base image 中不存在，按 `onion-ai-data` skill 在 Dockerfile 中添加 Volcengine extra-tools apt source 并安装 `onion-ai-data`。
  - 主计划 `M7` 增加 values 契约：`onion.enabled=true`、`onion.model=DeepSeek-V4-Flash`、`onion.dir=/data01`。
  - 主计划 `M7` 明确 Onion model prepare initContainer image 必须是同一个 `global.image`，不引入专用 Onion 工具镜像。
  - 命令引用 `C7` 增加 Dockerfile/build workflow 中 `onion-ai-data`、Volcengine apt source、`command -v oniond` 的结构检查。
  - 命令引用 `C13` render 检查增加 `oniond download model`、Onion model/dir、同一新镜像和 runtime `apt install` forbidden pattern。
  - 命令引用 `C15` deploy 默认打开 Onion 模型准备，且传入同一个 `global.image`。
  - 命令引用 `C16` 收集 `onion-model-prepare` 或 `init-model` 日志，并对非 router pod 硬检查模型目录中的 `config.json`、tokenizer 和 safetensors index/shards。

## Initial Plan Summary

当前仅完成计划编写，尚未实施迁移。执行建议使用 `superpowers:executing-plans`，第一步从命令引用 `C1` 开始；`C2`、`C2A`、计划内发布动作、dev-cluster 部署和 benchmark 已获本线程授权。`C10` 也已获授权，但只能作为最后发布门执行；若目标、参数、验证门槛或执行顺序偏离计划必须重新确认。

### P13: ByteIAAS workflow first run failed on rustup bootstrap

- Summary: 第一次 ByteIAAS workflow 构建未产出可部署镜像；失败点是 wheel image build 阶段 `build_rust.sh` 下载 rustup installer 时 `https://sh.rustup.rs` 返回 HTTP 504。该 run 已取消以释放构建资源。
- Evidence:
  - Workflow run id: `28360410848`。
  - Wheel job id: `84013412316`。
  - API log path: `/tmp/byteiaas-vllm-84013412316-api.log`。
  - 关键日志：`rustup not found, installing...` 后 `curl: (22) The requested URL returned error: 504`，随后 `bash build_rust.sh` 退出 `22`。
  - 本机 `docker info` 无法连接 `/var/run/docker.sock`，因此按计划使用 workflow 构建。

### P14: Hardened Rust bootstrap for workflow retry

- Summary: 将 `build_rust.sh` 的 rustup installer 下载和 `rustup toolchain install` 改为重试执行，避免 transient 504 直接使 ByteIAAS workflow 失败；未引入新镜像源或新依赖。
- Evidence:
  - `build_rust.sh` 新增 `retry_command` 和 `install_rustup`。
  - rustup installer 使用 `curl --retry 5 --retry-all-errors --retry-delay 2 -o /tmp/rustup-init.sh` 下载后执行。
  - `rustup toolchain install "$TOOLCHAIN"` 通过 `retry_command` 执行。
  - `bash -n build_rust.sh` 通过。

### P15: Enforced P/D different-node constraint in Helm chart

- Summary: 用户指出 servingkit 模板中 prefill/decode 各请求 8 GPU，因此必须分布到不同节点；迁移 chart 已新增 Helm validation，从模板层拒绝空节点、非 8 GPU 形态和 P/D 节点交集。
- Evidence:
  - 新增 `examples/deployment/deepseek-v4-flash-pd/templates/validations.yaml`。
  - validation 强制 `global.gpuCount == 8`、`prefill.nodeAffinity.enabled == true`、`decode.nodeAffinity.enabled == true`、两侧 nodeAffinity values 非空。
  - validation 遍历 prefill node list，若任一节点也在 decode node list 中则 `fail`，错误包含 `prefill and decode nodeAffinity values must be disjoint`。
  - 命令引用 `C13` 增加同节点负例渲染，要求 Helm validation 必须拒绝。

### P16: Static validation after rustup and placement fixes

- Summary: 当前修正通过静态验证、tag script 验证、Helm lint、正常渲染和同节点负例渲染。
- Evidence:
  - `bash -n build_rust.sh` 通过。
  - `git diff --check` 通过。
  - `python3 scripts/ci/get_byteiaas_image_tag.py --mode dev --image-flavor openai --cuda-suffix cu130 --timestamp 202606291700` 输出 `v0.10.1.1.iaas.dev.202606291700-cu130`。
  - `python3 scripts/ci/get_byteiaas_image_tag.py --mode dev --image-flavor openai-devel --cuda-suffix cu130 --timestamp 202606291700` 输出 `v0.10.1.1.iaas.dev.202606291700-openai-devel-cu130`。
  - `helm lint examples/deployment/deepseek-v4-flash-pd ...` 通过，`1 chart(s) linted, 0 chart(s) failed`。
  - 正常 `helm template` 输出包含 `kind: StormService`、`node-prefill-8gpu`、`node-decode-8gpu`、`oniond download model`、prefill/decode `nvidia.com/gpu: 8` request/limit。
  - 同节点负例 `helm template ... prefill.nodeAffinity.values[0]=same-node decode.nodeAffinity.values[0]=same-node` 失败，错误为 `prefill and decode nodeAffinity values must be disjoint because each role requests 8 GPUs; overlapping node: same-node`。

### P17: Second ByteIAAS workflow passed wheel and failed image on get-pip proxy 504

- Summary: 第二次 ByteIAAS workflow run `28361351639` 越过 rustup bootstrap；`build-wheel / build-wheel` 成功，但 `build-image / build-and-publish-image` 在 `vllm-base` 阶段下载 `GET_PIP_URL` 时遇到 proxy 504，未产出可部署镜像。
- Evidence:
  - Run id: `28361351639`。
  - Wheel job id `84016509746` conclusion: `success`。
  - Image job id `84016509717` conclusion: `failure`。
  - Image job API log path: `/tmp/byteiaas-vllm-84016509717-api.log`。
  - 关键日志：`curl: (56) Received HTTP code 504 from proxy after CONNECT`，失败命令为 `curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION}`，Docker build 返回 `exit code: 1`。

### P18: Hardened get-pip download for image build retry

- Summary: 将 `docker/Dockerfile` 中 `GET_PIP_URL` 下载改为有 retry/timeout 的临时文件执行，避免 proxy 504 或半开连接直接导致 image build 失败；同时给 rustup installer 下载补 `connect-timeout` 和 `max-time`。
- Evidence:
  - `docker/Dockerfile` 将 `curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION}` 改为 `curl --fail --show-error --location --retry 5 --retry-all-errors --retry-delay 5 --connect-timeout 20 --max-time 300 -o /tmp/get-pip.py ${GET_PIP_URL}`，随后 `python${PYTHON_VERSION} /tmp/get-pip.py` 并删除临时文件。
  - `build_rust.sh` 的 rustup installer curl 追加 `--connect-timeout 20 --max-time 300`。
  - `bash -n build_rust.sh` 通过。
  - `git diff --check` 通过。

### P19: ByteIAAS workflow succeeded and deployment render passed with real image

- Summary: 第三次 ByteIAAS workflow run `28389076984` 成功，产出可部署 `openai` 和 `openai-devel` 镜像；随后使用真实 `openai` 镜像完成 Helm render/lint、runtime install forbidden pattern 检查和同节点负例检查。
- Evidence:
  - Run id: `28389076984`，workflow conclusion: `success`。
  - Image job id `84111193573` conclusion: `success`；wheel job id `84111193610` conclusion: `success`。
  - Image job API log path: `/tmp/byteiaas-vllm-84111193573-api.log`。
  - Published openai image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606300110-cu130`。
  - Openai digest: `sha256:574c3dc2023be9300df8e699994798f76e3f048bff81f3e6719e8726197de113`。
  - Published openai-devel image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606300110-openai-devel-cu130`。
  - Openai-devel digest: `sha256:93e3326c6ca055a7e6986305ceb77a8b77d611bc2f9c4f92c52d1d2305bc6fd4`。
  - `helm lint examples/deployment/deepseek-v4-flash-pd ...` 通过。
  - Render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`。
  - Render 使用同一 `global.image`，包含 `kind: StormService`、`oniond download model`、prefill/decode `nvidia.com/gpu: 8`，并且同节点负例被 Helm validation 拒绝。

### P20: dev-cluster preflight and GPU permit granted

- Summary: `dev-cluster` preflight 通过，StormService CRD 存在，选择两个不同 8-GPU 节点并成功获取 16 GPU permit。
- Evidence:
  - `envctl info dev-cluster` 显示 kubeconfig 存在；`envctl validate dev-cluster` 返回 `OK dev-cluster`。
  - CRD `stormservices.orchestration.aibrix.ai` 存在，created at `2026-04-07T11:59:19Z`。
  - 当前 Running 8-GPU workloads 占用节点：`192.168.1.143`、`192.168.1.146`、`192.168.1.149`、`192.168.1.154`、`192.168.1.220`。
  - 选择 `PREFILL_NODE=192.168.1.148`、`DECODE_NODE=192.168.1.186`、`ROUTER_NODE=192.168.1.186`；两节点 allocatable GPU 均为 8，且 `UNSCHEDULABLE=false`。
  - Workspace-env session id: `codex-vllm-dsv4-flash-pd-20260630-093559-214441`。
  - Permit id: `3813ef32-5e13-42b7-9a1c-a36aa463dd5b`，status `granted`，requested GPUs `16`。

### P21: Added workspace-env labels to deployment template

- Summary: 为满足 workspace-env GPU resource tracking 要求，chart 增加 `workspaceEnv.sessionId/owner/purpose` values，并把对应 labels 注入 prefill/decode StormService pod template 和 router Deployment pod template。
- Evidence:
  - 修改 `examples/deployment/deepseek-v4-flash-pd/values.yaml`，新增 `workspaceEnv`。
  - 修改 `templates/stormservice.yaml`，在 prefill/decode `template.metadata.labels` 添加 `workspace-env/session-id`、`workspace-env/owner`、`workspace-env/purpose`。
  - 修改 `templates/router.yaml`，在 router pod template labels 添加同样标签。
  - Actual-node render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd-actual-nodes.yaml`。
  - Render 验证包含 workspace-env labels、`PREFILL_NODE=192.168.1.148`、`DECODE_NODE=192.168.1.186`、prefill/decode `nvidia.com/gpu: 8`，且无 runtime install forbidden pattern。

### P22: dev-cluster first deployment proved P/D different-node placement but image startup failed

- Summary: 首轮 `dev-cluster` 部署使用新镜像和 `StormService` 成功创建 1P1D 形态；实际调度证明 prefill 和 decode 分别占用不同 8-GPU 节点。服务未进入 benchmark，因为 prefill/decode engine 初始化失败于镜像内缺少 `vllm._moe_C` 模块。
- Evidence:
  - Helm release `dsv4-flash-pd` 部署到 namespace `vllm-dsv4-flash-pd`。
  - Render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd-actual-nodes.yaml`。
  - Render 中 prefill nodeAffinity values 为 `192.168.1.148`，decode nodeAffinity values 为 `192.168.1.186`，router nodeAffinity values 为 `192.168.1.186`。
  - Render 中 prefill/decode containers 均包含 `requests.nvidia.com/gpu: 8` 和 `limits.nvidia.com/gpu: 8`。
  - Cluster evidence: `dsv4-flash-pd-roleset-4s2b6-prefill-7df8bb5fcd-0` 调度到 `192.168.1.148`，container `prefill` 请求 8 GPU。
  - Cluster evidence: `dsv4-flash-pd-roleset-4s2b6-decode-5498c49684-0` 调度到 `192.168.1.186`，container `decode` 请求 8 GPU。
  - Cluster evidence: router `dsv4-flash-pd-router-6c9f646bcf-l7cwz` 调度到 `192.168.1.186`，不请求 GPU。
  - Onion init logs 显示 prefill/decode 均执行模型准备并因已有完整目录幂等跳过：`Model directory /data01/DeepSeek-V4-Flash is already complete, skip Onion download.`
  - Failure evidence: prefill/decode previous logs 均包含 `RuntimeError: Worker failed with error 'No module named 'vllm._moe_C''` 和 `RuntimeError: Engine core initialization failed.`
  - Latest evidence saved under `artifacts/2026-06-29-vllm-dsv4-flash-pd/logs/` and `artifacts/2026-06-29-vllm-dsv4-flash-pd/cluster/`.
  - 按 workspace-env 规则，已执行 `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd` 并删除 namespace；`namespace-after-cleanup.txt` 记录 namespace 不存在。
  - Permit `3813ef32-5e13-42b7-9a1c-a36aa463dd5b` 已释放，release artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/permit-release.json`。

### P23: Superseded runtime fallback attempt for `_moe_C`

- Summary: ByteIAAS workflow 产出的 wheel 只包含 `vllm/_moe_C_stable_libtorch.abi3.so`，与 `setup.py` 和 `current_platform.import_kernels()` 中稳定 libtorch 扩展路径一致；当时曾尝试在 `vllm/_custom_ops.py::topk_hash_softplus_sqrt` 中加入 `_moe_C` 到 `_moe_C_stable_libtorch` 的 runtime fallback。该路线已在 P26 被用户明确拒绝，不得作为后续实现依据，必须从待发布分支移除。
- Evidence:
  - Wheel artifact downloaded from workflow run `28389076984` under `/tmp/vllm-wheel-28389076984/`。
  - Wheel content includes `vllm/_moe_C_stable_libtorch.abi3.so` and does not include `vllm/_moe_C.abi3.so`。
  - `setup.py` 的 package extraction list 和 extension list 均包含 `vllm._moe_C_stable_libtorch`。
  - `vllm/platforms/interface.py::import_kernels` 已用 `contextlib.suppress(ImportError)` 同时尝试 `vllm._moe_C` 和 `vllm._moe_C_stable_libtorch`。
  - 修改文件：`vllm/_custom_ops.py`。
  - Historical validation: `python3 -m py_compile vllm/_custom_ops.py` 当时通过，但该验证不再构成可接受依据。

### P24: ByteIAAS workflow rerun queued without runner after 30 minutes

- Summary: `_moe_C_stable_libtorch` fallback 修复提交后，已触发新的 ByteIAAS dev image workflow，但 30 分钟内仍未分配 runner；本地 `gh run watch` 已停止，GitHub run 未取消，等待外部 runner 队列恢复。
- Evidence:
  - Fix commit: `51b135cef854e6d72cb704068644c52d047706e5`。
  - Branch pushed: `codex/vllm-dsv4-fork-base-byteiaas-build`。
  - Workflow run id: `28414886195`。
  - Run URL: `https://github.com/bytedance-iaas/vllm/actions/runs/28414886195`。
  - `build-image / build-and-publish-image` job id `84195572689` status `queued`。
  - `build-wheel / build-wheel` job id `84195572744` status `queued`。
  - 结构化状态保存到 `artifacts/2026-06-29-vllm-dsv4-flash-pd/workflow/run-28414886195-queued-after-30m.json`。
  - 同 workflow 另有 scheduled `iaas_main` run `28397967919` 也处于 `queued`，看起来是 ByteIAAS runner 队列容量问题，不是本分支 concurrency 阻塞。
  - 当前未占用 `dev-cluster` GPU；首轮 permit 已释放，namespace 已删除。

### P25: Resume check confirmed ByteIAAS runner is still externally blocked

- Summary: 2026-06-30 resume 后重新检查 run `28414886195`，仍为 `queued`，且 `updatedAt` 仍停留在创建时间；当前没有进入 wheel/image build，因此不能提取新镜像、不能重新部署、不能 benchmark，也不能更新 `iaas_main`。
- Evidence:
  - `gh run view 28414886195 --repo bytedance-iaas/vllm` 返回 status `queued`，headSha `51b135cef854e6d72cb704068644c52d047706e5`。
  - Job `84195572689` (`build-image / build-and-publish-image`) status `queued`。
  - Job `84195572744` (`build-wheel / build-wheel`) status `queued`。
  - `gh api repos/bytedance-iaas/vllm/actions/runners` 显示仓库只有 1 台 runner `vllm-byteiaas-build-01`，status `online`，busy `true`，labels 包含 `x64-vllm-wheel-build-node` 和 `x64-vllm-docker-build-node`。
  - `gh api --method GET repos/bytedance-iaas/vllm/actions/runs -f per_page=30` 过滤 `queued/in_progress` 后，仅看到 run `28414886195` 和 scheduled `iaas_main` run `28397967919` 处于 `queued`，没有当前仓库内 `in_progress` run。
  - 本次尝试 `gh run list --status ...` 失败，因为本机 `gh` 版本不支持 `--status`；随后改用 REST API 查询成功。
  - 当前 `dev-cluster` namespace `vllm-dsv4-flash-pd` 不存在，首轮 permit 已释放，没有 GPU 占用。

### P26: User rejected runtime fallback source modification; plan switched back to build-only changes

- Summary: 用户在 2026-06-30 明确要求计划修改：不希望在 vLLM runtime Python 源码中加入 `import vllm._moe_C` / fallback 到 `vllm._moe_C_stable_libtorch`；vLLM 源码修改应只包含构建过程中遇到的问题。因此此前 `vllm/_custom_ops.py` fallback 提交和基于该提交触发的 workflow run 被标记为无效路线。
- Evidence:
  - 被拒绝的 runtime 修改提交：`51b135cef854e6d72cb704068644c52d047706e5`。
  - 被拒绝路线触发的 workflow run：`28414886195`，headSha `51b135cef854e6d72cb704068644c52d047706e5`。
  - 主计划新增全局约束：不得用 `vllm/` Python runtime fallback、模型逻辑、算子调用逻辑或调度逻辑修改来绕过部署失败。
  - 主计划更新 M5：下一步必须执行 `C9A`，取消或忽略 run `28414886195`，并从待发布分支移除 `vllm/_custom_ops.py` fallback 修改。
  - 命令引用新增 `C9A`：撤销 runtime fallback 路线并回到 build-only 修复。
  - 后续 `_moe_C` import 类问题只能从 build/package artifact 层处理，例如 `setup.py`、CMake、wheel extraction/package data 或 Dockerfile wheel 安装路径；如果这些路径不能解决，应停止并报告 blocker，而不是改 vLLM runtime 逻辑。
- Prevention note: 后续执行前必须先检查 `git diff <fork_sha>...HEAD -- vllm/`，确认没有 runtime Python fallback 修改；新镜像不能来自 run `28414886195` 或任何包含 `51b135cef` runtime fallback 的 ref。

### P27: Tightened plan to prohibit `_moe_C` runtime fallback

- Summary: 按用户最新要求再次收紧计划：不允许在 `vllm/_custom_ops.py` 或其它 `vllm/**` runtime Python 文件中加入 `_moe_C` 到 `_moe_C_stable_libtorch` 的 `ImportError` fallback；vLLM 源码修改只允许解决构建过程中暴露的问题。
- Evidence:
  - 主计划 `Global Constraints` 明确禁止 runtime fallback，并把 M5 从“已完成”改为“需重新打开”：成功构建的旧镜像只作为历史事实，不能进入 benchmark 或更新 `iaas_main`。
  - 命令引用 `C9A` 从整提交 `git revert` 改为 `git restore --source="${INVALID_COMMIT}^" -- vllm/_custom_ops.py`，只回滚 runtime source 文件，保留计划日志和约束记录。
  - `C9A` 验证改为禁止 `_moe_C_stable_libtorch` 和 `except ImportError` 出现在 `vllm/_custom_ops.py` 中；fork baseline 原有的 hard import `vllm._moe_C` 不被视为违规。
  - 后续处理 `No module named 'vllm._moe_C'` 只能进入 build/package artifact 调查：`setup.py`、CMake、wheel extraction/package data、扩展产物命名或 Dockerfile wheel 安装路径。

### P28: C9A completed; runtime fallback removed from pending branch

- Summary: 按 `C9A` 执行撤销 runtime fallback 路线：取消无效 workflow run，恢复 `vllm/_custom_ops.py` 到 fork baseline 的 hard import，并把后续方向固定为 build/package artifact 修复。当前不使用 `superpowers:subagent-driven-development`；C9A 和接下来的 `_moe_C` package 修复都集中在同一分支、同一 build/runtime failure chain 和同一组计划文件，主线程串行执行能减少并发写冲突。后续如只读调查范围扩大，可再把 review lane 分给 subagent。
- Evidence:
  - `gh run view 28414886195 --repo bytedance-iaas/vllm --json status,conclusion,headSha,url` 初始返回 status `queued`、headSha `51b135cef854e6d72cb704068644c52d047706e5`。
  - 已执行 `gh run cancel 28414886195 --repo bytedance-iaas/vllm`；复查返回 status `completed`、conclusion `cancelled`。
  - 已执行 `git restore --source="51b135cef854e6d72cb704068644c52d047706e5^" -- vllm/_custom_ops.py`。
  - `topk_hash_softplus_sqrt` 当前只包含 `import vllm._moe_C  # noqa: F401`，不包含 `_moe_C_stable_libtorch` fallback。
  - `uv run --no-project python -m py_compile vllm/_custom_ops.py` 通过，遵守本仓库 AGENTS.md 中禁止 bare `python3` 的要求。
  - 首次本地验证使用 `rg "_moe_C_stable_libtorch|except ImportError" vllm/_custom_ops.py`，误伤文件顶部既有 `torch.library.register_fake` 兼容 fallback；已修正命令引用，只检查 `_moe_C_stable_libtorch` 和 `topk_hash_softplus_sqrt` 函数范围内的 `try`/`except ImportError`。
- Prevention note: 后续验证不能用全文件 `except ImportError` 判断 runtime fallback；必须限定到 `_moe_C`/`_moe_C_stable_libtorch` 目标或具体函数范围。

### P29: Superseded build-side `_moe_C` rename attempt

- Summary: `_moe_C` 缺失曾尝试按 build/package artifact 层修复，不修改 `vllm/**` runtime Python fallback。修复方式是继续使用 `csrc/libtorch_stable/moe/**` 的 stable ABI MoE extension 源码，但将 Python extension module 名导出为 fork runtime baseline 期望的 `vllm._moe_C`，并让 C++ init 入口跟随 CMake target 名。该路线在 P30 被上游对比推翻，已撤销，不得继续构建或发布。
- Touched files:
  - `CMakeLists.txt`：MoE stable ABI extension target 从 `_moe_C_stable_libtorch` 改为 `_moe_C`，对应 compile definitions/link target 同步改名。
  - `csrc/libtorch_stable/moe/torch_bindings.cpp`：`REGISTER_EXTENSION(_moe_C_stable_libtorch)` 改为 `REGISTER_EXTENSION(TORCH_EXTENSION_NAME)`，使 build target 名决定 `PyInit_*`；torch library fragment/impl 仍是 `_moe_C`，算子 schema 和实现未改。
  - `setup.py`：`ext_modules` 改为 `CMakeExtension(name="vllm._moe_C")`；precompiled exact member 从 `vllm/_moe_C_stable_libtorch.abi3.so` 改为 `vllm/_moe_C.abi3.so`，避免 wheel 同时携带两个会注册同一 torch library 的 module。
- Evidence:
  - `rg -n "_moe_C_stable_libtorch|_moe_C" setup.py CMakeLists.txt csrc/libtorch_stable/moe/torch_bindings.cpp vllm/platforms/interface.py vllm/_custom_ops.py` 显示构建/打包侧只导出 `vllm._moe_C`；仅 `vllm/platforms/interface.py` 保留 suppress 的 `vllm._moe_C_stable_libtorch` import，这是 fork baseline runtime 逻辑且本次未改。
  - `uv run --no-project python -m py_compile setup.py vllm/_custom_ops.py` 通过。
  - `git diff --check` 通过。
  - C7 workflow/tag/Onion 静态验证通过：四个 ByteIAAS workflow YAML parsed；tag script 输出 `v0.10.1.1.iaas.dev.202606301136-cu130` 和 `v0.10.1.1.iaas.dev.202606301136-openai-devel-cu130`；Dockerfile 中存在 build-time `onion-ai-data`/`oniond` 安装和 `command -v oniond` 校验。
  - 本机 `docker info` 仍失败：`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`；因此不执行本地 C8，下一步走 C9 ByteIAAS workflow。
- Prevention note: 不要通过复制或同时打包 `_moe_C_stable_libtorch` 与 `_moe_C` 解决该问题；两个 module 若都被 import，可能重复注册 `_moe_C` torch library。

### P30: Upstream comparison invalidated build-side rename; waiting on runtime alignment approval

- Summary: 用户指出 commit `4fcea785fd66874046f9b828eb2fad7fbd527a63` 比 runtime fallback 更异常后，已停止该路线。对比 `vllm-project/vllm` upstream main、`v0.10.1.1`、`v0.10.1`、`v0.10.0` 证明：上游不出问题是因为每条代码线内部自洽，而当前 fork base 混入了 stable build artifact 与残留 hard import。
- Evidence:
  - 已取消 workflow run `28418542564`；复查 `gh run view 28418542564 --repo bytedance-iaas/vllm --json status,conclusion,headSha,url` 返回 status `completed`、conclusion `cancelled`、headSha `4fcea785fd66874046f9b828eb2fad7fbd527a63`。
  - `git fetch --no-tags https://github.com/vllm-project/vllm.git refs/heads/main:refs/tmp/upstream-vllm-main refs/tags/v0.10.0:refs/tmp/upstream-vllm-v0.10.0 refs/tags/v0.10.1:refs/tmp/upstream-vllm-v0.10.1 refs/tags/v0.10.1.1:refs/tmp/upstream-vllm-v0.10.1.1` 成功；`git fetch ... 4fcea785fd66874046f9b828eb2fad7fbd527a63` 也能取到对象，但该 SHA 仅由当前本地/远端 integration branch 包含。
  - upstream main：`CMakeLists.txt`、`setup.py`、`csrc/libtorch_stable/moe/torch_bindings.cpp` 仍构建/打包 `vllm._moe_C_stable_libtorch`；`vllm/platforms/interface.py::import_kernels()` import stable extension；`vllm/_custom_ops.py::topk_hash_softplus_sqrt` 不再 hard import `vllm._moe_C`，直接调用 `torch.ops._moe_C.topk_softplus_sqrt`。
  - upstream `v0.10.0`/`v0.10.1`/`v0.10.1.1`：构建/打包 `vllm._moe_C`，且 runtime import `vllm._moe_C`；module 名和 import 名一致。
  - fork base `cde7799cc66c5a4cb349156a3ca3228f9798dbc9`：构建/打包 `vllm._moe_C_stable_libtorch`，但 `topk_hash_softplus_sqrt` 仍 hard import `vllm._moe_C`，这是首轮部署 `No module named 'vllm._moe_C'` 的直接不一致点。
  - 已按命令引用 `C9B` 恢复 `CMakeLists.txt`、`setup.py`、`csrc/libtorch_stable/moe/torch_bindings.cpp` 到 `4fcea785fd66874046f9b828eb2fad7fbd527a63^`，即 upstream main/fork baseline 的 stable build artifact 命名。
- Current blocker: 合理修复应是按 upstream main 对齐 `vllm/_custom_ops.py::topk_hash_softplus_sqrt`，删除 fork 残留的 `import vllm._moe_C` hard import，让 `current_platform.import_kernels()` 负责加载 `vllm._moe_C_stable_libtorch`。这不是 fallback，但属于 `vllm/**` runtime Python 逻辑修改；用户此前要求 vLLM 源码修改只包含构建过程中遇到的问题，因此执行前需要用户确认。
- Prevention note: 不再尝试 build-side rename 或 runtime fallback。再次推进前必须改变假设：这是 fork 与 upstream main 的 runtime/source 对齐问题，不是 wheel artifact 缺文件问题。

### P31: Approved upstream-main alignment for hash topk MoE import

- Summary: 用户确认按 upstream main 对齐修复。该问题不是社区 fork base `a331589394d95d462f2993c32fe3c063146c74e8` 原有问题，而是 `wangyicong52/vllm.git` `dev/dsv4-mooncake-pp-megamoe` fork 后的提交 `f7c4c621d2595cba06a135306b7709b6d5af7804` 引入：它在 `topk_hash_softplus_sqrt` 中新增了非 suppress 的 `import vllm._moe_C`，但 build artifact 已经是 `_moe_C_stable_libtorch`。本次只删除该 hard import，不做 fallback，不改 build artifact，不改算子语义。
- Evidence:
  - `git merge-base refs/tmp/wang-dsv4-mooncake-pp-megamoe refs/tmp/upstream-vllm-main` 返回 `a331589394d95d462f2993c32fe3c063146c74e8`。
  - 在 base `a331589394d95d462f2993c32fe3c063146c74e8` 上：build artifact 是 `_moe_C_stable_libtorch`，`import_kernels()` import `vllm._moe_C_stable_libtorch`，`topk_hash_softplus_sqrt` 不 hard import `vllm._moe_C`。
  - `git show f7c4c621d -- vllm/_custom_ops.py` 显示新增 `import vllm._moe_C  # noqa: F401`。
  - `git show 103f86c7b -- vllm/platforms/interface.py` 显示新增 suppress 的 legacy `vllm._moe_C` import；该点不会直接失败，因为 `ImportError` 被 suppress。
  - 当前修改删除 `vllm/_custom_ops.py::topk_hash_softplus_sqrt` 内的 hard import，保留 `hash_indices_table` dtype 兼容逻辑和 `torch.ops._moe_C.topk_softplus_sqrt` 调用。
- Validation plan: 执行命令引用 `C9C`；若通过，提交并按 `C9` 重启 ByteIAAS workflow。

### P32: Successful user build image accepted for next deployment validation

- Summary: 用户已在 `codex/vllm-dsv4-fork-base-byteiaas-build` 上继续修正构建并完成成功 ByteIAAS workflow run `28442949331`。该 run 的 headSha 是当前分支 HEAD `7186cf328963d12daabe8ee47087a29111c0cb75`，且包含此前批准的 upstream-main alignment 提交 `d3f23315c`；候选 `openai-devel` 镜像 registry/static 检查通过。当前不需要重新构建，后续直接使用该镜像进入 C13-C16 部署验证。
- Evidence:
  - `git rev-parse HEAD` 返回 `7186cf328963d12daabe8ee47087a29111c0cb75`，当前分支为 `codex/vllm-dsv4-fork-base-byteiaas-build`。
  - `git merge-base --is-ancestor d3f23315c 7186cf328963d12daabe8ee47087a29111c0cb75` 成功，说明删除 `topk_hash_softplus_sqrt` hard import 的 upstream-main alignment 已包含在当前构建 SHA 中。
  - `rg -n "import vllm\\._moe_C|_moe_C_stable_libtorch|topk_hash_softplus_sqrt" vllm/_custom_ops.py setup.py CMakeLists.txt cmake -S` 显示 `vllm/_custom_ops.py::topk_hash_softplus_sqrt` 不再 hard import `vllm._moe_C`；build/package artifact 仍为 `_moe_C_stable_libtorch`。
  - `gh run view 28442949331 --repo bytedance-iaas/vllm --json status,conclusion,headSha,url,createdAt,updatedAt,jobs` 返回 `status=completed`、`conclusion=success`、`headSha=7186cf328963d12daabe8ee47087a29111c0cb75`；`build-wheel / build-wheel` 和 `build-image / build-and-publish-image` 两个 job 均为 success。
  - Workflow image job logs 记录发布 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-cu130` 和 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130`；其中 `openai-devel` digest 为 `sha256:f38ccbf3f1b126e1aaf5621ed6abf51b590424ac48fad74d03e3f24b73a153a7`。
  - `docker buildx imagetools inspect iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130` 返回 OCI image index digest `sha256:f38ccbf3f1b126e1aaf5621ed6abf51b590424ac48fad74d03e3f24b73a153a7`，包含 `linux/amd64` manifest `sha256:429013e1edb16888a0a0ad3776b5ae6366eb2cbea733b6107dff88021cfe65f0`。
  - 本机 `docker info` 无法访问 `/var/run/docker.sock`，`ctr version` 无法访问 `/run/containerd/containerd.sock`；因此本步骤只做 registry/static image check，不做本地容器运行、镜像内 import/CLI smoke，也不向构建流程补充 smoke。
- Decision: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130` 可作为下一轮 `dev-cluster` 部署验证候选镜像；runtime 内容是否满足要求由 C13 render、C14 preflight、C15 deploy 和 C16 真实 router smoke 验证。

### P33: Re-aligned deployment template with current servingkit chart semantics

- Summary: 用户指出计划要求部署方式不应与 servingkit `perf/vllm_dsv4/vllm/deepseek/deepseek-v4-flash-pd` 严重不对齐。已重新 fetch servingkit `perf/vllm_dsv4` 并以 SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 为当前参考基准，收敛 vLLM 仓库中的部署模板和计划说明。
- Evidence:
  - `git fetch origin perf/vllm_dsv4:refs/tmp/perf-vllm-dsv4` 成功，`git rev-parse refs/tmp/perf-vllm-dsv4` 返回 `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69`。
  - servingkit 当前参考 values 的关键语义：prefill `kvTransfer.role=kv_producer`、`port=8000`、`dataParallelSize=1`、`tensorParallelSize=4`、`pipelineParallelSize=2`、`all2allBackend=""`、`enableExpertParallel=false`、`maxNumBatchedTokens=32768`、`maxNumSeqs=16`；decode `kvTransfer.role=kv_consumer`、`port=8001`、`dataParallelSize=8`、`cpKvCacheInterleaveSize=256`、`enableExpertParallel=true`、`moeBackend=deep_gemm_mega_moe`、`enablePrefixCaching=true`、MTP speculative config；router `serviceDiscovery.enabled=false`、静态 `--prefill/--decode`、`intraNodeDataParallelSize=1`。
  - 已更新 `examples/deployment/deepseek-v4-flash-pd/values.yaml`：移除调试残留 `prefill.args.maxNumBatchedTokens=2048` 和 `env.decode.NVSHMEM_QP_DEPTH=2048`；默认 P/D/router 参数回到 servingkit 当前语义；保留计划允许的差异：`global.image` 由执行时填入、Onion 替代 TOS、节点参数化、删除 runtime install/hotfix。
  - 已更新 `_helpers.tpl` 和 `configmap.yaml`：新增 `kvTransferConfigJsonForRole`，prefill/decode 分别渲染 `kv_producer`/`kv_consumer`；支持 servingkit 当前 TP/PP/CP/MegaMoE/MTP/静态 router 参数；保留 `command -v vllm-router` 检查但不做 runtime install。
  - 已更新主计划 M7 和命令引用 C13/C15：chart 默认值跟随 servingkit 当前语义；当时曾把 `vllm.maxModelLen=66000` 和 `decode.args.maxNumSeqs=512` 作为执行期覆盖，该点已在 P34 被用户要求撤销，后续只保留 `decode.args.maxNumSeqs=512`。
  - `helm template dsv4-flash-pd examples/deployment/deepseek-v4-flash-pd ... --set vllm.maxModelLen=66000 --set decode.args.maxNumSeqs=512` 曾通过，并保存到 `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`；该 render 已被 P34 supersede，后续不得使用带 `vllm.maxModelLen=66000` 的 manifest。forbidden pattern `runtimePatch|git clone|pip install|apt install|install_deepgemm|ensure_pip_package|wheelURL|wheelPath|/tmp/vllm-runtime-patch|vllm-router.*pip` 无命中。
  - rendered manifest 关键参数检查通过：`kv_producer`、`kv_consumer`、prefill `--tensor-parallel-size "4"`、`--pipeline-parallel-size "2"`、decode `--port "8001"`、`--data-parallel-size "8"`、`--cp-kv-cache-interleave-size "256"`、`--moe-backend "deep_gemm_mega_moe"`、`--speculative-config`、router `--prefill http://192.168.1.148:8000 8998`、`--decode http://192.168.1.186:8001`、`--intra-node-data-parallel-size "1"` 均存在。
  - 同节点负例 `prefill.nodeAffinity.values[0]=same-node` 与 `decode.nodeAffinity.values[0]=same-node` 被 Helm validation 拒绝，错误包含 `prefill and decode nodeAffinity values must be disjoint`。
  - `helm lint examples/deployment/deepseek-v4-flash-pd ...` 通过，只有 `Chart.yaml: icon is recommended` informational message；`git diff --check` 通过。
- Decision: 后续部署验证不得继续使用之前发散的 DP8 prefill/high_throughput、decode 8000、router service discovery、router intra-node DP8 或调试注入的 `NVSHMEM_QP_DEPTH=2048` 路线，除非用户再次明确要求偏离 servingkit 当前 chart。

### P34: Removed explicit vllm.maxModelLen override

- Summary: 用户明确要求不使用 `vllm.maxModelLen=66000`。已更新主计划和命令引用，C13/C15 不再设置 `vllm.maxModelLen`，部署沿用 servingkit 当前 `maxModelLen: null` 行为；rendered command 中不应出现 `--max-model-len`。
- Evidence:
  - 从 C13/C15 命令引用删除 `MAX_MODEL_LEN="${MAX_MODEL_LEN:-66000}"` 和 `--set vllm.maxModelLen="${MAX_MODEL_LEN}"`。
  - 主计划 M7 的允许差异改为只保留 `decode.args.maxNumSeqs=512` 作为 BS512/1.5k benchmark 显式覆盖，并新增禁止使用 `vllm.maxModelLen=66000` 的说明。
  - `examples/deployment/deepseek-v4-flash-pd/values.yaml` 保持 `vllm.maxModelLen: null`。
  - 重新执行 `helm template dsv4-flash-pd examples/deployment/deepseek-v4-flash-pd ... --set decode.args.maxNumSeqs=512` 并覆盖 `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`；`rg -- "--max-model-len|\"66000\"|vllm.maxModelLen"` 对 rendered manifest 无命中。
  - 新 rendered manifest 仍保留 servingkit 对齐关键项：prefill `--tensor-parallel-size "4"`、`--pipeline-parallel-size "2"`、`kv_role":"kv_producer"`；decode `--cp-kv-cache-interleave-size "256"`、`--moe-backend "deep_gemm_mega_moe"`、`--max-num-seqs "512"`、`kv_role":"kv_consumer"`；router 静态 `--prefill http://192.168.1.148:8000 8998`、`--decode http://192.168.1.186:8001`、`--intra-node-data-parallel-size "1"`。
- Decision: 后续 render/deploy/benchmark 不得通过 values、命令行或临时 patch 显式设置 `vllm.maxModelLen=66000`；如果 64k/1 请求因此受限，应记录为与 servingkit 对齐后的实际行为，而不是重新加 `--max-model-len`。

### P35: Second deployment failed on CUDA 12 Mooncake wheel inside CUDA 13 image

- Summary: 使用用户成功构建的候选镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130` 执行第二轮 `dev-cluster` 部署验证。部署模板、节点约束、Onion init 和 router 静态 P/D 模式均按预期生效，但 prefill/decode 在 vLLM 初始化阶段失败。根因是镜像内通用 `mooncake-transfer-engine` wheel 依赖 CUDA 12 runtime，在 CUDA 13.0.2 image 中导入 `mooncake.engine` 时报 `libcudart.so.12` 缺失；这是镜像构建依赖问题，不应通过 runtime hotfix/install 规避。
- Deployment evidence:
  - C14 session: `codex-vllm-dsv4-flash-pd-20260630-214036-660370`。
  - GPU permit: `96a89e60-ddba-41a0-89a9-b06ecbc07379`，请求 16 GPU，部署失败后已释放。
  - 选定节点：prefill `192.168.1.148`，decode `192.168.1.186`，router `192.168.1.186`。
  - Namespace: `vllm-dsv4-flash-pd`；Helm release: `dsv4-flash-pd`。
  - Prefill pod `dsv4-flash-pd-roleset-dncp6-prefill-7b899c7478-0` 调度到 `192.168.1.148`，container `prefill` 请求 8 GPU。
  - Decode pod `dsv4-flash-pd-roleset-dncp6-decode-6497676765-0` 调度到 `192.168.1.186`，container `decode` 请求 8 GPU。
  - Router pod `dsv4-flash-pd-router-5b9fb8f67b-2lssk` 调度到 `192.168.1.186`，不请求 GPU。
  - Onion init 已完成并因已有模型目录幂等跳过；router 启动为静态 P/D URL 模式：`prefill_urls: [("http://192.168.1.148:8000", Some(8998))]`，`decode_urls: ["http://192.168.1.186:8001"]`，`discovery: None`。
- Failure evidence:
  - 失败证据保存于 `artifacts/2026-06-29-vllm-dsv4-flash-pd/failure-20260630-214421/` 和 `artifacts/2026-06-29-vllm-dsv4-flash-pd/failure-latest/`。
  - Prefill previous log `failure-latest/dsv4-flash-pd-roleset-dncp6-prefill-7b899c7478-0-prefill-previous.log` 包含 `RuntimeError: Mooncake is not available`。
  - 对应代码路径 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` 在 `from mooncake.engine import TransferEngine` 导入失败后将 `TransferEngine=None`，随后抛出 `RuntimeError("Mooncake is not available")`。
  - 在 router container 使用同一镜像检查：`mooncake-transfer-engine` 已安装，版本 `0.3.11.post1`；`import mooncake` 可找到 `/usr/local/lib/python3.12/dist-packages/mooncake/__init__.py`；但 `from mooncake.engine import TransferEngine` 失败，错误为 `ImportError: libcudart.so.12: cannot open shared object file: No such file or directory`。
  - `uv pip install --system --dry-run 'mooncake-transfer-engine-cuda13==0.3.11'` 可解析到 CUDA 13 专用包，说明构建层可以改用 `mooncake-transfer-engine-cuda13`。
- Cleanup evidence:
  - 已中断等待中的 Helm 命令并执行 `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd`。
  - 已执行 `kubectl delete namespace vllm-dsv4-flash-pd --ignore-not-found --wait=true --timeout=180s`，`namespace-after-delete.txt` 记录 namespace 已不存在。
  - Permit `96a89e60-ddba-41a0-89a9-b06ecbc07379` 已释放，release artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/permit-release-after-mooncake-failure.json`。
  - Workspace-env registry active after cleanup 为 `[]`。
- Fix:
  - 已在 `docker/Dockerfile` 的 `INSTALL_KV_CONNECTORS` 分支中加入 CUDA 13 特化构建逻辑：当 `CUDA_MAJOR=13` 时卸载通用 `mooncake-transfer-engine`，并安装 `mooncake-transfer-engine-cuda13>=0.3.8`。
  - 该修复位于镜像构建阶段，不修改 `vllm/**` runtime Python 逻辑，不添加部署时安装，也不添加构建流程 import/CLI smoke。
- Decision:
  - 旧镜像 `v0.10.0.iaas.dev.202606302005-openai-devel-cu130` 不得继续用于 benchmark 或更新 `iaas_main`。
  - 下一步必须提交并推送 Dockerfile 修复，按 C9 重新触发 ByteIAAS workflow，等待新镜像产出后重新执行 C13-C16。

### P36: CUDA 13 Mooncake package fix rebuilt successfully

- Summary: 已提交并推送 Dockerfile 修复，重新触发 ByteIAAS dev image workflow。新 workflow 成功产出 openai/openai-devel 镜像；后续部署验证固定使用新的 `openai-devel` 镜像，不再使用 P35 中失败的旧镜像。
- Code and workflow evidence:
  - Fix commit: `aaa5f958a4e8b156d789963174b445cae239fa53`。
  - Commit message: `Fix CUDA13 Mooncake package for ByteIAAS image`。
  - Pushed branch: `codex/vllm-dsv4-fork-base-byteiaas-build`。
  - Workflow run id: `28449545514`。
  - Workflow URL: `https://github.com/bytedance-iaas/vllm/actions/runs/28449545514`。
  - `gh run view 28449545514 --repo bytedance-iaas/vllm --json status,conclusion,headSha,url,createdAt,updatedAt,jobs` 返回 `status=completed`、`conclusion=success`、`headSha=aaa5f958a4e8b156d789963174b445cae239fa53`。
  - Jobs: `build-image / build-and-publish-image` success；`build-wheel / build-wheel` success。
- Published images:
  - `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302152-cu130`，digest `sha256:ae9c4b999fac0c7e14ba10730d7963a0a7db648841003e8881370541ca82d8fd`。
  - `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302152-openai-devel-cu130`，digest `sha256:aaf3098637aa709286668aa380c74e6540c24b15ea14c3bc806530c56f7e6e2a`。
  - Job log evidence saved at `artifacts/2026-06-29-vllm-dsv4-flash-pd/workflow/job-84307773674-logs-api.txt`，其中包含 `Resolved openai-devel image tag v0.10.0.iaas.dev.202606302152-openai-devel-cu130` 和 `Published openai-devel image: iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302152-openai-devel-cu130`。
  - `docker buildx imagetools inspect` 通过：openai-devel image index digest 为 `sha256:aaf3098637aa709286668aa380c74e6540c24b15ea14c3bc806530c56f7e6e2a`，包含 `linux/amd64` manifest `sha256:16392b32883e2e4766b3f39d9dc7343af916913aea30222952cdb11a05b96a14`。
- Static validation before push:
  - `git diff --check` 通过。
  - `uv pip install --system --dry-run 'mooncake-transfer-engine-cuda13>=0.3.8'` 可解析到 `mooncake-transfer-engine-cuda13==0.3.11.post1`。
  - `helm template` 使用 `decode.args.maxNumSeqs=512`、不同 P/D 节点和旧候选 image 做 precommit render 通过；render 中无 `--max-model-len`、`66000`、runtime hotfix/install forbidden pattern。
  - `helm lint examples/deployment/deepseek-v4-flash-pd ...` 通过，仅有 `Chart.yaml: icon is recommended` informational message。
- Decision:
  - 新部署候选固定为 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302152-openai-devel-cu130`。
  - 下一步重新执行 C13-C16；如果该新镜像仍失败，需要按新的日志证据重新定位，不得回退到运行时安装或 runtime fallback。

### P37: Third deployment failed on missing DeepGEMM fork MegaMoE SM90 symbols

- Summary: 使用 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302152-openai-devel-cu130` 执行第三轮 `dev-cluster` 部署验证。Mooncake CUDA 13 包问题已越过，prefill 推进到 API server startup，router 以静态 P/D URL 模式启动并看到 prefill healthy；decode 在加载 DSV4 MegaMoE 权重时失败。根因是镜像内没有安装 servingkit chart 指向的 `wangyicong52/DeepGEMM` fork wheel，vLLM 回落到 vendored DeepGEMM 后缺少 `transform_weights_for_mega_moe_sm90_fp4`。
- Deployment evidence:
  - C14 session: `codex-vllm-dsv4-flash-pd-20260630-221836-711763`。
  - GPU permit: `bf793b14-0583-4840-80f0-0741b4a99fa4`，请求 16 GPU，部署失败后已释放。
  - 选定节点：prefill `192.168.1.148`，decode `192.168.1.186`，router `192.168.1.186`。
  - Namespace: `vllm-dsv4-flash-pd`；Helm release: `dsv4-flash-pd`。
  - Prefill pod `dsv4-flash-pd-roleset-jwnnx-prefill-548567f887-0` 调度到 `192.168.1.148`，container `prefill` 请求 8 GPU。
  - Decode pod `dsv4-flash-pd-roleset-jwnnx-decode-c46c79c8b-0` 调度到 `192.168.1.186`，container `decode` 请求 8 GPU。
  - Router pod `dsv4-flash-pd-router-7dbc4d6c5d-vxvg6` 调度到 `192.168.1.186`，不请求 GPU。
  - Onion init 已完成并因已有模型目录幂等跳过：`Model directory /data01/DeepSeek-V4-Flash is already complete, skip Onion download.`
  - Router static mode evidence: `prefill_urls: [("http://192.168.1.148:8000", Some(8998))]`，`decode_urls: ["http://192.168.1.186:8001"]`，`discovery: None`。
- Failure evidence:
  - Failure artifact directory: `artifacts/2026-06-29-vllm-dsv4-flash-pd/failure-deepgemm-20260630-222819/`。
  - Full previous decode log: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-debug-new-image/decode-previous-main-full.log`。
  - Decode log root cause: `NotImplementedError: The resolved DeepGEMM build is missing required MegaMoE symbols for SM90: ['transform_weights_for_mega_moe_sm90_fp4']. Update the DeepGEMM wheel/image.`
  - Stack path: `vllm/models/deepseek_v4/nvidia/model.py::finalize_mega_moe_weights -> DeepseekV4MegaMoEExperts._check_runtime_supported`。
  - Log also shows `deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm`，说明当前镜像没有外部 fork `deep_gemm`，只使用了 vLLM wheel 内的 vendored DeepGEMM。
- Cleanup evidence:
  - 已中断等待中的 Helm 命令并执行 `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd`。
  - 已删除 namespace `vllm-dsv4-flash-pd` 并复查不存在。
  - Permit `bf793b14-0583-4840-80f0-0741b4a99fa4` 已释放，release artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/permit-release-after-deepgemm-failure.json`。
  - Workspace-env registry active after cleanup 为 `[]`；未保留 GPU pod。
- DeepGEMM fork evidence:
  - `git ls-remote --tags https://github.com/wangyicong52/DeepGEMM.git refs/tags/deep_gemm-2.5.0-1wyc-vllm-mega-moe196439b72` 返回 `88965b078186ee7510ab9fc4f1d5ebc19adfa8d1`。
  - 同名 tag 源码中的 `deep_gemm/mega/__init__.py` 只有 `transform_weights_for_mega_moe`，不包含 `transform_weights_for_mega_moe_sm90` 或 `transform_weights_for_mega_moe_sm90_fp4`。
  - servingkit chart 已出现的 release wheel URL 可访问，wheel `deep_gemm-2.5.0-1wyc_vllm_mega_moe196439b72-cp312-cp312-linux_x86_64.whl` 包含 `deep_gemm/_C.cpython-312-x86_64-linux-gnu.so`，且 `deep_gemm/__init__.py` 导出 `transform_weights_for_mega_moe_sm90`、`transform_weights_for_mega_moe_sm90_fp4` 和 `fp8_fp4_mega_moe`。
- Fix:
  - 已在 `docker/Dockerfile` 的 `vllm-openai-base` 阶段新增可选 `DEEPGEMM_WHEEL_X86_64` build arg；当 `TARGETPLATFORM=linux/amd64` 且参数非空时，image build-time 执行 `uv pip install --system "${DEEPGEMM_WHEEL_X86_64}"`，并用 Python 检查 `transform_weights_for_mega_moe_sm90`、`transform_weights_for_mega_moe_sm90_fp4`、`fp8_fp4_mega_moe` 三个符号。
  - 已在 `.github/workflows/_byteiaas-build-and-publish-image.yml` 中为 ByteIAAS openai image build 传入 chart 已出现的 `wangyicong52/DeepGEMM` release wheel URL。
  - 该修复位于镜像构建阶段，不修改 `vllm/**` runtime Python 逻辑，不在部署模板中添加 wheel 下载或安装，也不引入运行时 hotfix。
- Decision:
  - 镜像 `v0.10.0.iaas.dev.202606302152-openai-devel-cu130` 不得继续用于 benchmark 或更新 `iaas_main`。
  - 下一步提交并推送 DeepGEMM fork wheel 修复，按 C9 重新触发 ByteIAAS workflow，等待新镜像产出后重新执行 C13-C16。

### P38: DeepGEMM fork wheel image rebuild succeeded

- Summary: 已提交并推送 DeepGEMM fork wheel 的 image build-time 安装修复，重新触发 ByteIAAS dev image workflow。新 workflow 成功产出 openai/openai-devel 镜像，且 image job log 确认安装了 `deep-gemm==2.5.0` 并完成 MegaMoE 符号检查。后续部署验证固定使用新的 `openai-devel` 镜像，不再使用 P35/P37 中失败的旧镜像。
- Code and workflow evidence:
  - Fix commit: `d6fe62d15643d5619e6c5ac95201a060938a839f`。
  - Commit message: `Install DeepGEMM fork wheel in ByteIAAS image`。
  - Pushed branch: `codex/vllm-dsv4-fork-base-byteiaas-build`。
  - Workflow run id: `28452612809`。
  - Workflow URL: `https://github.com/bytedance-iaas/vllm/actions/runs/28452612809`。
  - `gh run view 28452612809 --repo bytedance-iaas/vllm --json status,conclusion,headSha,url,createdAt,updatedAt,jobs` 返回 `status=completed`、`conclusion=success`、`headSha=d6fe62d15643d5619e6c5ac95201a060938a839f`。
  - Jobs: `build-image / build-and-publish-image` success；`build-wheel / build-wheel` success。
- Published images:
  - `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-cu130`，digest `sha256:fedbbdac93f15356b2a9afea25f8ad671c719c37f7343424b32c99cdd1fd9cfa`。
  - `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，digest `sha256:57cb7a44de57b09bb8a45d214210dc8c4e76cd601c0ea0a8c78fc81f05e2d32a`。
  - Job log evidence saved at `artifacts/2026-06-29-vllm-dsv4-flash-pd/workflow/job-84318680934-logs-api.txt`，其中包含 `DeepGEMM MegaMoE symbols verified`、`+ deep-gemm==2.5.0`、`Published openai-devel image: iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`。
  - `docker buildx imagetools inspect` 通过：openai-devel image index digest 为 `sha256:57cb7a44de57b09bb8a45d214210dc8c4e76cd601c0ea0a8c78fc81f05e2d32a`，包含 `linux/amd64` manifest `sha256:fb00b955042689ed001658ede9ff15e8678ac8b6cec7f250d986b0c5bcf6b182`。
- Static validation before push:
  - `git diff --check` 通过。
  - ByteIAAS workflow YAML 通过 `yaml.safe_load` 解析。
  - `uv pip install --python <temporary-py312-venv> --dry-run <DeepGEMM wheel URL>` 可解析 `deep-gemm` wheel。
  - `helm template` 使用 `decode.args.maxNumSeqs=512`、不同 P/D 节点和测试 image 做 render 通过；render 中无 `--max-model-len`、`66000`、runtime hotfix/install forbidden pattern。
- Decision:
  - 新部署候选固定为 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`。
  - 下一步重新执行 C13-C16；如果该新镜像仍失败，需要按新的日志证据重新定位，不得回退到运行时安装或 runtime fallback。

### P39: Fourth deployment and router smoke passed with new image

- Summary: 使用 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130` 重新执行 C13-C16，部署方式继续对齐 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69`：StormService、1P1D、prefill/decode 各 8 GPU、router 静态 P/D URL、同一镜像执行 Onion 模型准备，无 runtime hotfix/install。
- Deployment evidence:
  - C14 session: `codex-vllm-dsv4-flash-pd-20260630-225852-738107`。
  - GPU permit: `7db945db-d65b-4d55-8f10-7c1ea453dfdd`，请求 16 GPU。
  - 选定节点：prefill `192.168.1.148`，decode `192.168.1.186`，router `192.168.1.186`。
  - Namespace: `vllm-dsv4-flash-pd`；Helm release: `dsv4-flash-pd`。
  - Render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`。
  - Render scan 无 `--max-model-len`、`66000`、runtime hotfix/install forbidden pattern；prefill/decode nodeAffinity 非空且不同，`global.gpuCount=8`。
  - Prefill pod `dsv4-flash-pd-roleset-pb4l6-prefill-86b499d88b-0` 调度到 `192.168.1.148`，container `prefill` 请求 8 GPU。
  - Decode pod `dsv4-flash-pd-roleset-pb4l6-decode-57748944b9-0` 调度到 `192.168.1.186`，container `decode` 请求 8 GPU。
  - Router pod `dsv4-flash-pd-router-59894c4b86-tkq6b` 调度到 `192.168.1.186`，不请求 GPU；启动早期因 servingkit-aligned TCP liveness probe 在 backend warmup 期间重启 2 次，最终 Ready。
- Readiness and smoke evidence:
  - `kubectl wait` 后 prefill、decode、router 均 Ready。
  - Onion init 在 P/D 侧均完成并幂等跳过已存在模型目录：`Model directory /data01/DeepSeek-V4-Flash is already complete, skip Onion download.`。
  - 模型完整性检查确认存在 `config.json`、tokenizer 文件和 safetensors index/shards。
  - Router `/v1/models` 返回 HTTP 200，模型 `max_model_len=1048576`，说明没有通过 CLI 强设 `vllm.maxModelLen=66000`。
  - Router `/v1/completions` 返回 HTTP 200 且非空输出；completion id 显示实际路由到 `prefill_addr_192.168.1.148:8000` 与 `decode_addr_192.168.1.186:8001`。
  - C16 原 label selector 只覆盖 router，已补采 StormService-managed P/D pod 的 describe、logs、init status、argv/env、model files、package evidence。
  - Package evidence 显示 `vllm 0.10.1.dev9890+gd6fe62d15`、`deep-gemm 2.5.0`、`mooncake-transfer-engine-cuda13 0.3.11.post1`、`vllm-router 0.1.14`、`oniond /usr/bin/oniond`；DeepGEMM 三个 MegaMoE 符号检查为 `True`。
  - Bad-log scan for `Mooncake found no common KV transfer regions|KV group count mismatch|KV load failed|handshake compatibility failure|request timeout during KV pull` 为 0；runtime install/hotfix scan 为 0。
- Artifacts:
  - `pods-ready.txt`、`pods-ready-all-labels.txt`、`services.txt`、`router-models.json`、`router-completion-smoke.json`。
  - `*-argv-env.txt`、`*-package-evidence.txt`、`*-model-files.txt`、`*-logs-tail5000.txt`。
  - `bad-log-scan.txt`、`runtime-install-scan.txt`、`c16-scan-summary.txt`。
- Decision:
  - M8 acceptance 已满足，可以进入 M9 benchmark。

### P40: Benchmark completed; TTFT passed but decode throughput gate failed

- Summary: M9 已完成，evalscope raw artifacts、timestamps、pod-local metrics heads、serving logs 和 summary 已保存。64k/1 Avg TTFT 通过用户阈值，但 BS512/1536 decode Avg output throughput 未达到 14000 tokens/s，因此不得更新远端 `iaas_main`。
- Harness adjustments:
  - C17 初始 `evalscope` 安装后 `perf` 子命令缺 `uvicorn`，错误要求 `pip install 'evalscope[perf]'`；已在本地 `.venv-evalscope` 中安装 `evalscope[perf]`，`evalscope --version` 为 `1.8.1`。
  - 本机没有 `/data01/DeepSeek-V4-Flash` tokenizer；已从运行中的 prefill pod 只复制 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json` 到 artifact `tokenizer/`，未复制模型权重，未修改 serving pod。
  - evalscope base URL 会自动拼 `/chat/completions` 并得到 404；实际 benchmark 使用显式 `http://127.0.0.1:30000/v1/completions`。
  - C20 使用 `--dataset-offset 1`，避免复用 C19 cache seed 的同一 64k token 序列污染 TTFT。
  - C20 wrapper 原先在 zsh 中使用 bash-only `PIPESTATUS`，evalscope 已成功但脚本尾部失败；已补写 timestamps end 和 exit code，并在 C21 改用 `bash -lc`。
- Warmup and seed evidence:
  - evalscope connection smoke 16 input / 1 output 成功。
  - C19 warmup 1024 input / 1 output 成功，Avg TTFT `2658.39 ms`。
  - C19 64k prefix cache seed 成功，Avg TTFT `6776.92 ms`，Avg Input Tokens `65536.00`。
- Measured results:
  - C20 64k input / 1 output: exit `0`，Avg TTFT `6590.02 ms`，Avg Input Tokens `65536.00`，Avg Output Tokens `1.00`；通过 `< 10s` gate。
  - C21 BS512 / 1536 output: exit `0`，Total / Success / Failed `512 / 512 / 0`，Avg Output Tokens `1536.00`，Avg Latency `74.59 s`，Avg TTFT `38948.52 ms`，Avg TPOT `23.22 ms`，Avg ITL `103.78 ms`，Avg Output Throughput `8361.71 tok/s`；低于 `>= 14000 tok/s` gate。
  - C21 Workload Completion tok/s: Overall `8361.71`，Last 30s `28671.27`，Steady `28671.27`；gate 按用户要求看 Avg/Overall，不看 Last 30s/Steady 或 percentiles。
  - 该 throughput 是 1P1D router path 结果，output 来自单个 decode 节点，不是多 decode 聚合。
- Cleanup evidence:
  - 已保存 `summary.md`，明确 Prometheus skipped by user，不声称完整服务侧 monitoring 诊断。
  - 已保存 `final-pods.txt`、`final-events.txt`、`post-benchmark/*`、`bad-log-scan-after-benchmark.txt`。
  - `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd` 已执行；namespace 删除后复查为 `NotFound`。
  - Router port-forward pid `749095` 已 kill 且进程不存在。
  - Permit `7db945db-d65b-4d55-8f10-7c1ea453dfdd` 已释放，artifact `permit-release-after-benchmark.json` 状态为 `released`。
  - workspace-env registry 中本任务 HelmRelease 记录已更新为 `released`；按本任务 namespace/session 过滤 active 结果为 `[]`。
- Artifacts:
  - Summary: `artifacts/2026-06-29-vllm-dsv4-flash-pd/summary.md`。
  - TTFT: `evalscope-ttft-64k-1out.log`、`ttft-64k-1out.timestamps`。
  - Decode: `evalscope-decode-bs512-cache-hit-1p5kout.log`、`decode-bs512-cache-hit-1p5kout.timestamps`。
  - Cleanup: `cleanup-verification.txt`、`registry-helmrelease-released-after-benchmark.json`、`registry-active-after-benchmark-cleanup.json`。
- Decision:
  - M10 不执行。性能未达标是用户明确 stop rule：`性能过差不更新iaas_main`。
  - 下一步若继续推进，应先诊断 BS512/1536 Avg throughput 低于目标的原因，而不是更新远端 `iaas_main`。

### P41: Align deployment semantics back to servingkit for scheduler and max-num-seqs

- Summary: 用户确认 `prefill.args.noAsyncScheduling` 必须与 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 对齐为 `false`；`decode.args.maxNumSeqs` 是每个 worker 的值，不应为 BS512 benchmark 覆盖为 `512`，应保持 `96`。本条 supersede 之前 P33/P34/P38/P39 中关于 `decode.args.maxNumSeqs=512` 可作为允许差异的结论，但不改写历史已执行命令记录。
- Evidence:
  - `examples/deployment/deepseek-v4-flash-pd/values.yaml` 已将 `prefill.args.noAsyncScheduling` 从 `true` 改为 `false`。
  - 命令引用 `C13` 与 `C15` 的 `DECODE_MAX_NUM_SEQS` 默认值已从 `512` 改为 `96`。
  - 主计划 `M7` 已明确：`decode.args.maxNumSeqs` 必须保持 servingkit 当前语义 `96`，不得为了 BS512/1.5k benchmark 覆盖为 `512`。
  - 当前状态已标记：此前 benchmark 使用的 `decode.args.maxNumSeqs=512` run 不能作为当前部署语义的有效 gate；后续需以 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96` 重新 render/deploy/benchmark。

### P42: Resume execution decision and ownership

- Summary: 继续按 `superpowers:executing-plans` 执行主计划；本轮不启用 `subagent-driven-development`。
- Rationale: 当前剩余工作不是独立可并行实现流，而是同一 Helm release、同一 namespace、同一 GPU permit、同一 router port-forward 和同一 artifact 目录上的串行状态机。并行子代理会增加 Kubernetes ownership、permit、port-forward 和 artifact 覆盖冲突。
- Ownership boundaries: 主线程独占 `dev-cluster` namespace `vllm-dsv4-flash-pd`、Helm release `dsv4-flash-pd`、workspace-env permit/session、`artifacts/2026-06-29-vllm-dsv4-flash-pd/` 以及计划三份文档的更新。
- Validation strategy: 重新执行 `C13` render，确认 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`、无 runtime hotfix/install；随后执行 `C14` 获取 16 GPU permit，`C15/C16` 部署和真实 router smoke，最后执行 `C19-C21` benchmark 并由 `C22` 清理和更新 gate 结果。

### P43: C13 rerender passed with maxNumSeqs=96

- Summary: 使用当前新候选镜像和原 P/D 节点重新执行 C13 render，确认部署语义已回到 servingkit 对齐状态。
- Evidence:
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`。
  - Render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`。
  - Positive grep artifacts: `c13-render-positive-grep.txt`、`c13-render-shape-grep.txt`。
  - Render 中 prefill 参数包含 `--max-num-seqs "16"`，没有 `--no-async-scheduling`。
  - Render 中 decode 参数包含 `--max-num-seqs "96"`、`--all2all-backend "deepep_low_latency"`、`--moe-backend "deep_gemm_mega_moe"`、`--cp-kv-cache-interleave-size "256"` 和 MTP speculative config。
  - Render 中没有 `--max-model-len`、`"66000"`、`runtimePatch`、`git clone`、`pip install`、`apt install`、wheel download/install 或 runtime router install。
  - `helm template` 同节点负例被拒绝，`rendered-invalid-same-node.err` 包含 `prefill and decode nodeAffinity values must be disjoint`。

### P44: C14 GPU permit granted; local env-file write bug repaired

- Summary: C14 集群 preflight 与 GPU permit 申请已完成，permit 返回 `granted`；命令尾部用于写 `c14-current-env.sh` 的 Python snippet 错误引用了未定义变量，导致 shell exit `1`，但错误发生在 permit 已写入 artifact 后、任何 GPU workload 创建之前。已用 artifact 中的 JSON 修正本地状态文件。
- Evidence:
  - `envctl info dev-cluster` 和 `envctl validate dev-cluster` 通过，输出保存到 `c14-env-info.txt`、`c14-env-validate.txt`。
  - `stormservices.orchestration.aibrix.ai` CRD 存在。
  - 节点 `192.168.1.148` 与 `192.168.1.186` 均 `ALLOCATABLE_GPU=8` 且 `UNSCHEDULABLE=false`。
  - `gpu-permit.json` 返回 `status=granted`、`permit_id=585fd741-c3a7-490b-935f-cb761b5652fc`、`session_id=codex-vllm-dsv4-flash-pd-20260630-234927-790782`、`requested_gpus=16.0`。
  - `permit-list --active` 确认该 permit 仍 active 且状态为 `granted`。
  - 已写入 `artifacts/2026-06-29-vllm-dsv4-flash-pd/c14-current-env.sh`，供 C15/C16 使用。
- Prevention: 后续命令不再在 f-string 中引用未定义 JSON key 变量；需要写 permit 环境文件时直接从已保存的 JSON 字段取值。

### P45: C15 deployed after node reselect

- Summary: 初始 C15 使用旧节点 `192.168.1.148/192.168.1.186` 失败在调度阶段，原因是这些节点已有其它 hostNetwork P/D 服务占用 8000/8001 端口。已停止 Helm wait、卸载本任务刚创建的失败 release，改选当前空闲的 `192.168.1.149` 与 `192.168.1.154` 后重新 render/deploy；Helm release 已成功 deployed。
- Failed old-node evidence:
  - `c15-failed-old-node-events.txt` 中 prefill/decode Pending 原因：`1 node(s) didn't have free ports for the requested pod ports`。
  - `c15-node-occupancy-before-reselect.txt` 显示 `192.168.1.148` 上已有 `hank/vllm-45112-ppcp...prefill` 占用 `8000,8998`，`192.168.1.186` 上已有 `hank/vllm-45112-ppcp...decode` 占用 `8001,8999`。
  - 本任务旧 release 已通过 `c15-uninstall-old-node-release.log` 卸载，旧 router pod delete wait 成功。
- Reselect evidence:
  - `c15-node-occupancy-before-reselect.txt` 显示 `192.168.1.149` 与 `192.168.1.154` 当前 GPU request 为 `0`，hostNetwork 仅有系统/monitoring 端口，不占用 P/D/router 端口。
  - `rendered-dsv4-flash-pd.yaml` 已重新渲染到 prefill `192.168.1.149:8000`、decode `192.168.1.154:8001`、router `192.168.1.154:30000`，且 `decode.args.maxNumSeqs=96`。
  - `c15-reselected-nodes.txt` 记录 `192.168.1.149` 与 `192.168.1.154` 均 `ALLOCATABLE_GPU=8`、`UNSCHEDULABLE=false`。
- Deploy evidence:
  - `helm-upgrade-dsv4-flash-pd.log` 返回 `STATUS: deployed`。
  - `kubectl-get-all-after-deploy.txt` 显示 prefill pod `dsv4-flash-pd-roleset-596xb-prefill-846b58fc57-0` Ready on `192.168.1.149`，decode pod `dsv4-flash-pd-roleset-596xb-decode-6598c55574-0` Ready on `192.168.1.154`，router pod `dsv4-flash-pd-router-6d8fb49b5f-tbfln` Ready on `192.168.1.154`。
  - Permit `585fd741-c3a7-490b-935f-cb761b5652fc` 已通过 `c15-permit-running.json` 标记为 `running`。

### P46: C16 readiness and real router smoke passed

- Summary: C16 证据链通过。P/D/router 均 Ready，Onion init 幂等跳过已有模型目录，模型完整性检查通过，router `/v1/models` 与 `/v1/completions` 均通过真实 router path。
- Evidence:
  - `pods-ready.txt` 显示 prefill `1/1 Running` on `192.168.1.149`，decode `1/1 Running` on `192.168.1.154`，router `1/1 Running` on `192.168.1.154`。router 在后端未 ready 期间有 3 次 liveness restart，最终 Ready。
  - `*-onion-model-prepare.log` 显示 `Model directory /data01/DeepSeek-V4-Flash is already complete, skip Onion download.`。
  - `router-models.json` 返回 `deepseek-v4-flash`，`max_model_len=1048576`，确认没有通过 CLI 设置 `--max-model-len=66000`。
  - `router-completion-smoke.json` 返回 HTTP 成功且文本为 `","`；completion id 包含 `prefill_addr_192.168.1.149:8000` 与 `decode_addr_192.168.1.154:8001`。
  - `bad-log-scan.txt` 对 `Mooncake found no common KV transfer regions|KV group count mismatch|KV load failed|handshake compatibility failure|request timeout during KV pull` 无命中。
  - `runtime-install-scan.txt` 对 runtime hotfix/install pattern 无命中。
  - `*-package-evidence.txt`、`*-argv-env.txt`、`*-model-files.txt` 已保存到 artifact 目录。
- Issue: 第一次 C16 shell 由于 nested zsh/bash quoting 失败，未执行到集群；随后改为 `shell=bash` 直接运行同一逻辑成功。后续长脚本优先用 bash shell 直接执行，避免把脚本嵌进 zsh 单引号。

### P47: C19-C21 rerun with maxNumSeqs=96; C21 invalid due Mooncake KV transfer failures

- Summary: 继续在 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96` 的当前部署上执行 M9。C19 warmup/cache seed 和 C20 64k/1 TTFT 成功；C21 BS512/1536 cache-hit decode run 未完成且无有效 throughput gate 结果。日志出现大量 Mooncake KV transfer 失败和 producer 侧 480s timeout，因此 M10 仍不得执行。
- Access path issue:
  - 继续使用 `envctl port-forward` 时，port-forward 进程在处理一次健康检查连接后退出，导致 C19 多次在 `curl http://127.0.0.1:30000/health` 处 exit `7`。
  - 直接访问 router Node IP `http://192.168.1.154:30000/health` 从本机超时。
  - 改用前台 PTY session 执行原生 `kubectl port-forward -n vllm-dsv4-flash-pd svc/dsv4-flash-pd-router 30000:30000` 后，连续健康检查通过，C19-C21 可继续执行。
  - Prevention: 未来长 benchmark 不应在短生命周期 shell 中用后台 `envctl port-forward`；需要保持 port-forward 前台会话，或使用更稳定的 cluster-internal benchmark pod。
- C19 evidence:
  - Health check: `c19-router-health-before-warmup.txt` 返回 `All servers healthy`。
  - Warmup artifact: `evalscope-warmup.log`，1024 input / 1 output 成功，`Total / Success / Failed = 1 / 1 / 0`，Avg TTFT `10578.51 ms`。
  - Cache seed artifact: `evalscope-cache-seed.log`，64k prefix / 1 output 成功，`Total / Success / Failed = 1 / 1 / 0`，Avg TTFT `6804.19 ms`。
- C20 evidence:
  - Artifact: `evalscope-ttft-64k-1out.log`、`ttft-64k-1out.timestamps`。
  - Measured run: start `2026-07-01T00:19:47+0800`，end `2026-07-01T00:19:59+0800`，exit `0`。
  - Result: `Total / Success / Failed = 1 / 1 / 0`，Avg TTFT `6633.05 ms`，Avg Input Tokens `65536.00`，Avg Output Tokens `1.00`。
  - Gate: 通过 `<10s` Avg TTFT gate。
- C21 evidence:
  - Artifact: `evalscope-decode-bs512-cache-hit-1p5kout.log`、`decode-bs512-cache-hit-1p5kout.timestamps`。
  - Command shape: `--parallel 512 --number 512 --prefix-length 65536 --min-tokens 1536 --max-tokens 1536 --seed 42`，部署端仍为 `decode.args.maxNumSeqs=96`。
  - Request generation completed after about 4m46s, then entered `Processing[parallel_512_number_512]`。
  - Partial evalscope progress reached at least 300 successes, but then stalled; last raw progress in `evalscope-decode-bs512-cache-hit-1p5kout.log` stayed around `Processing ... 68%| 348/512` before no further progress.
  - During the stall, decode pod logs captured many `mooncake_connector.py:2208 ... failed: Mooncake transfer engine returned -1` lines.
  - Prefill pod logs captured many `mooncake_connector.py:2072 ... timed out after 480 seconds without being sent. Freeing its blocks on the producer side.` warnings.
  - Evidence files: `c21-invalid-mooncake-evidence.txt`、`c21-*-logs-during-hang.txt`、`c21-*-logs-final.txt`、`c21-evalscope-log-tail-during-hang.txt`。
  - Because this is a KV/Mooncake runtime failure and the run did not finish 512/512, it has no valid Avg/Overall output throughput for the `>=14000 tokens/s` gate.
  - The run was interrupted after failure evidence was collected; timestamps record end `2026-07-01T00:35:19+0800` and `MEASURED_RUN_EXIT_CODE=130` with invalid reason.
- Cleanup evidence:
  - `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd` completed; artifact `cleanup-helm-uninstall-invalid-c21.txt`。
  - Namespace `vllm-dsv4-flash-pd` deleted and subsequent lookup returned `NotFound`; artifacts `cleanup-namespace-delete-invalid-c21.txt`、`cleanup-namespace-after-invalid-c21.txt`。
  - PTY port-forward session exited after local `kubectl port-forward` process was killed.
  - Workspace-env permit `585fd741-c3a7-490b-935f-cb761b5652fc` released; artifact `permit-release-after-invalid-c21.json` shows `status=released`。
  - Workspace-env active registry for `dev-cluster/vllm-dsv4-flash-pd` is empty; artifact `registry-active-after-invalid-c21-cleanup.json`。
  - Updated benchmark summary: `artifacts/2026-06-29-vllm-dsv4-flash-pd/summary.md`。
- Decision:
  - M10 不执行。C20 TTFT gate 通过，但 C21 是不可接受的 KV/Mooncake runtime failure，不是外部资源 blocker，也没有满足 BS512/1.5k output throughput gate。
  - 下一步必须先诊断 Mooncake transfer engine `-1` 与 producer-side timeout 的根因；当前分支不得更新到远端 `iaas_main`。

### P48: Re-confirm servingkit-aligned scheduling values after user correction

- Summary: 用户再次确认 `noAsyncScheduling` 必须对齐 servingkit 为 `false`，且 `maxNumSeqs` 是每个 worker 的值，所以 decode 侧应保持 `96`，不能为了 BS512 benchmark 改成 `512`。本条只收敛模板和计划口径，不改变 P/D 拓扑、镜像或 benchmark 阈值。
- Evidence:
  - `examples/deployment/deepseek-v4-flash-pd/values.yaml` 当前为 `prefill.args.noAsyncScheduling: false`、`decode.args.maxNumSeqs: 96`。
  - 命令引用 `C13` 与 `C15` 当前默认 `DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-96}"`，并在 Helm render/deploy 时传入 `--set decode.args.maxNumSeqs="${DECODE_MAX_NUM_SEQS}"`。
  - 主计划 `M7` 与 Current Status 已明确：`decode.args.maxNumSeqs` 按每 worker admission 值理解，必须保持 `96`，不得因 BS512/1.5k benchmark 覆盖为 `512`。
  - 执行只读 render 校验生成 `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd-maxseqs96-check.yaml`，检查通过：prefill rendered command 不包含 `--no-async-scheduling`；decode `start-decode.sh` 包含 `"--max-num-seqs" "96"`；rendered manifest 不包含 `--max-model-len`、`runtimePatch`、`git clone`、`pip install`、`apt install`、runtime wheel/install 或 runtime router install pattern。
- Outcome:
  - 计划中历史 P40 的 `decode.args.maxNumSeqs=512` run 已明确标记为废弃历史结果，不再作为 M9 gate 依据。
  - 当前有效后续执行入口仍是 P47 之后的 Mooncake transfer failure 诊断或在健康节点上按 `noAsyncScheduling=false`、`maxNumSeqs=96` 重新跑 C13-C21。
- Prevention:
  - 后续执行 C13/C15 时可显式留空 `DECODE_MAX_NUM_SEQS` 使用默认 `96`，或显式设置 `DECODE_MAX_NUM_SEQS=96`；不得设置为 `512`。

### P49: Reverse-node C21 invalid and cleanup

- Date: 2026-07-01
- Summary: 在 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96` 的当前部署语义下，尝试将 P/D 节点对调为 prefill `192.168.1.154`、decode/router `192.168.1.149`。C16 smoke 和 C20 64k/1 TTFT 通过，但旧 C21 BS512/512 请求形态仍因 Mooncake KV transfer 错误无法形成有效 throughput gate。
- Evidence:
  - Reverse render artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd-reverse-154-149.yaml`。
  - Reverse permit: `aaa9b695-2390-4e2f-ad21-2b52d0a27fd0`，session `codex-vllm-dsv4-flash-pd-20260701-reverse-149-154`。
  - Reverse deployment: prefill pod on `192.168.1.154`，decode/router on `192.168.1.149`，均 Ready；Onion init 幂等跳过；router `/v1/models` 和 `/v1/completions` 成功。
  - C20 artifact: `evalscope-ttft-64k-1out-reverse.log`；Avg TTFT `6636.64 ms`，通过 `<10s` gate；该结果是计算结果，不是 cache 命中结果，因为 measured run 使用 `prefix_length=0`，且 `Cached Prompt tok/s=0.00`。
  - 旧 C21 artifact: `evalscope-decode-bs512-cache-hit-1p5kout-reverse.log`、`decode-bs512-cache-hit-1p5kout-reverse.timestamps`。请求生成完成 `512/512` 后进入 processing，阶段性统计到 `400` success，最后约停在 `487/512`，无有效最终 Avg/Overall throughput。
  - Prefill bad-log evidence saved to `c21-reverse-prefill-tail-after-invalid.log`，包含 `Sending to 192.168.1.149:15927 failed (ret=-1)`。
  - Decode bad-log evidence saved to `c21-reverse-decode-tail-after-invalid.log`，包含 `Mooncake transfer engine returned -1`。
  - Evalscope 被中断并写入 `MEASURED_RUN_EXIT_CODE=120`。
- Cleanup:
  - Helm release `dsv4-flash-pd` 已卸载；namespace `vllm-dsv4-flash-pd` 已删除。
  - Permit `aaa9b695-2390-4e2f-ad21-2b52d0a27fd0` 已释放；registry 资源已标记 `released`。
  - 本地 evalscope 和 router port-forward 已退出；残留 `kubectl logs -f` 已停止。
- Outcome:
  - 反向节点只能说明旧 C21 BS512/512 请求口径下仍会触发 Mooncake KV transfer 失败；不能作为 BS512/1.5k throughput gate。
  - M10 继续不执行。

### P50: Decode throughput benchmark request count and fallback BS sweep

- Date: 2026-07-01
- Summary: 用户修改 benchmark 策略：压测时发送请求数应为 BS 的 4 倍，以达到更好的 decode 端吞吐；如果 BS512 跑不过，需要在 128-512 之间寻找能够完整通过压测的 BS。
- Plan changes:
  - 命令引用 `C21` 已从 `--parallel 512 --number 512` 改为 `DECODE_BS=512`、`DECODE_REQUESTS=$((DECODE_BS * 4))`，即 `--parallel 512 --number 2048`。
  - `C21` artifact 命名改为包含总请求数：`evalscope-decode-bs512-cache-hit-1p5kout-n2048.log` 和 `decode-bs512-cache-hit-1p5kout-n2048.timestamps`。
  - 新增命令引用 `C21A`：当 BS512/2048 请求 run invalid 或没有有效 throughput 结果时，按默认候选 `384 256 192 128` 降档尝试；每个候选都使用 `number = 4 * BS`，并写入 `decode-bs-sweep-cache-hit-1p5kout.summary.tsv`。
  - 主计划 `M9` 已恢复为未完成状态：需要重新执行 C21；若 BS512 invalid，还必须执行 C21A。
  - 主计划 `M10` 已明确：只有 BS512 本身的 `number=2048` run 达到 `>=14000 tokens/s` 且无不可接受 runtime 错误，才能更新远端 `iaas_main`；降档 BS 通过只能作为容量诊断，不替代 BS512 gate。
- Next:
  - 重新部署当前 image 和模板后，先执行 C21 BS512/2048 请求；若再次出现 KV/Mooncake 错误或未完成，再执行 C21A 降档 sweep，并同步保存 P/D/router bad-log scan。

### P51: Monitoring, service restart isolation, and vLLM bench comparison

- Date: 2026-07-01
- Summary: 用户补充要求：失败后应重启服务，避免把坏状态带到下一轮；需要补充对比 vLLM 自带压测脚本是否和 evalscope 有差异；需要使用 servingkit `hanhan_dev/llm-serving-monitoring` 部署监控系统，并检查是否达到预期 running BS，达不到时不测试高于实际 BS 的情况。
- Source inspection:
  - servingkit URL 对应本地 repo `origin/hanhan_dev` 下路径 `llm-serving-monitoring/`，不是独立分支。
  - 该 chart 包含 `Chart.yaml`、`values.yaml`、`templates/prometheus-*`、`templates/grafana-*`、`dashboards/vllm-v4-full-metrics.json` 和示例 `examples/dev-cluster-vllm-dsv4-fp8-pd-values.yaml`。
  - README 说明该 chart 部署独立 Prometheus/Grafana，默认不依赖集群级 Prometheus/VMP；vLLM dashboard 中使用 `vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:generation_tokens_total` 等指标。
  - 当前 vLLM fork 中存在 `vllm/benchmarks/serve.py`，CLI 为 `vllm bench serve` / `python -m vllm.benchmarks.serve`，支持 `--request-rate inf`、`--max-concurrency`、`--num-prompts`、`--random-prefix-len`、`--random-input-len`、`--random-output-len`、`--ignore-eos`、`--extra-body`、`--save-result` 等参数，可与 evalscope C21/C21A 做口径对比。
- Plan changes:
  - 主计划全局约束新增：benchmark 必须部署 servingkit `origin/hanhan_dev:llm-serving-monitoring` 的最小 Prometheus 监控；对比 vLLM 自带压测脚本，但 gate 仍默认使用 evalscope，除非对比证明口径等价。
  - 命令引用 `C18` 从“Skip Prometheus”改为部署 servingkit monitoring chart：归档 `llm-serving-monitoring/` 到 artifact 目录，使用本任务专属 namespace/release `vllm-dsv4-flash-pd-monitoring` / `dsv4-flash-pd-monitoring`，不得接管或清理已有共享 `vllm-monitoring`；生成本任务 values，scrape `${RELEASE}-prefill.${NAMESPACE}.svc.cluster.local:8000`、`${RELEASE}-decode.${NAMESPACE}.svc.cluster.local:8001`、`${RELEASE}-router.${NAMESPACE}.svc.cluster.local:30000`，并校验三类 target `up == 1`。
  - 新增 `C21M`：每次 C21/C21A 候选结束、失败或中断后，用 Prometheus query_range 记录 `max_decode_running`、`max_decode_waiting` 和 `max_decode_output_tps_30s`。
  - 新增 `C21R`：任何 C21/C21A 候选失败、被中断、出现 KV/Mooncake runtime 错误或队列/运行态异常后，继续下一轮前必须删除并重建 P/D/router pods，重新等待 Ready，并确认 router `/health` 和 `/v1/models` 成功。
  - 新增 `C21V`：保存 vLLM 自带 benchmark help、对比说明，并在不高于 observed running BS 的候选上运行一次 `python -m vllm.benchmarks.serve` 对照；该结果只用于判断工具差异，不替代 C21 gate。
  - `C22` summary/cleanup 增加 monitoring release/namespace 清理和 running BS evidence。
- Supersedes:
  - P7 的“此处先跳过 Prometheus”已被本条新要求覆盖。后续不得再把 Prometheus skipped 作为当前计划状态。
  - P50 的降档 sweep 继续有效，但每次失败后必须先执行 C21R；若 C21M 显示 observed running BS 低于候选目标，下一轮不得测试高于 observed capacity 的候选。
- Gate impact:
  - 远端 `iaas_main` 更新仍必须由 BS512/1.5k gate 决定：`number=2048`、Avg/Overall output throughput `>=14000 tokens/s`、Prometheus 证明实际 running BS 达标、无 KV/Mooncake/DeepEP/DeepGEMM runtime 错误。
  - 降档 BS 通过只能作为容量诊断；vLLM bench 对比通过也不能单独替代 evalscope gate，除非计划后续被明确修改。

### P52: Redeploy current image and fix monitoring namespace creation

- Date: 2026-07-01
- Summary: 继续执行当前 plan，使用候选镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130` 重新部署当前模板，节点选择 prefill `192.168.1.149`、decode/router `192.168.1.154`。C13 render、C14 permit、C15 deploy、C16 real router smoke 已完成；C18 首次执行暴露出 monitoring chart namespace 创建命令 bug，已修正命令引用后准备重跑。
- Evidence:
  - C13 artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/rendered-dsv4-flash-pd.yaml`；检查通过：`prefill.args.noAsyncScheduling=false`，decode `--max-num-seqs "96"`，不包含 `--max-model-len`、`66000`、runtime hotfix/install，且同节点负例被 Helm validation 拒绝。
  - C14 permit: session `codex-vllm-dsv4-flash-pd-20260701-014137-931102`，permit `1ee454b6-f76a-480c-8345-92b7fc6463f0`，状态 `granted`，节点 `192.168.1.149` 与 `192.168.1.154` 均 allocatable `nvidia.com/gpu=8`。
  - C15 deploy: release `dsv4-flash-pd` deployed；prefill pod `dsv4-flash-pd-roleset-cpgck-prefill-64697d968-0` 在 `192.168.1.149`，decode pod `dsv4-flash-pd-roleset-cpgck-decode-775d4df7f-0` 在 `192.168.1.154`，router pod `dsv4-flash-pd-router-68d87564cb-pvr98` 在 `192.168.1.154`；三者均 Ready。decode 首次启动花约 8.5 分钟完成 TileLang/DeepGEMM 初始化，router 早期因等待 worker 健康重启 2 次，最终 Ready。
  - C16 smoke: router `/v1/models` 返回 `max_model_len=1048576`；router `/v1/completions` 返回非空文本 `","`，completion id 显示 route 到 prefill `192.168.1.149:8000` 和 decode `192.168.1.154:8001`；bad-log scan 与 runtime install/hotfix scan 未命中。
  - C17/C17A: evalscope `1.8.1` 可用；tokenizer 小文件已从当前 prefill pod 刷新到 `artifacts/2026-06-29-vllm-dsv4-flash-pd/tokenizer/`。
  - C18 首次失败: `Error: namespaces "vllm-dsv4-flash-pd-monitoring" already exists`。原因是命令引用同时使用 Helm `--create-namespace` 和 chart values `namespace.create: true`，Helm 先创建 namespace 后 chart 再尝试创建同名 Namespace。
- Plan fix:
  - 命令引用 `C18` 已改为 `namespace.create: false`，由 Helm `--create-namespace` 负责创建本任务专属 namespace；scrape targets、Prometheus 配置、nodeAffinity 和 cleanup 语义不变。
- Next:
  - 重跑 C18；通过后继续 C19 warmup/cache seed、C20 64k/1 TTFT、C21 BS512/2048 decode throughput，并在每个候选后执行 C21M。

### P53: Fix monitoring Prometheus service discovery

- Date: 2026-07-01
- Summary: C18 第二次执行成功部署过 Prometheus Pod，但随后 port-forward 失败，因为命令引用假设 Service 名称为 `${MONITORING_RELEASE}-prometheus`；servingkit chart 实际渲染名称为 `${MONITORING_RELEASE}-llm-serving-monitoring-prometheus`。执行失败后未保留 release/namespace，当前 `vllm-dsv4-flash-pd-monitoring` 不存在。
- Evidence:
  - C18 output: Prometheus Pod `dsv4-flash-pd-monitoring-llm-serving-monitoring-prometheus6qxzp` 曾 Ready，Service 名称显示为 `dsv4-flash-pd-monitoring-llm-serving-monitoring-prometheus`。
  - 失败原因: `Error from server (NotFound): services "dsv4-flash-pd-monitoring-prometheus" not found`。
  - 后续检查: `helm status dsv4-flash-pd-monitoring -n vllm-dsv4-flash-pd-monitoring` 返回 `release: not found`；`kubectl get ns vllm-dsv4-flash-pd-monitoring` 返回 `NotFound`。
- Plan fix:
  - 命令引用 `C18` 和 `C21M` 已改为通过 label `app.kubernetes.io/instance=${MONITORING_RELEASE},app.kubernetes.io/component=prometheus` 动态发现 Prometheus Service，并把实际名称写入 artifact。
- Next:
  - 重新执行 C18；若 scrape target `up == 1` 通过，再继续 benchmark。

### P54: Treat router metrics as unavailable, keep worker metrics gate

- Date: 2026-07-01
- Summary: C18 第三次执行后，Prometheus 已能 scrape prefill/decode vLLM worker metrics，但 router target `up=0`。诊断确认 router `/metrics` 对 Prometheus GET 返回 `405 Method Not Allowed`；这不是 P/D worker running BS 指标缺失。计划已改为 C18 只要求 prefill/decode `up == 1`，并将 router 可用性保留在 C16 的 `/health`、`/v1/models` 和真实 completion smoke。
- Evidence:
  - Prometheus query `up{stack="vllm",release="dsv4-flash-pd"}` 返回 prefill `1`、decode `1`、router `0`。
  - Prometheus target error for router: `server returned HTTP status 405 Method Not Allowed` on `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000/metrics`。
  - 从 Prometheus Pod 内直接访问 prefill/decode `/metrics` 返回 `vllm:*` 指标；访问 router `/metrics` 返回 `HTTP/1.1 405 Method Not Allowed`。
  - 从 router Pod 内访问 `/health` 返回 `All servers healthy`；C16 已通过 router `/v1/models` 与真实 completion smoke。
- Plan fix:
  - 命令引用 `C18` 将 router scrape target 设为 `enabled: false`，required roles 改为 `prefill` 和 `decode`。
  - 主计划全局约束和 M9 已明确：running BS gate 使用 decode worker metrics；router metrics 不可用时记录证据但不阻塞 benchmark。
- Next:
  - 重新执行 C18，确认 prefill/decode worker target `up == 1` 后继续 C19/C20/C21。

### P55: C20/C21 latest measured results with monitoring

- Date: 2026-07-01
- Summary: 在当前有效部署语义 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、`1P1D`、P/D 不同 8-GPU 节点下，C20 TTFT 通过，但 C21 BS512/2048 请求 decode throughput run 无效。
- C20 evidence:
  - Artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/evalscope-ttft-64k-1out.log`、`ttft-64k-1out.timestamps`。
  - Result: `Total / Success / Failed = 1 / 1 / 0`，Avg TTFT `6775.64 ms`，通过 `<10s` gate。
  - Cache status: `prefix_length=0` 且 `Cached Prompt tok/s=0.00`，因此这是计算结果，不是 cache 命中结果。
- C21 evidence:
  - Artifact: `evalscope-decode-bs512-cache-hit-1p5kout-n2048.log`、`decode-bs512-cache-hit-1p5kout-n2048.timestamps`。
  - Command shape: `--parallel 512 --number 2048 --prefix-length 65536 --min/max-prompt-length 0 --min/max-tokens 1536 --seed 42`，满足 `number = 4 * BS`。
  - Timestamps: start `2026-07-01T02:02:49+0800`，end `2026-07-01T02:25:44+0800`，exit `120`。
  - Runtime failure evidence: decode logs 出现 `Mooncake transfer engine returned -1`；prefill logs 出现 Mooncake transfer timeout，例如 `Sync batch data transfer timeout`。
  - C21M monitoring: `monitoring/running-bs-bs512.summary.txt` 记录 `max_decode_running=231.0`、`max_decode_waiting=512.0`、`max_decode_output_tps_30s=8578.58620689655`。
- Outcome: C21 BS512 无有效 Avg/Overall throughput，且属于不可接受的 KV/Mooncake runtime failure；不得更新远端 `iaas_main`。

### P56: C21R restart isolated failed BS512 state

- Date: 2026-07-01
- Summary: 按用户要求，BS512 失败后先重启 P/D/router，再执行降档候选，避免把 Mooncake/KV、router 或队列坏状态带入下一轮。
- Evidence:
  - Restart artifact dir: `artifacts/2026-06-29-vllm-dsv4-flash-pd/service-restarts/20260701-022656/`。
  - 第一次 wait 命令存在竞态：`kubectl wait` 命中了旧 pod ready 状态，已中断并改用 corrected wait。
  - Corrected wait 后 prefill pod `dsv4-flash-pd-roleset-cpgck-prefill-64697d968-0` Ready on `192.168.1.149`，decode pod `dsv4-flash-pd-roleset-cpgck-decode-775d4df7f-0` Ready on `192.168.1.154`，router pod `dsv4-flash-pd-router-68d87564cb-gfbgm` Ready on `192.168.1.154`。
  - Post-restart router `/health`、`/v1/models` 和真实 completion smoke 通过；completion artifact 为 `router-completion-after-c21r.json`。
- Prevention: 后续 C21R 应等待被删除 pod 退出或等待新 pod UID/创建时间变化后再判断 Ready，避免旧 pod ready 竞态。

### P57: C21A fallback BS192 completed but below gate

- Date: 2026-07-01
- Summary: 因 C21M 显示 BS512 observed max running 仅 `231`，没有继续测试 384 或 256，直接选择不高于 observed capacity 的 BS192。该降档候选完整通过，但吞吐显著低于 14000 tokens/s，且不能替代 BS512 gate。
- Evidence:
  - Artifact: `evalscope-decode-bs192-cache-hit-1p5kout-n768.log`、`decode-bs192-cache-hit-1p5kout-n768.timestamps`。
  - Command shape: `--parallel 192 --number 768 --prefix-length 65536 --min/max-prompt-length 0 --min/max-tokens 1536 --seed 42`。
  - Timestamps: start `2026-07-01T02:36:59+0800`，end `2026-07-01T02:47:32+0800`，exit `0`。
  - Result: `Total / Success / Failed = 768 / 768 / 0`，`Avg Output Tokens=1536.00`，`Output Throughput=6354.03 tok/s`，workload `Completion tok/s` overall `6354.78`，Last30s `8283.68`，steady drop 20% `9042.32`。
  - C21M monitoring: `monitoring/running-bs-bs192.summary.txt` 记录 `max_decode_running=192.0`、`max_decode_waiting=164.0`、`max_decode_output_tps_30s=10668.137931034482`。
  - Bad-log scan: `log-scans/bad-log-scan-bs192.txt` 记录 `bad_log_scan=clean`。
- Outcome: BS192 是当前部署下能完成的降档诊断点，但不足以发布；M10 继续不执行。

### P58: C21V local vLLM benchmark comparison unavailable

- Date: 2026-07-01
- Summary: 按 C21V 尝试运行 vLLM 自带 benchmark help，当前本地客户端环境缺少 `torch`，因此无法执行 vLLM bench 对照 run。
- Evidence:
  - Artifact dir: `artifacts/2026-06-29-vllm-dsv4-flash-pd/vllm-bench-compare/`。
  - `vllm-bench-serve-help.exitcode` 为 `1`。
  - `vllm-bench-serve-help.err` 包含 `ModuleNotFoundError: No module named 'torch'`。
  - 已写入 `vllm-bench-serve-unavailable.md`，明确该缺口不改变 evalscope gate。
- Outcome: C21V 作为口径对比有验证缺口；没有证据支持切换 benchmark harness。最终 gate 仍使用 evalscope C20/C21/C21A。

### P59: Current gate decision before cleanup

- Date: 2026-07-01
- Summary: 当前 image、部署模板和 benchmark 证据已足以做 gate 判断：不更新远端 `iaas_main`。
- Decision:
  - `64k/1` Avg TTFT `6775.64 ms`，通过 `<10s`。
  - BS512/1.5k `number=2048` 无效，属于 Mooncake/KV runtime failure，且 Prometheus observed max running 只有 `231`。
  - BS192/1.5k 完整通过但 evalscope Output Throughput `6354.03 tok/s`，低于 `14000 tok/s`，且只是降档容量诊断。
  - vLLM built-in benchmark 本地对比不可用，不能替代 evalscope gate。
- Artifact summary: `artifacts/2026-06-29-vllm-dsv4-flash-pd/summary.md` 已更新为最新结果。
- Next: 执行 C22 清理 serving/monitoring Helm release、namespace、本地 port-forward 和 GPU permit `1ee454b6-f76a-480c-8345-92b7fc6463f0`。

### P60: C22 cleanup completed

- Date: 2026-07-01
- Summary: 已清理本次 live benchmark 创建的 serving、monitoring、port-forward 和 GPU permit 资源。
- Cleanup evidence:
  - `cleanup/helm-uninstall-serving.txt`: `release "dsv4-flash-pd" uninstalled`。
  - `cleanup/delete-serving-namespace.txt`: `namespace "vllm-dsv4-flash-pd" deleted`。
  - `cleanup/helm-uninstall-monitoring.txt`: `release "dsv4-flash-pd-monitoring" uninstalled`。
  - `cleanup/delete-monitoring-namespace.txt`: `namespace "vllm-dsv4-flash-pd-monitoring" deleted`。
  - `cleanup/permit-release.json`: permit `1ee454b6-f76a-480c-8345-92b7fc6463f0` 状态为 `released`。
  - `cleanup/namespaces-after-cleanup.txt`: 两个 namespace 均返回 `NotFound`。
  - `cleanup/registry-active-after-cleanup-final.json` 和 `cleanup/permit-list-after-cleanup-final.json`: 均为 `[]`。
  - `cleanup/local-processes-after-cleanup-final.txt`: 无本任务 `kubectl port-forward`、`evalscope` 或 `vllm.benchmarks.serve` 残留。
- Final decision:
  - 远端 `iaas_main` 未更新。
  - 当前 goal 停止于不可接受的 BS512 KV/Mooncake runtime failure 和性能 gate 未达标；后续若要推进，需要先修复或重新定位 C21 BS512 failure，再重新构建/部署/benchmark。

### P61: Plan update for containerized vLLM benchmark comparison

- Date: 2026-07-01
- Summary: 用户要求修改计划：vLLM 自带压测必须在 vLLM 容器中执行。此前 P58 的本地 `python3 -m vllm.benchmarks.serve --help` 缺 `torch` 只能保留为历史证据，不再满足 C21V 的计划要求。
- Plan changes:
  - 主计划 Global Constraints 已明确：`vllm bench serve` / `vllm.benchmarks.serve` 对比必须在同一个新构建 vLLM 镜像启动的容器内执行，不能使用本地工作站 Python 环境；该容器不请求 GPU，不做运行时安装、代码 clone 或 hotfix，通过集群内 router Service 发请求，结束后删除。
  - M9 的 C21V checkbox 已重新置为未完成，要求后续保存容器内 help、对比说明、一个不高于 observed running BS 的对照运行或容器内无法运行原因。
  - 命令引用 `C21V` 已改为创建临时 benchmark Pod `dsv4-flash-pd-vllm-bench`，使用同一 vLLM 镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`，挂载 `/data01` 只读作为 tokenizer/model path，通过 `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000` 访问 router。
  - C21V 不再使用本地 port-forward，也不在本地执行 `python3 -m vllm.benchmarks.serve`；help 和 benchmark 都通过 `kubectl exec` 在 benchmark Pod 内运行。
  - 如果容器内 help 或 benchmark 失败，应归类为候选镜像/容器 runtime 对 vLLM bench 支持不足，而不是本地环境缺依赖。
- Outcome:
  - 当前 live 资源已在 P60 清理，未立即执行新的 C21V。
  - 后续恢复执行时需要重新部署服务和 monitoring 后再运行新的容器内 C21V。

### P62: C21V containerized vLLM benchmark comparison completed

- Date: 2026-07-01
- Summary: 按用户要求，C21V 已在同一个新构建 vLLM 镜像启动的临时 benchmark Pod 内执行，而不是本地 Python 环境。镜像内 `python3 -m vllm.benchmarks.serve` 是空入口，没有输出和结果目录；实际可用入口为 `vllm bench serve`。
- Benchmark pod:
  - Namespace: `vllm-dsv4-flash-pd`
  - Pod: `dsv4-flash-pd-vllm-bench`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - GPU request: none
  - Target: `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000/v1/completions`
  - Tokenizer/model path: `/data01/DeepSeek-V4-Flash`
- Command shape:
  - `vllm bench serve --backend openai --base-url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000 --endpoint /v1/completions --model deepseek-v4-flash --tokenizer /data01/DeepSeek-V4-Flash --dataset-name random --random-prefix-len 65536 --random-input-len 0 --random-output-len 1536 --request-rate inf --max-concurrency 128 --num-prompts 512 --ignore-eos --temperature 0 --seed 42 --save-result --save-detailed`
- Result:
  - Timestamps: start `2026-07-01T07:45:50+0800`, end `2026-07-01T07:50:36+0800`, exit `0`。
  - Successful / Failed: `512 / 0`。
  - Output token throughput: `5018.14 tok/s`。
  - Mean TTFT: `18268.25 ms`。
  - Mean TPOT: `11.17 ms`。
  - Mean ITL: `46.78 ms`。
  - vLLM-reported maximum request concurrency: `128`；vLLM-reported peak concurrent requests: `180`，该字段与 `max_concurrency` 的口径不同，不能直接作为服务端 running BS gate。
- Monitoring:
  - Prometheus window artifacts: `vllm-bench-compare-c21v-container/monitoring/window-summary.json`。
  - Decode running: max `128.0`，avg `31.97`。
  - Decode waiting: max `64.0`，avg `4.31`。
  - Decode 30s output TPS: max `8772.66`，avg `2531.23`。
  - Bad-log scan: `vllm-bench-compare-c21v-container/bad-log-scan-c21v.txt`，没有命中 Mooncake/KV 坏模式。
- Outcome:
  - C21V 证明 vLLM 镜像内置 CLI 可用，但该口径只是对照，不能替代 evalscope gate。
  - C21V 的 BS128/1.5k output throughput 仍显著低于 `14000 tok/s`，且 Prometheus 显示 decode running 平均远低于 128；继续测试更高 BS 没有发布意义。
  - 远端 `iaas_main` 仍不得更新。

### P63: C22 cleanup after C21V completed

- Date: 2026-07-01
- Summary: 已清理本次为 C21V 容器内 benchmark 重新创建的 serving、monitoring、临时 benchmark Pod、namespace 和 workspace-env permit。
- Cleanup evidence:
  - Cleanup artifact dir: `artifacts/2026-06-29-vllm-dsv4-flash-pd/cleanup-c21v/`。
  - `helm-uninstall-serving.txt`: `release "dsv4-flash-pd" uninstalled`。
  - `delete-serving-namespace.txt`: `namespace "vllm-dsv4-flash-pd" deleted`。
  - `helm-uninstall-monitoring.txt`: `release "dsv4-flash-pd-monitoring" uninstalled`。
  - `delete-monitoring-namespace.txt`: `namespace "vllm-dsv4-flash-pd-monitoring" deleted`。
  - `namespaces-after-cleanup-final.txt`: `vllm-dsv4-flash-pd` 与 `vllm-dsv4-flash-pd-monitoring` 均返回 `NotFound`。
  - `permit-release-c21v.json` 和 `permit-status-after-cleanup-c21v.json`: permit `626cbb46-0c8f-4166-8ffc-c7737180a180` 状态为 `released`。
  - `registry-active-after-cleanup-c21v.json`: `[]`。
  - `permit-list-unreleased-after-cleanup-c21v.json`: `[]`。
  - `local-processes-after-cleanup-c21v-final.txt`: 无本任务 `kubectl port-forward`、`kubectl wait/exec`、`evalscope`、`vllm bench` 或 `vllm.benchmarks` 残留。
- Final decision:
  - 远端 `iaas_main` 未更新。
  - 当前 goal 停止于不可接受的 BS512 KV/Mooncake runtime failure 和性能 gate 未达标；后续需要先修复或重新定位 BS512 failure/throughput 问题，再重新构建、部署和 benchmark。

### P64: Supplemental vLLM bench serve BS256/400/512 sweep completed

- Date: 2026-07-01
- Summary: 按用户要求补充 vLLM 自带压测方式下 BS256、BS400、BS512 三组结果。该 sweep 在同一个新构建 vLLM 镜像启动的 no-GPU benchmark Pod 内执行，通过 in-cluster router Service 发请求，固定 64K prefix、1536 output、`number = 4 * BS`，部署语义仍为 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、`1P1D`。
- Benchmark pod:
  - Namespace: `vllm-dsv4-flash-pd`
  - Pod: `dsv4-flash-pd-vllm-bench-sweep`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - GPU request: none
  - Target: `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000/v1/completions`
  - Tokenizer/model path: `/data01/DeepSeek-V4-Flash`
- Command shape:
  - `vllm bench serve --backend openai --base-url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000 --endpoint /v1/completions --model deepseek-v4-flash --tokenizer /data01/DeepSeek-V4-Flash --dataset-name random --random-prefix-len 65536 --random-input-len 0 --random-output-len 1536 --request-rate inf --max-concurrency <BS> --num-prompts <4*BS> --ignore-eos --temperature 0 --seed 42 --save-result --save-detailed --result-dir /tmp/vllm-bench-bs-sweep --result-filename vllm-bench-serve-bs<BS>-n<4*BS>.json --disable-tqdm`
- Results:
  - BS256: start `2026-07-01T08:28:58+08:00`，end `2026-07-01T08:36:22+08:00`，exit `0`，`Successful / Failed = 1024 / 0`，output throughput `8052.52 tok/s`，Mean TTFT `23914.05 ms`，Mean TPOT `14.90 ms`，Mean ITL `69.75 ms`。
  - BS400: start `2026-07-01T08:36:24+08:00`，end `2026-07-01T08:46:20+08:00`，exit `0`，`Successful / Failed = 1600 / 0`，output throughput `11091.25 tok/s`，Mean TTFT `10276.96 ms`，Mean TPOT `26.36 ms`，Mean ITL `87.89 ms`。
  - BS512: start `2026-07-01T08:46:22+08:00`，end `2026-07-01T08:58:13+08:00`，exit `0`，`Successful / Failed = 2048 / 0`，output throughput `15281.44 tok/s`，Mean TTFT `6876.14 ms`，Mean TPOT `26.39 ms`，Mean ITL `106.91 ms`。
- Monitoring:
  - BS256: Prometheus max decode running `256.0`，max decode 30s output TPS `13468.83`。
  - BS400: Prometheus max decode running `400.0`，max decode 30s output TPS `14246.62`。
  - BS512: Prometheus max decode running `510.0`，max decode 30s output TPS `18210.17`。
  - 三组 bad-log scan 均未命中 Mooncake/KV 坏模式。
- Artifacts:
  - Summary: `artifacts/2026-06-29-vllm-dsv4-flash-pd/vllm-bench-bs-sweep-20260701/summary.md`。
  - TSV: `artifacts/2026-06-29-vllm-dsv4-flash-pd/vllm-bench-bs-sweep-20260701/runs/summary.tsv`。
  - Per-BS logs/results/Prometheus windows: `runs/bs256/`、`runs/bs400/`、`runs/bs512/`。
- Outcome:
  - vLLM bench BS512 口径下吞吐超过 `14000 tok/s`，且 Prometheus 显示 decode running 达到 `510`。
  - 该结果仍是 vLLM harness 对照，不能自动替代此前 invalid 的 evalscope BS512 gate；远端 `iaas_main` 仍不更新，除非用户明确接受 vLLM bench 作为新的 gate 或后续 evalscope gate 重新跑通。

### P65: Cleanup after supplemental vLLM BS sweep completed

- Date: 2026-07-01
- Summary: 已清理补充 vLLM BS sweep 创建或复用的 serving、monitoring、benchmark Pod、namespace 和 workspace-env permit。
- Cleanup evidence:
  - Cleanup artifact dir: `artifacts/2026-06-29-vllm-dsv4-flash-pd/vllm-bench-bs-sweep-20260701/cleanup/`。
  - Namespace lookup after cleanup: `vllm-dsv4-flash-pd` 与 `vllm-dsv4-flash-pd-monitoring` 均返回 `NotFound`。
  - Permit release: `b2a762f1-8271-4a9b-b544-7cd872237760` 状态为 `released`。
  - `registry-active-after-cleanup-final.json`: `[]`。
  - `permit-list-active-after-cleanup-final.json`: `[]`。
  - `local-processes-after-cleanup.txt`: 无本任务相关本地残留进程。
- Note:
  - 第一次清理脚本中的本地 `pkill` pattern 误匹配自身导致 exit `143`，但当时 serving/monitoring namespace 已删除；随后使用不匹配自身的方式补完 permit release、registry 验证和本地进程检查。

### P66: Plan update for evalscope vs vLLM bench divergence analysis

- Date: 2026-07-01
- Trigger: 用户要求修改计划，分析为什么 evalscope 的结果和 vLLM 自带 benchmark 的结果差异这么大。
- Plan changes:
  - 主计划 Global Constraints 新增差异分析 gate：evalscope 与 vLLM bench 结果显著不一致时，必须先执行 `C21W`，从请求构造、prefix cache 命中语义、client 网络路径、并发/限流模型、超时与失败处理、统计口径、Prometheus running/waiting/output TPS、Mooncake/KV 错误窗口和服务状态延续角度归因。
  - M9 新增未完成项：`C21W` 先基于现有 artifacts 做离线分析，不创建 GPU workload、不更新 gate；如果证据不足，再执行 `C21X` 最小配对复现实验。
  - M9 Acceptance 新增：`C21W` 必须产出 `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/summary.md`，并把结论分类为 `workload-mismatch`、`client-path`、`harness-timeout/statistics`、`service-state/kv-transfer`、`mixed` 或 `inconclusive`。
  - 命令引用新增 `C21W`：读取 evalscope BS512/BS192、vLLM bench BS512、Prometheus window、bad-log scan 和 timestamps，生成 `summary.md`、`evalscope-excerpts.txt`、`vllm-bench-excerpts.txt`、`monitoring-comparison.txt`、`bad-log-comparison.txt`。
  - 命令引用新增 `C21X`：仅当 `C21W` 无法解释差异时，重新部署相同 `1P1D` 服务并按 BS128/256/512 做配对复现实验；每个失败候选后必须 C21R 重启服务。
- Current decision:
  - vLLM bench BS512 成功结果仍只作为 supplemental/harness comparison。
  - 在 `C21W` 完成且差异原因可审计前，不得用 vLLM bench BS512 替代 evalscope BS512 gate，不得进入 M10 更新远端 `iaas_main`。
- Next:
  - 执行 `C21W` 离线分析；只有 `C21W` 结论为 `inconclusive` 时才重新申请 GPU permit 执行 `C21X`。

### P67: C21W execution strategy

- Date: 2026-07-01
- Summary: 开始按 `superpowers:executing-plans` 执行 `C21W`。本轮不启用 `subagent-driven-development`。
- Rationale:
  - `C21W` 是单一离线 artifact 分析，输入集中在 `artifacts/2026-06-29-vllm-dsv4-flash-pd/`，写入集中在 `harness-diff-analysis-20260701/summary.md` 和计划进展日志。
  - 当前不创建 GPU workload、不访问 live cluster、不修改 vLLM 源码、不触碰远端分支或 `iaas_main`。
  - 分派并行 subagent 会增加同一 summary 结论和计划文件的写入冲突，收益低于协调成本。
- Ownership:
  - 主线程负责读取 evidence、生成差异分析、更新主计划 compact status、更新 progress log，并决定是否需要进入 `C21X`。
- Validation:
  - `C21W` 产物必须包含 `summary.md`、`evalscope-excerpts.txt`、`vllm-bench-excerpts.txt`、`monitoring-comparison.txt`、`bad-log-comparison.txt`。
  - 结论必须覆盖 workload、client path、concurrency/running BS、failure window、stats denominator 和 Mooncake/KV bad-log 证据。

### P68: C21W divergence analysis completed

- Date: 2026-07-01
- Summary: C21W 离线差异分析已完成；本轮没有创建 GPU workload、没有访问 live cluster、没有重新部署、没有执行 C21X。
- Artifacts:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/summary.md`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/request-db-summary.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/evalscope-excerpts.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/vllm-bench-excerpts.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/monitoring-comparison.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-diff-analysis-20260701/bad-log-comparison.txt`
- Evidence:
  - evalscope BS512 timestamps: start `2026-07-01T02:02:49+0800`，end `2026-07-01T02:25:44+0800`，exit `120`。
  - evalscope BS512 request DB: `result_count=0`，因此没有完整请求结果，也没有可比较的 Avg/Overall throughput。
  - evalscope BS512 Prometheus: `max_decode_running=231.0`，`max_decode_waiting=512.0`，`max_decode_output_tps_30s=8578.58620689655`。
  - evalscope BS512 bad logs: prefill 侧有 `Sync batch data transfer timeout`、`Sending to ... failed (ret=-1)`，producer-side 有 `timed out after 480 seconds without being sent`。
  - evalscope BS192 fallback request DB: `result_count=768`，`success_counts=[(1, 768)]`，sample row 为 `prompt_tokens=65536`、`completion_tokens=1536`，证明完成请求的 token 长度对齐；但 output throughput 只有 `6354.03 tok/s`。
  - vLLM bench BS512: `completed=2048`，`failed=0`，`total_input_tokens=134217728`，`total_output_tokens=3145728`，`output_throughput=15281.439904082581`，Prometheus decode running max `510.0`，waiting max `16.0`，bad-log scan clean。
- Conclusion:
  - 分类为 `mixed`，主因是 `service-state/kv-transfer` 与 `harness-timeout/statistics` 叠加；`client-path` 是可信放大因素。
  - evalscope BS512 不是低吞吐完整结果，而是 invalid run；vLLM bench BS512 是后续健康服务状态下的完整结果，两者不能直接互相替代。
  - 现有 artifacts 已足以解释差异，因此当前不进入 `C21X`；若用户决定切换 gate 或要求 same-state harness 等价性证明，再修改计划并执行 C21X。
- Gate:
  - C21W 不改变 M10 决策。evalscope BS512 仍未通过，且 KV/Mooncake runtime failure 是不可接受 blocker；vLLM bench BS512 仍只作为 supplemental/harness comparison，不替代 evalscope gate。

### P69: Plan update for evalscope Mooncake/KV failure diagnosis

- Date: 2026-07-01
- Trigger: 用户要求修改计划，分析 evalscope 结果为什么和 vLLM 结果差异这么大；C21W 已解释差异主要来自 evalscope BS512 invalid run 与 vLLM bench healthy run 的不可比，但还需要定位 evalscope BS512 触发的 Mooncake/KV transfer failure。
- Plan changes:
  - 主计划 Global Constraints 新增 C23 gate：在 C21W 解释差异但根因落在 Mooncake/KV transfer failure 时，必须先做离线根因初筛，不修改源码、不改变部署语义、不创建 GPU workload。
  - M9 新增 C23 checkbox：分析 evalscope BS512 的失败是否属于部署语义发散、metadata/握手不一致、Mooncake/RDMA transfer timeout/descriptor pressure、client burst/服务状态污染，或证据不足。
  - M9 Acceptance 新增 C23 artifact：`artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/summary.md`，并保存失败日志统计、MooncakeConnector 代码路径、servingkit SHA 对齐/差异、节点与 rendered command 对比。
  - 命令引用新增 `C23: Offline Mooncake/KV failure root-cause triage`，只使用已有 artifacts、当前部署模板、servingkit reference SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 和本仓库代码。
- Execution strategy:
  - 本轮继续不启用 `subagent-driven-development`；C23 是单一离线诊断链，写入集中在一个 artifact 目录和三份计划文件，并行分派会增加状态冲突。
  - 当前不申请 workspace-env permit，不访问 live cluster，不修改 vLLM runtime 源码，不触碰远端 `iaas_main`。
- Next:
  - 执行 C23 离线命令并记录 summary；若 C23 仍无法解释 transfer failure，再规划 live diagnostic/repro。

### P70: C23 offline Mooncake/KV failure diagnosis completed

- Date: 2026-07-01
- Summary: C23 离线根因初筛已完成；本轮没有申请 GPU permit、没有访问 live cluster、没有重新部署、没有修改 vLLM runtime 源码，也没有改变 evalscope gate。
- Artifacts:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/summary.md`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/failure-counts.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/code-path-excerpts.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/servingkit-values-diff.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/mooncake-failure-diagnosis-20260701/node-and-render-comparison.txt`
- Evidence:
  - 日志统计覆盖 9 个 C21 during/final log 文件。
  - `sync_timeout=62`、`ret_failed=62`、`xfer_returned=93`、`producer_timeout=535`。
  - `KV group count mismatch=0`、`handshake compatibility failure=0`、`Mooncake found no common KV transfer regions=0`。
  - `ret_failed` 远端 session 都指向 decode 节点 `192.168.1.154` 的多个 Mooncake 端口：`16737`、`15047`、`16936`、`15365`、`16445`、`15763`、`15628`、`16078`。
  - 失败 transfer 统计：`duration_s` min `30.1319`、p50 `31.4379`、max `32.0157`；descriptors min `7642`、p50 `82033`、max `115257`；bytes min `130947840`、p50 `1427616000`、max `1998662400`。
  - MooncakeConnector 代码路径显示 `_send_blocks` 调用 `batch_transfer_sync_write`，ret 非 0 时记录 `Sending to ... failed (ret=...)`；consumer 侧随后记录 `Mooncake transfer engine returned -1`；producer 侧未发送请求按 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 默认 480s 超时释放。
  - servingkit diff 的主要差异是计划内的 runtime hotfix/install 删除、`mooncakePackageVersion` pin 删除、节点参数化；当前没有看到足以解释 failure 的严重 P/D/router 语义发散。
- Conclusion:
  - 当前不是 evalscope “低吞吐完成结果”，而是 Mooncake transfer failure invalid run。
  - 现有证据更支持 evalscope BS512 请求突发/客户端路径触发 Mooncake/RDMA transfer timeout 或 descriptor pressure；不像 metadata/握手不一致。
  - vLLM bench BS512 后续健康 run 成功，不能证明 evalscope gate 通过，也不能消除该 transfer failure blocker。
- Next:
  - 如果继续推进 evalscope gate，应新增或执行 live diagnostic/repro：同一新部署、同一节点、同一 prefix/tokenizer、控制 harness 顺序，采集 Mooncake debug/RDMA/descriptor 证据；每次失败后必须 C21R 重启服务。
  - 当前仍不得更新远端 `iaas_main`。

### P71: Plan update for live repro, flakiness check, and debug-code authorization

- Date: 2026-07-01
- Trigger: 用户要求修改计划，继续部署并分析根因，同时考虑这可能是偶现问题、可能和压测方式无关；用户允许修改调试代码并推送。
- Plan changes:
  - 主计划新增 M11：Live 复现与 Mooncake/KV 根因诊断。
  - 命令引用新增 `C24: Live Mooncake/KV repro, flakiness check, and harness cross-check`。
  - 命令引用新增 `C25: Add gated Mooncake diagnostic logging, push branch, build debug image, and retest`。
  - Global Constraints 明确：先用当前成功构建镜像做无代码改动 live 复现和偶现性判断；只有 C24 证据不足时，才加最小调试代码、推送诊断分支、触发 dev image 构建。
  - Approval Forecast 记录追加授权：允许为根因诊断修改最小调试代码、推送诊断分支、触发 dev image 构建，并用调试镜像复测；该授权不包含更新远端 `iaas_main`。
- Diagnostic policy:
  - C24 必须至少跑 evalscope BS512/2048 两次 independent attempts；attempt 之间必须执行 C21R 重启 P/D/router。
  - 如果第一次成功，仍需第二次判断是否只是偶现恢复；如果第一次失败，重启后第二次判断是否稳定复现。
  - 同一 live 服务状态下还要跑 vLLM bench BS512 cross-check，避免仅凭历史 healthy run 推断压测方式差异。
  - C24 summary 必须分类为 `stable-repro`、`intermittent`、`harness-specific`、`node/environment-specific`、`instrumentation-needed` 或 `resolved-by-successful-rerun`。
  - C25 调试代码只允许加可开关日志/计数，例如 `VLLM_DSV4_MOONCAKE_DIAG=1`；不得改变 Mooncake transfer、调度、fallback、TP/PP/DP/EP、maxNumSeqs 或 benchmark 语义。
- Current decision:
  - 下一步执行 C24 前置环境、render 和 workspace-env permit 检查。
  - 当前仍不得更新远端 `iaas_main`。

### P72: C24 live deployment and first evalscope BS512 attempt

- Date: 2026-07-01
- Summary: C24 已重新申请 workspace-env permit 并部署当前成功构建镜像 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`。第一次 deployment candidate 使用 prefill node `192.168.1.149` 时，该节点在 benchmark 前变为 `NodeNotReady`/unreachable，pod 被 evict 且 stuck terminating；该事件归类为 node/environment-specific，不纳入 evalscope/vLLM harness 对比。随后清理并改用 prefill `192.168.1.148`、decode/router `192.168.1.154`。
- Permit:
  - Session: `codex-vllm-dsv4-live-diag-20260701-095856`
  - Thread: `codex-thread-vllm-dsv4-live-diag`
  - Permit: `6751b814-3d8d-40f8-80f0-0fd8fe6bb4e2`
  - Requested GPUs: `16`
- Effective deployment semantics:
  - `1P1D`，prefill 8 GPU，decode 8 GPU，P/D different nodes。
  - `prefill.args.noAsyncScheduling=false`。
  - `decode.args.maxNumSeqs=96`。
  - Rendered command does not contain `--max-model-len`。
  - No runtime hotfix/install path was observed.
- Smoke and seed:
  - Router `/v1/models` and `/v1/completions` succeeded.
  - Evalscope seed for attempt 2 later confirmed seed prompt hash equals formal prompt hash.
- Evalscope attempt 1:
  - Directory: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/evalscope-bs512-attempt-1/`
  - Command shape: `evalscope perf --parallel 512 --number 2048 --dataset random --prefix-length 65536 --min-prompt-length 0 --max-prompt-length 0 --min-tokens 1536 --max-tokens 1536 --seed 42 --extra-args '{"temperature":0,"ignore_eos":true}'`
  - Processing stalled around `1762/2048` after several minutes without progress.
  - Client interrupted and recorded exit `130`.
  - Partial DB summary: `1000` success rows, avg latency `70.439972s`, avg TTFT `47.572388s`, avg output tokens `1536`, output tokens `1,536,000`.
  - Stuck Prometheus snapshot: decode running `0`, decode waiting `286`, generation TPS `0`, prefill running/waiting `0`.
  - Corrected log counts: producer-side `timed out after 480 seconds without being sent` count `2065`; `out-of-order step` count `732`; `KV group count mismatch`/`handshake compatibility failure`/`Mooncake found no common KV transfer regions` all `0`.
- Decision:
  - Attempt 1 is a real producer-side timeout/stall signal, but not yet stable repro because service was restarted and attempt 2 completed.

### P73: C24 restart and second evalscope BS512 attempt

- Date: 2026-07-01
- Summary: Attempt 1 后按要求重启 P/D/router，修正了首次 restart 脚本中错误使用 `role-group` label 的问题，改为使用 `role-name=prefill` 和 `role-name=decode` 删除 P/D pods，并删除旧 router pod 释放 hostNetwork port `30000`。
- Restart evidence:
  - Directory: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/restart-after-attempt-1b/`
  - Current pods after restart:
    - Prefill: `dsv4-flash-pd-roleset-2dd5z-prefill-744499bdd7-0` on `192.168.1.148`
    - Decode: `dsv4-flash-pd-roleset-2dd5z-decode-8cd55cb47-0` on `192.168.1.154`
    - Router: `dsv4-flash-pd-router-6748845866-kjcxj` on `192.168.1.154`
- Attempt 2 seed:
  - Directory: `evalscope-bs512-attempt-2-seed/`
  - Seed request succeeded with TTFT `14101.70 ms`.
  - Seed prompt hash: `b88adb6042683505ee443673a4c578f643293ad60a126f0ada5cc27b2b1260d3`.
  - Wrapper exited nonzero only because zsh did not support the bash `PIPESTATUS` collection expression after evalscope completion; actual evalscope request succeeded.
- Attempt 2 formal run:
  - Directory: `evalscope-bs512-attempt-2/`
  - Exit: `0`
  - Result: `2048/2048` success, `0` failed.
  - Formal prompt hash: `b88adb6042683505ee443673a4c578f643293ad60a126f0ada5cc27b2b1260d3`; seed and formal run prompt match.
  - First 10 formal prompts are identical; this rules out evalscope generating different prefixes per request for this run.
  - Avg output throughput: `12291.66 tok/s`.
  - Steady/drop20% completion throughput: `16963.67 tok/s`.
  - Last 30s completion throughput: `18608.77 tok/s`.
  - Avg TTFT: `20907.75 ms`; p50 TTFT `18604.52 ms`; p99 TTFT `52169.56 ms`.
  - DB summary: `2048` rows, `2048` success, avg latency `57.61893s`, avg TTFT `20.90775s`, avg output tokens `1536`, total output tokens `3,145,728`.
  - Corrected log counts in attempt-local artifacts: producer timeout `0`; `ret == TransferStatus::COMPLETED` `0`; `KV group count mismatch` `0`; `handshake compatibility failure` `0`; `Mooncake found no common KV transfer regions` `0`; `out-of-order step` `0`.
  - Exact Prometheus window `2026-07-01T11:39:54+08:00` to `2026-07-01T11:44:11+08:00`: decode running min/avg/max `0/294.54/512`; decode waiting `0/18.44/187`; prefill running `0/3.33/16`; prefill waiting `0/63.33/479`; decode generation TPS 30s `0/11889.85/18978.79`.
- Decision:
  - Attempt 2 shows the producer-side timeout is not deterministic under the current deployment.
  - However evalscope Avg gate still fails: output throughput `<14000 tok/s` and Avg TTFT `>10s`.
  - Because user specified `看Avg`, steady/drop20% and last30s values are evidence only, not gate pass.

### P74: C24 vLLM bench BS512 cross-check in the same vLLM image

- Date: 2026-07-01
- Summary: 在同一 live deployment 上，用同一个 vLLM image 启动 no-GPU benchmark Pod 执行 `vllm bench serve` BS512/2048 cross-check；该 Pod 挂载 `/data01` hostPath 读取 tokenizer/model，通过 in-cluster router Service 发请求，没有本地 Python 环境、运行时安装、代码 clone 或 hotfix。
- Benchmark pod:
  - Namespace: `vllm-dsv4-flash-pd`
  - Pod: `vllm-bench-bs512-crosscheck`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - Node: `192.168.1.148`
  - GPU request: none
  - Tokenizer/model path check: `/data01/DeepSeek-V4-Flash` exists with `tokenizer_config.json` and `tokenizer.json`.
- Command:
  - `vllm bench serve --backend openai --base-url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000 --endpoint /v1/completions --model deepseek-v4-flash --tokenizer /data01/DeepSeek-V4-Flash --dataset-name random --random-prefix-len 65536 --random-input-len 0 --random-output-len 1536 --request-rate inf --max-concurrency 512 --num-prompts 2048 --ignore-eos --temperature 0 --seed 42 --save-result --save-detailed --result-dir /tmp/vllm-bench-bs512-crosscheck --result-filename vllm-bench-serve-bs512-n2048.json`
- Result:
  - Exit: `0`
  - Successful / Failed: `2048 / 0`
  - Output token throughput: `15152.89 tok/s`
  - Mean TTFT: `7964.35 ms`
  - Mean TPOT: `26.01 ms`
  - Mean ITL: `105.79 ms`
  - P99 TTFT: `33665.59 ms`
  - Total input tokens: `134,217,728`
  - Total generated tokens: `3,145,728`
  - Max concurrency argument: `512`; reported peak concurrent requests: `539`
- Log scan:
  - Producer timeout `0`
  - `KV group count mismatch` `0`
  - `handshake compatibility failure` `0`
  - `Mooncake found no common KV transfer regions` `0`
  - `out-of-order step` `650`; because this run completed successfully, this pattern is not by itself fatal.
- Exact Prometheus window `2026-07-01T11:55:26+08:00` to `2026-07-01T11:58:54+08:00`:
  - decode running min/avg/max `0/391.93/512`
  - decode waiting min/avg/max `0/4.31/16`
  - prefill running min/avg/max `0/3.79/15`
  - prefill waiting min/avg/max `0/28.48/288`
  - decode generation TPS 30s min/avg/max `0/14365.81/18456.14`
- Interpretation:
  - vLLM bench meets both user Avg gates on the same image/deployment: Mean TTFT `<10s` and output throughput `>14000 tok/s`.
  - Compared with evalscope attempt 2, vLLM bench kept higher average decode running and much lower decode waiting, which supports request arrival/admission shape as a major factor in the harness gap.
  - This remains a cross-check, not a substitute for evalscope gate unless the plan gate is explicitly changed.

### P75: C24 conclusion and cleanup

- Date: 2026-07-01
- C24 summary artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/summary.md`
- Classification: `intermittent + harness-specific`
- Root-cause assessment:
  - The original evalscope BS512 producer timeout/stall is not a stable deterministic failure under the current deployment because attempt 2 completed after a full restart.
  - It is not safe to call the issue resolved because evalscope attempt 2 still failed the required Avg gate and attempt 1 produced real producer-side timeout evidence.
  - The live evalscope vs vLLM bench difference is reproducible and correlates with different admission/waiting behavior: evalscope exact window had decode running avg `294.54` and decode waiting max `187`; vLLM bench exact window had decode running avg `391.93` and decode waiting max `16`.
  - Evalscope prefix-cache intent was aligned within evalscope: seed and formal prompt hashes matched, and formal requests were identical. Evalscope still reported `Cached Prompt tok/s 0.00`, so cache accounting/metric semantics remain unresolved.
  - C25 debug code is not required to decide M10, because M10 is blocked by evalscope Avg gate failure and intermittent producer timeout. C25 should be used only if the next objective is to deeply root-cause attempt-1 Mooncake/RDMA descriptor timeout behavior.
- Cleanup:
  - Cleanup artifact dir: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/cleanup/`
  - Serving namespace `vllm-dsv4-flash-pd`: `NotFound` after cleanup.
  - Monitoring namespace `vllm-dsv4-flash-pd-monitoring`: `NotFound` after cleanup.
  - Benchmark pod `vllm-bench-bs512-crosscheck`: deleted with namespace.
  - Local task processes: no remaining `vllm-bench`, `evalscope`, or task-specific `kubectl wait/exec/port-forward`.
  - Permit `6751b814-3d8d-40f8-80f0-0fd8fe6bb4e2`: status `released`.
- Release decision:
  - Remote `iaas_main` remains unchanged.
  - Current plan still blocks M10 unless the user explicitly changes the performance gate from evalscope Avg to vLLM bench Avg and requests the corresponding plan update.

### P76: C26 evalscope vs vLLM benchmark deep dive completed

- Date: 2026-07-01
- Trigger: 用户要求修改计划，分析 vLLM benchmark 和 evalscope benchmark 的差异，深挖 evalscope benchmark 无法达标的原因，并重点看 speculative decoding 接受率是否有区别。
- Summary: C26 离线深挖已完成；本轮没有创建 GPU workload、没有访问 live cluster、没有修改源码、没有执行 C25 调试代码、没有更新远端 `iaas_main`。
- Artifacts:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/summary.md`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/metrics-comparison.txt`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/chunk-granularity.txt`
- Evidence:
  - evalscope attempt 2 与 vLLM bench cross-check 都完成 `2048/2048`，总 output tokens 均为 `3,145,728`。
  - evalscope attempt 2: duration `255.9237s`，output throughput `12291.6638 tok/s`，Avg TTFT `20907.75 ms`，Avg TPOT `23.92 ms`，Avg ITL `101.84 ms`，Decoded Tok/Iter `4.3615`，Spec. Accept Rate `0.7707`。
  - vLLM bench cross-check: duration `207.5992s`，output throughput `15152.8905 tok/s`，Mean TTFT `7964.35 ms`，Mean TPOT `26.01 ms`，Mean ITL `105.79 ms`；JSON 中没有 `spec`、`accept` 或 `draft` 字段。
  - 首批 512 请求差异更明显：evalscope Avg TTFT `44907.21 ms`，vLLM bench Mean TTFT `21885.30 ms`。
  - 末批 512 请求仍有差距：evalscope Avg TTFT `7535.16 ms`，vLLM bench Mean TTFT `2016.18 ms`。
  - Prometheus exact window: evalscope decode running avg/max `294.54/512`、decode waiting avg/max `18.44/187`、prefill waiting avg/max `63.33/479`、decode gen TPS avg/max `11889.85/18978.79`；vLLM bench decode running avg/max `391.93/512`、decode waiting avg/max `4.31/16`、prefill waiting avg/max `28.48/288`、decode gen TPS avg/max `14365.81/18456.14`。
  - evalscope seed prompt 与 formal prompt SHA 相同：`b88adb6042683505ee443673a4c578f643293ad60a126f0ada5cc27b2b1260d3`；first 10 formal prompts identical。
  - evalscope prompt UTF-8 bytes `434688`；2048 请求约 `890 MB` prompt payload，首批 512 请求约 `223 MB`，不含 JSON/HTTP overhead；evalscope 通过本地 `127.0.0.1:30000` port-forward，vLLM bench 在集群内 no-GPU Pod 直连 router Service。
  - chunk/ITL 代理：evalscope 约 `4.2610` output tokens / ITL event，vLLM bench 约 `4.0699` output tokens / ITL event；该代理不能替代 server-side spec metrics，但不支持 evalscope 接受率更差。
- Conclusion:
  - 分类为 `client-path + TTFT/admission dominated; speculative-acceptance-not-primary`。
  - 当前证据不支持“evalscope 因投机解码接受率更差导致无法达标”。evalscope 的 per-output decode cadence 不差，直接报告的 `Spec. Accept Rate=0.7707` 也不是异常低值；主要差距来自 TTFT、admission/waiting、client path 和请求 payload 上传路径。
  - 不能离线给出 vLLM bench 的真实 speculative accept rate，因为 vLLM bench JSON 没有 spec 字段，C24 Prometheus 也未抓 `vllm:spec_decode_*`。因此只能说“没有证据表明 evalscope 接受率更差”，不能宣称两者接受率完全相同。
  - 若需要 definitive 结论，应执行新增 `C26B`：重新部署相同镜像/语义，分别跑 local port-forward evalscope、in-cluster evalscope no-GPU Pod 和 in-cluster vLLM bench，并在每个窗口抓 `vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total`、`vllm:spec_decode_num_accepted_tokens_per_pos_total`、`vllm:spec_decode_num_drafts_total`。
- Gate:
  - C26 不改变 M10 决策。evalscope Avg gate 仍未通过；vLLM bench 通过仍只作为 harness cross-check；远端 `iaas_main` 仍不得更新。

### P77: C26B follow-up for port-forward/client path and attempt-1 RDMA details

- Date: 2026-07-01
- Trigger: 用户要求定位 evalscope 不达标是否只是本地 port-forward/client path 导致，并定位 attempt 1 的 producer timeout、Mooncake descriptor 和 RDMA 细节。
- Summary:
  - 重新确认 `dev-cluster` 可用，并通过 idempotent workspace-env permit 继续使用 `c52190ad-8c5c-4fe7-a392-7e9a325edc25`。
  - 当前部署语义保持 servingkit 对齐：prefill `192.168.1.148`、decode/router `192.168.1.154`、`prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`、无 runtime hotfix/install。
  - smoke 成功后执行 local port-forward evalscope BS512/2048、尝试 in-cluster evalscope、执行 in-cluster `vllm bench serve` no-GPU Pod，并补充 attempt 1 RDMA/descriptor 统计。
  - 本轮结束后已清理 Helm release、serving/monitoring namespaces、benchmark Pod、local port-forward，并释放 permit `c52190ad-8c5c-4fe7-a392-7e9a325edc25`。
- Artifacts:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/smoke-after-restart/`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/runs/local-portforward-evalscope-bs512-full/`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/runs/incluster-evalscope-bs512/`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/runs/incluster-evalscope-bs512-venv/`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/runs/incluster-vllm-bench-bs512-spec/summary.md`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/evalscope-bs512-attempt-1/evidence-summary/attempt1-rdma-descriptor-stats.txt`
- Evidence:
  - local port-forward evalscope BS512/2048：exit `0`，`2048/2048` success，request generation 耗时 `18:44`，Avg output throughput `11875.18 tok/s`，Avg TTFT `28649.67 ms`，Avg TPOT `21.54 ms`，Avg ITL `96.69 ms`，Decoded Tok/Iter `4.58`，Spec. Accept Rate `78.2%`。
  - local evalscope processing window：post-generation decode running avg/max `262.02/512`，decode waiting avg/max `13.74/165`，prefill waiting avg/max `82.68/474`，decode generation TPS avg/max `11158.13/20730.45`，weighted spec accept rate `0.8263`。first-completion-to-end window decode generation TPS avg/max `13829.31/20730.45`，weighted spec accept rate `0.8266`。
  - in-cluster evalscope blocked：同镜像 no-GPU Pod 内 `pip install evalscope==1.8.1` 卡在依赖下载；复用本地 `.venv-evalscope` 的 hostPath 失败，报 `/data00/.../.venv-evalscope/bin/evalscope: No such file or directory`，因为 Kubernetes 节点 `/data00` 不是本地工作站文件系统。
  - in-cluster `vllm bench serve` no-GPU Pod：image 与 serving 相同，直连 `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000`，`2048/2048` completed，0 failed，output throughput `10688.35 tok/s`，Mean TTFT `27152.21 ms`，Mean TPOT `25.34 ms`，Mean ITL `100.11 ms`，`MEASURED_RUN_START=2026-07-01T06:00:14+0000`，`MEASURED_RUN_END=2026-07-01T06:13:54+0000`，vLLM bench duration `294.31s`。
  - in-cluster vLLM bench processing window：decode running avg/max `305.34/512`，decode waiting avg/max `20.06/308`，prefill waiting avg/max `89.76/496`，decode generation TPS avg/max `10568.30/17652.03`，weighted spec accept rate `0.7234`。
  - in-cluster vLLM bench bad-log scan：0 行，无 `Sync batch data transfer timeout`、`Mooncake transfer engine returned -1`、producer 480s timeout、KV mismatch、handshake failure 或 no-common-region。
  - C24 attempt 1 单独 RDMA/descriptor 统计：`Sync batch data transfer timeout=204`、producer 480s timeout `2063`、Mooncake `ret=-1` `212`、KV mismatch/handshake/no-common-region 均为 `0`。
  - attempt 1 `Sending to ... failed (ret=-1)` records `204`；duration_s p50/p90/max `30.58/35.09/36.75`；descriptors p50/p90/max `33062/139169/173006`；bytes p50/p90/max `571046400/2426947200/2997993600`。
- Conclusion:
  - evalscope 不达标不是“只是本地 port-forward/client path”单因导致。port-forward 和本地 client path 会放大问题，但 in-cluster vLLM bench 绕开 port-forward 后仍未达 Avg gate。
  - 当前更准确的分类是：大 prompt client 构造/上传、router admission/waiting、decode running 不持续贴满 512、以及 Mooncake/RDMA 偶发大批量 transfer timeout 共同影响。
  - speculative decoding 不是 evalscope 独有劣化点：local evalscope 的 Prometheus weighted spec accept rate 约 `0.826`，in-cluster vLLM bench 约 `0.723`，没有证据支持 evalscope 因接受率更差而不达标。
  - attempt 1 的失败路径更像 Mooncake/RDMA descriptor pressure 或 transfer timeout：大批量 transfer 在约 30-37s 超时，consumer 侧 `ret=-1`，producer 侧随后 480s unsent timeout；metadata/握手/region mismatch 证据为 0。
- Cleanup:
  - `helm uninstall dsv4-flash-pd -n vllm-dsv4-flash-pd` 完成，namespace `vllm-dsv4-flash-pd` deleted。
  - `helm uninstall dsv4-flash-pd-monitoring -n vllm-dsv4-flash-pd-monitoring` 完成，namespace `vllm-dsv4-flash-pd-monitoring` deleted。
  - 本地无残留 `port-forward`、`kubectl wait`、`evalscope` 或 `vllm bench` 进程。
  - Permit `c52190ad-8c5c-4fe7-a392-7e9a325edc25` status `released`。
- Gate:
  - 远端 `iaas_main` 仍不得更新。evalscope BS512 Avg gate 未通过；本轮 in-cluster vLLM bench 也未达到 `>=14000 tok/s`。
  - C25 暂不执行，除非下一步目标明确变为继续定位 attempt 1 的 Mooncake/RDMA timeout 并构建调试镜像。

### P78: Proxy install path for in-cluster evalscope verified

- Date: 2026-07-01
- Trigger: 用户询问此前安装 evalscope 是否使用代理，并要求使用代理安装 evalscope。
- Summary:
  - 使用 `envctl info dev-cluster` 中的代理 `100.68.170.29:3128`，在 `dev-cluster` 创建临时 no-GPU Pod 验证同一 vLLM 镜像内安装 `evalscope[perf]==1.8.1`。
  - 该步骤不创建 GPU workload，不占用 workspace-env GPU permit。
  - 验证完成后已删除临时 namespace `vllm-dsv4-evalscope-install`，并确认查询为空。
- Artifact:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/evalscope-proxy-install-20260701/pod.yaml`
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/evalscope-proxy-install-20260701/pod.log`
- Environment:
  - Namespace: `vllm-dsv4-evalscope-install`
  - Pod: `evalscope-proxy-install`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - Node: `192.168.1.154`
  - Proxy env: `HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy=http://100.68.170.29:3128`
- Command:
  - `python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'`
  - `evalscope --version`
  - Python import check for `evalscope.__version__`
- Evidence:
  - `evalscope 1.8.1`
  - `evalscope_import_ok 1.8.1`
  - `jieba-0.42.1.tar.gz` 等依赖通过代理正常下载；此前 C26B 无代理 Pod 卡在依赖下载。
  - Pod runtime: `POD_START=2026-07-01T07:23:13+0000`，`POD_END=2026-07-01T07:24:40+0000`。
- Plan update:
  - C26B 的 in-cluster evalscope Pod scaffold 已更新为显式设置代理并安装 `evalscope[perf]==1.8.1`，不再使用无代理 `pip install evalscope==1.8.1`。
  - 该更新只影响 benchmark client Pod 的依赖安装方式，不改变 serving 部署镜像、P/D/router 语义、Mooncake/DeepEP/DeepGEMM/vLLM runtime 行为或 M10 gate。
- Next:
  - 若继续补齐 definitive C26B 对照，应重新按 C14 获取 workspace-env GPU permit，部署相同镜像和 servingkit-aligned `1P1D` 服务，然后运行 in-cluster evalscope BS512/2048 并采集 Prometheus running/waiting/output TPS 与 `vllm:spec_decode_*` 指标。

### P79: C26B proxy rerun blocked by prefill node NotReady before benchmark

- Date: 2026-07-01
- Trigger: 继续执行目标；C26B 的 Pod 内 evalscope 安装阻塞已通过 P78 的代理安装验证解除，因此尝试重新部署同镜像同语义 `1P1D` 服务以补跑 in-cluster evalscope。
- Subagent decision:
  - 不启用 `subagent-driven-development`。
  - 原因：本轮工作是单一 live cluster mutation、permit 生命周期、Helm release 和 cleanup 的串行流程；并行子代理会增加同一 namespace/permit/artifact 的状态冲突风险。
  - 主线程负责所有集群 mutation、artifact、计划更新和 gate 判断。
- Issue log:
  - `C14` 第一次前置检查在 `kubectl get pods -o custom-columns=...containers[*]...` 被 zsh glob 解析打断，错误为 `zsh: no matches found: custom-columns=...containers[*]...`。
  - Outcome: 错误发生在 permit 申请之前，没有创建 GPU workload。
  - Fix/prevention: C14 命令引用已改为给 `custom-columns=...` 参数加单引号，后续 zsh 环境不会再把 `[*]` 当 glob。
- Permit:
  - Session: `codex-vllm-dsv4-c26b-proxy-20260701-153108`
  - Thread: `019f02ab-92f4-73f3-870b-5f981a254020`
  - Permit: `f50acdc0-e818-43d2-bfa9-c88e5dab11a8`
  - Requested GPUs: `16`
  - Status: `granted` at `2026-07-01T07:31:09+00:00`; released at `2026-07-01T07:47:10+00:00`
- Deployment attempt:
  - Artifact root: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/proxy-rerun/`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - Intended shape: `1P1D`，prefill `192.168.1.148`，decode/router `192.168.1.154`，`prefill.args.noAsyncScheduling=false`，`decode.args.maxNumSeqs=96`，无 `--max-model-len`。
  - Render evidence: `deploy/rendered.yaml` and `deploy/render-grep.txt` show `StormService`、prefill/decode `nvidia.com/gpu: 8`、decode `--max-num-seqs "96"`、Onion model preparation；未出现 `--max-model-len`、runtime hotfix/install pattern。
  - Helm install started and created the expected pods:
    - decode `dsv4-flash-pd-roleset-p6nnt-decode-74586976c-0` on `192.168.1.154` reached `Ready=True`。
    - router `dsv4-flash-pd-router-d596fd5bd-59kxk` started on `192.168.1.154` but restarted while waiting for workers。
    - prefill `dsv4-flash-pd-roleset-p6nnt-prefill-74ddd7796c-0` started on `192.168.1.148` but never reached readiness before node eviction。
- Blocker evidence:
  - Event: `TaintManagerEviction` marked prefill pod for deletion.
  - Node `192.168.1.148` status became `NotReady` / `Ready=Unknown` with reason `NodeStatusUnknown` and message `Kubelet stopped posting node status`。
  - Node taints: `node.kubernetes.io/unreachable` with `NoSchedule` and `NoExecute`。
  - This happened before router smoke, monitoring deploy, or in-cluster evalscope measured run; therefore no C26B benchmark result was produced in this attempt.
- Cleanup:
  - Helm release `dsv4-flash-pd` uninstalled.
  - Namespace `vllm-dsv4-flash-pd` deleted after force-deleting the stuck task-owned prefill pod.
  - Monitoring release was not created in this attempt; `dsv4-flash-pd-monitoring` was not present.
  - Permit `f50acdc0-e818-43d2-bfa9-c88e5dab11a8` released.
  - Independent checks after cleanup found no `vllm-dsv4-flash-pd` namespace, no task pods, and no task-specific `port-forward`/`evalscope`/`helm upgrade` processes.
- Gate:
  - C26B in-cluster evalscope remains incomplete due to external node failure before benchmark.
  - M10 remains blocked because evalscope BS512 Avg gate has not passed.
  - Next retry should not use `192.168.1.148` until it returns Ready; it should reacquire workspace-env permit and select two currently Ready/free 8-GPU nodes.

### P80: C26B in-cluster evalscope proxy rerun completed on 186/154 and failed Avg gate

- Date: 2026-07-01
- Trigger: P79 的首个 proxy rerun 被 `192.168.1.148` 节点 NotReady 阻断后，按当前 Ready/free 节点重试，补齐用户要求的“使用代理安装 evalscope”并验证是否只是本地 port-forward/client path 导致。
- Permit:
  - Session: `codex-vllm-dsv4-c26b-proxy-186154-20260701-154948`
  - Thread: `019f02ab-92f4-73f3-870b-5f981a254020`
  - Permit: `a7389cb9-4e81-4a34-9c65-c8a4c27b22d0`
  - Requested GPUs: `16`
  - Status during benchmark: `running`
- Deployment:
  - Artifact root: `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/proxy-rerun-186-154/`
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - Shape: `1P1D`，prefill `192.168.1.186`，decode/router `192.168.1.154`，`prefill.args.noAsyncScheduling=false`，`decode.args.maxNumSeqs=96`，无 `--max-model-len`。
  - Router smoke 成功，completion id 显示 prefill `192.168.1.186:8000`、decode `192.168.1.154:8001`。
  - Monitoring `up{stack="vllm",release="dsv4-flash-pd"}` 对 prefill/decode 均为 `1`。
- Benchmark command:
  - Pod: `evalscope-bench-incluster-proxy`
  - Client placement: no-GPU Pod on `192.168.1.186`。
  - Proxy install: `python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'`。
  - Formal shape: `parallel=512`，`number=2048`，`prefix-length=65536`，`min/max-tokens=1536`，`temperature=0`，`ignore_eos=true`。
- Results:
  - Seed run completed; TTFT about `13976.75 ms`。
  - Formal run completed with `MEASURED_RUN_EXIT_CODE=0` and Pod phase `Succeeded`。
  - Total / success / failed: `2048 / 2047 / 1`。
  - Avg TTFT: `20492.39 ms`，未通过 `< 10000 ms` gate。
  - Evalscope Overall Avg output throughput: `12613.28 tok/s`，未通过 `>= 14000 tok/s` gate。
  - Workload completion throughput: Overall `12630.03 tok/s`，Last 30s `21912.98 tok/s`，Steady drop 20% `16094.52 tok/s`；用户 gate 按 Avg/Overall，因此仍失败。
  - Speculative decoding: `Decoded Tok/Iter=4.35`，evalscope `Spec. Accept Rate=0.77`。
- Prometheus processing-window evidence:
  - Window: approximately `2026-07-01T16:23:25+0800` to `2026-07-01T16:27:35+0800`。
  - Decode running avg/max: `300.68 / 512.0`。
  - Decode waiting avg/max: `45.30 / 398.0`。
  - Decode generation tps 30s avg/max: `12047.57 / 18842.41`。
  - Spec accepted/draft tps 30s avg: `9143.93 / 11623.09`，rate-derived acceptance `0.7867`。
  - Interpretation: in-cluster evalscope 移除了本地 port-forward，但服务端仍未长期维持 512 running；Prometheus output TPS 与 evalscope Overall Avg 同向，因此不达标不能只归因于本地 port-forward。
- Log scan:
  - Prefill bad-log matches: `0`。
  - Decode matches: `334`，样本均为 `Received stats for out-of-order step ...` warning；未见 Mooncake transfer-region/KV-load/producer timeout 类错误样本。
  - Router matches: `4`，其中 benchmark 窗口内 1 条 `Two-stage processing failed ... Prefill request failed ... connection closed before message completed`，对应 evalscope 1 个 failed request。
- Artifact:
  - `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701/proxy-rerun-186-154/runs/incluster-evalscope-bs512-proxy/summary.md`
  - `.../pod.log` and `.../pod-final.log`
  - `.../prometheus/range-summary.json`
  - `.../logs/{prefill,decode,router}-since40m.log`
- Artifact gap:
  - `kubectl cp` 在 Pod `Succeeded` 后失败：`cannot exec into a container in a completed pod`。
  - 文件级 evalscope outputs 未能复制出 `/tmp`；完整文本摘要已保存在 `pod.log`/`pod-final.log`。后续 benchmark Pod 若需 DB/HTML/JSON 文件，应在命令末尾 sleep 或把 output 写到可持久化卷。
- Gate:
  - 本轮仍阻止 M10，不得更新远端 `iaas_main`。
  - 若继续按计划寻找 128-512 之间可完整通过的 evalscope batch，必须先重启 P/D/router，再按 monitoring 的实际 running capacity 决定候选。
- Cleanup:
  - Benchmark Pod `evalscope-bench-incluster-proxy` 已删除。
  - Helm release `dsv4-flash-pd` 已卸载，namespace `vllm-dsv4-flash-pd` 已删除。
  - Monitoring release `dsv4-flash-pd-monitoring` 已卸载，namespace `vllm-dsv4-flash-pd-monitoring` 已删除。
  - Prometheus port-forward session 已停止。
  - Permit `a7389cb9-4e81-4a34-9c65-c8a4c27b22d0` status `released` at `2026-07-01T08:32:42+00:00`。
  - Final check: 查询两个任务 namespace 返回空；本地无本任务 `port-forward`/`evalscope` 残留。仍存在一个无关 `default svc/tmp-router 18080:8000` port-forward，未清理。

### P81: C26C evalscope proxy BS 降档 sweep 已完成并清理

- Date: 2026-07-01
- Trigger: 用户要求 evalscope 安装必须使用代理，并在 BS512 不达标后继续检查 `128-512` 间是否存在能通过 Avg gate 的 BS；每个失败候选后必须重启服务，避免把坏状态带到下一轮。
- Permit:
  - Session: `codex-vllm-dsv4-evalscope-bs-sweep-20260701-163549`
  - Thread: `019f02ab-92f4-73f3-870b-5f981a254020`
  - Permit: `e927012a-ff9e-4626-a769-d80bc8cac77f`
  - Requested GPUs: `16`
  - Release status: `released` at `2026-07-01T10:09:45+00:00`
- Deployment:
  - Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130`
  - Shape: `1P1D`，prefill `192.168.1.186`，decode/router `192.168.1.154`，P/D 均 8 GPU 且不同节点；`prefill.args.noAsyncScheduling=false`，`decode.args.maxNumSeqs=96`，无 `--max-model-len`。
  - Router smoke after final restart: `/v1/models` model count `1`，`/v1/completions` returned non-empty text `World`。
- Evalscope install:
  - All benchmark Pods used proxy env `HTTP_PROXY/HTTPS_PROXY=http://100.68.170.29:3128` with `NO_PROXY=localhost,127.0.0.1,.svc,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16`。
  - Explicit install command in Pod YAML: `python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'`。
  - Pod output confirmed `evalscope 1.8.1`。
- Benchmark commands:
  - BS400: `evalscope perf --url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000/v1/completions --model deepseek-v4-flash --tokenizer-path /data01/DeepSeek-V4-Flash --parallel 400 --number 1600 --dataset random --prefix-length 65536 --min-prompt-length 0 --max-prompt-length 0 --min-tokens 1536 --max-tokens 1536 --seed 42 --extra-args '{"temperature":0,"ignore_eos":true}' --outputs /tmp/evalscope-bs400/formal`
  - BS256: same arguments with `--parallel 256 --number 1024 --outputs /tmp/evalscope-bs256/formal`
  - BS128: same arguments with `--parallel 128 --number 512 --outputs /tmp/evalscope-bs128/formal`
- Results:
  - BS400 completed `1600 / 1600 / 0`，Avg TTFT `28498.44 ms`，Output Throughput `10525.87 tok/s`，TPOT `17.80 ms`，Decoded Tok/Iter `4.78`，Spec. Accept Rate `0.79`。未通过 `>= 14000 tok/s` Avg gate。
  - BS256 completed `1024 / 1024 / 0`，Avg TTFT `24342.75 ms`，Output Throughput `8042.26 tok/s`，TPOT `14.84 ms`，Decoded Tok/Iter `4.56`，Spec. Accept Rate `0.78`。未通过 Avg gate。
  - BS128 completed `512 / 512 / 0`，Avg TTFT `21165.30 ms`，Output Throughput `4920.15 tok/s`，TPOT `11.09 ms`，Decoded Tok/Iter `4.38`，Spec. Accept Rate `0.77`。未通过 Avg gate。
  - 因用户要求 gate 看 Avg，BS400 的 steady/drop20% `14008.14 tok/s` 不作为通过依据。
- Prometheus processing-window evidence:
  - BS400: decode running avg/max `185.50/400.00`，waiting avg/max `18.85/245.00`，decode tps 30s avg/max `9581.17/18099.00`，rate-derived spec accept ratio avg `0.8642`。
  - BS256: decode running avg/max `121.80/256.00`，waiting avg/max `10.28/131.00`，decode tps 30s avg/max `7042.59/13333.38`，rate-derived spec accept ratio avg `0.7674`。
  - BS128: decode running avg/max `55.91/128.00`，waiting avg/max `7.70/36.00`，decode tps 30s avg/max `4220.63/6990.69`，rate-derived spec accept ratio avg `0.7616`。
- Failure/restart hygiene:
  - BS400 failed Avg gate 后删除 benchmark Pod，并重启 P/D/router 后再跑 BS256。
  - BS256 failed Avg gate 后第一次删除 serving pods 命令把多个 Pod 名拼成单个参数，返回 `NotFound`；随后用 `xargs` 重试成功，等待 P/D/router Ready，并通过 router smoke 后再跑 BS128。
  - BS128 failed Avg gate 后未再启动更小 BS，因为用户要求寻找 `128-512` 间候选；BS128 已是下界且仍远低于门槛。
- Log scan:
  - BS128 broad bad-log regex 命中 `55` 行，主要是 `VLLM_RPC_TIMEOUT` unknown env warning、decode coordinator `out-of-order step` warning 和 router startup timeout text。
  - Mooncake/RDMA 关键词命中大量正常 RDMA 初始化日志；error-like 过滤命中 `8` 条 `Failed to open device mlx5_7 ... GID 3`，均发生在 startup 阶段。
  - 未见本轮请求处理期间 `Mooncake found no common KV transfer regions`、`KV group count mismatch`、`KV load failed`、`handshake compatibility failure`、`request timeout during KV pull` 或 `Sync batch data transfer timeout` 这类 producer timeout/KV pull failure。
- Artifacts:
  - Root: `artifacts/2026-06-29-vllm-dsv4-flash-pd/evalscope-bs-downgrade-sweep-20260701/`
  - Per-run: `runs/evalscope-bs400/`、`runs/evalscope-bs256/`、`runs/evalscope-bs128/`
  - BS128 copied outputs: `runs/evalscope-bs128/pod-output/evalscope-bs128/`
  - Monitoring summaries: `runs/evalscope-bs{400,256,128}/monitoring/range-summary.txt`
  - Cleanup evidence: `final-cleanup/`
- Cleanup:
  - Benchmark Pod `evalscope-bs128-proxy` deleted with namespace cleanup.
  - Helm release `dsv4-flash-pd` uninstalled.
  - Namespace `vllm-dsv4-flash-pd` deleted.
  - Monitoring release `dsv4-flash-pd-monitoring` uninstalled.
  - Namespace `vllm-dsv4-flash-pd-monitoring` deleted.
  - Local router port-forward `18082:30000` and Prometheus port-forward `19091:9090` stopped.
  - Final checks: both namespaces return `NotFound`；local process scan found no task `port-forward` or `evalscope-bs*` process；permit `e927012a-ff9e-4626-a769-d80bc8cac77f` no longer appears in granted/running permits.
- Gate:
  - No evalscope candidate in the tested `128/256/400/512` set passed the Avg output throughput gate.
  - M10 remains blocked; do not update remote `iaas_main` unless gate changes or a later run satisfies the original Avg thresholds.

### P82: C25 gated Mooncake diagnostic logging started

- Date: 2026-07-01
- Trigger: 继续执行 goal；C26C 已证明 evalscope proxy 降档没有满足 Avg gate，M10 继续阻止；用户此前要求定位 attempt 1 producer timeout / Mooncake descriptor / RDMA 细节，且已授权为诊断修改最小调试代码并推送。
- Subagent decision:
  - 不启用 `superpowers:subagent-driven-development`。
  - Rationale: C25 涉及同一个诊断分支、两个紧耦合文件 `vllm/envs.py` 与 `mooncake_connector.py`、同一 ByteIAAS workflow、同一后续 GPU deployment/permit 生命周期；并行子代理会增加 dirty worktree、branch、namespace、permit 和 artifact 冲突风险。
  - Ownership: 主线程负责代码、验证、提交、推送、workflow 触发和后续复测；不分派并行写入。
- Branch:
  - Base branch before C25: `codex/vllm-dsv4-fork-base-byteiaas-build`
  - Diagnostic branch: `codex/vllm-dsv4-mooncake-transfer-diagnostics`
- Code changes:
  - `vllm/envs.py`: added default-off env flag `VLLM_DSV4_MOONCAKE_DIAG`。
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`: added gated `MooncakeDiag` logs for:
    - producer ready timeout: rank, pending request count/sample, timeout seconds;
    - producer batch transfer: remote session, request count/sample, transfer id sample, ret, elapsed seconds, descriptor count, total bytes;
    - producer expired request: pending total, expired count, timeout, need/sent/sending counters, local group count;
    - consumer receive/pulling error: worker address, request count/sample, err_reqs, err_msg, encoded metadata bytes.
  - Behavior guard: all new diagnostic logs are behind `VLLM_DSV4_MOONCAKE_DIAG`; normal values do not enable it. No transfer behavior, timeout, scheduling, retry, fallback import, Mooncake/DeepEP/DeepGEMM config, or deployment semantic value changed.
- Validation:
  - `uv run --no-project python -m py_compile vllm/envs.py vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` passed.
  - `git diff --check -- vllm/envs.py vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` passed.
  - Code diff archived at `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/c25-debug-code.diff`。
- Issue log:
  - CodeGraph was not initialized in this workspace, so codegraph context lookup failed before editing.
  - Outcome: used narrow file reads and existing plan command references instead; no code was edited before the failed codegraph lookup.
  - Prevention: if broader call graph review becomes necessary, initialize CodeGraph in a separate step or continue using narrow symbol/file reads for this two-file diagnostic change.
- Next:
  - Commit, push diagnostic branch, trigger ByteIAAS dev image workflow for `openai-devel` / `cu130`。

### P83: C25 diagnostic branch pushed and ByteIAAS dev workflow queued

- Date: 2026-07-01
- Branch: `codex/vllm-dsv4-mooncake-transfer-diagnostics`
- Commit: `66e76d57e2dd1cefb3b2122054cf0b907892eb61`
- Push: `origin/codex/vllm-dsv4-mooncake-transfer-diagnostics`
- Commit issue:
  - First `git commit -m "chore: add gated Mooncake transfer diagnostics"` failed due to missing DCO sign-off.
  - Error: `ERROR: commit is missing the expected DCO sign-off. Expected exactly: Signed-off-by: Hank Han <hanhan7630@outlook.com>`
  - Fix: reran `git commit -s -m "chore: add gated Mooncake transfer diagnostics"`; commit succeeded with required sign-off.
  - Prevention: use `git commit -s` for this repo.
- Workflow:
  - `gh workflow view byteiaas-release-dev.yml --ref codex/vllm-dsv4-mooncake-transfer-diagnostics --yaml` showed current inputs are only `checkout_ref` and `vllm_version`; older command reference inputs `image_flavors` and `cuda_version` do not exist for this workflow.
  - Trigger command:
    `gh workflow run byteiaas-release-dev.yml --ref codex/vllm-dsv4-mooncake-transfer-diagnostics -f checkout_ref=codex/vllm-dsv4-mooncake-transfer-diagnostics`
  - Run id: `28510442949`
  - Initial status: `queued`
  - Run URL: `https://github.com/bytedance-iaas/vllm/actions/runs/28510442949`
- Current local-only ledger note:
  - This P83 plan update is intentionally not pushed after the workflow trigger, to avoid moving the branch ref for a run that was dispatched with `checkout_ref` as a branch name.
- Next:
  - Poll run `28510442949` until terminal.
  - If success, extract debug image tag/digest from workflow logs and continue C25 deployment only with `VLLM_DSV4_MOONCAKE_DIAG=1` added.
  - If failure, inspect failed job/step and record blocker or minimal fix.

### P84: C25 debug image workflow failed on buildx driver and was patched

- Date: 2026-07-01
- Failed run: `28510442949`
- Failed job: `build-image / build-and-publish-image`，job id `84509182184`
- Failed step: `Build and push AMD64 image by digest`
- Root cause:
  - Log artifact: `artifacts/2026-06-29-vllm-dsv4-flash-pd/c25-workflow-28510442949/build-image.log`
  - Error: `ERROR: failed to build: push-by-digest is currently not implemented for docker driver, please create a new builder instance`
  - This happened before the Dockerfile build began, so it is a CI/buildx builder configuration issue, not a diagnostic code compile/runtime failure.
- Immediate action:
  - Submitted `gh run cancel 28510442949 --repo bytedance-iaas/vllm` to stop the already-failed run and avoid continuing the unrelated wheel job.
- Fix:
  - File: `.github/workflows/_byteiaas-build-and-publish-image.yml`
  - Changed `docker/setup-buildx-action@v3` setup to create a per-run `docker-container` builder:
    - `name: byteiaas-vllm-image-builder-${{ github.run_id }}`
    - `driver: docker-container`
    - `keep-state: false`
    - `cleanup: true`
  - Local BuildKit cache remains controlled by existing `BYTEIAAS_BUILDX_CACHE_ROOT` and `--cache-from/--cache-to type=local` logic, so this does not remove the intended image build cache path.
- Validation:
  - YAML parse check passed for `.github/workflows/_byteiaas-build-and-publish-image.yml` and `.github/workflows/byteiaas-release-dev.yml` using Python `yaml.safe_load`.
  - `git diff --check -- .github/workflows/_byteiaas-build-and-publish-image.yml` passed.
- Next:
  - Commit and push workflow fix to `codex/vllm-dsv4-mooncake-transfer-diagnostics`.
  - Trigger a new ByteIAAS dev workflow run from the same diagnostic branch.
