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
- `I7`: 用户已明确此处先跳过 Prometheus；summary 必须标记 Prometheus skipped by user，不得声称有完整服务侧 measured-window monitoring 诊断。

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
