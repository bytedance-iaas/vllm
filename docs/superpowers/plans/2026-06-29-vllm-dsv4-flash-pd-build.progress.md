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
