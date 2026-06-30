# vLLM Build Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:subagent-driven-development` only for independent read-only review of GitHub Actions logs while the main worker owns code edits and workflow reruns. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解决当前 ByteIAAS vLLM 构建卡在 DeepGEMM Python bootstrap、代理下载过慢、wheel 构建不复用稳定 BuildKit cache、CUDA/C++ 构建线程数过低的问题，并在当前分支完成一次完整构建和一次 cache 验证构建后停下来讨论是否合入 `iaas_main`。

**Architecture:** 先把 DeepGEMM 多 Python 准备从 `uv` managed Python 下载路径改为优先使用镜像内本地 `/usr/bin/python3.x` 或 manylinux `/opt/python/cpXY-cpXY/bin/python3.x`，只在缺失时带超时重试 fallback 到 `uv`。然后把 wheel workflow 切到与 image workflow 同类的 local BuildKit cache，并让 image/wheel 的 CUDA/C++ 默认总并发接近机器核数：`MAX_JOBS=nproc`，`NVCC_THREADS=min(4,nproc)`，由 vLLM `setup.py` 计算 CMake jobs 为 `MAX_JOBS/NVCC_THREADS`。最后用 GitHub Actions 和构建机现场进程做闭环验证：每次出现失败或明显低速卡点，都先定位新的具体卡点，再做最小修复、重启构建并继续监控，直到当前分支完成完整构建，且同 SHA 复跑证明 cache 被读取、无明显低速空转卡点。

**Tech Stack:** `bytedance-iaas/vllm`, GitHub Actions self-hosted runners, Docker Buildx, BuildKit local cache, CUDA 13.0.2, Ubuntu 22.04, manylinux2_28, `uv`, DeepGEMM, `ve ecs RunCommand`, `/home/hanhan.hank/.local/bin/gh-github`.

## Global Constraints

- 计划、命令引用、进展日志、目标、背景、假设、约束、非目标、里程碑、风险、审批说明、验证说明、进展摘要和最终摘要使用中文。
- 命令、文件路径、代码标识、错误信息、API 名称和专有名词保持原文。
- 主计划路径：`docs/superpowers/plans/2026-06-30-vllm-build-unblock.md`
- 命令引用路径：`docs/superpowers/plans/2026-06-30-vllm-build-unblock.commands.md`
- 进展日志路径：`docs/superpowers/plans/2026-06-30-vllm-build-unblock.progress.md`
- 不创建 `.codex/plans`。
- 保持范围只在构建系统、Dockerfile、DeepGEMM 构建辅助脚本和验证命令内；不修改 vLLM runtime/model/scheduler/operator 逻辑。
- 不新增镜像内 import/CLI smoke；验证通过 workflow 日志、构建产物、进程现场和 cache 证据完成。
- 不更新远端 `iaas_main`；本计划只修复当前 integration branch 的构建路径。
- 当前分支完整构建和效率验证完成后必须停下来和用户讨论是否合入 `iaas_main`；不得自动更新 `iaas_main`。
- 不引入远端 cache 或外部新服务；本次先只做本机构建机 local cache。
- 不把代理问题用关闭代理绕过；优先减少不必要外网下载，并保留现有代理参数。
- 执行不是单次尝试：如果重启构建后出现新的失败或低速卡点，必须继续按“观察 -> 定位 -> 最小修复 -> push -> 重启构建”的循环处理，直到达到本计划验收标准或遇到需要用户决策的边界。

---

## Context Summary

- 当前运行 `28434471432` 的两个 job 均在 Docker build 步骤内长时间运行：
  - `build-image / build-and-publish-image` job `84256928677`，step `Build and push AMD64 image by digest`。
  - `build-wheel / build-wheel` job `84256928772`，step `Build wheel stage image`。
- 两台构建机现场都显示 CPU/load 几乎为 0，活跃 BuildKit 子进程均为 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
- 卡点来源：
  - `docker/Dockerfile` 约 `437` 行：`ENV DEEPGEMM_VENV_PREFIX=/opt/dgenv`
  - `docker/Dockerfile` 约 `438` 行：`tools/setup_deepgemm_pythons.sh > /tmp/dg_pythons.txt`
  - `tools/setup_deepgemm_pythons.sh` 约 `32` 行：`uv venv --python "$V" "$venv" --python-preference only-managed --seed`
- `pyproject.toml` 当前 `requires-python = ">=3.10,<3.15"`，因此 helper 默认会尝试 `3.10 3.11 3.12 3.13 3.14`。
- 主 `/opt/venv` 的 Python 3.12 bootstrap 已改为 `/usr/bin/python3.12`，但 DeepGEMM `/opt/dgenv/*` 仍走 managed Python 下载路径。
- `docker/Dockerfile` 当前已支持 Ubuntu path 安装 `python${PYTHON_VERSION}`、`python${PYTHON_VERSION}-dev`、`python${PYTHON_VERSION}-venv`；还没有为 DeepGEMM Python matrix 安装 `3.10-3.14`。
- `.github/workflows/_byteiaas-build-and-publish-image.yml` 已有 local BuildKit cache wrapper，`BYTEIAAS_BUILDX_CACHE_ROOT=/data/buildkit/byteiaas-vllm-image-cache`，并带 `--cache-from` 与 `--cache-to`。
- `.github/workflows/_byteiaas-build-wheel.yml` 仍使用 `DOCKER_BUILDKIT=1 docker build`，没有类似 image workflow 的 stable local cache wrapper。
- 两个 workflow 当前 `BYTEIAAS_BUILD_NVCC_THREADS: "1"`，虽然 `max_jobs` 默认已等于 `nproc`。

## Owner And Write Scope

- Owner：当前执行 agent。
- Claimed write scope：
  - `docker/Dockerfile`
  - `tools/setup_deepgemm_pythons.sh`
  - `.github/workflows/_byteiaas-build-wheel.yml`
  - `.github/workflows/_byteiaas-build-and-publish-image.yml`
  - `docs/superpowers/plans/2026-06-30-vllm-build-unblock*.md`
- Explicitly out of scope：
  - `vllm/**` runtime Python 逻辑
  - `csrc/**` 算子语义
  - `cmake/**` 非构建入口逻辑
  - 远端 `iaas_main` 更新
  - 新增远端 cache、S3 cache、付费服务或生产配置

## Milestones

### M1: 建立当前卡点和基线证据

- [ ] 按命令引用 `C1` 记录当前分支、工作区状态、目标文件片段和运行 `28434471432` 的 GitHub 状态。
- [ ] 按命令引用 `C2` 记录 build-01/build-02 现场进程、磁盘、内存和当前卡住的 `uv venv --python 3.10 /opt/dgenv/3.10` 证据。
- Acceptance: 进展日志包含 run/job/step、两台机器的进程证据，以及 “当前卡在 DeepGEMM managed Python bootstrap，不是主 `/opt/venv` Python 3.12 bootstrap” 的结论。
- References: commands `C1-C2`，progress entries `P1-P2`。

### M2: 让 DeepGEMM Python matrix 优先使用本地 Python

- [ ] 修改 `docker/Dockerfile` Ubuntu base 依赖安装：新增 `ARG DEEPGEMM_PYTHON_VERSIONS=3.10,3.11,3.12,3.13,3.14`，在非 manylinux path 中确保这些版本的 `pythonX.Y`、`pythonX.Y-dev`、`pythonX.Y-venv` 可从系统 apt/deadsnakes 安装。
- [ ] 修改 `tools/setup_deepgemm_pythons.sh`：对每个版本先查找系统解释器并验证 `Python.h` 存在，成功时直接输出该解释器路径，不创建 `/opt/dgenv/$V`。
- [ ] 保留 fallback：系统解释器缺失或 headers 不完整时，才创建 `$DEEPGEMM_VENV_PREFIX/$V`，并使用带 `timeout` 和重试的 `uv venv --python "$V" ... --python-preference only-managed --seed`。
- [ ] 在 helper 日志中把路径选择写到 stderr，例如 `DeepGEMM Python 3.10: using system interpreter /usr/bin/python3.10`，stdout 只保留冒号分隔解释器列表。
- Acceptance: 构建日志中 DeepGEMM 步骤优先显示 `using system interpreter`；构建机不再出现长时间 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
- References: commands `C3-C6`，progress entries `P3-P5`。

### M3: 收敛代理网速风险和失败时间

- [ ] 保留现有 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`/`NO_PROXY` 和 `UV_NATIVE_TLS=true`。
- [ ] 通过 M2 消除 DeepGEMM 常规路径上的 `uv` managed Python 下载。
- [ ] 对 fallback `uv` managed Python 加 `timeout 600s`、最多 5 次重试、清理 partial venv 和 uv 临时目录，避免无界卡住。
- [ ] 在 workflow 日志中打印 `uv --version`、`UV_NATIVE_TLS`、`UV_CACHE_DIR`、`DEEPGEMM_PYTHON_VERSIONS`，但不打印 secret。
- Acceptance: 如果 fallback 被触发，单次 fallback 不超过 600s，失败会明确输出版本和 attempt；正常 Ubuntu CUDA 构建不触发 fallback。
- References: commands `C3-C7`，progress entries `P3-P6`。

### M4: 让 wheel 构建复用 local BuildKit cache

- [ ] 修改 `.github/workflows/_byteiaas-build-wheel.yml`：增加 `BYTEIAAS_BUILDX_CACHE_ROOT=/data/buildkit/byteiaas-vllm-wheel-cache`。
- [ ] 在 wheel workflow 增加 `docker/setup-buildx-action@v3`，使用固定 builder name、`network=host`、Docker Hub mirror、`keep-state: true`、`cleanup: false`。
- [ ] 把 `DOCKER_BUILDKIT=1 docker build --target build ...` 改为 `docker buildx build --target build --load ...`，并添加 `--cache-from type=local,src=${cache_dir}` 与 `--cache-to type=local,dest=${cache_new},mode=max` 的原子 cache promote 逻辑。
- [ ] 保留最终本地镜像 tag `local/byteiaas-vllm-wheel:${GITHUB_RUN_ID}`，让后续 `docker create`/`docker cp` 步骤不变。
- Acceptance: wheel job 日志出现 `Using local BuildKit cache: /data/buildkit/byteiaas-vllm-wheel-cache/build` 或首次出现 `No local BuildKit cache yet`，成功后出现 `Updated local BuildKit cache`；第二次同 SHA rerun 能看到 BuildKit cache 命中证据。
- References: commands `C4-C9`，progress entries `P6-P8`。

### M5: 提升 CUDA/C++ 构建线程默认值

- [ ] 修改 `.github/workflows/_byteiaas-build-and-publish-image.yml`：把 `BYTEIAAS_BUILD_NVCC_THREADS` 默认从 `"1"` 改成社区模式 `min(4,nproc)`。
- [ ] 修改 `.github/workflows/_byteiaas-build-wheel.yml`：同样把 `BYTEIAAS_BUILD_NVCC_THREADS` 默认从 `"1"` 改成社区模式 `min(4,nproc)`。
- [ ] 修改两个 workflow 的 shell：`build_nvcc_threads="${BYTEIAAS_BUILD_NVCC_THREADS:-${default_nvcc_threads}}"`，继续允许手工 env 覆盖；`build_max_jobs` 仍默认 `nproc`，由 vLLM `setup.py` 把 CMake jobs 算为 `MAX_JOBS/NVCC_THREADS`，使总 CUDA 编译并发接近核数。
- [ ] 保持 `build_max_jobs="${BYTEIAAS_BUILD_MAX_JOBS:-${build_cpus}}"`。
- Acceptance: 新 workflow 日志打印 `max_jobs=<nproc>`、`nvcc_threads=4`、`effective_cuda_jobs=<nproc/4>`；build-01 预期约 `max_jobs=144, nvcc_threads=4, effective_cuda_jobs=36`，build-02 预期约 `max_jobs=120, nvcc_threads=4, effective_cuda_jobs=30`，除非 runner 实际 `nproc` 不同。
- References: commands `C4-C7`，progress entries `P6-P8`。

### M6: 静态验证、提交和推送修复分支

- [ ] 按命令引用 `C5` 做 shell/YAML/static 检查。
- [ ] 按命令引用 `C6` 在本地运行 helper 的非破坏性校验；如果本机缺少 Python matrix，只验证语法和显式版本参数行为，不要求本机安装 3.10-3.14。
- [ ] 按命令引用 `C7` review diff，确认没有 runtime 源码改动。
- [ ] 按命令引用 `C8` commit 并 push 当前分支。
- Acceptance: `git diff --name-only` 只包含 claimed write scope；commit pushed 到当前 integration branch。
- References: commands `C5-C8`，progress entries `P8-P9`。

### M7: 取消旧卡住 run，触发并监控新构建

- [ ] 按命令引用 `C9` 取消当前已知卡住 run `28434471432`，避免继续占用 build-01/build-02。
- [ ] 按命令引用 `C10` 触发 dev workflow 或 rerun 目标 workflow，使用修复后的 branch/ref。
- [ ] 按命令引用 `C11` 监控 GitHub job 状态和两台机器现场进程。
- [ ] 如果出现新的失败或低速卡点，按命令引用 `C15` 记录卡点分类，回到 M2-M6 做最小修复并重启构建。
- Acceptance: 新 run 中 DeepGEMM Python setup 不再长时间停在 `uv venv --python 3.10 /opt/dgenv/3.10`；如果失败，失败点有明确日志而不是无界低速下载；若出现新卡点，进展日志必须包含下一轮修复记录。
- References: commands `C9-C11`，progress entries `P10-P12`。

### M8: 验证 cache 和构建完成

- [ ] 第一次新构建成功后，按命令引用 `C12` 检查 build-01/build-02 的 `/data/buildkit/byteiaas-vllm-*-cache` 目录。
- [ ] 按命令引用 `C13` 对同一 SHA 再触发一次构建或 rerun，验证 local cache 是否被读取。
- [ ] 按命令引用 `C14` 抓取第二次构建日志，检查 `Using local BuildKit cache`、`CACHED` 或明显跳过慢层的时间证据。
- Acceptance: wheel 和 image 至少各有一次成功构建；第二次同 SHA 构建能证明读取 local cache；若 cache 未命中，进展日志必须记录具体 cache-miss 原因和下一步。
- References: commands `C12-C14`，progress entries `P13-P15`。

### M9: 完成效率验收并停在合入决策前

- [ ] 按命令引用 `C16` 汇总完整构建 run 和 cache 验证 run 的 job 时长、关键慢层、cache 命中证据、构建机 CPU/IO/网络现场摘要。
- [ ] 确认当前分支已完成完整构建：wheel artifact 上传成功，openai/openai-devel image tag 或 digest 发布成功。
- [ ] 确认无明显低速卡点：没有超过 10 分钟的 `uv` managed Python 下载；没有超过 30 分钟的低 CPU、低 IO、无日志进展 BuildKit 空转；CUDA/C++ 编译阶段看到 `max_jobs` 和 `nvcc_threads` 默认等于 `nproc`，且编译阶段不是单线程低负载。
- [ ] 若 M9 任一效率门不满足，回到 M7 继续定位和修复，不进入 `iaas_main` 讨论。
- [ ] M9 全部满足后，停止执行并向用户报告是否建议合入 `iaas_main`，等待用户决策。
- Acceptance: 进展日志包含“完整构建通过”和“效率验收通过”的证据；最终响应停在 `iaas_main` 合入讨论，不执行远端 `iaas_main` 更新。
- References: commands `C16`，progress entries `P16-P17`。

## Acceptance Criteria

- DeepGEMM Python matrix 在 Ubuntu CUDA build 中优先使用本地 Python，不再正常路径触发 `uv --python-preference only-managed` 下载。
- fallback `uv` managed Python 有明确 timeout、重试和清理，不会无界卡住。
- wheel workflow 使用 fixed Buildx builder 和 local BuildKit cache，保留后续 `docker create` / `docker cp` wheel 提取路径。
- image workflow 继续使用 local BuildKit cache，并保留现有 Volcengine CR publish 行为。
- image/wheel workflow 默认 `max_jobs=nproc`、`nvcc_threads=min(4,nproc)`，总 CUDA 编译并发接近核数，同时保留 env override。
- 新 run 不再卡在 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
- 至少一次 wheel 构建成功并上传 wheel artifact。
- 至少一次 image 构建成功并发布 openai/openai-devel image tag 或 digest。
- 同 SHA rerun 证明 local cache 被读取；如果 cache 未命中，必须有具体原因和下一步，不把“存在 cache 目录”当作命中证明。
- 构建过程没有明显低速卡点：没有超过 10 分钟的 `uv` managed Python 下载；没有超过 30 分钟的低 CPU、低 IO、无日志进展 BuildKit 空转；CUDA/C++ 编译阶段不是单线程低负载。
- 如果重启后的构建出现新卡点，必须继续处理并重启构建，直到当前分支完整构建通过并达到效率验收，或遇到必须用户决策的边界。
- 达成完整构建和效率验收后，停止并和用户讨论是否合入 `iaas_main`，不得自动合入。

## Approval Forecast

- 本计划写入完成不需要审批；本轮不执行代码修改。
- 执行阶段按用户继续指令可直接进行的动作：
  - 修改 claimed write scope 内的源码和 workflow 文件。
  - commit 并 push 当前 integration branch。
  - 取消当前卡住的 GitHub Actions run `28434471432`，避免占用构建机。
  - 触发 ByteIAAS dev build workflow；该 workflow 会向 Volcengine CR 发布 dev image。
  - 如新 run 出现新的构建失败或明显低速卡点，在 claimed write scope 内继续做最小修复、push 并重启构建。
- 需要重新确认的情况：
  - 要更新远端 `iaas_main`。
  - 要 force push 或改写已有远端历史。
  - 要引入远端 cache、S3、付费服务、生产凭据或生产部署。
  - 要修改 `vllm/**` runtime 逻辑来绕过构建或运行问题。
- 如执行时需要确认，建议审批措辞：
  - `确认取消 run 28434471432，并触发当前分支的新 ByteIAAS dev build，允许发布 dev image 到 Volcengine CR。`

## Risks And Fallbacks

- `python3.13` 或 `python3.14` 在 Ubuntu 22.04 apt/deadsnakes 镜像不可用：fallback 是把缺失版本记录为外部包源 blocker，或将 DeepGEMM matrix 限制为当前实际 wheel 需要版本，但后者需要用户确认，因为会改变 Python ABI 覆盖。
- `nvcc_threads=nproc` 可能造成 CPU 过度并发或内存压力：fallback 是保留 `BYTEIAAS_BUILD_NVCC_THREADS` override，失败后用较低值重跑并记录证据。
- local BuildKit cache 首次构建无法加速：这是预期；必须通过第二次同 SHA rerun 验证 cache 读取。
- BuildKit local cache 因磁盘空间或损坏无法读取：fallback 是清理该 job 对应 cache 目录后重建，不能删除无关 cache。
- GitHub in-progress job logs API 可能继续返回 404：fallback 是通过 `ve ecs RunCommand` 查看 runner 进程，job 完成后再抓 logs。

## Current Status

- 状态：计划已创建，尚未执行代码修改。
- 当前已知 blocker：run `28434471432` 的 image/wheel job 正卡在 DeepGEMM `/opt/dgenv/3.10` managed Python bootstrap。
- Next action：使用 `superpowers:executing-plans` 从 `M1` 开始执行，先记录当前证据，再修改 `docker/Dockerfile` 和 `tools/setup_deepgemm_pythons.sh`；执行后必须持续跟踪构建，直到当前分支完整构建和效率验收完成，再停下来讨论是否合入 `iaas_main`。

## References

- 命令引用：`docs/superpowers/plans/2026-06-30-vllm-build-unblock.commands.md`
- 进展日志：`docs/superpowers/plans/2026-06-30-vllm-build-unblock.progress.md`
- 相关旧计划：`docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.md`
