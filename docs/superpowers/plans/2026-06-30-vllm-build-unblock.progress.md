# vLLM Build Unblock Progress Log

本文档记录详细进展、证据摘录、失败尝试、issue log、subagent 结果和最终摘要。主计划只保留 compact status 和指针。

## 2026-06-30 Initial Planning State

### 目标

- 解决当前 ByteIAAS vLLM 构建卡点。
- 解决 DeepGEMM Python bootstrap 走 `uv` managed Python 导致的代理低速下载问题。
- 解决 wheel workflow 没有稳定 local BuildKit cache 的问题。
- 解决默认 CUDA/C++ build thread 过低，尤其 `BYTEIAAS_BUILD_NVCC_THREADS="1"` 的问题。
- 修改代码后必须持续跟踪构建；如果出现新的失败或明显低速卡点，需要继续定位、最小修复、push、重启构建，重复直到当前分支完成完整构建并达到效率验收。
- 当前分支完整构建和效率验收通过后必须停下来和用户讨论是否合入 `iaas_main`，不得自动合入。

### 当前现场证据

- 当前 run：`28434471432`
- 当前 commit：`67d49b7f7103bbb8338e87decf234fdd82c83b7e`
- `build-image / build-and-publish-image`：
  - job id `84256928677`
  - runner `vllm-byteiaas-build-01`
  - step `Build and push AMD64 image by digest`
- `build-wheel / build-wheel`：
  - job id `84256928772`
  - runner `vllm-byteiaas-build-02`
  - step `Build wheel stage image`
- build-01 现场：
  - load average 接近 0。
  - `/data` 2T，已用约 349G，可用约 1.6T。
  - Docker buildx 正在运行 `--target vllm-openai`。
  - BuildKit 子进程为 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
- build-02 现场：
  - load average 接近 0。
  - `/data` 2T，已用约 43G，可用约 1.9T。
  - Docker build 正在运行 `--target build`。
  - BuildKit 子进程为 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。

### 代码证据

- `tools/setup_deepgemm_pythons.sh` 当前默认从 `pyproject.toml` 的 `requires-python = ">=3.10,<3.15"` 推导 `3.10 3.11 3.12 3.13 3.14`。
- `tools/setup_deepgemm_pythons.sh` 当前对每个版本执行：
  - `uv venv --python "$V" "$venv" --python-preference only-managed --seed`
- `docker/Dockerfile` 当前已将主 `/opt/venv` 的 `uv venv` 改为使用本地 `/usr/bin/python${PYTHON_VERSION}` 或 manylinux `/opt/python/cpXY-cpXY/bin/python${PYTHON_VERSION}`。
- `docker/Dockerfile` 当前未给 DeepGEMM Python matrix 安装全部本地 Python 版本。
- `.github/workflows/_byteiaas-build-and-publish-image.yml` 当前已有 local BuildKit cache wrapper。
- `.github/workflows/_byteiaas-build-wheel.yml` 当前仍使用 `DOCKER_BUILDKIT=1 docker build`，没有 stable local cache wrapper。
- 两个 workflow 当前均设置 `BYTEIAAS_BUILD_NVCC_THREADS: "1"`。

### 假设

- 当前代理不是完全不可用；之前用 `curl --range` 拉同一类 GitHub release asset 能达到可接受速度，慢点主要集中在 `uv` managed Python 下载路径。
- Ubuntu CUDA build 可以通过 apt/deadsnakes 获得 `python3.10` 到 `python3.14` 及 headers；如果某些版本不可用，需要记录为包源 blocker。
- manylinux build path 可以优先使用 `/opt/python/cpXY-cpXY/bin/pythonX.Y`，不需要 apt 安装。
- DeepGEMM CMake 只需要可执行 Python 和 headers；直接传系统解释器路径比强制创建 venv 更适合当前构建。

### 约束

- 不修改 vLLM runtime/model/scheduler/operator 逻辑。
- 不新增镜像内 import/CLI smoke。
- 不更新远端 `iaas_main`。
- 本次不设计远端 cache/S3 cache，只修复本地构建机 cache。
- 不泄露 GitHub、Volcengine、CR 或代理凭据。

### 非目标

- 不做 P/D 部署模板、benchmark 或性能 gate。
- 不解决所有未来 Python 版本支持策略，只解决当前 `>=3.10,<3.15` 构建卡点。
- 不重构 Dockerfile stage 结构。
- 不替换 DeepGEMM、Mooncake、DeepEP 或 `vllm-router` 来源策略。

## Issue Log

- `I1`: 当前 DeepGEMM helper 强制 `--python-preference only-managed`，即使 Dockerfile 已为主 venv 安装了本地 Python 3.12，也会重新走 `uv` managed Python 下载。
- `I2`: `uv` managed Python 在代理下表现为长时间小块读取，CPU 近乎空闲，失败前没有明确超时。
- `I3`: wheel workflow 不使用固定 Buildx builder 和 local cache，导致和 image workflow 的 cache 设计不对齐。
- `I4`: image workflow 已有 local cache，但首次成功前不会产生可复用 cache；如果构建一直卡住，cache 自然无法验证命中。
- `I5`: `BYTEIAAS_BUILD_NVCC_THREADS="1"` 与用户希望 CUDA 相关构建并发跟随核数不一致。
- `I6`: `nvcc_threads=nproc` 可能过度并发；计划保留 env override 作为 fallback。
- `I7`: 一次修复可能只解决当前 `uv /opt/dgenv` 卡点，后续仍可能暴露 apt、pip、cmake、ninja、nvcc、image push 或 cache miss 新卡点；执行计划必须把这些卡点纳入同一闭环，而不是第一次重启构建后就停止。
- `I8`: “效率达标”在本计划中按可观察门槛定义：没有超过 10 分钟的 `uv` managed Python 下载；没有超过 30 分钟的低 CPU、低 IO、无日志进展 BuildKit 空转；CUDA/C++ 编译阶段默认使用 `nproc` 级别并发；同 SHA 复跑读取 local cache。

## Progress Entries

### P0: Plan created

- Summary: 创建 2026-06-30 构建解卡计划，范围限定为 DeepGEMM Python bootstrap、网络慢路径收敛、local BuildKit cache 和构建线程默认值。
- Files:
  - `docs/superpowers/plans/2026-06-30-vllm-build-unblock.md`
  - `docs/superpowers/plans/2026-06-30-vllm-build-unblock.commands.md`
  - `docs/superpowers/plans/2026-06-30-vllm-build-unblock.progress.md`
- Current status: 仅写计划，未修改构建代码。

### P1: User tightened execution target

- Summary: 用户明确目标不是只提交修复，而是实现当前分支完整构建，且没有明显低速卡点；代码修改后需要持续跟踪构建，出现新卡点继续处理并重启构建，直到完整构建和效率达到预期，然后停下来讨论是否合入 `iaas_main`。
- Plan updates:
  - 主计划 `Goal`、`Architecture`、`Global Constraints`、`M7`、`M9`、`Acceptance Criteria`、`Approval Forecast` 和 `Current Status` 已加入闭环执行和停在合入决策前的要求。
  - 命令引用新增 `C15` 新卡点分类和下一轮修复入口。
  - 命令引用新增 `C16` 完整构建和效率验收汇总。
  - 进展日志新增 `I7` 和 `I8`。

### P2: M1 preflight evidence captured

- Summary: 完成 M1/C1/C2。GitHub run `28434471432` 仍在进行；image 与 wheel job 都停在 Docker build 步骤。两台构建机现场均为低 CPU/load，活跃 BuildKit 子进程均是 DeepGEMM Python helper 触发的 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
- Evidence files:
  - `artifacts/2026-06-30-vllm-build-unblock/preflight/C1.txt`
  - `artifacts/2026-06-30-vllm-build-unblock/preflight/C2-build01.txt`
  - `artifacts/2026-06-30-vllm-build-unblock/preflight/C2-build02.txt`
- Evidence excerpts:
  - build-01: `docker buildx build --target vllm-openai ... --build-arg max_jobs=144 --build-arg nvcc_threads=1 ...`
  - build-01: `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`
  - build-02: `docker build --target build ... --build-arg max_jobs=120 --build-arg nvcc_threads=1 ...`
  - build-02: `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`
- Conclusion: 当前卡点是 DeepGEMM managed Python bootstrap，不是主 `/opt/venv` Python 3.12 bootstrap；同时两个 workflow 仍传入 `nvcc_threads=1`。

### P3: M2-M6 local edits and static validation

- Summary: 完成 claimed write scope 内的构建修复并通过本地静态验证。
- Code changes:
  - `tools/setup_deepgemm_pythons.sh`: 对每个目标 Python 版本优先查找系统解释器或 manylinux `/opt/python/cpXY-cpXY` 解释器，并验证 `Python.h`；只有缺失时才 fallback 到带 `timeout 600s`、最多 5 次重试的 `uv venv --python ... --python-preference only-managed --seed`。
  - `docker/Dockerfile`: 新增 `DEEPGEMM_PYTHON_VERSIONS=3.10,3.11,3.12,3.13,3.14`，Ubuntu build path 尝试安装这些版本的 `pythonX.Y`、`pythonX.Y-dev`、`pythonX.Y-venv`；DeepGEMM setup 显式传入该 matrix 并打印最终解释器列表。
  - `.github/workflows/_byteiaas-build-wheel.yml`: 新增 fixed Buildx builder 和 `/data/buildkit/byteiaas-vllm-wheel-cache` local cache；wheel build 改为 `docker buildx build --load`，保留 `local/byteiaas-vllm-wheel:${GITHUB_RUN_ID}`。
  - `.github/workflows/_byteiaas-build-and-publish-image.yml`: 移除固定 `BYTEIAAS_BUILD_NVCC_THREADS: "1"`，默认 `nvcc_threads=${build_cpus}`。
  - 两个 workflow 均保留 `BYTEIAAS_BUILD_NVCC_THREADS` override，默认 `max_jobs` 和 `nvcc_threads` 跟随 `nproc`。
- Validation:
  - `bash -n tools/setup_deepgemm_pythons.sh`: pass。
  - Python YAML parse `_byteiaas-build-wheel.yml` 和 `_byteiaas-build-and-publish-image.yml`: pass。
  - `git diff --check`: pass。
  - claimed write scope guard: pass。
  - 本地 helper 校验：`tools/setup_deepgemm_pythons.sh 3.11` 输出 `/usr/bin/python3.11`，stderr 显示 `DeepGEMM Python 3.11: using system interpreter /usr/bin/python3.11`。
- Risk carried forward: Docker build 仍需验证 deadsnakes/apt 是否提供 `python3.13` 和 `python3.14` packages；若新 run 失败，按 C15 分类后继续最小修复。

### P4: Commit pushed and new build started

- Summary: 完成 M6/C8 和 M7/C9-C10。旧 run 已取消，新 ByteIAAS dev build 已触发。
- Commit:
  - `207612ca3b6aa30cf1ed623ca647df0dcb70cfad ci: unblock vllm docker builds`
  - Pushed branch: `codex/vllm-dsv4-fork-base-byteiaas-build`
- Old run:
  - `28434471432` 已进入 `completed/cancelled`。
  - image job `84256928677` 和 wheel job `84256928772` 均在旧 Docker build step 被取消。
- New run:
  - URL: `https://github.com/bytedance-iaas/vllm/actions/runs/28437275387`
  - run id: `28437275387`
  - head SHA: `207612ca3b6aa30cf1ed623ca647df0dcb70cfad`
  - initial jobs:
    - `84266201333 build-wheel / build-wheel`
    - `84266201334 build-image / build-and-publish-image`
- Next: 按 C11 持续监控新 run，若出现新失败或低速卡点则执行 C15 分类并继续最小修复。

### P5: First live monitoring after new run

- Summary: 新 run 已越过旧的 DeepGEMM `/opt/dgenv/3.10` managed Python 卡点。image 和 wheel build 参数均显示 `nvcc_threads` 默认跟随 `nproc`，wheel workflow 已使用 Buildx local cache 路径。
- Evidence:
  - build-01 image process: `--build-arg max_jobs=144 --build-arg nvcc_threads=144`。
  - build-01 compile process: `nvcc ... --threads=144`，并进入 `cmake --build` / `ninja` 阶段。
  - build-02 wheel process: `docker buildx build --cache-to type=local,dest=/data/buildkit/byteiaas-vllm-wheel-cache/build.new,mode=max --load ... --build-arg max_jobs=120 --build-arg nvcc_threads=120`。
  - No process matching `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed` in current probes.
- Evidence files:
  - `artifacts/2026-06-30-vllm-build-unblock/monitor/C11-build02-182440.txt`
  - `artifacts/2026-06-30-vllm-build-unblock/monitor/C11-ivk-yepkqh1022lbds39tq1b-182923.txt`
  - `artifacts/2026-06-30-vllm-build-unblock/monitor/C11-ivk-yepkqh2emslbds65uvb0-182924.txt`
  - `artifacts/2026-06-30-vllm-build-unblock/monitor/C11-ivk-yepkqusytplbdsrn6sit-183501.txt`
- Watch item: build-01 currently shows `cmake --build . -j=1` because `nvcc_threads=144` consumes the parallelism budget in vLLM's build logic. This may still be acceptable if `nvcc --threads=144` uses CPU effectively; if CPU remains low for the low-speed window, classify as a CUDA build parallelism efficiency issue and adjust.

### P6: CUDA build parallelism adjusted after live efficiency issue

- Summary: 新 run `28437275387` 越过旧 uv 卡点后暴露新效率问题：workflow 将 `nvcc_threads` 直接设为 `nproc`，vLLM `setup.py::compute_num_jobs` 会把 CMake jobs 降为 `MAX_JOBS/NVCC_THREADS=1`；现场只看到单个 `cc1plus` 约 1 核运行，不符合“CUDA/C++ 编译阶段不是单线程低负载”的验收门。
- Evidence:
  - build-01: `cmake --build . -j=1`，`ninja -j 1`，`nvcc --threads=144`。
  - build-02: `cmake --build . -j=1`，`ninja -j 1`，`nvcc --threads=120`，单个 `cc1plus` 约 109% CPU。
  - repo reference: `tools/generate_cmake_presets.py` 使用社区模式 `nvcc_threads = min(4, cpu_cores)`，`cmake_jobs = cpu_cores // nvcc_threads`。
- Fix: 将两个 workflow 的默认 `BYTEIAAS_BUILD_NVCC_THREADS` 改为 `min(4,nproc)`，保留 `MAX_JOBS=nproc`，使 vLLM `setup.py` 计算出的 CMake jobs 接近 `nproc/4`，总 CUDA 编译并发接近机器核数。
- Next: 重新静态验证、提交、push，取消 run `28437275387` 并触发下一轮构建。

## Approval Notes

- 本次 `/plan` 只写计划，不执行审批敏感动作。
- 执行计划时，取消卡住 run、push branch、触发 dev image publish workflow 都属于计划内可预见动作。
- 若执行阶段需要更新 `iaas_main`、force push、引入远端 cache 或修改 runtime 逻辑，必须重新确认。

## Validation Notes

- 静态验证：
  - `bash -n tools/setup_deepgemm_pythons.sh`
  - Python YAML parse 两个 workflow。
  - `git diff --check`
  - claimed write scope guard。
- 动态验证：
  - 新 run 不再长时间卡在 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`。
  - wheel 和 image 构建至少各成功一次。
  - 同 SHA rerun 能证明 local cache 被读取。
  - 每次新 run 若出现失败或低速卡点，必须生成卡点分类 artifact 并进入下一轮修复。
  - 通过 C16 汇总完整构建和效率验收后，停下来讨论 `iaas_main` 合入。

## Final Summary

- 暂无；执行完成后填写。
