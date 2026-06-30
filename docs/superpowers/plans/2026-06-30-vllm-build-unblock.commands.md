# vLLM Build Unblock Commands

本文档只保存可运行命令、用途、工作目录、预期结果和运行时机。执行时每个命令的摘要结果应写入 `docs/superpowers/plans/2026-06-30-vllm-build-unblock.progress.md`。

## C1: 记录本地分支、目标文件和 GitHub run 状态

**When:** 执行阶段第一步，任何代码修改前。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

git status --short
git branch --show-current
git rev-parse HEAD
git remote -v

/home/hanhan.hank/.local/bin/gh-github run view 28434471432 \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,jobs \
  | jq '.'

sed -n '80,245p' docker/Dockerfile
sed -n '410,450p' docker/Dockerfile
sed -n '1,110p' tools/setup_deepgemm_pythons.sh
sed -n '1,260p' .github/workflows/_byteiaas-build-wheel.yml
sed -n '1,260p' .github/workflows/_byteiaas-build-and-publish-image.yml
```

**Expected result:** 输出当前分支、HEAD、run/job/step 状态和待修改文件片段；工作区除 `artifacts/` 和本计划文件外无意外改动。

## C2: 记录 build-01/build-02 当前卡点现场

**When:** C1 后，修改前。

**Working directory:** `/data00/home/hanhan.hank/workspace/v1/byteiaas_ci`

```bash
set -euo pipefail

probe='date "+%F %T %z"; uptime; free -h; df -h /data /data/docker /data/buildkit 2>/dev/null || true; ps -eo pid,ppid,etimes,stat,pcpu,pmem,comm,args --sort=-pcpu | egrep "docker build|buildx|buildkit|containerd-shim|ninja|cmake|gcc|g\\+\\+|nvcc|uv venv|python3.1|pip|apt|Runner.Worker" | head -120 || true'
encoded="$(printf '%s' "${probe}" | base64 -w0)"

ve ecs RunCommand \
  --Region cn-beijing \
  --InstanceIds.1 i-yep2egsxdsxjd1vvsiua \
  --Type Shell \
  --InvocationName probe-vllm-build01 \
  --CommandContent "${encoded}" \
  --Timeout 60 \
  | tee /tmp/vllm-build-unblock-build01-invocation.json

ve ecs RunCommand \
  --Region cn-beijing \
  --InstanceIds.1 i-yepk0h5clc4c5qwd9zsy \
  --Type Shell \
  --InvocationName probe-vllm-build02 \
  --CommandContent "${encoded}" \
  --Timeout 60 \
  | tee /tmp/vllm-build-unblock-build02-invocation.json
```

**Expected result:** 返回两个 `InvocationId`。随后用 `C2A` 读取输出。

## C2A: 读取 Cloud Assistant 探针输出

**When:** C2 下发后 10-30 秒。

**Working directory:** `/data00/home/hanhan.hank/workspace/v1/byteiaas_ci`

```bash
set -euo pipefail

BUILD01_INVOCATION_ID="$(jq -r '.Result.InvocationId' /tmp/vllm-build-unblock-build01-invocation.json)"
BUILD02_INVOCATION_ID="$(jq -r '.Result.InvocationId' /tmp/vllm-build-unblock-build02-invocation.json)"

for id in "${BUILD01_INVOCATION_ID}" "${BUILD02_INVOCATION_ID}"; do
  ve ecs DescribeInvocationResults \
    --Region cn-beijing \
    --InvocationId "${id}" \
    --PageSize 10 \
    | tee "/tmp/${id}.json" \
    | jq -r '.Result.InvocationResults[0].InvocationResultStatus, .Result.InvocationResults[0].Output' \
    | tail -n 1 \
    | base64 -d
done
```

**Expected result:** 输出包含 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed` 时，将其作为当前卡点证据写入 progress log。

## C3: 修改 DeepGEMM Python bootstrap

**When:** M1 证据记录完成后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 用 `apply_patch` 修改 `docker/Dockerfile` 和 `tools/setup_deepgemm_pythons.sh`。

**Expected edit shape:**

- `docker/Dockerfile`
  - 在 base stage 增加 `ARG DEEPGEMM_PYTHON_VERSIONS=3.10,3.11,3.12,3.13,3.14`。
  - Ubuntu path 中，在 deadsnakes source 准备后，安装 `python${PYTHON_VERSION}` 之外，再安装 `${DEEPGEMM_PYTHON_VERSIONS}` 对应的 `pythonX.Y`、`pythonX.Y-dev`、`pythonX.Y-venv`。
  - `ENV DEEPGEMM_PYTHON_VERSIONS=${DEEPGEMM_PYTHON_VERSIONS}`。
  - DeepGEMM setup RUN 继续写 `/tmp/dg_pythons.txt`，但日志输出 `cat /tmp/dg_pythons.txt` 或 stderr 选择信息。
- `tools/setup_deepgemm_pythons.sh`
  - 对每个版本优先查找 `/usr/bin/python$V`、`python$V`、manylinux `/opt/python/cpXY-cpXY/bin/python$V`。
  - 用解释器执行 `sysconfig.get_paths()["include"]` 并检查 `Python.h`。
  - 成功时输出解释器路径，不运行 `uv venv`。
  - fallback `uv venv` 使用 `timeout 600s`、最多 5 次、清理 partial venv 和 `${UV_CACHE_DIR:-/root/.cache/uv}/.temp`。

**Expected result:** `git diff -- docker/Dockerfile tools/setup_deepgemm_pythons.sh` 只包含 DeepGEMM Python bootstrap 和本地 Python 安装相关改动。

## C4: 修改 wheel/image workflow cache 和线程默认值

**When:** C3 后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 用 `apply_patch` 修改两个 workflow。

**Expected edit shape:**

- `.github/workflows/_byteiaas-build-wheel.yml`
  - 新增 `BYTEIAAS_BUILDX_CACHE_ROOT: /data/buildkit/byteiaas-vllm-wheel-cache`。
  - 移除或置空 `BYTEIAAS_BUILD_NVCC_THREADS: "1"`。
  - 增加 `docker/setup-buildx-action@v3`，固定 builder name，例如 `byteiaas-vllm-wheel-builder`。
  - 将 wheel build step 改为 `docker buildx build --load`。
  - 添加与 image workflow 同风格的 `buildx_with_local_cache build ...` wrapper。
- `.github/workflows/_byteiaas-build-and-publish-image.yml`
  - 移除或置空 `BYTEIAAS_BUILD_NVCC_THREADS: "1"`。
  - 修改默认值为 `default_nvcc_threads=min(4, build_cpus)` 和 `build_nvcc_threads="${BYTEIAAS_BUILD_NVCC_THREADS:-${default_nvcc_threads}}"`。
  - 打印 `max_jobs`、`nvcc_threads`、`cpus`、`effective_cuda_jobs` 和 cache source。

**Expected result:** `git diff -- .github/workflows/_byteiaas-build-wheel.yml .github/workflows/_byteiaas-build-and-publish-image.yml` 显示 wheel workflow 使用 local BuildKit cache，两个 workflow 默认 `max_jobs=nproc`、`nvcc_threads=min(4,nproc)`，并打印 `effective_cuda_jobs`。

## C5: 静态检查

**When:** C3-C4 edits 后，commit 前。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

bash -n tools/setup_deepgemm_pythons.sh

python3 - <<'PY'
import pathlib
import yaml
for path in [
    ".github/workflows/_byteiaas-build-wheel.yml",
    ".github/workflows/_byteiaas-build-and-publish-image.yml",
]:
    with open(path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print(f"YAML OK: {path}")
PY

rg -n "uv venv --python.*only-managed|DEEPGEMM_PYTHON_VERSIONS|BYTEIAAS_BUILDX_CACHE_ROOT|BYTEIAAS_BUILD_NVCC_THREADS|build_nvcc_threads" \
  docker/Dockerfile \
  tools/setup_deepgemm_pythons.sh \
  .github/workflows/_byteiaas-build-wheel.yml \
  .github/workflows/_byteiaas-build-and-publish-image.yml
```

**Expected result:** `bash -n` 成功，两个 YAML 文件可解析，`rg` 输出显示 fallback 仍存在但不再是正常路径，workflow cache/thread 变量存在。

## C6: 本地 helper 行为校验

**When:** C5 成功后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

DEEPGEMM_VENV_PREFIX="${tmpdir}/dgenv" \
UV_CACHE_DIR="${tmpdir}/uv-cache" \
tools/setup_deepgemm_pythons.sh "$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
```

**Expected result:** 在本机当前 Python 版本有 headers 时，输出一个解释器路径且 stderr 显示使用 system interpreter；如果本机没有 headers，命令可能触发 fallback，此时记录结果但不以本机失败否定 Docker build 修复。

## C7: Diff review guard

**When:** C5-C6 后，commit 前。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

git diff --name-only
git diff --check

unexpected="$(git diff --name-only | grep -Ev '^(docker/Dockerfile|tools/setup_deepgemm_pythons.sh|\.github/workflows/_byteiaas-build-wheel\.yml|\.github/workflows/_byteiaas-build-and-publish-image\.yml|docs/superpowers/plans/2026-06-30-vllm-build-unblock(\.commands|\.progress)?\.md)$' || true)"
if [ -n "${unexpected}" ]; then
  echo "Unexpected files changed:"
  echo "${unexpected}"
  exit 1
fi
```

**Expected result:** `git diff --check` 成功；changed files 只在 claimed write scope 内。

## C8: Commit and push

**When:** C7 通过后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

branch="$(git branch --show-current)"
git add \
  docker/Dockerfile \
  tools/setup_deepgemm_pythons.sh \
  .github/workflows/_byteiaas-build-wheel.yml \
  .github/workflows/_byteiaas-build-and-publish-image.yml \
  docs/superpowers/plans/2026-06-30-vllm-build-unblock.md \
  docs/superpowers/plans/2026-06-30-vllm-build-unblock.commands.md \
  docs/superpowers/plans/2026-06-30-vllm-build-unblock.progress.md

git commit -m "ci: unblock vllm docker builds"
git push origin "${branch}"
```

**Expected result:** 新 commit 推送到当前 integration branch。不要更新 `iaas_main`。

## C9: 取消旧卡住 run

**When:** C8 push 成功后，触发新构建前。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

/home/hanhan.hank/.local/bin/gh-github run cancel 28434471432 \
  --repo bytedance-iaas/vllm

/home/hanhan.hank/.local/bin/gh-github run view 28434471432 \
  --repo bytedance-iaas/vllm \
  --json status,conclusion \
  | jq '.'
```

**Expected result:** run 进入 `completed/cancelled` 或取消请求已接受。如果 run 已自然结束，记录实际状态。

## C10: 触发新 ByteIAAS dev build

**When:** C9 后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

branch="$(git branch --show-current)"

/home/hanhan.hank/.local/bin/gh-github workflow run byteiaas-release-dev.yml \
  --repo bytedance-iaas/vllm \
  --ref "${branch}"

sleep 10
/home/hanhan.hank/.local/bin/gh-github run list \
  --repo bytedance-iaas/vllm \
  --workflow byteiaas-release-dev.yml \
  --branch "${branch}" \
  --limit 5 \
  --json databaseId,headSha,status,conclusion,createdAt \
  | tee /tmp/vllm-build-unblock-runs.json

jq -r '.[0].databaseId' /tmp/vllm-build-unblock-runs.json \
  | tee /tmp/vllm-build-unblock-run-id
```

**Expected result:** 输出新的 run id，headSha 等于 C8 commit。

## C11: 监控新 run 和现场进程

**When:** C10 触发后，每 5-10 分钟一次，直到 job 成功、失败或明确卡住；如果出现新卡点，继续执行 C15，而不是停在第一次失败。

**Working directory:** `/data00/home/hanhan.hank/workspace/v1/byteiaas_ci`

```bash
set -euo pipefail

RUN_ID="$(cat /tmp/vllm-build-unblock-run-id)"

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,jobs \
  | jq '.'

probe='date "+%F %T %z"; uptime; ps -eo pid,ppid,etimes,stat,pcpu,pmem,comm,args --sort=-pcpu | egrep "docker build|buildx|buildkit|ninja|cmake|gcc|g\\+\\+|nvcc|uv venv|python3.1|Runner.Worker" | head -120 || true'
encoded="$(printf '%s' "${probe}" | base64 -w0)"

for instance in i-yep2egsxdsxjd1vvsiua i-yepk0h5clc4c5qwd9zsy; do
  ve ecs RunCommand \
    --Region cn-beijing \
    --InstanceIds.1 "${instance}" \
    --Type Shell \
    --InvocationName "probe-${RUN_ID}-${instance##*-}" \
    --CommandContent "${encoded}" \
    --Timeout 60
done
```

**Expected result:** 新 run 不出现长时间 `uv venv --python 3.10 /opt/dgenv/3.10 --python-preference only-managed --seed`；若出现任何低速卡点或失败，记录 elapsed、活跃进程、CPU/load 和日志，并进入 C15。

## C12: 检查 local cache 目录

**When:** 新 run 第一次成功后。

**Working directory:** `/data00/home/hanhan.hank/workspace/v1/byteiaas_ci`

```bash
set -euo pipefail

probe='date "+%F %T %z"; du -sh /data/buildkit/byteiaas-vllm-image-cache/* /data/buildkit/byteiaas-vllm-wheel-cache/* 2>/dev/null || true; find /data/buildkit -maxdepth 3 -type d -name "byteiaas-vllm-*-cache" -print'
encoded="$(printf '%s' "${probe}" | base64 -w0)"

for instance in i-yep2egsxdsxjd1vvsiua i-yepk0h5clc4c5qwd9zsy; do
  ve ecs RunCommand \
    --Region cn-beijing \
    --InstanceIds.1 "${instance}" \
    --Type Shell \
    --InvocationName "cache-check-${instance}" \
    --CommandContent "${encoded}" \
    --Timeout 60
done
```

**Expected result:** build-01 至少有 image cache，build-02 至少有 wheel cache；大小非 0。

## C13: 同 SHA rerun 验证 cache

**When:** C12 确认 cache 存在后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

RUN_ID="$(cat /tmp/vllm-build-unblock-run-id)"

/home/hanhan.hank/.local/bin/gh-github run rerun "${RUN_ID}" \
  --repo bytedance-iaas/vllm

sleep 10
/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,jobs \
  | jq '.'
```

**Expected result:** rerun 启动；如果 GitHub 为 rerun 生成新的 attempt，后续用同一 run id 查看 attempt logs。

## C14: 获取完成后日志并验证 cache 命中

**When:** C13 rerun 完成后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

RUN_ID="$(cat /tmp/vllm-build-unblock-run-id)"
LOG_DIR="artifacts/2026-06-30-vllm-build-unblock/logs-${RUN_ID}"
mkdir -p "${LOG_DIR}"

/home/hanhan.hank/.local/bin/gh-github run download "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --dir "${LOG_DIR}" || true

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --log > "${LOG_DIR}/run.log"

rg -n "Using local BuildKit cache|No local BuildKit cache yet|Updated local BuildKit cache|CACHED|DeepGEMM Python .*using system interpreter|uv venv --python 3.10 /opt/dgenv" "${LOG_DIR}"
```

**Expected result:** 日志包含 `Using local BuildKit cache` 和 DeepGEMM system interpreter 证据；不包含长时间卡住的 `/opt/dgenv/3.10` managed Python 路径。若未命中 cache，记录 exact log。

## C15: 新卡点分类和下一轮修复入口

**When:** C11 监控发现新 run 失败、明显低速卡点、或超过 30 分钟低 CPU/低 IO/无日志进展时。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

RUN_ID="$(cat /tmp/vllm-build-unblock-run-id)"
ARTIFACT_DIR="artifacts/2026-06-30-vllm-build-unblock/stall-${RUN_ID}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${ARTIFACT_DIR}"

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,jobs \
  | tee "${ARTIFACT_DIR}/run-status.json"

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --log > "${ARTIFACT_DIR}/run.log" || true

rg -n "error|failed|timeout|uv venv|DeepGEMM|CACHED|RUN |Building wheel|nvcc|ninja|cmake|Using local BuildKit cache|No local BuildKit cache" \
  "${ARTIFACT_DIR}/run.log" \
  > "${ARTIFACT_DIR}/interesting-lines.txt" || true

cat > "${ARTIFACT_DIR}/classification.md" <<'EOF'
# 新卡点分类

- run id:
- head sha:
- 卡点类型: network-download / dependency-install / cmake-configure / cuda-compile / docker-cache / image-push / wheel-copy / unknown
- 证据:
- 最小修复范围:
- 是否仍在 claimed write scope 内:
- 下一步:
EOF

echo "${ARTIFACT_DIR}"
```

**Expected result:** 生成一个卡点 artifact 目录，包含 run status、日志、interesting lines 和人工补全的分类模板。执行者必须把分类摘要写入 progress log，然后回到 M2-M6 做最小修复、push，并用 C10 重新触发构建。

## C16: 完整构建和效率验收汇总

**When:** 一次完整构建成功，并且同 SHA cache 验证构建完成后。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

RUN_ID="$(cat /tmp/vllm-build-unblock-run-id)"
SUMMARY_DIR="artifacts/2026-06-30-vllm-build-unblock/summary-${RUN_ID}"
mkdir -p "${SUMMARY_DIR}"

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,jobs \
  | tee "${SUMMARY_DIR}/run-status.json"

/home/hanhan.hank/.local/bin/gh-github run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --log > "${SUMMARY_DIR}/run.log"

python3 - <<'PY' "${SUMMARY_DIR}/run-status.json" "${SUMMARY_DIR}/jobs-summary.tsv"
import json
import sys
from datetime import datetime, timezone

src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
rows = []
for job in data.get("jobs", []):
    started = job.get("startedAt")
    completed = job.get("completedAt")
    duration = ""
    if started and completed and not completed.startswith("0001-"):
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        c = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        duration = str(c - s)
    rows.append([
        str(job.get("databaseId", "")),
        job.get("name", ""),
        job.get("status", ""),
        job.get("conclusion", ""),
        duration,
    ])
with open(dst, "w", encoding="utf-8") as f:
    f.write("job_id\tname\tstatus\tconclusion\tduration\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
PY

rg -n "Using build parallelism|max_jobs=|nvcc_threads=|DeepGEMM Python .*using system interpreter|Using local BuildKit cache|Updated local BuildKit cache|CACHED|Published openai image|Published openai-devel image|Upload wheel artifact" \
  "${SUMMARY_DIR}/run.log" \
  > "${SUMMARY_DIR}/evidence-lines.txt" || true

cat > "${SUMMARY_DIR}/acceptance.md" <<'EOF'
# 构建和效率验收

- 完整构建 run:
- head sha:
- wheel artifact: pass/fail
- openai image publish: pass/fail
- openai-devel image publish: pass/fail
- cache rerun 读取 local cache: pass/fail
- 无超过 10 分钟 uv managed Python 下载: pass/fail
- 无超过 30 分钟低 CPU/低 IO/无日志进展 BuildKit 空转: pass/fail
- CUDA/C++ 编译线程默认等于 nproc: pass/fail
- 是否停止并进入 iaas_main 合入讨论: yes/no
EOF

echo "${SUMMARY_DIR}"
```

**Expected result:** 生成 summary artifact，包含 job 时长、关键日志证据和验收表。只有 `acceptance.md` 全部满足时，才能停止并和用户讨论是否合入 `iaas_main`；不能自动合入。
