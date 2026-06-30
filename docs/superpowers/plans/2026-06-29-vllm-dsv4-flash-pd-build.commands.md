# vLLM DSV4 Flash P/D Fork-Base Build Migration Commands

本文档只保存可运行命令、用途、工作目录、预期结果和运行时机。执行时每个命令的摘要结果应写入 `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.progress.md`。

## C1: 获取当前远端状态

**When:** 执行阶段第一步，任何分支变更或远端备份之前。

**Working directory:** 干净的 `bytedance-iaas/vllm` 工作区。

```bash
set -euo pipefail

git status --short
git status --short -- ':!docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.md' ':!docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.commands.md' ':!docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build.progress.md'
git remote -v
git fetch origin iaas_main --tags
git fetch https://github.com/wangyicong52/vllm.git \
  dev/dsv4-mooncake-pp-megamoe:refs/remotes/wangyicong52/dev-dsv4-mooncake-pp-megamoe

ORIGINAL_IAAS_MAIN_SHA="$(git rev-parse origin/iaas_main)"
FORK_BASE_SHA="$(git rev-parse refs/remotes/wangyicong52/dev-dsv4-mooncake-pp-megamoe)"

echo "ORIGINAL_IAAS_MAIN_SHA=${ORIGINAL_IAAS_MAIN_SHA}"
echo "FORK_BASE_SHA=${FORK_BASE_SHA}"
git log -1 --oneline "${ORIGINAL_IAAS_MAIN_SHA}"
git log -1 --oneline "${FORK_BASE_SHA}"
```

**Expected result:** 除本计划三份 `docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build*.md` 文件外，工作区无未提交改动；输出旧 `origin/iaas_main` SHA 和 fork base SHA。

## C2: 创建 GitHub 远端备份分支

**When:** C1 完成后。用户已授权本计划内远端备份，除非目标仓库或分支名偏离本计划，否则不需要再次停下确认。

**Working directory:** 同 C1。

```bash
set -euo pipefail

BACKUP_BRANCH="backup/iaas_main-20260629"
ORIGINAL_IAAS_MAIN_SHA="$(git rev-parse origin/iaas_main)"

git push origin \
  "${ORIGINAL_IAAS_MAIN_SHA}:refs/heads/${BACKUP_BRANCH}"

git ls-remote origin "refs/heads/${BACKUP_BRANCH}"
```

**Expected result:** `git ls-remote` 输出的 SHA 等于 `ORIGINAL_IAAS_MAIN_SHA`。

## C2A: 为备份分支格式添加 GitHub 保护规则

**When:** C2 远端备份分支创建并验证成功后。

**Working directory:** 同 C1。

```bash
set -euo pipefail

RULESET_NAME="Protect backup iaas_main branches"
RULESET_JSON="/tmp/protect-backup-iaas-main-ruleset.json"

cat > "${RULESET_JSON}" <<'JSON'
{
  "name": "Protect backup iaas_main branches",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/backup/iaas_main-*"],
      "exclude": []
    }
  },
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"}
  ]
}
JSON

RULESET_ID="$(gh api /repos/bytedance-iaas/vllm/rulesets \
  --jq ".[] | select(.name==\"${RULESET_NAME}\") | .id" | head -n 1 || true)"

if [ -n "${RULESET_ID}" ]; then
  gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    "/repos/bytedance-iaas/vllm/rulesets/${RULESET_ID}" \
    --input "${RULESET_JSON}"
else
  gh api \
    --method POST \
    -H "Accept: application/vnd.github+json" \
    /repos/bytedance-iaas/vllm/rulesets \
    --input "${RULESET_JSON}"
fi

gh api /repos/bytedance-iaas/vllm/rulesets \
  --jq '.[] | select(.name=="Protect backup iaas_main branches") | {id,name,target,enforcement,conditions,rules}'
```

**Expected result:** GitHub ruleset 创建成功，覆盖 `refs/heads/backup/iaas_main-*`，并包含 `deletion` 和 `non_fast_forward` rules。若 GitHub Enterprise 不支持 rulesets 或权限不足，记录 exact error，并改由仓库管理员在 GitHub UI 添加同名规则。

## C3: 从 fork 创建新的集成分支

**When:** C2 远端备份验证通过后。

**Working directory:** 同 C1。

```bash
set -euo pipefail

INTEGRATION_BRANCH="codex/vllm-dsv4-fork-base-byteiaas-build"
FORK_REF="refs/remotes/wangyicong52/dev-dsv4-mooncake-pp-megamoe"

git switch --create "${INTEGRATION_BRANCH}" "${FORK_REF}"
git status --short
git log -1 --oneline
```

**Expected result:** 当前分支为 `codex/vllm-dsv4-fork-base-byteiaas-build`，HEAD 等于 fork base SHA，工作区干净。

## C4: 从旧 `iaas_main` 回拷 ByteIAAS 构建文件

**When:** C3 完成后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

OLD_BASE="origin/iaas_main"

git checkout "${OLD_BASE}" -- \
  .github/workflows/byteiaas-release-dev.yml \
  .github/workflows/byteiaas-release.yml \
  .github/workflows/_byteiaas-build-and-publish-image.yml \
  .github/workflows/_byteiaas-build-wheel.yml \
  scripts/ci/get_byteiaas_image_tag.py \
  docker/byteiaas-openai-devel.Dockerfile

git status --short
```

**Expected result:** 只出现上述 ByteIAAS 构建相关文件的新增或修改；不出现 `vllm/`、`csrc/`、`cmake/` 旧源码逻辑文件。

## C5: 审计 Dockerfile 差异，决定最小手工 edits

**When:** C4 后，修改 `docker/Dockerfile` 前。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

git diff --no-ext-diff --unified=80 origin/iaas_main -- docker/Dockerfile > /tmp/old-iaas-main-dockerfile.diff
git diff --no-ext-diff --unified=80 "refs/remotes/wangyicong52/dev-dsv4-mooncake-pp-megamoe" -- docker/Dockerfile > /tmp/fork-base-dockerfile.diff

rg -n "INSTALL_KV_CONNECTORS|MOONCAKE_WHEEL|vllm-openai-base|VLLM_BUILD_COMMIT|VLLM_IMAGE_TAG|DeepEP|deepgemm|byteiaas" docker/Dockerfile
```

**Expected result:** 确认可保留 fork/current Dockerfile 的 Mooncake/KV connector 处理；只需补 ByteIAAS workflow 依赖的 image metadata 或 openai-devel 构建兼容内容。

## C6: DeepGEMM fork wheel 处理验证

**When:** 修改 Dockerfile 或 workflow 前，用于确认 DeepGEMM fork release 是否可复现。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

git ls-remote --tags https://github.com/wangyicong52/DeepGEMM.git \
  "refs/tags/deep_gemm-2.5.0-1wyc-vllm-mega-moe196439b72" || true

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
git clone --depth 1 --branch deep_gemm-2.5.0-1wyc-vllm-mega-moe196439b72 \
  --recurse-submodules --shallow-submodules \
  https://github.com/wangyicong52/DeepGEMM.git "${tmpdir}/DeepGEMM"
rg -n "transform_weights_for_mega_moe_sm90|transform_weights_for_mega_moe_sm90_fp4" \
  "${tmpdir}/DeepGEMM/deep_gemm" || true

python3 - <<'PY'
from urllib.parse import urlparse
import tempfile
import urllib.request
import zipfile

url = "https://github.com/wangyicong52/DeepGEMM/releases/download/deep_gemm-2.5.0-1wyc-vllm-mega-moe196439b72/deep_gemm-2.5.0-1wyc_vllm_mega_moe196439b72-cp312-cp312-linux_x86_64.whl"
parsed = urlparse(url)
print(parsed.netloc)
print(parsed.path)

with tempfile.NamedTemporaryFile(suffix=".whl") as f:
    with urllib.request.urlopen(url, timeout=60) as r:
        f.write(r.read())
    f.flush()
    with zipfile.ZipFile(f.name) as zf:
        init_py = zf.read("deep_gemm/__init__.py").decode()
        mega_py = zf.read("deep_gemm/mega/__init__.py").decode()
        for symbol in [
            "transform_weights_for_mega_moe_sm90",
            "transform_weights_for_mega_moe_sm90_fp4",
            "fp8_fp4_mega_moe",
        ]:
            assert symbol in init_py or symbol in mega_py, symbol
        print("release wheel exports required MegaMoE symbols")
PY
```

**Expected result:** 确认 tag 是否存在，以及 tag 源码是否包含 `transform_weights_for_mega_moe_sm90_fp4`。如果 tag 源码包含所需符号，优先在 Docker build 中从该 tag 构建；如果 tag 源码缺少该符号但 chart 已出现的 release wheel 包含所需符号，则记录证据并使用本线程已授权的 Helm chart release wheel URL 做 image build-time 安装；不得在部署模板或 Pod 启动时下载/安装 wheel。

## C7: Workflow 和 tag 脚本验证

**When:** ByteIAAS workflow 文件回拷和必要 edits 后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

if command -v actionlint >/dev/null 2>&1; then
  actionlint \
    .github/workflows/byteiaas-release-dev.yml \
    .github/workflows/byteiaas-release.yml \
    .github/workflows/_byteiaas-build-and-publish-image.yml \
    .github/workflows/_byteiaas-build-wheel.yml
else
  uv run --no-project --with pyyaml python - <<'PY'
from pathlib import Path
import yaml

for path in [
    ".github/workflows/byteiaas-release-dev.yml",
    ".github/workflows/byteiaas-release.yml",
    ".github/workflows/_byteiaas-build-and-publish-image.yml",
    ".github/workflows/_byteiaas-build-wheel.yml",
]:
    yaml.safe_load(Path(path).read_text())
    print(f"parsed {path}")
PY
fi

uv run --no-project python scripts/ci/get_byteiaas_image_tag.py \
  --mode dev \
  --image-flavor openai \
  --cuda-suffix cu130

uv run --no-project python scripts/ci/get_byteiaas_image_tag.py \
  --mode dev \
  --image-flavor openai-devel \
  --cuda-suffix cu130

rg -n "onion-ai-data|oniond|GPG-KEY-system|volctools.list|signed-by=/etc/apt/trusted.gpg.d/volc-extra-tools.gpg|command -v oniond" \
  docker/Dockerfile docker/byteiaas-openai-devel.Dockerfile .github/workflows
```

**Expected result:** workflow lint 或 parse 成功；tag script 输出合法 dev tag；Dockerfile/build workflow 中存在按 `onion-ai-data` skill 安装或确认 `oniond` 的逻辑，且该逻辑属于镜像构建阶段而非 Pod runtime。

## C8: 本地或 build-node Docker build

**When:** Dockerfile/workflow edits 后，有 Docker buildx 环境时运行。只构建镜像，不做镜像内 import/CLI smoke。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

docker buildx build \
  --target vllm-openai \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION=13.0.2 \
  --build-arg FINAL_BASE_IMAGE="nvidia/cuda:13.0.2-base-ubuntu22.04" \
  --build-arg INSTALL_KV_CONNECTORS=true \
  --build-arg RUN_WHEEL_CHECK=false \
  --load \
  -t local/vllm:dsv4-fork-base-byteiaas-build \
  .
```

**Expected result:** image build 成功。若本机无 Docker/GPU/buildx，记录环境限制并转 C9 的 ByteIAAS workflow build。

## C9: ByteIAAS workflow 构建并发布镜像

**When:** C7 通过后。必须在更新远端 `iaas_main` 前用 integration branch 的 `checkout_ref` 构建，产出的 image tag/digest 供 C13-C22 部署与 benchmark 使用。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

BUILD_REF="$(git branch --show-current)"
RUN_START="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "BYTEIAAS_WORKFLOW_DISPATCH_START=${RUN_START}"

git push origin "HEAD:refs/heads/${BUILD_REF}"

gh workflow run byteiaas-release-dev.yml \
  --repo bytedance-iaas/vllm \
  --ref "${BUILD_REF}" \
  -f checkout_ref="${BUILD_REF}" \
  -f vllm_version=""

sleep 20
RUN_ID="$(gh run list \
  --repo bytedance-iaas/vllm \
  --workflow byteiaas-release-dev.yml \
  --branch "${BUILD_REF}" \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')"

echo "BYTEIAAS_WORKFLOW_RUN_ID=${RUN_ID}"
gh run watch "${RUN_ID}" --repo bytedance-iaas/vllm --exit-status
gh run view "${RUN_ID}" --repo bytedance-iaas/vllm --log > "/tmp/byteiaas-vllm-${RUN_ID}.log"

RUN_END="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "BYTEIAAS_WORKFLOW_DISPATCH_END=${RUN_END}"

rg "Published openai image:|Published openai-devel image:" "/tmp/byteiaas-vllm-${RUN_ID}.log"
```

**Expected result:** workflow 成功；日志中出现 `Published openai image:` 和 `Published openai-devel image:`，将 image tag/digest 记录到进展日志。不要在构建流程中补充镜像内 import/CLI smoke。

## C9A: 撤销 runtime fallback 路线并回到 build-only 修复

**When:** 用户明确不接受 `vllm/_custom_ops.py` 这类 runtime Python fallback，且要求 vLLM 源码修改只包含构建过程中遇到的问题时。必须在再次构建、部署、benchmark 或更新 `iaas_main` 前执行。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

INVALID_RUN_ID="28414886195"
INVALID_COMMIT="51b135cef854e6d72cb704068644c52d047706e5"

gh run view "${INVALID_RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,url \
  --jq '{status, conclusion, headSha, url}'

# 如果该 run 仍 queued/in_progress，取消以避免发布包含 runtime fallback 的镜像。
RUN_STATUS="$(gh run view "${INVALID_RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status \
  --jq '.status')"
if [ "${RUN_STATUS}" = "queued" ] || [ "${RUN_STATUS}" = "in_progress" ]; then
  gh run cancel "${INVALID_RUN_ID}" --repo bytedance-iaas/vllm
fi

# 只回滚 runtime fallback 文件，保留计划文档中对该失败路线的历史记录和后续约束。
# 不使用整提交 revert，因为该提交同时包含需要保留的计划/进展日志内容。
git restore --source="${INVALID_COMMIT}^" -- vllm/_custom_ops.py

git diff -- vllm/_custom_ops.py
uv run --no-project python -m py_compile vllm/_custom_ops.py

# 确认不再包含 `_moe_C` 到 stable libtorch extension 的 runtime fallback。
if rg -n "_moe_C_stable_libtorch" vllm/_custom_ops.py; then
  echo "vllm/_custom_ops.py still contains runtime fallback" >&2
  exit 1
fi
if awk '/def topk_hash_softplus_sqrt/,/^def / { print }' vllm/_custom_ops.py | \
  rg -n "try:|except ImportError|_moe_C_stable_libtorch"; then
  echo "topk_hash_softplus_sqrt still contains runtime fallback" >&2
  exit 1
fi

git add vllm/_custom_ops.py docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build*.md
git commit -s -m "Constrain DSV4 migration to build-only source changes"
```

**Expected result:** run `28414886195` 不再继续发布 runtime fallback 镜像；当前 HEAD 仅撤销 `vllm/_custom_ops.py` fallback 修改并保留计划日志。后续若继续处理 `No module named 'vllm._moe_C'`，只能从 build/package artifact 层修复，例如 `setup.py`、CMake、wheel extraction/package data 或 Dockerfile wheel 安装路径；不得再修改 `vllm/` runtime Python 逻辑。

## C9B: 撤销错误的 build-side `_moe_C` rename 路线

**When:** 上游对比证明 `_moe_C_stable_libtorch` build artifact 与 `torch.ops._moe_C` namespace 是 upstream main 的自洽设计，而当前分支的 build-side rename 会偏离 upstream main 时。必须在再次构建、部署、benchmark 或更新 `iaas_main` 前执行。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

BAD_COMMIT="4fcea785fd66874046f9b828eb2fad7fbd527a63"
BAD_RUN_ID="28418542564"

RUN_STATUS="$(gh run view "${BAD_RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status \
  --jq '.status')"
if [ "${RUN_STATUS}" = "queued" ] || [ "${RUN_STATUS}" = "in_progress" ]; then
  gh run cancel "${BAD_RUN_ID}" --repo bytedance-iaas/vllm
fi
gh run view "${BAD_RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,url \
  --jq '{status, conclusion, headSha, url}'

git restore --source="${BAD_COMMIT}^" -- \
  CMakeLists.txt \
  setup.py \
  csrc/libtorch_stable/moe/torch_bindings.cpp

uv run --no-project python -m py_compile setup.py
git diff --check
```

**Expected result:** run `28418542564` 不再发布 build-side rename 镜像；`CMakeLists.txt`、`setup.py` 和 `csrc/libtorch_stable/moe/torch_bindings.cpp` 恢复 upstream main/fork baseline 的 `_moe_C_stable_libtorch` build artifact 命名。后续若要解决 `topk_hash_softplus_sqrt` 的 `import vllm._moe_C` 失败，必须先确认是否允许按 upstream main 对齐 runtime hard import，而不是再做 fallback 或 build-side rename。

## C9C: 按 upstream main 对齐 `topk_hash_softplus_sqrt`

**When:** 用户确认允许删除 `wangyicong52` fork 提交 `f7c4c621d` 引入的 `import vllm._moe_C` hard import，并明确该改动是 upstream main 对齐，不是 runtime fallback。必须在再次构建、部署、benchmark 或更新 `iaas_main` 前执行。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

uv run --no-project python -m py_compile vllm/_custom_ops.py

# `topk_hash_softplus_sqrt` 不应再 hard import `vllm._moe_C`，
# 也不得 fallback 到 `_moe_C_stable_libtorch`。
if awk '/def topk_hash_softplus_sqrt/,/^def / { print }' vllm/_custom_ops.py | \
  rg -n "import vllm\\._moe_C|_moe_C_stable_libtorch|try:|except ImportError"; then
  echo "topk_hash_softplus_sqrt still contains forbidden import/fallback" >&2
  exit 1
fi

# Build artifact 命名保持 upstream main/fork baseline 的 stable-libtorch 路线。
rg -n "vllm\\._moe_C_stable_libtorch|_moe_C_stable_libtorch" \
  setup.py CMakeLists.txt csrc/libtorch_stable/moe/torch_bindings.cpp
git diff --check
```

**Expected result:** `topk_hash_softplus_sqrt` 只调用 `torch.ops._moe_C.topk_softplus_sqrt`，不再 import `vllm._moe_C`，也不包含 fallback；build/package artifact 仍为 `vllm._moe_C_stable_libtorch`。该状态与 upstream main 的 `_moe_C_stable_libtorch` Python module + `torch.ops._moe_C` namespace 设计一致。

## C9D: 检查已成功构建的 openai-devel 镜像

**When:** 用户已在同一 integration branch 上完成 ByteIAAS workflow 成功构建，且希望复用现有镜像继续部署验证而不是重新构建。本次固定 run id 为 `28442949331`，固定候选镜像为 `iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130`。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

RUN_ID="28442949331"
EXPECTED_SHA="7186cf328963d12daabe8ee47087a29111c0cb75"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302005-openai-devel-cu130"

test "$(git rev-parse HEAD)" = "${EXPECTED_SHA}"
git merge-base --is-ancestor d3f23315c "${EXPECTED_SHA}"

gh run view "${RUN_ID}" \
  --repo bytedance-iaas/vllm \
  --json status,conclusion,headSha,url,createdAt,updatedAt,jobs \
  --jq '{status, conclusion, headSha, url, createdAt, updatedAt, jobs: [.jobs[] | {name, status, conclusion, url}]}'

docker buildx imagetools inspect "${IMAGE}"

# 本机 Docker/containerd socket 不可访问时，不把镜像内 import/CLI smoke 作为本步骤要求；
# runtime 内容由 C13-C16 的 dev-cluster 部署和真实 router request 验证。
docker info >/tmp/vllm-docker-info.txt 2>&1 || true
ctr version >/tmp/vllm-ctr-version.txt 2>&1 || true
cat /tmp/vllm-docker-info.txt
cat /tmp/vllm-ctr-version.txt
```

**Expected result:** workflow `28442949331` 为 `completed/success`，headSha 等于 `7186cf328963d12daabe8ee47087a29111c0cb75`；候选镜像 registry manifest 可读，包含 `linux/amd64` manifest。若本机 Docker/containerd socket 因权限不可访问，记录该限制即可，不重新构建，不补充构建流程 smoke；下一步使用该 `openai-devel` image 执行 C13-C16。

## C10: Final gate 后更新远端 `iaas_main`

**When:** C9 image build/publish 成功，C13 render 通过，C16 real router smoke 通过，C20/C21 measured runs 完成或对应 blocker 已在进展日志中被明确接受，并且 C22 summary 写入 artifacts 后。可接受 blocker 仅限外部环境或资源问题，例如 GPU permit 长时间排队、`dev-cluster` 临时资源不足、CR/image pull 临时失败、Onion 模型源临时不可用；render/config、镜像缺依赖、Onion init、模型完整性、vLLM 启动、router real request、KV transfer、DeepGEMM/DeepEP/Mooncake import 或 runtime 错误都必须阻止本步骤。benchmark 跑通但性能未达阈值也必须阻止本步骤；本计划性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，BS512/1.5k evalscope overall output token throughput >= 14000 tokens/s。远端备份分支和保护规则必须仍存在。用户已在 2026-06-29 本线程授权本步骤；只要目标仓库、源 SHA、备份分支、验证门槛和更新方式符合本计划，不需要再次停下来审批。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

BACKUP_BRANCH="backup/iaas_main-20260629"
EXPECTED_OLD_SHA="$(git rev-parse origin/iaas_main)"
NEW_SHA="$(git rev-parse HEAD)"

git ls-remote origin "refs/heads/${BACKUP_BRANCH}"
gh api /repos/bytedance-iaas/vllm/rulesets \
  --jq '.[] | select(.name=="Protect backup iaas_main branches") | {id,name,enforcement}'

git push --force-with-lease=refs/heads/iaas_main:"${EXPECTED_OLD_SHA}" \
  origin \
  "HEAD:refs/heads/iaas_main"

git fetch origin iaas_main
git rev-parse origin/iaas_main
echo "NEW_SHA=${NEW_SHA}"
```

**Expected result:** `origin/iaas_main` 指向 `NEW_SHA`；远端备份分支仍指向旧 SHA。

## C11: Branch protection fallback

**When:** C10 被 branch protection 拒绝。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

INTEGRATION_BRANCH="codex/vllm-dsv4-fork-base-byteiaas-build"
git push origin "HEAD:refs/heads/${INTEGRATION_BRANCH}"

gh pr create \
  --repo bytedance-iaas/vllm \
  --base iaas_main \
  --head "${INTEGRATION_BRANCH}" \
  --title "Replace iaas_main with DSV4 fork base plus ByteIAAS build support" \
  --body-file /tmp/vllm-dsv4-fork-base-pr-body.md
```

**Expected result:** integration branch exists on GitHub；if GitHub cannot create a normal PR due unrelated history or policy, record the exact error and escalate for repository admin branch migration.

## C12: Final status checks

**When:** Before final report or PR.

**Working directory:** fork-base integration branch or final `iaas_main` checkout.

```bash
set -euo pipefail

git diff --check
git status --short
git diff --name-status "refs/remotes/wangyicong52/dev-dsv4-mooncake-pp-megamoe"...HEAD
git ls-remote origin "refs/heads/backup/iaas_main-20260629"
gh api /repos/bytedance-iaas/vllm/rulesets \
  --jq '.[] | select(.name=="Protect backup iaas_main branches") | {id,name,enforcement}'
```

**Expected result:** no whitespace errors；only planned ByteIAAS build/deployment/doc files differ from fork base；remote backup exists.

## C13: Render 无 runtime hotfix/install 的部署模板

**When:** `examples/deployment/deepseek-v4-flash-pd/` 文件创建完成，并且 C9 已产出 image tag/digest 后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
IMAGE="${IMAGE:?set IMAGE to the ByteIAAS image tag or digest from C9}"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
HOST_NETWORK="${HOST_NETWORK:-true}"
GLOBAL_GPU_COUNT="${GLOBAL_GPU_COUNT:-8}"
STORM_REPLICAS="${STORM_REPLICAS:-1}"
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
DECODE_REPLICAS="${DECODE_REPLICAS:-1}"
ROUTER_REPLICAS="${ROUTER_REPLICAS:-1}"
PREFILL_NODE="${PREFILL_NODE:?set PREFILL_NODE to the 8-GPU prefill node name}"
DECODE_NODE="${DECODE_NODE:?set DECODE_NODE to the 8-GPU decode node name}"
ROUTER_NODE="${ROUTER_NODE:-${DECODE_NODE}}"
ONION_ENABLED="${ONION_ENABLED:-true}"
ONION_MODEL="${ONION_MODEL:-DeepSeek-V4-Flash}"
ONION_DIR="${ONION_DIR:-/data01}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data01}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-512}"
WORKSPACE_ENV_SESSION_ID="${WORKSPACE_ENV_SESSION_ID:-render-only}"

if [ "${GLOBAL_GPU_COUNT}" != "8" ]; then
  echo "GLOBAL_GPU_COUNT must remain 8 for the servingkit-equivalent 1P1D shape" >&2
  exit 1
fi
if [ "${PREFILL_NODE}" = "${DECODE_NODE}" ]; then
  echo "PREFILL_NODE and DECODE_NODE must be different because each role requests 8 GPUs" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"

helm template "${RELEASE}" examples/deployment/deepseek-v4-flash-pd \
  --namespace "${NAMESPACE}" \
  --set global.image="${IMAGE}" \
  --set global.gpuCount="${GLOBAL_GPU_COUNT}" \
  --set stormService.replicas="${STORM_REPLICAS}" \
  --set prefill.replicas="${PREFILL_REPLICAS}" \
  --set decode.replicas="${DECODE_REPLICAS}" \
  --set router.replicas="${ROUTER_REPLICAS}" \
  --set prefill.hostNetwork="${HOST_NETWORK}" \
  --set decode.hostNetwork="${HOST_NETWORK}" \
  --set router.hostNetwork="${HOST_NETWORK}" \
  --set prefill.nodeAffinity.enabled=true \
  --set prefill.nodeAffinity.key=kubernetes.io/hostname \
  --set prefill.nodeAffinity.operator=In \
  --set "prefill.nodeAffinity.values[0]=${PREFILL_NODE}" \
  --set decode.nodeAffinity.enabled=true \
  --set decode.nodeAffinity.key=kubernetes.io/hostname \
  --set decode.nodeAffinity.operator=In \
  --set "decode.nodeAffinity.values[0]=${DECODE_NODE}" \
  --set router.nodeAffinity.enabled=true \
  --set router.nodeAffinity.key=kubernetes.io/hostname \
  --set router.nodeAffinity.operator=In \
  --set "router.nodeAffinity.values[0]=${ROUTER_NODE}" \
  --set onion.enabled="${ONION_ENABLED}" \
  --set onion.model="${ONION_MODEL}" \
  --set onion.dir="${ONION_DIR}" \
  --set model.name="${MODEL_NAME}" \
  --set model.basePath="${MODEL_BASE_PATH}" \
  --set decode.args.maxNumSeqs="${DECODE_MAX_NUM_SEQS}" \
  --set workspaceEnv.sessionId="${WORKSPACE_ENV_SESSION_ID}" \
  --set workspaceEnv.owner=codex \
  --set workspaceEnv.purpose=vllm-dsv4-flash-pd \
  > "${ARTIFACT_DIR}/rendered-${RELEASE}.yaml"

forbidden='runtimePatch|git clone|pip install|apt(-get)?[[:space:]]+(update|install)|install_deepgemm|ensure_pip_package|wheelURL|wheelPath|/tmp/vllm-runtime-patch|vllm-router.*pip'
if rg -n "${forbidden}" "${ARTIFACT_DIR}/rendered-${RELEASE}.yaml"; then
  echo "render contains forbidden runtime hotfix/install pattern" >&2
  exit 1
fi

rg -n "kind: StormService|replicas: 1|hostNetwork: true|oniond download model|${ONION_MODEL}|${ONION_DIR}|image: ${IMAGE}|vllm serve|vllm-router|MooncakeConnector|kv_producer|kv_consumer|deep_gemm_mega_moe|deepep_low_latency|--tensor-parallel-size|--pipeline-parallel-size|--cp-kv-cache-interleave-size|--speculative-config|--prefill|--decode|--intra-node-data-parallel-size" \
  "${ARTIFACT_DIR}/rendered-${RELEASE}.yaml"
rg -n "nvidia.com/gpu: 8|kubernetes.io/hostname|${PREFILL_NODE}|${DECODE_NODE}" \
  "${ARTIFACT_DIR}/rendered-${RELEASE}.yaml"

if helm template "${RELEASE}" examples/deployment/deepseek-v4-flash-pd \
  --namespace "${NAMESPACE}" \
  --set global.image="${IMAGE}" \
  --set global.gpuCount="${GLOBAL_GPU_COUNT}" \
  --set "prefill.nodeAffinity.values[0]=same-node" \
  --set "decode.nodeAffinity.values[0]=same-node" \
  > "${ARTIFACT_DIR}/rendered-invalid-same-node.yaml" \
  2> "${ARTIFACT_DIR}/rendered-invalid-same-node.err"; then
  echo "expected Helm validation to reject same prefill/decode node" >&2
  exit 1
fi
rg -n "prefill and decode nodeAffinity values must be disjoint" \
  "${ARTIFACT_DIR}/rendered-invalid-same-node.err"
```

**Expected result:** render 成功；prefill/decode workload 继续使用 `kind: StormService`；replica shape 为 `1P1D`，即 `stormService.replicas=1`、`prefill.replicas=1`、`decode.replicas=1`、`router.replicas=1`；`global.gpuCount=8` 渲染为 prefill 和 decode 的 `nvidia.com/gpu: 8` request/limit；`PREFILL_NODE` 与 `DECODE_NODE` 必须非空且不同，并分别出现在 prefill/decode required nodeAffinity；router 默认使用 `ROUTER_NODE=${DECODE_NODE}`；prefill、decode、router 默认保留 `hostNetwork: true`；serving container 和 Onion model prepare initContainer 都使用同一个新镜像 `global.image`；Onion 模型准备路径存在；prefill/decode KV roles 分别为 `kv_producer`/`kv_consumer`；prefill 渲染 servingkit 对齐的 `dataParallelSize=1`、`tensorParallelSize=4`、`pipelineParallelSize=2`；decode 渲染 servingkit 对齐的 `dataParallelSize=8`、`port=8001`、`cpKvCacheInterleaveSize=256`、`deep_gemm_mega_moe`、MTP speculative config；router 默认关闭 service discovery 并渲染静态 `--prefill http://${PREFILL_NODE}:8000 8998` 与 `--decode http://${DECODE_NODE}:8001`，`intra-node-data-parallel-size=1`；`--max-model-len` 不应出现在 rendered command 中；只有 `decode.args.maxNumSeqs=512` 是为 BS512/1.5k benchmark 做的显式执行期覆盖；没有 runtime hotfix、`git clone`、`pip install`、`apt install`、wheel download/install 或 runtime router install；同一节点负例必须被 Helm validation 拒绝。

## C14: dev-cluster preflight, registry, and GPU permit

**When:** C13 通过后，任何 Kubernetes GPU workload 创建之前。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
SKILL_DIR="/data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
GPU_TOTAL="16"
GLOBAL_GPU_COUNT="${GLOBAL_GPU_COUNT:-8}"
PREFILL_NODE="${PREFILL_NODE:?set PREFILL_NODE to the 8-GPU prefill node name}"
DECODE_NODE="${DECODE_NODE:?set DECODE_NODE to the 8-GPU decode node name}"
ROUTER_NODE="${ROUTER_NODE:-${DECODE_NODE}}"
SESSION_ID="codex-vllm-dsv4-flash-pd-$(date +%Y%m%d-%H%M%S)-$$"
THREAD_ID="${CODEX_THREAD_ID:-manual}"
PURPOSE="vllm-dsv4-flash-pd-build-deploy-benchmark"
CLEANUP_COMMAND="eval \"\$(${ENV_ROOT}/bin/envctl use ${ENVIRONMENT})\"; helm uninstall ${RELEASE} -n ${NAMESPACE} || true; kubectl delete namespace ${NAMESPACE} --ignore-not-found"

if [ "${GLOBAL_GPU_COUNT}" != "8" ]; then
  echo "GLOBAL_GPU_COUNT must remain 8 for this servingkit-equivalent deployment" >&2
  exit 1
fi
if [ "${PREFILL_NODE}" = "${DECODE_NODE}" ]; then
  echo "PREFILL_NODE and DECODE_NODE must be different 8-GPU nodes" >&2
  exit 1
fi

mkdir -p "artifacts/2026-06-29-vllm-dsv4-flash-pd"

"${ENV_ROOT}/bin/envctl" info "${ENVIRONMENT}"
"${ENV_ROOT}/bin/envctl" validate "${ENVIRONMENT}"
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get crd stormservices.orchestration.aibrix.ai
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu,UNSCHED:.spec.unschedulable
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get pods -A -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase,GPU:.spec.containers[*].resources.requests.nvidia\\.com/gpu,NODE:.spec.nodeName | sed -n '1,200p'

for selected_node in "${PREFILL_NODE}" "${DECODE_NODE}"; do
  alloc_gpu="$("${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get node "${selected_node}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
  unsched="$("${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get node "${selected_node}" -o jsonpath='{.spec.unschedulable}')"
  echo "NODE=${selected_node} ALLOCATABLE_GPU=${alloc_gpu:-0} UNSCHEDULABLE=${unsched:-false}"
  if [ "${alloc_gpu:-0}" -lt 8 ]; then
    echo "selected node ${selected_node} has fewer than 8 allocatable GPUs" >&2
    exit 1
  fi
  if [ "${unsched:-false}" = "true" ]; then
    echo "selected node ${selected_node} is unschedulable" >&2
    exit 1
  fi
done

python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" init
python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  session-start \
  --session-id "${SESSION_ID}" \
  --thread-id "${THREAD_ID}" \
  --owner codex \
  --task "${PURPOSE}" \
  --cwd "$PWD"

python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  permit-acquire \
  --session-id "${SESSION_ID}" \
  --thread-id "${THREAD_ID}" \
  --environment "${ENVIRONMENT}" \
  --namespace "${NAMESPACE}" \
  --gpu "${GPU_TOTAL}" \
  --purpose "${PURPOSE}" \
  --release-condition "release after benchmark artifacts and cleanup" \
  --cleanup-command "${CLEANUP_COMMAND}" \
  --wait-seconds 180 \
  | tee "artifacts/2026-06-29-vllm-dsv4-flash-pd/gpu-permit.json"
```

**Expected result:** `envctl validate dev-cluster` 通过；`PREFILL_NODE` 和 `DECODE_NODE` 是两个不同节点，且每个节点 allocatable `nvidia.com/gpu` 至少为 8；GPU permit 返回 `granted` 或 `running` 后才允许继续。若返回 `queued`、`denied`、`blocked` 或其他状态，不创建 Kubernetes GPU 资源。

## C15: dev-cluster Helm deploy

**When:** C14 GPU permit granted/running 后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
IMAGE="${IMAGE:?set IMAGE to the ByteIAAS image tag or digest from C9}"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
HOST_NETWORK="${HOST_NETWORK:-true}"
GLOBAL_GPU_COUNT="${GLOBAL_GPU_COUNT:-8}"
STORM_REPLICAS="${STORM_REPLICAS:-1}"
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
DECODE_REPLICAS="${DECODE_REPLICAS:-1}"
ROUTER_REPLICAS="${ROUTER_REPLICAS:-1}"
PREFILL_NODE="${PREFILL_NODE:?set PREFILL_NODE to the 8-GPU prefill node name}"
DECODE_NODE="${DECODE_NODE:?set DECODE_NODE to the 8-GPU decode node name}"
ROUTER_NODE="${ROUTER_NODE:-${DECODE_NODE}}"
ONION_ENABLED="${ONION_ENABLED:-true}"
ONION_MODEL="${ONION_MODEL:-DeepSeek-V4-Flash}"
ONION_DIR="${ONION_DIR:-/data01}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data01}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-512}"
WORKSPACE_ENV_SESSION_ID="${WORKSPACE_ENV_SESSION_ID:?set WORKSPACE_ENV_SESSION_ID to the granted workspace-env session id from C14}"

if [ "${GLOBAL_GPU_COUNT}" != "8" ]; then
  echo "GLOBAL_GPU_COUNT must remain 8 for the servingkit-equivalent 1P1D shape" >&2
  exit 1
fi
if [ "${PREFILL_NODE}" = "${DECODE_NODE}" ]; then
  echo "PREFILL_NODE and DECODE_NODE must be different because each role requests 8 GPUs" >&2
  exit 1
fi

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

helm status "${RELEASE}" -n "${NAMESPACE}" >/tmp/${RELEASE}-helm-status.txt 2>&1 && {
  echo "Helm release ${RELEASE} already exists; inspect owner before upgrading" >&2
  cat /tmp/${RELEASE}-helm-status.txt >&2
  exit 1
} || true

helm upgrade --install "${RELEASE}" examples/deployment/deepseek-v4-flash-pd \
  --namespace "${NAMESPACE}" \
  --set global.image="${IMAGE}" \
  --set global.gpuCount="${GLOBAL_GPU_COUNT}" \
  --set stormService.replicas="${STORM_REPLICAS}" \
  --set prefill.replicas="${PREFILL_REPLICAS}" \
  --set decode.replicas="${DECODE_REPLICAS}" \
  --set router.replicas="${ROUTER_REPLICAS}" \
  --set prefill.hostNetwork="${HOST_NETWORK}" \
  --set decode.hostNetwork="${HOST_NETWORK}" \
  --set router.hostNetwork="${HOST_NETWORK}" \
  --set prefill.nodeAffinity.enabled=true \
  --set prefill.nodeAffinity.key=kubernetes.io/hostname \
  --set prefill.nodeAffinity.operator=In \
  --set "prefill.nodeAffinity.values[0]=${PREFILL_NODE}" \
  --set decode.nodeAffinity.enabled=true \
  --set decode.nodeAffinity.key=kubernetes.io/hostname \
  --set decode.nodeAffinity.operator=In \
  --set "decode.nodeAffinity.values[0]=${DECODE_NODE}" \
  --set router.nodeAffinity.enabled=true \
  --set router.nodeAffinity.key=kubernetes.io/hostname \
  --set router.nodeAffinity.operator=In \
  --set "router.nodeAffinity.values[0]=${ROUTER_NODE}" \
  --set onion.enabled="${ONION_ENABLED}" \
  --set onion.model="${ONION_MODEL}" \
  --set onion.dir="${ONION_DIR}" \
  --set model.name="${MODEL_NAME}" \
  --set model.basePath="${MODEL_BASE_PATH}" \
  --set decode.args.maxNumSeqs="${DECODE_MAX_NUM_SEQS}" \
  --set workspaceEnv.sessionId="${WORKSPACE_ENV_SESSION_ID}" \
  --set workspaceEnv.owner=codex \
  --set workspaceEnv.purpose=vllm-dsv4-flash-pd \
  --wait \
  --timeout 60m \
  2>&1 | tee "${ARTIFACT_DIR}/helm-upgrade-${RELEASE}.log"

kubectl get all -n "${NAMESPACE}" -o wide | tee "${ARTIFACT_DIR}/kubectl-get-all-after-deploy.txt"
```

**Expected result:** Helm release 安装成功，`1P1D` prefill/decode/router workloads 开始创建；prefill 和 decode 分别带不同节点的 required nodeAffinity，并各自请求 8 张 GPU；P/D/router 运行参数与 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 保持语义一致，只有 image、Onion 模型准备、节点参数化、删除 runtime install/hotfix 和 `decode.args.maxNumSeqs=512` 是本计划允许的显式差异；`--max-model-len` 不应出现在 rendered command 中；router 默认跟随 decode 节点。若 release 已存在且不是本任务创建，停止，不接管未知 owner。

## C16: Deployment readiness and real router smoke

**When:** C15 完成后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
MODEL_PATH="${MODEL_PATH:-/data01/DeepSeek-V4-Flash}"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl wait -n "${NAMESPACE}" --for=condition=Ready pod \
  -l app.kubernetes.io/instance="${RELEASE}" \
  --timeout=60m

kubectl get pods -n "${NAMESPACE}" -o wide | tee "${ARTIFACT_DIR}/pods-ready.txt"
kubectl get svc -n "${NAMESPACE}" -o wide | tee "${ARTIFACT_DIR}/services.txt"

for pod in $(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/instance="${RELEASE}" -o name); do
  kubectl describe -n "${NAMESPACE}" "${pod}" > "${ARTIFACT_DIR}/$(basename "${pod}")-describe.txt"
  kubectl logs -n "${NAMESPACE}" "${pod}" --all-containers --tail=300 > "${ARTIFACT_DIR}/$(basename "${pod}")-logs-tail.txt" || true
  kubectl get -n "${NAMESPACE}" "${pod}" -o jsonpath='{range .status.initContainerStatuses[*]}{.name}{" ready="}{.ready}{" exit="}{.state.terminated.exitCode}{" reason="}{.state.terminated.reason}{"\n"}{end}' \
    > "${ARTIFACT_DIR}/$(basename "${pod}")-init-status.txt" || true
  for init_container in onion-model-prepare init-model; do
    kubectl logs -n "${NAMESPACE}" "${pod}" -c "${init_container}" --tail=300 \
      > "${ARTIFACT_DIR}/$(basename "${pod}")-${init_container}.log" 2>&1 || true
  done
  kubectl exec -n "${NAMESPACE}" "${pod}" -- sh -lc 'tr "\0" " " </proc/1/cmdline; echo; env | sort | grep -E "VLLM|MOONCAKE|DEEP|NCCL|CUDA|MODEL|ROUTER" || true' \
    > "${ARTIFACT_DIR}/$(basename "${pod}")-argv-env.txt" || true
  case "$(basename "${pod}")" in
    *router*) ;;
    *)
      kubectl exec -n "${NAMESPACE}" "${pod}" -- sh -lc "test -f '${MODEL_PATH}/config.json' && (test -f '${MODEL_PATH}/tokenizer.json' || test -f '${MODEL_PATH}/tokenizer.model') && (test -f '${MODEL_PATH}/model.safetensors.index.json' || find '${MODEL_PATH}' -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .) && find '${MODEL_PATH}' -maxdepth 1 \\( -name 'config.json' -o -name 'tokenizer*' -o -name '*.safetensors' -o -name 'model.safetensors.index.json' \\) | sort | sed -n '1,80p'" \
        > "${ARTIFACT_DIR}/$(basename "${pod}")-model-files.txt"
      ;;
  esac
done

("${ENV_ROOT}/bin/envctl" port-forward "${ENVIRONMENT}" "${NAMESPACE}" "svc/${RELEASE}-router" 30000:30000 \
  > "${ARTIFACT_DIR}/router-port-forward.log" 2>&1 & echo $! > "${ARTIFACT_DIR}/router-port-forward.pid")
sleep 5

curl -sf http://127.0.0.1:30000/v1/models | tee "${ARTIFACT_DIR}/router-models.json"
curl -sf http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"hello","max_tokens":1,"temperature":0}' \
  | tee "${ARTIFACT_DIR}/router-completion-smoke.json"
```

**Expected result:** Onion model prepare init container completed successfully or logged an idempotent existing-model skip；model directory contains `config.json`, tokenizer file, and safetensors index/shards；Pods Ready through bounded wait；actual argv/env show no runtime hotfix/install；router `/v1/models` and `/v1/completions` succeed with non-empty output. This smoke is live service readiness for benchmark, not a build-flow image import/CLI smoke.

## C17: Prepare evalscope environment

**When:** C16 real router smoke passes.

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVAL_VENV="$PWD/.venv-evalscope"

uv venv --python 3.12 "${EVAL_VENV}"
uv pip install --python "${EVAL_VENV}/bin/python" evalscope
"${EVAL_VENV}/bin/evalscope" --version | tee "${ARTIFACT_DIR}/evalscope-version.txt"
```

**Expected result:** evalscope 可执行。若 `uv`、Python、network、package index 或权限阻塞安装，停止并记录 blocker，不切换自定义 harness。

## C18: Skip Prometheus and capture lightweight pod metrics

**When:** C16 real router smoke passes, C20/C21 measured run 之前。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
METRICS_DIR="${ARTIFACT_DIR}/metrics-lightweight"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
mkdir -p "${METRICS_DIR}"

cat > "${METRICS_DIR}/prometheus-skipped.md" <<'MD'
# Prometheus skipped

User explicitly requested to skip Prometheus for this benchmark plan revision.
No temporary Prometheus, Grafana, scrape config, PromQL query, or measured-window monitoring stack is deployed.
The benchmark summary must state this limitation and must not claim full service-side monitoring diagnosis.
MD

kubectl get svc -A | rg -i 'prometheus|grafana' | tee "${METRICS_DIR}/existing-monitoring-services.txt" || true
kubectl get pods -A | rg -i 'prometheus|grafana' | tee "${METRICS_DIR}/existing-monitoring-pods.txt" || true

for pod in $(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/instance="${RELEASE}" -o name); do
  kubectl logs -n "${NAMESPACE}" "${pod}" --all-containers --tail=300 \
    > "${METRICS_DIR}/$(basename "${pod}")-logs-tail-before-benchmark.txt" || true
  kubectl exec -n "${NAMESPACE}" "${pod}" -- sh -lc 'curl -sf http://127.0.0.1:${PORT:-8000}/metrics | head -n 40' \
    > "${METRICS_DIR}/$(basename "${pod}")-metrics-head.txt" || true
done
```

**Expected result:** 不部署 Prometheus；写入 `metrics-lightweight/prometheus-skipped.md`，保存现有监控资源列表、pod logs tail 和 pod-local `/metrics` head。后续 summary 必须明确 Prometheus skipped by user；这些轻量证据不能替代 measured-window PromQL。

## C19: Benchmark warmup and cache seed

**When:** C17 evalscope ready and C16 service ready.

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="/data01/DeepSeek-V4-Flash"

"${EVALSCOPE}" perf \
  --parallel 1 \
  --number 1 \
  --model "${MODEL}" \
  --url "${URL}" \
  --api openai \
  --dataset random \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --min-tokens 1 \
  --max-tokens 1 \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs-dir "${ARTIFACT_DIR}/evalscope-warmup" \
  2>&1 | tee "${ARTIFACT_DIR}/evalscope-warmup.log"

"${EVALSCOPE}" perf \
  --parallel 1 \
  --number 1 \
  --model "${MODEL}" \
  --url "${URL}" \
  --api openai \
  --dataset random \
  --prefix-length 65536 \
  --min-prompt-length 0 \
  --max-prompt-length 0 \
  --min-tokens 1 \
  --max-tokens 1 \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs-dir "${ARTIFACT_DIR}/evalscope-cache-seed" \
  2>&1 | tee "${ARTIFACT_DIR}/evalscope-cache-seed.log"
```

**Expected result:** warmup 和 prefix cache seed 均成功；seed 使用 `--seed 42`，供 C21 使用同一 prefix。

## C20: Measured 64k input / 1 output TTFT

**When:** C19 warmup passes.

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="/data01/DeepSeek-V4-Flash"

run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "MEASURED_RUN_START=${run_start}" | tee "${ARTIFACT_DIR}/ttft-64k-1out.timestamps"

"${EVALSCOPE}" perf \
  --parallel 1 \
  --number 1 \
  --model "${MODEL}" \
  --url "${URL}" \
  --api openai \
  --dataset random \
  --prefix-length 0 \
  --min-prompt-length 65536 \
  --max-prompt-length 65536 \
  --min-tokens 1 \
  --max-tokens 1 \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs-dir "${ARTIFACT_DIR}/evalscope-ttft-64k-1out" \
  2>&1 | tee "${ARTIFACT_DIR}/evalscope-ttft-64k-1out.log"
run_status=${PIPESTATUS[0]}

run_end="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "MEASURED_RUN_END=${run_end}"
  echo "MEASURED_RUN_EXIT_CODE=${run_status}"
} | tee -a "${ARTIFACT_DIR}/ttft-64k-1out.timestamps"
exit "${run_status}"
```

**Expected result:** evalscope measured run completes with exit code 0；raw log and outputs are preserved；TTFT metric is reported from streaming request path；64k/1 Avg TTFT 必须小于 10s，否则不得进入 C10 更新远端 `iaas_main`；P50/P95/P99 归档但不作为 gate。

## C21: Measured cache-hit decode BS512 / 1.5k output throughput

**When:** C19 cache seed passes and target remains stable.

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="/data01/DeepSeek-V4-Flash"
DECODE_OUTPUT_TOKENS=1536

run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "MEASURED_RUN_START=${run_start}" | tee "${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout.timestamps"

"${EVALSCOPE}" perf \
  --parallel 512 \
  --number 512 \
  --model "${MODEL}" \
  --url "${URL}" \
  --api openai \
  --dataset random \
  --prefix-length 65536 \
  --min-prompt-length 0 \
  --max-prompt-length 0 \
  --min-tokens "${DECODE_OUTPUT_TOKENS}" \
  --max-tokens "${DECODE_OUTPUT_TOKENS}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs-dir "${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout" \
  2>&1 | tee "${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout.log"
run_status=${PIPESTATUS[0]}

run_end="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "MEASURED_RUN_END=${run_end}"
  echo "MEASURED_RUN_EXIT_CODE=${run_status}"
} | tee -a "${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout.timestamps"
exit "${run_status}"
```

**Expected result:** 512 并发、512 总请求以同一 64k prefix cache 命中场景完成；每个请求固定 1536 output tokens，并通过 `--extra-args '{"temperature":0,"ignore_eos":true}'` 避免提前 EOS；平均输出 token 数必须达到 `1536 * 95% = 1459.2` 以上才可作为 throughput capacity 结果；BS512/1.5k evalscope overall output token throughput 必须达到 14000 tokens/s 以上，否则不得进入 C10 更新远端 `iaas_main`；该 throughput 是 `1P1D` router-path 输出吞吐，全部 output 来自单个 decode 节点，不是多 decode 聚合；P50/P95/P99 归档但不作为 gate；保存 raw output、timestamps 和 summary。

## C22: Benchmark artifact summary and cleanup

**When:** C20/C21 完成或被明确判定 invalid/blocked 后。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
SKILL_DIR="/data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl get pods -n "${NAMESPACE}" -o wide > "${ARTIFACT_DIR}/final-pods.txt" || true
kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp > "${ARTIFACT_DIR}/final-events.txt" || true

cat > "${ARTIFACT_DIR}/summary.md" <<'MD'
# vLLM DSV4 Flash P/D Benchmark Summary

## Target

- Environment: dev-cluster
- Namespace: vllm-dsv4-flash-pd
- Release: dsv4-flash-pd
- Model: deepseek-v4-flash

## Runs

- 64k input / 1 output TTFT: see `evalscope-ttft-64k-1out.log`
- cache-hit decode BS512 / 1.5k output throughput: see `evalscope-decode-bs512-cache-hit-1p5kout.log`
- Prometheus: skipped by user; see `metrics-lightweight/prometheus-skipped.md`
- Lightweight metrics/log evidence: see `metrics-lightweight/`

## Performance Gate

- Gate uses Avg metrics only; P50/P95/P99 are archived but do not block.
- 64k/1 Avg TTFT must be < 10s.
- BS512/1.5k evalscope overall output throughput must be >= 14000 tokens/s.
- Deployment shape is 1P1D, so router-path output throughput is produced by one decode node, not aggregated across multiple decode replicas.
- If either threshold is not met, do not update remote `iaas_main`.

## Artifacts

Raw artifacts are in this directory.
MD

helm uninstall "${RELEASE}" -n "${NAMESPACE}" || true
kubectl delete namespace "${NAMESPACE}" --ignore-not-found

if [ -f "${ARTIFACT_DIR}/router-port-forward.pid" ]; then
  kill "$(cat "${ARTIFACT_DIR}/router-port-forward.pid")" 2>/dev/null || true
fi

PERMIT_ID="$(python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/2026-06-29-vllm-dsv4-flash-pd/gpu-permit.json")
if not path.exists():
    raise SystemExit(0)
data = json.loads(path.read_text())
print(data.get("permit_id", ""))
PY
)"

if [ -n "${PERMIT_ID}" ]; then
  python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
    permit-release --permit-id "${PERMIT_ID}"
fi
```

**Expected result:** summary.md exists；temporary Helm release, benchmark namespace, and router port-forward are cleaned unless intentionally retained and reported；GPU permit is released by permit id from `gpu-permit.json`；summary 明确 Prometheus skipped by user。
