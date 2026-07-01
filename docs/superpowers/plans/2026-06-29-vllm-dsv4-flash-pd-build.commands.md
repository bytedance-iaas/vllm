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

**When:** C9 image build/publish 成功，C13 render 通过，C16 real router smoke 通过，C20 measured TTFT run 和 vLLM bench BS512/2048 measured run 完成，C21W/C23/C27/C28 已解释 evalscope 差异，并且对应 summary/artifacts 已写入后。2026-07-02 用户已接受 `vllm bench serve` 结果并认为总体通过，因此当前发布性能 gate 看 Avg，不看 P50/P95/P99；阈值为 64k/1 Avg TTFT < 10s，`1P1D` router-path `vllm bench serve` BS512/1.5k output throughput >= 14000 tokens/s，BS512 必须使用 `num-prompts=2048` 且 Prometheus 证明实际 running BS 接近目标。可接受 blocker 仅限外部环境或资源问题，例如 GPU permit 长时间排队、`dev-cluster` 临时资源不足、CR/image pull 临时失败、Onion 模型源临时不可用；render/config、镜像缺依赖、Onion init、模型完整性、vLLM 启动、router real request、DeepGEMM/DeepEP/Mooncake import 或 runtime 错误仍必须阻止本步骤。历史 evalscope 波动不再单独阻止本步骤，但必须作为风险写入发布摘要。远端备份分支和保护规则必须仍存在；不得使用 C25/C28 诊断分支直接更新 `iaas_main`。用户已在 2026-06-29 本线程授权远端更新，并在 2026-07-02 授权当前 gate 口径；只要目标仓库、源 SHA、备份分支、验证门槛和更新方式符合本计划，不需要再次停下来审批。

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
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-96}"
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

**Expected result:** render 成功；prefill/decode workload 继续使用 `kind: StormService`；replica shape 为 `1P1D`，即 `stormService.replicas=1`、`prefill.replicas=1`、`decode.replicas=1`、`router.replicas=1`；`global.gpuCount=8` 渲染为 prefill 和 decode 的 `nvidia.com/gpu: 8` request/limit；`PREFILL_NODE` 与 `DECODE_NODE` 必须非空且不同，并分别出现在 prefill/decode required nodeAffinity；router 默认使用 `ROUTER_NODE=${DECODE_NODE}`；prefill、decode、router 默认保留 `hostNetwork: true`；serving container 和 Onion model prepare initContainer 都使用同一个新镜像 `global.image`；Onion 模型准备路径存在；prefill/decode KV roles 分别为 `kv_producer`/`kv_consumer`；prefill 渲染 servingkit 对齐的 `dataParallelSize=1`、`tensorParallelSize=4`、`pipelineParallelSize=2`、`noAsyncScheduling=false`；decode 渲染 servingkit 对齐的 `dataParallelSize=8`、`port=8001`、`cpKvCacheInterleaveSize=256`、`deep_gemm_mega_moe`、`maxNumSeqs=96`、MTP speculative config；router 默认关闭 service discovery 并渲染静态 `--prefill http://${PREFILL_NODE}:8000 8998` 与 `--decode http://${DECODE_NODE}:8001`，`intra-node-data-parallel-size=1`；`--max-model-len` 不应出现在 rendered command 中；没有 runtime hotfix、`git clone`、`pip install`、`apt install`、wheel download/install 或 runtime router install；同一节点负例必须被 Helm validation 拒绝。

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
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get nodes -o 'custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu,UNSCHED:.spec.unschedulable'
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get pods -A -o 'custom-columns=NS:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase,GPU:.spec.containers[*].resources.requests.nvidia\\.com/gpu,NODE:.spec.nodeName' | sed -n '1,200p'

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
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-96}"
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

**Expected result:** Helm release 安装成功，`1P1D` prefill/decode/router workloads 开始创建；prefill 和 decode 分别带不同节点的 required nodeAffinity，并各自请求 8 张 GPU；P/D/router 运行参数与 servingkit `perf/vllm_dsv4` SHA `53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69` 保持语义一致，只有 image、Onion 模型准备、节点参数化、删除 runtime install/hotfix 是本计划允许的显式差异；`prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`；`--max-model-len` 不应出现在 rendered command 中；router 默认跟随 decode 节点。若 release 已存在且不是本任务创建，停止，不接管未知 owner。

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
  -l storm-service-name="${RELEASE}" \
  --timeout=60m
kubectl wait -n "${NAMESPACE}" --for=condition=Ready pod \
  -l app.kubernetes.io/instance="${RELEASE}" \
  --timeout=60m

kubectl get pods -n "${NAMESPACE}" -o wide | tee "${ARTIFACT_DIR}/pods-ready.txt"
kubectl get svc -n "${NAMESPACE}" -o wide | tee "${ARTIFACT_DIR}/services.txt"

for pod in $(kubectl get pods -n "${NAMESPACE}" -o name | rg "${RELEASE}"); do
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
PROXY_URL="http://100.68.170.29:3128"

uv venv --python 3.12 "${EVAL_VENV}"
HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}" http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}" \
  uv pip install --python "${EVAL_VENV}/bin/python" 'evalscope[perf]==1.8.1'
"${EVAL_VENV}/bin/evalscope" --version | tee "${ARTIFACT_DIR}/evalscope-version.txt"
```

**Expected result:** evalscope `perf` 子命令可执行，版本为 `1.8.1`。所有后续 in-cluster evalscope Pod 也必须通过 `python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'` 安装；若 `uv`、Python、network、package index 或权限阻塞安装，停止并记录 blocker，不切换自定义 harness。

## C17A: Copy tokenizer files for local evalscope

**When:** C17 evalscope ready and C16 service ready；本机没有目标模型 tokenizer 路径时执行。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
TOKENIZER_DIR="${ARTIFACT_DIR}/tokenizer"
MODEL_PATH="/data01/DeepSeek-V4-Flash"

mkdir -p "${TOKENIZER_DIR}"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
prefill_pod="$(kubectl get pods -n "${NAMESPACE}" -l storm-service-name="${RELEASE}",role-name=prefill -o jsonpath='{.items[0].metadata.name}')"

for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
  kubectl cp -n "${NAMESPACE}" -c prefill "${prefill_pod}:${MODEL_PATH}/${f}" "${TOKENIZER_DIR}/${f}"
done
find "${TOKENIZER_DIR}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort | tee "${ARTIFACT_DIR}/tokenizer-files-local.txt"
```

**Expected result:** 只复制 tokenizer/config 小文件到 artifact 目录供 evalscope 生成固定 token 长度请求；不复制模型权重，不修改 serving pod。

## C18: Deploy servingkit monitoring chart and verify scrape targets

**When:** C16 real router smoke passes, C20/C21 measured run 之前。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
MONITORING_NAMESPACE="vllm-dsv4-flash-pd-monitoring"
MONITORING_RELEASE="dsv4-flash-pd-monitoring"
SERVINGKIT_REPO="/data00/home/hanhan.hank/workspace/servingkit"
SERVINGKIT_REF="origin/hanhan_dev"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
MONITORING_DIR="${ARTIFACT_DIR}/monitoring"
MONITORING_CHART="${MONITORING_DIR}/llm-serving-monitoring"
MONITORING_VALUES="${MONITORING_DIR}/values-vllm-dsv4-flash-pd.yaml"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
mkdir -p "${MONITORING_DIR}"
rm -rf "${MONITORING_CHART}"
git -C "${SERVINGKIT_REPO}" archive --format=tar "${SERVINGKIT_REF}" llm-serving-monitoring \
  | tar -C "${MONITORING_DIR}" -xf -

cat > "${MONITORING_VALUES}" <<YAML
namespace:
  create: false
  name: ${MONITORING_NAMESPACE}

prometheus:
  enabled: true
  scrapeInterval: 1s
  scrapeTimeout: 900ms
  externalLabels:
    cluster: ${ENVIRONMENT}
  nodeAffinity:
    enabled: true
    key: kubernetes.io/hostname
    values:
      - "${DECODE_NODE:?set DECODE_NODE to the current decode/router node}"

grafana:
  enabled: false

nodeExporter:
  enabled: false

scrapeTargets:
  - name: ${RELEASE}-prefill
    enabled: true
    metricsPath: /metrics
    targets:
      - address: ${RELEASE}-prefill.${NAMESPACE}.svc.cluster.local:8000
        labels:
          stack: vllm
          release: ${RELEASE}
          role: prefill
          model: deepseek-v4-flash
  - name: ${RELEASE}-decode
    enabled: true
    metricsPath: /metrics
    targets:
      - address: ${RELEASE}-decode.${NAMESPACE}.svc.cluster.local:8001
        labels:
          stack: vllm
          release: ${RELEASE}
          role: decode
          model: deepseek-v4-flash
  - name: ${RELEASE}-router
    enabled: false
    metricsPath: /metrics
    targets:
      - address: ${RELEASE}-router.${NAMESPACE}.svc.cluster.local:30000
        labels:
          stack: vllm
          release: ${RELEASE}
          role: router
          model: deepseek-v4-flash
YAML

helm template "${MONITORING_RELEASE}" "${MONITORING_CHART}" \
  --namespace "${MONITORING_NAMESPACE}" \
  -f "${MONITORING_VALUES}" \
  > "${MONITORING_DIR}/rendered-monitoring.yaml"

helm upgrade --install "${MONITORING_RELEASE}" "${MONITORING_CHART}" \
  --namespace "${MONITORING_NAMESPACE}" \
  --create-namespace \
  -f "${MONITORING_VALUES}" \
  2>&1 | tee "${MONITORING_DIR}/helm-upgrade-monitoring.log"

kubectl wait -n "${MONITORING_NAMESPACE}" \
  --for=condition=Ready pod \
  -l app.kubernetes.io/instance="${MONITORING_RELEASE}",app.kubernetes.io/component=prometheus \
  --timeout=180s \
  | tee "${MONITORING_DIR}/prometheus-ready.txt"

kubectl get all -n "${MONITORING_NAMESPACE}" -o wide \
  | tee "${MONITORING_DIR}/kubectl-get-monitoring.txt"

PROMETHEUS_SERVICE="$(kubectl get svc -n "${MONITORING_NAMESPACE}" \
  -l app.kubernetes.io/instance="${MONITORING_RELEASE}",app.kubernetes.io/component=prometheus \
  -o jsonpath='{.items[0].metadata.name}')"
echo "PROMETHEUS_SERVICE=${PROMETHEUS_SERVICE}" | tee "${MONITORING_DIR}/prometheus-service.txt"

kubectl port-forward -n "${MONITORING_NAMESPACE}" "svc/${PROMETHEUS_SERVICE}" 19090:9090 \
  > "${MONITORING_DIR}/prometheus-port-forward.log" 2>&1 &
pf_pid=$!
trap 'kill "${pf_pid}" >/dev/null 2>&1 || true; wait "${pf_pid}" >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:19090/-/ready" > "${MONITORING_DIR}/prometheus-ready-http.txt"; then
    break
  fi
  sleep 1
  if [ "${i}" = 60 ]; then
    cat "${MONITORING_DIR}/prometheus-port-forward.log" >&2
    exit 1
  fi
done

curl -fsS --get "http://127.0.0.1:19090/api/v1/query" \
  --data-urlencode "query=up{stack=\"vllm\",release=\"${RELEASE}\"}" \
  | tee "${MONITORING_DIR}/prometheus-up-query.json"

python3 - "${MONITORING_DIR}/prometheus-up-query.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
series = data.get("data", {}).get("result", [])
roles = {item.get("metric", {}).get("role"): float(item.get("value", [0, "0"])[1]) for item in series}
required = {"prefill", "decode"}
missing = sorted(role for role in required if roles.get(role) != 1.0)
if missing:
    raise SystemExit(f"Prometheus scrape target not healthy for roles: {missing}; roles={roles}")
print("Prometheus scrape targets healthy:", roles)
PY

kill "${pf_pid}" >/dev/null 2>&1 || true
wait "${pf_pid}" >/dev/null 2>&1 || true
trap - EXIT
```

**Expected result:** 从 servingkit `origin/hanhan_dev:llm-serving-monitoring` 归档出的 Helm chart 成功渲染和部署；最小监控启用 Prometheus，默认关闭 Grafana 和 nodeExporter；Prometheus scrape `prefill`、`decode` 两个 vLLM worker targets 且 `up == 1`。router `/metrics` 当前对 Prometheus GET 返回 `405 Method Not Allowed`，不作为 running BS gate；router 可用性由 C16 `/health`、`/v1/models` 和真实 completion smoke 证明。后续 C21/C21A 必须用该 Prometheus 的 decode `vllm:num_requests_running` / `vllm:num_requests_waiting` 判断实际 running BS 和排队状态。

## C19: Benchmark warmup and cache seed

**When:** C17 evalscope ready and C16 service ready.

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="${ARTIFACT_DIR}/tokenizer"

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
TOKENIZER_PATH="${ARTIFACT_DIR}/tokenizer"

run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "MEASURED_RUN_START=${run_start}" | tee "${ARTIFACT_DIR}/ttft-64k-1out.timestamps"

"${EVALSCOPE}" perf \
  --parallel 1 \
  --number 1 \
  --model "${MODEL}" \
  --url "${URL}" \
  --api openai \
  --dataset random \
  --dataset-offset 1 \
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

**Expected result:** evalscope measured run completes with exit code 0；raw log and outputs are preserved；TTFT metric is reported from streaming request path；`--dataset-offset 1` 避免该请求复用 C19 cache seed 的同一 token 序列；64k/1 Avg TTFT 必须小于 10s，否则不得进入 C10 更新远端 `iaas_main`；P50/P95/P99 归档但不作为 gate。

## C21: Measured cache-hit decode BS512 / 1.5k output throughput

**When:** C19 cache seed passes and target remains stable.

**Working directory:** fork-base integration 分支。

```bash
bash <<'BASH'
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="${ARTIFACT_DIR}/tokenizer"
DECODE_OUTPUT_TOKENS=1536
DECODE_BS=512
DECODE_REQUESTS=$((DECODE_BS * 4))

run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "MEASURED_RUN_START=${run_start}"
  echo "DECODE_BS=${DECODE_BS}"
  echo "DECODE_REQUESTS=${DECODE_REQUESTS}"
} | tee "${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout-n${DECODE_REQUESTS}.timestamps"

"${EVALSCOPE}" perf \
  --parallel "${DECODE_BS}" \
  --number "${DECODE_REQUESTS}" \
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
  --outputs-dir "${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout-n${DECODE_REQUESTS}" \
  2>&1 | tee "${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout-n${DECODE_REQUESTS}.log"
run_status=${PIPESTATUS[0]}

run_end="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "MEASURED_RUN_END=${run_end}"
  echo "MEASURED_RUN_EXIT_CODE=${run_status}"
} | tee -a "${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout-n${DECODE_REQUESTS}.timestamps"
exit "${run_status}"
BASH
```

**Expected result:** 512 并发、2048 总请求以同一 64k prefix cache 命中场景完成；总请求数必须是并发 BS 的 4 倍，用于让 decode 端达到更充分的吞吐状态；每个请求固定 1536 output tokens，并通过 `--extra-args '{"temperature":0,"ignore_eos":true}'` 避免提前 EOS；平均输出 token 数必须达到 `1536 * 95% = 1459.2` 以上才可作为 throughput capacity 结果；该 throughput 是 `1P1D` router-path 输出吞吐，全部 output 来自单个 decode 节点，不是多 decode 聚合；P50/P95/P99 归档但不作为 gate；保存 raw output、timestamps 和 summary。C21 结束或被判定 invalid 后必须执行 C21M 查询实际 running BS；若 BS512 无法完整完成或触发 KV/Mooncake runtime 错误，必须先执行 C21R 重启服务，再执行 C21A 在 128-512 范围内寻找可通过压测的最大 BS，不能直接把 BS512 invalid 当作最终容量结论。该段是原 evalscope gate 的命令要求；2026-07-02 后当前 M10 发布 gate 已切换为用户接受的 `vllm bench serve` BS512/2048 结果，evalscope 结果保留为风险和诊断证据。

## C21A: Fallback cache-hit decode BS sweep between 128 and 512

**When:** C21 的 BS512/2048 请求 run 未完成、出现 KV/Mooncake runtime 错误、或没有有效 Avg/Overall throughput 结果时。

**Working directory:** fork-base integration 分支。

```bash
bash <<'BASH'
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="${ARTIFACT_DIR}/tokenizer"
DECODE_OUTPUT_TOKENS=1536
BS_CANDIDATES="${BS_CANDIDATES:-384 256 192 128}"
SUMMARY="${ARTIFACT_DIR}/decode-bs-sweep-cache-hit-1p5kout.summary.tsv"

printf 'bs\trequests\texit_code\tlog\ttimestamps\n' > "${SUMMARY}"

for bs in ${BS_CANDIDATES}; do
  case "${bs}" in
    ''|*[!0-9]*)
      echo "invalid bs candidate: ${bs}" >&2
      exit 2
      ;;
  esac
  if [ "${bs}" -lt 128 ] || [ "${bs}" -gt 512 ]; then
    echo "bs candidate out of supported range [128,512]: ${bs}" >&2
    exit 2
  fi

  requests=$((bs * 4))
  log="${ARTIFACT_DIR}/evalscope-decode-bs${bs}-cache-hit-1p5kout-n${requests}.log"
  timestamps="${ARTIFACT_DIR}/decode-bs${bs}-cache-hit-1p5kout-n${requests}.timestamps"
  outputs_dir="${ARTIFACT_DIR}/evalscope-decode-bs${bs}-cache-hit-1p5kout-n${requests}"

  run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "MEASURED_RUN_START=${run_start}"
    echo "DECODE_BS=${bs}"
    echo "DECODE_REQUESTS=${requests}"
  } | tee "${timestamps}"

  set +e
  "${EVALSCOPE}" perf \
    --parallel "${bs}" \
    --number "${requests}" \
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
    --outputs-dir "${outputs_dir}" \
    2>&1 | tee "${log}"
  run_status=${PIPESTATUS[0]}
  set -e

  run_end="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "MEASURED_RUN_END=${run_end}"
    echo "MEASURED_RUN_EXIT_CODE=${run_status}"
  } | tee -a "${timestamps}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${bs}" "${requests}" "${run_status}" "${log}" "${timestamps}" >> "${SUMMARY}"

  if [ "${run_status}" -eq 0 ]; then
    echo "C21A first passing BS candidate: ${bs}; inspect ${log} for Avg/Overall throughput and output length gate."
    break
  fi
done
BASH
```

**Expected result:** 每个候选 BS 的总请求数均为 `4 * BS`；候选从高到低尝试，默认覆盖 `384, 256, 192, 128`，并与 C21 的 BS512 一起构成 128-512 区间的降档检查。每个候选结束或被判定 invalid 后都必须执行 C21M；如果某个候选失败、被中断或出现 KV/Mooncake runtime 错误，继续下一个候选前必须先执行 C21R 重启服务，避免把坏状态带到下一轮。第一个 exit code 0 且无 bad-log 的候选是当前部署和节点组合下“能够通过压测的最大已测 BS”；仍需检查平均输出 token 数 `>=1459.2`、evalscope Avg/Overall output throughput、以及 P/D/router bad-log scan。若监控显示实际 decode running BS 明显低于目标候选 BS，下一轮不得测试高于该实际 running BS 的候选，应直接降到不高于 observed max running 的候选。降档 BS 通过只能作为容量诊断结果，不能替代 BS512 gate；2026-07-02 后当前 M10 发布 gate 以用户接受的 `vllm bench serve` BS512/2048 结果为准。

## C21M: Query running BS and queue state from servingkit monitoring

**When:** 每次 C21 或 C21A 候选运行结束、被中断、或被判定 invalid 后立即执行；C21A 选择下一候选前必须读取本步骤结果。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
RELEASE="dsv4-flash-pd"
MONITORING_NAMESPACE="vllm-dsv4-flash-pd-monitoring"
MONITORING_RELEASE="dsv4-flash-pd-monitoring"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
MONITORING_DIR="${ARTIFACT_DIR}/monitoring"
TIMESTAMPS_FILE="${TIMESTAMPS_FILE:?set TIMESTAMPS_FILE to the measured run timestamps file}"
BS_UNDER_TEST="${BS_UNDER_TEST:?set BS_UNDER_TEST to the candidate BS}"

mkdir -p "${MONITORING_DIR}"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

start_raw="$(grep '^MEASURED_RUN_START=' "${TIMESTAMPS_FILE}" | tail -1 | cut -d= -f2-)"
end_raw="$(grep '^MEASURED_RUN_END=' "${TIMESTAMPS_FILE}" | tail -1 | cut -d= -f2- || true)"
if [ -z "${end_raw}" ]; then
  end_raw="$(date -Is)"
fi
start_epoch="$(date -d "${start_raw}" +%s)"
end_epoch="$(date -d "${end_raw}" +%s)"
if [ "${end_epoch}" -le "${start_epoch}" ]; then
  end_epoch=$((start_epoch + 1))
fi

PROMETHEUS_SERVICE="$(kubectl get svc -n "${MONITORING_NAMESPACE}" \
  -l app.kubernetes.io/instance="${MONITORING_RELEASE}",app.kubernetes.io/component=prometheus \
  -o jsonpath='{.items[0].metadata.name}')"
echo "PROMETHEUS_SERVICE=${PROMETHEUS_SERVICE}" | tee "${MONITORING_DIR}/prometheus-service-c21m.txt"

kubectl port-forward -n "${MONITORING_NAMESPACE}" "svc/${PROMETHEUS_SERVICE}" 19090:9090 \
  > "${MONITORING_DIR}/prometheus-port-forward-c21m.log" 2>&1 &
pf_pid=$!
trap 'kill "${pf_pid}" >/dev/null 2>&1 || true; wait "${pf_pid}" >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:19090/-/ready" >/dev/null; then
    break
  fi
  sleep 1
  if [ "${i}" = 60 ]; then
    cat "${MONITORING_DIR}/prometheus-port-forward-c21m.log" >&2
    exit 1
  fi
done

query_range() {
  local name="$1"
  local query="$2"
  curl -fsS --get "http://127.0.0.1:19090/api/v1/query_range" \
    --data-urlencode "query=${query}" \
    --data-urlencode "start=${start_epoch}" \
    --data-urlencode "end=${end_epoch}" \
    --data-urlencode "step=1" \
    > "${MONITORING_DIR}/${name}-bs${BS_UNDER_TEST}.json"
}

query_range "decode-running" "sum(vllm:num_requests_running{stack=\"vllm\",release=\"${RELEASE}\",role=\"decode\"})"
query_range "decode-waiting" "sum(vllm:num_requests_waiting{stack=\"vllm\",release=\"${RELEASE}\",role=\"decode\"})"
query_range "decode-output-tps" "sum(rate(vllm:generation_tokens_total{stack=\"vllm\",release=\"${RELEASE}\",role=\"decode\"}[30s]))"

python3 - "${MONITORING_DIR}" "${BS_UNDER_TEST}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
bs = int(sys.argv[2])

def max_value(name: str) -> float:
    path = directory / f"{name}-bs{bs}.json"
    data = json.loads(path.read_text())
    values = []
    for series in data.get("data", {}).get("result", []):
        values.extend(float(v[1]) for v in series.get("values", []))
    return max(values) if values else 0.0

max_running = max_value("decode-running")
max_waiting = max_value("decode-waiting")
max_output_tps = max_value("decode-output-tps")
summary = directory / f"running-bs-bs{bs}.summary.txt"
summary.write_text(
    f"bs_under_test={bs}\n"
    f"max_decode_running={max_running}\n"
    f"max_decode_waiting={max_waiting}\n"
    f"max_decode_output_tps_30s={max_output_tps}\n"
)
print(summary.read_text(), end="")
if max_running < bs * 0.80:
    print(
        f"WARNING: observed max_decode_running {max_running} is below 80% of target BS {bs}; "
        "do not test higher BS candidates before explaining the admission/routing limit.",
        file=sys.stderr,
    )
PY

kill "${pf_pid}" >/dev/null 2>&1 || true
wait "${pf_pid}" >/dev/null 2>&1 || true
trap - EXIT
```

**Expected result:** 写入 `monitoring/running-bs-bs<BS>.summary.txt`，包含 `max_decode_running`、`max_decode_waiting`、`max_decode_output_tps_30s`。若 `max_decode_running < 0.8 * BS_UNDER_TEST`，说明服务端实际没有达到目标 running BS；后续不得测试高于 observed running capacity 的候选，必须先降低 BS 或解释 admission/routing/queue 限制。

## C21R: Restart service after failed benchmark candidate

**When:** C21 或任一 C21A 候选失败、被中断、出现 KV/Mooncake runtime 错误、或 C21M 显示队列/运行态异常后；继续下一轮压测前必须执行。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
RESTART_DIR="${ARTIFACT_DIR}/service-restarts/$(date '+%Y%m%d-%H%M%S')"

mkdir -p "${RESTART_DIR}"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl get pods -n "${NAMESPACE}" -o wide | tee "${RESTART_DIR}/pods-before-restart.txt"
kubectl logs -n "${NAMESPACE}" -l storm-service-name="${RELEASE}" --all-containers --tail=1000 \
  > "${RESTART_DIR}/stormservice-logs-before-restart.txt" || true
kubectl logs -n "${NAMESPACE}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --all-containers --tail=1000 \
  > "${RESTART_DIR}/router-logs-before-restart.txt" || true

kubectl delete pod -n "${NAMESPACE}" -l storm-service-name="${RELEASE}" --wait=false \
  | tee "${RESTART_DIR}/delete-stormservice-pods.txt"
kubectl delete pod -n "${NAMESPACE}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --wait=false \
  | tee "${RESTART_DIR}/delete-router-pods.txt"

kubectl wait -n "${NAMESPACE}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=prefill --timeout=900s \
  | tee "${RESTART_DIR}/prefill-ready-after-restart.txt"
kubectl wait -n "${NAMESPACE}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=decode --timeout=900s \
  | tee "${RESTART_DIR}/decode-ready-after-restart.txt"
kubectl wait -n "${NAMESPACE}" --for=condition=Ready pod -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --timeout=300s \
  | tee "${RESTART_DIR}/router-ready-after-restart.txt"

kubectl get pods -n "${NAMESPACE}" -o wide | tee "${RESTART_DIR}/pods-after-restart.txt"

kubectl port-forward -n "${NAMESPACE}" "svc/${RELEASE}-router" 30000:30000 \
  > "${RESTART_DIR}/router-port-forward-after-restart.log" 2>&1 &
pf_pid=$!
trap 'kill "${pf_pid}" >/dev/null 2>&1 || true; wait "${pf_pid}" >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:30000/health > "${RESTART_DIR}/router-health-after-restart.txt"; then
    break
  fi
  sleep 1
  if [ "${i}" = 120 ]; then
    cat "${RESTART_DIR}/router-port-forward-after-restart.log" >&2
    exit 1
  fi
done

curl -fsS --max-time 10 http://127.0.0.1:30000/v1/models \
  > "${RESTART_DIR}/router-models-after-restart.json"

kill "${pf_pid}" >/dev/null 2>&1 || true
wait "${pf_pid}" >/dev/null 2>&1 || true
trap - EXIT
```

**Expected result:** 失败候选后的 P/D/router pods 被删除并重建，三类 pod 重新 Ready，router `/health` 与 `/v1/models` 成功；后续 C21A 候选不得复用失败前的服务进程状态。

## C21V: Compare vLLM built-in benchmark script against evalscope inside the vLLM image

**When:** C18 monitoring 已部署，且至少一个 evalscope C21/C21A 候选有完整结果或 invalid evidence 后。该步骤原用于口径对比；2026-07-02 用户接受 `vllm bench serve` 结果后，该步骤的 BS512/2048、64K prefix、1536 output、in-cluster router Service 结果成为当前发布性能 gate 的主要证据。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
VLLM_BENCH_DIR="${ARTIFACT_DIR}/vllm-bench-compare"
ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
BENCH_POD="${BENCH_POD:-dsv4-flash-pd-vllm-bench}"
BENCH_IMAGE="${BENCH_IMAGE:-iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130}"
BENCH_NODE="${BENCH_NODE:-192.168.1.154}"
URL_BASE="http://${RELEASE}-router.${NAMESPACE}.svc.cluster.local:30000"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="/data01/DeepSeek-V4-Flash"
COMPARE_BS="${COMPARE_BS:-128}"
COMPARE_REQUESTS=$((COMPARE_BS * 4))
OUTPUT_TOKENS=1536

mkdir -p "${VLLM_BENCH_DIR}"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

cat > "${VLLM_BENCH_DIR}/comparison-notes.md" <<'MD'
# evalscope vs vLLM bench serve comparison

- evalscope C21/C21A was the original gate; as of 2026-07-02 the user accepts vLLM bench serve as the release performance gate after C21W/C23/C27/C28 explained evalscope instability.
- vLLM built-in `vllm bench serve` must run inside a container created from the same new vLLM image used by the deployment, not from the local workstation Python environment.
- The benchmark container must not install packages, clone code, or apply runtime hotfixes. If `python3 -m vllm.benchmarks.serve --help` fails inside that image, record it as an image/runtime packaging issue.
- vLLM built-in `vllm bench serve` supports `--request-rate inf`, `--max-concurrency`, `--num-prompts`, `--random-prefix-len`, `--random-input-len`, `--random-output-len`, `--ignore-eos`, and `--extra-body`.
- For a cache-hit decode workload, map evalscope `--parallel BS --number 4*BS --prefix-length 65536 --min/max-prompt-length 0 --min/max-tokens 1536` to vLLM `--max-concurrency BS --num-prompts 4*BS --random-prefix-len 65536 --random-input-len 0 --random-output-len 1536 --ignore-eos`.
- Compare output token count, success/fail count, Avg/Overall output throughput, TTFT/TPOT/ITL, and Prometheus `max_decode_running`.
MD

cat > "${VLLM_BENCH_DIR}/vllm-bench-pod.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${BENCH_POD}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: vllm-bench-serve
    app.kubernetes.io/instance: ${RELEASE}
    workspace-env/session-id: codex-vllm-dsv4-flash-pd-bench
    workspace-env/owner: codex
    workspace-env/purpose: vllm-bench-compare
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: "${BENCH_NODE}"
  containers:
    - name: bench
      image: ${BENCH_IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["bash", "-lc", "sleep infinity"]
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
        limits:
          cpu: "8"
          memory: "32Gi"
      volumeMounts:
        - name: model-data
          mountPath: /data01
          readOnly: true
  volumes:
    - name: model-data
      hostPath:
        path: /data01
        type: Directory
EOF

kubectl delete pod -n "${NAMESPACE}" "${BENCH_POD}" --ignore-not-found --wait=true \
  > "${VLLM_BENCH_DIR}/delete-old-bench-pod.txt" 2>&1 || true
kubectl apply -f "${VLLM_BENCH_DIR}/vllm-bench-pod.yaml" \
  > "${VLLM_BENCH_DIR}/apply-bench-pod.txt"
kubectl wait -n "${NAMESPACE}" --for=condition=Ready "pod/${BENCH_POD}" --timeout=300s \
  > "${VLLM_BENCH_DIR}/bench-pod-ready.txt"
kubectl get pod -n "${NAMESPACE}" "${BENCH_POD}" -o wide \
  > "${VLLM_BENCH_DIR}/bench-pod-wide.txt"

set +e
kubectl exec -n "${NAMESPACE}" "${BENCH_POD}" -- \
  python3 -m vllm.benchmarks.serve --help \
  > "${VLLM_BENCH_DIR}/vllm-bench-serve-help.txt" \
  2> "${VLLM_BENCH_DIR}/vllm-bench-serve-help.err"
help_status=$?
set -e
echo "${help_status}" > "${VLLM_BENCH_DIR}/vllm-bench-serve-help.exitcode"
if [ "${help_status}" -ne 0 ]; then
  cat > "${VLLM_BENCH_DIR}/vllm-bench-serve-unavailable.md" <<MD
# vLLM bench serve unavailable

\`python3 -m vllm.benchmarks.serve --help\` exited with ${help_status} inside image \`${BENCH_IMAGE}\`.
See \`vllm-bench-serve-help.err\`.

This means the vLLM built-in benchmark comparison could not run inside the same vLLM image used by deployment.
Do not use vLLM bench as the release gate if this comparison cannot run inside the same vLLM image.
MD
  kubectl delete pod -n "${NAMESPACE}" "${BENCH_POD}" --ignore-not-found --wait=false \
    > "${VLLM_BENCH_DIR}/delete-bench-pod-after-help-failure.txt" 2>&1 || true
  exit 0
fi

for i in $(seq 1 120); do
  if kubectl exec -n "${NAMESPACE}" "${BENCH_POD}" -- \
    curl -fsS --max-time 2 "${URL_BASE}/health" \
    > "${VLLM_BENCH_DIR}/router-health-before-vllm-bench.txt"; then
    break
  fi
  sleep 1
  if [ "${i}" = 120 ]; then
    exit 1
  fi
done

set +e
kubectl exec -n "${NAMESPACE}" "${BENCH_POD}" -- \
  python3 -m vllm.benchmarks.serve \
  --backend openai \
  --base-url "${URL_BASE}" \
  --endpoint /v1/completions \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --dataset-name random \
  --random-prefix-len 65536 \
  --random-input-len 0 \
  --random-output-len "${OUTPUT_TOKENS}" \
  --request-rate inf \
  --max-concurrency "${COMPARE_BS}" \
  --num-prompts "${COMPARE_REQUESTS}" \
  --ignore-eos \
  --temperature 0 \
  --seed 42 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/vllm-bench-compare \
  --result-filename "vllm-bench-serve-bs${COMPARE_BS}-n${COMPARE_REQUESTS}.json" \
  2>&1 | tee "${VLLM_BENCH_DIR}/vllm-bench-serve-bs${COMPARE_BS}-n${COMPARE_REQUESTS}.log"
bench_status=${PIPESTATUS[0]}
set -e
echo "${bench_status}" > "${VLLM_BENCH_DIR}/vllm-bench-serve-bs${COMPARE_BS}-n${COMPARE_REQUESTS}.exitcode"

kubectl cp -n "${NAMESPACE}" "${BENCH_POD}:/tmp/vllm-bench-compare" \
  "${VLLM_BENCH_DIR}/pod-result-dir" \
  > "${VLLM_BENCH_DIR}/kubectl-cp-results.txt" 2>&1 || true
kubectl delete pod -n "${NAMESPACE}" "${BENCH_POD}" --ignore-not-found --wait=false \
  > "${VLLM_BENCH_DIR}/delete-bench-pod-after-run.txt" 2>&1 || true
exit 0
```

**Expected result:** 保存 vLLM 自带压测脚本在同一 vLLM 镜像内的 help、运行日志、JSON 结果和对比说明。不能再用本地工作站缺 `torch` 作为 C21V 的最终状态；若容器内 help 或 benchmark 失败，应归类为候选镜像/容器 runtime 对 `vllm bench serve` 支持不足，并保存 stderr。对比结论必须说明它和 evalscope 在请求生成、concurrency 限制、prefix cache 构造、输出长度固定、统计口径上的差异。2026-07-02 用户已接受 vLLM benchmark 作为当前发布性能 gate，因此用于 M10 的 BS512 gate 结果必须来自同一个新构建 vLLM 镜像、in-cluster router Service、`num-prompts=2048`、`max-concurrency=512`、64K prefix、1536 output，并保存 Prometheus running/waiting 与 bad-log evidence。C21V benchmark Pod 不请求 GPU，使用同一个新构建 vLLM 镜像，不做运行时安装、代码 clone 或 hotfix，结束后必须删除。

## C21W: Analyze why evalscope and vLLM bench results diverge

**When:** C21/C21A evalscope 结果与 C21V/vLLM bench sweep 结果出现显著差异后立即执行。当前触发条件是：evalscope BS512/2048 invalid 且出现 Mooncake/KV transfer failure；vLLM bench BS512/2048 exit `0`、`2048/2048` success、output throughput `15281.44 tok/s`。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
VLLM_SWEEP_DIR="${ARTIFACT_DIR}/vllm-bench-bs-sweep-20260701"
ANALYSIS_DIR="${ARTIFACT_DIR}/harness-diff-analysis-20260701"
mkdir -p "${ANALYSIS_DIR}"

{
  echo "# evalscope vs vLLM bench divergence analysis"
  echo
  echo "## Inputs"
  echo
  echo "- evalscope BS512 log: \`${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout-n2048.log\`"
  echo "- evalscope BS512 timestamps: \`${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout-n2048.timestamps\`"
  echo "- evalscope fallback BS192 log: \`${ARTIFACT_DIR}/evalscope-decode-bs192-cache-hit-1p5kout-n768.log\`"
  echo "- vLLM bench sweep summary: \`${VLLM_SWEEP_DIR}/runs/summary.tsv\`"
  echo "- vLLM bench BS512 log/result: \`${VLLM_SWEEP_DIR}/runs/bs512/vllm-bench.log\`, \`${VLLM_SWEEP_DIR}/runs/bs512/vllm-bench-result.json\`"
  echo "- Prometheus windows: evalscope \`${ARTIFACT_DIR}/monitoring/running-bs-bs512.summary.txt\` and vLLM \`${VLLM_SWEEP_DIR}/runs/bs512/prom-window-summary.json\`"
  echo
  echo "## Required Questions"
  echo
  echo "1. Workload equivalence: endpoint, request format, prompt/prefix construction, tokenizer, output token cap, ignore_eos/temperature, streaming mode, and total request count."
  echo "2. Cache semantics: whether evalscope and vLLM bench both produced the intended cache-hit decode workload, including prefix cache seed behavior and prompt token counters."
  echo "3. Client path: local workstation + port-forward vs in-cluster benchmark Pod, connection reuse, timeout behavior, request generation cost, and whether client-side throttling or timeout explains the difference."
  echo "4. Concurrency model: evalscope \`--parallel/--number\` vs vLLM bench \`--max-concurrency/--num-prompts/--request-rate inf\`, and observed Prometheus running/waiting."
  echo "5. Failure window: whether Mooncake/KV errors align with evalscope only, vLLM bench only, or service state before/after restart."
  echo "6. Statistics: output token throughput denominator, failed/incomplete request handling, elapsed window, Avg vs percentile, and whether partial evalscope progress can be compared to vLLM completed throughput."
  echo
} > "${ANALYSIS_DIR}/summary.md"

{
  echo "## evalscope command and result excerpts"
  echo
  echo "### BS512 timestamps"
  sed -n '1,80p' "${ARTIFACT_DIR}/decode-bs512-cache-hit-1p5kout-n2048.timestamps" 2>/dev/null || true
  echo
  echo "### BS512 result/failure lines"
  rg -n "Total|Success|Failed|Avg Output|Output Throughput|Completion tok|Cached Prompt|TTFT|TPOT|ITL|Mooncake|timeout|Error|Exception|Traceback|exit|Processing" \
    "${ARTIFACT_DIR}/evalscope-decode-bs512-cache-hit-1p5kout-n2048.log" 2>/dev/null || true
  echo
  echo "### BS192 fallback result lines"
  rg -n "Total|Success|Failed|Avg Output|Output Throughput|Completion tok|Cached Prompt|TTFT|TPOT|ITL|Mooncake|timeout|Error|Exception|Traceback|Processing" \
    "${ARTIFACT_DIR}/evalscope-decode-bs192-cache-hit-1p5kout-n768.log" 2>/dev/null || true
} > "${ANALYSIS_DIR}/evalscope-excerpts.txt"

{
  echo "## vLLM bench command and result excerpts"
  echo
  echo "### Sweep summary"
  cat "${VLLM_SWEEP_DIR}/runs/summary.tsv" 2>/dev/null || true
  echo
  echo "### BS512 timestamps"
  sed -n '1,80p' "${VLLM_SWEEP_DIR}/runs/bs512/timestamps.env" 2>/dev/null || true
  echo
  echo "### BS512 command"
  cat "${VLLM_SWEEP_DIR}/runs/bs512/command.txt" 2>/dev/null || true
  echo
  echo "### BS512 result lines"
  rg -n "Successful requests|Failed requests|Benchmark duration|Total input tokens|Total generated tokens|Request throughput|Output token throughput|Mean TTFT|Mean TPOT|Mean ITL|Maximum request concurrency|Max request concurrency|Peak concurrency" \
    "${VLLM_SWEEP_DIR}/runs/bs512/vllm-bench.log" 2>/dev/null || true
} > "${ANALYSIS_DIR}/vllm-bench-excerpts.txt"

{
  echo "## Monitoring comparison"
  echo
  echo "### evalscope BS512 C21M"
  sed -n '1,160p' "${ARTIFACT_DIR}/monitoring/running-bs-bs512.summary.txt" 2>/dev/null || true
  echo
  echo "### evalscope BS192 C21M"
  sed -n '1,160p' "${ARTIFACT_DIR}/monitoring/running-bs-bs192.summary.txt" 2>/dev/null || true
  echo
  echo "### vLLM bench BS512 Prometheus window"
  python3 - <<'PY' "${VLLM_SWEEP_DIR}/runs/bs512/prom-window-summary.json"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(0)
data = json.loads(p.read_text())
print(json.dumps(data, indent=2, sort_keys=True))
PY
} > "${ANALYSIS_DIR}/monitoring-comparison.txt"

python3 - <<'PY' > "${ANALYSIS_DIR}/request-db-summary.txt"
import json, sqlite3
from pathlib import Path

root = Path("artifacts/2026-06-29-vllm-dsv4-flash-pd")
items = [
    ("evalscope BS512 n2048", sorted((root / "evalscope-decode-bs512-cache-hit-1p5kout-n2048").glob("*/deepseek-v4-flash/parallel_512_number_2048/benchmark_data.db"))),
    ("evalscope BS192 n768", sorted((root / "evalscope-decode-bs192-cache-hit-1p5kout-n768").glob("*/deepseek-v4-flash/parallel_192_number_768/benchmark_data.db"))),
]

print("# Request/result database summary")
for name, matches in items:
    print(f"## {name}")
    if not matches:
        print("exists=0")
        continue
    db = matches[-1]
    print(f"path={db}")
    print("exists=1")
    con = sqlite3.connect(db)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table' order by name")]
    print("tables=" + ",".join(tables))
    if "result" in tables:
        cols = [r[1] for r in cur.execute("pragma table_info(result)")]
        print("columns=" + ",".join(cols))
        print("result_count=" + str(cur.execute("select count(*) from result").fetchone()[0]))
        select_cols = [c for c in ["success", "prompt_tokens", "completion_tokens", "latency"] if c in cols]
        if select_cols:
            row = cur.execute("select " + ",".join(select_cols) + " from result limit 1").fetchone()
            print("sample_columns=" + ",".join(select_cols))
            print("sample_row=" + repr(row))
        for c in ["prompt_tokens", "completion_tokens"]:
            if c in cols:
                row = cur.execute(f"select min({c}), avg({c}), max({c}) from result").fetchone()
                print(f"{c}_min_avg_max={row}")
        if "success" in cols:
            print("success_counts=" + repr(cur.execute("select success, count(*) from result group by success order by success").fetchall()))
    con.close()

print("## vLLM bench BS512 JSON")
p = root / "vllm-bench-bs-sweep-20260701" / "runs" / "bs512" / "vllm-bench-result.json"
print(f"path={p}")
if p.exists():
    d = json.loads(p.read_text())
    for k in [
        "completed", "failed", "duration", "total_input_tokens", "total_output_tokens",
        "request_throughput", "output_throughput", "mean_ttft_ms", "mean_tpot_ms",
        "mean_itl_ms", "max_concurrency", "num_prompts",
    ]:
        print(f"{k}={d.get(k)}")
PY

{
  echo "## Bad log comparison"
  echo
  echo "### evalscope BS512 focused bad patterns"
  rg -n "Mooncake found no common|KV group count mismatch|KV load failed|handshake compatibility failure|request timeout during KV pull|Mooncake transfer engine returned -1|Sync batch data transfer timeout|timed out after 480 seconds|Sending to .* failed \\(ret=-1\\)" \
    "${ARTIFACT_DIR}/c21-invalid-mooncake-evidence.txt" \
    "${ARTIFACT_DIR}/result-extract-invalid-c21.txt" \
    "${ARTIFACT_DIR}"/c21-dsv4-flash-pd-roleset-*-logs-during-*.txt \
    "${ARTIFACT_DIR}"/c21-dsv4-flash-pd-roleset-*-logs-final.txt \
    2>/dev/null || true
  echo
  echo "### vLLM bench BS512 bad patterns"
  rg -n "Mooncake found no common|KV group count mismatch|KV load failed|handshake compatibility failure|request timeout during KV pull|Mooncake transfer engine returned -1|Sync batch data transfer timeout|timed out after 480 seconds|Sending to .* failed \\(ret=-1\\)" \
    "${VLLM_SWEEP_DIR}/runs/bs512" "${VLLM_SWEEP_DIR}/smoke" 2>/dev/null || true
} > "${ANALYSIS_DIR}/bad-log-comparison.txt"

cat >> "${ANALYSIS_DIR}/summary.md" <<MD

## Evidence Files

- \`evalscope-excerpts.txt\`
- \`vllm-bench-excerpts.txt\`
- \`monitoring-comparison.txt\`
- \`bad-log-comparison.txt\`
- \`request-db-summary.txt\`

## Analysis Checklist

- [ ] 明确 evalscope 和 vLLM bench 是否真的是同一个 workload。
- [ ] 明确 evalscope 的 BS512 invalid 是否来自服务端 Mooncake/KV 失败，而不是单纯统计口径差异。
- [ ] 明确 vLLM bench BS512 成功是否发生在一个重启后的健康服务状态，且 bad-log scan clean。
- [ ] 对比 evalscope local/port-forward client 与 vLLM in-cluster Pod client 的网络路径和 timeout 行为。
- [ ] 对比 Prometheus 的 \`max_decode_running\`、\`max_decode_waiting\`、\`max_decode_output_tps_30s\`，说明服务端实际压力是否相近。
- [ ] 给出结论分类：\`workload-mismatch\`、\`client-path\`、\`harness-timeout/statistics\`、\`service-state/kv-transfer\`、\`mixed\` 或 \`inconclusive\`。
- [ ] 若结论是 \`inconclusive\`，执行或规划 C21X。

## Provisional Conclusion

请在执行 C21W 后填写。禁止在此处直接接受 vLLM bench BS512 作为 evalscope gate 替代，除非上面的 checklist 有充分证据。
MD

echo "C21W artifacts written to ${ANALYSIS_DIR}"
```

**Expected result:** `harness-diff-analysis-20260701/summary.md` 和配套 evidence 文件存在。结论必须明确说明差异主因，至少覆盖 workload、client path、concurrency/running BS、failure window、stats denominator 和 Mooncake/KV bad-log 证据。若结论是 `inconclusive`，必须执行或规划 C21X；若结论认为 vLLM bench 可替代 evalscope，也必须说明替代的风险、失去的 comparability、以及为什么 evalscope 的 KV/Mooncake failure 不再阻塞发布。

## C21X: Optional paired reproduction for evalscope vs vLLM bench divergence

**When:** 仅当 C21W 无法从现有 artifacts 解释 evalscope 与 vLLM bench 差异时执行。本步骤会重新创建 GPU serving workload，必须重新按 C14 获取 workspace-env GPU permit。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd/harness-paired-repro-20260701"
mkdir -p "${ARTIFACT_DIR}"

cat > "${ARTIFACT_DIR}/paired-repro-plan.md" <<'MD'
# Paired Reproduction Plan

## Goal

在同一次重新部署的健康 `1P1D` 服务进程上，对 evalscope 和 `vllm bench serve` 做最小配对复现，解释为什么历史 evalscope BS512 invalid 而 vLLM bench BS512 成功。

## Required Setup

- 先执行 C13/C14/C15/C16/C18，部署相同 image、相同 chart、相同 `prefill.args.noAsyncScheduling=false`、相同 `decode.args.maxNumSeqs=96`、相同 `1P1D`，并部署 monitoring。
- 重新部署后不要改变 P/D/router runtime 参数。
- 每个失败候选后必须执行 C21R 重启服务，再继续下一候选。

## Paired Matrix

1. BS128 / requests 512 / output 1536：先跑 evalscope，再 C21R 重启，再跑 vLLM bench；或反向顺序重复一次，避免顺序污染。
2. BS256 / requests 1024 / output 1536：仅在 BS128 两种 harness 都 clean 时执行。
3. BS512 / requests 2048 / output 1536：仅在 BS256 两种 harness 都 clean 时执行；若任一 harness 出现 KV/Mooncake failure，停止并归因。

## Required Evidence Per Candidate

- Exact command and timestamps.
- Success / Failed / output tokens / throughput / TTFT / TPOT / ITL.
- Prometheus window: decode running, waiting, generation TPS, prompt TPS, TTFT counters.
- Bad-log scan for prefill, decode, router.
- Service state before and after run: pod restarts, `/health`, `/v1/models`, one real completion if needed.

## Stop Rules

- 任一 harness 出现 Mooncake/KV bad-log：停止该 BS，执行 C21R；不要把坏状态带入下一轮。
- Prometheus 显示实际 decode running 明显低于候选 BS：不测试更高 BS。
- 如果 BS128 都无法 clean 对齐，不继续 BS256/BS512。
MD

echo "C21X is a live repro wrapper. Execute C13/C14/C15/C16/C18 first, then run the paired matrix described in ${ARTIFACT_DIR}/paired-repro-plan.md."
```

**Expected result:** 只有在 C21W 证据不足时才进入本步骤。执行后必须新增一个 `paired-repro` summary，明确同一服务状态下 evalscope 与 vLLM bench 的差异是否复现、是否与 client path 或 service-state 相关、以及是否仍阻止 M10。

## C23: Offline Mooncake/KV failure root-cause triage

**When:** C21W 已解释 evalscope 与 vLLM bench 差异，但 evalscope BS512 仍因 Mooncake/KV transfer failure invalid 时执行。该步骤只使用现有 artifacts、当前部署模板、servingkit reference SHA 和本仓库代码；不创建 GPU workload、不访问 live cluster、不修改源码或部署语义。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
OUT="${ARTIFACT_DIR}/mooncake-failure-diagnosis-20260701"
SERVINGKIT_DIR="/data00/home/hanhan.hank/workspace/servingkit"
SERVINGKIT_SHA="53a6d6a27e59fe1cc620b85c5ee20f51d27e9b69"
MOONCAKE_CONNECTOR="vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

mkdir -p "${OUT}"

uv run --no-project python - <<'PY' > "${OUT}/failure-counts.txt"
from collections import Counter
from pathlib import Path
import re
import statistics

root = Path("artifacts/2026-06-29-vllm-dsv4-flash-pd")
log_files = sorted(root.glob("c21-*logs-during-*.txt")) + sorted(root.glob("c21-*logs-final.txt"))
patterns = {
    "ret_failed": re.compile(r"Sending to ([^ ]+) failed \(ret=(-?\d+)\) after ([0-9.]+)s? \((\d+) descriptors, (\d+) bytes\)"),
    "sync_timeout": re.compile(r"Sync batch data transfer timeout"),
    "producer_timeout": re.compile(r"timed out after 480 seconds without being sent"),
    "xfer_returned": re.compile(r"Mooncake transfer engine returned (-?\d+)"),
    "kv_group_mismatch": re.compile(r"KV group count mismatch"),
    "handshake_failure": re.compile(r"handshake compatibility failure"),
    "no_common_regions": re.compile(r"Mooncake found no common KV transfer regions"),
}

counts = Counter()
sessions = Counter()
durations = []
descriptors = []
bytes_values = []
examples = []

for path in log_files:
    text = path.read_text(errors="replace")
    for name, pattern in patterns.items():
        matches = list(pattern.finditer(text))
        counts[name] += len(matches)
        if name == "ret_failed":
            for match in matches:
                session, ret, duration_s, descriptor_count, byte_count = match.groups()
                sessions[session] += 1
                durations.append(float(duration_s))
                descriptors.append(int(descriptor_count))
                bytes_values.append(int(byte_count))
                if len(examples) < 12:
                    examples.append((path.name, session, ret, duration_s, descriptor_count, byte_count))

def stat_line(name, values):
    if not values:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(values)} min={min(values)} "
        f"p50={statistics.median(values)} max={max(values)}"
    )

print(f"log_files={len(log_files)}")
for path in log_files:
    print(f"  {path}")
print("")
print("counts:")
for key in sorted(patterns):
    print(f"  {key}={counts[key]}")
print("")
print("ret_failed_by_remote_session:")
for session, count in sessions.most_common():
    print(f"  {session}={count}")
print("")
print("transfer_failure_stats:")
print("  " + stat_line("duration_s", durations))
print("  " + stat_line("descriptors", descriptors))
print("  " + stat_line("bytes", bytes_values))
print("")
print("first_ret_failed_examples:")
for item in examples:
    print("  file={} session={} ret={} duration_s={} descriptors={} bytes={}".format(*item))
PY

{
  echo "# MooncakeConnector code path excerpts"
  echo
  echo "## send loop and _send_blocks result handling"
  sed -n '1598,1650p' "${MOONCAKE_CONNECTOR}"
  echo
  echo "## batch_transfer_sync_write failure log"
  sed -n '1868,1892p' "${MOONCAKE_CONNECTOR}"
  echo
  echo "## producer-side unsent timeout"
  sed -n '2058,2088p' "${MOONCAKE_CONNECTOR}"
  echo
  echo "## consumer receive error path"
  sed -n '2148,2190p' "${MOONCAKE_CONNECTOR}"
  echo
  echo "## Mooncake env defaults"
  rg -n "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT|VLLM_MOONCAKE_BOOTSTRAP_PORT|MOONCAKE_PREFERRED_SEGMENT|MOONCAKE_REQUESTER_LOCAL_HOSTNAME" vllm/envs.py
} > "${OUT}/code-path-excerpts.txt"

{
  echo "# servingkit reference vs local deployment values"
  echo
  echo "Reference: ${SERVINGKIT_DIR}@${SERVINGKIT_SHA}:vllm/deepseek/deepseek-v4-flash-pd/values.yaml"
  echo "Local: examples/deployment/deepseek-v4-flash-pd/values.yaml"
  echo
  if [ -d "${SERVINGKIT_DIR}/.git" ]; then
    diff -u \
      <(git -C "${SERVINGKIT_DIR}" show "${SERVINGKIT_SHA}:vllm/deepseek/deepseek-v4-flash-pd/values.yaml" | sed -n '/^vllm:/,/^service:/p') \
      <(sed -n '/^vllm:/,/^service:/p' examples/deployment/deepseek-v4-flash-pd/values.yaml) || true
  else
    echo "servingkit checkout not found at ${SERVINGKIT_DIR}"
  fi
} > "${OUT}/servingkit-values-diff.txt"

{
  echo "# Node, render, and benchmark comparison"
  echo
  echo "## C21W summary"
  sed -n '1,220p' "${ARTIFACT_DIR}/harness-diff-analysis-20260701/summary.md" || true
  echo
  echo "## vLLM bench sweep summary"
  sed -n '1,180p' "${ARTIFACT_DIR}/vllm-bench-bs-sweep-20260701/summary.md" || true
  echo
  echo "## Rendered command and scheduling grep"
  rg -n -- '--kv-transfer-config|--max-num-seqs|--max-num-batched-tokens|--cp-kv-cache-interleave-size|--max-model-len|--no-async-scheduling|nodeAffinity|nvidia.com/gpu|image:' \
    "${ARTIFACT_DIR}/rendered-dsv4-flash-pd.yaml" \
    "${ARTIFACT_DIR}/vllm-bench-bs-sweep-20260701/rendered-dsv4-flash-pd-vllmbench-sweep.yaml" || true
} > "${OUT}/node-and-render-comparison.txt"

{
  echo "# Mooncake/KV Failure Offline Diagnosis"
  echo
  echo "## 结论"
  echo
  echo "- 当前 C23 是离线初筛，不修改代码、不修改部署语义、不创建 GPU workload。"
  echo "- evalscope BS512 的失败主路径是 Mooncake producer 侧同步批量 transfer 在约 30-32s 超时，随后 consumer 侧收到 \`Mooncake transfer engine returned -1\`，producer 侧堆积请求在 480s 后按 abort timeout 释放。"
  echo "- 现有日志未显示 \`KV group count mismatch\`、\`handshake compatibility failure\` 或 \`Mooncake found no common KV transfer regions\`，因此不像 metadata/握手不一致。"
  echo "- servingkit 对齐检查显示当前部署仍需以 diff 文件为准审计有意差异；如果 diff 只包含 image、Onion、节点参数化和 runtime install 删除，则部署语义发散不是首要假设。"
  echo "- 当前首要假设是 evalscope BS512 的请求突发/客户端路径触发了 Mooncake/RDMA transfer timeout 或 descriptor pressure；vLLM bench 在后续健康服务状态下成功不能证明 evalscope gate 已通过。"
  echo "- 如需继续推进，应做 live diagnostic：同一新部署、同一节点、同一 prefix/tokenizer、先 evalscope 再 vLLM 或交错顺序，采集 Mooncake debug/RDMA/descriptor 证据，并在每次失败后重启服务。"
  echo
  echo "## Evidence files"
  echo
  echo "- \`failure-counts.txt\`"
  echo "- \`code-path-excerpts.txt\`"
  echo "- \`servingkit-values-diff.txt\`"
  echo "- \`node-and-render-comparison.txt\`"
} > "${OUT}/summary.md"

echo "C23 artifacts written to ${OUT}"
```

**Expected result:** `mooncake-failure-diagnosis-20260701/summary.md` 和配套 evidence 文件存在。`failure-counts.txt` 应给出 Mooncake transfer timeout、`ret=-1`、480s producer timeout、metadata/handshake mismatch 的计数与失败 transfer 大小统计；summary 必须给出当前根因假设、已排除项和是否需要 live diagnostic。若结果显示 metadata/部署语义不一致，先修正部署/镜像对齐；若结果只显示 Mooncake/RDMA timeout 或 descriptor pressure，后续 live diagnostic 必须先采集更细粒度 Mooncake/RDMA 证据，不能直接改 vLLM runtime 源码。

## C24: Live Mooncake/KV repro, flakiness check, and harness cross-check

**When:** C23 完成后，用户要求继续部署并分析根因时执行。该步骤使用当前已成功构建镜像，不改代码；目标是判断 evalscope BS512 failure 是稳定复现、偶现、harness-specific、节点/环境相关，还是需要 C25 调试镜像。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
LIVE_DIR="${ARTIFACT_DIR}/live-mooncake-diagnosis-20260701"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
DECODE_BS=512
DECODE_REQUESTS=2048
DECODE_OUTPUT_TOKENS=1536

mkdir -p "${LIVE_DIR}"

cat > "${LIVE_DIR}/execution-plan.md" <<MD
# C24 Live Mooncake/KV Diagnosis Execution Plan

## 目标

- 使用当前成功构建镜像：\`${IMAGE}\`
- 保持部署语义：\`1P1D\`、P/D 不同 8-GPU 节点、\`prefill.args.noAsyncScheduling=false\`、\`decode.args.maxNumSeqs=96\`、无 \`--max-model-len\`
- 先按 C13-C18 重新 render、申请 permit、部署、smoke、部署 monitoring
- 执行 evalscope BS512/2048 repeated repro，尝试之间必须执行 C21R 重启服务
- 执行 vLLM bench BS512 cross-check，仍在同一 vLLM 镜像的 no-GPU benchmark Pod 内
- 保存每次 run 的 timestamps、exit code、Prometheus running/waiting/output TPS、bad-log scan、P/D/router logs、events 和 request DB summary

## 必填执行参数

\`\`\`bash
export IMAGE="${IMAGE}"
export PREFILL_NODE="<8-GPU-prefill-node>"
export DECODE_NODE="<different-8-GPU-decode-node>"
export ROUTER_NODE="\${DECODE_NODE}"
export GLOBAL_GPU_COUNT=8
export WORKSPACE_ENV_SESSION_ID="<returned-by-C14>"
\`\`\`

## 执行顺序

1. 用上述参数执行 C13。
2. 执行 C14 获取 workspace-env GPU permit；状态不是 \`granted\` 或 \`running\` 时停止。
3. 执行 C15 部署当前镜像。
4. 执行 C16 real router smoke。
5. 执行 C17/C17A 确认 evalscope 和 tokenizer。
6. 执行 C18 部署 monitoring。
7. 执行 C19 warmup/cache seed。
8. 执行本 C24 的 repeat matrix。
9. 若失败或切换候选，每次下一轮前执行 C21R。
10. 结束后执行 C22 清理。
MD

cat > "${LIVE_DIR}/repeat-matrix.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"
LIVE_DIR="${ARTIFACT_DIR}/live-mooncake-diagnosis-20260701"
EVALSCOPE="$PWD/.venv-evalscope/bin/evalscope"
URL="http://127.0.0.1:30000/v1/completions"
MODEL="deepseek-v4-flash"
TOKENIZER_PATH="${ARTIFACT_DIR}/tokenizer"
DECODE_BS=512
DECODE_REQUESTS=2048
DECODE_OUTPUT_TOKENS=1536

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

run_evalscope_bs512() {
  local attempt="$1"
  local out_dir="${LIVE_DIR}/evalscope-bs512-attempt-${attempt}"
  mkdir -p "${out_dir}"

  kubectl get pods -n "${NAMESPACE}" -o wide > "${out_dir}/pods-before.txt"
  kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp > "${out_dir}/events-before.txt" || true

  local timestamps="${out_dir}/timestamps.txt"
  local log="${out_dir}/evalscope.log"
  local outputs_dir="${out_dir}/evalscope-output"
  local run_start
  run_start="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "MEASURED_RUN_START=${run_start}"
    echo "DECODE_BS=${DECODE_BS}"
    echo "DECODE_REQUESTS=${DECODE_REQUESTS}"
    echo "DECODE_OUTPUT_TOKENS=${DECODE_OUTPUT_TOKENS}"
    echo "HARNESS=evalscope"
    echo "ATTEMPT=${attempt}"
  } | tee "${timestamps}"

  set +e
  "${EVALSCOPE}" perf \
    --parallel "${DECODE_BS}" \
    --number "${DECODE_REQUESTS}" \
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
    --outputs-dir "${outputs_dir}" \
    2>&1 | tee "${log}"
  local run_status=${PIPESTATUS[0]}
  set -e

  local run_end
  run_end="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "MEASURED_RUN_END=${run_end}"
    echo "MEASURED_RUN_EXIT_CODE=${run_status}"
  } | tee -a "${timestamps}"

  kubectl get pods -n "${NAMESPACE}" -o wide > "${out_dir}/pods-after.txt" || true
  kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp > "${out_dir}/events-after.txt" || true
  kubectl logs -n "${NAMESPACE}" -l storm-service-name="${RELEASE}" --all-containers --tail=2000 > "${out_dir}/stormservice-logs-tail.txt" || true
  kubectl logs -n "${NAMESPACE}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --all-containers --tail=1000 > "${out_dir}/router-logs-tail.txt" || true
  rg -n "Sync batch data transfer timeout|Sending to .* failed|Mooncake transfer engine returned|timed out after 480 seconds|KV group count mismatch|handshake compatibility failure|Mooncake found no common KV transfer regions" \
    "${out_dir}" > "${out_dir}/bad-log-scan.txt" || true

  TIMESTAMPS_FILE="${timestamps}" BS_UNDER_TEST="${DECODE_BS}" bash -lc '
    set -euo pipefail
    # Inline C21M-compatible marker. Full Prometheus C21M should be run by the operator after each attempt.
    echo "Run C21M now with TIMESTAMPS_FILE=${TIMESTAMPS_FILE} BS_UNDER_TEST=${BS_UNDER_TEST}"
  ' | tee "${out_dir}/c21m-reminder.txt"

  return "${run_status}"
}

summary="${LIVE_DIR}/repeat-summary.tsv"
printf 'attempt\tharness\texit_code\tpath\n' > "${summary}"

for attempt in 1 2; do
  set +e
  run_evalscope_bs512 "${attempt}"
  status=$?
  set -e
  printf '%s\tevalscope\t%s\t%s\n' "${attempt}" "${status}" "${LIVE_DIR}/evalscope-bs512-attempt-${attempt}" >> "${summary}"

  # Even after success, restart once before the second attempt to distinguish
  # stable success from stateful leftover effects. After failure this is required.
  if [ "${attempt}" = "1" ]; then
    echo "Run C21R before attempt 2; do not carry service state forward." | tee "${LIVE_DIR}/restart-required-before-attempt-2.txt"
    exit 20
  fi
done
BASH
chmod +x "${LIVE_DIR}/repeat-matrix.sh"

cat > "${LIVE_DIR}/summary.md" <<MD
# C24 Live Mooncake/KV Diagnosis

## 当前状态

- C24 scaffolding 已生成。
- 先按 \`execution-plan.md\` 执行 C13-C19，再运行 \`repeat-matrix.sh\`。
- \`repeat-matrix.sh\` 在 attempt 1 后会以 exit 20 停止，强制执行 C21R 重启，再继续 attempt 2，避免把失败或成功状态带到下一轮。

## 分类规则

- \`stable-repro\`: evalscope 两次 BS512 都失败，并出现同类 Mooncake/KV transfer bad logs。
- \`intermittent\`: 两次结果不一致，或多次执行中同一部署语义有成有败。
- \`harness-specific\`: evalscope 失败但同一 live 服务状态下 vLLM bench BS512 成功，且服务端证据显示差异集中在请求/客户端/限流/超时路径。
- \`node/environment-specific\`: 换节点或节点事件/链路/RDMA 证据指向环境差异。
- \`instrumentation-needed\`: C24 仍不能定位，需要 C25 调试镜像。
- \`resolved-by-successful-rerun\`: 至少两次 evalscope BS512 连续成功、throughput 达标、bad-log clean；仍需解释历史失败差异后才能考虑 M10。
MD

echo "C24 scaffolding written to ${LIVE_DIR}"
```

**Expected result:** 生成 `live-mooncake-diagnosis-20260701/execution-plan.md`、`repeat-matrix.sh` 和初始 `summary.md`。执行者必须先按 `execution-plan.md` 跑 C13-C19，再运行 `repeat-matrix.sh`；attempt 1 后必须执行 C21R 重启，再继续 attempt 2。每次 attempt 后必须补跑 C21M，把 Prometheus summary 保存到对应 attempt 目录或在 summary 中引用。若 C24 仍不能定位根因，则进入 C25。

## C24A: Actual live repro commands and evidence from 2026-07-01

**When:** C24 已执行完毕后用于复现相同证据采集或审计本轮结果；不应在未重新获取 workspace-env permit 的情况下直接创建 GPU workload。

**Working directory:** fork-base integration 分支。

**实际 evalscope BS512/2048 命令形态：**

```bash
set -euo pipefail

ATT="artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/evalscope-bs512-attempt-2"
mkdir -p "${ATT}/evalscope-output"

.venv-evalscope/bin/evalscope perf \
  --parallel 512 \
  --number 2048 \
  --model deepseek-v4-flash \
  --url http://127.0.0.1:30000/v1/completions \
  --api openai \
  --dataset random \
  --prefix-length 65536 \
  --min-prompt-length 0 \
  --max-prompt-length 0 \
  --min-tokens 1536 \
  --max-tokens 1536 \
  --tokenizer-path artifacts/2026-06-29-vllm-dsv4-flash-pd/tokenizer \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs "${ATT}/evalscope-output" \
  2>&1 | tee "${ATT}/evalscope.log"
```

**实际 vLLM bench BS512/2048 cross-check 命令形态：**

```bash
set -euo pipefail

NS="vllm-dsv4-flash-pd"
ENV="/data00/home/hanhan.hank/workspace/env/bin/envctl"
POD="vllm-bench-bs512-crosscheck"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130"

"${ENV}" kubectl dev-cluster delete pod -n "${NS}" "${POD}" --ignore-not-found --wait=true
"${ENV}" kubectl dev-cluster apply -n "${NS}" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  labels:
    app.kubernetes.io/name: vllm-bench-bs512-crosscheck
spec:
  restartPolicy: Never
  nodeName: 192.168.1.148
  containers:
  - name: bench
    image: ${IMAGE}
    imagePullPolicy: IfNotPresent
    command: ["bash", "-lc", "sleep 86400"]
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: TZ
      value: Asia/Shanghai
    volumeMounts:
    - name: models
      mountPath: /data01
    - name: shared-mem
      mountPath: /dev/shm
  volumes:
  - name: models
    hostPath:
      path: /data01
      type: DirectoryOrCreate
  - name: shared-mem
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
EOF

"${ENV}" kubectl dev-cluster wait -n "${NS}" --for=condition=Ready "pod/${POD}" --timeout=300s
"${ENV}" kubectl dev-cluster exec -n "${NS}" "${POD}" -- bash -lc '
python3 - <<PY
from pathlib import Path
p = Path("/data01/DeepSeek-V4-Flash")
print("model_path_exists", p.exists(), "tokenizer_config", (p/"tokenizer_config.json").exists(), "tokenizer_json", (p/"tokenizer.json").exists())
PY
vllm bench serve \
  --backend openai \
  --base-url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000 \
  --endpoint /v1/completions \
  --model deepseek-v4-flash \
  --tokenizer /data01/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-prefix-len 65536 \
  --random-input-len 0 \
  --random-output-len 1536 \
  --request-rate inf \
  --max-concurrency 512 \
  --num-prompts 2048 \
  --ignore-eos \
  --temperature 0 \
  --seed 42 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/vllm-bench-bs512-crosscheck \
  --result-filename vllm-bench-serve-bs512-n2048.json
'
```

**实际 Prometheus exact window 对比命令形态：**

```bash
set -euo pipefail

OUT="artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/prometheus-window-comparison"
mkdir -p "${OUT}"

if ! ss -ltn '( sport = :19090 )' | rg -q ':19090'; then
  /data00/home/hanhan.hank/workspace/env/bin/envctl port-forward dev-cluster \
    vllm-dsv4-flash-pd-monitoring \
    svc/dsv4-flash-pd-monitoring-llm-serving-monitoring-prometheus \
    19090:9090 > "${OUT}/prom-port-forward-exact.log" 2>&1 &
  echo $! > "${OUT}/prom-port-forward-exact.pid"
fi

for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:19090/-/healthy >/dev/null 2>&1 && break
  sleep 1
done

python3 - <<'PY'
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

windows = {
    "evalscope_attempt2_exact": ("2026-07-01T11:39:54+08:00", "2026-07-01T11:44:11+08:00"),
    "vllm_bench_exact_from_json": ("2026-07-01T11:55:26+08:00", "2026-07-01T11:58:54+08:00"),
}
queries = {
    "decode_running": 'sum(vllm:num_requests_running{stack="vllm",release="dsv4-flash-pd",role="decode"})',
    "decode_waiting": 'sum(vllm:num_requests_waiting{stack="vllm",release="dsv4-flash-pd",role="decode"})',
    "prefill_running": 'sum(vllm:num_requests_running{stack="vllm",release="dsv4-flash-pd",role="prefill"})',
    "prefill_waiting": 'sum(vllm:num_requests_waiting{stack="vllm",release="dsv4-flash-pd",role="prefill"})',
    "decode_gen_tps_30s": 'sum(rate(vllm:generation_tokens_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s]))',
}
base = "http://127.0.0.1:19090/api/v1/query_range"
out = []
for window_name, (start_s, end_s) in windows.items():
    start = dt.datetime.fromisoformat(start_s).timestamp()
    end = dt.datetime.fromisoformat(end_s).timestamp()
    out.append(f"[{window_name}] {start_s} {end_s}")
    for metric_name, query in queries.items():
        params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": "5s"})
        data = json.load(urllib.request.urlopen(base + "?" + params, timeout=30))
        vals = []
        for series in data.get("data", {}).get("result", []):
            vals.extend(float(v[1]) for v in series.get("values", []) if v[1] not in ("NaN", "nan"))
        if vals:
            out.append(f"  {metric_name}: min={min(vals):.2f} avg={sum(vals)/len(vals):.2f} max={max(vals):.2f} n={len(vals)}")
        else:
            out.append(f"  {metric_name}: no-data")
    out.append("")
Path("artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/prometheus-window-comparison/exact-summary.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
PY
```

**Observed 2026-07-01 result:** evalscope attempt 1 是 `intermittent` producer timeout/stall；evalscope attempt 2 完成但 Avg output throughput `12291.66 tok/s` 未达 gate；vLLM bench cross-check 完成且 Avg output throughput `15152.89 tok/s`、Mean TTFT `7964.35 ms` 通过。详见进展日志 `P72`-`P75` 与 `artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/summary.md`。

## C25: Add gated Mooncake diagnostic logging, push branch, build debug image, and retest

**When:** 仅当 C24 结果为 `instrumentation-needed` 或无法区分偶现、harness-specific、Mooncake/RDMA timeout、descriptor pressure 时执行。用户已在 2026-07-01 允许修改调试代码并推送；该授权只覆盖诊断分支和调试镜像，不覆盖更新 `iaas_main`。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

DIAG_BRANCH="codex/vllm-dsv4-mooncake-transfer-diagnostics"
BASE_BRANCH="codex/vllm-dsv4-fork-base-byteiaas-build"
DIAG_ENV="VLLM_DSV4_MOONCAKE_DIAG"
TARGET_FILE="vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

git switch "${BASE_BRANCH}"
git switch -C "${DIAG_BRANCH}"

rg -n "_send_blocks|batch_transfer_sync_write|fetch_finished_sending_reqs|receive_kv_from_single_worker|VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT" "${TARGET_FILE}" vllm/envs.py

cat <<'PLAN'
Apply a minimal patch before push:

1. Add an env-backed flag such as VLLM_DSV4_MOONCAKE_DIAG in vllm/envs.py.
2. In MooncakeConnector producer send path, when the flag is enabled, log:
   - role/rank or worker identity when available
   - remote_session
   - request count in the batch
   - descriptor count
   - total bytes
   - elapsed seconds
   - ret value
   - first few request ids only if already present in normal logs; do not log prompts or tokens
3. In consumer receive error path, when the flag is enabled, log:
   - request count
   - returned error
   - remote prefill/decode address metadata already present in request ids
4. In producer timeout path, when the flag is enabled, log:
   - pending/expired request count snapshot
   - timeout threshold
5. Do not change transfer behavior, timeout, scheduling, connector config, retry behavior, or fallback imports.
6. Do not enable the diagnostic flag by default in normal values; only set it for C25 debug deployment.
PLAN

uv run --no-project python -m py_compile vllm/envs.py "${TARGET_FILE}"
git diff --check -- vllm/envs.py "${TARGET_FILE}"
git diff -- vllm/envs.py "${TARGET_FILE}" > "artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701/c25-debug-code.diff"

git status --short
git add vllm/envs.py "${TARGET_FILE}" docs/superpowers/plans/2026-06-29-vllm-dsv4-flash-pd-build*.md
git commit -s -m "chore: add gated Mooncake transfer diagnostics"
git push origin "${DIAG_BRANCH}"

gh workflow run byteiaas-release-dev.yml \
  --ref "${DIAG_BRANCH}" \
  -f checkout_ref="${DIAG_BRANCH}"

echo "Wait for the ByteIAAS dev workflow to produce a debug openai-devel image, then rerun C13-C19 and C24 with:"
echo "  --set prefill.extraEnv.${DIAG_ENV}=1 --set decode.extraEnv.${DIAG_ENV}=1"
echo "Run any evalscope benchmark Pod with explicit proxy install:"
echo "  python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'"
echo "Render-grep diagnostic env and proxy benchmark Pod YAML before deploy/run."
```

**Expected result:** 调试分支推送成功，workflow 触发成功，调试镜像构建成功后用同一部署语义复测。调试代码默认不改变行为，只有 `VLLM_DSV4_MOONCAKE_DIAG=1` 开启额外日志；调试镜像不得用于 M10。若 C25 发现明确代码或配置根因，再单独规划最小修复，不把调试日志当作最终修复。

## C26: Offline evalscope vs vLLM bench deep dive, including speculative decoding

**When:** C24 已完成，且用户要求继续分析 vLLM benchmark 与 evalscope benchmark 的差异，尤其 speculative decoding 接受率是否不同。该步骤只读已有 artifacts，不创建 GPU workload，不访问 live cluster，不修改源码，不改变 gate。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

BASE="artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701"
OUT="${BASE}/benchmark-diff-deep-dive-20260701"
mkdir -p "${OUT}"

python3 - <<'PY'
import json
import pickle
import sqlite3
from pathlib import Path
from statistics import mean

base = Path("artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701")
out = base / "benchmark-diff-deep-dive-20260701"
out.mkdir(parents=True, exist_ok=True)

eval_summary = json.loads((base / "evalscope-bs512-attempt-2/benchmark_summary.json").read_text())
eval_workload = json.loads((base / "evalscope-bs512-attempt-2/workload_throughput.json").read_text())
vllm_json = json.loads((base / "vllm-bench-bs512-crosscheck/vllm-bench-serve-bs512-n2048.json").read_text())

def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)

def summarize(name, values):
    return (
        f"{name}: n={len(values)}, avg={mean(values):.4f}, "
        f"p50={percentile(values, 50):.4f}, p90={percentile(values, 90):.4f}, "
        f"p99={percentile(values, 99):.4f}, min={min(values):.4f}, max={max(values):.4f}"
    )

conn = sqlite3.connect(base / "evalscope-bs512-attempt-2/benchmark_data.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "select started_time, completed_time, latency, ttft, tpot, output_tokens "
    "from result order by started_time"
).fetchall()
conn.close()

eval_start0 = rows[0]["started_time"]
eval_all = {
    "start_span_s": rows[-1]["started_time"] - rows[0]["started_time"],
    "complete_span_s": max(r["completed_time"] for r in rows) - min(r["started_time"] for r in rows),
    "completed_span_s": max(r["completed_time"] for r in rows) - min(r["completed_time"] for r in rows),
}

def eval_segment(label, seg):
    return "\n".join([
        f"[evalscope {label}]",
        f"start_span_s={seg[-1]['started_time'] - seg[0]['started_time']:.4f} "
        f"complete_span_s={max(r['completed_time'] for r in seg) - min(r['started_time'] for r in seg):.4f} "
        f"completed_span_s={max(r['completed_time'] for r in seg) - min(r['completed_time'] for r in seg):.4f}",
        summarize("latency_s", [r["latency"] for r in seg]),
        summarize("ttft_ms", [r["ttft"] for r in seg]),
        summarize("tpot_ms", [r["tpot"] for r in seg]),
        "",
    ])

v_start = vllm_json["request_start_times"]
v_ttft = vllm_json["ttfts"]
v_itls = vllm_json["itls"]
v_order = sorted(range(len(v_start)), key=lambda i: v_start[i])

def v_segment(label, idxs):
    flat_itls = [x for i in idxs for x in v_itls[i]]
    starts = [v_start[i] for i in idxs]
    return "\n".join([
        f"[vllm {label}]",
        f"start_span_s={max(starts) - min(starts):.4f}",
        summarize("ttft_ms", [v_ttft[i] for i in idxs]),
        summarize("itl_ms_flat", [x * 1000 for x in flat_itls]),
        "",
    ])

lines = []
lines.append(eval_segment("all", rows))
lines.append(eval_segment("first512", rows[:512]))
lines.append(eval_segment("last512", rows[-512:]))
lines.append(v_segment("all", v_order))
lines.append(v_segment("first512", v_order[:512]))
lines.append(v_segment("last512", v_order[-512:]))
lines.append("[summary metrics]")
lines.append(f"evalscope output_throughput={eval_summary.get('Output Throughput')} duration={eval_summary.get('Duration')} total_output={eval_summary.get('Output Tokens')}")
lines.append(f"evalscope decoded_tok_iter={eval_summary.get('Decoded Tok/Iter')} spec_accept_rate={eval_summary.get('Spec. Accept Rate')}")
lines.append(f"vllm output_throughput={vllm_json.get('output_throughput')} duration={vllm_json.get('duration')} total_output={vllm_json.get('total_output_tokens')}")
lines.append(f"vllm json_has_spec_fields={[k for k in vllm_json if any(s in k.lower() for s in ['spec','accept','draft'])]}")
for item in eval_workload:
    lines.append(f"evalscope workload {json.dumps(item, ensure_ascii=False)}")
(out / "metrics-comparison.txt").write_text("\n".join(lines) + "\n")
print(out / "metrics-comparison.txt")
PY

python3 - <<'PY'
import base64
import json
import pickle
import sqlite3
from pathlib import Path

base = Path("artifacts/2026-06-29-vllm-dsv4-flash-pd/live-mooncake-diagnosis-20260701")
out = base / "benchmark-diff-deep-dive-20260701"
vllm_json = json.loads((base / "vllm-bench-bs512-crosscheck/vllm-bench-serve-bs512-n2048.json").read_text())

v_total_output = vllm_json["total_output_tokens"]
v_total_itl_events = sum(len(x) for x in vllm_json["itls"])

conn = sqlite3.connect(base / "evalscope-bs512-attempt-2/benchmark_data.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("select output_tokens, chunks from result order by id").fetchall()
conn.close()

eval_total_output = sum(r["output_tokens"] for r in rows)
eval_chunks = 0
eval_nonempty = 0
eval_itl_events = 0
decode_ok = 0
decode_err = 0
for r in rows:
    raw = r["chunks"]
    try:
        chunks = pickle.loads(base64.b64decode(raw))
        decode_ok += 1
    except Exception:
        decode_err += 1
        continue
    eval_chunks += len(chunks)
    for chunk in chunks:
        text = ""
        try:
            text = chunk["choices"][0].get("text") or ""
        except Exception:
            pass
        if text:
            eval_nonempty += 1
    eval_itl_events += max(len(chunks) - 1, 0)

lines = [
    "[vllm bench chunk estimates]",
    f"requests={len(vllm_json['itls'])} total_output_tokens={v_total_output} total_itl_events={v_total_itl_events}",
    f"tokens_per_itl_event={v_total_output / v_total_itl_events:.4f}",
    "",
    "[evalscope response chunk estimates]",
    f"requests={len(rows)} decode_ok={decode_ok} decode_err={decode_err} total_output_tokens={eval_total_output}",
    f"total_response_chunks={eval_chunks} tokens_per_response_chunk={eval_total_output / eval_chunks:.4f}",
    f"total_nonempty_text_chunks={eval_nonempty} tokens_per_nonempty_text_chunk={eval_total_output / eval_nonempty:.4f}",
    f"total_itl_events={eval_itl_events} tokens_per_itl_event={eval_total_output / eval_itl_events:.4f}",
]
(out / "chunk-granularity.txt").write_text("\n".join(lines) + "\n")
print(out / "chunk-granularity.txt")
PY

cat > "${OUT}/summary.md" <<'MD'
# vLLM bench vs evalscope Deep Dive

## Classification

`client-path + TTFT/admission dominated; speculative-acceptance-not-primary`

Evalscope attempt 2 did not meet the Avg gate, but current evidence does not support speculative decoding accept-rate degradation as the primary cause.

## Key Findings

- Evalscope attempt 2 completed `2048/2048`, output throughput `12291.66 tok/s`, Avg TTFT `20907.75 ms`, Avg TPOT `23.92 ms`, Avg ITL `101.84 ms`.
- vLLM bench cross-check completed `2048/2048`, output throughput `15152.89 tok/s`, Mean TTFT `7964.35 ms`, Mean TPOT `26.01 ms`, Mean ITL `105.79 ms`.
- Evalscope TPOT/ITL is not worse; the gap is dominated by TTFT/admission and total duration.
- Evalscope reports `Spec. Accept Rate = 0.7707` and `Decoded Tok/Iter = 4.3615`.
- vLLM bench JSON does not contain `spec`, `accept`, or `draft` fields, and the C24 Prometheus window did not capture `vllm:spec_decode_*` metrics.
- Chunk/ITL proxy does not indicate worse evalscope speculative behavior: evalscope is about `4.261` output tokens per ITL event, vLLM bench about `4.070`.
- Evalscope used local `127.0.0.1:30000` through port-forward and sent about `890 MB` of prompt payload for 2048 requests before JSON/HTTP overhead; vLLM bench ran in cluster against the router Service.

## Next Live Check If Needed

Run evalscope in a no-GPU Pod inside the cluster against the router Service, and capture `vllm:spec_decode_*` for both evalscope and vLLM bench measured windows.

## Gate

This analysis does not change the gate. Evalscope Avg gate remains failed, and vLLM bench remains a harness cross-check.
MD

sed -n '1,240p' "${OUT}/metrics-comparison.txt"
sed -n '1,160p' "${OUT}/chunk-granularity.txt"
sed -n '1,220p' "${OUT}/summary.md"
```

**Expected result:** `benchmark-diff-deep-dive-20260701/summary.md`、`metrics-comparison.txt`、`chunk-granularity.txt` 存在。结论必须说明 evalscope 不达标是否由 speculative decoding 接受率导致；若 vLLM bench 没有 spec 指标，必须明确该缺口，不能伪造接受率对比。

## C26B: Live paired evalscope/vLLM benchmark with server-side spec metrics

**When:** 仅当需要 definitive 证明 evalscope 失败是否来自 client path/admission，或需要真实比较 evalscope 与 vLLM bench 的 speculative decoding 接受率时执行。本步骤会重新创建 GPU serving workload，必须重新按 C14 获取 workspace-env GPU permit；每个失败候选后必须按 C21R 重启服务。

**Working directory:** fork-base integration 分支。

```bash
set -euo pipefail

NAMESPACE="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
MONITORING_NAMESPACE="vllm-dsv4-flash-pd-monitoring"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202606302238-openai-devel-cu130"
OUT="artifacts/2026-06-29-vllm-dsv4-flash-pd/live-spec-paired-20260701"
mkdir -p "${OUT}"

cat > "${OUT}/required-live-steps.md" <<'MD'
# C26B Live Paired Benchmark

1. Re-run C14-C19 with the same image and same servingkit-aligned semantics:
   - `prefill.args.noAsyncScheduling=false`
   - `decode.args.maxNumSeqs=96`
   - no `--max-model-len`
   - 1P1D, P/D on different 8-GPU nodes
2. Run local port-forward evalscope BS512/2048 exactly as C24 attempt 2.
3. Restart P/D/router with C21R.
4. Run in-cluster evalscope from a no-GPU Pod, targeting `http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000`.
5. Restart P/D/router with C21R.
6. Run in-cluster `vllm bench serve` from the same image/no-GPU Pod.
7. For every measured window, query running/waiting/output TPS and the spec decode metrics below.
MD

cat > "${OUT}/spec-promql.txt" <<'PROMQL'
sum(rate(vllm:spec_decode_num_accepted_tokens_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s]))
sum(rate(vllm:spec_decode_num_draft_tokens_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s]))
sum(rate(vllm:spec_decode_num_accepted_tokens_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s])), 1)
sum by (index) (rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s]))
sum(rate(vllm:spec_decode_num_drafts_total{stack="vllm",release="dsv4-flash-pd",role="decode"}[30s]))
PROMQL

cat > "${OUT}/in-cluster-evalscope-pod.yaml" <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: evalscope-bench-incluster
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["bash", "-lc"]
      args:
        - |
          set -euo pipefail
          export HTTP_PROXY="http://100.68.170.29:3128"
          export HTTPS_PROXY="http://100.68.170.29:3128"
          export http_proxy="http://100.68.170.29:3128"
          export https_proxy="http://100.68.170.29:3128"
          export NO_PROXY="localhost,127.0.0.1,.svc,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
          export no_proxy="${NO_PROXY}"
          python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'
          evalscope --version
          evalscope perf \
            --url http://${RELEASE}-router.${NAMESPACE}.svc.cluster.local:30000/v1/completions \
            --model deepseek-v4-flash \
            --parallel 512 \
            --number 2048 \
            --dataset random \
            --prefix-length 65536 \
            --min-prompt-length 0 \
            --max-prompt-length 0 \
            --min-tokens 1536 \
            --max-tokens 1536 \
            --seed 42 \
            --extra-args '{"temperature":0,"ignore_eos":true}'
      volumeMounts:
        - name: model
          mountPath: /data01
  volumes:
    - name: model
      hostPath:
        path: /data01
        type: Directory
YAML

echo "C26B scaffolding written to ${OUT}"
echo "Do not apply this pod until C14-C19 have been rerun and a workspace-env permit is granted."
```

**Expected result:** 产生 live pairing 执行脚手架。真正执行后，summary 必须分别给出 local evalscope、in-cluster evalscope、in-cluster vLLM bench 的 Avg TTFT、output throughput、running/waiting、spec accept/draft rate、bad-log scan 和是否仍阻止 M10。若 in-cluster evalscope 接近 vLLM bench 且 spec 指标相近，则根因可确认在本地 port-forward/client path/admission；若 in-cluster evalscope 仍差且 spec accept rate 明显低，再进入 C25 或单独规划模型侧诊断。
2026-07-01 更新：in-cluster evalscope 安装必须使用 `envctl info dev-cluster` 给出的代理 `http://100.68.170.29:3128`，并安装 `evalscope[perf]==1.8.1`，因为无代理安装在 C26B 首次执行中卡依赖下载；代理安装已由 `evalscope-proxy-install-20260701` no-GPU Pod 验证通过。

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
MONITORING_NAMESPACE="vllm-dsv4-flash-pd-monitoring"
MONITORING_RELEASE="dsv4-flash-pd-monitoring"
ARTIFACT_DIR="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
mkdir -p "${ARTIFACT_DIR}/monitoring"

kubectl get pods -n "${NAMESPACE}" -o wide > "${ARTIFACT_DIR}/final-pods.txt" || true
kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp > "${ARTIFACT_DIR}/final-events.txt" || true
kubectl get all -n "${MONITORING_NAMESPACE}" -o wide > "${ARTIFACT_DIR}/monitoring/final-monitoring.txt" || true

if [ ! -f "${ARTIFACT_DIR}/summary.md" ]; then
cat > "${ARTIFACT_DIR}/summary.md" <<'MD'
# vLLM DSV4 Flash P/D Benchmark Summary

## Target

- Environment: dev-cluster
- Namespace: vllm-dsv4-flash-pd
- Release: dsv4-flash-pd
- Model: deepseek-v4-flash

## Runs

- 64k input / 1 output TTFT: see `evalscope-ttft-64k-1out.log`
- cache-hit decode BS512 / 1.5k output throughput: see `evalscope-decode-bs512-cache-hit-1p5kout-n2048.log`
- fallback BS sweep, if BS512 is invalid: see `decode-bs-sweep-cache-hit-1p5kout.summary.tsv`
- servingkit monitoring chart: see `monitoring/`
- running BS evidence: see `monitoring/running-bs-bs*.summary.txt`
- vLLM built-in benchmark comparison, if run: see `vllm-bench-compare/`

## Performance Gate

- Gate uses Avg metrics only; P50/P95/P99 are archived but do not block.
- 64k/1 Avg TTFT must be < 10s.
- BS512/1.5k evalscope overall output throughput must be >= 14000 tokens/s.
- BS512 measured run uses 2048 total requests, i.e. `4 * BS`; if BS512 is invalid, fallback BS sweep also uses `number = 4 * BS` for each candidate between 128 and 512.
- Before testing a higher BS, Prometheus `max_decode_running` must show the service can actually reach the target running BS; do not run candidates higher than observed capacity without explaining the admission/routing limit.
- Deployment shape is 1P1D, so router-path output throughput is produced by one decode node, not aggregated across multiple decode replicas.
- If either threshold is not met, do not update remote `iaas_main`.

## Artifacts

Raw artifacts are in this directory.
MD
fi

helm uninstall "${RELEASE}" -n "${NAMESPACE}" || true
kubectl delete namespace "${NAMESPACE}" --ignore-not-found
helm uninstall "${MONITORING_RELEASE}" -n "${MONITORING_NAMESPACE}" || true
kubectl delete namespace "${MONITORING_NAMESPACE}" --ignore-not-found

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

SESSION_ID="$(grep '^SESSION_ID=' "${ARTIFACT_DIR}/c14-session.env" | cut -d= -f2- || true)"
if [ -n "${SESSION_ID}" ]; then
  python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
    resource-upsert \
    --session-id "${SESSION_ID}" \
    --environment "${ENVIRONMENT}" \
    --namespace "${NAMESPACE}" \
    --kind HelmRelease \
    --name "${RELEASE}" \
    --gpu 16 \
    --status released \
    --purpose vllm-dsv4-flash-pd \
    --release-condition "released after benchmark artifacts and cleanup completed" \
    --cleanup-command "eval \"\$(${ENV_ROOT}/bin/envctl use ${ENVIRONMENT})\"; helm uninstall ${RELEASE} -n ${NAMESPACE} || true; kubectl delete namespace ${NAMESPACE} --ignore-not-found"
fi
```

**Expected result:** summary.md exists；temporary serving Helm release, benchmark namespace, monitoring Helm release, monitoring namespace, and router/Prometheus port-forward are cleaned unless intentionally retained and reported；GPU permit is released by permit id from `gpu-permit.json`；summary 明确记录 servingkit monitoring chart、running BS evidence、vLLM benchmark comparison outcome, and whether BS512 gate passed.

## C26C: evalscope proxy BS 降档 sweep 命令

**When to run:** 当 BS512 evalscope 不达标或 invalid 后，需要在 `128-512` 范围内继续寻找可通过 Avg gate 的候选，并且必须避免本地 port-forward/client path 影响。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 在 no-GPU benchmark Pod 内显式通过代理安装 `evalscope[perf]==1.8.1`，直连 in-cluster router Service，按 `number = 4 * BS` 跑 64K prefix / 1536 output；每个失败候选后重启 P/D/router，再进入下一候选。

**Proxy install command used in each Pod:**

```bash
python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'
evalscope --version
```

**Benchmark command template inside Pod:**

```bash
BS=400
NUM=$((BS * 4))
OUT="/tmp/evalscope-bs${BS}/formal"

evalscope perf \
  --url http://dsv4-flash-pd-router.vllm-dsv4-flash-pd.svc.cluster.local:30000/v1/completions \
  --model deepseek-v4-flash \
  --tokenizer-path /data01/DeepSeek-V4-Flash \
  --parallel "${BS}" \
  --number "${NUM}" \
  --dataset random \
  --prefix-length 65536 \
  --min-prompt-length 0 \
  --max-prompt-length 0 \
  --min-tokens 1536 \
  --max-tokens 1536 \
  --seed 42 \
  --extra-args '{"temperature":0,"ignore_eos":true}' \
  --outputs "${OUT}"
```

**Candidate values used:** `BS=400` with `NUM=1600`，`BS=256` with `NUM=1024`，`BS=128` with `NUM=512`。

**Expected result:** Pod log shows `evalscope 1.8.1`；formal run exits `0`；summary reports `Total / Success / Failed` and Avg `Output Throughput (tok/s)`；Prometheus processing-window evidence records decode running/waiting/TPS and `vllm:spec_decode_*` rates. Gate passes only if Avg output throughput is `>= 14000 tok/s` for the candidate and required TTFT gate remains satisfied.

## C26C: restart after failed candidate

**When to run:** 每个候选 BS 未通过 Avg gate、出现 Mooncake/KV bad state、或 benchmark 被中断后，下一候选开始前必须执行。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 删除当前 P/D/router Pod，让 Helm/StormService 重新拉起同一配置，避免把 router queue、Mooncake/KV 或 worker 状态污染带入下一轮。

```bash
set -euo pipefail
ENVCTL=/data00/home/hanhan.hank/workspace/env/bin/envctl
NS=vllm-dsv4-flash-pd

"${ENVCTL}" kubectl dev-cluster get pods -n "${NS}" -o name \
  | grep -E 'roleset|router' \
  | xargs -r "${ENVCTL}" kubectl dev-cluster delete -n "${NS}"

"${ENVCTL}" kubectl dev-cluster wait pod -n "${NS}" \
  -l app.kubernetes.io/instance=dsv4-flash-pd \
  --for=condition=Ready \
  --timeout=1800s
```

**Expected result:** prefill、decode、router pod 均重新 `Ready`；router `/v1/models` 和一次小 completion smoke 成功后才可继续下一组 benchmark。

## C26C: final cleanup and permit release

**When to run:** C26C sweep 完成、失败、或用户要求停止时。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 清理临时 serving、monitoring、benchmark Pod、namespace、port-forward，并释放本轮 GPU permit。

```bash
set -euo pipefail
ENVCTL=/data00/home/hanhan.hank/workspace/env/bin/envctl
ENV_ROOT=/data00/home/hanhan.hank/workspace/env
SKILL_DIR=/data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env
NS=vllm-dsv4-flash-pd
MNS=vllm-dsv4-flash-pd-monitoring
RELEASE=dsv4-flash-pd
MREL=dsv4-flash-pd-monitoring
PERMIT_ID=e927012a-ff9e-4626-a769-d80bc8cac77f

eval "$("${ENVCTL}" use dev-cluster)"
helm uninstall "${RELEASE}" -n "${NS}" || true
helm uninstall "${MREL}" -n "${MNS}" || true
kubectl delete pod -n "${NS}" evalscope-bs128-proxy --ignore-not-found || true
kubectl delete namespace "${NS}" --ignore-not-found
kubectl delete namespace "${MNS}" --ignore-not-found
kubectl wait --for=delete namespace/"${NS}" --timeout=180s || true
kubectl wait --for=delete namespace/"${MNS}" --timeout=180s || true

python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  permit-release --permit-id "${PERMIT_ID}"

"${ENVCTL}" kubectl dev-cluster get ns "${NS}" "${MNS}" || true
ps -eo pid,ppid,cmd | grep -E 'port-forward.*(vllm-dsv4-flash-pd|19091|18082)|evalscope-bs(128|256|400)' | grep -v grep || true
```

**Expected result:** both namespaces return `NotFound`；no task `port-forward` or `evalscope-bs*` process remains；permit status is `released` and does not appear in granted/running permit list.

## C27: decode 场景 evalscope TTFT 高于 vLLM bench 的专项离线分析

**When to run:** 用户要求解释 decode 场景下 evalscope Avg TTFT 为什么高于 `vllm bench serve`；应在 C25/P87 已完成、且重新部署前优先执行。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Purpose:** 只使用已有 artifacts 归因 TTFT 差异，避免把 C24/C25/C26B/C26C 的 throughput 结论误当成 TTFT 解释。该步骤不创建 GPU workload、不修改源码、不更新 `iaas_main`。

```bash
set -euo pipefail

ROOT="$PWD"
ART="${ROOT}/artifacts/2026-06-29-vllm-dsv4-flash-pd"
OUT="${ART}/decode-ttft-divergence-analysis-20260701"
mkdir -p "${OUT}"

{
  echo "# Decode TTFT Divergence Evidence"
  echo
  echo "## Known artifact roots"
  printf -- "- C24 live diagnosis: %s\n" "${ART}/live-mooncake-diagnosis-20260701"
  printf -- "- C26 offline deep dive: %s\n" "${ART}/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701"
  printf -- "- C26B in-cluster proxy rerun: %s\n" "${ART}/live-spec-paired-20260701/proxy-rerun-186-154"
  printf -- "- C26C evalscope BS sweep: %s\n" "${ART}/evalscope-bs-downgrade-sweep-20260701"
  printf -- "- C25 debug retest: %s\n" "${ART}/c25-debug-retest-20260701"
  printf -- "- vLLM bench BS sweep: %s\n" "${ART}/vllm-bench-bs-sweep-20260701"
  echo
  echo "## TTFT / throughput / request generation excerpts"
  rg -n --no-heading \
    "Avg TTFT|Mean TTFT|Request generation|request generation|first 512|last 512|Output Throughput|output throughput|Total / Success / Failed|Successful requests|Failed requests|duration|Duration" \
    "${ART}/live-mooncake-diagnosis-20260701/summary.md" \
    "${ART}/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/summary.md" \
    "${ART}/live-spec-paired-20260701/proxy-rerun-186-154/runs/incluster-evalscope-bs512-proxy/summary.md" \
    "${ART}/evalscope-bs-downgrade-sweep-20260701/summary.md" \
    "${ART}/c25-debug-retest-20260701/runs/incluster-evalscope-bs512-proxy/pod.log" \
    "${ART}/c25-debug-retest-20260701/runs/incluster-evalscope-bs512-proxy/pod-key-summary.txt" \
    "${ART}/vllm-bench-bs-sweep-20260701/summary.md" \
    2>/dev/null || true
  echo
  echo "## Prometheus running/waiting/TPS/spec excerpts"
  rg -n --no-heading \
    "running|waiting|TPS|tps|generation|Spec|Accept|Decoded Tok|spec_decode|num_requests" \
    "${ART}/live-mooncake-diagnosis-20260701/summary.md" \
    "${ART}/live-mooncake-diagnosis-20260701/benchmark-diff-deep-dive-20260701/summary.md" \
    "${ART}/live-spec-paired-20260701/proxy-rerun-186-154/runs/incluster-evalscope-bs512-proxy/summary.md" \
    "${ART}/evalscope-bs-downgrade-sweep-20260701/summary.md" \
    "${ART}/c25-debug-retest-20260701/runs/incluster-evalscope-bs512-proxy/monitoring" \
    "${ART}/vllm-bench-bs-sweep-20260701/summary.md" \
    2>/dev/null || true
  echo
  echo "## Mooncake/RDMA/timeout excerpts"
  rg -n --no-heading \
    "Mooncake|RDMA|descriptor|timed out|timeout|transfer engine returned|producer|ret=-1|prefill connection|connection closed|Failed to open device|Local segment descriptor" \
    "${ART}/live-mooncake-diagnosis-20260701/summary.md" \
    "${ART}/mooncake-failure-diagnosis-20260701/summary.md" \
    "${ART}/live-spec-paired-20260701/proxy-rerun-186-154/runs/incluster-evalscope-bs512-proxy/summary.md" \
    "${ART}/evalscope-bs-downgrade-sweep-20260701/summary.md" \
    "${ART}/c25-debug-retest-20260701/server-logs" \
    "${ART}/c25-debug-retest-20260701/runs/incluster-evalscope-bs512-proxy/pod.log" \
    2>/dev/null || true
} > "${OUT}/ttft-excerpts.txt"

cat > "${OUT}/hypothesis-matrix.md" <<'MD'
# Decode TTFT Divergence Hypothesis Matrix

| 假设 | 需要检查的证据 | 当前判定 | 证据路径 |
| --- | --- | --- | --- |
| evalscope request generation 被计入 TTFT | evalscope log 的 request generation 阶段、processing 阶段开始时间、TTFT 定义 | 执行后填写：confirmed / likely / inconclusive | `ttft-excerpts.txt` |
| evalscope 首批请求 TTFT 更高并拉高 Avg | first 512 TTFT、last 512 TTFT、request start span | 执行后填写 | `ttft-excerpts.txt` |
| evalscope 无法稳定维持 decode running BS512 | Prometheus running avg/max、waiting avg/max、output TPS | 执行后填写 | `ttft-excerpts.txt` |
| admission/waiting 比 vLLM bench 更重 | `vllm:num_requests_waiting`、router/server queue 证据 | 执行后填写 | `ttft-excerpts.txt` |
| TPOT/ITL 或 decode token 生成速度是主因 | TPOT、ITL、server-side generation TPS、output tokens | 执行后填写 | `ttft-excerpts.txt` |
| 投机解码接受率差异导致 TTFT/throughput 差异 | evalscope `Spec. Accept Rate`、`Decoded Tok/Iter`、server-side `vllm:spec_decode_*` | 执行后填写 | `ttft-excerpts.txt` |
| local port-forward/client path 是主因 | local vs in-cluster evalscope、in-cluster vLLM bench、NO_PROXY/Service path | 执行后填写 | `ttft-excerpts.txt` |
| HTTP streaming/connection reuse/statistics semantics 差异 | evalscope/vLLM bench client implementation、failure inclusion、TTFT calculation | 执行后填写 | source inspection + `ttft-excerpts.txt` |
| Mooncake/RDMA timeout 或 descriptor pressure 放大 TTFT | bad-log counts、producer timeout、descriptor/bytes、RDMA init warning | 执行后填写 | `ttft-excerpts.txt` |
MD

cat > "${OUT}/summary.md" <<'MD'
# Decode 场景 evalscope TTFT 高于 vLLM bench 的专项分析

## 问题

解释在 64K prefix / 1536 output / BS512 decode 场景下，为什么 evalscope 的 Avg TTFT 明显高于 `vllm bench serve`。

## 必须对齐的事实

- 历史执行时 gate 仍使用 evalscope Avg/Overall；2026-07-02 用户已修改当前 gate，接受 vLLM benchmark 作为发布性能 gate。
- `1P1D` 输出来自单台 decode 节点。
- 当前部署语义保持 `prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`。
- C27 默认离线分析，不创建 GPU workload。

## 需要填写的结论

### Confirmed

- 执行后填写。

### Likely

- 执行后填写。

### Inconclusive / Evidence Gap

- 执行后填写。

## 必须覆盖的检查项

- evalscope request generation 是否进入 TTFT 统计窗口。
- C24 first/last 512 request TTFT 是否显示 evalscope 在早期 admission 更慢。
- evalscope 与 vLLM bench 的 decode running avg/max、waiting avg/max、server-side output TPS 是否一致。
- TPOT/ITL 是否支持“decode token 生成慢”这个解释。
- speculative decoding 接受率是否有 server-side 证据支持差异。
- in-cluster evalscope 是否仍高 TTFT，用于排除纯本地 port-forward/client path。
- 失败请求、timeout、Mooncake/RDMA descriptor 或 producer timeout 是否进入或放大 TTFT。
- evalscope 和 vLLM bench 的 TTFT/statistics 定义是否不同。

## 初始证据摘录

见 `ttft-excerpts.txt` 和 `hypothesis-matrix.md`。

## 发布判断

C27 执行时不改变 M10；2026-07-02 用户已修改 gate，当前可按用户接受的 vLLM benchmark gate 推进 M10 发布前收口。
MD

echo "C27 offline analysis scaffold written to ${OUT}"
echo "Next: fill ${OUT}/summary.md and ${OUT}/hypothesis-matrix.md from ${OUT}/ttft-excerpts.txt plus targeted source inspection if needed."
```

**Expected result:** 创建 `artifacts/2026-06-29-vllm-dsv4-flash-pd/decode-ttft-divergence-analysis-20260701/`，包含 `ttft-excerpts.txt`、`hypothesis-matrix.md`、`summary.md`。最终填写后的 summary 必须明确哪些因素是 confirmed、likely、inconclusive。预期优先验证的方向是：evalscope 的更高 Avg TTFT 主要来自 admission/running-BS/client request scheduling 和统计窗口差异，而不是 TPOT/ITL 变差；投机解码接受率目前只有 evalscope 与部分 server-side 代理证据，不能在没有同窗口 vLLM bench `vllm:spec_decode_*` 的情况下宣称完全等价或完全不同。

## C28A: live TTFT correlation preflight and GPU permit

**When to run:** C27 离线分析完成后，仍需要同窗口 live 指标证明 evalscope TTFT 高于 `vllm bench serve` 的根因时执行。该步骤只做 preflight 和 workspace-env permit；如果 permit queued，不创建 GPU workload。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

**Benchmark goal and monitoring plan:** 重新部署同一 `1P1D` 服务，使用 C25 调试镜像、相同 P/D/router 语义、相同 64K prefix / 1536 output / BS512 / 2048 requests。先部署 servingkit monitoring chart，确认 prefill/decode scrape `up=1`，然后分别运行 in-cluster evalscope 和 in-cluster `vllm bench serve`，每个 measured window 保存 Prometheus running/waiting/output TPS、`vllm:spec_decode_*`、vLLM TTFT/TPOT 指标、bad-log 和 client timeline。该 run 执行时不改变 M10 gate、不更新 `iaas_main`；2026-07-02 用户已在 C28 完成后接受 vLLM benchmark 作为当前发布 gate。

```bash
set -euo pipefail

ROOT="$PWD"
ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
MNS="vllm-dsv4-flash-pd-monitoring"
RELEASE="dsv4-flash-pd"
MRELEASE="dsv4-flash-pd-monitoring"
GPU_TOTAL=16
THREAD_ID="019f02ab-92f4-73f3-870b-5f981a254020"
SESSION_ID="${SESSION_ID:-codex-vllm-c28-ttft-$(date '+%Y%m%d-%H%M%S')}"
SKILL_DIR="/data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env"
ART="${ROOT}/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
mkdir -p "${ART}/preflight"

"${ENV_ROOT}/bin/envctl" info "${ENVIRONMENT}" | tee "${ART}/preflight/envctl-info.txt"
"${ENV_ROOT}/bin/envctl" validate "${ENVIRONMENT}" | tee "${ART}/preflight/envctl-validate.txt"

python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" init
python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  session-start \
  --session-id "${SESSION_ID}" \
  --thread-id "${THREAD_ID}" \
  --owner codex \
  --task "vllm dsv4 C28 live TTFT correlation" \
  --cwd "${ROOT}" \
  | tee "${ART}/preflight/session-start.json"

python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  active --environment "${ENVIRONMENT}" --namespace "${NS}" \
  | tee "${ART}/preflight/active-resources.json"
python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  permit-list --environment "${ENVIRONMENT}" --namespace "${NS}" --active \
  | tee "${ART}/preflight/active-permits.json"

"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get nodes -o wide \
  | tee "${ART}/preflight/nodes-wide.txt"
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get pods -A -o wide \
  | tee "${ART}/preflight/pods-all-wide.txt"

CLEANUP_COMMAND="helm -n ${NS} uninstall ${RELEASE} --ignore-not-found; helm -n ${MNS} uninstall ${MRELEASE} --ignore-not-found; kubectl delete ns ${NS} ${MNS} --ignore-not-found"
python3 "${SKILL_DIR}/scripts/resource_registry.py" --env-root "${ENV_ROOT}" \
  permit-acquire \
  --session-id "${SESSION_ID}" \
  --thread-id "${THREAD_ID}" \
  --environment "${ENVIRONMENT}" \
  --namespace "${NS}" \
  --gpu "${GPU_TOTAL}" \
  --purpose "vllm-dsv4-c28-live-ttft-correlation" \
  --release-condition "C28 paired evalscope/vllm benchmark evidence collected and resources cleaned" \
  --cleanup-command "${CLEANUP_COMMAND}" \
  --wait-seconds 180 \
  | tee "${ART}/preflight/permit-acquire.json"

python3 - "${ART}/preflight/permit-acquire.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
status = data.get("status")
print(f"PERMIT_STATUS={status}")
print(f"PERMIT_ID={data.get('permit_id')}")
if status not in {"granted", "running"}:
    raise SystemExit(f"permit not usable for GPU workload: {status}")
PY
```

**Expected result:** `permit-acquire.json` status 为 `granted` 或 `running` 才能继续 C28B；若为 `queued`、`blocked`、`denied`、`expired`、`released`、`failed` 或 `stale`，不得创建 GPU workload，只记录等待/阻塞状态。

## C28B: deploy C25 debug image with unchanged 1P1D semantics

**When to run:** C28A permit status 为 `granted` 或 `running`。执行前必须填写 `PREFILL_NODE` 和 `DECODE_NODE`，二者必须不同且各自可调度 8 GPU；推荐先复用最近成功组合 `PREFILL_NODE=192.168.1.186`、`DECODE_NODE=192.168.1.154`，但执行时以 C28A live node 状态为准。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

ROOT="$PWD"
ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
SESSION_ID="${SESSION_ID:?reuse SESSION_ID from C28A}"
PERMIT_ID="${PERMIT_ID:?set PERMIT_ID from C28A permit-acquire.json}"
PREFILL_NODE="${PREFILL_NODE:?set explicit prefill node}"
DECODE_NODE="${DECODE_NODE:?set explicit decode node}"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202607011829-openai-devel-cu130"
ART="${ROOT}/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
RENDER="${ART}/render/rendered.yaml"
mkdir -p "${ART}/render" "${ART}/deploy" "${ART}/smoke"

if [ "${PREFILL_NODE}" = "${DECODE_NODE}" ]; then
  echo "PREFILL_NODE and DECODE_NODE must differ for 1P1D full-node deployment" >&2
  exit 2
fi

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
python3 /data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env/scripts/resource_registry.py --env-root "${ENV_ROOT}" \
  permit-running --permit-id "${PERMIT_ID}" | tee "${ART}/deploy/permit-running.json"
python3 /data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env/scripts/resource_registry.py --env-root "${ENV_ROOT}" \
  resource-upsert \
  --session-id "${SESSION_ID}" \
  --environment "${ENVIRONMENT}" \
  --namespace "${NS}" \
  --kind HelmRelease \
  --name "${RELEASE}" \
  --gpu 16 \
  --status create \
  --purpose "vllm-dsv4-c28-live-ttft-correlation" \
  --release-condition "C28 paired evalscope/vllm benchmark evidence collected and resources cleaned" \
  --cleanup-command "helm -n ${NS} uninstall ${RELEASE} --ignore-not-found; kubectl delete ns ${NS} --ignore-not-found" \
  | tee "${ART}/deploy/resource-create-helmrelease.json"

kubectl get node "${PREFILL_NODE}" "${DECODE_NODE}" -o wide | tee "${ART}/deploy/selected-nodes.txt"

helm template "${RELEASE}" examples/deployment/deepseek-v4-flash-pd \
  -n "${NS}" \
  --set global.image="${IMAGE}" \
  --set workspaceEnv.sessionId="${SESSION_ID}" \
  --set prefill.nodeAffinity.values[0]="${PREFILL_NODE}" \
  --set decode.nodeAffinity.values[0]="${DECODE_NODE}" \
  --set router.nodeAffinity.values[0]="${DECODE_NODE}" \
  --set prefill.extraEnv.VLLM_DSV4_MOONCAKE_DIAG="1" \
  --set decode.extraEnv.VLLM_DSV4_MOONCAKE_DIAG="1" \
  > "${RENDER}"

grep -n "VLLM_DSV4_MOONCAKE_DIAG\\|--max-num-seqs\\|--max-model-len\\|--no-async-scheduling\\|pip install\\|git clone\\|apt install" "${RENDER}" \
  | tee "${ART}/render/render-scan.txt" || true
if grep -n -- "--max-model-len\\|--no-async-scheduling\\|pip install\\|git clone\\|apt install" "${RENDER}"; then
  echo "render contains forbidden runtime or semantic drift pattern" >&2
  exit 3
fi
grep -Eq -- '"--max-num-seqs"[[:space:]]+"96"|--max-num-seqs[[:space:]]+"96"' "${RENDER}"
grep -q "VLLM_DSV4_MOONCAKE_DIAG" "${RENDER}"

kubectl create namespace "${NS}" --dry-run=client -o yaml | kubectl apply -f - | tee "${ART}/deploy/namespace.txt"
helm upgrade --install "${RELEASE}" examples/deployment/deepseek-v4-flash-pd \
  -n "${NS}" \
  --set global.image="${IMAGE}" \
  --set workspaceEnv.sessionId="${SESSION_ID}" \
  --set prefill.nodeAffinity.values[0]="${PREFILL_NODE}" \
  --set decode.nodeAffinity.values[0]="${DECODE_NODE}" \
  --set router.nodeAffinity.values[0]="${DECODE_NODE}" \
  --set prefill.extraEnv.VLLM_DSV4_MOONCAKE_DIAG="1" \
  --set decode.extraEnv.VLLM_DSV4_MOONCAKE_DIAG="1" \
  --wait --timeout 30m \
  | tee "${ART}/deploy/helm-upgrade.txt"
python3 /data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env/scripts/resource_registry.py --env-root "${ENV_ROOT}" \
  resource-upsert \
  --session-id "${SESSION_ID}" \
  --environment "${ENVIRONMENT}" \
  --namespace "${NS}" \
  --kind HelmRelease \
  --name "${RELEASE}" \
  --gpu 16 \
  --status running \
  --purpose "vllm-dsv4-c28-live-ttft-correlation" \
  --release-condition "C28 paired evalscope/vllm benchmark evidence collected and resources cleaned" \
  --cleanup-command "helm -n ${NS} uninstall ${RELEASE} --ignore-not-found; kubectl delete ns ${NS} --ignore-not-found" \
  | tee "${ART}/deploy/resource-running-helmrelease.json"

kubectl get pods -n "${NS}" -o wide | tee "${ART}/deploy/pods-wide.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=prefill --timeout=900s | tee "${ART}/deploy/prefill-ready.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=decode --timeout=900s | tee "${ART}/deploy/decode-ready.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --timeout=300s | tee "${ART}/deploy/router-ready.txt"

kubectl port-forward -n "${NS}" "svc/${RELEASE}-router" 30000:30000 > "${ART}/smoke/router-port-forward.log" 2>&1 &
pf_pid=$!
trap 'kill "${pf_pid}" >/dev/null 2>&1 || true; wait "${pf_pid}" >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do
  curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null && break
  sleep 1
done
curl -fsS http://127.0.0.1:30000/v1/models | tee "${ART}/smoke/models.json"
curl -fsS http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"hello","max_tokens":4,"temperature":0}' \
  | tee "${ART}/smoke/completion.json"
kill "${pf_pid}" >/dev/null 2>&1 || true
wait "${pf_pid}" >/dev/null 2>&1 || true
trap - EXIT
```

**Expected result:** Rendered command still has `decode.args.maxNumSeqs=96` and no `--max-model-len` / runtime install / hotfix; P/D/router Ready；router `/v1/models` and `/v1/completions` succeed through real router path；`VLLM_DSV4_MOONCAKE_DIAG=1` appears only as diagnostic env.

## C28C: paired in-cluster evalscope and vLLM bench with Prometheus windows

**When to run:** C28B smoke succeeds。该步骤会创建 no-GPU benchmark Pods；每个 measured command 必须打印 start/end timestamps 和 exit code，输出写入 artifact。若 evalscope run 失败或中断，必须先执行 C28R 重启服务，再运行 vLLM bench。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

ROOT="$PWD"
ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
MNS="vllm-dsv4-flash-pd-monitoring"
RELEASE="dsv4-flash-pd"
MRELEASE="dsv4-flash-pd-monitoring"
IMAGE="iaas-gpu-cn-beijing.cr.volces.com/serving/vllm:v0.10.0.iaas.dev.202607011829-openai-devel-cu130"
ART="${ROOT}/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
mkdir -p "${ART}/runs" "${ART}/monitoring"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

# Monitoring: use the same servingkit chart flow as C18.
SERVINGKIT_REPO="/data00/home/hanhan.hank/workspace/servingkit"
SERVINGKIT_REF="origin/hanhan_dev"
MON_DIR="${ART}/monitoring/chart"
MON_CHART="${MON_DIR}/llm-serving-monitoring"
MON_VALUES="${ART}/monitoring/values-c28.yaml"
rm -rf "${MON_CHART}"
mkdir -p "${MON_DIR}"
git -C "${SERVINGKIT_REPO}" archive --format=tar "${SERVINGKIT_REF}" llm-serving-monitoring \
  | tar -C "${MON_DIR}" -xf -
cat > "${MON_VALUES}" <<YAML
namespace:
  create: false
  name: ${MNS}

prometheus:
  enabled: true
  scrapeInterval: 1s
  scrapeTimeout: 900ms
  externalLabels:
    cluster: ${ENVIRONMENT}
  nodeAffinity:
    enabled: true
    key: kubernetes.io/hostname
    values:
      - "192.168.1.149"

grafana:
  enabled: false

nodeExporter:
  enabled: false

scrapeTargets:
  - name: ${RELEASE}-prefill
    enabled: true
    metricsPath: /metrics
    targets:
      - address: ${RELEASE}-prefill.${NS}.svc.cluster.local:8000
        labels:
          stack: vllm
          release: ${RELEASE}
          role: prefill
          model: deepseek-v4-flash
  - name: ${RELEASE}-decode
    enabled: true
    metricsPath: /metrics
    targets:
      - address: ${RELEASE}-decode.${NS}.svc.cluster.local:8001
        labels:
          stack: vllm
          release: ${RELEASE}
          role: decode
          model: deepseek-v4-flash
YAML
kubectl create namespace "${MNS}" --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install "${MRELEASE}" "${MON_CHART}" -n "${MNS}" \
  -f "${MON_VALUES}" \
  --wait --timeout 10m \
  | tee "${ART}/monitoring/helm-upgrade.txt"

PROM_POD="$(kubectl get pod -n "${MNS}" -l app.kubernetes.io/component=prometheus -o jsonpath='{.items[0].metadata.name}')"
kubectl wait -n "${MNS}" --for=condition=Ready "pod/${PROM_POD}" --timeout=10m
kubectl get pod -n "${MNS}" -o wide | tee "${ART}/monitoring/pods-wide.txt"

write_query() {
  local out="$1"
  local query="$2"
  kubectl exec -n "${MNS}" "${PROM_POD}" -- wget -qO- \
    "http://127.0.0.1:9090/api/v1/query?query=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "${query}")" \
    > "${out}"
}
write_query "${ART}/monitoring/up.json" "up"
write_query "${ART}/monitoring/vllm-series-count.json" "count({__name__=~\"vllm:.*\"})"

run_evalscope_pod() {
  local name="evalscope-c28-bs512"
  local dir="${ART}/runs/${name}"
  mkdir -p "${dir}"
  cat > "${dir}/pod.yaml" <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${name}
  namespace: ${NS}
  labels:
    workspace-env/owner: codex
    workspace-env/purpose: vllm-c28-ttft-correlation
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: "192.168.1.149"
  containers:
    - name: bench
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      env:
        - name: HTTP_PROXY
          value: http://100.68.170.29:3128
        - name: HTTPS_PROXY
          value: http://100.68.170.29:3128
        - name: NO_PROXY
          value: localhost,127.0.0.1,.svc,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
        - name: no_proxy
          value: localhost,127.0.0.1,.svc,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
      command: ["bash", "-lc"]
      volumeMounts:
        - name: models
          mountPath: /data01
      args:
        - |
          set -euo pipefail
          python3 -m pip install --proxy http://100.68.170.29:3128 -U 'evalscope[perf]==1.8.1'
          evalscope --version
          run_start=\$(date '+%Y-%m-%dT%H:%M:%S%z')
          echo "MEASURED_RUN_START=\${run_start}"
          set +e
          evalscope perf \
            --parallel 512 \
            --number 2048 \
            --model deepseek-v4-flash \
            --url http://${RELEASE}-router.${NS}.svc.cluster.local:30000/v1/completions \
            --api openai \
            --dataset random \
            --prefix-length 65536 \
            --min-prompt-length 0 \
            --max-prompt-length 0 \
            --min-tokens 1536 \
            --max-tokens 1536 \
            --tokenizer-path /data01/DeepSeek-V4-Flash \
            --seed 42 \
            --extra-args '{"temperature":0,"ignore_eos":true}' \
            --outputs /tmp/evalscope-c28-bs512
          status=\$?
          set -e
          run_end=\$(date '+%Y-%m-%dT%H:%M:%S%z')
          echo "MEASURED_RUN_END=\${run_end}"
          echo "MEASURED_RUN_EXIT_CODE=\${status}"
          exit "\${status}"
  volumes:
    - name: models
      hostPath:
        path: /data01
        type: DirectoryOrCreate
YAML
  kubectl delete pod -n "${NS}" "${name}" --ignore-not-found
  kubectl apply -f "${dir}/pod.yaml"
  kubectl wait -n "${NS}" --for=condition=Ready "pod/${name}" --timeout=300s || true
  kubectl logs -n "${NS}" -f "pod/${name}" | tee "${dir}/pod.log"
  kubectl get pod -n "${NS}" "${name}" -o yaml > "${dir}/pod-final.yaml" || true
}

run_vllm_bench_pod() {
  local name="vllm-bench-c28-bs512"
  local dir="${ART}/runs/${name}"
  mkdir -p "${dir}"
  cat > "${dir}/pod.yaml" <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${name}
  namespace: ${NS}
  labels:
    workspace-env/owner: codex
    workspace-env/purpose: vllm-c28-ttft-correlation
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: "192.168.1.149"
  containers:
    - name: bench
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["bash", "-lc"]
      volumeMounts:
        - name: models
          mountPath: /data01
      args:
        - |
          set -euo pipefail
          vllm bench serve --help >/tmp/vllm-bench-help.txt
          run_start=\$(date '+%Y-%m-%dT%H:%M:%S%z')
          echo "MEASURED_RUN_START=\${run_start}"
          set +e
          vllm bench serve \
            --backend openai \
            --base-url http://${RELEASE}-router.${NS}.svc.cluster.local:30000 \
            --endpoint /v1/completions \
            --model deepseek-v4-flash \
            --tokenizer /data01/DeepSeek-V4-Flash \
            --dataset-name random \
            --num-prompts 2048 \
            --request-rate inf \
            --max-concurrency 512 \
            --random-prefix-len 65536 \
            --random-input-len 0 \
            --random-output-len 1536 \
            --ignore-eos \
            --temperature 0 \
            --seed 42 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --save-detailed \
            --result-dir /tmp/vllm-bench-c28-bs512 \
            --result-filename result.json \
            --disable-tqdm
          status=\$?
          set -e
          run_end=\$(date '+%Y-%m-%dT%H:%M:%S%z')
          echo "MEASURED_RUN_END=\${run_end}"
          echo "MEASURED_RUN_EXIT_CODE=\${status}"
          exit "\${status}"
  volumes:
    - name: models
      hostPath:
        path: /data01
        type: DirectoryOrCreate
YAML
  kubectl delete pod -n "${NS}" "${name}" --ignore-not-found
  kubectl apply -f "${dir}/pod.yaml"
  kubectl wait -n "${NS}" --for=condition=Ready "pod/${name}" --timeout=300s || true
  kubectl logs -n "${NS}" -f "pod/${name}" | tee "${dir}/pod.log"
  kubectl get pod -n "${NS}" "${name}" -o yaml > "${dir}/pod-final.yaml" || true
}

run_evalscope_pod
# If evalscope produced timeout/Mooncake failure, run C28R before vLLM bench.
run_vllm_bench_pod
```

**Expected result:** 保存两个 in-cluster no-GPU benchmark Pod 的 YAML、日志、终态 YAML；evalscope 使用同一 C25 调试镜像作为 client image，通过代理安装 `evalscope[perf]==1.8.1`，请求集群内 router Service 且 `NO_PROXY` 保证服务流量不走代理；vLLM bench 使用同一 C25 调试镜像；二者均使用 BS512、2048 requests、64K prefix、1536 output。执行后必须按 C28M 采集每个 measured window 的 Prometheus 指标并写 summary。若需要 first/last request TTFT 或 DB/JSON per-request artifact，不能把输出只写入 `/tmp` 后让 Pod 直接 `Succeeded`；应使用 hostPath 持久化输出或在命令末尾 sleep，再复制所需小文件。

## C28M: collect measured-window Prometheus, logs, and summary

**When to run:** C28C 的 evalscope/vLLM bench Pod 到达 terminal 后立即执行；如果某个 Pod 未产生 end timestamp，则用 Pod start/end time和日志中的最后时间作为 invalid window 证据。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

ROOT="$PWD"
ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
MNS="vllm-dsv4-flash-pd-monitoring"
RELEASE="dsv4-flash-pd"
ART="${ROOT}/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
mkdir -p "${ART}/monitoring/windows" "${ART}/server-logs"

eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"
PROM_POD="$(kubectl get pod -n "${MNS}" -l app.kubernetes.io/component=prometheus -o jsonpath='{.items[0].metadata.name}')"

extract_window() {
  local log="$1"
  local out="$2"
  local start end
  start="$(grep -m1 '^MEASURED_RUN_START=' "${log}" | cut -d= -f2- || true)"
  end="$(grep -m1 '^MEASURED_RUN_END=' "${log}" | cut -d= -f2- || true)"
  if [ -z "${start}" ]; then start="$(date -Is)"; fi
  if [ -z "${end}" ]; then end="$(date -Is)"; fi
  {
    echo "start=${start}"
    echo "end=${end}"
    echo "start_epoch=$(date -d "${start}" +%s)"
    echo "end_epoch=$(date -d "${end}" +%s)"
  } > "${out}"
}

query_range() {
  local out="$1"
  local query="$2"
  local start="$3"
  local end="$4"
  kubectl exec -n "${MNS}" "${PROM_POD}" -- wget -qO- \
    "http://127.0.0.1:9090/api/v1/query_range?query=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "${query}")&start=${start}&end=${end}&step=1" \
    > "${out}"
}

for run in evalscope-c28-bs512 vllm-bench-c28-bs512; do
  log="${ART}/runs/${run}/pod.log"
  win="${ART}/monitoring/windows/${run}.window"
  extract_window "${log}" "${win}"
  . "${win}"
  if [ "${end_epoch}" -le "${start_epoch}" ]; then end_epoch=$((start_epoch + 1)); fi
  outdir="${ART}/monitoring/windows/${run}"
  mkdir -p "${outdir}"
  query_range "${outdir}/decode-running.json" "sum(vllm:num_requests_running{release=\"${RELEASE}\",role=\"decode\"})" "${start_epoch}" "${end_epoch}"
  query_range "${outdir}/decode-waiting.json" "sum(vllm:num_requests_waiting{release=\"${RELEASE}\",role=\"decode\"})" "${start_epoch}" "${end_epoch}"
  query_range "${outdir}/decode-output-tps.json" "sum(rate(vllm:generation_tokens_total{release=\"${RELEASE}\",role=\"decode\"}[30s]))" "${start_epoch}" "${end_epoch}"
  query_range "${outdir}/ttft.json" "sum(rate(vllm:time_to_first_token_seconds_sum{release=\"${RELEASE}\"}[30s])) / clamp_min(sum(rate(vllm:time_to_first_token_seconds_count{release=\"${RELEASE}\"}[30s])), 1)" "${start_epoch}" "${end_epoch}"
  query_range "${outdir}/tpot.json" "sum(rate(vllm:request_time_per_output_token_seconds_sum{release=\"${RELEASE}\"}[30s])) / clamp_min(sum(rate(vllm:request_time_per_output_token_seconds_count{release=\"${RELEASE}\"}[30s])), 1)" "${start_epoch}" "${end_epoch}"
  query_range "${outdir}/spec-draft-tokens.json" "sum(rate(vllm:spec_decode_num_draft_tokens_total{release=\"${RELEASE}\",role=\"decode\"}[30s]))" "${start_epoch}" "${end_epoch}" || true
  query_range "${outdir}/spec-accepted-tokens.json" "sum(rate(vllm:spec_decode_num_accepted_tokens_total{release=\"${RELEASE}\",role=\"decode\"}[30s]))" "${start_epoch}" "${end_epoch}" || true
done

kubectl logs -n "${NS}" -l storm-service-name="${RELEASE}" --all-containers --tail=20000 > "${ART}/server-logs/stormservice-tail.log" || true
kubectl logs -n "${NS}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --all-containers --tail=10000 > "${ART}/server-logs/router-tail.log" || true
rg -n "MooncakeDiag|Mooncake transfer engine returned|timed out after|Failed to open device|prefill connection|connection closed|KV group count mismatch|handshake compatibility failure|found no common KV" \
  "${ART}/server-logs" "${ART}/runs" > "${ART}/bad-log-scan.txt" || true

cat > "${ART}/summary.md" <<'MD'
# C28 Live TTFT Correlation Summary

## Scope

- 同一 C25 调试镜像。
- 同一 `1P1D` 语义：`prefill.args.noAsyncScheduling=false`、`decode.args.maxNumSeqs=96`、无 `--max-model-len`。
- in-cluster evalscope 与 in-cluster `vllm bench serve` 均请求 router Service。
- 本 summary 原本不改变 M10 gate；2026-07-02 用户已基于 C28 结论接受 vLLM benchmark 作为当前 M10 性能 gate。

## Summary Requirements

- evalscope result:
- vLLM bench result:
- decode running/waiting comparison:
- server-side spec metric comparison:
- TTFT/TPOT metric comparison:
- MooncakeDiag / bad-log comparison:
- client/request timeline comparison:
- conclusion: confirmed / revised / inconclusive relative to C27.

## Gate

2026-07-02 用户已修改 gate：接受 vLLM benchmark 结果，认为总体通过。远端 `iaas_main` 更新前仍必须回到非诊断发布候选分支，核对候选 SHA、镜像来源、远端备份和 diff scope；不得把 C25/C28 诊断分支直接发布。
MD
```

**Expected result:** `c28-live-ttft-correlation-20260701/summary.md` 存在，且配套 Prometheus window、server logs、bad-log scan、benchmark Pod logs 都已保存。最终人工填写或后续脚本补全 summary 时，必须明确 C27 的 likely 结论是否被同窗口指标确认、修正或仍缺证据。

## C28R: restart service between failed candidates

**When to run:** C28C 中任一 benchmark Pod failed、timeout、被中断、或日志出现 Mooncake/KV bad-log 后；继续下一 measured run 前必须执行。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
RELEASE="dsv4-flash-pd"
ART="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
RESTART_DIR="${ART}/service-restarts/$(date '+%Y%m%d-%H%M%S')"
mkdir -p "${RESTART_DIR}"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl get pods -n "${NS}" -o wide | tee "${RESTART_DIR}/pods-before.txt"
kubectl logs -n "${NS}" -l storm-service-name="${RELEASE}" --all-containers --tail=5000 > "${RESTART_DIR}/stormservice-before.log" || true
kubectl logs -n "${NS}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --all-containers --tail=2000 > "${RESTART_DIR}/router-before.log" || true
kubectl delete pod -n "${NS}" -l storm-service-name="${RELEASE}" --wait=false | tee "${RESTART_DIR}/delete-stormservice.txt"
kubectl delete pod -n "${NS}" -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --wait=false | tee "${RESTART_DIR}/delete-router.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=prefill --timeout=900s | tee "${RESTART_DIR}/prefill-ready.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l storm-service-name="${RELEASE}",role-name=decode --timeout=900s | tee "${RESTART_DIR}/decode-ready.txt"
kubectl wait -n "${NS}" --for=condition=Ready pod -l app.kubernetes.io/instance="${RELEASE}",app.kubernetes.io/component=router --timeout=300s | tee "${RESTART_DIR}/router-ready.txt"
kubectl get pods -n "${NS}" -o wide | tee "${RESTART_DIR}/pods-after.txt"
```

**Expected result:** P/D/router 均重新 Ready；下一轮 measured run 从干净服务状态开始。

## C28Z: cleanup live TTFT correlation resources

**When to run:** C28 完成、失败、被中断、permit 不再需要、或用户要求停止时执行。

**Working directory:** `/data00/home/hanhan.hank/workspace/_codex_src/vllm-iaas`

```bash
set -euo pipefail

ENV_ROOT="/data00/home/hanhan.hank/workspace/env"
ENVIRONMENT="dev-cluster"
NS="vllm-dsv4-flash-pd"
MNS="vllm-dsv4-flash-pd-monitoring"
RELEASE="dsv4-flash-pd"
MRELEASE="dsv4-flash-pd-monitoring"
PERMIT_ID="${PERMIT_ID:-}"
ART="$PWD/artifacts/2026-06-29-vllm-dsv4-flash-pd/c28-live-ttft-correlation-20260701"
mkdir -p "${ART}/cleanup"
eval "$("${ENV_ROOT}/bin/envctl" use "${ENVIRONMENT}")"

kubectl delete pod -n "${NS}" \
  evalscope-c28-bs512 vllm-bench-c28-bs512 \
  evalscope-c28-bs512-persist vllm-bench-c28-bs512-persist \
  --ignore-not-found | tee "${ART}/cleanup/delete-benchmark-pods.txt" || true
helm -n "${NS}" uninstall "${RELEASE}" --ignore-not-found | tee "${ART}/cleanup/uninstall-serving.txt" || true
helm -n "${MNS}" uninstall "${MRELEASE}" --ignore-not-found | tee "${ART}/cleanup/uninstall-monitoring.txt" || true
kubectl delete ns "${NS}" "${MNS}" --ignore-not-found | tee "${ART}/cleanup/delete-namespaces.txt" || true
if [ -n "${PERMIT_ID}" ]; then
  python3 /data00/home/hanhan.hank/workspace/obsidian_remote/codex/skills/workspace-env/scripts/resource_registry.py --env-root "${ENV_ROOT}" \
    permit-release --permit-id "${PERMIT_ID}" | tee "${ART}/cleanup/permit-release.json"
fi
"${ENV_ROOT}/bin/envctl" kubectl "${ENVIRONMENT}" get ns "${NS}" "${MNS}" | tee "${ART}/cleanup/verify-ns.txt" || true
ps -eo pid,ppid,cmd | grep -E 'port-forward.*(vllm-dsv4-flash-pd|30000|19090)|evalscope-c28|vllm-bench-c28' | grep -v grep \
  | tee "${ART}/cleanup/local-processes.txt" || true
```

**Expected result:** task namespaces are gone or terminating with no remaining task pods；permit released；no local C28 port-forward or harness process remains.
