#!/usr/bin/env bash
# Provision one bare Python per `requires-python` entry (or per argument) and
# print their paths as ":"-separated DEEPGEMM_PYTHON_INTERPRETERS. Skip this
# entirely if you already have interpreter paths.
#
# Usage:
#   export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh)
#   python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
#
# Optional:
#   DEEPGEMM_VENV_PREFIX (default: /tmp/dgenv)
#   DEEPGEMM_UV_VENV_TIMEOUT (default: 600s)
set -euo pipefail

if [ "$#" -eq 0 ]; then
  # Derive the matrix from `requires-python = ">=3.X,<3.Y"` in pyproject.toml.
  pyproject="$(dirname "$0")/../pyproject.toml"
  spec=$(grep -E '^requires-python' "$pyproject" \
         | grep -oE '>=3\.[0-9]+,<3\.[0-9]+')
  lo=${spec#>=3.}; lo=${lo%%,*}
  hi=${spec##*<3.}
  set -- $(seq "$lo" $((hi - 1)) | sed 's/^/3./')
fi

prefix="${DEEPGEMM_VENV_PREFIX:-/tmp/dgenv}"
mkdir -p "$prefix"

python_has_headers() {
  local py="$1"
  "$py" - <<'PY'
import os
import sys
import sysconfig

include = sysconfig.get_paths().get("include")
if not include or not os.path.exists(os.path.join(include, "Python.h")):
    sys.exit(1)
PY
}

candidate_pythons() {
  local version="$1"
  local nodot
  nodot="$(printf '%s' "$version" | tr -d '.')"

  printf '/usr/bin/python%s\n' "$version"
  printf '/opt/python/cp%s-cp%s/bin/python%s\n' "$nodot" "$nodot" "$version"
  command -v "python${version}" 2>/dev/null || true
}

find_system_python() {
  local version="$1"
  local py

  while IFS= read -r py; do
    if [ -x "$py" ] && python_has_headers "$py"; then
      printf '%s\n' "$py"
      return 0
    fi
  done < <(candidate_pythons "$version" | awk '!seen[$0]++')

  return 1
}

create_managed_venv() {
  local version="$1"
  local venv="$2"
  local timeout_s="${DEEPGEMM_UV_VENV_TIMEOUT:-600}"
  local attempt rc

  for attempt in 1 2 3 4 5; do
    if timeout "${timeout_s}s" \
        uv venv --python "$version" "$venv" \
          --python-preference only-managed --seed >/dev/null; then
      return 0
    else
      rc=$?
    fi

    if [ "$attempt" = "5" ]; then
      return "$rc"
    fi

    echo "DeepGEMM Python ${version}: uv managed venv failed on attempt ${attempt}/5; retrying" >&2
    rm -rf "$venv" \
      "${UV_PYTHON_INSTALL_DIR:-/opt/uv/python}/.temp" \
      "${UV_CACHE_DIR:-/root/.cache/uv}/.temp"
    sleep $((attempt * 30))
  done
}

paths=""
for V in "$@"; do
  venv="$prefix/$V"
  if py="$(find_system_python "$V")"; then
    echo "DeepGEMM Python ${V}: using system interpreter ${py}" >&2
    paths="$paths:$py"
    continue
  fi

  if [ -x "$venv/bin/python" ] && python_has_headers "$venv/bin/python"; then
    echo "DeepGEMM Python ${V}: reusing managed venv ${venv}" >&2
    paths="$paths:$venv/bin/python"
    continue
  fi

  echo "DeepGEMM Python ${V}: system interpreter with Python.h not found; falling back to uv managed Python" >&2
  create_managed_venv "$V" "$venv"
  if ! python_has_headers "$venv/bin/python"; then
    echo "DeepGEMM Python ${V}: ${venv}/bin/python does not expose Python.h" >&2
    exit 1
  fi
  paths="$paths:$venv/bin/python"
done
echo "${paths#:}"
