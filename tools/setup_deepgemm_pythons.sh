#!/usr/bin/env bash
# Provision one bare Python per `requires-python` entry (or per argument) and
# print their paths as ":"-separated DEEPGEMM_PYTHON_INTERPRETERS. Skip this
# entirely if you already have interpreter paths.
#
# Usage:
#   export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh)
#   python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
#
# Optional: DEEPGEMM_VENV_PREFIX (default: /tmp/dgenv).
set -euo pipefail

log() {
  printf '[setup_deepgemm_pythons] %s\n' "$*" >&2
}

python_is_usable() {
  local py="$1"
  local version="$2"
  "$py" - "$version" <<'PY' >/dev/null 2>&1
import os
import sys
import sysconfig

want = sys.argv[1]
got = f"{sys.version_info.major}.{sys.version_info.minor}"
if got != want:
    raise SystemExit(1)

include = sysconfig.get_config_var("INCLUDEPY")
if not include or not os.path.exists(os.path.join(include, "Python.h")):
    raise SystemExit(1)

if not sysconfig.get_config_var("EXT_SUFFIX"):
    raise SystemExit(1)
PY
}

if [ "$#" -eq 0 ]; then
  # Derive the matrix from `requires-python = ">=3.X,<3.Y"` in pyproject.toml.
  pyproject="$(dirname "$0")/../pyproject.toml"
  spec=$(grep -E '^requires-python' "$pyproject" \
         | grep -oE '>=3\.[0-9]+,<3\.[0-9]+')
  lo=${spec#>=3.}; lo=${lo%%,*}
  hi=${spec##*<3.}
  readarray -t versions < <(seq "$lo" $((hi - 1)) | sed 's/^/3./')
  set -- "${versions[@]}"
fi

prefix="${DEEPGEMM_VENV_PREFIX:-/tmp/dgenv}"
mkdir -p "$prefix"

log "Target Python versions: $*"
if [ -n "${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-}" ]; then
  log "Proxy configured: HTTP_PROXY=${HTTP_PROXY:-<unset>} HTTPS_PROXY=${HTTPS_PROXY:-<unset>} ALL_PROXY=${ALL_PROXY:-<unset>}"
fi

paths=""
for V in "$@"; do
  venv="$prefix/$V"
  nodot="${V/./}"
  py=""

  # Prefer Python interpreters already shipped in manylinux builder images.
  # They include headers and avoid uv-managed Python downloads on cold builds.
  candidates=(
    "$venv/bin/python"
    "/opt/python/cp${nodot}-cp${nodot}/bin/python${V}"
    "/opt/python/cp${nodot}-cp${nodot}/bin/python3"
    "/opt/python/cp${nodot}-cp${nodot}/bin/python"
  )
  if system_py="$(command -v "python${V}" 2>/dev/null)"; then
    candidates+=("$system_py")
  fi

  for candidate in "${candidates[@]}"; do
    if [ -x "$candidate" ] && python_is_usable "$candidate" "$V"; then
      py="$candidate"
      break
    fi
  done

  if [ -n "$py" ]; then
    log "Using existing Python ${V}: ${py}"
    paths="$paths:$py"
    continue
  fi

  # uv-managed Python ensures Python.h is present; system 3.X-dev packages
  # on the manylinux / Ubuntu build bases are not always installed.
  log "Provisioning Python ${V} with uv at ${venv}"
  start_time="$(date +%s)"
  uv venv --python "$V" "$venv" --python-preference only-managed --seed 1>&2
  py="$venv/bin/python"
  if ! python_is_usable "$py" "$V"; then
    log "Provisioned Python ${V} is missing Python.h, EXT_SUFFIX, or has the wrong version: ${py}"
    exit 1
  fi
  elapsed="$(( $(date +%s) - start_time ))"
  log "Provisioned Python ${V}: ${py} (${elapsed}s)"
  paths="$paths:$py"
done
echo "${paths#:}"
