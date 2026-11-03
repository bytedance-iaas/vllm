#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

uv_version=0.11.1
uv_sha256=41ccc438dbb905240a3630265feb25be1bda61656ec7c32682a83648a686f4aa
uv_url="https://mirrors.volces.com/pypi/packages/c3/25/9ef73c8b6ef04b0cead7d8f1547034568e3e58f3397b55b83167e587f84a/uv-${uv_version}-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
uv_dir="${RUNNER_TEMP:?}/byteiaas-uv-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
uv_wheel="${uv_dir}/uv.whl"

test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
mkdir -p "${uv_dir}"
env NO_PROXY=mirrors.volces.com no_proxy=mirrors.volces.com \
    curl --fail --location --connect-timeout 10 --max-time 60 \
    --retry 2 --retry-all-errors --retry-delay 2 --retry-max-time 180 \
    --output "${uv_wheel}" "${uv_url}"
printf '%s  %s\n' "${uv_sha256}" "${uv_wheel}" | sha256sum --check -
for tool in uv uvx; do
    unzip -p "${uv_wheel}" "uv-${uv_version}.data/scripts/${tool}" \
        > "${uv_dir}/${tool}"
    chmod +x "${uv_dir}/${tool}"
done
case "$("${uv_dir}/uv" --version)" in
    "uv ${uv_version}" | "uv ${uv_version} "*) ;;
    *) echo "::error::Unexpected uv version"; exit 1 ;;
esac
echo "${uv_dir}" >> "${GITHUB_PATH:?}"
