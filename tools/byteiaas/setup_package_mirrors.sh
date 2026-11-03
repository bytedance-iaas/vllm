#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# An optional root allows source transformations to be checked in isolation.
package_root="${1:-/}"
export UBUNTU_MIRROR="${UBUNTU_MIRROR:-}"
export ALMALINUX_MIRROR="${ALMALINUX_MIRROR:-}"
export EPEL_MIRROR="${EPEL_MIRROR:-}"

rewrite_sources() {
    local source_file="$1"
    local source_kind="$2"
    local rewritten
    rewritten="$(mktemp)"
    if ! awk -v kind="${source_kind}" '
        BEGIN {
            ubuntu = ENVIRON["UBUNTU_MIRROR"]
            alma = ENVIRON["ALMALINUX_MIRROR"]
            epel = ENVIRON["EPEL_MIRROR"]
            sub(/\/+$/, "", ubuntu)
            sub(/\/+$/, "", alma)
            sub(/\/+$/, "", epel)
        }
        function flush(    i, line) {
            for (i = 1; i <= count; i++) {
                line = lines[i]
                if (mapped && line ~ /^[ \t]*(metalink|mirrorlist)=/)
                    sub(/^[ \t]*/, "#", line)
                print line
            }
            count = 0
            mapped = 0
        }
        kind == "apt" {
            if (ubuntu != "") {
                gsub(/https?:\/\/([a-z][a-z]\.)?(archive|security)\.ubuntu\.com\/ubuntu/, ubuntu)
                gsub(/https?:\/\/ports\.ubuntu\.com\/ubuntu-ports/, ubuntu)
            }
            print
            next
        }
        /^\[/ { flush() }
        {
            line = $0
            if (alma != "" && line ~ /^[ \t]*#?[ \t]*baseurl=https?:\/\/(repo|vault)\.almalinux\.org\/almalinux/) {
                sub(/^[ \t]*#?[ \t]*baseurl=/, "baseurl=", line)
                sub(/https?:\/\/(repo|vault)\.almalinux\.org\/almalinux/, alma, line)
                mapped = 1
            }
            if (epel != "" && line ~ /^[ \t]*#?[ \t]*baseurl=https?:\/\/[^ \t]*\/epel\//) {
                sub(/^[ \t]*#?[ \t]*baseurl=/, "baseurl=", line)
                sub(/https?:\/\/[^ \t]*\/epel\//, epel "/", line)
                mapped = 1
            }
            lines[++count] = line
        }
        END { if (kind != "apt") flush() }
    ' "${source_file}" > "${rewritten}"; then
        rm -f "${rewritten}"
        return 1
    fi
    cat "${rewritten}" > "${source_file}"
    rm -f "${rewritten}"
}

if [ -d "${package_root}/etc/apt" ]; then
    for source_file in "${package_root}/etc/apt/sources.list" \
        "${package_root}"/etc/apt/sources.list.d/*.list \
        "${package_root}"/etc/apt/sources.list.d/*.sources; do
        [ -f "${source_file}" ] || continue
        rewrite_sources "${source_file}" apt
    done
    mkdir -p "${package_root}/etc/apt/apt.conf.d"
    cat > "${package_root}/etc/apt/apt.conf.d/80byteiaas-downloads" <<'EOF'
Acquire::Retries "5";
Acquire::http::Timeout "20";
Acquire::https::Timeout "20";
APT::Update::Error-Mode "any";
EOF
fi

for source_file in "${package_root}"/etc/yum.repos.d/*.repo; do
    [ -f "${source_file}" ] || continue
    rewrite_sources "${source_file}" dnf
done
