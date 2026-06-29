#!/bin/bash
# Build vLLM Rust artifacts and install them into the vllm package.
# Usage: ./build_rust.sh [--debug]
#
# By default builds in release mode. Pass --debug for faster compile times
# during development.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read the required toolchain from rust-toolchain.toml.
TOOLCHAIN=$(grep '^channel' "$REPO_ROOT/rust-toolchain.toml" | sed 's/.*= *"\(.*\)"/\1/')

retry_command() {
    local attempt
    local max_attempts=5
    local delay_seconds=2
    local status=0

    for attempt in $(seq 1 "$max_attempts"); do
        if "$@"; then
            return 0
        fi
        status=$?
        if [[ "$attempt" == "$max_attempts" ]]; then
            return "$status"
        fi
        echo "Command failed with status $status, retrying ($attempt/$max_attempts): $*" >&2
        sleep "$delay_seconds"
    done
}

install_rustup() {
    local installer="/tmp/rustup-init.sh"
    local rustup_url="${RUSTUP_INIT_URL:-https://sh.rustup.rs}"

    retry_command curl \
        --proto '=https' \
        --tlsv1.2 \
        --fail \
        --show-error \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --connect-timeout 20 \
        --max-time 300 \
        -o "$installer" \
        "$rustup_url"
    sh "$installer" -y --default-toolchain none
    rm -f "$installer"
}

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    install_rustup
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    retry_command rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARG="--debug"
else
    PROFILE_ARG="--release"
fi

python3 "$REPO_ROOT/tools/build_rust.py" "$PROFILE_ARG"
