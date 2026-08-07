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

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    rustup_installer=$(mktemp)
    trap 'rm -f "$rustup_installer"' EXIT
    curl --retry 5 --retry-all-errors --retry-delay 2 --retry-max-time 120 \
        --connect-timeout 20 --proto '=https' --tlsv1.2 -sSf \
        -o "$rustup_installer" https://sh.rustup.rs
    sh "$rustup_installer" -y --default-toolchain none
    rm -f "$rustup_installer"
    trap - EXIT
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARG="--debug"
else
    PROFILE_ARG="--release"
fi

python3 "$REPO_ROOT/tools/build_rust.py" "$PROFILE_ARG"
