#!/bin/bash
# Script to build and/or install DeepGEMM from source
# Default: build and install immediately
# Optional: build wheels to a directory for later installation (useful in multi-stage builds)
set -e

# Default values
# Keep these defaults in sync with cmake/external_projects/deepgemm.cmake and
# docker/Dockerfile.
DEEPGEMM_GIT_REPO="${DEEPGEMM_GIT_REPOSITORY:-https://github.com/vllm-project/DeepGEMM.git}"
# NOTE: This is currently targeting nv-dev branch due to sm120 support
DEEPGEMM_GIT_REF="${DEEPGEMM_GIT_COMMIT:-e21c821f39a2056d68067a466c64ddc942200106}"
WHEEL_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --repo requires an argument." >&2
                exit 1
            fi
            DEEPGEMM_GIT_REPO="$2"
            shift 2
            ;;
        --ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --ref requires an argument." >&2
                exit 1
            fi
            DEEPGEMM_GIT_REF="$2"
            shift 2
            ;;
        --cuda-version)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --cuda-version requires an argument." >&2
                exit 1
            fi
            CUDA_VERSION="$2"
            shift 2
            ;;
        --wheel-dir)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --wheel-dir requires a directory path." >&2
                exit 1
            fi
            WHEEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --repo URL         Git repository (default: $DEEPGEMM_GIT_REPO)"
            echo "  --ref COMMIT       Exact 40-character commit (default: $DEEPGEMM_GIT_REF)"
            echo "  --cuda-version VER CUDA version (auto-detected if not provided)"
            echo "  --wheel-dir PATH   If set, build wheel into PATH but do not install"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! "$DEEPGEMM_GIT_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: DeepGEMM ref must be an exact 40-character commit." >&2
    exit 1
fi
DEEPGEMM_GIT_REF="${DEEPGEMM_GIT_REF,,}"

# Auto-detect CUDA version if not provided
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "Auto-detected CUDA version: $CUDA_VERSION"
    else
        echo "Warning: Could not auto-detect CUDA version. Please specify with --cuda-version"
        exit 1
    fi
fi

# Extract major and minor version numbers
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#"${CUDA_MAJOR}".}"
CUDA_MINOR="${CUDA_MINOR%%.*}"
echo "CUDA version: $CUDA_VERSION (major: $CUDA_MAJOR, minor: $CUDA_MINOR)"

# Check CUDA version requirement
if [ "$CUDA_MAJOR" -lt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]; }; then
    echo "Skipping DeepGEMM build/installation (requires CUDA 12.8+ but got ${CUDA_VERSION})"
    exit 0
fi

echo "Preparing DeepGEMM build..."
echo "Repository: $DEEPGEMM_GIT_REPO"
echo "Reference: $DEEPGEMM_GIT_REF"

# Create a temporary directory for the build
INSTALL_DIR=$(mktemp -d)
trap 'rm -rf "$INSTALL_DIR"' EXIT

# Fetch only the selected commit, then initialize its pinned submodules.
git init "$INSTALL_DIR/deepgemm"
pushd "$INSTALL_DIR/deepgemm"
git remote add origin "$DEEPGEMM_GIT_REPO"
git fetch --depth=1 origin "$DEEPGEMM_GIT_REF"
git checkout --detach "$DEEPGEMM_GIT_REF"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$DEEPGEMM_GIT_REF" ]; then
    echo "Error: DeepGEMM checkout mismatch: expected $DEEPGEMM_GIT_REF, got $ACTUAL_COMMIT" >&2
    exit 1
fi
git submodule update --init --recursive --depth=1

case "${DEEPGEMM_REQUIRE_SM90_MEGA_MOE:-0}" in
    1|ON|on|TRUE|true|YES|yes)
        python3 "$SCRIPT_DIR/check_deepgemm_source.py" "$PWD"
        ;;
esac

# Clean previous build artifacts
# (Based on https://github.com/deepseek-ai/DeepGEMM/blob/main/install.sh)
rm -rf -- build dist *.egg-info 2>/dev/null || true

# Build wheel
echo "🏗️  Building DeepGEMM wheel..."
python3 setup.py bdist_wheel

# If --wheel-dir was specified, copy wheels there and exit
if [ -n "$WHEEL_DIR" ]; then
    mkdir -p "$WHEEL_DIR"
    cp dist/*.whl "$WHEEL_DIR"/
    echo "✅ Wheel built and copied to $WHEEL_DIR"
    popd
    exit 0
fi

# Default behaviour: install built wheel
if command -v uv >/dev/null 2>&1; then
    echo "Installing DeepGEMM wheel using uv..."
    if [ -n "$VLLM_DOCKER_BUILD_CONTEXT" ]; then
        uv pip install --system dist/*.whl
    else
        uv pip install dist/*.whl
    fi
else
    echo "Installing DeepGEMM wheel using pip..."
    python3 -m pip install dist/*.whl
fi

popd
echo "✅ DeepGEMM installation completed successfully"
