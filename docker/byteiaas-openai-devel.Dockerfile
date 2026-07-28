ARG VLLM_OPENAI_DEVEL_BASE_IMAGE
FROM ${VLLM_OPENAI_DEVEL_BASE_IMAGE}

ARG CUDA_VERSION=13.0.2
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive

RUN CUDA_VERSION_DASH="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr "." "-")" \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cuda-libraries-dev-${CUDA_VERSION_DASH} \
        cuda-minimal-build-${CUDA_VERSION_DASH} \
        cmake \
        curl \
        gdb \
        git \
        git-lfs \
        less \
        lsof \
        ninja-build \
        pkg-config \
        rsync \
        strace \
        tmux \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Match the CUDA 13 image runtime. INSTALL_KV_CONNECTORS currently installs the
# generic Mooncake wheel, so replace it with the CUDA 13 build validated by the
# PCP+Mooncake deployment.
RUN python3 -m pip uninstall -y mooncake-transfer-engine \
    && python3 -m pip install --no-cache-dir \
        mooncake-transfer-engine-cuda13==0.3.12.post1 \
    && python3 -c "from importlib.metadata import version; from pathlib import Path; assert version('mooncake-transfer-engine') == '0.3.12.post1'; assert Path('/usr/local/lib/python3.12/dist-packages/mooncake/engine.so').is_file()" \
    && ldd /usr/local/lib/python3.12/dist-packages/mooncake/engine.so \
        | tee /tmp/mooncake-engine-ldd.txt \
    && grep -q 'libcudart\\.so\\.13 =>' /tmp/mooncake-engine-ldd.txt \
    && ! grep -q 'libcudart\\.so\\.12' /tmp/mooncake-engine-ldd.txt

WORKDIR /workspace
