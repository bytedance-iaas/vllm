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

# The published Mooncake wheel is linked against libcudart.so.12. Keep that
# runtime alongside CUDA 13 until Mooncake publishes a CUDA 13 wheel.
RUN python3 -m pip install --no-cache-dir nvidia-cuda-runtime-cu12==12.9.79

WORKDIR /workspace
