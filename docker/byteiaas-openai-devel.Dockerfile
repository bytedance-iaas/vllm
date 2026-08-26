ARG VLLM_OPENAI_DEVEL_BASE_IMAGE
FROM ${VLLM_OPENAI_DEVEL_BASE_IMAGE}

ARG CUDA_VERSION=13.0.2
ARG UBUNTU_MIRROR=http://mirrors.byted.org/ubuntu
ARG UBUNTU_MIRROR_HOSTS="10.8.6.125 mirrors.byted.org"
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    if [ -n "${UBUNTU_MIRROR_HOSTS}" ]; then \
        echo "${UBUNTU_MIRROR_HOSTS}" >> /etc/hosts; \
    fi; \
    if [ -n "${UBUNTU_MIRROR}" ]; then \
        if [ -f /etc/apt/sources.list ]; then \
            sed -i -E "s#https?://(archive|security)[.]ubuntu[.]com/ubuntu#${UBUNTU_MIRROR}#g" /etc/apt/sources.list; \
        fi; \
        find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) \
            -exec sed -i -E "s#https?://(archive|security)[.]ubuntu[.]com/ubuntu#${UBUNTU_MIRROR}#g" {} +; \
    fi; \
    rm -f /etc/apt/sources.list.d/*deadsnakes*; \
    CUDA_VERSION_DASH="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr "." "-")" \
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

WORKDIR /workspace
