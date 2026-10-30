ARG VLLM_OPENAI_DEVEL_BASE_IMAGE=local/vllm-openai:dev
FROM ${VLLM_OPENAI_DEVEL_BASE_IMAGE}

ARG CUDA_VERSION=13.0.3
ARG CUDA_PACKAGE_VERSION=13.0.3-1
ARG UBUNTU_MIRROR=http://mirrors.volces.com/ubuntu
ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG=local/vllm-openai-devel:dev
ARG VLLM_IMAGE_SOURCE=https://github.com/vllm-project/vllm
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
    if [ -n "${UBUNTU_MIRROR}" ]; then \
        if [ -f /etc/apt/sources.list ]; then \
            sed -i -E "s#https?://(archive|security)[.]ubuntu[.]com/ubuntu#${UBUNTU_MIRROR}#g" /etc/apt/sources.list; \
        fi; \
        if [ -d /etc/apt/sources.list.d ]; then \
            find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) \
                -exec sed -i -E "s#https?://(archive|security)[.]ubuntu[.]com/ubuntu#${UBUNTU_MIRROR}#g" {} +; \
        fi; \
    fi; \
    rm -f /etc/apt/sources.list.d/*deadsnakes*; \
    CUDA_VERSION_DASH="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr "." "-")" \
    && apt-get -o Acquire::Retries=5 update -y \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cuda-libraries-dev-${CUDA_VERSION_DASH}=${CUDA_PACKAGE_VERSION} \
        cuda-minimal-build-${CUDA_VERSION_DASH}=${CUDA_PACKAGE_VERSION} \
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

ENV VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT} \
    VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE} \
    VLLM_BUILD_URL=${VLLM_BUILD_URL} \
    VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG}
LABEL org.opencontainers.image.source="${VLLM_IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.opencontainers.image.version="${VLLM_IMAGE_TAG}" \
      org.opencontainers.image.url="${VLLM_BUILD_URL}" \
      ai.vllm.build.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}" \
      ai.vllm.build.url="${VLLM_BUILD_URL}" \
      ai.vllm.image.tag="${VLLM_IMAGE_TAG}"

WORKDIR /workspace
