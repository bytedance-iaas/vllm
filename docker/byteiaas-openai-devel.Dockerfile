ARG VLLM_OPENAI_DEVEL_BASE_IMAGE
FROM ${VLLM_OPENAI_DEVEL_BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
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
