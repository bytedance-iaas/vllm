# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_BUILD_COMMIT
LABEL org.opencontainers.image.source="https://github.com/bytedance-iaas/vllm" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}"

# Wheel contents must invalidate the install layer even on a warm builder.
COPY --from=byteiaas-wheels wheel.sha256 source-commit /opt/byteiaas/
RUN --mount=type=bind,from=byteiaas-wheels,target=/wheels \
    set -eu; \
    cd /wheels; \
    test "$(cat source-commit)" = "${VLLM_BUILD_COMMIT}"; \
    sha256sum --check --strict wheel.sha256; \
    set -- vllm-*.whl; \
    test "$#" -eq 1; \
    uv pip install --system --no-deps --reinstall "$1"; \
    uv pip check --system; \
    rm -rf /opt/tk300-overlay/vllm

WORKDIR /vllm-workspace
ENTRYPOINT ["vllm", "serve"]
