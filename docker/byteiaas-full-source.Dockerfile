ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_BUILD_COMMIT
ARG FLASHINFER_BUILD_COMMIT
LABEL org.opencontainers.image.source="https://github.com/bytedance-iaas/vllm" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.byteiaas.flashinfer.revision="${FLASHINFER_BUILD_COMMIT}"

ENV VIRTUAL_ENV="" \
    PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}" \
    PYTHONPATH=""
WORKDIR /opt/byteiaas

COPY --from=byteiaas-wheels wheel.sha256 source-commit /opt/byteiaas/vllm/
COPY --from=byteiaas-flashinfer-wheels wheel.sha256 source-commit /opt/byteiaas/flashinfer/
RUN --mount=type=bind,from=byteiaas-wheels,target=/wheels/vllm \
    --mount=type=bind,from=byteiaas-flashinfer-wheels,target=/wheels/flashinfer \
    set -eu; \
    uv pip check --system > /tmp/byteiaas-base-dependencies 2>&1 || test "$?" -eq 1; \
    sed -n '/^The package /p' /tmp/byteiaas-base-dependencies | sort > /opt/byteiaas/base-dependency-conflicts.txt; \
    test "$(cat /wheels/vllm/source-commit)" = "${VLLM_BUILD_COMMIT}"; \
    test "$(cat /wheels/flashinfer/source-commit)" = "${FLASHINFER_BUILD_COMMIT}"; \
    test -n "${VLLM_BUILD_COMMIT}"; test -n "${FLASHINFER_BUILD_COMMIT}"; \
    cd /wheels/vllm; sha256sum --check --strict wheel.sha256; \
    set -- vllm-*.whl; test "$#" -eq 1; \
    uv pip install --system --no-deps --reinstall "$1"; \
    cd /wheels/flashinfer; sha256sum --check --strict wheel.sha256; \
    set -- flashinfer_python-*.whl; test "$#" -eq 1; \
    uv pip install --system --no-deps --reinstall "$1"; \
    uv pip check --system > /tmp/byteiaas-source-dependencies 2>&1 || test "$?" -eq 1; \
    sed -n '/^The package /p' /tmp/byteiaas-source-dependencies | sort > /opt/byteiaas/source-dependency-conflicts.txt; \
    diff -u /opt/byteiaas/base-dependency-conflicts.txt /opt/byteiaas/source-dependency-conflicts.txt; \
    cat /opt/byteiaas/source-dependency-conflicts.txt; \
    rm -rf /opt/tk300-overlay/vllm /opt/tk300-overlay/flashinfer

RUN VLLM_BUILD_COMMIT="${VLLM_BUILD_COMMIT}" FLASHINFER_BUILD_COMMIT="${FLASHINFER_BUILD_COMMIT}" python3 - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

record = {"schema_version": 1, "delivery": "complete-source-wheels", "packages": {}}
for name, variable in (("vllm", "VLLM_BUILD_COMMIT"), ("flashinfer", "FLASHINFER_BUILD_COMMIT")):
    distribution = "vllm" if name == "vllm" else "flashinfer-python"
    spec = importlib.util.find_spec(name)
    package = Path(next(iter(spec.submodule_search_locations)))
    files = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.rglob("*"))
             if path.is_file() and path.suffix != ".pyc"}
    record["packages"][name] = {"commit": os.environ[variable],
                               "version": importlib.metadata.version(distribution),
                               "files": files}
Path("/opt/byteiaas/source-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
PY

ENTRYPOINT ["vllm", "serve"]
