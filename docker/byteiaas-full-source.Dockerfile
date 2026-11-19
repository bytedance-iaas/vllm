ARG BASE_IMAGE=iaas-gpu-cn-beijing.cr.volces.com/serving/vllm@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3
FROM ${BASE_IMAGE}

ARG VLLM_BUILD_COMMIT
ARG FLASHINFER_BUILD_COMMIT
LABEL org.opencontainers.image.source="https://github.com/bytedance-iaas/vllm" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.byteiaas.flashinfer.revision="${FLASHINFER_BUILD_COMMIT}"

COPY --from=byteiaas-wheels wheel.sha256 source-commit /opt/byteiaas/vllm/
COPY --from=byteiaas-flashinfer-wheels wheel.sha256 source-commit /opt/byteiaas/flashinfer/
RUN --mount=type=bind,from=byteiaas-wheels,target=/wheels/vllm \
    --mount=type=bind,from=byteiaas-flashinfer-wheels,target=/wheels/flashinfer \
    set -eu; \
    test "$(cat /wheels/vllm/source-commit)" = "${VLLM_BUILD_COMMIT}"; \
    test "$(cat /wheels/flashinfer/source-commit)" = "${FLASHINFER_BUILD_COMMIT}"; \
    test -n "${VLLM_BUILD_COMMIT}"; test -n "${FLASHINFER_BUILD_COMMIT}"; \
    cd /wheels/vllm; sha256sum --check --strict wheel.sha256; \
    set -- vllm-*.whl; test "$#" -eq 1; \
    uv pip install --system --no-deps --reinstall "$1"; \
    cd /wheels/flashinfer; sha256sum --check --strict wheel.sha256; \
    set -- flashinfer_python-*.whl; test "$#" -eq 1; \
    uv pip install --system --no-deps --reinstall "$1"; \
    uv pip check --system; \
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
if record["packages"]["vllm"]["version"] != "0.33.0":
    raise ValueError("Preserve the qualified vLLM/NIXL version identity")
Path("/opt/byteiaas/source-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
PY

ENV PYTHONPATH=""
WORKDIR /vllm-workspace
ENTRYPOINT ["vllm", "serve"]
