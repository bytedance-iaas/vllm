# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject stale release build inputs before running expensive CI builds."""

import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if (root / ".github/byteiaas-build-contract").read_text().strip() != "5":
        raise ValueError("Expected ByteIAAS build contract 5")
    dockerfile = (root / "docker/Dockerfile").read_text()
    for name, value in {
        "CUDA_VERSION": "13.0.3",
        "PYTHON_VERSION": "3.12",
        "UBUNTU_VERSION": "24.04",
        "UV_VERSION": "0.11.1",
    }.items():
        if f"ARG {name}={value}\n" not in dockerfile:
            raise ValueError(f"Unexpected release build default: {name}")
    if "torch==2.13.0\n" not in (root / "requirements/build/cuda.txt").read_text():
        raise ValueError("Unexpected native PyTorch build contract")
    for workflow_name in (
        "_byteiaas-build-wheel.yml",
        "_byteiaas-build-and-publish-image.yml",
    ):
        workflow = (root / ".github/workflows" / workflow_name).read_text()
        if '--build-arg UV_INDEX="${BYTEIAAS_PIP_EXTRA_INDEX_URL}"' not in workflow:
            raise ValueError(f"Missing supplemental Python index: {workflow_name}")
        for distribution in ("UBUNTU", "ALMALINUX", "EPEL"):
            variable = f"BYTEIAAS_{distribution}_MIRROR"
            value = f"http://mirrors.volces.com/{distribution.lower()}"
            if f"{variable}: {value}\n" not in workflow:
                raise ValueError(f"Missing internal mirror: {workflow_name}/{variable}")
            if f'--build-arg {distribution}_MIRROR="${{{variable}}}"' not in workflow:
                raise ValueError(
                    f"Missing mirror build arg: {workflow_name}/{variable}"
                )
    wheel_workflow = (root / ".github/workflows/_byteiaas-build-wheel.yml").read_text()
    if "--target mooncake-system-deps" not in wheel_workflow or (
        wheel_workflow.index("--target mooncake-system-deps")
        > wheel_workflow.index("--target byteiaas-wheel-artifacts")
    ):
        raise ValueError("Missing system dependency preflight before wheel compilation")
    for stage in ("base", "mooncake-system-deps", "vllm-runtime-base"):
        marker = f" AS {stage}\n"
        stage_body = (
            dockerfile.split(marker, 1)[1].split("\nFROM ", 1)[0]
            if marker in dockerfile
            else ""
        )
        if "RUN bash /usr/local/bin/setup_package_mirrors.sh" not in stage_body:
            raise ValueError(f"Missing package mirror setup: {stage}")
        if stage in ("base", "vllm-runtime-base"):
            for setting in (
                "ARG PIP_INDEX_URL UV_INDEX_URL\n",
                "ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL\n",
                "ARG UV_INDEX\n",
                "ARG UV_VERSION\n",
            ):
                if setting not in stage_body:
                    raise ValueError(f"Missing Python bootstrap argument: {stage}")
    if "https://astral.sh/uv/install.sh" in dockerfile:
        raise ValueError("uv must use a pinned package from the Python index")
    if 'apt-cache show "python${PYTHON_VERSION}-dev"' not in dockerfile:
        raise ValueError("Runtime must prefer distribution Python before adding a PPA")
    if "Preflight supplemental Python index" not in wheel_workflow or (
        wheel_workflow.index("Preflight supplemental Python index")
        > wheel_workflow.index("--target byteiaas-wheel-artifacts")
    ):
        raise ValueError("Missing supplemental index check before wheel compilation")
    mooncake = dockerfile.split(" AS mooncake-build\n", 1)[1].split("\nFROM ", 1)[0]
    if "ARG UV_INDEX\n" not in mooncake:
        raise ValueError("Mooncake must retain the supplemental Python index")
    system_deps = dockerfile.split(" AS mooncake-system-deps\n", 1)[1].split(
        "\nFROM ", 1
    )[0]
    if "COPY --from=" in system_deps or "uv " in system_deps:
        raise ValueError(
            "System dependency preflight depends on expensive build stages"
        )
    if "FROM mooncake-system-deps AS mooncake-build\n" not in dockerfile:
        raise ValueError("Mooncake build must reuse the system dependency preflight")
    sources = {}
    for kind in ("deepgemm", "humming"):
        path = root / "docker/byteiaas/native" / kind / "source.json"
        config = json.loads(path.read_text())
        if (
            not config["repository"].startswith("https://")
            or not config["package_version"]
        ):
            raise ValueError(f"Invalid native source config: {kind}")
        subprocess.run(
            ["git", "check-ref-format", f"refs/heads/{config['ref']}"], check=True
        )
        sources[kind] = config
    print(json.dumps({"contract": 5, "native_sources": sources}, indent=2))


if __name__ == "__main__":
    main()
