# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HELPER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/ci/get_byteiaas_image_tag.py"
)
SPEC = importlib.util.spec_from_file_location("get_byteiaas_image_tag", HELPER_PATH)
assert SPEC is not None
helper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(helper)


def test_build_dev_tag_with_cuda_suffix() -> None:
    assert (
        helper.build_tag(
            mode="dev",
            image_flavor="openai",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="",
            cuda_suffix="cu130",
        )
        == "v0.12.0.iaas.dev.202606231234-cu130"
    )


def test_build_release_tag_with_internal_tag_and_cuda_suffix() -> None:
    assert (
        helper.build_tag(
            mode="release",
            image_flavor="openai",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="0.0.11",
            cuda_suffix="cu130",
        )
        == "v0.12.0.byted.0.0.11.202606231234-cu130"
    )


def test_build_dev_openai_devel_tag_with_cuda_suffix() -> None:
    assert (
        helper.build_tag(
            mode="dev",
            image_flavor="openai-devel",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="",
            cuda_suffix="cu130",
        )
        == "v0.12.0.iaas.dev.202606231234-openai-devel-cu130"
    )


def test_build_release_openai_devel_tag_with_cuda_suffix() -> None:
    assert (
        helper.build_tag(
            mode="release",
            image_flavor="openai-devel",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="0.0.11",
            cuda_suffix="cu130",
        )
        == "v0.12.0.byted.0.0.11.202606231234-openai-devel-cu130"
    )


def test_build_tag_rejects_unknown_image_flavor() -> None:
    with pytest.raises(SystemExit, match="unsupported image flavor"):
        helper.build_tag(
            mode="dev",
            image_flavor="runtime",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="",
            cuda_suffix="cu130",
        )


def test_release_requires_tag_value() -> None:
    with pytest.raises(SystemExit, match="--tag-value is required"):
        helper.build_tag(
            mode="release",
            image_flavor="openai",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="",
            cuda_suffix="cu130",
        )


def test_release_rejects_unsafe_tag_value() -> None:
    with pytest.raises(SystemExit, match="Docker tag-safe suffix"):
        helper.build_tag(
            mode="release",
            image_flavor="openai",
            vllm_version="0.12.0",
            timestamp="202606231234",
            tag_value="../bad",
            cuda_suffix="cu130",
        )


def test_timestamp_must_be_yyyymmddhhmm() -> None:
    with pytest.raises(SystemExit, match="YYYYMMDDHHMM"):
        helper.current_timestamp("2026-06-23T12:34")


def test_normalize_version_strips_leading_v() -> None:
    assert helper.normalize_version("v0.12.0") == "0.12.0"


def test_normalize_version_rejects_invalid_version() -> None:
    with pytest.raises(SystemExit, match="invalid vLLM version"):
        helper.normalize_version("release-candidate")


def test_normalize_version_rejects_pep440_local_suffix() -> None:
    with pytest.raises(SystemExit, match="invalid vLLM version"):
        helper.normalize_version("0.12.0+local")
