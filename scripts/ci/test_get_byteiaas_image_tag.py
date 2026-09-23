#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_byteiaas_image_tag import (
    build_tag,
    get_vllm_version_for_mode,
    normalize_version,
)


class GetByteiaasImageTagTest(unittest.TestCase):
    def test_dev_openai_tag_preserves_existing_format(self) -> None:
        self.assertEqual(
            build_tag(
                mode="dev",
                image_flavor="openai",
                vllm_version="0.27.0",
                timestamp="202608141200",
                tag_value="",
                cuda_suffix="cu130",
            ),
            "v0.27.0.iaas.dev.202608141200-cu130",
        )

    def test_format_suffix_trails_flavor_and_cuda_suffixes(self) -> None:
        self.assertEqual(
            build_tag(
                mode="dev",
                image_flavor="openai-devel",
                vllm_version="0.27.0",
                timestamp="202608141200",
                tag_value="",
                cuda_suffix="cu130",
                format_suffix="zstd",
            ),
            "v0.27.0.iaas.dev.202608141200-openai-devel-cu130-zstd",
        )
        self.assertEqual(
            build_tag(
                mode="release",
                image_flavor="openai",
                vllm_version="0.27.0",
                timestamp="202608141200",
                tag_value="0.27.0.post1",
                cuda_suffix="cu130",
                format_suffix="nydus",
            ),
            "v0.27.0.post1.byted.202608141200-cu130-nydus",
        )

    def test_format_suffix_must_be_tag_safe(self) -> None:
        for value in ("-zstd", "zstd/", "zs td", "z$td"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    SystemExit, "--format-suffix must be a Docker tag-safe suffix"
                ),
            ):
                build_tag(
                    mode="dev",
                    image_flavor="openai",
                    vllm_version="0.27.0",
                    timestamp="202608141200",
                    tag_value="",
                    cuda_suffix="cu130",
                    format_suffix=value,
                )

    def test_generated_tag_must_fit_docker_tag_limit(self) -> None:
        with self.assertRaisesRegex(SystemExit, "maximum length is 128"):
            build_tag(
                mode="release",
                image_flavor="openai-devel",
                vllm_version="",
                timestamp="202608141200",
                tag_value="a" * 110,
                cuda_suffix="cu130",
                format_suffix="nydus",
            )

    def test_version_requires_ascii_digits(self) -> None:
        with self.assertRaisesRegex(SystemExit, "invalid vLLM version"):
            normalize_version("１２.27.0")

    def test_release_mode_does_not_resolve_unused_vllm_version(self) -> None:
        with patch(
            "get_byteiaas_image_tag.get_vllm_version",
            side_effect=AssertionError("must not resolve"),
        ) as get_version:
            self.assertEqual(get_vllm_version_for_mode("release", ""), "")
        get_version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
