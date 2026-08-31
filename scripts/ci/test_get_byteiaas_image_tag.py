#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_byteiaas_image_tag import build_tag


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


if __name__ == "__main__":
    unittest.main()
