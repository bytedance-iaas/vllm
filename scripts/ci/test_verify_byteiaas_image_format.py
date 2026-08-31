#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_byteiaas_image_format import image_repository, verify_image_format


class VerifyByteiaasImageFormatTest(unittest.TestCase):
    def test_image_repository_preserves_registry_port(self) -> None:
        self.assertEqual(
            image_repository("registry.example.com:5000/serving/vllm:tag"),
            "registry.example.com:5000/serving/vllm",
        )
        self.assertEqual(
            image_repository("registry.example.com/serving/vllm@sha256:index"),
            "registry.example.com/serving/vllm",
        )

    def test_zstd_verification_follows_oci_index(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [{"digest": "sha256:image"}],
            },
            "registry.example.com/serving/vllm@sha256:image": {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "layers": [
                    {"mediaType": "application/vnd.oci.image.layer.v1.tar+zstd"}
                ],
            },
        }

        markers = verify_image_format(
            "registry.example.com/serving/vllm:zstd",
            "zstd",
            inspect=manifests.__getitem__,
        )

        self.assertEqual(markers, ["application/vnd.oci.image.layer.v1.tar+zstd"])

    def test_nydus_verification_follows_nested_indexes(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:nydus": {
                "manifests": [{"digest": "sha256:platform"}],
            },
            "registry.example.com/serving/vllm@sha256:platform": {
                "manifests": [{"digest": "sha256:image"}],
            },
            "registry.example.com/serving/vllm@sha256:image": {
                "layers": [
                    {
                        "mediaType": "application/vnd.oci.image.layer.v1.tar",
                        "annotations": {"containerd.io/snapshot/nydus-blob": "true"},
                    }
                ]
            },
        }

        markers = verify_image_format(
            "registry.example.com/serving/vllm:nydus",
            "nydus",
            inspect=manifests.__getitem__,
        )

        self.assertIn("containerd.io/snapshot/nydus-blob", markers)

    def test_missing_format_marker_fails(self) -> None:
        manifest = {
            "registry.example.com/serving/vllm:zstd": {
                "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}]
            }
        }

        with self.assertRaisesRegex(ValueError, "expected zstd layer media types"):
            verify_image_format(
                "registry.example.com/serving/vllm:zstd",
                "zstd",
                inspect=manifest.__getitem__,
            )


if __name__ == "__main__":
    unittest.main()
