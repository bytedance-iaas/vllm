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

    def test_every_runnable_child_must_match(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "manifests": [
                    {"digest": "sha256:zstd"},
                    {"digest": "sha256:gzip"},
                ],
            },
            "registry.example.com/serving/vllm@sha256:zstd": {
                "layers": [
                    {"mediaType": "application/vnd.oci.image.layer.v1.tar+zstd"}
                ],
            },
            "registry.example.com/serving/vllm@sha256:gzip": {
                "layers": [
                    {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}
                ],
            },
        }

        with self.assertRaisesRegex(ValueError, "every runnable manifest"):
            verify_image_format(
                "registry.example.com/serving/vllm:zstd",
                "zstd",
                inspect=manifests.__getitem__,
            )

    def test_attestation_manifest_is_ignored(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "manifests": [
                    {
                        "digest": "sha256:image",
                        "platform": {"os": "linux", "architecture": "amd64"},
                    },
                    {
                        "digest": "sha256:attestation",
                        "platform": {"os": "unknown", "architecture": "unknown"},
                        "annotations": {
                            "vnd.docker.reference.type": "attestation-manifest"
                        },
                    },
                ],
            },
            "registry.example.com/serving/vllm@sha256:image": {
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

    def test_shared_child_is_not_a_cycle(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "manifests": [
                    {"digest": "sha256:shared"},
                    {"digest": "sha256:shared"},
                ],
            },
            "registry.example.com/serving/vllm@sha256:shared": {
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

    def test_empty_or_attestation_only_index_fails(self) -> None:
        for manifests in (
            {"registry.example.com/serving/vllm:zstd": {"manifests": []}},
            {
                "registry.example.com/serving/vllm:zstd": {
                    "manifests": [
                        {
                            "digest": "sha256:attestation",
                            "artifactType": "application/vnd.in-toto+json",
                        }
                    ]
                }
            },
        ):
            with (
                self.subTest(manifests=manifests),
                self.assertRaisesRegex(
                    ValueError,
                    "no layers or child manifests|no runnable child manifests",
                ),
            ):
                verify_image_format(
                    "registry.example.com/serving/vllm:zstd",
                    "zstd",
                    inspect=manifests.__getitem__,
                )

    def test_actual_manifest_cycle_fails(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "manifests": [{"digest": "sha256:child"}],
            },
            "registry.example.com/serving/vllm@sha256:child": {
                "manifests": [{"digest": "sha256:root"}],
            },
            "registry.example.com/serving/vllm@sha256:root": {
                "manifests": [{"digest": "sha256:child"}],
            },
        }

        with self.assertRaisesRegex(ValueError, "manifest cycle detected"):
            verify_image_format(
                "registry.example.com/serving/vllm@sha256:root",
                "zstd",
                inspect=manifests.__getitem__,
            )

    def test_missing_format_marker_fails(self) -> None:
        manifest = {
            "registry.example.com/serving/vllm:zstd": {
                "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}]
            }
        }

        with self.assertRaisesRegex(ValueError, "expected zstd markers"):
            verify_image_format(
                "registry.example.com/serving/vllm:zstd",
                "zstd",
                inspect=manifest.__getitem__,
            )

    def test_similar_marker_names_do_not_pass(self) -> None:
        manifests: dict[str, dict[str, Any]] = {
            "registry.example.com/serving/vllm:zstd": {
                "layers": [
                    {"mediaType": "application/vnd.example.not-zstd"},
                ],
            },
            "registry.example.com/serving/vllm:nydus": {
                "layers": [
                    {
                        "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                        "annotations": {
                            "example.com/nydus-disabled": "false",
                            "containerd.io/snapshot/nydus-blob": "false",
                        },
                    }
                ],
            },
        }

        for reference, image_format in (
            ("registry.example.com/serving/vllm:zstd", "zstd"),
            ("registry.example.com/serving/vllm:nydus", "nydus"),
        ):
            with (
                self.subTest(reference=reference),
                self.assertRaisesRegex(ValueError, f"expected {image_format} markers"),
            ):
                verify_image_format(
                    reference,
                    image_format,
                    inspect=manifests.__getitem__,
                )


if __name__ == "__main__":
    unittest.main()
