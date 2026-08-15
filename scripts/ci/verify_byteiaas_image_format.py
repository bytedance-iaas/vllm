#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from typing import Any

Manifest = dict[str, Any]
InspectManifest = Callable[[str], Manifest]


def image_repository(reference: str) -> str:
    if "@" in reference:
        return reference.rsplit("@", 1)[0]

    prefix, separator, name = reference.rpartition("/")
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    return f"{prefix}{separator}{name}"


def inspect_manifest(reference: str) -> Manifest:
    raw = subprocess.check_output(
        ["docker", "buildx", "imagetools", "inspect", "--raw", reference]
    )
    return json.loads(raw)


def collect_layers(
    reference: str,
    inspect: InspectManifest = inspect_manifest,
    seen: set[str] | None = None,
) -> list[Manifest]:
    if seen is None:
        seen = set()
    if reference in seen:
        raise ValueError(f"manifest cycle detected at {reference}")
    seen.add(reference)

    manifest = inspect(reference)
    layers = manifest.get("layers")
    if isinstance(layers, list):
        return layers

    repository = image_repository(reference)
    collected: list[Manifest] = []
    for descriptor in manifest.get("manifests", []):
        digest = descriptor.get("digest", "")
        if digest:
            collected.extend(
                collect_layers(f"{repository}@{digest}", inspect=inspect, seen=seen)
            )
    return collected


def verify_image_format(
    reference: str,
    image_format: str,
    inspect: InspectManifest = inspect_manifest,
) -> list[str]:
    layers = collect_layers(reference, inspect=inspect)
    media_types = sorted(
        {layer.get("mediaType", "") for layer in layers if layer.get("mediaType")}
    )

    if image_format == "zstd":
        if not any("zstd" in media_type.lower() for media_type in media_types):
            raise ValueError(
                f"expected zstd layer media types for {reference}, got {media_types}"
            )
        return media_types

    if image_format == "nydus":
        annotation_markers = {
            str(marker)
            for layer in layers
            for marker in (
                *(layer.get("annotations", {}) or {}).keys(),
                *(layer.get("annotations", {}) or {}).values(),
            )
        }
        markers = media_types + sorted(annotation_markers)
        if not any("nydus" in marker.lower() for marker in markers):
            raise ValueError(
                f"expected nydus markers for {reference}; "
                f"layers={media_types}, annotations={sorted(annotation_markers)}"
            )
        return markers

    raise ValueError(f"unsupported image format: {image_format}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ByteIAAS image layer formats in a remote registry."
    )
    parser.add_argument("--format", choices=["zstd", "nydus"], required=True)
    parser.add_argument("references", nargs="+")
    args = parser.parse_args()

    for reference in args.references:
        try:
            markers = verify_image_format(reference, args.format)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Verified {args.format} image {reference}: {markers}")


if __name__ == "__main__":
    main()
