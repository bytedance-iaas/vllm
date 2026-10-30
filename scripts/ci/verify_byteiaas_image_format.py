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
ZSTD_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+zstd"
NYDUS_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.nydus.blob.v1"
NYDUS_MANIFEST_ARTIFACT_TYPE = "application/vnd.nydus.image.manifest.v1+json"
NYDUS_LAYER_ANNOTATIONS = {
    "containerd.io/snapshot/nydus-blob",
    "containerd.io/snapshot/nydus-bootstrap",
}


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


def is_attestation_descriptor(descriptor: Manifest) -> bool:
    artifact_type = str(descriptor.get("artifactType", "")).lower()
    if "in-toto" in artifact_type or "attestation" in artifact_type:
        return True
    annotations = descriptor.get("annotations")
    if isinstance(annotations, dict):
        reference_type = annotations.get("vnd.docker.reference.type", "")
        if "attestation" in str(reference_type).lower():
            return True

    platform = descriptor.get("platform")
    if isinstance(platform, dict):
        return (
            platform.get("os") == "unknown" or platform.get("architecture") == "unknown"
        )

    return False


def select_child_descriptors(
    descriptors: list[Manifest],
    image_format: str | None,
) -> list[Manifest]:
    candidates = [
        descriptor
        for descriptor in descriptors
        if not is_attestation_descriptor(descriptor)
    ]
    if image_format == "nydus":
        nydus_artifacts = [
            descriptor
            for descriptor in candidates
            if descriptor.get("artifactType") == NYDUS_MANIFEST_ARTIFACT_TYPE
        ]
        if nydus_artifacts:
            return nydus_artifacts
    elif image_format == "zstd":
        candidates = [
            descriptor
            for descriptor in candidates
            if not descriptor.get("artifactType")
        ]
    return candidates


def collect_layer_sets(
    reference: str,
    inspect: InspectManifest = inspect_manifest,
    ancestors: frozenset[str] = frozenset(),
    image_format: str | None = None,
) -> list[list[Manifest]]:
    if reference in ancestors:
        raise ValueError(f"manifest cycle detected at {reference}")

    manifest = inspect(reference)
    layers = manifest.get("layers")
    if isinstance(layers, list):
        return [layers]

    repository = image_repository(reference)
    raw_descriptors = manifest.get("manifests")
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        raise ValueError(f"manifest {reference} has no layers or child manifests")

    descriptors = select_child_descriptors(
        [descriptor for descriptor in raw_descriptors if isinstance(descriptor, dict)],
        image_format,
    )
    collected: list[list[Manifest]] = []
    child_ancestors = ancestors | {reference}
    for descriptor in descriptors:
        digest = descriptor.get("digest", "")
        if not digest:
            raise ValueError(f"runnable child manifest in {reference} has no digest")
        collected.extend(
            collect_layer_sets(
                f"{repository}@{digest}",
                inspect=inspect,
                ancestors=child_ancestors,
                image_format=image_format,
            )
        )

    if not collected:
        raise ValueError(f"manifest {reference} has no runnable child manifests")
    return collected


def collect_layers(
    reference: str,
    inspect: InspectManifest = inspect_manifest,
) -> list[Manifest]:
    return [
        layer
        for layer_set in collect_layer_sets(reference, inspect=inspect)
        for layer in layer_set
    ]


def layer_markers(layers: list[Manifest]) -> tuple[set[str], dict[str, set[str]]]:
    media_types = {
        str(layer.get("mediaType", "")) for layer in layers if layer.get("mediaType")
    }
    annotations_by_key: dict[str, set[str]] = {}
    for layer in layers:
        annotations = layer.get("annotations")
        if isinstance(annotations, dict):
            for key, value in annotations.items():
                annotations_by_key.setdefault(str(key), set()).add(str(value))
    return media_types, annotations_by_key


def has_requested_format(layers: list[Manifest], image_format: str) -> bool:
    media_types, annotations_by_key = layer_markers(layers)
    if image_format == "zstd":
        return bool(layers) and all(
            layer.get("mediaType") == ZSTD_LAYER_MEDIA_TYPE for layer in layers
        )

    if NYDUS_LAYER_MEDIA_TYPE in media_types:
        return True
    return any(
        annotations_by_key.get(key, set()) == {"true"}
        or "true" in annotations_by_key.get(key, set())
        for key in NYDUS_LAYER_ANNOTATIONS
    )


def verify_image_format(
    reference: str,
    image_format: str,
    inspect: InspectManifest = inspect_manifest,
) -> list[str]:
    if image_format not in {"zstd", "nydus"}:
        raise ValueError(f"unsupported image format: {image_format}")

    all_markers: set[str] = set()
    for child_index, layers in enumerate(
        collect_layer_sets(reference, inspect=inspect, image_format=image_format),
        start=1,
    ):
        media_types, annotations_by_key = layer_markers(layers)
        if not has_requested_format(layers, image_format):
            raise ValueError(
                f"expected {image_format} markers for every runnable manifest "
                f"in {reference}; child {child_index} has media types "
                f"{sorted(media_types)} and annotations {annotations_by_key}"
            )
        all_markers.update(media_types)
        all_markers.update(annotations_by_key)

    return sorted(all_markers)


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
