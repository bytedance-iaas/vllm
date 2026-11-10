# ByteIAAS DeepSeek V4.1 Flash on v0.32.0

This port starts from vLLM `v0.32.0` (`5df63a7203`) and keeps its native
build contracts. The runtime requires Torch 2.13.0, CUDA 13.0.3, Python 3.12,
and SM90 hardware for the internal MegaMoE path.

## Build and source identity

Build from a checkout containing the ported commits and a complete Git
directory. Run the inexpensive contract check before building:

```bash
.venv/bin/python tools/byteiaas/validate_build_contract.py
docker build --target vllm-openai -f docker/Dockerfile \
  --build-arg BYTEIAAS_BUILD_NATIVE=true \
  --build-arg BYTEIAAS_BUILD_TOKENIZERS=true \
  --build-arg BYTEIAAS_BUILD_ROUTER=true \
  --build-arg DEEPGEMM_GIT_COMMIT="$(.venv/bin/python tools/byteiaas/prepare_native_source.py \
    --config docker/byteiaas/native/deepgemm/source.json resolve)" \
  --build-arg HUMMING_GIT_COMMIT="$(.venv/bin/python tools/byteiaas/prepare_native_source.py \
    --config docker/byteiaas/native/humming/source.json resolve)" \
  --build-arg VLLM_ROUTER_GIT_COMMIT="$(.venv/bin/python tools/byteiaas/router_wheel.py resolve)" \
  --build-arg BYTEIAAS_MOONCAKE_WHEEL_SOURCE=mooncake-build \
  --build-arg BYTEIAAS_REQUIRE_SOURCE_MOONCAKE=true \
  --build-arg INSTALL_NIXL=false \
  --build-arg BUILD_NIXL=false \
  --build-arg BYTEIAAS_NIXL_WHEEL_SOURCE=nixl-empty \
  --build-arg VLLM_BUILD_COMMIT="$(git rev-parse HEAD)" \
  -t "$BUILT_IMAGE" .
```

The DeepGEMM, Humming, and router builds follow their ByteIAAS branches.
The wheel workflow resolves each branch once and builds the selected commit.
The native manifests record the repository, branch, commit, Git tree, and
wheel checksum. The router manifest records the commit and Cargo.lock digest.

DeepGEMM retains its SM90 MegaMoE operators on the 2.8 baseline. Humming
provides the unified Prefill and Decode kernels and includes the launcher,
device-info extension, cubin patcher, NVRTC binary, and native manifest.
Tokenizers is fixed to `441393c7`. Mooncake is fixed to `ffd73707`, including the PR #4208 backport,
and built with `STORE_USE_JEMALLOC=ON`. The image enables jemalloc's background
purge thread with `MALLOC_CONF=background_thread:true`.

The release manylinux_2_28 builder replaces earlier AlmaLinux and dependency
bootstrap iterations. The runtime follows release Ubuntu 24.04 defaults.
Internal workflows pass regional Ubuntu/AlmaLinux/EPEL and Python indexes
to the build and runtime stages, together with the local Docker Hub mirror.
The `mooncake-system-deps` target installs fresh Ubuntu dependencies before
native compilation on the same CI builder. APT retries transient downloads
and rejects incomplete index updates. Python bootstrap prefers distribution
packages and installs pinned uv 0.11.1 from the configured Python index.
Supplemental indexes use `UV_INDEX` so PyTorch's command-line
`--extra-index-url` cannot hide the public PyPI fallback when the regional
mirror has not yet synchronized a release. CI checks the known supplemental
index requirement for Python 3.12/manylinux_2_28 before building images.
Build contract 5 reuses the branch snapshots through same-run artifacts.
The wheel workflow uploads vLLM, DeepGEMM, unified Humming, DeepEP, patched
tokenizers, router, and Mooncake wheels with source/provenance manifests and
checksums. The image workflows download that same run's artifact and install
its wheels without invoking the component builders again. The optional
Triton source build has an explicit disabled marker. NIXL is neither built
nor included in the artifact; internal KV dependency resolution excludes
NIXL and its transitive variants.

## Prefill and Decode use the same Humming package

Use the same built image for both roles. Remove the previous role-specific
paths `/opt/humming-dsv41-prefill-3632a505` and `/opt/humming-0.1.12` from
`PYTHONPATH`. A higher-priority source directory would override the installed
package and bypass the verified wheel.

For the ServingKit `deepseek-v4.1-flash-offload` chart, render the built image
with those two paths cleared:

```bash
helm template dsv41-flash \
  vllm/deepseek/deepseek-v4.1-flash-offload -n wyc \
  --set-string global.image="$BUILT_IMAGE" \
  --set-json 'prefill.extraEnv.PYTHONPATH=null' \
  --set-json 'decode.extraEnv.PYTHONPATH=null'
```

Keep the encoder-only Prefill, weight pruning, PP layer split, PCP alignment,
cache-only router, DSpark, CUDA graph, bounded replay, and canonical Store
settings from the validated profile. The encoder-only transfer contract is
specific to the supported PP2 producer/PP1 consumer layout. In heterogeneous
PP transfer, preserve each layer's exact region instead of coalescing layers
into one transfer span.

## Verify the installed runtime

The build retains `/opt/byteiaas/image-metadata.json` and per-package
installed-file proofs under `/opt/byteiaas/provenance`. Check that both roles
report the same Humming version, source tree, wheel hash, and Torch/ABI
contract. Devel images retain the base image identity while recording their
own tag and build identity.

The image installation verifies:

- DeepGEMM SM90 MegaMoE native operator schemas.
- Humming native artifacts, ordered scheduler and input descriptors.
- Tokenizers native bytes and `DecodeStream.prefill_pending` behavior.
- Router native bytes and `--encoder-only-prefill`.

These checks establish package identity and interfaces. Validate model
outputs and serving performance separately on the target hardware. Use a
text/image smoke first, followed by controlled long-prefix and cancellation
checks. Record first-use compilation separately from warm latency.
