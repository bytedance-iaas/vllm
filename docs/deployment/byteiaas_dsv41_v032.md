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
  --build-arg VLLM_BUILD_COMMIT="$(git rev-parse HEAD)" \
  -t "$BUILT_IMAGE" .
```

The DeepGEMM lock reconstructs SM90 MegaMoE on release base `1e1842a8`.
The Humming lock reconstructs the unified Prefill/Decode implementation on
upstream main `f970d0b3`. Both locks include patch SHA256 and the exact
resulting Git tree. Local source commit IDs are attribution; remote fetches
use upstream base commits and apply the checked-in patches.

These sources produce DeepGEMM `2.8.0+local` and Humming
`0.1.18+unified.g596bb89`. The Humming build includes its launcher,
device-info extension, cubin patcher, NVRTC binary, and native manifest.
Tokenizers is fixed to `441393c7`; the encoder-only router is fixed to
`ef339bc7`.

The release manylinux_2_28 builder replaces earlier AlmaLinux and dependency
bootstrap iterations. The runtime follows release Ubuntu 24.04 defaults.
Internal workflows keep the HTTP/index configuration and local Docker Hub
mirror, but no longer pass obsolete DeepGEMM fork pin or AlmaLinux/Ubuntu
mirror arguments to the release Dockerfile.

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
