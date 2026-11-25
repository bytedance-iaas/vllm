# DeepEP v2 SM120 PCIe kernels

This builds the PCIe Magic implementation from DeepEP PR #821 as a separate
extension, while retaining the ElasticBuffer-based DeepEP v2 package used by
vLLM v0.33.0. The pinned upstream revision is
cd63ec7f79b27192675700b5ede4c30b8c41f783. The native kernel code remains in that
checkout. [Upstream implementation](https://github.com/deepseek-ai/DeepEP/pull/821)

Set DEEPEP_PCIE_SOURCE_DIR to the pinned checkout and EP_MAGIC_BUILD_DIR to a
build directory in the execution host's existing task session.

~~~bash
DEEPEP_PCIE_SOURCE_DIR=/path/to/DeepEP-pr821 \
uv run --no-project --python .venv/bin/python python \
  tools/ep_kernels/deepep_sm120_pcie/setup.py build_ext \
  --build-lib "$EP_MAGIC_BUILD_DIR/lib" \
  --build-temp "$EP_MAGIC_BUILD_DIR/objects"

PYTHONPATH="$EP_MAGIC_BUILD_DIR/lib:$PYTHONPATH" \
uv run --no-project --python .venv/bin/python python -m torch.distributed.run \\\n  --standalone --nproc-per-node 8 \
  tools/ep_kernels/deepep_sm120_pcie/validate.py
~~~

Run with the same Torch, NCCL and CUDA runtime used by the serving container.
The build defaults to SM120 and supports CPU-only compilation. CUDA/NCCL
headers and a CUDA toolkit are required.

For serving, put the built extension on PYTHONPATH, set
VLLM_DEEPEP_V2_USE_PCIE=1 and VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=0, then select
--all2all-backend deepep_v2. NCCL GIN must be available. On the tested TK300
host NCCL_GIN_TYPE=3 was used, together with explicit RDMA device passthrough.

The adapter implements the non-expanded Decode layout used with CUDA Graphs.
The validation covers BF16, packed MXFP8 scales, an idle route rank, and
CUDA Graph capture/replay on eight GPUs.
