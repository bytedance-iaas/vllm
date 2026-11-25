# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the PCIe Magic kernels from DeepEP PR #821."""

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
project_dir = Path(__file__).resolve().parent
source_dir = Path(os.environ["DEEPEP_PCIE_SOURCE_DIR"]).resolve()
nccl_dir = Path(
    os.environ.get(
        "EP_NCCL_ROOT_DIR",
        str(Path(torch.__file__).resolve().parent.parent / "nvidia/nccl"),
    )
)
cuda_dir = Path(CUDA_HOME or "/usr/local/cuda")
# These component headers are needed by the C++ Torch bindings. Passing them
# to nvcc can override its matching runtime headers with a pip CUDA version.
component_headers = [f"-I{path}" for path in nccl_dir.parent.glob("*/include")]

setup(
    name="deep-ep-magic-sm120",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="deep_ep_magic_C",
            sources=[
                str(project_dir / "bindings.cpp"),
                str(source_dir / "csrc/kernels/magic/runtime.cu"),
            ],
            include_dirs=[
                str(source_dir / "deep_ep/include"),
                str(source_dir),
                str(nccl_dir / "include"),
                str(cuda_dir / "include/cccl"),
            ],
            library_dirs=[str(nccl_dir / "lib")],
            extra_link_args=[
                f"-Wl,-rpath,{nccl_dir / 'lib'}",
                "-l:libnccl.so.2",
            ],
            extra_compile_args={
                "cxx": ["-std=c++20", "-O2", *component_headers],
                "nvcc": ["-std=c++20", "-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
