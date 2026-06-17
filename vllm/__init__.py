# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

def _patch_cutlass_dsl_mlir_global_dtors() -> None:
    """Patch CUTLASS DSL/MLIR global destructor API drift at import time.

    Some CUDA 13 CUTLASS DSL wheels call llvm.mlir_global_dtors with the old
    two-argument form while the bundled MLIR binding requires the newer `data`
    attribute. Patch the provider before CUTLASS kernels are compiled in worker
    subprocesses.
    """
    import site
    import sysconfig
    from pathlib import Path

    roots = set()
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(path)
    try:
        roots.update(site.getsitepackages())
    except AttributeError:
        pass

    global_dtors_marker = """                global_dtors = llvm.mlir_global_dtors(
                    dtors=[],
                    priorities=[],
                )
"""
    global_dtors_replacement = """                global_dtors = llvm.mlir_global_dtors(
                    dtors=[],
                    priorities=[],
                    data=[],
                )
"""
    global_dtors_append_marker = """        global_dtors.attributes["priorities"] += [
            ir.IntegerAttr.get(self.i32_type, 65535)
        ]  # the default priority
        return current_block
"""
    global_dtors_append_replacement = """        global_dtors.attributes["priorities"] += [
            ir.IntegerAttr.get(self.i32_type, 65535)
        ]  # the default priority
        global_dtors.attributes["data"] += [
            ir.FlatSymbolRefAttr.get(unload_func_wrapper_symbol)
        ]
        return current_block
"""

    for root in roots:
        provider_path = (
            Path(root)
            / "nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/tvm_ffi_provider.py"
        )
        if not provider_path.exists():
            continue

        text = provider_path.read_text()
        patched = False
        if global_dtors_marker in text:
            text = text.replace(global_dtors_marker, global_dtors_replacement, 1)
            patched = True
        if global_dtors_append_marker in text:
            text = text.replace(
                global_dtors_append_marker,
                global_dtors_append_replacement,
                1,
            )
            patched = True
        if patched:
            provider_path.write_text(text)
        return


_patch_cutlass_dsl_mlir_global_dtors()


MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
