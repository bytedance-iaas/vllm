# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.machinery
import os
import sys
import types

# In symlink mode (VLLM_FLASH_ATTN_SRC_DIR), cute/ is a symlink to the real
# source tree and its files use `flash_attn.cute.*` imports (not rewritten).
# Register a virtual `flash_attn` package so those imports resolve.
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.islink(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    _fa_mod.__package__ = "flash_attn"
    _fa_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", None, is_package=True
    )
    _fa_mod.__spec__.submodule_search_locations = _fa_mod.__path__
    sys.modules["flash_attn"] = _fa_mod

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Keep C-extension imports lazy so `vllm.vllm_flash_attn.cute.*` users do
    # not need the FA2/FA3 extension when they only need CUTE helper modules.
    from vllm.vllm_flash_attn import flash_attn_interface

    value = getattr(flash_attn_interface, name)
    globals()[name] = value
    return value
