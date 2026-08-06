# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_deepgemm_source.py"


def _write_source_tree(root: Path) -> None:
    files = {
        "deep_gemm/__init__.py": """
from ._C import transform_sf_into_required_layout
from .mega import (
    fp8_fp4_mega_moe,
    fp8_mega_moe,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe_sm90,
    transform_weights_for_mega_moe_sm90_fp4,
)
""",
        "deep_gemm/mega/__init__.py": """
def get_symm_buffer_for_mega_moe():
    pass

def transform_weights_for_mega_moe_sm90():
    pass

def transform_weights_for_mega_moe_sm90_fp4():
    pass

def fp8_fp4_mega_moe():
    return _C.fp8_fp4_mega_moe_sm90

def fp8_mega_moe():
    return _C.fp8_mega_moe
""",
        "csrc/python_api.cpp": """
#include "apis/sm90_mega.hpp"
deep_gemm::mega::register_sm90_apis(m);
""",
        "csrc/apis/layout.hpp": """
m.def("transform_sf_into_required_layout", &transform);
""",
        "csrc/apis/sm90_mega.hpp": """
m.def("fp8_fp4_mega_moe_sm90", &fp8_fp4);
m.def("fp8_mega_moe", &fp8);
""",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _run_checker(source_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(source_dir)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_complete_sm90_mega_moe_source_contract(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)

    result = _run_checker(tmp_path)

    assert result.returncode == 0
    assert "source contract verified" in result.stdout


def test_rejects_missing_sm90_fp8_fp4_dispatch_binding(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    binding = tmp_path / "csrc" / "apis" / "sm90_mega.hpp"
    binding.write_text('m.def("fp8_mega_moe", &fp8);\n')

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "fp8_fp4_mega_moe_sm90" in result.stderr
