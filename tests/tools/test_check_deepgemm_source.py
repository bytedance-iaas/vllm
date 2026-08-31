# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest

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
    transform_weights_for_mega_moe,
    transform_weights_for_mega_moe_sm90,
    transform_weights_for_mega_moe_sm90_fp4,
)
""",
        "deep_gemm/mega/__init__.py": """
class SymmBuffer:
    def __init__(self):
        self.buffer_size_fn = _C.get_symm_buffer_size_for_sm90_mega_moe

def get_symm_buffer_for_mega_moe(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch=True,
    activation="swiglu",
):
    alignment_fn = _C.get_token_alignment_for_sm90_mega_moe
    return alignment_fn

def transform_weights_for_mega_moe(l1_weights, l2_weights):
    pass

def transform_weights_for_mega_moe_sm90(l1_weights, l2_weights):
    pass

def transform_weights_for_mega_moe_sm90_fp4(l1_weights, l2_weights):
    pass

def fp8_fp4_mega_moe(
    y,
    l1_weights,
    l2_weights,
    sym_buffer,
    recipe=(1, 1, 32),
    activation="swiglu",
    activation_clamp=None,
    fast_math=True,
):
    return _C.fp8_fp4_mega_moe_sm90

def fp8_mega_moe(
    y,
    l1_weights,
    l2_weights,
    sym_buffer,
    recipe=(128, 128, 128),
    activation="swiglu",
    activation_clamp=None,
    fast_math=True,
):
    return _C.fp8_mega_moe
""",
        "csrc/python_api.cpp": """
#include "apis/sm90_mega.hpp"
deep_gemm::mega::register_sm90_apis(m);
""",
        "csrc/apis/layout.hpp": """
m.def(
    "transform_sf_into_required_layout",
    &transform_sf_into_required_layout,
    py::arg("sf"),
    py::arg("mn"),
    py::arg("k"),
    py::arg("recipe"),
    py::arg("num_groups"),
    py::arg("is_sfa") = None,
    py::arg("disable_ue8m0_cast"));
""",
        "csrc/apis/sm90_mega.hpp": """
m.def("get_token_alignment_for_sm90_mega_moe",
      &get_token_alignment_for_sm90_mega_moe);
m.def("get_symm_buffer_size_for_sm90_mega_moe",
      &get_symm_buffer_size_for_sm90_mega_moe);
m.def("fp8_fp4_mega_moe_sm90", &fp8_fp4_mega_moe_sm90);
m.def("fp8_mega_moe", &fp8_mega_moe);
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
    binding.write_text(
        binding.read_text().replace(
            'm.def("fp8_fp4_mega_moe_sm90", &fp8_fp4_mega_moe_sm90);\n',
            '// m.def("fp8_fp4_mega_moe_sm90", &fp8_fp4_mega_moe_sm90);\n',
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "fp8_fp4_mega_moe_sm90" in result.stderr


@pytest.mark.parametrize(
    "binding_name",
    [
        "get_token_alignment_for_sm90_mega_moe",
        "get_symm_buffer_size_for_sm90_mega_moe",
    ],
)
def test_rejects_missing_sm90_buffer_binding(
    tmp_path: Path,
    binding_name: str,
) -> None:
    _write_source_tree(tmp_path)
    binding = tmp_path / "csrc" / "apis" / "sm90_mega.hpp"
    binding.write_text(
        binding.read_text().replace(
            f'm.def("{binding_name}",\n'
            f"      &{binding_name});\n",
            f'// m.def("{binding_name}", &{binding_name});\n',
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert binding_name in result.stderr


def test_rejects_incompatible_runtime_wrapper_signature(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    mega_path = tmp_path / "deep_gemm" / "mega" / "__init__.py"
    mega_path.write_text(
        mega_path.read_text().replace(
            "def fp8_mega_moe(\n"
            "    y,\n"
            "    l1_weights,\n"
            "    l2_weights,\n"
            "    sym_buffer,\n"
            "    recipe=(128, 128, 128),\n"
            '    activation="swiglu",\n'
            "    activation_clamp=None,\n"
            "    fast_math=True,\n"
            "):\n",
            "def fp8_mega_moe(\n"
            "    recipe,\n"
            "    activation,\n"
            "    activation_clamp,\n"
            "    fast_math,\n"
            "    **kwargs,\n"
            "):\n",
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "incompatible wrapper signature for fp8_mega_moe" in result.stderr


def test_rejects_unreachable_sm90_buffer_api_use(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    mega_path = tmp_path / "deep_gemm" / "mega" / "__init__.py"
    mega_path.write_text(
        mega_path.read_text().replace(
            "        self.buffer_size_fn = "
            "_C.get_symm_buffer_size_for_sm90_mega_moe\n",
            "        return\n"
            "        self.buffer_size_fn = "
            "_C.get_symm_buffer_size_for_sm90_mega_moe\n",
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "SymmBuffer.__init__ is missing reachable _C APIs" in result.stderr


def test_rejects_sm90_buffer_api_use_in_constant_false_branch(
    tmp_path: Path,
) -> None:
    _write_source_tree(tmp_path)
    mega_path = tmp_path / "deep_gemm" / "mega" / "__init__.py"
    mega_path.write_text(
        mega_path.read_text().replace(
            "        self.buffer_size_fn = "
            "_C.get_symm_buffer_size_for_sm90_mega_moe\n",
            "        if False:\n"
            "            self.buffer_size_fn = "
            "_C.get_symm_buffer_size_for_sm90_mega_moe\n",
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "SymmBuffer.__init__ is missing reachable _C APIs" in result.stderr


def test_rejects_incompatible_layout_binding_signature(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    binding = tmp_path / "csrc" / "apis" / "layout.hpp"
    binding.write_text(
        binding.read_text().replace(
            '    py::arg("disable_ue8m0_cast"));\n',
            ");\n",
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "transform_sf_into_required_layout cannot accept" in result.stderr


def test_rejects_required_layout_binding_optional_argument(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    binding = tmp_path / "csrc" / "apis" / "layout.hpp"
    binding.write_text(
        binding.read_text().replace(
            '    py::arg("is_sfa") = None,\n',
            '    py::arg("is_sfa"),\n',
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "transform_sf_into_required_layout cannot accept" in result.stderr


def test_rejects_missing_sm100_weight_transform(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    init_path = tmp_path / "deep_gemm" / "__init__.py"
    init_path.write_text(
        init_path.read_text().replace(
            "    transform_weights_for_mega_moe,\n",
            "",
        )
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "transform_weights_for_mega_moe" in result.stderr
