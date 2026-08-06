#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Statically validate DeepGEMM's SM90 MegaMoE source contract."""

import argparse
import ast
import sys
from pathlib import Path

PUBLIC_APIS = {
    "fp8_fp4_mega_moe",
    "fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "transform_sf_into_required_layout",
    "transform_weights_for_mega_moe_sm90",
    "transform_weights_for_mega_moe_sm90_fp4",
}
MEGA_WRAPPERS = PUBLIC_APIS - {"transform_sf_into_required_layout"}
REQUIRED_TOKENS = {
    "csrc/python_api.cpp": {
        '"apis/sm90_mega.hpp"',
        "deep_gemm::mega::register_sm90_apis(m)",
    },
    "csrc/apis/layout.hpp": {
        'm.def("transform_sf_into_required_layout"',
    },
    "csrc/apis/sm90_mega.hpp": {
        'm.def("fp8_fp4_mega_moe_sm90"',
        'm.def("fp8_mega_moe"',
    },
    "deep_gemm/mega/__init__.py": {
        "_C.fp8_fp4_mega_moe_sm90",
        "_C.fp8_mega_moe",
    },
}


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc


def _imported_names(tree: ast.Module) -> set[str]:
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _defined_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def check_source(source_dir: Path) -> list[str]:
    source_dir = source_dir.resolve()
    init_path = source_dir / "deep_gemm" / "__init__.py"
    mega_path = source_dir / "deep_gemm" / "mega" / "__init__.py"
    errors = []

    try:
        public_names = _imported_names(_parse(init_path))
        missing = sorted(PUBLIC_APIS - public_names)
        if missing:
            errors.append(f"{init_path}: missing public imports: {', '.join(missing)}")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        wrappers = _defined_functions(_parse(mega_path))
        missing = sorted(MEGA_WRAPPERS - wrappers)
        if missing:
            errors.append(f"{mega_path}: missing wrappers: {', '.join(missing)}")
    except ValueError as exc:
        errors.append(str(exc))

    for relative_path, tokens in REQUIRED_TOKENS.items():
        path = source_dir / relative_path
        try:
            content = path.read_text()
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        missing = sorted(token for token in tokens if token not in content)
        if missing:
            errors.append(f"{path}: missing source contracts: {', '.join(missing)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()

    errors = check_source(args.source_dir)
    if errors:
        print("DeepGEMM source is missing required SM90 MegaMoE APIs:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"DeepGEMM SM90 MegaMoE source contract verified: {args.source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
