#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Statically validate DeepGEMM's SM90 MegaMoE source contract."""

import argparse
import ast
import re
import sys
from pathlib import Path

PUBLIC_APIS = {
    "fp8_fp4_mega_moe",
    "fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "transform_sf_into_required_layout",
    "transform_weights_for_mega_moe",
    "transform_weights_for_mega_moe_sm90",
    "transform_weights_for_mega_moe_sm90_fp4",
}
MEGA_WRAPPER_CALLS = {
    "fp8_fp4_mega_moe": (
        4,
        frozenset({"recipe", "activation", "activation_clamp", "fast_math"}),
    ),
    "fp8_mega_moe": (
        4,
        frozenset({"recipe", "activation", "activation_clamp", "fast_math"}),
    ),
    "get_symm_buffer_for_mega_moe": (
        6,
        frozenset({"use_fp8_dispatch", "activation"}),
    ),
    "transform_weights_for_mega_moe": (2, frozenset()),
    "transform_weights_for_mega_moe_sm90": (2, frozenset()),
    "transform_weights_for_mega_moe_sm90_fp4": (2, frozenset()),
}
REQUIRED_CPP_TOKENS = {
    "csrc/python_api.cpp": {
        '"apis/sm90_mega.hpp"',
        "deep_gemm::mega::register_sm90_apis(m)",
    },
}
CPP_BINDINGS = {
    "csrc/apis/layout.hpp": {
        "transform_sf_into_required_layout": (
            "transform_sf_into_required_layout",
            (5, frozenset({"disable_ue8m0_cast"})),
        ),
    },
    "csrc/apis/sm90_mega.hpp": {
        "fp8_fp4_mega_moe_sm90": ("fp8_fp4_mega_moe_sm90", None),
        "fp8_mega_moe": ("fp8_mega_moe", None),
        "get_symm_buffer_size_for_sm90_mega_moe": (
            "get_symm_buffer_size_for_sm90_mega_moe",
            None,
        ),
        "get_token_alignment_for_sm90_mega_moe": (
            "get_token_alignment_for_sm90_mega_moe",
            None,
        ),
    },
}
C_API_USES = {
    ("function", "fp8_fp4_mega_moe"): {"fp8_fp4_mega_moe_sm90"},
    ("function", "fp8_mega_moe"): {"fp8_mega_moe"},
    ("function", "get_symm_buffer_for_mega_moe"): {
        "get_token_alignment_for_sm90_mega_moe"
    },
    ("method", "SymmBuffer", "__init__"): {
        "get_symm_buffer_size_for_sm90_mega_moe"
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


def _defined_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reachable_c_api_uses(
    body: list[ast.stmt],
) -> set[str]:
    def attributes(root: ast.AST) -> set[str]:
        return {
            child.attr
            for child in ast.walk(root)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "_C"
        }

    def constant_truth(node: ast.expr) -> bool | None:
        if isinstance(node, ast.Constant):
            return bool(node.value)
        return None

    def collect(statements: list[ast.stmt]) -> tuple[set[str], bool]:
        names = set()
        for statement in statements:
            if isinstance(statement, ast.If):
                names.update(attributes(statement.test))
                truth = constant_truth(statement.test)
                if truth is not None:
                    branch_names, terminates = collect(
                        statement.body if truth else statement.orelse
                    )
                else:
                    body_names, body_terminates = collect(statement.body)
                    else_names, else_terminates = collect(statement.orelse)
                    branch_names = body_names | else_names
                    terminates = (
                        bool(statement.orelse)
                        and body_terminates
                        and else_terminates
                    )
                names.update(branch_names)
                if terminates:
                    return names, True
                continue

            names.update(attributes(statement))
            if isinstance(statement, (ast.Raise, ast.Return)):
                return names, True
        return names, False

    return collect(body)[0]


def _find_c_api_scope(
    tree: ast.Module,
    scope: tuple[str, ...],
) -> list[ast.stmt] | None:
    if scope[0] == "function":
        function = _defined_functions(tree).get(scope[1])
        return function.body if function is not None else None

    _, class_name, method_name = scope
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        return None
    method = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ),
        None,
    )
    return method.body if method is not None else None


def _accepts_call(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    num_positional: int,
    keyword_names: frozenset[str],
) -> bool:
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if num_positional > len(positional) and arguments.vararg is None:
        return False

    num_bound_regular = min(
        max(num_positional - len(arguments.posonlyargs), 0),
        len(arguments.args),
    )
    bound_regular_names = {
        arg.arg for arg in arguments.args[:num_bound_regular]
    }
    if bound_regular_names & keyword_names:
        return False

    provided_names = {arg.arg for arg in positional[:num_positional]}
    provided_names.update(keyword_names)

    num_required_positional = len(positional) - len(arguments.defaults)
    required_names = {arg.arg for arg in positional[:num_required_positional]}
    if not required_names.issubset(provided_names):
        return False

    if arguments.kwarg is None:
        keyword_capable = {arg.arg for arg in arguments.args}
        keyword_capable.update(arg.arg for arg in arguments.kwonlyargs)
        if not keyword_names.issubset(keyword_capable):
            return False

    required_kwonly = {
        arg.arg
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is None
    }
    return required_kwonly.issubset(keyword_names)


def _strip_cpp_comments(content: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", content, flags=re.DOTALL)


def _find_cpp_binding(content: str, api_name: str) -> str | None:
    match = re.search(
        rf'\bm\s*\.\s*def\s*\(\s*"{re.escape(api_name)}"\s*,',
        content,
    )
    if match is None:
        return None
    end = content.find(";", match.end())
    return content[match.start() : end + 1] if end >= 0 else None


def _accepts_named_call(
    parameter_names: tuple[str, ...],
    default_names: frozenset[str],
    num_positional: int,
    keyword_names: frozenset[str],
) -> bool:
    if num_positional > len(parameter_names):
        return False
    bound_positionally = set(parameter_names[:num_positional])
    if bound_positionally & keyword_names:
        return False
    if not keyword_names.issubset(parameter_names):
        return False
    provided_names = bound_positionally | keyword_names
    required_names = set(parameter_names) - default_names
    return required_names.issubset(provided_names)


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
        mega_tree = _parse(mega_path)
        wrappers = _defined_functions(mega_tree)
        missing = sorted(MEGA_WRAPPER_CALLS.keys() - wrappers.keys())
        if missing:
            errors.append(f"{mega_path}: missing wrappers: {', '.join(missing)}")
        for name, (num_positional, keyword_names) in MEGA_WRAPPER_CALLS.items():
            function = wrappers.get(name)
            if function is not None and not _accepts_call(
                function, num_positional, keyword_names
            ):
                keywords = ", ".join(sorted(keyword_names)) or "none"
                errors.append(
                    f"{mega_path}: incompatible wrapper signature for {name}: "
                    f"requires {num_positional} positional arguments and "
                    f"keywords {keywords}"
                )
        for scope, required_names in C_API_USES.items():
            body = _find_c_api_scope(mega_tree, scope)
            found_names = _reachable_c_api_uses(body) if body is not None else set()
            missing_names = sorted(required_names - found_names)
            if missing_names:
                scope_name = ".".join(scope[1:])
                errors.append(
                    f"{mega_path}: {scope_name} is missing reachable _C APIs: "
                    f"{', '.join(missing_names)}"
                )
    except ValueError as exc:
        errors.append(str(exc))

    for relative_path, tokens in REQUIRED_CPP_TOKENS.items():
        path = source_dir / relative_path
        try:
            content = _strip_cpp_comments(path.read_text())
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        missing = sorted(token for token in tokens if token not in content)
        if missing:
            errors.append(f"{path}: missing source contracts: {', '.join(missing)}")

    for relative_path, bindings in CPP_BINDINGS.items():
        path = source_dir / relative_path
        try:
            content = _strip_cpp_comments(path.read_text())
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        for api_name, (symbol_name, required_call) in bindings.items():
            statement = _find_cpp_binding(content, api_name)
            if statement is None:
                errors.append(f"{path}: missing C++ binding for {api_name}")
                continue
            if re.search(rf"&(?:\w+::)*{re.escape(symbol_name)}\b", statement) is None:
                errors.append(
                    f"{path}: C++ binding {api_name} does not reference "
                    f"{symbol_name}"
                )
            if required_call is not None:
                parameter_names = tuple(
                    re.findall(r'py\s*::\s*arg\s*\(\s*"([^"]+)"', statement)
                )
                default_names = frozenset(
                    re.findall(
                        r'py\s*::\s*arg\s*\(\s*"([^"]+)"\s*\)\s*=',
                        statement,
                    )
                )
                num_positional, keyword_names = required_call
                if not _accepts_named_call(
                    parameter_names,
                    default_names,
                    num_positional,
                    keyword_names,
                ):
                    keywords = ", ".join(sorted(keyword_names))
                    errors.append(
                        f"{path}: C++ binding {api_name} cannot accept "
                        f"{num_positional} positional arguments and keywords "
                        f"{keywords}; parameters={parameter_names}, "
                        f"defaults={sorted(default_names)}"
                    )

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
