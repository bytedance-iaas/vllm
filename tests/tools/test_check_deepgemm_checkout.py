# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_deepgemm_checkout.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "DeepGEMM checkout test")
    _git(repo, "config", "user.email", "deepgemm-checkout-test@example.com")


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_checkout(tmp_path: Path) -> tuple[Path, str]:
    submodule = tmp_path / "cutlass"
    _init_repo(submodule)
    first_submodule_commit = _commit_file(
        submodule, "version.txt", "v1\n", "cutlass v1"
    )
    _commit_file(submodule, "version.txt", "v2\n", "cutlass v2")

    checkout = tmp_path / "deepgemm"
    _init_repo(checkout)
    _commit_file(checkout, "README.md", "DeepGEMM\n", "initial source")
    _git(
        checkout,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule),
        "third-party/cutlass",
    )
    _git(checkout, "commit", "-am", "add cutlass")
    return checkout, first_submodule_commit


def _run_checker(checkout: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(checkout)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_clean_checkout_with_exact_initialized_submodules(
    tmp_path: Path,
) -> None:
    checkout, _ = _make_checkout(tmp_path)

    result = _run_checker(checkout)

    assert result.returncode == 0
    assert "clean and reusable" in result.stdout


def test_rejects_dirty_tracked_checkout(tmp_path: Path) -> None:
    checkout, _ = _make_checkout(tmp_path)
    (checkout / "README.md").write_text("modified\n")

    result = _run_checker(checkout)

    assert result.returncode == 1
    assert "worktree is dirty" in result.stderr
    assert "README.md" in result.stderr


def test_rejects_uninitialized_submodule(tmp_path: Path) -> None:
    checkout, _ = _make_checkout(tmp_path)
    _git(checkout, "submodule", "deinit", "--force", "--all")

    result = _run_checker(checkout)

    assert result.returncode == 1
    assert "submodules are not initialized at recorded commits" in result.stderr
    assert any(line.startswith("-") for line in result.stderr.splitlines())
    assert " third-party/cutlass" in result.stderr


def test_rejects_submodule_at_wrong_commit(tmp_path: Path) -> None:
    checkout, first_submodule_commit = _make_checkout(tmp_path)
    _git(checkout / "third-party/cutlass", "checkout", first_submodule_commit)

    result = _run_checker(checkout)

    assert result.returncode == 1
    assert "submodules are not initialized at recorded commits" in result.stderr
    assert any(line.startswith("+") for line in result.stderr.splitlines())
    assert " third-party/cutlass" in result.stderr
