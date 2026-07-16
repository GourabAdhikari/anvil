"""Detect and run a repository's Python or npm test suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _has_pytest_config(repo: Path) -> bool:
    config_files = ("pytest.ini", "tox.ini", "setup.cfg")
    for filename in config_files:
        path = repo / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace").lower()
        if filename == "pytest.ini" or "pytest" in content:
            return True

    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        content = pyproject.read_text(encoding="utf-8", errors="replace").lower()
        if "[tool.pytest" in content or "pytest" in content:
            return True

    for filename in ("requirements.txt", "requirements-dev.txt"):
        requirements = repo / filename
        if requirements.is_file():
            for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
                package = line.split("#", 1)[0].strip().lower()
                if package.startswith("pytest") and (len(package) == 6 or package[6] in "<>=!~ "):
                    return True

    tests_dir = repo / "tests"
    if tests_dir.is_dir() and any(
        file.name.startswith("test_") or file.name.endswith("_test.py")
        for file in tests_dir.rglob("*.py")
    ):
        return True
    return False


def _has_npm_test(repo: Path) -> bool:
    package_json = repo / "package.json"
    if not package_json.is_file():
        return False
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(package.get("scripts"), dict) and bool(package["scripts"].get("test"))


def _runner(repo: Path) -> tuple[str, list[str]] | None:
    if _has_pytest_config(repo):
        return "pytest", [sys.executable, "-m", "pytest"]
    if _has_npm_test(repo):
        return "npm", ["npm", "test"]
    return None


def run_tests(repo_path: str) -> dict[str, Any]:
    """Detect and execute pytest or npm tests in ``repo_path``."""
    if not isinstance(repo_path, str) or not repo_path.strip():
        return {"success": False, "repo_path": repo_path, "error": "repo_path must be a non-empty path."}

    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        return {"success": False, "repo_path": repo_path, "error": f"Directory does not exist: {repo_path}"}

    detected = _runner(path)
    if detected is None:
        return {
            "success": False,
            "repo_path": str(path),
            "error": "No supported test project detected. Expected pytest configuration or a package.json test script.",
        }

    runner, command = detected
    try:
        completed = subprocess.run(
            command,
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "success": False,
            "repo_path": str(path),
            "runner": runner,
            "command": command,
            "error": str(exc),
            "stdout": "",
            "stderr": "",
        }

    passed = completed.returncode == 0
    return {
        "success": passed,
        "passed": passed,
        "repo_path": str(path),
        "runner": runner,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


__all__ = ["run_tests"]
