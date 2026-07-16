"""Local Git operations used by Anvil.

These functions intentionally do not push, create pull requests, or run tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from git.exc import BadName


def _repo(repo_path: str) -> Repo:
    if not isinstance(repo_path, str) or not repo_path.strip():
        raise ValueError("repo_path must be a non-empty path")
    return Repo(Path(repo_path).expanduser().resolve(), search_parent_directories=False)


def _error(error: Exception, repo_path: str) -> dict[str, Any]:
    if isinstance(error, NoSuchPathError):
        message = f"Repository path does not exist: {repo_path}"
    elif isinstance(error, InvalidGitRepositoryError):
        message = f"Not a Git repository: {repo_path}"
    else:
        message = str(error)
    return {"success": False, "repo_path": repo_path, "error": message}


def _branch(repo: Repo) -> str | None:
    if repo.head.is_detached:
        return None
    return repo.active_branch.name


def _staged_paths(repo: Repo) -> list[str]:
    try:
        return [diff.a_path for diff in repo.index.diff("HEAD")]
    except BadName:
        # An unborn repository has no HEAD for GitPython to diff against.
        return [path for path in repo.git.diff("--cached", "--name-only").splitlines() if path]


def git_status(repo_path: str) -> dict[str, Any]:
    """Return the current branch and working-tree status for a local repo."""
    try:
        repo = _repo(repo_path)
        untracked = list(repo.untracked_files)
        staged = _staged_paths(repo)
        unstaged = [diff.a_path for diff in repo.index.diff(None)]
        return {
            "success": True,
            "repo_path": str(repo.working_tree_dir),
            "branch": _branch(repo),
            "is_dirty": repo.is_dirty(untracked_files=True),
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
        }
    except Exception as exc:
        return _error(exc, repo_path)


def git_commit(repo_path: str, message: str) -> dict[str, Any]:
    """Stage all local changes and create a commit with ``message``."""
    if not isinstance(message, str) or not message.strip():
        return {"success": False, "repo_path": repo_path, "error": "Commit message must not be empty."}

    try:
        repo = _repo(repo_path)
        if not repo.is_dirty(untracked_files=True):
            return {"success": False, "repo_path": str(repo.working_tree_dir), "error": "No changes to commit."}

        repo.git.add(A=True)
        commit = repo.index.commit(message.strip())
        return {
            "success": True,
            "repo_path": str(repo.working_tree_dir),
            "branch": _branch(repo),
            "commit": commit.hexsha,
            "message": commit.message.strip(),
        }
    except Exception as exc:
        return _error(exc, repo_path)


__all__ = ["git_status", "git_commit"]
