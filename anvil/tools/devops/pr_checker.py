"""List open pull requests from GitHub."""

from __future__ import annotations

import os
from typing import Any

Github: Any = None


def _github_client(token: str) -> Any:
    global Github
    if Github is None:
        try:
            from github import Github as GithubClient
        except ImportError as exc:
            raise RuntimeError("PyGithub is not installed") from exc
        Github = GithubClient
    return Github(token)


def check_prs(repo_name: str) -> dict[str, Any]:
    """Return metadata for all open pull requests in ``repo_name``.

    ``repo_name`` must use GitHub's ``owner/repository`` format. Pull request
    descriptions are intentionally not included; summarization is a later task.
    """
    if not isinstance(repo_name, str) or not repo_name.strip():
        return {"success": False, "repo_name": repo_name, "error": "repo_name must not be empty."}

    repository_name = repo_name.strip()
    if repository_name.count("/") != 1 or any(not part for part in repository_name.split("/")):
        return {
            "success": False,
            "repo_name": repository_name,
            "error": "repo_name must use the owner/repository format.",
        }

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        return {"success": False, "repo_name": repository_name, "error": "GITHUB_TOKEN is not configured."}

    try:
        github = _github_client(token)
        repository = github.get_repo(repository_name)
        pull_requests = repository.get_pulls(state="open")
        results = [
            {
                "title": pull_request.title,
                "author": pull_request.user.login,
                "url": pull_request.html_url,
                "state": pull_request.state,
            }
            for pull_request in pull_requests
        ]
    except Exception as exc:
        return {"success": False, "repo_name": repository_name, "error": str(exc)}

    return {
        "success": True,
        "repo_name": repository_name,
        "count": len(results),
        "pull_requests": results,
    }


__all__ = ["check_prs"]
