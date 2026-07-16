"""Create an empty GitHub repository for an Anvil project.

Template scaffolding is intentionally out of scope for this phase.  The
repository is initialized with GitHub's default README only.
"""

from __future__ import annotations

import os
from typing import Any

Github: Any = None
SUPPORTED_STACKS = frozenset({"nextjs-drizzle", "fastapi-ml"})


def _github_client(token: str) -> Any:
    global Github
    if Github is None:
        try:
            from github import Github as GithubClient
        except ImportError as exc:
            raise RuntimeError("PyGithub is not installed") from exc
        Github = GithubClient
    return Github(token)


def create_repo(name: str, stack: str, private: bool = True) -> dict[str, Any]:
    """Create and initialize a GitHub repository.

    Args:
        name: GitHub repository name.
        stack: Anvil stack identifier (currently ``nextjs-drizzle`` or
            ``fastapi-ml``). It is recorded in the result; no template is
            applied yet.
        private: Whether the repository should be private.

    Returns:
        A JSON-serializable result containing repository metadata, or a
        structured error result when creation cannot be completed.
    """
    if not isinstance(name, str) or not name.strip():
        return {"success": False, "error": "Repository name must not be empty."}
    if not isinstance(stack, str) or stack not in SUPPORTED_STACKS:
        supported = ", ".join(sorted(SUPPORTED_STACKS))
        return {"success": False, "error": f"Unsupported stack. Choose one of: {supported}."}
    if not isinstance(private, bool):
        return {"success": False, "error": "private must be a boolean."}

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        return {"success": False, "error": "GITHUB_TOKEN is not configured."}

    repository_name = name.strip()
    try:
        github = _github_client(token)
        user = github.get_user()
        repository = user.create_repo(
            name=repository_name,
            description=f"Anvil project ({stack})",
            private=private,
            auto_init=True,
        )
    except Exception as exc:
        return {"success": False, "name": repository_name, "stack": stack, "error": str(exc)}

    return {
        "success": True,
        "name": repository.name,
        "stack": stack,
        "private": repository.private,
        "url": repository.html_url,
        "clone_url": repository.clone_url,
        "id": repository.id,
    }


__all__ = ["create_repo"]
