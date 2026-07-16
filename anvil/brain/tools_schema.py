"""Groq/OpenAI-compatible function-calling schemas for Anvil tools."""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_repo",
            "description": "Create a new GitHub repo, scaffold it from a template, and push the initial commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "stack": {"type": "string", "enum": ["nextjs-drizzle", "fastapi-ml"]},
                    "private": {"type": "boolean", "default": True},
                },
                "required": ["name", "stack"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Get the current git status of a local repo. Defaults to the current working directory when repo_path is omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Optional local repository path; defaults to the current working directory.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and commit with a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["repo_path", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Detect and run the test suite for a repo, defaulting to the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string", "description": "Optional; defaults to the current working directory."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_prs",
            "description": "List open pull requests for a repo with a short summary of each.",
            "parameters": {
                "type": "object",
                "properties": {"repo_name": {"type": "string"}},
                "required": ["repo_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_error",
            "description": "Explain a stack trace or error message in plain English with a likely fix.",
            "parameters": {
                "type": "object",
                "properties": {"error_text": {"type": "string"}},
                "required": ["error_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicates",
            "description": "Scan given repos for near-duplicate functions using code embeddings and cosine similarity, return matches above a similarity threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_paths": {"type": "array", "items": {"type": "string"}},
                    "similarity_threshold": {"type": "number", "default": 0.85},
                },
                "required": ["repo_paths"],
            },
        },
    },
]

__all__ = ["TOOL_SCHEMAS"]
