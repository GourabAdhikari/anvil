"""Text command router for Anvil's LLM brain.

The router deliberately keeps tool imports lazy.  This lets the text-only core be
used before optional integrations are installed, and makes the same function
safe to call from the future voice input path.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from typing import Any

MODEL = "llama-3.3-70b-versatile"

# Kept here as a fallback while the schemas module is being built.  If that
# module exports TOOL_SCHEMAS, its value is used instead.
_DEFAULT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_repo",
            "description": "Create a new GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "stack": {"type": "string"},
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
            "description": "Get the current git status of a local repository.",
            "parameters": {"type": "object", "properties": {"repo_path": {"type": "string"}}, "required": ["repo_path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and commit them.",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string"}, "message": {"type": "string"}},
                "required": ["repo_path", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Detect and run the test suite for a repository.",
            "parameters": {"type": "object", "properties": {"repo_path": {"type": "string"}}, "required": ["repo_path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_prs",
            "description": "List open pull requests for a GitHub repository.",
            "parameters": {"type": "object", "properties": {"repo_name": {"type": "string"}}, "required": ["repo_name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_error",
            "description": "Explain an error in plain English.",
            "parameters": {"type": "object", "properties": {"error_text": {"type": "string"}}, "required": ["error_text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicates",
            "description": "Find near-duplicate code across repositories.",
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

_TOOL_MODULES = {
    "create_repo": "anvil.tools.scaffold.create_repo",
    "git_status": "anvil.tools.devops.git_ops",
    "git_commit": "anvil.tools.devops.git_ops",
    "run_tests": "anvil.tools.devops.test_runner",
    "check_prs": "anvil.tools.devops.pr_checker",
    "explain_error": "anvil.tools.devops.error_explainer",
    "find_duplicates": "anvil.tools.devops.duplicate_detector",
}


def _schemas() -> list[dict[str, Any]]:
    try:
        module = importlib.import_module("anvil.brain.tools_schema")
        schemas = getattr(module, "TOOL_SCHEMAS", None)
        if schemas:
            return schemas
    except ImportError:
        pass
    return _DEFAULT_TOOL_SCHEMAS


def _client() -> Any:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The groq package is not installed") from exc
    return Groq(api_key=api_key)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tool_calls(message: Any) -> list[Any]:
    return list(_value(message, "tool_calls", None) or [])


def _tool_name(call: Any) -> str:
    function = _value(call, "function", {})
    return _value(function, "name", "")


def _call_payload(call: Any) -> dict[str, Any]:
    """Convert an SDK tool-call object to the dict accepted in a request."""
    function = _value(call, "function", {})
    return {
        "id": _value(call, "id", ""),
        "type": _value(call, "type", "function"),
        "function": {
            "name": _value(function, "name", ""),
            "arguments": _value(function, "arguments", "{}"),
        },
    }


def _tool_arguments(call: Any) -> dict[str, Any]:
    function = _value(call, "function", {})
    raw = _value(function, "arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid arguments for {_tool_name(call)}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"arguments for {_tool_name(call)} must be a JSON object")
    return raw


def _tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def _handler(name: str, handlers: Mapping[str, Callable[..., Any]] | None) -> Callable[..., Any] | None:
    if handlers and name in handlers:
        return handlers[name]
    module_name = _TOOL_MODULES.get(name)
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, name, None)


def run(command: str, *, client: Any = None, handlers: Mapping[str, Callable[..., Any]] | None = None) -> str:
    """Send ``command`` to Groq, execute requested tools, and return the reply.

    ``client`` and ``handlers`` are injectable for tests and local integrations.
    Tool failures are returned to the model so it can explain them naturally;
    malformed model responses and API failures are raised to the caller.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    client = client or _client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are Anvil, a concise personal developer agent."},
        {"role": "user", "content": command.strip()},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=_schemas(), tool_choice="auto")
    message = _value(_value(response, "choices", [])[0], "message", {})
    calls = _tool_calls(message)
    if not calls:
        return _value(message, "content", "") or ""

    messages.append({
        "role": "assistant",
        "content": _value(message, "content", None),
        "tool_calls": [_call_payload(call) for call in calls],
    })
    for call in calls:
        name = _tool_name(call)
        try:
            function = _handler(name, handlers)
            if function is None:
                raise RuntimeError(f"tool '{name}' is not available")
            result = function(**_tool_arguments(call))
        except Exception as exc:  # tool errors should be explained by the model
            result = {"error": str(exc)}
        messages.append({"role": "tool", "tool_call_id": _value(call, "id", name), "name": name, "content": _tool_result(result)})

    final = client.chat.completions.create(model=MODEL, messages=messages)
    final_message = _value(_value(final, "choices", [])[0], "message", {})
    return _value(final_message, "content", "") or ""


route = run
route_command = run
process_command = run
