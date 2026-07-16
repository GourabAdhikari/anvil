"""Text command router for Anvil's LLM brain.

The router deliberately keeps tool imports lazy.  This lets the text-only core be
used before optional integrations are installed, and makes the same function
safe to call from the future voice input path.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex
from collections.abc import Callable, Mapping
from typing import Any

from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = (
    "You are Anvil, a concise personal developer agent. "
    "Only call functions that appear in the supplied tools list. "
    "If no supplied tool applies, answer the user directly. "
    "For git_status, omit repo_path to inspect the current working directory; never invent placeholder paths."
)

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
            "description": "Get the current git status of a local repository. Defaults to the current working directory when repo_path is omitted.",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string", "description": "Optional; defaults to the current working directory."}},
            },
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

_CONFIRMATION_REQUIRED = frozenset({"create_repo", "git_commit"})
_REQUIRED_TOOLS = frozenset({
    "create_repo",
    "git_status",
    "git_commit",
    "run_tests",
    "check_prs",
    "explain_error",
    "find_duplicates",
})

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


def _schema_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise RuntimeError("Invalid tool schema: every function tool must have a name")
        if name in names:
            raise RuntimeError(f"Duplicate tool schema: {name}")
        names.add(name)
    return names


def _validate_tool_registry(tools: list[dict[str, Any]]) -> set[str]:
    """Ensure advertised schemas and local dispatch registrations agree."""
    advertised = _schema_names(tools)
    registered = set(_TOOL_MODULES)
    missing_handlers = advertised - registered
    missing_schemas = registered - advertised
    missing_required = _REQUIRED_TOOLS - advertised
    unexpected_tools = advertised - _REQUIRED_TOOLS
    if missing_handlers or missing_schemas or missing_required or unexpected_tools:
        details = []
        if missing_handlers:
            details.append(f"advertised without handlers: {sorted(missing_handlers)}")
        if missing_schemas:
            details.append(f"registered without schemas: {sorted(missing_schemas)}")
        if missing_required:
            details.append(f"required tools missing from schemas: {sorted(missing_required)}")
        if unexpected_tools:
            details.append(f"unexpected tools in schemas: {sorted(unexpected_tools)}")
        raise RuntimeError("Tool registry mismatch: " + "; ".join(details))
    return advertised


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


def _raw_payload(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(value, default=str, sort_keys=True)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), default=str, sort_keys=True)
    return repr(value)


def _tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def _validate_executor_registry(handlers: Mapping[str, Callable[..., Any]] | None) -> None:
    """Validate that every registered tool resolves to an executable function."""
    missing: list[str] = []
    for name, module_name in _TOOL_MODULES.items():
        if handlers and name in handlers:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(f"Executor dependency for '{name}' is unavailable: {exc}") from exc
        if not callable(getattr(module, name, None)):
            missing.append(name)
    if missing:
        raise RuntimeError(f"Executor functions missing: {sorted(missing)}")


def _handler(name: str, handlers: Mapping[str, Callable[..., Any]] | None) -> Callable[..., Any] | None:
    print("Requested tool:", name)
    print("Execution registry keys:", sorted(_TOOL_MODULES))
    if handlers and name in handlers:
        return handlers[name]
    module_name = _TOOL_MODULES.get(name)
    if not module_name:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, name, None)


def _confirm_tool(name: str, arguments: dict[str, Any], confirm: Callable[[str], bool] | None) -> bool:
    if name not in _CONFIRMATION_REQUIRED:
        return True
    prompt = f"Confirm {name} with arguments {json.dumps(arguments, default=str)}? [y/N] "
    if confirm is not None:
        return bool(confirm(prompt))
    try:
        return input(prompt).strip().casefold() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _memory_command(command: str) -> str | None:
    """Handle memory commands locally without involving the LLM."""
    try:
        parts = shlex.split(command.strip())
    except ValueError as exc:
        return json.dumps({"success": False, "error": f"Invalid memory command: {exc}"})
    if len(parts) < 2 or parts[0].casefold() != "memory":
        return None

    from anvil.memory import store

    action = parts[1].casefold()
    if action == "remember" and len(parts) >= 3:
        return json.dumps(store.remember_statement(" ".join(parts[2:])), default=str)
    if action == "list" and len(parts) == 2:
        return json.dumps(store.list_memories(), default=str)
    if action == "search" and len(parts) >= 3:
        return json.dumps(store.search_memories(" ".join(parts[2:])), default=str)
    if action == "clear" and len(parts) == 2:
        return json.dumps(store.clear_memories(), default=str)
    return json.dumps({
        "success": False,
        "error": "Usage: memory remember <statement> | memory list | memory search <query> | memory clear",
    })


def _memory_context(command: str) -> str:
    try:
        from anvil.memory import store

        result = store.search_memories(command, limit=5)
        if not result.get("success") or not result.get("matches"):
            return ""
        memories = []
        for match in result["matches"]:
            # Conversation transcripts can contain stale model errors and must
            # never be treated as authoritative instructions or tool context.
            if match.get("id") == "conversation_history":
                continue
            value = match.get("value", "")
            if isinstance(value, (dict, list)):
                continue
            memory = str(value).strip()
            if memory:
                memories.append(memory)
        if not memories:
            return ""
        return "\nRelevant stored memories:\n" + "\n".join(f"- {memory}" for memory in memories)
    except Exception:
        return ""


def run(
    command: str,
    *,
    client: Any = None,
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    confirm: Callable[[str], bool] | None = None,
) -> str:
    """Send ``command`` to Groq, execute requested tools, and return the reply.

    ``client`` and ``handlers`` are injectable for tests and local integrations.
    Tool failures are returned to the model so it can explain them naturally;
    malformed model responses and API failures are raised to the caller.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    local_result = _memory_command(command)
    if local_result is not None:
        return local_result

    client = client or _client()
    tools = _schemas()
    advertised_tools = _validate_tool_registry(tools)
    _validate_executor_registry(handlers)
    system_prompt = SYSTEM_PROMPT + _memory_context(command)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": command.strip()},
    ]
    request_payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    print(json.dumps({"event": "available_tools", "tools": sorted(advertised_tools)}))
    print("Groq raw request #1:", _raw_payload(request_payload))
    if os.getenv("ANVIL_DEBUG") == "1":
        print("Groq system prompt:", repr(system_prompt))
    response = client.chat.completions.create(**request_payload)
    print("Groq raw response #1:", _raw_payload(response))
    message = _value(_value(response, "choices", [])[0], "message", {})
    calls = _tool_calls(message)
    if not calls:
        return _value(message, "content", "") or ""

    unknown_tools = sorted({_tool_name(call) for call in calls} - advertised_tools)
    if unknown_tools:
        raise RuntimeError(
            "Model requested unavailable tool(s): "
            + ", ".join(unknown_tools)
            + ". Available tools: "
            + ", ".join(sorted(advertised_tools))
        )

    messages.append({
        "role": "assistant",
        "content": _value(message, "content", None),
        "tool_calls": [_call_payload(call) for call in calls],
    })
    for call in calls:
        name = _tool_name(call)
        try:
            arguments = _tool_arguments(call)
            if name == "git_status":
                path = arguments.get("repo_path")
                if not path or path in {"current_repo", "./current_repo"}:
                    arguments["repo_path"] = os.getcwd()
            if not _confirm_tool(name, arguments, confirm):
                result = {"success": False, "cancelled": True, "tool": name, "message": "Action cancelled by the user."}
            else:
                function = _handler(name, handlers)
                if function is None:
                    raise RuntimeError(f"tool '{name}' is not available")
                result = function(**arguments)
        except Exception as exc:  # tool errors should be explained by the model
            result = {"error": str(exc)}
        messages.append({"role": "tool", "tool_call_id": _value(call, "id", name), "name": name, "content": _tool_result(result)})

    final_payload = {"model": MODEL, "messages": messages}
    print("Groq raw request #2:", _raw_payload(final_payload))
    final = client.chat.completions.create(**final_payload)
    print("Groq raw response #2:", _raw_payload(final))
    final_message = _value(_value(final, "choices", [])[0], "message", {})
    return _value(final_message, "content", "") or ""


route = run
route_command = run
process_command = run
