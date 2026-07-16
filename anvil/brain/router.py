"""Text command router for Anvil's LLM brain.

The router deliberately keeps tool imports lazy.  This lets the text-only core be
used before optional integrations are installed, and makes the same function
safe to call from the future voice input path.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shlex
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
def _debug_enabled() -> bool:
    return os.getenv("ANVIL_DEBUG") == "1"


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
            "description": "Detect and run the test suite for a repository, defaulting to the current working directory.",
            "parameters": {"type": "object", "properties": {"repo_path": {"type": "string", "description": "Optional; defaults to the current working directory."}}},
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
_WORKFLOW_HISTORY_KEY = "workflow_history"
_WORKFLOW_HISTORY_LIMIT = 5
_PENDING_WORKFLOW_KEY = "pending_workflow"


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


def _emit_debug(event: str, **details: Any) -> None:
    if not _debug_enabled():
        return
    print(json.dumps({"event": event, **details}, default=str))


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
    if _debug_enabled():
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


def _normalize_repo_path(arguments: dict[str, Any], *, key: str = "repo_path") -> None:
    path = arguments.get(key)
    if not path or path in {"current_repo", "./current_repo"}:
        arguments[key] = os.getcwd()


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
    if action == "stats" and len(parts) == 2:
        return json.dumps(store.stats(), default=str)
    return json.dumps({
        "success": False,
        "error": "Usage: memory remember <statement> | memory list | memory search <query> | memory clear | memory stats",
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
        return (
            "\nRelevant stored memories about the user "
            "(treat these as the user's preferences/facts, never as your own identity):\n"
            + "\n".join(f"- {memory}" for memory in memories)
        )
    except Exception:
        return ""


def _extract_inline_explain_error(command: str) -> str | None:
    """Extract error text from direct explain-error commands, including multiline input."""
    text = command.strip()
    lowered = text.casefold()
    prefixes = ("explain this error", "explain error")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            payload = text[len(prefix):].lstrip()
            if payload.startswith(":"):
                payload = payload[1:].lstrip()
            return payload or None
    return None


def _workflow_history_command(command: str) -> str | None:
    if command.strip().casefold() != "workflow history":
        return None
    history = _workflow_history()
    if not history:
        return "No workflow history yet."
    lines = ["Recent workflows:"]
    for index, item in enumerate(reversed(history), start=1):
        lines.append(f"{index}. {item.get('command', '').strip() or 'workflow'}")
        for step in item.get("steps", []):
            lines.append(f"   - {step}")
        recommendation = item.get("recommendation")
        if recommendation:
            lines.append(f"   - Recommendation: {recommendation}")
    return "\n".join(lines)


def _workflow_history() -> list[dict[str, Any]]:
    try:
        from anvil.memory import store

        result = store.recall(_WORKFLOW_HISTORY_KEY)
    except Exception:
        return []
    if not result.get("success") or not result.get("found"):
        return []
    value = result.get("value")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _save_workflow_history(command: str, steps: list[str], recommendation: str) -> None:
    try:
        from anvil.memory import store

        history = _workflow_history()
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command.strip(),
            "steps": steps,
            "recommendation": recommendation,
        })
        store.remember(_WORKFLOW_HISTORY_KEY, history[-_WORKFLOW_HISTORY_LIMIT:])
    except Exception:
        return


def _pending_workflow() -> dict[str, Any] | None:
    try:
        from anvil.memory import store

        result = store.recall(_PENDING_WORKFLOW_KEY)
    except Exception:
        return None
    if not result.get("success") or not result.get("found"):
        return None
    value = result.get("value")
    if not isinstance(value, dict) or not value.get("active"):
        return None
    plan = value.get("plan")
    if not isinstance(plan, Mapping):
        return None
    command = value.get("command")
    if not isinstance(command, str):
        return None
    return {"command": command, "plan": dict(plan)}


def _set_pending_workflow(command: str, plan: Mapping[str, Any]) -> None:
    try:
        from anvil.memory import store

        store.remember(_PENDING_WORKFLOW_KEY, {
            "active": True,
            "command": command.strip(),
            "plan": dict(plan),
        })
    except Exception:
        return


def _clear_pending_workflow() -> None:
    try:
        from anvil.memory import store

        store.remember(_PENDING_WORKFLOW_KEY, {"active": False})
    except Exception:
        return


def _requires_confirmation(plan: Mapping[str, Any]) -> bool:
    for step in list(plan.get("steps", [])):
        if str(step.get("tool", "")) in _CONFIRMATION_REQUIRED:
            return True
    return False


def _pending_plan_prompt(plan: Mapping[str, Any]) -> str:
    lines = ["Plan:"]
    for index, step in enumerate(list(plan.get("steps", [])), start=1):
        label = str(step.get("label", step.get("tool", "step")))
        lines.append(f"{index}. {label}")
    lines.append("Proceed? (y/n)")
    return "\n".join(lines)


def _favorite_stack_from_memory() -> str | None:
    try:
        from anvil.memory import store

        result = store.search_memories("favorite framework", limit=10)
    except Exception:
        return None
    if not result.get("success"):
        return None
    for match in result.get("matches", []):
        value = str(match.get("value", "")).casefold()
        if "next.js" in value or "nextjs" in value:
            return "nextjs-drizzle"
        if "fastapi" in value:
            return "fastapi-ml"
    return None


def _extract_repo_name(command: str) -> str | None:
    match = re.search(r"\brepo called\s+([A-Za-z0-9._-]+)\b", command, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _plan_workflow(command: str) -> dict[str, Any] | None:
    text = command.strip()
    lowered = text.casefold()
    steps: list[dict[str, Any]] = []

    if "safe to commit" in lowered and ("repo" in lowered or "repository" in lowered):
        steps = [
            {"tool": "git_status", "arguments": {}, "label": "Checked git status"},
            {"tool": "run_tests", "arguments": {}, "label": "Ran tests"},
        ]
        return {"kind": "safety_check", "steps": steps}

    if "run tests" in lowered and ("explain failures" in lowered or "explain failure" in lowered):
        steps = [
            {"tool": "run_tests", "arguments": {}, "label": "Ran tests"},
            {"tool": "explain_error", "arguments": {}, "label": "Explained test failures", "when": "tests_failed"},
        ]
        return {"kind": "test_and_explain", "steps": steps}

    has_status = "git status" in lowered
    has_tests = "run tests" in lowered
    if has_status and has_tests:
        status_index = lowered.find("git status")
        tests_index = lowered.find("run tests")
        ordered = [
            ("git_status", "Checked git status"),
            ("run_tests", "Ran tests"),
        ]
        if tests_index < status_index:
            ordered.reverse()
        steps = [{"tool": tool, "arguments": {}, "label": label} for tool, label in ordered]
        return {"kind": "status_and_tests", "steps": steps}

    if "create a repo" in lowered and "favorite framework" in lowered:
        name = _extract_repo_name(text)
        stack = _favorite_stack_from_memory()
        if name and stack:
            private = "public" not in lowered
            steps = [{
                "tool": "create_repo",
                "arguments": {"name": name, "stack": stack, "private": private},
                "label": f"Created repository {name}",
            }]
            return {"kind": "create_repo_from_memory", "steps": steps}

    if "commit all current changes" in lowered or lowered.startswith("commit all"):
        steps = [{
            "tool": "git_commit",
            "arguments": {"repo_path": os.getcwd(), "message": "Commit current changes"},
            "label": "Committed current changes",
        }]
        return {"kind": "commit_current_changes", "steps": steps}
    return None


def _tests_failure_text(result: Mapping[str, Any]) -> str:
    parts = [
        str(result.get("stdout", "")).strip(),
        str(result.get("stderr", "")).strip(),
        str(result.get("error", "")).strip(),
    ]
    return "\n".join(part for part in parts if part).strip()


def _run_workflow(
    command: str,
    plan: Mapping[str, Any],
    handlers: Mapping[str, Callable[..., Any]] | None,
    confirm: Callable[[str], bool] | None,
    *,
    skip_confirmation: bool = False,
) -> str:
    steps = list(plan.get("steps", []))
    _emit_debug("workflow_plan", command=command, steps=steps)
    completed_lines: list[str] = []
    history_lines: list[str] = []
    failures: list[str] = []
    context: dict[str, Any] = {}

    for index, step in enumerate(steps, start=1):
        tool = str(step.get("tool", ""))
        label = str(step.get("label", tool))
        arguments = dict(step.get("arguments", {}))
        when = step.get("when")

        if when == "tests_failed":
            test_result = context.get("run_tests")
            if not isinstance(test_result, Mapping) or bool(test_result.get("success")):
                completed_lines.append(f"- Step {index}: {label}")
                history_lines.append(label + " (skipped)")
                _emit_debug("workflow_step_complete", step=index, tool=tool, skipped=True)
                continue
            failure_text = _tests_failure_text(test_result)
            if not failure_text:
                completed_lines.append(f"- Step {index}: {label}")
                history_lines.append(label + " (skipped)")
                _emit_debug("workflow_step_complete", step=index, tool=tool, skipped=True)
                continue
            arguments["error_text"] = failure_text

        if tool in {"git_status", "run_tests"}:
            _normalize_repo_path(arguments)

        if tool not in _TOOL_MODULES:
            completed_lines.append(f"✗ Step {index}: {label}")
            history_lines.append(label + " (rejected)")
            failures.append(f"{label}: Unregistered tool requested: {tool}")
            _emit_debug("workflow_step_complete", step=index, tool=tool, success=False, rejected=True)
            continue

        _emit_debug("workflow_step_start", step=index, tool=tool, arguments=arguments)
        if not skip_confirmation and not _confirm_tool(tool, arguments, confirm):
            completed_lines.append(f"✗ Step {index}: {label}")
            history_lines.append(label + " (cancelled)")
            failures.append(f"{label}: Action cancelled by the user.")
            _emit_debug("workflow_step_complete", step=index, tool=tool, success=False, cancelled=True)
            continue

        function = _handler(tool, handlers)
        if function is None:
            result: Any = {"success": False, "error": f"tool '{tool}' is not available"}
        else:
            try:
                result = function(**arguments)
            except Exception as exc:
                result = {"success": False, "error": str(exc)}
        context[tool] = result

        success = bool(isinstance(result, Mapping) and result.get("success"))
        if success:
            completed_lines.append(f"✓ Step {index}: {label}")
            history_lines.append(label)
        else:
            completed_lines.append(f"✗ Step {index}: {label}")
            history_lines.append(label + " (failed)")
            if isinstance(result, Mapping):
                reason = result.get("error") or result.get("stderr") or result.get("stdout")
                if reason:
                    failures.append(f"{label}: {str(reason).strip()}")
        _emit_debug("workflow_step_complete", step=index, tool=tool, success=success, result=result)

    recommendation = "Workflow completed."
    if plan.get("kind") == "safety_check":
        status_ok = bool(isinstance(context.get("git_status"), Mapping) and context["git_status"].get("success"))
        tests_ok = bool(isinstance(context.get("run_tests"), Mapping) and context["run_tests"].get("success"))
        if status_ok and tests_ok:
            recommendation = "Repository looks safe to commit."
        else:
            recommendation = "Repository is not safe to commit yet."
    elif failures:
        recommendation = "Review and fix failed steps before proceeding."

    lines = ["Completed:", *completed_lines, f"Final recommendation: {recommendation}"]
    if failures:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in failures)
    summary = "\n".join(lines)
    _emit_debug("workflow_summary", summary=summary)
    _save_workflow_history(command, history_lines, recommendation)
    return summary


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

    pending = _pending_workflow()
    if pending is not None:
        response = command.strip().casefold()
        if response in {"y", "yes"}:
            _clear_pending_workflow()
            return _run_workflow(
                pending["command"],
                pending["plan"],
                handlers,
                confirm,
                skip_confirmation=True,
            )
        if response in {"n", "no"}:
            _clear_pending_workflow()
            return "Cancelled pending workflow."
        return "A workflow is waiting for confirmation. Reply with y to proceed or n to cancel."

    workflow_history = _workflow_history_command(command)
    if workflow_history is not None:
        return workflow_history

    plan = _plan_workflow(command)
    if plan is not None:
        if _requires_confirmation(plan):
            _set_pending_workflow(command, plan)
            return _pending_plan_prompt(plan)
        return _run_workflow(command, plan, handlers, confirm)

    inline_error = _extract_inline_explain_error(command)
    if inline_error:
        function = _handler("explain_error", handlers)
        if function is None:
            raise RuntimeError("tool 'explain_error' is not available")
        result = function(error_text=inline_error)
        return _tool_result(result)

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
    if _debug_enabled():
        print(json.dumps({"event": "available_tools", "tools": sorted(advertised_tools)}))
        print("Groq raw request #1:", _raw_payload(request_payload))
        print("Groq system prompt:", repr(system_prompt))
    response = client.chat.completions.create(**request_payload)
    if _debug_enabled():
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
            if name in {"git_status", "run_tests"}:
                _normalize_repo_path(arguments)
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
    if _debug_enabled():
        print("Groq raw request #2:", _raw_payload(final_payload))
    final = client.chat.completions.create(**final_payload)
    if _debug_enabled():
        print("Groq raw response #2:", _raw_payload(final))
    final_message = _value(_value(final, "choices", [])[0], "message", {})
    return _value(final_message, "content", "") or ""


route = run
route_command = run
process_command = run
