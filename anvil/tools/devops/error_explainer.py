"""Explain errors with Anvil's shared Groq client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from anvil.brain import router


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _content(response: Any) -> str:
    choices = _value(response, "choices", [])
    if not choices:
        return ""
    message = _value(choices[0], "message", {})
    return _value(message, "content", "") or ""


def _parse_response(content: str) -> dict[str, str] | None:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None

    fields = ("error_summary", "likely_cause", "suggested_fix")
    if not all(isinstance(value.get(field), str) and value[field].strip() for field in fields):
        return None
    return {field: value[field].strip() for field in fields}


def explain_error(error_text: str, *, client: Any = None) -> dict[str, Any]:
    """Return a concise summary, likely cause, and suggested fix for an error."""
    if not isinstance(error_text, str) or not error_text.strip():
        return {"success": False, "error": "error_text must be a non-empty string."}

    prompt = (
        "Analyze the following error for a developer. Return only valid JSON with "
        'the string keys "error_summary", "likely_cause", and "suggested_fix". '
        "Keep each value concise and do not include markdown.\n\n"
        f"Error:\n{error_text.strip()}"
    )
    try:
        groq = client or router._client()
        response = groq.chat.completions.create(
            model=router.MODEL,
            messages=[
                {"role": "system", "content": "You explain software errors clearly and accurately."},
                {"role": "user", "content": prompt},
            ],
        )
        explanation = _parse_response(_content(response))
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    if explanation is None:
        return {"success": False, "error": "Groq returned an invalid explanation format."}
    return {"success": True, "error_text": error_text.strip(), **explanation}


__all__ = ["explain_error"]
