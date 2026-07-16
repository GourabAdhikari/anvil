from __future__ import annotations

import json
import os

from anvil.brain import router


class FakeCompletions:
    def __init__(self, tool_name, arguments):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": self.tool_name, "arguments": self.arguments},
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Tool completed successfully."}}]}


class FakeClient:
    def __init__(self, tool_name, arguments):
        self.completions = FakeCompletions(tool_name, arguments)
        self.chat = type("Chat", (), {"completions": self.completions})()


def run_tool(name, arguments, handler, confirm=None):
    client = FakeClient(name, arguments)
    result = router.run(
        f"run {name}",
        client=client,
        handlers={name: handler},
        confirm=confirm,
    )
    assert result == "Tool completed successfully."
    assert len(client.completions.calls) == 2


def test_create_repo_requires_confirmation():
    calls = []
    run_tool("create_repo", '{"name":"test-agent","stack":"nextjs-drizzle","private":true}', lambda **args: calls.append(args) or {"success": True}, confirm=lambda prompt: True)
    assert calls


def test_git_status_defaults_to_current_directory():
    calls = []
    run_tool("git_status", "{}", lambda **args: calls.append(args) or {"success": True})
    assert calls == [{"repo_path": os.getcwd()}]


def test_git_status_replaces_model_placeholder_path():
    calls = []
    run_tool("git_status", '{"repo_path":"./current_repo"}', lambda **args: calls.append(args) or {"success": True})
    assert calls == [{"repo_path": os.getcwd()}]


def test_git_commit_requires_confirmation():
    calls = []
    run_tool("git_commit", '{"repo_path":".","message":"Add memory system"}', lambda **args: calls.append(args) or {"success": True}, confirm=lambda prompt: True)
    assert calls


def test_run_tests():
    calls = []
    run_tool("run_tests", '{"repo_path":"."}', lambda **args: calls.append(args) or {"success": True})
    assert calls


def test_check_prs():
    calls = []
    run_tool("check_prs", '{"repo_name":"owner/repo"}', lambda **args: calls.append(args) or {"success": True})
    assert calls


def test_explain_error():
    calls = []
    run_tool("explain_error", '{"error_text":"ModuleNotFoundError"}', lambda **args: calls.append(args) or {"success": True})
    assert calls


def test_commit_is_cancelled_without_confirmation():
    calls = []
    run_tool("git_commit", '{"repo_path":".","message":"unsafe"}', lambda **args: calls.append(args) or {"success": True}, confirm=lambda prompt: False)
    assert calls == []


def test_multiline_explain_error_routes_to_tool():
    calls = []
    result = router.run(
        "explain this error:\n\nModuleNotFoundError: No module named 'pip'",
        handlers={"explain_error": lambda **args: calls.append(args) or {"success": True, "error_summary": "missing pip"}},
    )
    payload = json.loads(result)
    assert calls == [{"error_text": "ModuleNotFoundError: No module named 'pip'"}]
    assert payload["success"] is True


def test_workflow_runs_multiple_tools_in_sequence():
    calls = []

    def run_tests(**args):
        calls.append(("run_tests", args))
        return {"success": True, "passed": True}

    def git_status(**args):
        calls.append(("git_status", args))
        return {"success": True, "is_dirty": False}

    result = router.run(
        "show git status and run tests",
        handlers={"run_tests": run_tests, "git_status": git_status},
    )

    assert [name for name, _ in calls] == ["git_status", "run_tests"]
    assert "✓ Step 1" in result
    assert "✓ Step 2" in result


def test_workflow_failure_recovery_for_safety_check():
    calls = []

    def git_status(**args):
        calls.append(("git_status", args))
        return {"success": True, "is_dirty": True}

    def run_tests(**args):
        calls.append(("run_tests", args))
        return {"success": False, "error": "pytest not installed"}

    result = router.run(
        "check my repository and tell me if it is safe to commit",
        handlers={"git_status": git_status, "run_tests": run_tests},
    )

    assert [name for name, _ in calls] == ["git_status", "run_tests"]
    assert "Repository is not safe to commit yet." in result
    assert "pytest not installed" in result


def test_workflow_runs_explain_error_only_when_tests_fail():
    calls = []

    def run_tests(**args):
        calls.append(("run_tests", args))
        return {"success": False, "stdout": "", "stderr": "AssertionError: bad output"}

    def explain_error(**args):
        calls.append(("explain_error", args))
        return {"success": True, "error_summary": "A test assertion failed."}

    result = router.run(
        "run tests then explain failures",
        handlers={"run_tests": run_tests, "explain_error": explain_error},
    )

    assert [name for name, _ in calls] == ["run_tests", "explain_error"]
    assert "AssertionError: bad output" in calls[1][1]["error_text"]
    assert "✓ Step 2: Explained test failures" in result


def test_workflow_skips_explain_error_when_tests_pass():
    calls = []

    def run_tests(**args):
        calls.append(("run_tests", args))
        return {"success": True, "passed": True}

    result = router.run(
        "run tests then explain failures",
        handlers={"run_tests": run_tests},
    )

    assert [name for name, _ in calls] == ["run_tests"]
    assert "Step 2: Explained test failures" in result
