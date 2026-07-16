from __future__ import annotations

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
