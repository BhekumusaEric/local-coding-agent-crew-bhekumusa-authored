import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent import (
    ToolBinding,
    acceptance_checks_from_plan,
    apply_acceptance_check_results,
    load_config,
    looks_like_malformed_tool_call,
    normalize_tool_calls,
    ollama_tool_name,
    public_web_url_error,
    review_verdict,
    tool_is_read_only,
    tools_for_role,
)


class AgentHelperTests(unittest.TestCase):
    def test_tool_name_is_normalized_and_bounded(self):
        self.assertEqual(ollama_tool_name("my server", "read/file"), "my_server_read_file")
        self.assertLessEqual(len(ollama_tool_name("s" * 80, "tool")), 64)

    def test_only_explicit_read_only_hint_skips_approval(self):
        read_only = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=True))
        write_tool = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=False))
        unannotated = SimpleNamespace(annotations=None)
        self.assertTrue(tool_is_read_only(read_only))
        self.assertFalse(tool_is_read_only(write_tool))
        self.assertFalse(tool_is_read_only(unannotated))

    def test_normalizes_json_shaped_unknown_tool_for_safe_rejection(self):
        response = {
            "role": "assistant",
            "content": '{"name":"git_status","arguments":{"path":"."}}',
        }
        calls = normalize_tool_calls(response)
        self.assertEqual(calls[0]["function"]["name"], "git_status")
        self.assertEqual(calls[0]["function"]["arguments"], {"path": "."})
        unknown = normalize_tool_calls(response)
        self.assertEqual(unknown[0]["function"]["name"], "git_status")

    def test_normalizes_fenced_qwen_json_tool_call(self):
        response = {
            "role": "assistant",
            "content": '```json\n{"name":"fs_list","arguments":{"path":"."}}\n```',
        }
        calls = normalize_tool_calls(response)
        self.assertEqual(calls[0]["function"]["name"], "fs_list")

    def test_multiple_tool_json_blocks_are_flagged_not_executable(self):
        response = {
            "role": "assistant",
            "content": '{"name":"git_reset","arguments":{}}\n{"name":"git_commit","arguments":{}}',
        }
        self.assertEqual(normalize_tool_calls(response), [])
        self.assertTrue(looks_like_malformed_tool_call(response))

    def test_config_expands_workspace_and_validates_fixed_test_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "git": {
                                "command": "uvx",
                                "args": ["--repository", "${workspaceFolder}"],
                            }
                        },
                        "test_command": ["python", "-m", "pytest"],
                    }
                ),
                encoding="utf-8",
            )
            servers, test_command = load_config(config_path, Path(temp_dir))
        self.assertEqual(servers[0]["args"], ["--repository", temp_dir])
        self.assertEqual(test_command, ["python", "-m", "pytest"])

    def test_role_tool_access_keeps_research_and_commands_separate(self):
        read_tool = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=True))
        write_tool = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=False))
        navigate_tool = SimpleNamespace(name="browser_navigate", annotations=None)
        snapshot_tool = SimpleNamespace(name="browser_snapshot", annotations=None)
        unsafe_tool = SimpleNamespace(name="browser_evaluate", annotations=None)
        bindings = {
            "filesystem_read": ToolBinding({}, "filesystem", mcp_tool=read_tool),
            "filesystem_write": ToolBinding({}, "filesystem", mcp_tool=write_tool),
            "git_read": ToolBinding({}, "git", mcp_tool=read_tool),
            "git_write": ToolBinding({}, "git", mcp_tool=write_tool),
            "research_fetch": ToolBinding({}, "research"),
            "playwright_navigate": ToolBinding(
                {}, "playwright", mcp_tool=navigate_tool
            ),
            "playwright_snapshot": ToolBinding(
                {}, "playwright", mcp_tool=snapshot_tool
            ),
            "playwright_evaluate": ToolBinding({}, "playwright", mcp_tool=unsafe_tool),
            "run_project_command": ToolBinding({}, "commands", kind="command"),
            "run_project_tests": ToolBinding({}, "tests", kind="tests"),
        }
        self.assertEqual(
            set(tools_for_role("Researcher", bindings)),
            {"research_fetch", "playwright_navigate", "playwright_snapshot"},
        )
        self.assertIn("run_project_command", tools_for_role("Builder", bindings))
        self.assertIn("git_read", tools_for_role("Builder", bindings))
        self.assertNotIn("git_write", tools_for_role("Builder", bindings))
        self.assertNotIn("research_fetch", tools_for_role("Builder", bindings))
        self.assertEqual(
            set(tools_for_role("Reviewer", bindings)),
            {"filesystem_read", "git_read"},
        )
        self.assertIn("run_project_tests", tools_for_role("Tester", bindings))

    def test_review_verdict_is_detected_from_review_text(self):
        self.assertEqual(review_verdict("VERDICT: PASS\nNo findings."), "PASS")
        self.assertEqual(review_verdict("Notes\nVERDICT: FIXES_REQUIRED"), "FIXES_REQUIRED")
        self.assertIsNone(review_verdict("Could not determine status."))

    def test_acceptance_checks_require_matching_report_and_evidence(self):
        checks = acceptance_checks_from_plan(
            "Plan\nAcceptance checks:\n- [ ] Unit tests pass\n* [ ] CLI returns zero"
        )
        self.assertEqual([check["text"] for check in checks], ["Unit tests pass", "CLI returns zero"])

        updated = apply_acceptance_check_results(
            checks,
            "Acceptance results:\n"
            "- [x] Unit tests pass — pytest exited with code 0\n"
            "- [ ] CLI returns zero — command was not run\n",
        )
        self.assertEqual(
            [(check["status"], check["evidence"]) for check in updated],
            [("passed", "pytest exited with code 0"), ("not_met", "command was not run")],
        )

        unchanged = apply_acceptance_check_results(checks, "- [x] Unit tests pass")
        self.assertEqual(unchanged[0]["status"], "pending")

    def test_public_web_url_guard_allows_public_and_rejects_local_targets(self):
        self.assertIsNone(public_web_url_error("https://docs.ollama.com/capabilities/tool-calling"))
        self.assertIsNotNone(public_web_url_error("http://localhost:11434/api/tags"))
        self.assertIsNotNone(public_web_url_error("http://192.168.1.1/"))
        self.assertIsNotNone(public_web_url_error("http://[::1]/"))
        self.assertIsNotNone(public_web_url_error("http://intranet/"))
        self.assertIsNotNone(public_web_url_error("https://user:pass@example.com/"))
        self.assertIsNotNone(public_web_url_error("file:///C:/Windows/win.ini"))


if __name__ == "__main__":
    unittest.main()
