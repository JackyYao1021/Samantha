"""Review structure, provider isolation, and user-facing explanations."""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from command_review import (
    REVIEW_SCHEMA, command_action_id, format_review, review_commands, validate_review,
)
from test_workflow import review_fixture


class CommandReviewTests(unittest.TestCase):
    def test_agent_uses_qwen_schema_and_passes_exact_context_as_data(self):
        request = {
            "user_request": "创建文件", "clarified_request": "创建 test.txt",
            "plan": "Create the file", "commands": ['touch "test.txt"'],
            "cwd": "/work", "shell": "/bin/bash", "previous_error": "",
        }
        with patch("command_review.Chat") as chat:
            chat.return_value.chat_context.return_value = json.dumps(review_fixture())
            result = review_commands(request)
        kwargs = chat.call_args.kwargs
        self.assertFalse(kwargs["allow_azure_fallback"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], REVIEW_SCHEMA)
        self.assertEqual(json.loads(chat.return_value.chat_context.call_args.args[0]), request)
        self.assertEqual(result, review_fixture())
        self.assertIn("Do not execute, rewrite, or approve commands", kwargs["begin_messages"])
        self.assertIn("所有自然语言字段必须使用简体中文", kwargs["begin_messages"])

    def test_reviewer_disables_cloud_fallback_even_when_other_agents_enable_it(self):
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture-key"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("offline")
            with self.assertRaisesRegex(RuntimeError, "Qwen request failed"):
                review_commands({"commands": ["pwd"]})
        azure.assert_not_called()

    def test_unknowns_and_errors_require_evidence_and_malformed_fields_are_rejected(self):
        invalid = [
            {**review_fixture(), "risk": "safe"},
            {**review_fixture(), "effects": []},
            {**review_fixture(), "effects": "creates file"},
            {**review_fixture(), "uncertainties": [1]},
            {**review_fixture(), "recommendation": " "},
            {**review_fixture(), "verdict": "insufficient_information", "uncertainties": []},
            {**review_fixture(), "approved": True},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_review(value)

    def test_fenced_json_is_supported_and_non_json_is_rejected(self):
        with patch("command_review.Chat") as chat:
            chat.return_value.chat_context.return_value = "```json\n" + json.dumps(review_fixture()) + "\n```"
            self.assertEqual(review_commands({}), review_fixture())
            chat.return_value.chat_context.return_value = "These commands look fine."
            with self.assertRaises(ValueError):
                review_commands({})

    def test_reported_issues_override_a_conflicting_favorable_verdict(self):
        raw = {**review_fixture(), "issues": ["Creates a directory instead of a text file."]}
        review = validate_review(raw)
        self.assertEqual(review["verdict"], "issues_found")
        self.assertEqual(review["issues"], raw["issues"])
        self.assertEqual(raw["verdict"], "reasonable")
        self.assertIn("Assessment: Issues found", format_review(review, "Create a text file"))

    def test_natural_language_confirmation_includes_findings_and_recommendation(self):
        review = {**review_fixture(), "verdict": "issues_found", "risk": "high",
                  "summary": "命令会删除指定目录。", "effects": ["删除 /work/dist"],
                  "issues": ["目标与请求不符。"], "uncertainties": ["目录内容尚未核实。"],
                  "recommendation": "建议拒绝并修改目标路径。"}
        text = format_review(validate_review(review), "删除构建目录")
        for expected in ("命令审阅", "判断: 存在问题", "风险: 高", "删除 /work/dist",
                         "目标与请求不符。", "目录内容尚未核实。", "建议拒绝并修改目标路径。"):
            self.assertIn(expected, text)
        self.assertIn("Assessment: Reasonable", format_review(review_fixture(), "Create a file"))

    def test_action_identity_covers_command_order_quoting_directory_and_shell(self):
        original = command_action_id(["cd tmp", "touch a.txt"], "/work", "/bin/bash")
        for commands, cwd, shell in (
            (["touch a.txt", "cd tmp"], "/work", "/bin/bash"),
            (["cd tmp", 'touch "a.txt"'], "/work", "/bin/bash"),
            (["cd tmp", "touch a.txt"], "/elsewhere", "/bin/bash"),
            (["cd tmp", "touch a.txt"], "/work", "other-bash"),
        ):
            self.assertNotEqual(command_action_id(commands, cwd, shell), original)


if __name__ == "__main__":
    unittest.main()
