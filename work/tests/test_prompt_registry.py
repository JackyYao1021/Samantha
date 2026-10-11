"""Prompt provenance changes independently of rendering and request interfaces."""

import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import request_text
from clarify_intent import clarification_prompt, clarify_intent_once
from command_review import review_commands, REVIEW_PROMPT
from error_correction import error_correction_agent
from intent_explain import parse_user_intent
from n2c import natural_language_to_command_agent
from prompts.registry import bind_prompt, compose_prompt, describe_registry, render_prompt


class PromptRegistryTests(unittest.TestCase):
    def test_id_version_are_immutable_and_new_version_permits_changes(self):
        identifier = "test.version." + uuid.uuid4().hex
        prompt = bind_prompt(identifier, "hello")
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "hello")
        self.assertEqual(bind_prompt(identifier, "hello"), prompt)
        with self.assertRaisesRegex(ValueError, "new version"):
            bind_prompt(identifier, "changed")
        second = bind_prompt(identifier, "changed", version="2")
        self.assertNotEqual(prompt.prompt_metadata[0]["template_sha256"], second.prompt_metadata[0]["template_sha256"])

    def test_template_hash_stays_fixed_but_rendered_hash_tracks_variables(self):
        identifier = "test.render." + uuid.uuid4().hex
        template = "Current path: {path}\nCurrent time: {time}\n"
        first = render_prompt(identifier, template, variables={"path": "/one", "time": "first"})
        second = render_prompt(identifier, template, variables={"path": "/two", "time": "second"})
        self.assertEqual(first, "Current path: /one\nCurrent time: first\n")
        self.assertEqual(first.prompt_metadata[0]["template_sha256"], second.prompt_metadata[0]["template_sha256"])
        self.assertNotEqual(first.prompt_metadata[0]["rendered_sha256"], second.prompt_metadata[0]["rendered_sha256"])
        self.assertEqual(first.prompt_metadata[0]["rendered_sha256"], hashlib.sha256(first.encode()).hexdigest())

    def test_composition_keeps_role_and_envelope_provenance_without_text_export(self):
        role = bind_prompt("test.compose." + uuid.uuid4().hex, "private system text")
        combined = compose_prompt(role, "\nEnvelope.", template_id="test.envelope")
        self.assertEqual(combined, "private system text\nEnvelope.")
        self.assertEqual(len(combined.prompt_metadata), 2)
        self.assertEqual(combined.prompt_metadata[-1]["rendered_sha256"], hashlib.sha256(combined.encode()).hexdigest())
        self.assertNotIn("private system text", str(describe_registry()))

    def test_legacy_dynamic_clarification_text_is_preserved_and_source_hash_stable(self):
        values = []
        for directory, moment in (("/first", "2026-10-11 12:00:00"), ("/second", "2026-10-11 13:00:00")):
            with patch("clarify_intent.get_current_path", return_value=directory), \
                    patch("clarify_intent.get_current_time", return_value=moment), patch("clarify_intent.Chat") as chat:
                expected = clarification_prompt()
                chat.return_value.chat_context.return_value = '{"is_jump": false, "requirement_summary": "fixture"}'
                clarify_intent_once("fixture")
                sent = chat.call_args.kwargs["begin_messages"]
            self.assertEqual(sent, expected)
            self.assertEqual(sent.prompt_metadata[0]["template_id"], "terminal.clarification")
            self.assertEqual(sent.prompt_metadata[0]["template_kind"], "python_source")
            values.append(sent.prompt_metadata[0])
        self.assertEqual(values[0]["template_sha256"], values[1]["template_sha256"])
        self.assertNotEqual(values[0]["rendered_sha256"], values[1]["rendered_sha256"])

    def test_all_terminal_roles_bind_stable_registry_ids(self):
        with patch("intent_explain.request_text", return_value="plan") as request:
            parse_user_intent("fixture")
            self.assertEqual(request.call_args.args[0].prompt_metadata[0]["template_id"], "terminal.intent")
        with patch("error_correction.request_text", return_value="corrected") as request:
            error_correction_agent("fixture", ["echo fixture"], "fixture error")
            self.assertEqual(request.call_args.args[0].prompt_metadata[0]["template_id"], "terminal.correction")
        with patch("n2c.Chat") as chat:
            chat.return_value.chat_context.return_value = '{"Commands": ["echo fixture"]}'
            natural_language_to_command_agent("fixture")
            self.assertEqual(chat.call_args.args[0].prompt_metadata[0]["template_id"], "terminal.generation")
        review = {"verdict": "reasonable", "risk": "low", "summary": "fixture",
                  "effects": ["prints fixture"], "issues": [], "uncertainties": [], "recommendation": "fixture"}
        import json
        with patch("command_review.Chat") as chat:
            chat.return_value.chat_context.return_value = json.dumps(review)
            review_commands({"user_request": "fixture"})
            sent = chat.call_args.kwargs["begin_messages"]
        self.assertEqual(sent.prompt_metadata[0]["template_id"], "terminal.review.default")
        self.assertTrue(sent.startswith(REVIEW_PROMPT))

    def test_text_request_adds_envelope_and_keeps_original_text_interface(self):
        role = bind_prompt("test.text." + uuid.uuid4().hex, "system instructions")
        with patch("chat.Chat") as chat:
            chat.return_value.chat_context.return_value = '{"text": "# Plan\\nfixture"}'
            self.assertEqual(request_text(role, "fixture"), "# Plan\nfixture")
        sent = chat.call_args.kwargs["begin_messages"]
        self.assertEqual([item["template_id"] for item in sent.prompt_metadata],
                         [role.prompt_metadata[0]["template_id"], "terminal.text_envelope"])
        self.assertTrue(sent.startswith("system instructions\nReturn one JSON object"))


if __name__ == "__main__":
    unittest.main()
