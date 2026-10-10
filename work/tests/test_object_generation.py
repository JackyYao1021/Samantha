"""Opt-in model contracts retain legacy prompts while requiring P0 fields."""

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import request_text
from intent_explain import parse_user_intent
from interaction_log import InteractionJournal
from n2c import natural_language_to_command_agent
from object_tracking.effects import PROPOSAL_EFFECTS_SCHEMA, TRACKING_GENERATION_INSTRUCTIONS, bind_effects
from workflow import default_services


class ObjectGenerationContractTests(unittest.TestCase):
    def test_tracked_generator_requests_only_explanation_and_effects_schema(self):
        response = json.dumps({"Explanation": "Create notes.", "Effects": [{
                                   "kind": "create", "path": "notes.txt",
                                   "content": {"lines": ["alpha"], "ends_with_newline": False},
                                   "destination": None, "entity_id": None, "entity_version": None}]})
        with patch("n2c.Chat") as chat:
            chat.return_value.chat_context.return_value = response
            generated = json.loads(natural_language_to_command_agent("Create notes", object_tracking=True))
            self.assertEqual(len(generated["Commands"]), 1)
            self.assertEqual(generated["Explanation"], "Create notes.")
            self.assertEqual(generated["Effects"][0]["content"], "alpha")
            self.assertEqual(generated["Effects"][0]["path"], str(Path("notes.txt").resolve()))
            bind_effects(generated["Commands"], generated["Effects"], os.getcwd())
            chat.return_value.chat_context.assert_called_once_with("Create notes")
            self.assertIn(TRACKING_GENERATION_INSTRUCTIONS, chat.call_args.args[0])
            contract = chat.call_args.kwargs["response_format"]
        self.assertEqual(contract["type"], "json_schema")
        self.assertTrue(contract["json_schema"]["strict"])
        schema = contract["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"Explanation", "Effects"})
        self.assertEqual(set(schema["properties"]), {"Explanation", "Effects"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["Effects"], PROPOSAL_EFFECTS_SCHEMA)
        fields = schema["properties"]["Effects"]["items"]
        self.assertEqual(set(fields["required"]),
                         {"kind", "path", "destination", "content", "entity_id", "entity_version"})
        self.assertEqual(fields["properties"]["kind"]["enum"], ["create", "rename", "append"])
        self.assertFalse(fields["additionalProperties"])
        content = fields["properties"]["content"]
        self.assertEqual(set(content["type"]), {"object", "null"})
        self.assertEqual(set(content["required"]), {"lines", "ends_with_newline"})
        self.assertEqual(content["properties"]["lines"]["type"], "array")
        self.assertEqual(content["properties"]["lines"]["items"]["type"], "string")
        self.assertEqual(content["properties"]["ends_with_newline"]["type"], "boolean")
        self.assertFalse(content["additionalProperties"])

    def test_tracked_effects_render_exact_newlines_quotes_and_paths_before_review(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                filename = "会议 notes 'draft'.txt"
                destination = "todo '文件'.txt"
                content = "alpha\n$HOME; `whoami` 'quoted' \\ tail\nbeta\n"
                raw_content = {"lines": ["alpha", "$HOME; `whoami` 'quoted' \\ tail", "beta"],
                               "ends_with_newline": True}
                proposals = [
                    {"kind": "create", "path": filename, "destination": None, "content": raw_content,
                     "entity_id": None, "entity_version": None},
                    {"kind": "append", "path": filename, "destination": None, "content": raw_content,
                     "entity_id": "known-file", "entity_version": 2},
                    {"kind": "rename", "path": filename, "destination": destination, "content": None,
                     "entity_id": "known-file", "entity_version": 2},
                ]
                for effect in proposals:
                    with self.subTest(kind=effect["kind"]), patch("n2c.Chat") as chat:
                        chat.return_value.chat_context.return_value = json.dumps({
                            "Explanation": "Exact fixture effect", "Effects": [effect]},
                            ensure_ascii=False)
                        result = json.loads(natural_language_to_command_agent("Request", object_tracking=True))
                        self.assertEqual(len(result["Commands"]), 1)
                        self.assertEqual(result["Explanation"], "Exact fixture effect")
                        normalized = result["Effects"][0]
                        self.assertEqual(normalized["path"], str(Path(filename).resolve()))
                        self.assertEqual(normalized["entity_id"], effect["entity_id"])
                        self.assertEqual(normalized["entity_version"], effect["entity_version"])
                        self.assertEqual(normalized["content"], None if effect["kind"] == "rename" else content)
                        words = shlex.split(result["Commands"][0])
                        if effect["kind"] == "rename":
                            self.assertEqual(normalized["destination"], str(Path(destination).resolve()))
                            self.assertEqual(words, ["mv", "--", Path(filename).resolve().as_posix(),
                                                     Path(destination).resolve().as_posix()])
                        else:
                            self.assertEqual(words, ["printf", "%s", content,
                                                     ">>" if effect["kind"] == "append" else ">",
                                                     Path(filename).resolve().as_posix()])
                        bind_effects(result["Commands"], result["Effects"], os.getcwd())
                self.assertEqual(list(Path.cwd().iterdir()), [], "Generation must not execute or create files.")
            finally:
                os.chdir(previous)

    def test_unexpected_model_commands_cannot_be_silently_replaced_for_tracked_effect(self):
        effect = {"kind": "create", "path": "notes.txt",
                  "content": {"lines": ["alpha"], "ends_with_newline": False},
                  "destination": None, "entity_id": None, "entity_version": None}
        for commands in ([], ["printf '%s' 'alpha' > notes.txt"], ["rm unrelated.txt"],
                         ["pwd", "touch notes.txt"], None):
            with self.subTest(commands=commands), patch("n2c.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps({
                    "Commands": commands, "Explanation": "Proposal", "Effects": [effect]})
                with self.assertRaises(ValueError):
                    natural_language_to_command_agent("Request", object_tracking=True)
                chat.assert_called_once()

    def test_malformed_effects_are_rejected_before_canonical_generation(self):
        valid = {"kind": "create", "path": "notes.txt",
                 "content": {"lines": ["alpha"], "ends_with_newline": False},
                 "destination": None, "entity_id": None, "entity_version": None}
        cases = [None, [{**valid, "kind": "copy"}], [{**valid, "content": None}],
                 [{**valid, "path": "line\nbreak.txt"}], [{**valid, "entity_id": "forged", "entity_version": 1}],
                 [{**valid, "extra": "unexpected"}], [valid, valid]]
        for effects in cases:
            with self.subTest(effects=effects), patch("n2c.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps({
                    "Explanation": "Proposal", "Effects": effects})
                with self.assertRaises(ValueError):
                    natural_language_to_command_agent("Request", object_tracking=True)
                chat.assert_called_once()

    def test_line_content_encoding_preserves_actual_and_literal_newlines(self):
        cases = [({"lines": ["alpha"], "ends_with_newline": True}, "alpha\n"),
                 ({"lines": ["alpha", "beta"], "ends_with_newline": False}, "alpha\nbeta"),
                 ({"lines": ["alpha", "beta"], "ends_with_newline": True}, "alpha\nbeta\n"),
                 ({"lines": ["alpha", "", "beta"], "ends_with_newline": True}, "alpha\n\nbeta\n"),
                 ({"lines": [r"alpha\n"], "ends_with_newline": False}, r"alpha\n"),
                 ({"lines": [r"alpha\n", "beta"], "ends_with_newline": True}, "alpha\\n\nbeta\n")]
        for raw_content, expected in cases:
            with self.subTest(content=raw_content), patch("n2c.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps({
                    "Explanation": "Explicit line encoding", "Effects": [{
                        "kind": "create", "path": "notes.txt", "destination": None,
                        "content": raw_content, "entity_id": None, "entity_version": None}]})
                result = json.loads(natural_language_to_command_agent("Request", object_tracking=True))
                actual = result["Effects"][0]["content"]
                self.assertEqual(actual.encode("utf-8"), expected.encode("utf-8"))
                self.assertEqual(shlex.split(result["Commands"][0])[2], expected)
                bind_effects(result["Commands"], result["Effects"], os.getcwd())
                chat.assert_called_once()

    def test_invalid_line_content_encoding_fails_without_fallback_or_escape_rewrite(self):
        cases = ["alpha\\n", {"lines": ["alpha\nbeta"], "ends_with_newline": False},
                 {"lines": ["alpha"]}, {"ends_with_newline": False},
                 {"lines": "alpha", "ends_with_newline": False},
                 {"lines": [1], "ends_with_newline": False},
                 {"lines": [None], "ends_with_newline": False},
                 {"lines": ["alpha"], "ends_with_newline": "true"},
                 {"lines": ["alpha"], "ends_with_newline": 1},
                 {"lines": ["alpha"], "ends_with_newline": None},
                 {"lines": ["alpha"], "ends_with_newline": False, "extra": "unexpected"}]
        for raw_content in cases:
            with self.subTest(content=raw_content), patch("n2c.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps({
                    "Explanation": "Invalid raw line encoding", "Effects": [{
                        "kind": "create", "path": "notes.txt", "destination": None,
                        "content": raw_content, "entity_id": None, "entity_version": None}]})
                with self.assertRaises(ValueError):
                    natural_language_to_command_agent("Request", object_tracking=True)
                chat.assert_called_once()
                chat.return_value.chat_context.assert_called_once_with("Request")

    def test_explicit_empty_effects_make_one_baseline_fallback_and_preserve_commands(self):
        proposal = {"Commands": ["mkdir -p out", "cp '*.txt' out/"],
                    "Explanation": "Untracked operation"}
        with tempfile.TemporaryDirectory() as directory, \
                InteractionJournal(Path(directory) / "generation.sqlite3") as journal, \
                journal.start_session("Request") as session, patch("n2c.Chat") as chat:
            chat.return_value.chat_context.side_effect = [
                json.dumps({"Explanation": "Unsupported operation", "Effects": []}), json.dumps(proposal)]
            result = json.loads(natural_language_to_command_agent("Request", object_tracking=True))
            self.assertEqual(chat.call_count, 2)
            self.assertEqual(chat.return_value.chat_context.call_count, 2)
            self.assertEqual(chat.call_args_list[0].kwargs["response_format"]["type"], "json_schema")
            self.assertEqual(chat.call_args_list[1].kwargs["response_format"], {"type": "json_object"})
            self.assertNotIn(TRACKING_GENERATION_INSTRUCTIONS, chat.call_args_list[1].args[0])
            session.finish("succeeded")
            evidence = journal.read_session(session.session_id)
            fallback_starts = [event for event in evidence["events"] if event["kind"] == "operation_started"
                               and event["data"].get("name") == "agent.generate_untracked_commands"]
            self.assertEqual(len(fallback_starts), 1)
            fallback_finishes = [event for event in evidence["events"] if event["kind"] == "operation_finished"
                                 and event["data"].get("name") == "agent.generate_untracked_commands"]
            self.assertEqual(len(fallback_finishes), 1)
            self.assertEqual(fallback_finishes[0]["data"]["operation_id"],
                             fallback_starts[0]["data"]["operation_id"])
            self.assertEqual(json.loads(fallback_finishes[0]["data"]["result"]), proposal)
        self.assertEqual(result, {**proposal, "Effects": []})

    def test_missing_effects_and_unexpected_raw_commands_fail_without_fallback(self):
        responses = [{"Explanation": "Missing Effects"},
                     {"Explanation": "Unexpected commands", "Effects": [], "Commands": []},
                     {"Explanation": "Unexpected commands", "Effects": [], "Commands": ["pwd"]}]
        for response in responses:
            with self.subTest(response=response), patch("n2c.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps(response)
                with self.assertRaises(ValueError):
                    natural_language_to_command_agent("Request", object_tracking=True)
                chat.assert_called_once()
                chat.return_value.chat_context.assert_called_once_with("Request")

    def test_baseline_generator_keeps_json_object_and_original_prompt(self):
        response = '{"Commands":["pwd"],"Explanation":"Show directory"}'
        with patch("n2c.Chat") as chat:
            chat.return_value.chat_context.return_value = response
            self.assertEqual(natural_language_to_command_agent("Show directory"), response)
            self.assertEqual(chat.call_args.kwargs["response_format"], {"type": "json_object"})
            self.assertNotIn(TRACKING_GENERATION_INSTRUCTIONS, chat.call_args.args[0])

    def test_tracked_intent_uses_strict_text_schema_and_preserves_supplied_prompt(self):
        request = "Append beta to the verified current file object."
        with patch("intent_explain.request_text", return_value="# Intent Breakdown\n1. Append beta.") as text:
            result = parse_user_intent(request, object_tracking=True)
            self.assertEqual(result, "# Intent Breakdown\n1. Append beta.")
            self.assertEqual(text.call_args.args[1], request)
            self.assertIs(text.call_args.kwargs["strict_schema"], True)
            self.assertIn("supplied verified current file", text.call_args.args[0])
            self.assertIn("exact requested UTF-8 content/newlines", text.call_args.args[0])
            self.assertIn("explicit targets and constraints", text.call_args.args[0])
            self.assertIn("Do not add directory changes", text.call_args.args[0])

    def test_baseline_intent_keeps_existing_text_contract(self):
        with patch("intent_explain.request_text", return_value="Existing plan") as text:
            self.assertEqual(parse_user_intent("List files"), "Existing plan")
            self.assertNotIn("strict_schema", text.call_args.kwargs)
            self.assertNotIn("supplied verified current file", text.call_args.args[0])

    def test_default_services_enable_both_tracked_model_contracts_only_on_opt_in(self):
        with patch("intent_explain.parse_user_intent", return_value="Plan") as parse, \
                patch("n2c.natural_language_to_command_agent", return_value="Commands") as generate:
            tracked = default_services(object_tracking=True)
            self.assertEqual(tracked.parse_intent("Request"), "Plan")
            self.assertEqual(tracked.generate_commands("Plan"), "Commands")
            parse.assert_called_once_with("Request", object_tracking=True)
            generate.assert_called_once_with("Plan", object_tracking=True)
            parse.reset_mock()
            generate.reset_mock()
            baseline = default_services()
            baseline.parse_intent("Request")
            baseline.generate_commands("Plan")
            parse.assert_called_once_with("Request")
            generate.assert_called_once_with("Plan")


class StrictTextResponseTests(unittest.TestCase):
    def test_strict_text_response_requests_nonempty_string_and_returns_markdown(self):
        plan = "# Intent Breakdown\n1. Create notes.txt with alpha."
        with patch("chat.Chat") as chat:
            chat.return_value.chat_context.return_value = json.dumps({"text": plan})
            self.assertEqual(request_text("Instructions", "User request", max_tokens=128, strict_schema=True), plan)
            chat.return_value.chat_context.assert_called_once_with("User request")
            self.assertEqual(chat.call_args.kwargs["max_tokens"], 128)
            self.assertIn('one field "text"', chat.call_args.kwargs["begin_messages"])
            contract = chat.call_args.kwargs["response_format"]
        self.assertEqual(contract["type"], "json_schema")
        self.assertEqual(contract["json_schema"]["name"], "text_response")
        self.assertTrue(contract["json_schema"]["strict"])
        schema = contract["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["text"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["text"], {"type": "string", "minLength": 1})

    def test_strict_text_response_rejects_missing_empty_or_wrong_type_output(self):
        for output in ({}, {"text": None}, {"text": ""}, {"text": " \n\t"}, {"text": 1},
                       {"text": False}, {"text": []}, {"text": {}}):
            with self.subTest(output=output), patch("chat.Chat") as chat:
                chat.return_value.chat_context.return_value = json.dumps(output)
                with self.assertRaisesRegex(ValueError, "non-empty text"):
                    request_text("Instructions", "Request", strict_schema=True)

    def test_strict_text_response_rejects_json_array_and_invalid_json(self):
        for output in ('[{"text":"plan"}]', '{"text":'):
            with self.subTest(output=output), patch("chat.Chat") as chat:
                chat.return_value.chat_context.return_value = output
                with self.assertRaises(ValueError):
                    request_text("Instructions", "Request", strict_schema=True)


if __name__ == "__main__":
    unittest.main()
