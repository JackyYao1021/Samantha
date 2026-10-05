"""Persist provider-returned traces without changing model or answer behavior."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import Chat
from content_search.config import Settings
from content_search.models import QwenAnalyzer
from history_cli import main as history_main
from interaction_log import InteractionJournal, InteractionLogError, LogSession, operation
from model_reasoning import log_response_reasoning
from samantha import process
from test_interaction_log import JournalTestCase
from workflow import default_services


def response(content='{"ok": true}', finish="stop", **fields):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, **fields)
    )])


class ModelReasoningTests(JournalTestCase):
    def setUp(self):
        super().setUp()
        environment = patch.dict(os.environ, {
            "SAMANTHA_LOG_REASONING": "true", "SAMANTHA_AZURE_FALLBACK": "false",
            "QWEN_API_MODE": "openai", "QWEN_MODEL": "qwen3:4b", "QWEN_REASONING_EFFORT": "none",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def reasoning(self):
        return [event for event in self.read()["events"] if event["kind"] == "model_reasoning"]

    def call_openai(self, result, *, temporary=False):
        with InteractionJournal() as journal, journal.start_session("fixture request") as session:
            with operation("agent.clarify", {"prompt": "fixture"}), patch("chat.OpenAI") as client:
                client.return_value.chat.completions.create.return_value = result
                chat = Chat()
                answer = chat.temp_chat("hello") if temporary else chat.chat_context("hello")
            session.finish("succeeded")
        return answer, chat

    def test_compatibility_fields_preserve_text_and_link_to_operation(self):
        text = "  返回的思考文本\nfixture trace🙂  "
        for source in ("thinking", "reasoning_content", "reasoning"):
            with self.subTest(source=source):
                answer, chat = self.call_openai(response(**{source: text}))
                event = self.reasoning()[0]
                self.assertEqual(answer, '{"ok": true}')
                self.assertEqual(chat.messages[-1], {"role": "assistant", "content": answer})
                self.assertEqual(event["data"]["text"], text)
                self.assertEqual(event["data"]["source"], source)
                self.assertEqual(event["data"]["operation_name"], "agent.clarify")
                self.assertEqual(event["data"]["provider"], "qwen")
                self.assertTrue(event["data"]["complete"])
                start = next(e for e in self.read()["events"] if e["kind"] == "operation_started")
                self.assertEqual(event["data"]["operation_id"], start["data"]["operation_id"])
                self.assertEqual(event["turn"], 1)

    def test_actual_sdk_response_retains_extension_field(self):
        result = ChatCompletion.model_validate({
            "id": "fixture", "created": 1, "model": "qwen3:4b", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"ok": true}', "reasoning_content": "SDK fixture trace",
            }}],
        })
        self.call_openai(result)
        self.assertEqual(self.reasoning()[0]["data"]["text"], "SDK fixture trace")

    def test_inline_trace_is_saved_before_filtering_and_temporary_chat_is_covered(self):
        answer, chat = self.call_openai(response("<think>\nfixture trace\n</think>\n{\"ok\": true}"), temporary=True)
        self.assertEqual(answer, '{"ok": true}')
        self.assertEqual(len(chat.messages), 1)
        event = self.reasoning()[0]["data"]
        self.assertEqual(event["source"], "think_tag")
        self.assertEqual(event["text"], "\nfixture trace\n")

    def test_literal_tags_inside_json_are_not_misclassified(self):
        content = '{"Commands": ["echo </think>"]}'
        answer, _ = self.call_openai(response(content))
        self.assertEqual(answer, content)
        event = self.reasoning()[0]["data"]
        self.assertFalse(event["available"])
        self.assertIsNone(event["text"])

    def test_missing_trace_and_multiple_fields_have_explicit_provenance(self):
        self.call_openai(response(thinking="", reasoning=None, reasoning_content={"unsupported": True}))
        self.assertFalse(self.reasoning()[0]["data"]["available"])
        self.call_openai(response(thinking="native fixture", reasoning_content="compatibility fixture"))
        events = self.reasoning()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["data"]["call_id"], events[1]["data"]["call_id"])
        self.assertEqual({e["data"]["source"] for e in events}, {"thinking", "reasoning_content"})

    def test_truncated_and_unclosed_traces_survive_rejected_answer(self):
        for result in (response('<think>unfinished fixture', finish="stop"),
                       response('{"partial":', finish="length", reasoning_content="returned fixture")):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.call_openai(result)
            data = self.read()
            self.assertEqual(data["session"]["status"], "failed")
            event = self.reasoning()[0]["data"]
            self.assertTrue(event["available"])
            self.assertFalse(event["complete"])

    def test_ollama_native_trace_is_saved_even_if_response_is_incomplete(self):
        for done in (True, False):
            with self.subTest(done=done):
                with patch.dict(os.environ, {"QWEN_API_MODE": "ollama", "QWEN_REASONING_EFFORT": "none"}), \
                        patch("chat.OpenAI") as qwen, patch("chat.httpx.Client") as http:
                    qwen.return_value.base_url = "http://localhost:11434/v1/"
                    http.return_value.__enter__.return_value.post.return_value.json.return_value = {
                        "done": done, "done_reason": "stop", "message": {
                            "content": '{"ok": true}', "thinking": "  native fixture\n",
                        },
                    }
                    try:
                        with InteractionJournal() as journal, journal.start_session("native") as session:
                            chat = Chat()
                            answer = chat.chat_context("hello")
                            self.assertEqual(answer, '{"ok": true}')
                            session.finish("succeeded")
                    except RuntimeError:
                        self.assertFalse(done)
                    self.assertFalse(http.return_value.__enter__.return_value.post.call_args.kwargs["json"]["think"])
                event = self.reasoning()[0]["data"]
                self.assertEqual(event["text"], "  native fixture\n")
                self.assertEqual(event["transport"], "ollama")
                self.assertEqual(event["complete"], done)
                self.assertEqual(len(self.reasoning()), 1)

    def test_azure_fallback_response_has_correct_model_and_provider(self):
        with patch.dict(os.environ, {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture-key"}), \
                patch("chat.OpenAI") as qwen, patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
            azure.return_value.chat.completions.create.return_value = response(reasoning="fallback fixture")
            with InteractionJournal() as journal, journal.start_session("fallback") as session:
                chat = Chat()
                chat.chat_context("hello")
                session.finish("succeeded")
        data = self.reasoning()[0]["data"]
        self.assertEqual(data["provider"], "azure")
        self.assertEqual(data["model"], chat.model)
        self.assertNotIn("fixture-key", json.dumps(self.read()))

    def test_opt_out_preserves_answer_and_does_not_enable_thinking(self):
        with patch.dict(os.environ, {"SAMANTHA_LOG_REASONING": "false", "QWEN_REASONING_EFFORT": "none"}):
            answer, chat = self.call_openai(response(reasoning_content="fixture trace"))
        self.assertEqual(answer, '{"ok": true}')
        self.assertEqual(chat.reasoning_effort, "none")
        self.assertEqual(self.reasoning(), [])

    def test_reasoning_write_failure_does_not_trigger_azure_request(self):
        original = LogSession.record
        def fail_reasoning(session, kind, data, **kwargs):
            if kind == "model_reasoning":
                raise InteractionLogError("trace write failed")
            return original(session, kind, data, **kwargs)
        with patch.dict(os.environ, {"QWEN_API_MODE": "ollama", "SAMANTHA_AZURE_FALLBACK": "true",
                                     "AZURE_OPENAI_API_KEY": "fixture-key"}), \
                patch("chat.OpenAI") as qwen, patch("chat.AzureOpenAI") as azure, \
                patch("chat.httpx.Client") as http, patch.object(LogSession, "record", fail_reasoning):
            qwen.return_value.base_url = "http://localhost:11434/v1/"
            http.return_value.__enter__.return_value.post.return_value.json.return_value = {
                "done": True, "message": {"content": "answer", "thinking": "fixture"},
            }
            with self.assertRaisesRegex(InteractionLogError, "trace write failed"):
                with InteractionJournal() as journal, journal.start_session("failure"):
                    Chat().chat_context("hello")
            azure.return_value.chat.completions.create.assert_not_called()

    def test_workflow_associates_traces_with_agents_and_keeps_retry_traces(self):
        from test_workflow import review_fixture
        review_json = json.dumps(review_fixture())
        contents = [
            '{"is_jump": false, "requirement_summary": "Create a file."}',
            '{"text": "Create test.txt."}', '{"Commands": ["first"]}',
            review_json, '{"text": "Correct the request."}',
            '{"Commands": ["second"]}', review_json,
        ]
        fake = replace(default_services(), execute_commands=Mock(side_effect=[
            {"success": False, "output": "fixture error", "current_dir": os.getcwd()},
            {"success": True, "output": "done", "current_dir": os.getcwd()},
        ]))
        with patch("chat.OpenAI") as qwen:
            qwen.return_value.chat.completions.create.side_effect = [
                response(content, reasoning_content=f"fixture {index}") for index, content in enumerate(contents)
            ]
            state = process("Create a file", services=fake, input_fn=Mock(side_effect=["y", "y"]),
                            output_fn=lambda _: None)
        self.assertEqual(state["status"], "succeeded")
        events = self.reasoning()
        self.assertEqual(len(events), 7)
        self.assertEqual([e["data"]["operation_name"] for e in events], [
            "agent.clarify", "agent.parse_intent", "agent.generate_commands", "agent.review_commands",
            "agent.correct_error", "agent.generate_commands", "agent.review_commands",
        ])
        self.assertEqual([e["turn"] for e in events], [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(len({e["data"]["operation_id"] for e in events}), 7)
        self.assertEqual(len({e["data"]["call_id"] for e in events}), 7)

    def test_reasoning_only_history_and_full_exports_include_traces(self):
        self.call_openai(response(reasoning_content="export fixture"))
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(history_main(["show", "latest", "--reasoning-only", "--json"]), 0)
        shown = json.loads(stdout.getvalue())
        self.assertEqual(shown["events"], self.reasoning())
        target = self.path.parent / "trace.jsonl"
        with redirect_stdout(StringIO()):
            self.assertEqual(history_main(["export", "latest", "--format", "jsonl", "--output", str(target)]), 0)
        events = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()][1:]
        self.assertEqual(events, self.read()["events"])

    def test_content_analyzer_records_returned_traces_for_annotation_and_answer(self):
        with patch("content_search.models.OpenAI") as qwen:
            qwen.return_value.chat.completions.create.side_effect = [
                response('{"summary": "fixture", "description": "", "ocr_text": "", "tags": []}',
                         reasoning_content="annotation fixture"),
                response("Source [1]", reasoning_content="answer fixture"),
            ]
            with InteractionJournal() as journal, journal.start_session("content fixture") as session:
                analyzer = QwenAnalyzer(Settings())
                try:
                    with operation("content.index_file", {}):
                        self.assertEqual(analyzer.annotate("text")["summary"], "fixture")
                    with operation("content.ask", {}):
                        self.assertEqual(analyzer.answer("question", [{
                            "path": "a.txt", "text": "text", "kind": "native_text",
                        }]), "Source [1]")
                finally:
                    analyzer.close()
                session.finish("succeeded")
        events = self.reasoning()
        self.assertEqual([e["data"]["text"] for e in events], ["annotation fixture", "answer fixture"])
        self.assertEqual([e["data"]["provider"] for e in events], ["qwen_vl", "qwen_vl"])

    def test_operation_context_is_restored_after_nested_calls_and_failures(self):
        with InteractionJournal() as journal, journal.start_session("context") as session:
            with operation("agent.outer", {}):
                try:
                    with operation("agent.inner", {}):
                        raise ValueError("fixture")
                except ValueError:
                    pass
                log_response_reasoning(response(reasoning="outer fixture"), model="fixture", provider="fixture")
            log_response_reasoning(response(reasoning="unscoped fixture"), model="fixture", provider="fixture")
            session.finish("succeeded")
        events = self.reasoning()
        self.assertEqual(events[0]["data"]["operation_name"], "agent.outer")
        self.assertNotIn("operation_id", events[1]["data"])
