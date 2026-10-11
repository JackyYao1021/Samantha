"""Offline transport and privacy checks for physical model-call telemetry."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import Chat
from interaction_log import InteractionJournal, InteractionLogError, operation
from model_telemetry import capture_model_calls, model_request, summarize_calls, USAGE_FIELDS
from samantha import process
from workflow import WorkflowServices


def response(usage=None):
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="answer"))])


class ModelTelemetryTests(unittest.TestCase):
    def test_openai_parameters_usage_and_prompt_are_captured_without_text(self):
        raw_usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=3),
                                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2))
        messages = [{"role": "user", "content": "private input with sk-secret-credential"}]
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = response(raw_usage)
            with capture_model_calls() as calls:
                received = Chat(max_tokens=123, temperature=0.3, top_p=0.8).send_message(messages)
        self.assertEqual(received.usage, raw_usage)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["status"], "succeeded")
        self.assertIsNone(call["exception_type"])
        self.assertEqual(call["usage"], {"input_tokens": 12, "output_tokens": 7,
                                       "cache_read_tokens": 3, "cache_write_tokens": None,
                                       "reasoning_tokens": 2})
        self.assertGreaterEqual(call["duration_ms"], 0)
        params = call["effective_parameters"]
        self.assertEqual((params["max_tokens"], params["temperature"], params["top_p"]), (123, 0.3, 0.8))
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertEqual(len(params["prompt_sha256"]), 64)
        self.assertNotIn("private input", json.dumps(calls))
        self.assertNotIn("sk-secret-credential", json.dumps(calls))
        self.assertEqual(sdk.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"],
                         messages[0]["content"] + "\n/no_think")
        self.assertEqual(messages[0]["content"], "private input with sk-secret-credential")

    def test_failure_and_fallback_are_two_calls_without_url_or_exception_text(self):
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "azure-secret",
                       "QWEN_BASE_URL": "http://username:password@localhost:11434/v1?key=url-secret"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("sk-error-secret at user:password@host")
            azure.return_value.chat.completions.create.return_value = response({"prompt_tokens": 5, "completion_tokens": 0})
            with capture_model_calls() as calls:
                Chat().send_message([{"role": "user", "content": "hello"}])
            self.assertEqual(qwen.call_args.kwargs["max_retries"], 0)
            self.assertEqual(azure.call_args.kwargs["max_retries"], 0)
        self.assertEqual([(call["provider"], call["status"]) for call in calls],
                         [("qwen", "failed"), ("azure", "succeeded")])
        self.assertEqual(calls[0]["exception_type"], "RuntimeError")
        self.assertEqual(calls[0]["usage"], dict.fromkeys(USAGE_FIELDS))
        self.assertEqual(calls[1]["usage"]["output_tokens"], 0)
        self.assertNotEqual(calls[0]["call_id"], calls[1]["call_id"])
        self.assertNotEqual(calls[0]["effective_parameters"]["prompt_sha256"],
                            calls[1]["effective_parameters"]["prompt_sha256"])
        for secret in ("username", "password", "azure-secret", "url-secret", "sk-error-secret", "localhost"):
            self.assertNotIn(secret, json.dumps(calls))
        summary = summarize_calls(calls)
        self.assertEqual(summary["physical_llm_calls"], 2)
        self.assertIsNone(summary["usage"]["input_tokens"])
        self.assertEqual(summary["usage_coverage"]["input_tokens"], {
            "reported_calls": 1, "total_calls": 2, "ratio": 0.5, "reported_sum": 5})
        self.assertIsNone(summary["api_cost"])

    def test_native_usage_and_actual_payload_controls(self):
        with patch.dict(os.environ, {"QWEN_API_MODE": "ollama"}, clear=True), \
                patch("chat.OpenAI") as sdk, patch("chat.httpx.Client") as http:
            sdk.return_value.base_url = "http://localhost:11434/v1/"
            raw = http.return_value.__enter__.return_value.post.return_value
            raw.json.return_value = {"done": True, "message": {"content": "answer"},
                                     "prompt_eval_count": 11, "eval_count": 9}
            with capture_model_calls() as calls:
                Chat(response_format={"type": "json_object"}).send_message([])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["transport"], "ollama")
        self.assertEqual(calls[0]["usage"], {"input_tokens": 11, "output_tokens": 9,
                                          "cache_read_tokens": None, "cache_write_tokens": None,
                                          "reasoning_tokens": None})
        params = calls[0]["effective_parameters"]
        self.assertEqual(params["num_predict"], 4096)
        self.assertFalse(params["think"])
        self.assertFalse(params["stream"])
        self.assertEqual(len(params["format_sha256"]), 64)
        sdk.return_value.chat.completions.create.assert_not_called()

    def test_native_invalid_payload_and_client_setup_do_not_count_requests(self):
        for setup_error in (False, True):
            with self.subTest(setup_error=setup_error), \
                    patch.dict(os.environ, {"QWEN_API_MODE": "ollama"}, clear=True), \
                    patch("chat.OpenAI") as sdk, patch("chat.httpx.Client") as http:
                sdk.return_value.base_url = "http://localhost:11434/v1/"
                if setup_error:
                    http.return_value.__enter__.side_effect = RuntimeError("client setup failed")
                    fmt = None
                else:
                    fmt = {"type": "unsupported"}
                with capture_model_calls() as calls:
                    with self.assertRaises(RuntimeError):
                        Chat(response_format=fmt).send_message([])
                self.assertEqual(calls, [])
                http.return_value.__enter__.return_value.post.assert_not_called()

    def test_native_http_and_json_failure_count_once(self):
        for stage in ("post", "raise_for_status", "json"):
            with self.subTest(stage=stage), patch.dict(os.environ, {"QWEN_API_MODE": "ollama"}, clear=True), \
                    patch("chat.OpenAI") as sdk, patch("chat.httpx.Client") as http:
                sdk.return_value.base_url = "http://localhost:11434/v1/"
                client = http.return_value.__enter__.return_value
                target = client.post if stage == "post" else getattr(client.post.return_value, stage)
                target.side_effect = ValueError("private malformed response")
                with capture_model_calls() as calls:
                    with self.assertRaises(RuntimeError):
                        Chat().send_message([])
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["status"], "failed")
                self.assertEqual(calls[0]["exception_type"], "ValueError")
                self.assertNotIn("private malformed response", json.dumps(calls))

    def test_capture_resets_and_nested_calls_are_not_lost(self):
        with capture_model_calls() as outer:
            with capture_model_calls() as inner:
                with model_request(provider="qwen", transport="openai", model="model", parameters={}):
                    pass
            with model_request(provider="qwen", transport="openai", model="model", parameters={}):
                pass
        with capture_model_calls() as later:
            pass
        self.assertEqual((len(outer), len(inner), len(later)), (2, 1, 0))
        inner[0]["status"] = "edited"
        self.assertEqual(outer[0]["status"], "succeeded")
        self.assertEqual(summarize_calls(later)["usage"], dict.fromkeys(USAGE_FIELDS))
        self.assertIsNone(summarize_calls(later)["usage_coverage"]["input_tokens"]["ratio"])

    def test_tokens_are_provider_counts_without_coercion_or_estimation(self):
        records = []
        for raw_usage in ({"prompt_tokens": 0, "completion_tokens": 0},
                          {"prompt_tokens": "100", "completion_tokens": True},
                          {"prompt_tokens": -1, "completion_tokens": 1.5}):
            with capture_model_calls() as calls:
                with model_request(provider="qwen", transport="openai", model="model", parameters={}) as call:
                    call["response"] = response(raw_usage)
            records.extend(calls)
        self.assertEqual(records[0]["usage"]["input_tokens"], 0)
        self.assertEqual(summarize_calls(records[:1])["usage"]["input_tokens"], 0)
        for record in records[1:]:
            self.assertIsNone(record["usage"]["input_tokens"])
            self.assertIsNone(record["usage"]["output_tokens"])
        self.assertIsNone(summarize_calls(records)["usage"]["input_tokens"])

    def test_journal_model_event_is_linked_to_agent_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            with InteractionJournal(Path(directory) / "journal.sqlite3") as journal:
                with journal.start_session("fixture") as session:
                    with operation("agent.fixture", {}):
                        with model_request(provider="qwen", transport="openai", model="model", parameters={}):
                            pass
                    session.finish("succeeded")
                saved = journal.read_session(session.session_id)
        events = saved["events"]
        started = next(event for event in events if event["kind"] == "operation_started")
        model = next(event for event in events if event["kind"] == "model_call")
        self.assertEqual(model["data"]["operation_id"], started["data"]["operation_id"])
        self.assertEqual(model["data"]["operation_name"], "agent.fixture")

    def test_mock_http_transport_has_one_attempt_per_failed_sdk_request(self):
        # Exercise the real SDK and its retry setting without opening a socket.
        attempts = []

        def fail(request):
            attempts.append(request)
            return httpx.Response(503, json={"error": {"message": "fixture unavailable"}})

        from openai import OpenAI
        client = OpenAI(api_key="fixture-key", base_url="http://fixture.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(fail)))
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI", return_value=client):
            with capture_model_calls() as calls:
                with self.assertRaises(RuntimeError):
                    Chat().send_message([])
        client.close()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(calls), len(attempts))
        self.assertEqual(calls[0]["exception_type"], "InternalServerError")

    def test_real_workflow_captures_worker_request_and_preserves_confirmation(self):
        review = {"verdict": "reasonable", "risk": "low", "summary": "Print fixture.",
                  "effects": ["Prints fixture."], "issues": [], "uncertainties": [],
                  "recommendation": "Proceed if intended."}
        executed = []
        services = WorkflowServices(
            clarify=lambda *_: {"is_jump": False, "requirement_summary": "Print fixture."},
            parse_intent=lambda _: Chat().temp_chat("plan"),
            generate_commands=lambda _: '{"Commands": ["echo fixture"]}',
            review_commands=lambda _: review,
            correct_error=lambda *_: "corrected fixture",
            execute_commands=lambda commands: executed.append(commands),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("chat.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = response({"prompt_tokens": 2, "completion_tokens": 1})
            log_file = Path(directory) / "journal.sqlite3"
            with capture_model_calls() as calls:
                state = process("Print fixture", services=services, input_fn=lambda _: "n",
                                output_fn=lambda _: None, log_file=log_file)
            with InteractionJournal(log_file, readonly=True) as journal:
                saved = journal.read_session(state["log_session_id"])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(executed, [])
        self.assertEqual(len(calls), 1)
        event = next(event for event in saved["events"] if event["kind"] == "model_call")
        self.assertEqual(event["data"]["call_id"], calls[0]["call_id"])
        self.assertEqual(event["data"]["operation_name"], "agent.parse_intent")

    def test_journal_failure_does_not_replay_completed_request_as_fallback(self):
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture-key"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True), \
                patch("chat.OpenAI") as qwen, patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.return_value = response()
            with InteractionJournal(Path(directory) / "journal.sqlite3") as journal:
                with journal.start_session("fixture") as session:
                    original_record = session.record

                    def fail_model_event(kind, data, **kwargs):
                        if kind == "model_call":
                            raise InteractionLogError("fixture journal unavailable")
                        return original_record(kind, data, **kwargs)

                    with patch.object(session, "record", side_effect=fail_model_event), capture_model_calls() as calls:
                        with self.assertRaises(InteractionLogError):
                            Chat().send_message([])
                    session.finish("failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "succeeded")
        azure.return_value.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
