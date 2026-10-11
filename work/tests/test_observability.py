"""Privacy, trace propagation, and bounded optional-export failure tests."""

from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import Chat
from content_search.config import Settings
from content_search.models import BGEEmbedder, QwenAnalyzer
from interaction_log import InteractionJournal, operation
from model_budget import BudgetExceeded, budget_context
from model_telemetry import capture_model_calls, summarize_calls
from observability.events import capture_events, current_trace, emit_event, trace_context, use_exporter
from observability.langfuse_exporter import LangfuseExporter
from observability.redaction import external_event, redact
from prompts.registry import bind_prompt


def response(content="fixture", usage=None):
    return SimpleNamespace(id="chatcmpl-fixture", usage=usage, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content))])


def client():
    sdk = Mock()
    sdk.start_observation.side_effect = lambda **kwargs: Mock(id="1234567890abcdef")
    return sdk


class ObservabilityTests(unittest.TestCase):
    def test_local_capture_is_json_safe_for_paths_bytes_cycles_and_opaque_values(self):
        class PrivateOpaque:
            def __repr__(self):
                raise AssertionError("CANARY_REPR must never run")
        cyclic = []
        cyclic.append(cyclic)
        payload = {"path": Path("fixture.txt"), "windows": PureWindowsPath("C:/private/fixture.txt"),
                   "tuple": (1, Path("nested.txt")), "bytes": b"CANARY_BYTES",
                   "opaque": PrivateOpaque(), "cycle": cyclic, "nonfinite": float("nan")}
        with capture_events() as events:
            emitted = emit_event("fixture", payload)
        encoded = json.dumps(events, allow_nan=False)
        self.assertEqual(events[0]["data"]["path"], "fixture.txt")
        self.assertEqual(events[0]["data"]["windows"], "C:\\private\\fixture.txt")
        self.assertEqual(events[0]["data"]["tuple"], [1, "nested.txt"])
        self.assertEqual(events[0]["data"]["bytes"]["size"], 12)
        self.assertEqual(events[0]["data"]["opaque"], {"opaque_type": "PrivateOpaque"})
        self.assertNotIn("CANARY", encoded)
        self.assertIsInstance(payload["path"], Path)
        self.assertEqual(emitted, events[0])

    def test_trace_context_links_session_nested_operation_and_physical_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("chat.OpenAI") as qwen, capture_events() as events, capture_model_calls() as calls:
            qwen.return_value.chat.completions.create.return_value = response(usage={"prompt_tokens": 4, "completion_tokens": 2})
            with trace_context(experiment_id="experiment", episode_id="episode", attempt_id="attempt") as root:
                with InteractionJournal(Path(directory) / "journal.sqlite3") as journal:
                    with journal.start_session("private request") as session:
                        with operation("agent.outer", {}):
                            outer = current_trace()["operation_id"]
                            with operation("agent.inner", {}):
                                Chat(bind_prompt("test.trace", "private system")).send_message([])
                        session.finish("succeeded")
                    saved = journal.read_session(session.session_id)
        call = calls[0]
        self.assertEqual(call["trace"]["experiment_id"], "experiment")
        self.assertEqual(call["trace"]["episode_id"], "episode")
        self.assertEqual(call["trace"]["attempt_id"], "attempt")
        self.assertEqual(call["trace"]["parent_operation_id"], outer)
        self.assertEqual(call["trace"]["call_id"], call["call_id"])
        self.assertEqual(call["trace"]["session_id"], session.session_id)
        self.assertEqual(call["finish_reason"], "stop")
        self.assertEqual(call["provider_call_id"], "chatcmpl-fixture")
        self.assertEqual(call["prompt_templates"][0]["template_id"], "test.trace")
        self.assertEqual({event["trace"]["trace_id"] for event in events}, {root["trace_id"]})
        self.assertEqual({event["data"]["trace"]["trace_id"] for event in saved["events"]}, {root["trace_id"]})
        local_request = next(event for event in saved["events"] if event["kind"] == "user_input")
        self.assertEqual(local_request["data"]["text"], "private request")
        self.assertIsNone(current_trace()["session_id"])

    def test_journal_without_experiment_has_one_stable_session_trace(self):
        with tempfile.TemporaryDirectory() as directory, capture_events() as events:
            with InteractionJournal(Path(directory) / "journal.sqlite3") as journal:
                with journal.start_session("fixture") as session:
                    session.output("fixture")
                    session.finish("succeeded")
                saved = journal.read_session(session.session_id)
        self.assertEqual({event["trace"]["trace_id"] for event in events}, {session.session_id})
        self.assertEqual({event["data"]["trace"]["trace_id"] for event in saved["events"]}, {session.session_id})

    def test_trace_ids_are_reproducible_across_explicit_worker_context_and_thread_copy(self):
        with trace_context(experiment_id="experiment", episode_id="episode", attempt_id="attempt") as first:
            with capture_events() as events, ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copy_context().run, emit_event, "fixture", {}).result()
        with trace_context(experiment_id="experiment", episode_id="episode", attempt_id="attempt") as second:
            pass
        self.assertEqual(first["trace_id"], second["trace_id"])
        self.assertEqual(events[0]["trace"]["trace_id"], first["trace_id"])

    def test_recursive_secret_auth_paths_base64_and_reasoning_are_redacted(self):
        canary = "CANARY_CREDENTIAL_very_private"
        payload = {"headers": {"Authorization": "Bearer " + canary, "X-API-Key": canary},
                   "nested": [{"apiKey": canary}, {"password": canary}],
                   "error": "failed " + canary + " https://alice:password@host/path?token=secret",
                   "diagnostic": "at C:\\Users\\Alice\\private file.txt\n/home/alice/private.txt",
                   "image": "data:image/jpeg;base64," + "a" * 200,
                   "document": {"text": "CANARY_DOCUMENT"}, "prompt": "CANARY_PROMPT",
                   "reasoning_content": "CANARY_REASONING", "analysis": "<think>CANARY_THINK</think>",
                   "opaque": object(), "digest": "a" * 64}
        result = redact(payload)
        serialized = json.dumps(result)
        for secret in (canary, "alice", "Alice", "private file", "CANARY_DOCUMENT", "CANARY_PROMPT", "CANARY_REASONING", "CANARY_THINK"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(result["digest"], "a" * 64)
        self.assertEqual(payload["prompt"], "CANARY_PROMPT")
        cyclic = {}
        cyclic["self"] = cyclic
        self.assertIn("[redacted]", json.dumps(redact(cyclic)))

    def test_external_projection_drops_freeform_even_without_secret_patterns(self):
        canary = "CANARY_PRIVATE_FREEFORM"
        event = {"kind": "model_call", "role": "model", "metadata": {"freeform": canary},
                 "trace": {"experiment_id": "experiment"},
                 "data": {"metadata": {"freeform": canary}, "prompt": canary, "output": canary,
                          "image": {"base64": canary}, "reasoning": canary,
                          "usage": {"input_tokens": 0, "output_tokens": None},
                          "model": "qwen3:4b", "status": "succeeded", "call_type": "chat",
                          "prompt_templates": [{"template_id": "fixture", "version": "1", "text": canary}]}}
        safe = external_event(event)
        self.assertNotIn(canary, json.dumps(safe))
        self.assertNotIn("metadata", safe)
        self.assertEqual(safe["data"]["usage"]["input_tokens"], 0)
        self.assertIsNone(safe["data"]["usage"]["output_tokens"])

    def test_export_is_disabled_by_default_without_initializing_sdk(self):
        sdk = client()
        exporter = LangfuseExporter(client=sdk)
        self.assertFalse(exporter.submit({"kind": "fixture", "data": {}}))
        self.assertTrue(exporter.flush(0))
        sdk.start_observation.assert_not_called()
        sdk.flush.assert_not_called()

    def test_export_thread_start_failure_degrades_without_business_exception(self):
        sdk = client()
        with patch("observability.langfuse_exporter.threading.Thread.start", side_effect=RuntimeError("CANARY_RESOURCE")):
            exporter = LangfuseExporter(enabled=True, client=sdk)
        self.assertFalse(exporter.submit({"kind": "fixture", "data": {}}))
        self.assertFalse(exporter.flush(0))
        self.assertFalse(exporter.close(0))
        self.assertEqual(exporter.stats()["last_error_type"], "RuntimeError")
        self.assertNotIn("CANARY", json.dumps(exporter.stats()))
        sdk.start_observation.assert_not_called()

    def test_invalid_close_deadline_is_nonblocking_and_does_not_escape(self):
        for value in (None, "bad", True, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                exporter = LangfuseExporter()
                self.assertFalse(exporter.close(value))

    def test_export_failure_does_not_replay_model_or_trigger_cloud_fallback(self):
        sdk = client()
        sdk.start_observation.side_effect = RuntimeError("CANARY_EXPORT_SECRET")
        exporter = LangfuseExporter(enabled=True, client=sdk)
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure, use_exporter(exporter), capture_model_calls() as calls:
            qwen.return_value.chat.completions.create.return_value = response()
            answer = Chat().send_message([])
        self.assertEqual(answer.choices[0].message.content, "fixture")
        self.assertFalse(exporter.flush(1))
        exporter.close(1)
        self.assertEqual(len(calls), 1)
        qwen.return_value.chat.completions.create.assert_called_once()
        azure.return_value.chat.completions.create.assert_not_called()
        self.assertEqual(exporter.stats()["last_error_type"], "RuntimeError")
        self.assertNotIn("CANARY_EXPORT_SECRET", json.dumps(exporter.stats()))

    def test_actual_sdk_adapter_kwargs_contain_only_projected_metadata(self):
        sdk = client()
        exporter = LangfuseExporter(enabled=True, client=sdk)
        with trace_context(experiment_id="experiment", episode_id="episode", attempt_id="attempt"), use_exporter(exporter):
            emit_event("model_call", {"model": "qwen3:4b", "call_type": "chat", "status": "succeeded",
                                      "usage": {"input_tokens": 3, "output_tokens": None},
                                      "metadata": {"freeform": "CANARY_PRIVATE"}, "prompt": "CANARY_PROMPT"})
        self.assertTrue(exporter.flush(1))
        exporter.close(1)
        options = sdk.start_observation.call_args.kwargs
        self.assertEqual(options["as_type"], "generation")
        self.assertEqual(options["usage_details"], {"input": 3})
        self.assertNotIn("input", options)
        self.assertNotIn("output", options)
        self.assertNotIn("CANARY", json.dumps(options))
        self.assertEqual(len(options["trace_context"]["trace_id"]), 32)

    def test_export_usage_excludes_cache_and_reasoning_from_base_buckets(self):
        sdk = client()
        exporter = LangfuseExporter(enabled=True, client=sdk)
        exporter.submit({"kind": "model_call", "data": {"call_type": "chat", "usage": {
            "input_tokens": 20, "output_tokens": 10, "cache_read_tokens": 5,
            "cache_write_tokens": None, "reasoning_tokens": 4}}})
        self.assertTrue(exporter.flush(1))
        exporter.close(1)
        usage = sdk.start_observation.call_args.kwargs["usage_details"]
        self.assertEqual(usage, {"input": 15, "input_cached_tokens": 5, "output": 6,
                                 "output_reasoning_tokens": 4, "total": 30})
        self.assertEqual(sum(value for key, value in usage.items() if key != "total"), 30)

    def test_queue_is_bounded_and_flush_returns_at_deadline(self):
        entered, release = threading.Event(), threading.Event()
        sdk = client()

        def block(**kwargs):
            entered.set()
            release.wait(2)
            return Mock(id="1234567890abcdef")
        sdk.start_observation.side_effect = block
        exporter = LangfuseExporter(enabled=True, client=sdk, max_queue=1)
        self.assertTrue(exporter.submit({"kind": "fixture", "data": {}}))
        self.assertTrue(entered.wait(1))
        self.assertTrue(exporter.submit({"kind": "fixture", "data": {}}))
        self.assertFalse(exporter.submit({"kind": "fixture", "data": {}}))
        started = time.monotonic()
        self.assertFalse(exporter.flush(0.01))
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertEqual(exporter.stats()["dropped"], 1)
        release.set()
        self.assertTrue(exporter.flush(1))
        exporter.close(1)

    def test_sdk_flush_has_no_invented_timeout_argument_and_one_blocked_thread(self):
        release = threading.Event()
        sdk = client()
        sdk.flush.side_effect = lambda: release.wait(2)
        exporter = LangfuseExporter(enabled=True, client=sdk)
        exporter.submit({"kind": "fixture", "data": {}})
        self.assertFalse(exporter.flush(0.01))
        self.assertFalse(exporter.flush(0.01))
        self.assertEqual(sdk.flush.call_count, 1)
        self.assertEqual(sdk.flush.call_args.kwargs, {})
        release.set()
        exporter.close(1)

    def test_content_analysis_qa_and_embeddings_are_distinct_physical_boundaries(self):
        annotation = '{"summary": "fixture", "description": "", "ocr_text": "", "tags": []}'
        settings = Settings(embedding_backend="ollama", embedding_model="bge-m3")
        with patch("content_search.models.OpenAI") as qwen, patch("content_search.models.httpx.Client") as http:
            qwen.return_value.chat.completions.create.side_effect = [response(annotation, {"prompt_tokens": 4, "completion_tokens": 2}), response("answer", None)]
            http.return_value.post.return_value.json.return_value = {"embeddings": [[1.0] * 1024], "prompt_eval_count": 7}
            with capture_model_calls() as calls:
                analyzer = QwenAnalyzer(settings)
                analyzer.annotate(text="CANARY_DOCUMENT")
                analyzer.answer("fixture", [{"path": "/private/doc.txt", "kind": "text", "text": "CANARY_DOCUMENT"}])
                BGEEmbedder(settings).encode(["CANARY_DOCUMENT"])
        self.assertEqual([call["call_type"] for call in calls], ["content_analysis", "content_qa", "embedding"])
        self.assertEqual(qwen.call_args.kwargs["max_retries"], 0)
        self.assertEqual(calls[2]["usage"]["input_tokens"], 7)
        self.assertIsNone(calls[2]["usage"]["output_tokens"])
        self.assertEqual(summarize_calls(calls)["physical_llm_calls"], 2)
        self.assertEqual(summarize_calls(calls)["physical_embedding_calls"], 1)
        self.assertNotIn("CANARY_DOCUMENT", json.dumps(calls))
        self.assertNotIn("/private/doc.txt", json.dumps(calls))

    def test_local_embedding_is_not_a_physical_api_or_llm_request(self):
        with patch("content_search.models.httpx.Client"):
            embedder = BGEEmbedder(Settings())
            embedder.model = Mock()
            dense = SimpleNamespace(tolist=lambda: [1.0] * 1024)
            embedder.model.encode.return_value = {"dense_vecs": [dense], "lexical_weights": [{"1": 0.5}]}
            with capture_model_calls() as calls:
                embedder.encode(["fixture"])
        summary = summarize_calls(calls)
        self.assertEqual(summary["physical_llm_calls"], 0)
        self.assertEqual(summary["physical_embedding_calls"], 0)
        self.assertEqual(summary["local_embedding_calls"], 1)
        self.assertEqual(calls[0]["call_type"], "embedding")
        self.assertFalse(calls[0]["physical_request"])
        self.assertIsNone(calls[0]["usage"]["input_tokens"])

    def test_budget_rejection_never_becomes_cloud_fallback(self):
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure, capture_model_calls() as calls:
            qwen.return_value.chat.completions.create.return_value = response(usage={"prompt_tokens": 2, "completion_tokens": 1})
            chat = Chat()
            with budget_context(max_calls=1, max_seconds=10) as budget:
                chat.send_message([])
                with self.assertRaises(BudgetExceeded):
                    chat.send_message([])
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.calls, 1)
        self.assertEqual(budget.reported_tokens, 3)
        azure.return_value.chat.completions.create.assert_not_called()

    def test_local_only_budget_rejects_content_and_embedding_before_network(self):
        with patch("content_search.models.OpenAI") as qwen, patch("content_search.models.httpx.Client") as http, \
                capture_model_calls() as calls, budget_context(max_calls=1, max_seconds=10) as budget:
            analyzer = QwenAnalyzer(Settings())
            with self.assertRaises(BudgetExceeded):
                analyzer.annotate(text="fixture")
            embedder = BGEEmbedder(Settings(embedding_backend="ollama", embedding_model="bge-m3"))
            with self.assertRaises(BudgetExceeded):
                embedder.encode(["fixture"])
        qwen.return_value.chat.completions.create.assert_not_called()
        http.return_value.post.assert_not_called()
        self.assertEqual(calls, [])
        self.assertEqual(budget.calls, 0)

    def test_evaluation_output_cap_is_explicit_and_normal_defaults_are_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI"):
            self.assertEqual(Chat().max_tokens, 4096)
        with patch.dict(os.environ, {"QWEN_EVAL_MAX_TOKENS": "2048"}, clear=True), patch("chat.OpenAI"):
            self.assertEqual(Chat().max_tokens, 2048)
            self.assertEqual(Chat(max_tokens=512).max_tokens, 512)
        for value in ("0", "-1", "no", "1.5", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"QWEN_EVAL_MAX_TOKENS": value}, clear=True), patch("chat.OpenAI"):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    Chat()


if __name__ == "__main__":
    unittest.main()
