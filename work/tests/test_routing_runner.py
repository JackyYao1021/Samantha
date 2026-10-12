"""Admission, no-gold fallback, trace transport, and frozen-evidence contracts."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.routing_client import EvidenceWriteError, OllamaClient, NoRedirect, strict_model_json
from evals.routing_runner import load_config, solve_case, _messages, SOLVER_PROMPT
from evals.routing_tasks import load_cases
from model_budget import BudgetExceeded, budget_context


def sample_payload():
    return {"status": "completed", "actions": [], "answer": [], "question": ""}


class FakeClient:
    def __init__(self, values):
        self.values, self.calls, self.messages = list(values), [], []

    def chat(self, messages, schema, *, model, **kwargs):
        self.messages.append(deepcopy(messages))
        value = self.values.pop(0)
        if isinstance(value, (BudgetExceeded, EvidenceWriteError)):
            raise value
        self.calls.append({"model": model, "call_type": "chat", "duration_ms": 2.,
                           "status": "succeeded", "usage": {"input_tokens": 10, "output_tokens": 5},
                           "provider_compute_ms": 1.})
        if isinstance(value, Exception):
            raise value
        return value


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(WORK / "evals" / "experiments" / "routing_local.json")
        self.case = next(case for case in load_cases(WORK / "evals" / "datasets" / "routing_tasks_v1.jsonl")
                         if case["split"] == "development" and case["family"] == "write")
        self.selection = {"economical_model": self.config["models"][0],
                          "family_models": {family: self.config["models"][0]
                            for family in ("write", "rename", "append", "select", "multi_step", "clarify")}}

    def test_semantically_wrong_valid_output_does_not_trigger_oracle_fallback(self):
        client = FakeClient([sample_payload()])
        result, stopped = solve_case(self.case, "rule", 1, client=client,
                                    config=self.config, selection=self.selection)
        self.assertFalse(stopped)
        self.assertTrue(result["complete"])
        self.assertEqual(result["task_success"], 0)
        self.assertFalse(result["fallback"])
        self.assertEqual(len(client.calls), 1)

    def test_invalid_output_contract_escalates_once_to_other_model(self):
        client = FakeClient([{"unexpected": True}, sample_payload()])
        result, stopped = solve_case(self.case, "rule", 1, client=client,
                                    config=self.config, selection=self.selection)
        self.assertFalse(stopped)
        self.assertTrue(result["fallback"])
        self.assertNotEqual(result["initial_model"], result["selected_model"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["input_tokens"], 20)

    def test_fixed_model_control_never_escalates(self):
        client = FakeClient([ValueError("bad output")])
        result, stopped = solve_case(self.case, "fixed_a", 1, client=client, config=self.config)
        self.assertFalse(stopped)
        self.assertFalse(result["fallback"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["task_success"], 0)

    def test_persistence_failure_cannot_trigger_model_replay(self):
        client = FakeClient([EvidenceWriteError("evidence unavailable")])
        result, stopped = solve_case(self.case, "rule", 1, client=client,
                                    config=self.config, selection=self.selection)
        self.assertTrue(stopped)
        self.assertFalse(result["complete"])
        self.assertFalse(result["fallback"])
        self.assertEqual(result["error_category"], "evidence_persistence")
        self.assertEqual(len(client.messages), 1)

    def test_budget_stop_is_incomplete_and_does_not_retry(self):
        client = FakeClient([BudgetExceeded("exhausted")])
        result, stopped = solve_case(self.case, "rule", 1, client=client,
                                    config=self.config, selection=self.selection)
        self.assertTrue(stopped)
        self.assertFalse(result["complete"])
        self.assertEqual(result["error_category"], "budget_exhausted")
        self.assertFalse(result["fallback"])

    def test_model_router_sees_only_public_input_and_family_does_not_score_task(self):
        client = FakeClient([{"family": "append"}, sample_payload()])
        result, _ = solve_case(self.case, "model_driven", 1, client=client,
                               config=self.config, selection=self.selection)
        sent = json.loads(client.messages[0][1]["content"])
        self.assertEqual(set(sent), {"request", "files", "directories"})
        self.assertNotIn("expected", sent)
        self.assertNotIn("family", sent)
        self.assertEqual(result["classification_family"], "append")
        self.assertEqual(result["task_success"], 0)
        self.assertEqual(len(client.calls), 2)

    def test_missing_embedding_prototypes_cannot_score_a_default_model(self):
        client = FakeClient([])
        result, stopped = solve_case(self.case, "embedding", 1, client=client,
                                    config=self.config, selection=self.selection, prototypes=[])
        self.assertFalse(stopped)
        self.assertFalse(result["complete"])
        self.assertEqual(result["task_success"], 0)
        self.assertEqual(result["error_category"], "unsupported_router")
        self.assertEqual(client.calls, [])

    def test_gold_change_cannot_change_generation_messages(self):
        from evals.routing_tasks import public_input
        altered = deepcopy(self.case)
        altered["expected"]["files"] = {"hidden-answer": "secret"}
        self.assertEqual(_messages(public_input(self.case), SOLVER_PROMPT),
                         _messages(public_input(altered), SOLVER_PROMPT))

    def test_embedding_setup_persistence_failure_stops_heldout_requests(self):
        from evals.routing_runner import run

        class SetupFailingClient(FakeClient):
            def embed(self, texts, *, model):
                raise EvidenceWriteError("setup evidence unavailable")

        client = SetupFailingClient([sample_payload(), sample_payload()])
        identities = {name: {"name": name, "digest": "fixture"}
                      for name in [*self.config["models"], self.config["embedding_model"]]}
        with tempfile.TemporaryDirectory() as temp:
            with patch("evals.routing_runner.read_inventory", return_value=identities), \
                    patch("evals.routing_runner.OllamaClient", return_value=client):
                result = run(WORK / "evals/datasets/routing_tasks_v1.jsonl",
                             WORK / "evals/experiments/routing_local.json",
                             Path(temp) / "run", allow_local_model=True, case_limit=1)
        self.assertEqual(result["manifest"]["stop_reason"], "evidence_persistence")
        self.assertEqual(len(client.messages), 2)
        self.assertTrue(all(group["started"] == 0 for group in result["summary"]["policies"].values()))


class FakeOpener:
    def __init__(self, response):
        self.response, self.requests = response, []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return io.BytesIO(json.dumps(self.response).encode())


class ClientTests(unittest.TestCase):
    def response(self):
        return {"done": True, "done_reason": "stop", "message": {"content": '{"ok":true}'},
                "prompt_eval_count": 11, "eval_count": 4, "prompt_eval_duration": 1_000_000,
                "eval_duration": 2_000_000, "load_duration": 500_000}

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for value in ('{"status":"clarify","status":"completed"}', '{"value":NaN}', '{"value":Infinity}'):
            with self.assertRaises(ValueError):
                strict_model_json(value)

    def test_actual_request_boundary_saves_usage_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opener = FakeOpener(self.response())
            client = OllamaClient(root / "requests", allowed_models=("qwen3:4b",), opener=opener)
            with budget_context(max_calls=1, max_seconds=10, allowed_models=("qwen3:4b",)):
                value = client.chat([{"role": "user", "content": "public"}], {}, model="qwen3:4b")
            self.assertEqual(value, {"ok": True})
            self.assertEqual(client.calls[0]["usage"]["input_tokens"], 11)
            self.assertEqual(client.calls[0]["provider_compute_ms"], 3)
            self.assertEqual(client.calls[0]["provider_load_ms"], .5)
            request = opener.requests[0][0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            self.assertFalse(json.loads(request.data)["think"])
            self.assertTrue((root / "requests" / "request_0001" / "response.json").is_file())

    def test_provider_error_is_a_physical_request_but_budget_rejection_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            opener = FakeOpener(TimeoutError("provider timed out"))
            client = OllamaClient(Path(temp) / "requests", allowed_models=("qwen3:4b",), opener=opener)
            with budget_context(max_calls=1, max_seconds=10, allowed_models=("qwen3:4b",)):
                with self.assertRaises(TimeoutError):
                    client.chat([], {}, model="qwen3:4b")
                with self.assertRaises(BudgetExceeded):
                    client.chat([], {}, model="qwen3:4b")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0]["status"], "failed")
            self.assertEqual(len(opener.requests), 1)
            journal = (Path(temp) / "requests.jsonl").read_text().splitlines()
            self.assertEqual(len(journal), 2)

    def test_undeclared_model_and_remote_redirect_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            opener = FakeOpener(self.response())
            client = OllamaClient(Path(temp) / "requests", allowed_models=("qwen3:4b",), opener=opener)
            with self.assertRaises(ValueError):
                client.chat([], {}, model="another-model")
            self.assertEqual(opener.requests, [])
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://remote.invalid")

    def test_telemetry_save_failure_does_not_double_count_completed_request(self):
        from evals.harness import write_json
        with tempfile.TemporaryDirectory() as temp:
            opener = FakeOpener(self.response())
            client = OllamaClient(Path(temp) / "requests", allowed_models=("qwen3:4b",), opener=opener)
            failed = False
            def fail_one_save(path, value):
                nonlocal failed
                if path.name == "telemetry.json" and not failed:
                    failed = True
                    raise OSError("disk temporarily unavailable")
                return write_json(path, value)
            with patch("evals.routing_client.write_json", side_effect=fail_one_save):
                with self.assertRaises(EvidenceWriteError):
                    client.chat([], {}, model="qwen3:4b")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(client.calls[0]["usage"]["input_tokens"], 11)
            self.assertEqual(client.calls[0]["evidence_error"], "persistence_failed")

    def test_truncated_output_retains_usage_but_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            response = {**self.response(), "done_reason": "length"}
            client = OllamaClient(Path(temp) / "requests", allowed_models=("qwen3:4b",), opener=FakeOpener(response))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                client.chat([], {}, model="qwen3:4b")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0]["finish_reason"], "length")

    def test_request_directory_failure_is_fatal_before_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            opener = FakeOpener(self.response())
            client = OllamaClient(Path(temp) / "requests", allowed_models=("qwen3:4b",), opener=opener)
            with patch.object(Path, "mkdir", side_effect=PermissionError("read-only evidence directory")):
                with self.assertRaises(EvidenceWriteError):
                    client.chat([], {}, model="qwen3:4b")
            self.assertEqual(opener.requests, [])
            self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
