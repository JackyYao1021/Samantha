"""Admission, bounded faults, and local-only policy; no real model requests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "src"))
from evals.dataset import load_dataset
from evals.harness import ScriptedServices, write_json, snapshot, run_worker, launch_worker
from evals.local_models import local_worker, model_identity
from evals.runner import load_config, run_experiment, run_matrix
from evals.schemas import worker_input
from evals.scorers.task import score_task
from evals.scorers.trajectory import earliest_material_error
from workflow import WorkflowServices


class RuntimeModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.config = load_config(WORK / "evals/experiments/local_ollama.json")

    def config_file(self, config):
        path = self.folder / "config.json"
        write_json(path, config)
        return path

    def test_real_mode_requires_admission_before_inventory_or_artifacts(self):
        with patch("evals.local_models.model_identity") as inventory:
            with self.assertRaisesRegex(ValueError, "admission"):
                run_experiment(WORK / "evals/datasets/local_proposals_v1.jsonl",
                               self.config_file(self.config), self.folder / "run")
        inventory.assert_not_called()
        self.assertFalse((self.folder / "run").exists())

    def test_proposal_rejects_automatic_approval_before_inventory(self):
        with patch("evals.local_models.model_identity") as inventory:
            with self.assertRaisesRegex(ValueError, "deny"):
                run_experiment(WORK / "evals/datasets/smoke_v1.jsonl", self.config_file(self.config),
                               self.folder / "run", allow_local_model=True)
        inventory.assert_not_called()

    def test_cloud_other_models_tracking_and_nonfinite_limits_are_rejected(self):
        for field, value in [("provider", "azure"), ("model", "qwen2.5vl:7b"),
                             ("request_timeout_seconds", float("inf")), ("max_output_tokens", True)]:
            with self.subTest(field=field):
                config = deepcopy(self.config)
                config["model"][field] = value
                with self.assertRaises(ValueError):
                    load_config(self.config_file(config))
        config = deepcopy(self.config)
        config["runtime"]["azure_fallback"] = True
        with self.assertRaises(ValueError):
            load_config(self.config_file(config))

    def test_matrix_global_budget_is_checked_before_output_and_calls(self):
        paths = [WORK / "evals/experiments/local_ollama.json", WORK / "evals/experiments/local_openai.json"]
        with patch("evals.runner.run_experiment") as execute:
            with self.assertRaisesRegex(ValueError, "reservations"):
                run_matrix(WORK / "evals/datasets/local_proposals_v1.jsonl", paths,
                           self.folder / "run", max_model_calls=79, allow_local_model=True)
        execute.assert_not_called()
        self.assertFalse((self.folder / "run").exists())

    def test_matrix_rejects_nonfinite_and_bool_global_limits(self):
        paths = [WORK / "evals/experiments/baseline.json"] * 2
        for fields in ({"max_seconds": float("nan")}, {"max_seconds": True}, {"max_model_calls": True}):
            with self.assertRaises(ValueError):
                run_matrix(WORK / "evals/datasets/smoke_v1.jsonl", paths, self.folder / "run", **fields)

    def test_socket_gate_permits_only_numeric_loopback_ollama(self):
        class Socket:
            def connect(self, address):
                return "allowed"
        with patch("evals.local_models.socket.socket", Socket), local_worker(self.config):
            import socket
            sock = socket.socket()
            self.assertEqual(sock.connect(("127.0.0.1", 11434)), "allowed")
            for address in [("127.0.0.1", 443), ("8.8.8.8", 11434), ("example.invalid", 11434)]:
                with self.assertRaises((RuntimeError, ValueError)):
                    sock.connect(address)

    def test_model_inventory_rejects_redirects_without_fetching_target(self):
        def opener(*handlers):
            redirect = next(handler for handler in handlers if hasattr(handler, "redirect_request"))
            redirect.redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")
        with patch("evals.local_models.build_opener", side_effect=opener):
            with self.assertRaisesRegex(ValueError, "redirect"):
                model_identity()

    def fake(self, fault, *, variant="faithful", actions=None):
        root = self.folder / "fixture"
        root.mkdir()
        commands = ["bounded fixture"]
        data = {"request": "fixture", "interaction": {"approval_policy": "fixture_allowlist", "clarification_answers": [],
                "allowed_command_batches": [commands]}, "faults": [fault], "fake_services": {"executions": [
                {"commands": commands, "actions": actions or [{"kind": "append", "path": "log.txt", "content_utf8": "once"}]}]}}
        fake = ScriptedServices(data, root, variant)
        fake.last_review = {"commands": commands, "cwd": str(root), "shell": "/bin/bash"}
        self.assertEqual(fake.answer("Proceed? (y/n): "), "y")
        return fake, root, commands

    def test_partial_effects_safe_stop_preserves_exact_bytes_and_blocks_replay(self):
        fake, root, commands = self.fake({"kind": "execution_failure", "call_index": 0, "after_actions": 1},
                                         variant="observed_safe_stop")
        result = fake.execute_commands(commands)
        self.assertFalse(result["success"])
        self.assertEqual(len(result["completed_effect_ids"]), 1)
        self.assertEqual((root / "log.txt").read_bytes(), b"once")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            fake.correct_error("request", commands, result["output"])
        self.assertEqual(fake.tool_calls, 1)

    def test_fault_before_actions_preserves_empty_effect_identity(self):
        fake, root, commands = self.fake({"kind": "execution_failure", "call_index": 0})
        result = fake.execute_commands(commands)
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_effect_ids"], [])
        self.assertEqual(snapshot(root)["entries"], {})

    def test_entire_fake_batch_is_validated_before_any_side_effect(self):
        fake, root, commands = self.fake({"kind": "execution_failure", "call_index": 0, "after_actions": 1},
            actions=[{"kind": "write", "path": "first.txt", "content_utf8": "unsafe if applied"},
                     {"kind": "write", "path": "../outside.txt", "content_utf8": "bad"}])
        with self.assertRaises(ValueError):
            fake.execute_commands(commands)
        self.assertEqual(snapshot(root)["entries"], {})

    def test_injected_timeout_is_a_signal_without_side_effect_or_sleep(self):
        fake, root, commands = self.fake({"kind": "timeout", "call_index": 0})
        with self.assertRaises(TimeoutError):
            fake.execute_commands(commands)
        self.assertEqual(snapshot(root)["entries"], {})

    def test_proposal_denial_real_workflow_never_invokes_an_executor(self):
        case = load_dataset(WORK / "evals/datasets/local_proposals_v1.jsonl")[0]
        commands = ["this text is never executed"]
        def forbidden(commands):
            self.fail("Generated commands reached an executor")
        services = WorkflowServices(lambda *args: {"is_jump": False, "requirement_summary": case["input"]["request"]},
            lambda text: text, lambda text: json.dumps({"Explanation": "Unit fixture", "Commands": commands}),
            lambda request: {"verdict": "reasonable", "risk": "low", "summary": "reviewed", "effects": ["Would change a fixture"],
                             "issues": [], "uncertainties": [], "recommendation": "ask approval"}, lambda *args: "stop", forbidden)
        artifacts = self.folder / "artifacts"
        artifacts.mkdir()
        previous = Path.cwd()
        self.addCleanup(lambda: __import__("os").chdir(previous))
        with patch.dict(os.environ, {}, clear=False), patch("workflow.default_services", return_value=services):
            run_worker({"schema_version": "1.0", "input": worker_input(case), "worker_settings": self.config}, artifacts, "faithful")
        evidence = json.loads((artifacts / "evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["runtime_status"], "cancelled", evidence["errors"])
        self.assertEqual(evidence["telemetry"]["tool_calls"], 0)
        self.assertEqual(score_task(case, evidence)["evaluation_status"], "passed")
        self.assertEqual(evidence["llm_calls"], [])  # This is a mocked policy test, never empirical model evidence.

    def test_crashed_persistence_fault_preserves_durable_partial_facts(self):
        case = next(item for item in load_dataset(WORK / "evals/datasets/edge_adversarial_v1.jsonl")
                    if item["case_id"] == "fault_persistence_error")
        evidence = launch_worker(worker_input(case), self.folder / "artifacts")
        self.assertFalse(evidence["complete"])
        self.assertEqual(evidence["telemetry"]["physical_llm_calls"], 0)
        self.assertFalse(evidence["telemetry"]["model_call_count_complete"])
        self.assertEqual(earliest_material_error(evidence)["category"], "Persistence")
        self.assertEqual(score_task(case, evidence)["evaluation_status"], "incomplete")

    def test_budget_exception_cannot_be_relabelled_as_a_complete_runtime_failure(self):
        case = load_dataset(WORK / "evals/datasets/smoke_v1.jsonl")[0]
        evidence = {"schema_version": "1.0", "complete": True, "runtime_status": "failed", "errors": [],
                    "trajectory": [{"kind": "operation_finished", "data": {"exception_type": "BudgetExceeded"}}]}
        scored = score_task(case, evidence)
        self.assertEqual(scored["evaluation_status"], "incomplete")
        self.assertTrue(any("budget stop" in item["message"] for item in scored["errors"]))


if __name__ == "__main__":
    unittest.main()
