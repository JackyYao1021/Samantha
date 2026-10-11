"""Offline end-to-end checks for isolation, frozen replay, and failure evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "src"))

from command_review import command_action_id
from evals.dataset import load_dataset
from evals.harness import fixture_path, ScriptedServices, snapshot, write_json
from evals.manifest import file_hash, verify_frozen
from evals.report import metric
from evals.runner import regrade, run_experiment
from evals.schemas import worker_input


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class EvaluationIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="samantha-eval-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.cases = load_dataset(WORK / "evals" / "datasets" / "smoke_v1.jsonl")
        cls.dataset = cls.root / "smoke-input.json"
        write_json(cls.dataset, {"schema_version": "1.0", "cases": cls.cases})
        cls.baseline_config = WORK / "evals" / "experiments" / "baseline.json"
        cls.baseline = cls.root / "baseline"
        cls.parent_cwd = Path.cwd()
        cls.launches = []
        actual_popen = subprocess.Popen

        def recorded_popen(*args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            if command[:3] == [sys.executable, "-m", "evals.harness"]:
                cls.launches.append({"cwd": kwargs["cwd"], "env_keys": set(kwargs["env"]),
                                     "argv": command})
            return actual_popen(*args, **kwargs)

        environment = {"SAMANTHA_EVAL_CANARY": "synthetic-parent-only",
                       "QWEN_API_KEY": "synthetic-qwen-key",
                       "AZURE_OPENAI_API_KEY": "synthetic-azure-key",
                       "SAMANTHA_STATE_FILE": str(cls.root / "parent-state-must-not-exist.json")}
        with patch.dict(os.environ, environment), patch("evals.harness.subprocess.Popen", side_effect=recorded_popen):
            cls.summary = run_experiment(cls.dataset, cls.baseline_config, cls.baseline)
        cls.results = read_jsonl(cls.baseline / "results.jsonl")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="samantha-eval-case-", dir=self.root)
        self.addCleanup(temporary.cleanup)
        self.case_root = Path(temporary.name)

    def copied_baseline(self):
        target = self.case_root / "saved-run"
        shutil.copytree(self.baseline, target)
        return target

    def config(self, **changes):
        config = read_json(self.baseline_config)
        config.update(changes)
        path = self.case_root / "config.json"
        write_json(path, config)
        return path

    def small_dataset(self, cases=None):
        path = self.case_root / "cases.json"
        write_json(path, {"schema_version": "1.0", "cases": cases or [self.cases[0]]})
        return path

    def test_ten_real_workflow_smoke_protocols_and_six_task_successes(self):
        self.assertEqual(len(self.results), 10)
        self.assertEqual(self.summary["started_attempts"], 10)
        self.assertEqual(self.summary["planned_attempts"], 10)
        self.assertEqual(self.summary["protocol_passes"], 10)
        self.assertEqual(self.summary["task_successes"], 6)
        self.assertEqual(self.summary["attempted_task_success_rate"], 0.6)
        self.assertEqual(self.summary["coverage"], 1.0)
        self.assertEqual(self.summary["physical_llm_calls"], 0)
        self.assertEqual(len(load_dataset(self.baseline / "frozen" / "dataset.jsonl")), 10)
        self.assertTrue((self.baseline / "summary.md").is_file())
        self.assertTrue(all(result["evaluation_status"] == "passed" for result in self.results))
        for result in self.results:
            evidence = read_json(self.baseline / result["evidence_ref"])
            self.assertTrue(evidence["complete"])
            self.assertEqual(evidence["telemetry"]["physical_llm_calls"], 0)
            self.assertEqual(result["evidence_sha256"], file_hash(self.baseline / result["evidence_ref"]))
            self.assertIsNone(result["usage"]["input_tokens"])
            self.assertIsNone(result["usage"]["api_cost_usd"])

    def test_noop_successful_runtime_fails_file_postconditions(self):
        run = self.case_root / "negative-control"
        summary = run_experiment(self.small_dataset(), self.config(variant="noop"), run)
        result = read_jsonl(run / "results.jsonl")[0]
        self.assertEqual(result["runtime_status"], "succeeded")
        self.assertEqual(result["evaluation_status"], "failed")
        self.assertEqual(metric(result, "task_success"), 0)
        self.assertEqual(summary["protocol_passes"], 0)
        self.assertEqual(summary["task_successes"], 0)
        self.assertTrue(any(check["check"].startswith("file_bytes:") and not check["passed"]
                            for check in result["checks"]))
        evidence = read_json(run / result["evidence_ref"])
        self.assertEqual(evidence["before_snapshot"], evidence["after_snapshot"])
        self.assertEqual(evidence["telemetry"]["tool_calls"], 1)

    def test_rejected_and_cancelled_interactions_never_execute(self):
        cases = {result["case_id"]: result for result in self.results}
        for name in ("S06_approval_denied", "S07_clarification_cancelled"):
            with self.subTest(case_id=name):
                result = cases[name]
                evidence = read_json(self.baseline / result["evidence_ref"])
                self.assertEqual(result["runtime_status"], "cancelled")
                self.assertEqual(result["evaluation_status"], "passed")
                self.assertEqual(metric(result, "task_success"), 0)
                self.assertEqual(evidence["telemetry"]["tool_calls"], 0)
                self.assertEqual(evidence["before_snapshot"], evidence["after_snapshot"])
                self.assertFalse(any(event["kind"] == "operation_started" and
                                     event["data"].get("name") == "commands.execute"
                                     for event in evidence["trajectory"]))

    def test_worker_cwds_are_private_and_parent_canary_environment_is_excluded(self):
        self.assertEqual(Path.cwd(), self.parent_cwd)
        self.assertEqual(len(self.launches), 10)
        self.assertEqual(len({launch["cwd"] for launch in self.launches}), 10)
        for launch in self.launches:
            self.assertTrue(Path(launch["cwd"]).is_relative_to(self.baseline / "attempts"))
            self.assertTrue({"SAMANTHA_EVAL_CANARY", "QWEN_API_KEY", "AZURE_OPENAI_API_KEY",
                             "SAMANTHA_STATE_FILE"}.isdisjoint(launch["env_keys"]))
            self.assertEqual(launch["argv"][:3], [sys.executable, "-m", "evals.harness"])
        self.assertFalse((self.root / "parent-state-must-not-exist.json").exists())
        fixture_roots = []
        for result in self.results:
            artifact = (self.baseline / result["evidence_ref"]).parent
            evidence = read_json(artifact / "evidence.json")
            session = next(event for event in evidence["trajectory"] if event["kind"] == "session_started")
            fixture = artifact / "fixture"
            self.assertEqual(Path(session["data"]["cwd"]), fixture)
            self.assertEqual(read_json(artifact / "state.json")["current_dir"], str(fixture))
            fixture_roots.append(fixture)
            public = read_json(artifact / "input.json")["input"]
            self.assertNotIn("expected_output", public)
            self.assertNotIn("reference", public)
            self.assertNotIn("metadata", public)
        self.assertEqual(len(set(fixture_roots)), 10)

    def test_escape_actions_reject_entire_batch_before_any_effect(self):
        root = self.case_root / "fixture"
        root.mkdir()
        outside = self.case_root / "outside.txt"
        outside.write_bytes(b"outside guard")
        for escape in ("../outside.txt", "/absolute/outside.txt", str(outside), "C:/outside.txt"):
            with self.subTest(escape=escape):
                public = deepcopy(worker_input(self.cases[0]))
                commands = ["fixture batch"]
                public["fake_services"]["executions"] = [{"commands": commands, "actions": [
                    {"kind": "write", "path": "safe.txt", "content_utf8": "should never be written"},
                    {"kind": "write", "path": escape, "content_utf8": "escape"}]}]
                fake = ScriptedServices(public, root, "faithful")
                fake.approved_id = command_action_id(commands, str(root), "/bin/bash")
                with self.assertRaises(ValueError):
                    fake.execute_commands(commands)
                self.assertEqual(fake.scope_rejections, 1)
                self.assertFalse((root / "safe.txt").exists())
                self.assertEqual(outside.read_bytes(), b"outside guard")

    def test_symlink_or_reparse_escape_is_rejected_in_paths_and_snapshots(self):
        root = self.case_root / "fixture"
        root.mkdir()
        outside = self.case_root / "outside"
        outside.mkdir()
        (outside / "guard.txt").write_bytes(b"outside guard")
        link = root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            if os.name != "nt":
                self.skipTest("This host does not permit temporary symlinks: " + type(exc).__name__)
            # Directory junctions need no administrator privilege and exercise
            # the same Windows reparse-point rejection on this host. Both paths
            # are confined to this test's private temporary directory.
            self.assertTrue(link.resolve().is_relative_to(self.case_root.resolve()))
            self.assertTrue(outside.resolve().is_relative_to(self.case_root.resolve()))
            created = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                                     shell=False, capture_output=True, timeout=10)
            if created.returncode:
                self.skipTest("This host permits neither temporary symlinks nor directory junctions")
        with self.assertRaises(ValueError):
            fixture_path(root, "linked/guard.txt")
        with self.assertRaises(ValueError):
            snapshot(root)
        self.assertEqual((outside / "guard.txt").read_bytes(), b"outside guard")

    def test_attempt_budget_keeps_unrun_tasks_in_coverage_denominator(self):
        run = self.case_root / "limited"
        summary = run_experiment(self.dataset, self.config(max_attempts=2, repeats=2), run)
        self.assertEqual(summary["planned_attempts"], 20)
        self.assertEqual(summary["started_attempts"], 2)
        self.assertEqual(summary["verified_attempts"], 2)
        self.assertEqual(summary["coverage"], 0.1)
        self.assertEqual(summary["attempted_task_success_rate"], 1.0)
        self.assertEqual(len(read_jsonl(run / "attempts.jsonl")), 2)
        self.assertEqual(len(read_jsonl(run / "results.jsonl")), 2)
        self.assertEqual(read_json(run / "manifest.json")["stop_reason"], "attempt_budget")

    def test_timeout_is_saved_as_incomplete_with_started_attempt_and_evidence_hash(self):
        run = self.case_root / "timeout"
        summary = run_experiment(self.small_dataset(), self.config(timeout_seconds=0.000001), run)
        result = read_jsonl(run / "results.jsonl")[0]
        evidence = read_json(run / result["evidence_ref"])
        self.assertEqual(summary["planned_attempts"], 1)
        self.assertEqual(summary["started_attempts"], 1)
        self.assertEqual(summary["verified_attempts"], 0)
        self.assertEqual(summary["coverage"], 0)
        self.assertEqual(summary["attempted_task_success_rate"], 0)
        self.assertEqual(result["runtime_status"], "timeout")
        self.assertEqual(result["evaluation_status"], "incomplete")
        self.assertFalse(evidence["complete"])
        self.assertEqual(result["evidence_sha256"], file_hash(run / result["evidence_ref"]))
        self.assertEqual(len(read_jsonl(run / "attempts.jsonl")), 1)
        self.assertTrue((run / "errors.jsonl").is_file())
        self.assertTrue((run / "summary.md").is_file())

    def test_regrade_uses_frozen_dataset_validation_and_scorer_without_execution(self):
        run = self.copied_baseline()
        destination = self.case_root / "regraded"
        original_dataset = self.dataset.read_bytes()
        try:
            self.dataset.write_text("current dataset deliberately invalid", encoding="utf-8")
            with patch("evals.schemas.validate_case", side_effect=AssertionError("current validator used")), \
                    patch("evals.dataset.validate_case", side_effect=AssertionError("current dataset validator used")), \
                    patch("evals.scorers.task.score_task", side_effect=AssertionError("current scorer used")), \
                    patch("evals.runner.launch_worker", side_effect=AssertionError("worker relaunched")), \
                    patch("subprocess.Popen", side_effect=AssertionError("process launched")):
                summary = regrade(run, destination)
        finally:
            self.dataset.write_bytes(original_dataset)
        regraded = read_jsonl(destination / "results.jsonl")
        self.assertEqual(summary["protocol_passes"], 10)
        self.assertEqual(summary["task_successes"], 6)
        original = {result["case_id"]: result for result in self.results}
        self.assertEqual(len(regraded), 10)
        for result in regraded:
            self.assertEqual(result["scores"], original[result["case_id"]]["scores"])
            self.assertEqual(result["checks"], original[result["case_id"]]["checks"])
            self.assertEqual(result["evidence_sha256"], original[result["case_id"]]["evidence_sha256"])

    def test_saved_evidence_tampering_is_rejected_before_regrade(self):
        run = self.copied_baseline()
        evidence = run / self.results[0]["evidence_ref"]
        evidence.write_bytes(evidence.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "evidence integrity mismatch"):
            regrade(run, self.case_root / "tampered-regrade")
        self.assertFalse((self.case_root / "tampered-regrade").exists())

    def test_frozen_dataset_source_and_config_tampering_are_rejected(self):
        for target in ("frozen/dataset.jsonl", "frozen/sources/work/evals/scorers/task.py", "frozen/config.json"):
            with self.subTest(target=target):
                run = self.case_root / target.replace("/", "_")
                shutil.copytree(self.baseline, run)
                path = run / target
                if target.endswith("config.json"):
                    config = read_json(path)
                    config["max_attempts"] += 1
                    write_json(path, config)
                else:
                    path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    verify_frozen(run, read_json(run / "manifest.json"))

    def test_existing_run_and_regrade_destinations_are_never_overwritten(self):
        manifest_before = file_hash(self.baseline / "manifest.json")
        results_before = file_hash(self.baseline / "results.jsonl")
        with self.assertRaises(FileExistsError):
            run_experiment(self.dataset, self.baseline_config, self.baseline)
        self.assertEqual(file_hash(self.baseline / "manifest.json"), manifest_before)
        self.assertEqual(file_hash(self.baseline / "results.jsonl"), results_before)
        destination = self.case_root / "existing-destination"
        destination.mkdir()
        guard = destination / "guard.txt"
        guard.write_bytes(b"must remain")
        with self.assertRaises(FileExistsError):
            regrade(self.baseline, destination)
        self.assertEqual(guard.read_bytes(), b"must remain")
        self.assertEqual(set(destination.iterdir()), {guard})


if __name__ == "__main__":
    unittest.main()
