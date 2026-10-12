"""Task-level scoring must reject false success and retain independent repetitions."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "src"))

from evals.task_agent_eval import (ExactApprovalPolicy, ScriptedPlanner, classify_evaluation, grade_evidence,
                                  load_dataset, local_model_metadata, regrade_saved_suite, run_case, run_suite,
                                  score_filesystem, snapshot_tree)


class TaskAgentScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exact_bytes_pass(self):
        (self.root / "note.txt").write_bytes(b"alpha\n")
        self.assertTrue(score_filesystem({"note.txt": "alpha\n"}, snapshot_tree(self.root))["passed"])

    def test_success_claim_does_not_hide_noop(self):
        snapshot = snapshot_tree(self.root)
        snapshot["runtime_status"] = "succeeded"
        score = score_filesystem({"note.txt": "alpha\n"}, snapshot)
        self.assertFalse(score["passed"])
        self.assertEqual(score["missing"], ["note.txt"])

    def test_wrong_content_newline_and_duplicate_append_fail(self):
        for content in (b"wrong\n", b"alpha", b"alpha\r\n", b"alpha\nalpha\n"):
            with self.subTest(content=content):
                (self.root / "note.txt").write_bytes(content)
                self.assertFalse(score_filesystem({"note.txt": "alpha\n"}, snapshot_tree(self.root))["passed"])

    def test_unexpected_file_and_empty_directory_fail(self):
        (self.root / "note.txt").write_bytes(b"alpha\n")
        (self.root / "extra.txt").write_bytes(b"unexpected")
        (self.root / "empty").mkdir()
        score = score_filesystem({"note.txt": "alpha\n"}, snapshot_tree(self.root))
        self.assertFalse(score["passed"])
        self.assertEqual(score["unexpected"], ["empty", "extra.txt"])

    def test_incomplete_snapshot_never_passes(self):
        self.assertFalse(score_filesystem({}, {"complete": False, "entries": {}})["passed"])

    def test_nested_expected_directories_are_scored(self):
        (self.root / "reports").mkdir()
        (self.root / "reports" / "note.txt").write_bytes(b"done\n")
        self.assertTrue(score_filesystem({"reports/note.txt": "done\n"}, snapshot_tree(self.root))["passed"])

    def test_approval_is_exact_and_single_use(self):
        case = load_dataset()["cases"][0]
        policy = ExactApprovalPolicy(case)
        action = deepcopy(case["allowed_actions"][0])
        wrong = deepcopy(action)
        wrong["args"]["content"] = "unapproved"
        self.assertFalse(policy.allows(wrong))
        wrong = deepcopy(action)
        wrong["args"]["path"] = "../outside.txt"
        self.assertFalse(policy.allows(wrong))
        self.assertTrue(policy.allows(action))
        self.assertFalse(policy.allows(action))

    def test_scripted_planner_restores_from_observations(self):
        case = next(c for c in load_dataset()["cases"] if c["id"] == "copy_then_move")
        first = ScriptedPlanner(case)({"observations": [], "clarifications": []}, [])
        restored = ScriptedPlanner(case)({"observations": [{"tool": first["tool"]}], "clarifications": []}, [])
        self.assertEqual(first["tool"], "copy_file")
        self.assertEqual(restored["tool"], "move_file")

    def test_budget_and_infrastructure_are_not_model_quality_failures(self):
        task = {"status": "failed", "error": "Physical model request budget exceeded"}
        self.assertEqual(classify_evaluation(task, {}, False, live=True), "incomplete_budget")
        task = {"status": "failed", "error": "", "model_call_records": [{"status": "failed"}]}
        self.assertEqual(classify_evaluation(task, {}, False, live=True), "incomplete_infrastructure")
        task = {"status": "failed", "error": "Invalid planner decision schema.", "model_call_records": [{"status": "succeeded"}]}
        self.assertEqual(classify_evaluation(task, {}, False, live=True), "completed_fail")
        for error in ("Task active-time budget exhausted.", "Planner context budget exceeded."):
            with self.subTest(error=error):
                self.assertEqual(classify_evaluation({"status": "failed", "error": error}, {}, False, live=True), "incomplete_budget")

    def test_local_model_digest_is_read_without_download(self):
        body = {"models": [{"name": "fixture:4b", "digest": "known-digest", "size": 42, "details": {"quantization_level": "Q4"}}]}
        with patch.dict("os.environ", {"QWEN_BASE_URL": "http://127.0.0.1:11434/v1"}), patch("evals.task_agent_eval.build_opener") as factory:
            factory.return_value.open.return_value = io.BytesIO(json.dumps(body).encode())
            metadata = local_model_metadata("fixture:4b")
            request = factory.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")
            self.assertEqual(request.method, "GET")
            self.assertEqual(metadata["digest"], "known-digest")

    def test_remote_model_metadata_stays_unknown_without_request(self):
        with patch.dict("os.environ", {"QWEN_BASE_URL": "https://example.invalid/v1"}), patch("evals.task_agent_eval.build_opener") as factory:
            self.assertEqual(local_model_metadata("fixture:4b")["status"], "unknown")
            factory.assert_not_called()


class TaskAgentExecutionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = {case["id"]: case for case in load_dataset()["cases"]}

    def test_after_effect_restart_appends_once(self):
        result = run_case(self.cases["append_once"], self.root / "append")
        self.assertTrue(result["task_success"], result)
        self.assertTrue(result["protocol_pass"], result)
        self.assertTrue(result["controls"]["injected_crash"])
        self.assertTrue(result["controls"]["terminal_replay_unchanged"])
        self.assertEqual((self.root / "append/workspace/notes.txt").read_bytes(), b"alpha\nbeta\n")

    def test_repetitions_have_independent_files_and_databases(self):
        case = self.cases["create_note"]
        first = run_case(case, self.root / "r1")
        (self.root / "r1/workspace/notes.txt").write_bytes(b"contaminated")
        second = run_case(case, self.root / "r2")
        self.assertTrue(first["task_success"])
        self.assertTrue(second["task_success"])
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual((self.root / "r2/workspace/notes.txt").read_bytes(), b"alpha\n")
        self.assertEqual((self.root / "r1/workspace/notes.txt").read_bytes(), b"contaminated")

    def test_all_scripted_scenarios_and_manifest(self):
        output = self.root / "suite"
        report = run_suite(output)
        self.assertEqual(report["runs"], 10)
        self.assertEqual(report["full_passes"], 10, [(r["case_id"], r["filesystem_score"], r["protocol_issues"], r["error"]) for r in report["results"]])
        self.assertEqual(report["task_runs"], 7)
        self.assertEqual(report["physical_model_requests"], 0)
        self.assertTrue(all(r["task_success"] is None for r in report["results"] if r["kind"] == "protocol"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("summary.json", manifest["artifact_sha256"])
        self.assertIn("src/task_agent/runtime.py", {p.replace("\\", "/") for p in manifest["code_sha256"]})
        self.assertEqual(manifest["max_physical_requests"], 0)
        self.assertEqual(manifest["planned_runs"], 10)
        self.assertEqual(manifest["started_runs"], 10)
        self.assertEqual(manifest["completed_runs"], 10)
        self.assertTrue(manifest["coverage_complete"])
        self.assertTrue((output / "summary.md").is_file())

    def test_budget_incomplete_keeps_every_planned_case_record(self):
        report = run_suite(self.root / "limited", max_steps=1, max_cases=2)
        self.assertEqual(report["planned_runs"], 2)
        self.assertEqual(report["started_runs"], 2)
        self.assertEqual(report["task_runs"], 2)
        self.assertEqual(report["budget_exhausted_runs"], 2)
        self.assertEqual(report["incomplete_runs"], 2)
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["full_passes"], 0)
        self.assertEqual(report["task_successes"], 0)
        self.assertEqual(report["file_outcome_successes"], 2)
        self.assertTrue(all(r["file_outcome_success"] and not r["task_success"] for r in report["results"]))

    def test_correct_files_but_cancelled_is_not_end_to_end_success(self):
        case = deepcopy(self.cases["create_note"])
        case["decisions"].append({"type": "tool", "tool": "copy_file",
                                  "args": {"source": "notes.txt", "destination": "unapproved.txt"}})
        result = run_case(case, self.root / "cancelled-after-write")
        self.assertEqual(result["runtime_status"], "cancelled")
        self.assertTrue(result["file_outcome_success"])
        self.assertFalse(result["protocol_pass"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["evaluation_status"], "completed_fail")

    def test_success_status_required_even_with_passing_files_and_protocol(self):
        case = self.cases["create_note"]
        folder = self.root / "correct-files"
        result = run_case(case, folder)
        task = json.loads((folder / "trace.json").read_text(encoding="utf-8"))["task"]
        task["status"] = "cancelled"
        scores = grade_evidence(case, task, result["controls"], snapshot_tree(folder / "workspace"), True)
        self.assertTrue(scores["file_outcome_success"])
        self.assertTrue(scores["protocol_pass"])
        self.assertFalse(scores["task_success"])
        self.assertEqual(scores["evaluation_status"], "completed_fail")

    def test_saved_only_regrade_preserves_source_and_uses_saved_snapshot(self):
        source, output = self.root / "source", self.root / "regrade"
        run_suite(source, max_cases=1)
        original_manifest = (source / "manifest.json").read_bytes()
        original_summary = (source / "summary.json").read_bytes()
        (source / "create_note-r001/workspace/notes.txt").write_bytes(b"changed after original run")
        with patch("task_agent.runtime.AgentRuntime", side_effect=AssertionError("Regrade must not construct a runtime")):
            report = regrade_saved_suite(source, output)
        self.assertTrue(report["saved_only_regrade"])
        self.assertEqual(report["new_physical_model_requests"], 0)
        self.assertEqual(report["task_successes"], 1)
        self.assertEqual((source / "manifest.json").read_bytes(), original_manifest)
        self.assertEqual((source / "summary.json").read_bytes(), original_summary)
        self.assertTrue((output / "summary.md").exists())

    def test_regrade_rejects_changed_evidence(self):
        source = self.root / "source"
        run_suite(source, max_cases=1)
        (source / "create_note-r001/after.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash is missing or changed"):
            regrade_saved_suite(source, self.root / "regrade")

    def test_live_planner_receives_no_dataset_answers_or_scripts(self):
        case = self.cases["create_note"]
        contexts = []

        class RecordingPlanner:
            def __call__(self, state, tools):
                contexts.append(deepcopy(state))
                return {"type": "finish", "answer": "No operation performed."}

        with patch("task_agent.runtime.ModelPlanner", RecordingPlanner):
            result = run_case(case, self.root / "live-context", live=True)
        self.assertFalse(result["task_success"])
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["request"], case["request"])
        for forbidden in ("expected_files", "decisions", "allowed_actions", "files", "controls"):
            self.assertNotIn(forbidden, contexts[0])

    def test_model_success_without_write_fails_independent_score(self):
        case = deepcopy(self.cases["create_note"])
        case["decisions"] = [{"type": "finish", "answer": "Created successfully."}]
        result = run_case(case, self.root / "false-success")
        self.assertEqual(result["runtime_status"], "succeeded")
        self.assertFalse(result["task_success"])

    def test_output_folder_is_never_reused(self):
        path = self.root / "existing"
        path.mkdir()
        with self.assertRaises(FileExistsError):
            run_case(self.cases["create_note"], path)


if __name__ == "__main__":
    unittest.main()
