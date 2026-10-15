"""Frozen split, independent scoring and bounded A/B/C harness regression checks."""

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

from evals.intent_eval import (DATASET, FREEZE, PROTOCOL, DeadlineCall, classifier_records,
    filesystem_score, grade_case, load_dataset, local_configuration, make_plan, run_case, run_suite,
    snapshot_tree, ScriptedClassifier, ScriptedPlanner, regrade_saved_suite)


class IntentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.cases = {c["id"]: c for c in cls.data["cases"]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_frozen_annotations_grouped_and_contract_validated(self):
        self.assertEqual(len(self.data["cases"]), 64)
        self.assertEqual(len({c["group"] for c in self.data["cases"]}), 31)
        self.assertEqual(sum(c["split"] == "heldout" for c in self.data["cases"]), 26)
        self.assertEqual(self.cases["move_plain_en"]["group"], self.cases["move_negated_copy_zh"]["group"])
        self.assertEqual(self.data["annotation_version"], "2.0")

    def test_changed_frozen_data_rejected(self):
        path = self.root / "changed.json"
        path.write_text(DATASET.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen dataset/protocol changed"):
            load_dataset(path)

    def test_group_leakage_rejected_even_without_hash_check(self):
        data = deepcopy(self.data)
        data["cases"][1]["split"] = "heldout"
        path = self.root / "leak.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "group leakage"):
            load_dataset(path, verify_freeze=False)

    def test_formal_plan_is_complete_paired_and_budgeted(self):
        plan = make_plan(self.data, formal=True)
        self.assertEqual(plan["planned_runs"], 468)
        self.assertEqual(plan["maximum_physical_requests"], 3924)
        self.assertEqual(plan["semantic_group_count"], 13)
        self.assertTrue(all(p["max_model_calls"] == 8 for p in plan["runs"] if p["condition"] == "same_total"))
        self.assertTrue(all(p["max_model_calls"] == 8 for p in plan["runs"] if p["mode"] == "A"))
        self.assertEqual({p["max_model_calls"] for p in plan["runs"] if p["condition"] == "extra_classifier" and p["mode"] == "C"}, {9, 10})
        with self.assertRaisesRegex(ValueError, "subset"):
            make_plan(self.data, formal=True, max_cases=1)

    def test_counterbalanced_mode_orders(self):
        plan = make_plan(self.data, max_cases=3)
        self.assertEqual([p["mode"] for p in plan["runs"]], ["A", "B", "C", "B", "C", "A", "C", "A", "B"])

    def test_real_directory_outcome_requires_directory(self):
        case = self.cases["make_directory_en"]
        result = run_case(case, self.root / "directory", mode="C")
        self.assertTrue(result["task_success"], result)
        snapshot = snapshot_tree(self.root / "directory/workspace")
        del snapshot["entries"]["archive"]
        self.assertFalse(filesystem_score(case, snapshot)["passed"])

    def test_scripted_restart_and_append_once_in_all_arms(self):
        for mode in ("A", "B", "C"):
            with self.subTest(mode=mode):
                result = run_case(self.cases["append_once_en"], self.root / mode, mode=mode)
                self.assertTrue(result["task_success"], result)
                self.assertTrue(result["controls"]["injected_crash"])
                self.assertTrue(result["controls"]["terminal_replay_unchanged"])
                self.assertEqual(result["physical_model_requests"], 0)
                self.assertEqual((self.root / mode / "workspace/notes.txt").read_bytes(), b"alpha\nbeta\n")

    def test_clarification_correction_updates_annotations_and_state(self):
        result = run_case(self.cases["clarification_correction_en"], self.root / "correct", mode="C")
        self.assertTrue(result["task_success"], result)
        records = result["classification_records"]
        self.assertEqual([r["stage"] for r in records], ["initial", "updated"])
        self.assertEqual(records[1]["prediction"]["operations"][0]["parameters"]["content"]["value"], "final")
        self.assertEqual(result["controls"]["answers"], 1)

    def test_wrong_clarification_slot_does_not_receive_oracle_answer(self):
        wrong = ScriptedPlanner([{"type": "ask", "question": "What content should I write?"}])
        with patch("evals.intent_eval.ScriptedPlanner", return_value=wrong):
            result = run_case(self.cases["missing_name_en"], self.root / "wrong-slot", mode="C")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["controls"]["answers"], 0)
        self.assertTrue(result["controls"]["mismatched_clarifications"])
        self.assertTrue(result["missing_clarification"])

    def test_success_claim_and_unchanged_files_do_not_answer_question(self):
        wrong = ScriptedPlanner([{"type": "finish", "answer": "Successfully read the file."}])
        with patch("evals.intent_eval.ScriptedPlanner", return_value=wrong):
            result = run_case(self.cases["read_answer_en"], self.root / "empty-answer", mode="A")
        self.assertTrue(result["filesystem_pass"])
        self.assertEqual(result["runtime_status"], "succeeded")
        self.assertFalse(result["answer_pass"])
        self.assertFalse(result["protocol_pass"])
        self.assertFalse(result["task_success"])

    def test_context_is_verified_by_real_parent_actions(self):
        result = run_case(self.cases["reference_copy_en"], self.root / "context", mode="C")
        self.assertTrue(result["task_success"], result)
        setup = json.loads((self.root / "context/context_setup.json").read_text(encoding="utf-8"))
        self.assertEqual(setup["actions"][0]["status"], "verified")
        task = json.loads((self.root / "context/trace.json").read_text(encoding="utf-8"))["task"]
        self.assertEqual(task["parent_id"], setup["id"])
        self.assertEqual(task["intent"]["operations"][0]["parameters"]["source"]["source"], "verified_context")

    def test_live_model_inputs_contain_no_oracle(self):
        contexts = []
        case = self.cases["create_exact_en"]

        class Planner:
            def __call__(self, state, tools):
                contexts.append(deepcopy(state))
                return {"type": "finish", "answer": "No file created."}

        class Classifier:
            def __init__(self, mode):
                pass

            def __call__(self, state, tools):
                contexts.append(deepcopy(state))
                return deepcopy(case["gold"])

        with patch("task_agent.planner.ModelPlanner", Planner), patch("task_agent.intent.ModelIntentClassifier", Classifier):
            result = run_case(case, self.root / "live-input", mode="C", live=True)
        self.assertFalse(result["task_success"])
        self.assertEqual(len(contexts), 2)
        for state in contexts:
            self.assertEqual(state["request"], case["request"])
            for forbidden in ("gold", "script", "expected_files", "allowed_actions", "answer_terms", "files"):
                self.assertNotIn(forbidden, state)

    def test_budget_keeps_all_planned_denominators(self):
        from model_telemetry import model_request
        case = self.cases["create_exact_en"]

        class Classifier:
            def __init__(self, mode):
                pass

            def __call__(self, state, tools):
                # Synthetic telemetry admission, no network/model execution.
                with model_request(provider="qwen", transport="openai", model="qwen3:4b", parameters={}):
                    return deepcopy(case["gold"])

        class Planner:
            def __call__(self, state, tools):
                with model_request(provider="qwen", transport="openai", model="qwen3:4b", parameters={}):
                    return {"type": "finish", "answer": "No file created."}

        with patch("task_agent.planner.ModelPlanner", Planner), patch("task_agent.intent.ModelIntentClassifier", Classifier), \
                patch("evals.intent_eval.local_model_metadata", return_value={"status": "synthetic_test"}), \
                patch("sys.stdout", new_callable=io.StringIO):
            report = run_suite(self.root / "budget", live=True, case_ids=[case["id"]], modes=("C", "A", "B"), max_calls=1)
        self.assertEqual(report["planned_runs"], 3)
        self.assertEqual(report["started_runs"], 1)
        self.assertEqual(report["physical_model_requests"], 1)
        self.assertEqual(report["full_passes"], 0)
        self.assertEqual([r["evaluation_status"] for r in report["results"]],
                         ["incomplete_budget", "not_started_budget", "not_started_budget"])
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["groups"]["same_total/A"]["task_success_rate_planned"], 0)

    def test_missing_or_budget_classification_is_not_runtime_fallback(self):
        case = self.cases["create_exact_en"]
        self.assertIsNone(classifier_records(case, {}, "C")[0]["fallback_reason"])
        task = {"intent_history": [{"status": "budget_exhausted", "error": "budget", "intent": None}]}
        self.assertIsNone(classifier_records(case, task, "C")[0]["fallback_reason"])
        task["intent_history"][0]["status"] = "fallback"
        self.assertEqual(classifier_records(case, task, "C")[0]["fallback_reason"], "budget")

    def test_remote_endpoint_and_smoke_over_budget_rejected(self):
        with patch.dict("os.environ", {"QWEN_BASE_URL": "https://remote.invalid/v1"}):
            with self.assertRaisesRegex(ValueError, "loopback"):
                with local_configuration():
                    self.fail("Remote model configuration should not be admitted")
        with self.assertRaisesRegex(ValueError, "30 physical requests"):
            run_suite(self.root / "oversize", live=True, max_calls=31)
        self.assertFalse((self.root / "oversize").exists())

    def test_deadline_clamps_each_request(self):
        received = []
        wrapped = DeadlineCall(lambda state, tools: received.append(state["remaining_seconds"]), 110)
        with patch("evals.intent_eval.time.monotonic", return_value=100):
            wrapped({"remaining_seconds": 600}, [])
        self.assertEqual(received, [10])

    def test_regrade_uses_hash_verified_saved_evidence_without_execution(self):
        source, destination = self.root / "source", self.root / "regrade"
        with patch("sys.stdout", new_callable=io.StringIO):
            run_suite(source, modes=("C",), case_ids=["create_exact_en"])
        manifest_bytes = (source / "manifest.json").read_bytes()
        (source / "create_exact_en-C-same_total-r01/workspace/notes.txt").write_bytes(b"changed after snapshot")
        with patch("task_agent.runtime.AgentRuntime", side_effect=AssertionError("No runtime in regrade")):
            report = regrade_saved_suite(source, destination)
        self.assertTrue(report["saved_only_regrade"])
        self.assertEqual(report["new_physical_model_requests"], 0)
        self.assertEqual(report["full_passes"], 1)
        self.assertEqual(report["scorer_version"], "2.1")
        self.assertEqual((source / "manifest.json").read_bytes(), manifest_bytes)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("evals/intent_eval.py", manifest["original_code_sha256"])
        self.assertIn("scorer_code_sha256", manifest)

    def test_regrade_rejects_changed_original_trace(self):
        source = self.root / "source"
        with patch("sys.stdout", new_callable=io.StringIO):
            run_suite(source, modes=("A",), case_ids=["create_exact_en"])
        (source / "create_exact_en-A-same_total-r01/trace.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash is missing or changed"):
            regrade_saved_suite(source, self.root / "changed")


if __name__ == "__main__":
    unittest.main()
