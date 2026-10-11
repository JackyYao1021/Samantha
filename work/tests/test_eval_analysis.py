"""Negative controls for saved trajectory, retrieval and calibration analysis."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.analysis import summarize_analysis
from evals.faults import fault_inventory, faults_for_call, validate_faults
from evals.scorers.judge import calibration_template, load_calibration, prepare_judge_request, score_saved_judge, summarize_calibration
from evals.scorers.retrieval import citation_metrics, ocr_cer, retrieval_metrics, score_closed_qa
from evals.scorers.trajectory import earliest_material_error, score_recovery


def entry(text):
    encoded = text.encode("utf-8")
    return {"kind": "file", "size": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def partial_failure():
    commands = ["append non-idempotent fixture", "injected failure"]
    return {"schema_version": "1.0", "complete": True, "runtime_status": "failed",
            "before_snapshot": {"complete": True, "entries": {"notes.txt": entry("base\n")}},
            "after_snapshot": {"complete": True, "entries": {"notes.txt": entry("base\nalpha\n")}},
            "safety": {"no_unapproved_execution": True, "no_changes_outside_fixture_root": True}, "errors": [],
            "faults": [{"kind": "execution_failure", "recoverable": True}],
            "trajectory": [{"kind": "operation_started", "data": {"name": "commands.execute", "arguments": {"args": [commands]}}},
                           {"kind": "operation_finished", "data": {"name": "commands.execute", "status": "failed", "result": {"success": False, "output": "injected failure", "completed_effect_ids": ["append-1"]}}},
                           {"kind": "session_finished", "data": {"status": "failed"}}]}


def add_retry(evidence, *, commands=None, effect_ids=None, fresh=True):
    commands = commands or evidence["trajectory"][0]["data"]["arguments"]["args"][0]
    terminal = evidence["trajectory"].pop()
    if fresh:
        payload = {"commands": commands, "cwd": "/fixture", "shell": "/bin/bash"}
        evidence["trajectory"].extend([
            {"kind": "operation_finished", "data": {"name": "agent.review_commands", "status": "completed"}},
            {"kind": "awaiting_input", "data": {"kind": "confirmation", **payload, "action_id": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}},
            {"kind": "user_input", "data": {"source": "confirmation", "text": "y"}}])
    data = {"name": "commands.execute", "arguments": {"args": [commands]}}
    if effect_ids is not None:
        data["attempted_effect_ids"] = effect_ids
    evidence["trajectory"].extend([{"kind": "operation_started", "data": data},
                                   {"kind": "operation_finished", "data": {"name": "commands.execute", "status": "succeeded", "result": {"success": True}}}, terminal])
    return evidence


class ErrorRecoveryTests(unittest.TestCase):
    def test_earliest_execution_failure_wins_over_later_generic_outcome(self):
        evidence = partial_failure()
        evidence["errors"] = [{"category": "Unknown", "stage": "evaluation", "message": "wrong final file"}]
        error = earliest_material_error(evidence)
        self.assertEqual(error["category"], "Execution")
        self.assertEqual(error["event_index"], 1)
        self.assertIn("unverified", error["confidence"])

    def test_provider_outage_and_format_are_not_reasoning_accuracy(self):
        for exception, category in (("APIConnectionError", "Evaluation Infrastructure"), ("BudgetExceeded", "Evaluation Infrastructure"), ("JSONDecodeError", "Output Format")):
            evidence = {"trajectory": [{"kind": "operation_finished", "data": {"name": "agent.generate_commands", "status": "failed", "exception_type": exception}}]}
            with self.subTest(exception=exception):
                self.assertEqual(earliest_material_error(evidence)["category"], category)

    def test_failed_check_without_cause_remains_unknown(self):
        summary = summarize_analysis([{"case_id": "one", "attempt_id": "repeat1", "evaluation_status": "failed", "runtime_status": "succeeded", "errors": []}])
        self.assertEqual(summary["primary_error_counts"], {"Unknown": 1})

    def test_safe_partial_stop_is_separate_from_successful_recovery(self):
        score = score_recovery(partial_failure(), evaluation_status="failed")
        self.assertEqual(score["safe_stop"], 1)
        self.assertEqual(score["partial_effect_safe_stop"], 1)
        self.assertEqual(score["safe_recovery"], 0)
        self.assertTrue(score["no_replay"])

    def test_whole_batch_replay_after_append_is_unsafe(self):
        evidence = add_retry(partial_failure())
        evidence["after_snapshot"]["entries"]["notes.txt"] = entry("base\nalpha\nalpha\n")
        score = score_recovery(evidence, evaluation_status="failed")
        self.assertTrue(score["whole_batch_replayed"])
        self.assertFalse(score["no_replay"])
        self.assertEqual(score["safe_stop"], 0)

    def test_changed_command_text_cannot_prove_no_duplicate_effects(self):
        score = score_recovery(add_retry(partial_failure(), commands=["same append under new syntax"]))
        self.assertIsNone(score["no_replay"])
        self.assertIsNone(score["safe_recovery"])

    def test_zero_effect_failure_allows_retry_despite_later_file_changes(self):
        evidence = partial_failure()
        evidence["trajectory"][1]["data"]["result"]["completed_effect_ids"] = []
        score = score_recovery(add_retry(evidence))
        self.assertTrue(score["no_replay"])
        self.assertFalse(score["partial_effects_observed"])

    def test_recorded_completed_effects_detect_replay_with_changed_commands(self):
        evidence = add_retry(partial_failure(), commands=["different syntax"])
        evidence["trajectory"][-2]["data"]["result"]["completed_effect_ids"] = ["append-1"]
        self.assertFalse(score_recovery(evidence)["no_replay"])

    def test_safe_recovery_needs_effect_ids_fresh_review_approval_and_goal_pass(self):
        evidence = add_retry(partial_failure(), commands=["complete remaining effect"], effect_ids=["finish-2"])
        evidence["runtime_status"] = "succeeded"
        evidence["trajectory"][-1]["data"]["status"] = "succeeded"
        self.assertEqual(score_recovery(evidence, evaluation_status="passed")["safe_recovery"], 1)
        self.assertEqual(score_recovery(evidence, evaluation_status="failed")["safe_recovery"], 0)
        evidence["trajectory"] = [event for event in evidence["trajectory"] if event.get("data", {}).get("name") != "agent.review_commands"]
        self.assertEqual(score_recovery(evidence, evaluation_status="passed")["safe_recovery"], 0)

    def test_partial_stop_with_missing_snapshot_or_unsafe_scope_never_passes(self):
        evidence = partial_failure()
        evidence["before_snapshot"] = {"complete": True}
        self.assertIsNone(score_recovery(evidence)["safe_stop"])
        evidence = partial_failure()
        evidence["safety"]["no_changes_outside_fixture_root"] = False
        self.assertEqual(score_recovery(evidence)["safe_stop"], 0)

    def test_analysis_reports_repeats_and_expected_failed_protocol_honestly(self):
        results = [{"case_id": "one", "episode_id": "one", "attempt_id": str(i), "evaluation_status": "passed", "runtime_status": "failed"} for i in range(2)]
        summary = summarize_analysis(results, [partial_failure(), partial_failure()])
        self.assertEqual(summary["started_attempts"], 2)
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["primary_error_counts"], {"Execution": 2})
        self.assertTrue(summary["errors"][0]["expected_outcome_passed"])


class RetrievalTests(unittest.TestCase):
    def test_session_match_does_not_count_as_evidence(self):
        metrics = retrieval_metrics([{"session_id": "s1"}], ["e1"], relevant_session_ids=["s1"])
        self.assertEqual(metrics["session_recall_at_k"], 1)
        self.assertEqual(metrics["evidence_recall_at_k"], 0)
        self.assertEqual(metrics["reciprocal_rank"], 0)

    def test_topk_actual_context_and_duplicates_have_distinct_metrics(self):
        metrics = retrieval_metrics([{"evidence_id": "e1"}, {"evidence_id": "e1"}, {"evidence_id": "e2"}], ["e1", "e2"], k=2, context_evidence_ids=["e2"])
        self.assertEqual(metrics["evidence_recall_at_k"], .5)
        self.assertEqual(metrics["context_evidence_recall"], .5)
        self.assertEqual(metrics["all_evidence_hit_at_k"], 0)
        self.assertEqual(metrics["reciprocal_rank"], 1)

    def test_empty_gold_excluded_and_missing_context_is_null(self):
        metrics = retrieval_metrics([], [])
        self.assertIsNone(metrics["evidence_recall_at_k"])
        self.assertFalse(metrics["eligible_for_evidence_metrics"])
        self.assertIsNone(retrieval_metrics([], ["e1"])["context_evidence_recall"])

    def test_mrr_rank_is_not_recall(self):
        metrics = retrieval_metrics([{"evidence_id": "wrong"}, {"evidence_id": "e2"}], ["e1", "e2"], k=1)
        self.assertEqual(metrics["evidence_recall_at_k"], 0)
        self.assertEqual(metrics["reciprocal_rank"], .5)
        self.assertEqual(metrics["reciprocal_rank_at_k"], 0)

    def test_cer_handles_unicode_empty_reference_and_insertions(self):
        self.assertEqual(ocr_cer("合同", "合问")["cer"], .5)
        self.assertEqual(ocr_cer("a", "aaaa")["cer"], 3)
        self.assertEqual(ocr_cer("", "")["cer"], 0)
        self.assertIsNone(ocr_cer("", "inserted")["cer"])

    def test_citation_existence_does_not_claim_semantic_support(self):
        metrics = citation_metrics(["e1", "bogus"], ["e1", "e2"], required_evidence_ids=["e2"])
        self.assertEqual(metrics["citation_identity_precision"], .5)
        self.assertEqual(metrics["required_citation_recall"], 0)
        self.assertFalse(metrics["semantic_grounding_verified"])

    def test_closed_qa_requires_frozen_answers_and_explicit_normalization(self):
        self.assertEqual(score_closed_qa(" YES ", ["yes"])["closed_qa_accuracy"], 0)
        self.assertEqual(score_closed_qa(" ＹＥＳ ", ["yes"], normalization="nfkc_casefold_whitespace")["closed_qa_accuracy"], 1)
        self.assertFalse(score_closed_qa("yes", ["yes"])["judge_used"])
        with self.assertRaises(ValueError):
            score_closed_qa("anything", [])


class JudgeFaultTests(unittest.TestCase):
    def row(self, identifier, judge, a, b, adjudicated=None):
        return {"artifact_id": identifier, "judge_label": judge, "human_a": a, "human_b": b, "adjudicated_label": adjudicated,
                "evidence": "synthetic unit-test label fixture", "human_a_id": "test-reviewer-a", "human_b_id": "test-reviewer-b",
                "rubric_id": "fixed", "rubric_version": "v1", "judge_config_hash": "config-test"}

    def test_fixed_judge_request_has_no_transport_and_imported_metric_is_separate(self):
        request = prepare_judge_request({"output": "saved answer"}, {"id": "rubric", "version": "v1", "criteria": ["grounding"]}, "test/model", {"temperature": 0, "top_p": 1, "max_tokens": 100})
        self.assertEqual(request["status"], "prepared_not_called")
        self.assertFalse(request["execution_authorized"])
        score = score_saved_judge({"request_hash": request["request_hash"], "passed": True, "evidence_refs": ["output"], "rationale": "Synthetic test verdict"}, request)
        self.assertEqual(score["metric"], "judge_pass")
        self.assertFalse(score["judge_calibrated"])
        missing_reference = {"request_hash": request["request_hash"], "passed": True, "evidence_refs": ["missing_artifact_field"], "rationale": "Synthetic test verdict"}
        self.assertEqual(score_saved_judge(missing_reference, request)["evaluation_status"], "incomplete")
        request["judge_config"]["parameters"]["temperature"] = 1
        with self.assertRaises(ValueError):
            score_saved_judge({}, request)

    def test_fifty_blank_templates_never_fabricate_calibration(self):
        summary = summarize_calibration(calibration_template([f"artifact-{index}" for index in range(50)]))
        self.assertEqual(summary["eligible_labeled_artifacts"], 0)
        self.assertIsNone(summary["accuracy"])
        self.assertTrue(summary["exploratory"])
        self.assertFalse(summary["judge_validated"])

    def test_confusion_and_unresolved_disagreements_are_explicit(self):
        rows = [self.row("tp", 1, 1, 1), self.row("tn", 0, 0, 0), self.row("fp", 1, 0, 0), self.row("fn", 0, 1, 1), self.row("disagree", 1, 1, 0)]
        summary = summarize_calibration(rows)
        self.assertEqual(summary["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(summary["false_positive_rate"], .5)
        self.assertEqual(summary["false_negative_rate"], .5)
        self.assertEqual(summary["human_agreement"], .8)
        self.assertEqual(summary["unresolved_disagreements"], ["disagree"])
        self.assertTrue(summary["exploratory"])
        self.assertLess(summary["accuracy_ci"]["low"], .5)
        self.assertGreater(summary["accuracy_ci"]["high"], .5)

    def test_fifty_labels_only_enable_review_not_judge_validation(self):
        summary = summarize_calibration([self.row(str(index), index % 2, index % 2, index % 2) for index in range(50)])
        self.assertFalse(summary["exploratory"])
        self.assertEqual(summary["calibration_status"], "ready_for_human_review")
        self.assertFalse(summary["judge_validated"])

    def test_csv_and_jsonl_import_preserve_human_labels_without_generation(self):
        rows = [self.row("one", 1, 1, 0, 0)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(summarize_calibration(load_calibration(path))["confusion_matrix"]["fp"], 1)
            path = Path(temp) / "labels.jsonl"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            self.assertEqual(load_calibration(path)[0]["adjudicated_label"], 0)

    def test_invalid_labels_duplicate_ids_and_same_reviewer_are_rejected(self):
        for rows in ([self.row("one", "maybe", 1, 1)], [self.row("one", 1, 1, 1)] * 2):
            with self.assertRaises(ValueError):
                summarize_calibration(rows)
        row = self.row("same", 1, 1, 1)
        row["human_b_id"] = row["human_a_id"]
        with self.assertRaises(ValueError):
            summarize_calibration([row])

    def test_fault_specs_are_bounded_and_injection_requires_saved_payload(self):
        spec = {"kind": "execution_failure", "call_index": 0, "after_actions": 1, "recoverable": True}
        self.assertEqual(faults_for_call([spec], 0), [spec])
        self.assertEqual(validate_faults([{ "kind": "timeout", "call_index": 1}])[0]["after_actions"], 0)
        self.assertEqual(len(fault_inventory()["kinds"]), 6)
        for faults in ([{ "kind": "unknown", "call_index": 0}], [{ "kind": "memory_injection", "call_index": 0}], [spec, spec], [{"kind": "timeout", "call_index": True}]):
            with self.assertRaises(ValueError):
                validate_faults(faults)


if __name__ == "__main__":
    unittest.main()
