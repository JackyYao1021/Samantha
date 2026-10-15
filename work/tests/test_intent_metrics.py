"""Counterexamples for intent metrics, without classifier-generated gold labels."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.intent_metrics import classification_metrics, failure_analysis


def intent(*operations, domains=None, constraints=None):
    return {"operations": [{"id": "op" + str(i), "operation": name, "parameters": {
        key: {"value": value, "source": "request", "evidence": "request text"}
        for key, value in args.items()}, "depends_on": []} for i, (name, args) in enumerate(operations)],
        "domains": domains or ["file_management"], "constraints": constraints or [],
        "support": "supported", "fallback": "none"}


def metrics(gold, prediction, mode="C", **extra):
    return classification_metrics([{"mode": mode, "case_id": "test", "gold": gold,
                                    "prediction": prediction, "stage": "initial", **extra}])[mode]["initial"]


class IntentMetricTests(unittest.TestCase):
    def test_repeated_operations_match_by_slots_not_array_order(self):
        gold = intent(("read", {"path": "a.txt"}), ("read", {"path": "b.txt"}), domains=["content_query"])
        prediction = deepcopy(gold)
        prediction["operations"].reverse()
        result = metrics(gold, prediction)
        self.assertEqual(result["parameters"]["exact_match_rate"], 1)
        self.assertEqual(result["multi_intent_completeness"], 1)

    def test_one_read_cannot_satisfy_two_read_intents(self):
        gold = intent(("read", {"path": "a.txt"}), ("read", {"path": "b.txt"}), domains=["content_query"])
        prediction = deepcopy(gold)
        prediction["operations"].pop()
        result = metrics(gold, prediction)
        self.assertEqual(result["operation_macro_f1"], 1)  # Presence and multiplicity are distinct metrics.
        self.assertEqual(result["multi_intent_completeness"], 0)
        self.assertEqual(result["parameters"]["recall"], .5)

    def test_failed_classifier_counts_as_missing_not_dropped_denominator(self):
        gold = intent(("copy", {"source": "a", "destination": "b"}))
        result = metrics(gold, None, fallback_reason="invalid schema")
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["operation_macro_f1"], 0)
        self.assertEqual(result["parameters"]["recall"], 0)
        self.assertEqual(result["runtime_fallback_count"], 1)
        self.assertEqual(result["advised_fallback_count"], 0)

    def test_fine_grained_confusion_and_domain_are_separate(self):
        gold = intent(("copy", {"source": "a", "destination": "b"}))
        prediction = intent(("move", {"source": "a", "destination": "b"}))
        result = metrics(gold, prediction)
        self.assertEqual(result["domain_macro_f1"], 1)
        self.assertEqual(result["operation_macro_f1"], 0)
        self.assertEqual(result["critical_confusions"]["copy_as_move"], 1)

    def test_mixed_copy_move_is_not_a_confusion(self):
        gold = intent(("copy", {"source": "a", "destination": "b"}), ("move", {"source": "b", "destination": "c"}))
        self.assertEqual(metrics(gold, gold)["critical_confusions"]["copy_as_move"], 0)

    def test_extra_constraint_lowers_precision(self):
        gold = intent(("read", {"path": "a"}), constraints=[{"kind": "scope", "value": "read_only"}])
        pred = deepcopy(gold)
        pred["constraints"].append({"kind": "preserve_source", "value": True})
        result = metrics(gold, pred)
        self.assertEqual(result["constraints"]["precision"], .5)
        self.assertEqual(result["constraints"]["recall"], 1)
        self.assertEqual(result["constraints"]["exact_match_rate"], 0)

    def test_parameter_provenance_is_not_inferred_from_value_match(self):
        gold = intent(("read", {"path": "a"}))
        pred = deepcopy(gold)
        pred["operations"][0]["parameters"]["path"]["source"] = "verified_context"
        result = metrics(gold, pred)
        self.assertEqual(result["parameters"]["recall"], 1)
        self.assertEqual(result["parameter_sources"]["accuracy"], 0)

    def test_redundant_order_constraint_is_not_an_extraction_error(self):
        gold = intent(("copy", {"source": "a", "destination": "b"}))
        pred = deepcopy(gold)
        pred["constraints"] = [{"kind": "order", "value": "copy first"}]
        self.assertEqual(metrics(gold, pred)["constraints"]["exact_match_rate"], 1)

    def test_advised_fallback_is_not_runtime_classification_failure(self):
        gold = intent(("unknown", {}), domains=["unknown"])
        gold.update(support="unknown", fallback="clarify")
        result = metrics(gold, gold)
        self.assertEqual(result["runtime_fallback_count"], 0)
        self.assertEqual(result["advised_fallback_count"], 1)

    def test_baseline_and_flat_have_no_fabricated_domain_scores(self):
        gold = intent(("read", {"path": "a"}))
        self.assertFalse(classification_metrics([])["A"]["applicable"])
        self.assertIsNone(metrics(gold, gold, mode="B")["domain_macro_f1"])

    def test_unknown_safety_and_recognition_are_separate(self):
        gold = intent(("unknown", {}), domains=["unknown"])
        gold.update(support="unknown", fallback="clarify")
        result = metrics(gold, None, fallback_reason="parse failure")
        self.assertEqual(result["unknown"]["recognition_accuracy"], 0)
        self.assertEqual(result["unknown"]["fallback_rate"], 1)

    def test_failure_analysis_does_not_penalize_observation_tools(self):
        case = {"gold": intent(("copy", {"source": "a", "destination": "b"})),
                "allowed_actions": [{"tool": "copy_file", "args": {"source": "a", "destination": "b"}}]}
        task = {"actions": [{"tool": "list_files", "args": {"path": "."}, "status": "observed"},
                            {"tool": "copy_file", "args": {"source": "a", "destination": "b"}, "status": "verified"}]}
        self.assertEqual(failure_analysis(case, task, {}, True, True), [])
        task["actions"][1]["tool"] = "move_file"
        self.assertIn("tool_selection", failure_analysis(case, task, {}, False, False))

    def test_bad_reference_and_infrastructure_are_retained_as_overlapping_labels(self):
        gold = intent(("copy", {"source": "prior.txt", "destination": "b"}))
        gold["operations"][0]["parameters"]["source"]["source"] = "verified_context"
        case = {"gold": gold, "allowed_actions": [{"tool": "copy_file", "args": {"source": "prior.txt", "destination": "b"}}]}
        task = {"actions": [{"tool": "copy_file", "args": {"source": "guessed.txt", "destination": "b"}, "status": "failed"}],
                "model_call_records": [{"status": "failed", "exception_type": "ReadTimeout"}]}
        result = failure_analysis(case, task, {}, False, True)
        self.assertEqual(result, ["infrastructure", "omitted_operation", "parameters", "reference"])


if __name__ == "__main__":
    unittest.main()
