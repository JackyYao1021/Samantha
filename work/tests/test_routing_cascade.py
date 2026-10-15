"""Cascaded early exits, dev-only fitting, failure boundaries and saved metrics."""

from copy import deepcopy
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.routing_cascade import (CASCADE_VERSION, calibrate_embedding, cascade_route,
                                  embedding_candidate, rule_candidate, validate_settings, load_prototype_examples)
from evals.routing_client import EvidenceWriteError, OllamaClient
from evals.routing_runner import load_config, solve_case, planned_requests, run
from evals.routing_tasks import load_cases
from evals.routing_analysis import summarize_results
from model_budget import BudgetExceeded


SETTINGS = {"similarity_thresholds": [.4, .7, .9], "margin_thresholds": [.05, .2],
            "min_precision": 1., "min_accepted": 3}
SELECTION = {"economical_model": "default", "family_models": {"write": "writer", "append": "appender"}}


def prototype(case, family, vector, **extra):
    return {"case_id": case, "split": "development", "family": family, "vector": vector, **extra}


PROTOTYPES = [prototype("a", "write", [1, 0]), prototype("b", "write", [.99, .01]),
              prototype("c", "append", [0, 1]), prototype("d", "append", [.01, .99])]


class CalibrationTests(unittest.TestCase):
    def test_no_self_match_and_frozen_gate_is_independent_of_prototype_order(self):
        fitted = calibrate_embedding(PROTOTYPES, SETTINGS)
        self.assertEqual(fitted, calibrate_embedding(list(reversed(PROTOTYPES)), SETTINGS))
        self.assertTrue(fitted["gate"]["enabled"])
        self.assertEqual(fitted["gate"]["accepted"], 4)
        for row in fitted["rows"]:
            self.assertNotEqual(row["case_id"], row["prototype_id"])

    def test_grouped_translations_are_excluded_together(self):
        grouped = [{**p, "group_id": p["family"]} for p in PROTOTYPES]
        fitted = calibrate_embedding(grouped, SETTINGS)
        self.assertFalse(fitted["gate"]["enabled"])
        self.assertTrue(all(row["candidate_family"] != row["expected_family"] for row in fitted["rows"]))

    def test_heldout_labels_duplicate_ids_and_invalid_vectors_rejected(self):
        for prototypes in ([{**PROTOTYPES[0], "split": "heldout"}],
                           [PROTOTYPES[0], PROTOTYPES[0]],
                           [{**PROTOTYPES[0], "vector": [float("nan"), 0]}]):
            with self.subTest(prototypes=prototypes), self.assertRaises(ValueError):
                calibrate_embedding(prototypes, SETTINGS)

    def test_single_class_or_insufficient_support_cannot_enable_gate(self):
        for prototypes in ([], PROTOTYPES[:2], [PROTOTYPES[0], PROTOTYPES[2]]):
            with self.subTest(prototypes=prototypes):
                self.assertFalse(calibrate_embedding(prototypes, SETTINGS)["gate"]["enabled"])

    def test_margin_uses_different_classes_not_two_nearest_same_class_prototypes(self):
        fitted = calibrate_embedding(PROTOTYPES, SETTINGS)
        decision = embedding_candidate([1, 0], PROTOTYPES, fitted["gate"])
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["runner_up_family"], "append")
        self.assertGreater(decision["margin"], .9)

    def test_high_similarity_but_ambiguous_category_abstains(self):
        vectors = [prototype("a", "write", [1, 0]), prototype("b", "append", [1, .001])]
        decision = embedding_candidate([1, 0], vectors, {"enabled": True, "similarity_threshold": .8, "margin_threshold": .05})
        self.assertEqual(decision["status"], "abstain")
        self.assertIsNone(decision["family"])

    def test_invalid_grids_do_not_silently_enable_acceptance(self):
        for changed in ({"margin_thresholds": [0]}, {"similarity_thresholds": [True]},
                        {"similarity_thresholds": [float("nan")]}, {"min_precision": 0},
                        {"min_accepted": True}, {"margin_thresholds": [.1, .1]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_settings({**SETTINGS, **changed})


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.embed = Mock(return_value=[1, 0])
        self.classify = Mock(return_value={"family": "append"})
        self.calibration = calibrate_embedding(PROTOTYPES, SETTINGS)

    def route(self, request="Produce the requested artifact.", prototypes=None):
        return cascade_route({"request": request, "files": {}}, selection=SELECTION,
            prototypes=PROTOTYPES if prototypes is None else prototypes, calibration=self.calibration,
            embed=self.embed, classify=self.classify, trace=self.trace)

    def test_explicit_rule_exits_without_any_embedding_or_classifier_call(self):
        decision = self.route("Create note.txt containing 'rename then append'.")
        self.assertEqual(decision["stage"], "rule")
        self.assertEqual(decision["model"], "writer")
        self.embed.assert_not_called()
        self.classify.assert_not_called()

    def test_unknown_negated_conditional_and_conflicting_rules_abstain(self):
        for request in ("Produce the artifact.", "Do not rename report.txt.",
                        "If ready, append to log.txt.", "Rename a.txt to b.txt; search for summaries.",
                        "不要覆盖记录.txt。"):
            with self.subTest(request=request):
                self.assertEqual(rule_candidate({"request": request})["status"], "abstain")

    def test_embedding_acceptance_avoids_llm(self):
        decision = self.route()
        self.assertEqual(decision["stage"], "embedding")
        self.assertEqual([s["status"] for s in self.trace], ["abstain", "accepted"])
        self.embed.assert_called_once()
        self.classify.assert_not_called()

    def test_ambiguous_embedding_escalates_to_one_structured_llm_classification(self):
        self.embed.return_value = [1, 1]
        decision = self.route()
        self.assertEqual(decision["model"], "appender")
        self.assertEqual(decision["stage"], "llm")
        self.classify.assert_called_once()

    def test_transport_error_escalates_and_retains_evidence(self):
        self.embed.side_effect = TimeoutError("local request timed out")
        decision = self.route()
        self.assertEqual(decision["stage"], "llm")
        self.assertEqual(self.trace[1]["status"], "error")

    def test_unsupported_gate_skips_embedding_without_a_request(self):
        self.calibration = calibrate_embedding([], SETTINGS)
        self.assertEqual(self.route(prototypes=[])["stage"], "llm")
        self.assertEqual(self.trace[1]["status"], "skipped")
        self.embed.assert_not_called()

    def test_unknown_invalid_or_failed_llm_uses_default_without_fake_family(self):
        self.embed.return_value = [1, 1]
        for value in ({"family": "unknown"}, {"family": "write", "extra": 1}, {"family": []}, TimeoutError("timeout")):
            self.trace.clear()
            self.classify.side_effect = value if isinstance(value, Exception) else None
            self.classify.return_value = value
            decision = self.route()
            self.assertEqual(decision["model"], "default")
            self.assertIsNone(decision["family"])
            self.assertTrue(decision["default_used"])

    def test_budget_and_persistence_failures_never_trigger_downstream_requests(self):
        for error in (BudgetExceeded("budget"), EvidenceWriteError("disk")):
            self.trace.clear()
            self.embed.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.route()
            self.classify.assert_not_called()
            self.assertEqual(self.trace[-1]["status"], "stopped")

    def test_llm_budget_failure_does_not_become_a_default_route(self):
        self.embed.return_value = [1, 1]
        self.classify.side_effect = BudgetExceeded("budget")
        with self.assertRaises(BudgetExceeded):
            self.route()
        self.assertEqual(self.trace[-1]["stage"], "llm")
        self.assertEqual(self.trace[-1]["status"], "stopped")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(WORK / "evals/experiments/routing_cascade_local.json")
        self.cases = load_cases(WORK / "evals/datasets/routing_tasks_v1.jsonl")

    def test_six_policy_plan_counts_all_potential_cascade_and_fallback_requests(self):
        counts = planned_requests(self.cases, self.config)
        self.assertEqual(counts["planned_holdout_attempts"], 144)
        self.assertEqual(counts["minimum_physical_requests"], 217)
        self.assertEqual(counts["maximum_physical_requests_before_budget"], 361)

    def test_config_requires_explicit_calibration_settings(self):
        bad = deepcopy(self.config)
        bad.pop("cascade")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_augmented_corpus_is_development_only_with_bilingual_groups(self):
        rows = load_prototype_examples(WORK / "evals/datasets/routing_prototypes_v2.json")
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r["group_id"] for r in rows}), 24)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prototypes.json"
            for bad in ([{**rows[0], "split": "heldout"}], [rows[0], rows[0]],
                        [rows[0], {**rows[1], "family": "rename"}]):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    load_prototype_examples(path)

    def test_augmented_run_freezes_and_audits_actual_transport_and_calibration(self):
        from evals.routing_audit import audit, verify_sources
        class Opener:
            def open(self, request, timeout):
                body = json.loads(request.data)
                result = {"done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5,
                          "prompt_eval_duration": 1000000, "eval_duration": 2000000, "load_duration": 0}
                if "input" in body:
                    result["embeddings"] = [[1., (i + 1) / 100] for i in range(len(body["input"]))]
                else:
                    payload = ({"family": "write"} if "family" in body["format"]["properties"]
                               else {"status": "completed", "actions": [], "answer": [], "question": ""})
                    result["message"] = {"content": json.dumps(payload)}
                return io.BytesIO(json.dumps(result).encode())
        def make_client(output, **kwargs):
            return OllamaClient(output, opener=Opener(), **kwargs)
        identities = {n: {"name": n, "digest": "scripted-offline"}
                      for n in [*self.config["models"], self.config["embedding_model"]]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            with patch("evals.routing_runner.read_inventory", return_value=identities), \
                    patch("evals.routing_runner.OllamaClient", side_effect=make_client):
                result = run(WORK / "evals/datasets/routing_tasks_v1.jsonl",
                    WORK / "evals/experiments/routing_cascade_augmented_local.json", root,
                    allow_local_model=True, case_limit=1)
            audited = audit(root, Path(temp) / "audit")
            self.assertTrue(audited["ledger_complete"])
            self.assertTrue(audited["cascade_calibration_verified"])
            self.assertEqual(result["manifest"]["prototype_dataset"]["groups"], 24)
            self.assertEqual(audited["regrade_count"], audited["regrade_agreement"])
            path = root / "prototypes.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "prototypes integrity"):
                verify_sources(root, result["manifest"])

    def test_full_solver_path_and_early_error_metrics_are_independent_of_gold(self):
        case = deepcopy(next(c for c in self.cases if c["split"] == "heldout"))
        # Deliberately force a wrong early classification; the actual scorer
        # still rejects the empty solver response and never retries on gold.
        case["input"]["request"] = "Append 'tail' to note.txt."
        class Client:
            calls = []
            def chat(self, messages, schema, *, model, **kwargs):
                self.calls.append({"model": model, "call_type": "chat", "role": "solver"})
                return {"status": "completed", "actions": [], "answer": [], "question": ""}
            def embed(self, *args, **kwargs):
                raise AssertionError("early exit must not call embedding")
        selection = {"economical_model": self.config["models"][0],
                     "family_models": {"append": self.config["models"][0]}}
        client = Client()
        result, stopped = solve_case(case, "cascade", 1, client=client, config=self.config,
            selection=selection, prototypes=PROTOTYPES, cascade_calibration=calibrate_embedding(PROTOTYPES, SETTINGS))
        self.assertFalse(stopped)
        self.assertEqual(result["task_success"], 0)
        self.assertFalse(result["fallback"])
        self.assertEqual(len(client.calls), 1)
        summary = summarize_results([result], [{k: result[k] for k in ("case_id", "family", "split", "policy", "repeat")}])
        quality = summary["policies"]["cascade"]["routing_quality"]
        self.assertEqual(quality["correct"], 0)
        self.assertEqual(quality["cascade_stages"]["rule"]["wrong_acceptances"], 1)
        self.assertEqual(quality["llm_classifier_calls"], 0)


if __name__ == "__main__":
    unittest.main()
