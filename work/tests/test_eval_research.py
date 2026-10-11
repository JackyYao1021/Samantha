"""Paired episodes, deterministic uncertainty and saved-only research comparisons."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.research import compare_runs, create_preregistration
from evals.statistics import bootstrap_mean_ci, episode_rates, paired_episode_comparison, wilson_interval
from test_eval_analysis import add_retry, partial_failure


def result(episode, attempt, value):
    return {"case_id": episode, "episode_id": episode, "attempt_id": attempt,
            "scores": [{"metric": "task_success", "value": value}]}


class StatisticsTests(unittest.TestCase):
    def test_repeated_attempts_are_averaged_inside_episode(self):
        left = [result("one", str(index), index % 2) for index in range(10)] + [result("two", "1", 0)]
        right = [result("one", str(index), 1) for index in range(10)] + [result("two", "1", 1)]
        comparison = paired_episode_comparison(left, right, bootstrap_samples=200)
        self.assertEqual(comparison["paired_episodes"], 2)
        self.assertEqual(comparison["left_episode_mean"], .25)
        self.assertEqual(comparison["difference"]["mean"], .75)
        self.assertEqual(comparison["difference"]["n"], 2)
        self.assertTrue(comparison["exploratory"])

    def test_bootstrap_seed_and_variance_are_reproducible(self):
        values = [0, .5, 1, 1]
        self.assertEqual(bootstrap_mean_ci(values, seed=42, samples=100), bootstrap_mean_ci(values, seed=42, samples=100))
        rates = episode_rates([result("one", "1", 0), result("one", "2", 1)])
        self.assertEqual(rates["one"]["p_i"], .5)
        self.assertEqual(rates["one"]["repeat_variance"], .5)
        self.assertEqual(rates["one"]["binomial_variance"], .25)

    def test_unmatched_episodes_and_empty_intersection_are_visible(self):
        comparison = paired_episode_comparison([result("left", "1", 1)], [result("right", "1", 0)], bootstrap_samples=10)
        self.assertEqual(comparison["paired_episodes"], 0)
        self.assertEqual(comparison["left_only_episodes"], ["left"])
        self.assertIsNone(comparison["difference"]["mean"])

    def test_duplicate_attempts_missing_metric_nan_and_boolean_are_rejected(self):
        for values in ([result("one", "1", 1)] * 2, [{"case_id": "one", "scores": []}], [result("one", "1", float("nan"))], [result("one", "1", True)]):
            with self.assertRaises(ValueError):
                episode_rates(values)

    def test_wilson_empty_and_all_success_does_not_claim_certainty(self):
        self.assertIsNone(wilson_interval(0, 0)["low"])
        interval = wilson_interval(5, 5)
        self.assertLess(interval["low"], 1)
        self.assertAlmostEqual(interval["high"], 1)


class ResearchTests(unittest.TestCase):
    def write_run(self, root, name, evidence=None, dataset_hash="same"):
        run = root / name
        run.mkdir()
        manifest = {"schema_version": "1.0", "experiment_id": name, "dataset": {"sha256": dataset_hash}, "model": {"provider": "fake"}}
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        saved = result("partial-append", "repeat1", 0)
        saved["evaluation_status"] = "failed"
        if evidence is not None:
            data = json.dumps(evidence).encode()
            (run / "evidence.json").write_bytes(data)
            saved.update(evidence_ref="evidence.json", evidence_sha256=hashlib.sha256(data).hexdigest())
        (run / "results.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
        return run

    def test_preregistration_draft_contains_one_hypothesis_and_no_authorized_model(self):
        plan = create_preregistration()
        self.assertEqual(plan["status"], "draft_for_preregistration")
        self.assertEqual(plan["primary_metric"], "partial_effect_safety")
        self.assertEqual(plan["target_independent_episodes"], 60)
        self.assertFalse(plan["real_execution"]["authorized"])
        self.assertEqual(len(plan["variants"]), 2)

    def test_saved_partial_effects_comparison_distinguishes_safety_from_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = self.write_run(root, "replay", add_retry(partial_failure()))
            right = self.write_run(root, "stop", partial_failure())
            comparison = compare_runs([left, right], metric="partial_effect_safety", bootstrap_samples=100)
            values = comparison["comparisons"][0]
            self.assertEqual(values["left_episode_mean"], 0)
            self.assertEqual(values["right_episode_mean"], 1)
            self.assertFalse(comparison["model_improvement_proven"])
            self.assertEqual(compare_runs([left, right], bootstrap_samples=10)["comparisons"][0]["difference"]["mean"], 0)

    def test_missing_evidence_cannot_pass_research_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = self.write_run(root, "missing")
            right = self.write_run(root, "stop", partial_failure())
            values = compare_runs([left, right], metric="partial_effect_safety", bootstrap_samples=10)["comparisons"][0]
            self.assertEqual(values["left_episode_mean"], 0)
            self.assertEqual(values["left_recovery_coverage"]["complete_evidence_attempts"], 0)

    def test_mismatched_datasets_and_tampered_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = self.write_run(root, "left", partial_failure())
            mismatch = self.write_run(root, "mismatch", partial_failure(), dataset_hash="different")
            with self.assertRaisesRegex(ValueError, "dataset"):
                compare_runs([left, mismatch], bootstrap_samples=10)
            right = self.write_run(root, "right", partial_failure())
            (right / "evidence.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                compare_runs([left, right], metric="partial_effect_safety", bootstrap_samples=10)

    def test_saved_evidence_reference_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = self.write_run(root, "left", partial_failure())
            right = self.write_run(root, "right", partial_failure())
            record = json.loads((right / "results.jsonl").read_text())
            record["evidence_ref"] = "../left/evidence.json"
            (right / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                compare_runs([left, right], metric="partial_effect_safety", bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
