"""Controls for routing denominators, paired units and counterfactual evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.routing_analysis import compare_policies, routing_diagnostics, summarize_results, write_routing_report


def result(case="case1", policy="fixed_a", *, repeat=0, model="model:a", success=1,
           complete=True, started=True, split="heldout", tokens=12, compute=20):
    return {"case_id": case, "family": "extract", "split": split, "policy": policy,
            "policy_kind": "fixed" if policy.startswith("fixed") else "routed",
            "model": model, "repeat": repeat, "started": started, "complete": complete,
            "task_success": success, "duration_ms": 50 if started else None,
            "router_duration_ms": 0 if policy.startswith("fixed") else 5,
            "fallback": False, "fallback_reason": None, "error_category": None,
            "initial_model": model, "selected_model": model,
            "calls": [{"model": model, "call_type": "chat", "duration_ms": 25,
                       "usage": {"input_tokens": tokens, "output_tokens": 3},
                       "status": "succeeded", "provider_compute_ms": compute}] if started else []}


def plan(*rows):
    return [{key: row[key] for key in ("case_id", "family", "split", "policy", "repeat")} for row in rows]


def fixed_pair(*, a_success=1, b_success=1, a_tokens=12, b_tokens=6, a_compute=20, b_compute=10):
    return [result(policy="fixed_a", model="model:a", success=a_success, tokens=a_tokens, compute=a_compute),
            result(policy="fixed_b", model="model:b", success=b_success, tokens=b_tokens, compute=b_compute)]


class RoutingSummaryTests(unittest.TestCase):
    def test_missing_and_incomplete_tasks_stay_in_full_denominator(self):
        good = result()
        incomplete = result("case2", complete=False, success=1)
        missing = result("case3")
        group = summarize_results([good, incomplete], plan(good, incomplete, missing))["policies"]["fixed_a"]
        self.assertEqual((group["planned"], group["started"], group["complete"]), (3, 2, 1))
        self.assertEqual(group["task_successes"], 1)
        self.assertAlmostEqual(group["task_success_rate"], 1 / 3)
        self.assertEqual(group["success_interval"]["n"], 3)
        self.assertEqual(group["error_category_counts"], {"incomplete": 1, "not_started": 1})
        self.assertEqual(group["physical_calls"], 2)
        self.assertFalse(group["physical_call_count_complete"])
        self.assertIsNone(group["usage"]["input_tokens"])
        self.assertEqual(group["usage"]["coverage"]["input_tokens"]["reported_sum"], 24)

    def test_explicit_unstarted_stub_is_not_latency_zero(self):
        good, stub = result(), result("case2", started=False, complete=False, success=0)
        group = summarize_results([good, stub], plan(good, stub))["policies"]["fixed_a"]
        self.assertEqual(group["not_started"], 1)
        self.assertEqual(group["latency_ms"]["p50"], 50)
        self.assertEqual(group["latency_ms"]["eligible"], 1)
        self.assertEqual(group["fallback"]["rate"], 0)

    def test_partial_embedding_output_usage_is_unknown_not_zero(self):
        row = result(policy="embedding")
        row["calls"].append({"model": "bge-m3", "call_type": "embedding", "duration_ms": 8,
                             "provider_compute_ms": 6, "usage": {"input_tokens": 7, "output_tokens": None}})
        row["output_tokens"] = 3  # Saved aggregates must not hide missing call usage.
        group = summarize_results([row], plan(row))["policies"]["embedding"]
        self.assertEqual(group["usage"]["input_tokens"], 19)
        self.assertIsNone(group["usage"]["output_tokens"])
        self.assertEqual(group["usage"]["coverage"]["output_tokens"]["ratio"], .5)
        self.assertEqual((group["physical_llm_calls"], group["physical_embedding_calls"]), (1, 1))
        self.assertIsNone(group["api_cost_usd"])

    def test_filter_dev_and_retain_family_breakdown(self):
        held, dev = result(), result("dev1", split="dev", success=0)
        summary = summarize_results([held, dev], plan(held, dev))
        self.assertEqual(summary["ignored_dev_records"], 1)
        self.assertEqual(summary["ignored_dev_planned"], 1)
        group = summary["policies"]["fixed_a"]
        self.assertEqual(group["task_success_rate"], 1)
        self.assertEqual(group["families"]["extract"]["planned"], 1)

    def test_duplicate_records_and_plans_are_rejected(self):
        row = result()
        for records, planned in (([row, deepcopy(row)], plan(row)), ([row], plan(row, row))):
            with self.subTest(records=len(records), planned=len(planned)), self.assertRaisesRegex(ValueError, "Duplicate"):
                summarize_results(records, planned)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_policies([row, deepcopy(row)])

    def test_bad_metrics_and_boolean_success_are_rejected(self):
        alterations = [("duration_ms", -1), ("router_duration_ms", float("nan")),
                       ("duration_ms", float("inf")), ("task_success", True), ("input_tokens", -2)]
        for field, value in alterations:
            row = result()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize_results([row], plan(row))
        row = result()
        row["calls"][0]["usage"]["output_tokens"] = 1.5
        with self.assertRaises(ValueError):
            summarize_results([row], plan(row))

    def test_unknown_latency_and_usage_do_not_use_zero_or_top_level_totals(self):
        row = result()
        row.update(duration_ms=None, router_duration_ms=None, input_tokens=200, calls=None)
        group = summarize_results([row], plan(row))["policies"]["fixed_a"]
        self.assertIsNone(group["latency_ms"]["p95"])
        self.assertEqual(group["latency_ms"]["coverage"], 0)
        self.assertIsNone(group["usage"]["input_tokens"])
        self.assertFalse(group["physical_call_count_complete"])

    def test_unplanned_records_and_family_conflict_are_rejected(self):
        row = result()
        with self.assertRaisesRegex(ValueError, "planned"):
            summarize_results([row], [])
        planned = plan(row)
        planned[0]["family"] = "other"
        with self.assertRaisesRegex(ValueError, "family"):
            summarize_results([row], planned)


class RoutingComparisonTests(unittest.TestCase):
    def test_repeats_are_paired_then_averaged_as_two_case_units(self):
        rows = [result(case, policy, repeat=repeat, success=int(policy == "rule"))
                for case in ("case1", "case2") for policy in ("fixed_a", "rule") for repeat in range(10)]
        pair = compare_policies(rows)["comparisons"][0]
        self.assertEqual(pair["matched_attempts"], 20)
        self.assertEqual(pair["paired_cases"], 2)
        self.assertEqual(pair["difference"]["n"], 2)
        self.assertEqual(pair["difference"]["mean"], 1)
        self.assertTrue(pair["exploratory"])
        self.assertEqual(pair["complete_pair_coverage"], 1)

    def test_unmatched_repeats_are_explicit_and_never_cross_paired(self):
        rows = [result(policy="fixed_a", repeat=0, success=0),
                result(policy="fixed_a", repeat=1, success=1),
                result(policy="rule", repeat=0, success=1),
                result(policy="rule", repeat=2, success=0)]
        pair = compare_policies(rows)["comparisons"][0]
        self.assertEqual(pair["matched_attempts"], 1)
        self.assertEqual(pair["difference"]["mean"], 1)
        self.assertFalse(pair["identical_attempt_coverage"])
        self.assertEqual(pair["left_only_attempts"], [{"case_id": "case1", "repeat": 1}])
        self.assertEqual(pair["right_only_attempts"], [{"case_id": "case1", "repeat": 2}])

    def test_incomplete_and_unstarted_pairs_are_zero_with_explicit_coverage(self):
        left = result(success=1, complete=False)
        right = result(policy="rule", started=False, complete=False, success=0)
        pair = compare_policies([left, right])["comparisons"][0]
        self.assertEqual(pair["difference"]["mean"], 0)
        self.assertEqual(pair["complete_pair_coverage"], 0)


class RoutingDiagnosticTests(unittest.TestCase):
    def test_identifiable_initial_misroute_does_not_replace_successful_fallback(self):
        routed = result(policy="rule", model="model:b", success=1)
        routed.update(initial_model="model:a", fallback=True, fallback_reason="parse_failure")
        rows = fixed_pair(a_success=0, b_success=1) + [routed]
        before = deepcopy(rows)
        diagnostic = routing_diagnostics(rows)
        self.assertEqual(diagnostic["identifiable_misroutes"], 1)
        item = diagnostic["attempts"][0]
        self.assertEqual(item["actual_task_success"], 1)
        self.assertEqual(item["classification"], "identifiable_misroute")
        self.assertEqual(rows, before)
        summary = summarize_results(rows, plan(*rows))
        self.assertEqual(summary["policies"]["rule"]["task_successes"], 1)

    def test_missing_fixed_evidence_and_wrong_repeat_remain_unknown(self):
        routed = result(policy="rule", model="model:a", success=0)
        for rows in ([routed], [fixed_pair(a_success=0)[0], routed],
                     [fixed_pair(a_success=0)[0], result(policy="fixed_b", model="model:b", repeat=1), routed]):
            with self.subTest(fixed=len(rows) - 1):
                diagnostic = routing_diagnostics(rows)
                self.assertEqual(diagnostic["unknown_attempts"], 1)
                self.assertEqual(diagnostic["identifiable_misroutes"], 0)

    def test_both_fixed_failed_is_not_a_wrong_choice(self):
        rows = fixed_pair(a_success=0, b_success=0) + [result(policy="rule", model="model:a", success=0)]
        diagnostic = routing_diagnostics(rows)
        self.assertEqual(diagnostic["identifiable_misroutes"], 0)
        self.assertEqual(diagnostic["classification_counts"], {"both_fixed_failed": 1})

    def test_resource_escalation_requires_complete_comparable_evidence(self):
        rows = fixed_pair() + [result(policy="rule", model="model:a")]
        self.assertEqual(routing_diagnostics(rows)["avoidable_escalations"], 1)
        for field in ("provider_compute_ms", "output_tokens"):
            altered = deepcopy(rows)
            call = altered[1]["calls"][0]
            if field == "output_tokens":
                call["usage"][field] = None
            else:
                call[field] = None
            with self.subTest(field=field):
                diagnostic = routing_diagnostics(altered)
                self.assertEqual(diagnostic["avoidable_escalations"], 0)
                self.assertFalse(diagnostic["attempts"][0]["resource_comparable"])

    def test_resource_tradeoff_or_model_name_is_not_cost_evidence(self):
        # Higher compute but fewer tokens has no Pareto-dominant alternative.
        rows = fixed_pair(a_tokens=1, b_tokens=10, a_compute=20, b_compute=10)
        rows.append(result(policy="rule", model="model:a"))
        self.assertEqual(routing_diagnostics(rows)["avoidable_escalations"], 0)
        rows = fixed_pair(a_tokens=1, b_tokens=10, a_compute=5, b_compute=10)
        rows.append(result(policy="rule", model="model:a"))
        self.assertEqual(routing_diagnostics(rows)["avoidable_escalations"], 0)

    def test_incomplete_fixed_outcome_does_not_support_misroute(self):
        rows = fixed_pair(a_success=0, b_success=1)
        rows[1]["complete"] = False
        rows.append(result(policy="rule", model="model:a"))
        diagnostic = routing_diagnostics(rows)
        self.assertEqual(diagnostic["unknown_attempts"], 1)
        self.assertEqual(diagnostic["fixed_evidence_coverage"], 0)

    def test_missing_actual_selection_cannot_establish_avoidable_escalation(self):
        routed = result(policy="rule", model="model:a")
        routed["selected_model"] = None
        diagnostic = routing_diagnostics(fixed_pair() + [routed])
        self.assertEqual(diagnostic["avoidable_escalations"], 0)
        self.assertFalse(diagnostic["attempts"][0]["resource_comparable"])

    def test_counterfactual_success_does_not_replace_actual_route_failure(self):
        routed = result(policy="rule", model="model:a", success=0)
        diagnostic = routing_diagnostics(fixed_pair() + [routed])
        self.assertEqual(diagnostic["attempts"][0]["actual_task_success"], 0)
        self.assertEqual(diagnostic["attempts"][0]["actual_recorded_task_success"], 0)


class RoutingReportTests(unittest.TestCase):
    def test_chinese_report_keeps_frozen_rule_digests_and_limits(self):
        rows = fixed_pair() + [result(policy="rule", model="model:a")]
        manifest = {"experiment_id": "routing-test", "policy_version": "1.0",
                    "models": {"model:a": {"digest": "abc123"}, "model:b": {"digest": "xyz456"}},
                    "selection": {"selection_rule": "dev only, frozen before heldout"},
                    "limitations": ["同机顺序运行。"]}
        with tempfile.TemporaryDirectory() as temporary:
            path = write_routing_report(temporary, manifest, summarize_results(rows, plan(*rows)), compare_policies(rows))
            content = path.read_text(encoding="utf-8")
            self.assertIn("abc123", content)
            self.assertIn("dev only, frozen before heldout", content)
            self.assertIn("错路由证据覆盖", content)
            self.assertIn("API 成本为 null", content)
            self.assertIn("仅作探索", content)
            self.assertIn("Wilson", content)
            self.assertIn("同机顺序运行", content)


if __name__ == "__main__":
    unittest.main()
