"""Leakage and negative controls for pure, frozen routing policies."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.routing import (FAMILIES, POLICY_VERSION, build_capabilities, choose_models,
                           embedding_route, rule_family, rule_route)


def record(case_id, model, *, family="write", success=1, complete=True,
           duration=10, compute=5, input_tokens=20, output_tokens=10, split="development"):
    return {"case_id": case_id, "family": family, "split": split, "model": model,
            "task_success": success, "complete": complete, "duration_ms": duration,
            "provider_compute_ms": compute, "input_tokens": input_tokens, "output_tokens": output_tokens}


class CapabilityTests(unittest.TestCase):
    def test_heldout_or_unspecified_split_is_rejected_before_selection(self):
        for split in ("heldout", "test", None):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "development"):
                build_capabilities([record("c1", "model", split=split)], ["model"])

    def test_missing_and_incomplete_records_stay_in_shared_success_denominator(self):
        records = [record("c1", "a"), record("c2", "a", complete=False),
                   record("c1", "b", complete=False, duration=None, compute=None,
                          input_tokens=None, output_tokens=None)]
        capabilities = build_capabilities(records, ["b", "a"])
        a, b = (capabilities["models"][name]["overall"] for name in ("a", "b"))
        self.assertEqual(a["attempts"], 2)
        self.assertEqual(a["success_rate"], .5)
        self.assertEqual(a["incomplete_success_claims"], 1)
        self.assertEqual(b["attempts"], 2)
        self.assertEqual(b["recorded_attempts"], 1)
        self.assertEqual(b["success_rate"], 0)
        self.assertEqual(b["missing_case_ids"], ["c2"])
        self.assertEqual(b["case_coverage"], .5)
        self.assertIsNone(b["duration_ms"]["mean"])
        self.assertEqual(b["duration_ms"]["coverage"], 0)
        self.assertIsNone(b["input_tokens"]["total"])

    def test_stub_metrics_are_unknown_and_not_zero_cost(self):
        records = [record("c1", "a", complete=False, duration=None, compute=None,
                          input_tokens=None, output_tokens=None), record("c1", "b", compute=8)]
        result = choose_models(build_capabilities(records, ["a", "b"]))
        self.assertEqual(result["reference_model"], "b")
        self.assertEqual(result["economical_model"], "b")

    def test_observed_compute_selects_larger_named_model_without_size_assumption(self):
        records = [record("c1", "qwen3:4b", compute=50), record("c1", "qwen2.5vl:7b", compute=10)]
        result = choose_models(build_capabilities(records, ["qwen3:4b", "qwen2.5vl:7b"]))
        self.assertEqual(result["economical_model"], "qwen2.5vl:7b")
        self.assertEqual(result["family_models"]["write"], "qwen2.5vl:7b")
        self.assertIn("model size never", result["selection_rule"])
        self.assertIn("not an API-price", result["selection_rule"])

    def test_incomplete_compute_coverage_cannot_win_with_faster_partial_mean(self):
        records = [record("c1", "a", compute=100), record("c2", "a", compute=100),
                   record("c1", "b", compute=1), record("c2", "b", compute=None, duration=1)]
        result = choose_models(build_capabilities(records, ["a", "b"]))
        self.assertEqual(result["economical_model"], "a")
        self.assertIn("provider_compute_ms", result["selection_rule"])

    def test_duration_fallback_requires_full_timing_coverage(self):
        records = [record("c1", "a", compute=None, duration=50),
                   record("c1", "b", compute=None, duration=10)]
        result = choose_models(build_capabilities(records, ["a", "b"]))
        self.assertEqual(result["economical_model"], "b")
        self.assertIn("mean duration_ms", result["selection_rule"])

    def test_no_supported_economic_evidence_has_explicit_stable_fallback(self):
        records = [record("c1", name, complete=False, duration=None, compute=None,
                          input_tokens=None, output_tokens=None) for name in ("z", "a")]
        first = choose_models(build_capabilities(records, ["z", "a"]))
        second = choose_models(build_capabilities(list(reversed(records)), ["a", "z"]))
        self.assertEqual(first, second)
        self.assertEqual(first["economical_model"], "a")
        self.assertIn("unsupported", first["selection_rule"])

    def test_reference_success_precedes_cost_and_family_tie_prefers_economical(self):
        records = [record("c1", "a", family="write", compute=50),
                   record("c2", "a", family="append", compute=50),
                   record("c1", "b", family="write", compute=5),
                   record("c2", "b", family="append", success=0, compute=5)]
        capabilities = build_capabilities(records, ["b", "a"])
        result = choose_models(capabilities)
        self.assertEqual(result["reference_model"], "a")
        self.assertEqual(result["economical_model"], "b")
        self.assertEqual(result["family_models"]["write"], "b")
        self.assertEqual(result["family_models"]["append"], "a")
        self.assertEqual(result["family_models"]["clarify"], "b")
        self.assertIsNone(capabilities["models"]["a"]["families"]["clarify"]["success_rate"])

    def test_exact_ties_are_independent_of_input_order(self):
        records = [record("c1", name) for name in ("z", "a")]
        first = choose_models(build_capabilities(records, ["z", "a"]))
        second = choose_models(build_capabilities(list(reversed(records)), ["a", "z"]))
        self.assertEqual(first, second)
        self.assertEqual(first["reference_model"], "a")
        self.assertEqual(first["economical_model"], "a")

    def test_empty_records_are_unknown_not_perfect_success(self):
        capabilities = build_capabilities([], ["z", "a"])
        self.assertEqual(capabilities["policy_version"], POLICY_VERSION)
        self.assertIsNone(capabilities["models"]["a"]["overall"]["success_rate"])
        self.assertIn("unsupported", choose_models(capabilities)["selection_rule"])

    def test_duplicate_or_conflicting_case_labels_are_rejected(self):
        bad_records = [[record("c1", "a"), record("c1", "a")],
                       [record("c1", "a"), record("c1", "b", family="append")]]
        for records in bad_records:
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_capabilities(records, ["a", "b"])

    def test_invalid_metrics_and_success_are_not_coerced(self):
        variants = [{"duration_ms": float("nan")}, {"duration_ms": -1},
                    {"provider_compute_ms": float("inf")}, {"input_tokens": 1.5},
                    {"output_tokens": True}, {"task_success": True}, {"complete": 1},
                    {"family": "hidden_family"}, {"model": "undeclared"}]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                build_capabilities([{**record("c1", "a"), **variant}], ["a"])

    def test_selection_rejects_non_development_artifacts(self):
        capabilities = build_capabilities([record("c1", "a")], ["a"])
        capabilities["split"] = "heldout"
        with self.assertRaises(ValueError):
            choose_models(capabilities)


class RuleRoutingTests(unittest.TestCase):
    def test_basic_operations_ignore_quoted_operation_names(self):
        cases = [("Create note.txt containing exactly 'rename then append'.", "write"),
                 ("Rename report.txt to archive.txt.", "rename"),
                 ("Append exactly 'tail' to log.txt.", "append"),
                 ("给 log.txt 末尾写入内容。", "append")]
        for request, expected in cases:
            with self.subTest(request=request):
                self.assertEqual(rule_family({"request": request, "files": {}}), expected)

    def test_development_selection_uses_public_file_criteria(self):
        inputs = [{"request": "Return only the relative paths of regular files whose suffix is .txt and whose exact file content contains the text READY. Do not change any file.",
                   "files": {"a.txt": "READY\n", "b.txt": "WAIT\n", "c.md": "READY\n", "d.txt": "READY NOW\n"}},
                  {"request": "在这些文件中，筛出文件名以 .md 结尾且正文含“已审核”的文件，给出它们的相对路径。保持文件与目录原样。",
                   "files": {"资料/已审 文件.md": "已审核\n", "资料/待审.md": "待审核\n", "资料/明细.txt": "已审核\n", "说明.md": "已审核：通过\n"},
                   "directories": ["资料"]}]
        for public_input in inputs:
            with self.subTest(request=public_input["request"]):
                self.assertEqual(rule_family(public_input), "select")

    def test_development_ordered_operations_have_multi_step_priority(self):
        requests = ["Perform these steps in order: create directory bundle; create bundle/note.txt containing exactly \"seed\n\"; append exactly \"tail\n\" to that file; rename it to bundle/done.txt. Preserve control.dat and return the relative path of the finished file.",
                    "先创建目录“归档”，再将“输入/草稿.txt”重命名为“归档/稿件.txt”，然后在它末尾追加“确认\n”。不要改变“保留.ini”，回答处理完成的文件相对路径。"]
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(rule_family({"request": request, "files": {"control.dat": "keep"}}), "multi_step")

    def test_missing_target_or_rename_destination_requires_clarification(self):
        inputs = [{"request": "Write hello to a file.", "files": {}},
                  {"request": "Rename report.txt.", "files": {"report.txt": "draft"}},
                  {"request": "Append done to the file.", "files": {"a.txt": "a", "b.txt": "b"}},
                  {"request": "给这个文件追加完成。", "files": {"a.txt": "a", "b.txt": "b"}}]
        for public_input in inputs:
            with self.subTest(public_input=public_input):
                self.assertEqual(rule_family(public_input), "clarify")

    def test_unique_public_fixture_resolves_an_implicit_append_target(self):
        self.assertEqual(rule_family({"request": "Append done to the file.", "files": {"a.txt": "a"}}), "append")
        self.assertEqual(rule_family({"request": "Append done to the file.", "fixtures": [
            {"kind": "file", "path": "a.txt", "content_utf8": "a"}]}), "append")

    def test_hidden_fields_do_not_change_a_public_rule_decision(self):
        public_input = {"request": "Create note.txt containing hello.", "files": {}}
        with_hidden = {**public_input, "family": "rename", "expected": {"family": "clarify"},
                       "expected_output": {"oracle": object()}, "metadata": {"family": "append"}}
        self.assertEqual(rule_family(public_input), rule_family(with_hidden))
        self.assertEqual(rule_family(with_hidden), "write")

    def test_unknown_rule_is_explained_and_default_mapping_is_explicit(self):
        decision = rule_route({"request": "Summarize the project."}, {}, "default")
        self.assertEqual(set(decision), {"family", "model", "score", "reason", "prototype_id"})
        self.assertEqual(decision["family"], "multi_step")
        self.assertEqual(decision["model"], "default")
        self.assertIn("unknown", decision["reason"])
        self.assertIsNone(decision["score"])


class EmbeddingRoutingTests(unittest.TestCase):
    def test_nearest_cosine_prototype_maps_family_and_retains_identity(self):
        prototypes = [{"case_id": "d1", "split": "development", "family": "write", "vector": [1, 0]},
                      {"case_id": "d2", "split": "development", "family": "append", "vector": [0, 2]}]
        decision = embedding_route([0, 5], prototypes, {"write": "a", "append": "b"}, "fallback")
        self.assertEqual(decision["family"], "append")
        self.assertEqual(decision["model"], "b")
        self.assertEqual(decision["prototype_id"], "d2")
        self.assertAlmostEqual(decision["score"], 1)
        self.assertIn("development prototype d2", decision["reason"])

    def test_similarity_ties_are_stable_under_prototype_order(self):
        prototypes = [{"case_id": "z", "split": "development", "family": "append", "vector": [2, 0]},
                      {"case_id": "a", "split": "development", "family": "write", "vector": [1, 0]}]
        first = embedding_route([1, 0], prototypes, {"write": "w", "append": "a"}, "fallback")
        second = embedding_route([1, 0], list(reversed(prototypes)), {"write": "w", "append": "a"}, "fallback")
        self.assertEqual(first, second)
        self.assertEqual(first["prototype_id"], "a")
        self.assertIn("lexical", first["reason"])

    def test_heldout_prototypes_and_duplicate_ids_are_rejected(self):
        prototype = {"case_id": "d1", "split": "development", "family": "write", "vector": [1, 0]}
        for prototypes in ([{**prototype, "split": "heldout"}], [prototype, deepcopy(prototype)]):
            with self.subTest(prototypes=prototypes), self.assertRaises(ValueError):
                embedding_route([1, 0], prototypes, {}, "fallback")

    def test_invalid_vectors_cannot_silently_produce_default_decisions(self):
        prototype = {"case_id": "d1", "split": "development", "family": "write", "vector": [1, 0]}
        bad_vectors = [[], [0, 0], [1], [float("nan"), 0], [float("inf"), 0], [True, 0], ["1", 0]]
        for vector in bad_vectors:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                embedding_route(vector, [prototype], {}, "fallback")
        for vector in ([0, 0], [float("nan"), 1], [1, 0, 0]):
            with self.subTest(prototype_vector=vector), self.assertRaises(ValueError):
                embedding_route([1, 0], [{**prototype, "vector": vector}], {}, "fallback")

    def test_large_finite_vectors_normalize_without_overflow(self):
        prototypes = [{"case_id": "d1", "split": "development", "family": "write", "vector": [1e308, 1e308]}]
        decision = embedding_route([1e-308, 1e-308], prototypes, {"write": "model"}, "fallback")
        self.assertAlmostEqual(decision["score"], 1)

    def test_empty_prototypes_and_missing_family_mapping_have_explicit_fallbacks(self):
        empty = embedding_route([1, 0], [], {}, "fallback")
        self.assertIsNone(empty["prototype_id"])
        self.assertEqual(empty["model"], "fallback")
        self.assertIn("No development prototypes", empty["reason"])
        decision = embedding_route([1, 0], [{"case_id": "d1", "split": "development", "family": "write", "vector": [1, 0]}], {}, "fallback")
        self.assertEqual(decision["family"], "write")
        self.assertEqual(decision["model"], "fallback")
        self.assertIn("unavailable", decision["reason"])


if __name__ == "__main__":
    unittest.main()
