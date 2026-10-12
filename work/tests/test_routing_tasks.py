"""Byte-level, negative-control and path-safety tests; no model requests."""

from collections import Counter
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

from evals.routing_tasks import (
    FAMILIES, MAX_CONTENT_BYTES, OUTPUT_SCHEMA, PayloadValidationError,
    _preflight, evaluate_response, load_cases, public_input, validate_payload,
)


def action(tool, path, content="", destination=""):
    return {"tool": tool, "path": path, "content": content, "destination": destination}


def response(actions=None, answer=None, *, status="completed", question=""):
    return {"status": status, "actions": actions or [], "answer": answer or [], "question": question}


class RoutingTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case["case_id"]: case for case in load_cases()}

    def case(self, identifier="RD-write-01"):
        return deepcopy(self.cases[identifier])

    def assert_unchanged_failure(self, case, payload, *, schema_valid=None):
        report = evaluate_response(case, payload)
        self.assertEqual(report["task_success"], 0)
        self.assertEqual(report["before_snapshot"], report["after_snapshot"])
        if schema_valid is not None:
            self.assertEqual(report["schema_valid"], schema_valid)
        return report

    def test_dataset_has_balanced_independent_development_and_heldout_cases(self):
        cases = list(self.cases.values())
        self.assertEqual(len(cases), 24)
        counts = Counter((case["split"], case["family"]) for case in cases)
        self.assertEqual(counts, Counter({(split, family): 2
            for split in ("development", "heldout") for family in FAMILIES}))
        for family in FAMILIES:
            development = [case for case in cases if case["family"] == family and case["split"] == "development"]
            heldout = [case for case in cases if case["family"] == family and case["split"] == "heldout"]
            self.assertFalse({case["input"]["request"] for case in development}
                             & {case["input"]["request"] for case in heldout})
            self.assertFalse({path for case in development for path in case["expected"]["files"]}
                             & {path for case in heldout for path in case["expected"]["files"]})
            for subset in (development, heldout):
                self.assertEqual(sum(any(ord(char) > 127 for char in case["input"]["request"])
                                     for case in subset), 1)

    def test_public_input_excludes_labels_and_oracles_and_is_independent(self):
        for case in self.cases.values():
            public = public_input(case)
            self.assertEqual(set(public), {"request", "files", "directories"})
            original = deepcopy(case)
            public["files"]["probe.txt"] = "changed"
            public["directories"].append("probe")
            self.assertEqual(case, original)

    def test_load_rejects_duplicate_ids_and_json_object_keys(self):
        case = self.case()
        line = json.dumps(case, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "cases.jsonl"
            for content in (line + "\n" + line, line[:-1] + ',"split":"heldout"}'):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content[:40]), self.assertRaises(ValueError):
                    load_cases(path)

    def test_invalid_oracle_raises_instead_of_becoming_model_failure(self):
        cases = []
        case = self.case()
        case["expected"]["answer"] = ["not-in-final.txt"]
        cases.append(case)
        case = self.case()
        case["expected"]["files"]["../outside.txt"] = "outside"
        cases.append(case)
        case = self.case()
        case["expected"]["question_terms_any"] = ["file"]
        cases.append(case)
        case = self.case()
        case["expected"]["files"] = {"missing-parent/note.txt": "note"}
        cases.append(case)
        for case in cases:
            with self.subTest(expected=case["expected"]), self.assertRaises(ValueError):
                evaluate_response(case, response())

    def test_validate_payload_is_oracle_free_and_returns_an_independent_copy(self):
        payload = response([action("write", "arbitrary.txt", "hello")])
        with patch("evals.routing_tasks._execute", side_effect=AssertionError("must not execute")), \
                patch("evals.routing_tasks._validate_case", side_effect=AssertionError("must not read oracle")):
            canonical = validate_payload(payload)
        canonical["actions"][0]["content"] = "new"
        self.assertEqual(payload["actions"][0]["content"], "hello")
        self.assertEqual(set(OUTPUT_SCHEMA["properties"]), set(payload))

    def test_correct_write_checks_unicode_and_exact_newline_bytes(self):
        case = self.case("RD-write-02")
        report = evaluate_response(case, response([action("write", "日常/待办 清单.txt", "买牛奶\n散步\n")]))
        self.assertEqual(report["task_success"], 1)
        self.assertIsNone(report["error_category"])
        self.assertEqual(report["after_snapshot"]["entries"]["日常/待办 清单.txt"]["size"], len("买牛奶\n散步\n".encode("utf-8")))

    def test_completed_status_does_not_pass_without_required_artifact(self):
        report = self.assert_unchanged_failure(self.case(), response(), schema_valid=True)
        self.assertEqual(report["error_category"], "file_inventory")

    def test_wrong_bytes_and_modified_unrelated_files_fail(self):
        for content in ("Hello Samantha!", "Hello Samantha!\r\n", "wrong\n"):
            report = evaluate_response(self.case(), response([action("write", "hello.txt", content)]))
            self.assertEqual(report["task_success"], 0)
            self.assertEqual(report["error_category"], "file_content")
        report = evaluate_response(self.case(), response([
            action("write", "hello.txt", "Hello Samantha!\n"), action("write", "scratch.old", "changed")]))
        self.assertEqual(report["task_success"], 0)
        self.assertEqual(report["error_category"], "file_content")

    def test_extra_files_and_directories_fail_complete_inventory_check(self):
        for extra, category in ((action("write", "unexpected.txt", "extra"), "file_inventory"),
                                (action("mkdir", "unexpected"), "directory_inventory")):
            report = evaluate_response(self.case(), response([action("write", "hello.txt", "Hello Samantha!\n"), extra]))
            self.assertEqual(report["task_success"], 0)
            self.assertEqual(report["error_category"], category)

    def test_select_requires_exact_path_set_and_no_file_changes(self):
        case = self.case("RD-select-01")
        self.assertEqual(evaluate_response(case, response(answer=["d.txt", "a.txt"]))["task_success"], 1)
        for answer in (["a.txt"], ["a.txt", "d.txt", "c.md"], []):
            report = self.assert_unchanged_failure(case, response(answer=answer), schema_valid=True)
            self.assertEqual(report["error_category"], "answer_mismatch")
        report = evaluate_response(case, response([action("append", "a.txt", "extra")], ["a.txt", "d.txt"]))
        self.assertEqual(report["task_success"], 0)
        self.assertEqual(report["error_category"], "file_content")

    def test_clarify_requires_relevant_question_and_preserves_fixture(self):
        case = self.case("RD-clarify-01")
        correct = response(status="clarify", question="Which target file should I append to?")
        report = evaluate_response(case, correct)
        self.assertEqual(report["task_success"], 1)
        self.assertEqual(report["before_snapshot"], report["after_snapshot"])
        report = self.assert_unchanged_failure(case, response(status="clarify", question="Can you confirm?"), schema_valid=True)
        self.assertEqual(report["error_category"], "clarification")
        self.assert_unchanged_failure(case, response(status="clarify", question="Which file?",
                                                    actions=[action("write", "first.txt", "changed")]), schema_valid=False)

    def test_clarification_matching_is_unicode_and_case_insensitive(self):
        report = evaluate_response(self.case("RD-clarify-02"), response(status="clarify", question="请问要操作哪个文件？"))
        self.assertEqual(report["task_success"], 1)
        report = evaluate_response(self.case("RD-clarify-01"), response(status="clarify", question="Which FILE?"))
        self.assertEqual(report["task_success"], 1)

    def test_correct_multistep_uses_actual_sequential_file_effects(self):
        payload = response([
            action("mkdir", "bundle"), action("write", "bundle/note.txt", "seed\n"),
            action("append", "bundle/note.txt", "tail\n"),
            action("rename", "bundle/note.txt", destination="bundle/done.txt"),
        ], ["bundle/done.txt"])
        report = evaluate_response(self.case("RD-multi-01"), payload)
        self.assertEqual(report["task_success"], 1)
        self.assertNotIn("bundle/note.txt", report["after_snapshot"]["entries"])
        self.assertEqual(report["after_snapshot"]["entries"]["bundle/done.txt"]["size"], 10)

    def test_second_invalid_action_rejects_whole_batch_before_writing(self):
        first = action("write", "hello.txt", "Hello Samantha!\n")
        bad = action("write", "unexpected.txt", "bad")
        del bad["destination"]
        self.assert_unchanged_failure(self.case(), response([first, bad]), schema_valid=False)
        report = self.assert_unchanged_failure(self.case(), response([first, action("append", "missing.txt", "bad")]), schema_valid=True)
        self.assertEqual(report["error_category"], "action_precondition")

    def test_unsafe_paths_reject_whole_batch_without_effects(self):
        paths = ("../outside.txt", "/absolute.txt", "C:/outside.txt", "folder\\note.txt",
                 "CON.txt", "notes.txt:stream", "folder/../note.txt", "alias. ", "bad?.txt")
        for path in paths:
            payload = response([action("write", "hello.txt", "Hello Samantha!\n"), action("write", path, "bad")])
            with self.subTest(path=path):
                report = self.assert_unchanged_failure(self.case(), payload, schema_valid=False)
                self.assertEqual(report["error_category"], "path_safety")

    def test_symlink_parents_are_rejected_by_existing_harness_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                os.symlink(outside, root / "linked", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Host does not allow creating a test symlink")
            payload = validate_payload(response([action("write", "linked/note.txt", "bad")]))
            with self.assertRaises(ValueError):
                _preflight(root, payload, {"complete": True, "entries": {}})
            self.assertEqual(list(outside.iterdir()), [])

    def test_action_cap_utf8_byte_cap_and_unused_fields_are_enforced(self):
        invalid = [
            response([action("write", "note.txt", "a")] * 13),
            response([action("write", "note.txt", "中" * (MAX_CONTENT_BYTES // 3 + 1))]),
            response([action("mkdir", "folder", content="not empty")]),
            response([action("append", "scratch.old", "x", destination="unused.txt")]),
            response([action("bash", "scratch.old", "echo unsafe")]),
            response(answer=["scratch.old", "scratch.old"]),
            response(status="clarify"),
            {**response(), "expected": {}},
        ]
        for payload in invalid:
            with self.subTest(status=payload["status"], actions=len(payload["actions"])):
                self.assert_unchanged_failure(self.case(), payload, schema_valid=False)
                with self.assertRaises(PayloadValidationError):
                    validate_payload(payload)

    def test_rename_cannot_overwrite_an_unrelated_target(self):
        case = self.case("RD-rename-01")
        report = self.assert_unchanged_failure(case, response([
            action("write", "new.txt", "new"), action("rename", "draft.txt", destination="reference.md")]), schema_valid=True)
        self.assertEqual(report["error_category"], "action_precondition")

    def test_destination_paths_are_checked_before_any_action(self):
        payload = response([action("write", "new.txt", "new"),
                            action("rename", "draft.txt", destination="../outside.txt")])
        report = self.assert_unchanged_failure(self.case("RD-rename-01"), payload, schema_valid=False)
        self.assertEqual(report["error_category"], "path_safety")

    def test_directory_creation_is_required_before_write_to_new_parent(self):
        payload = response([action("write", "bundle/note.txt", "seed\n"), action("mkdir", "bundle")])
        self.assert_unchanged_failure(self.case("RD-multi-01"), payload, schema_valid=True)


if __name__ == "__main__":
    unittest.main()
