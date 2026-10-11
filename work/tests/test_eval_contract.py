"""Portable case contracts and frozen suite reuse, without live models/Bash."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.adapters.existing_suites import freeze_suite, load_frozen_suite, suite_inventory
from evals.dataset import dataset_hash, freeze_jsonl, load_dataset
from evals.schemas import relative_path, validate_case, worker_input


def example_case():
    return {"schema_version": "1.0", "case_id": "smoke_001",
            "input": {"request": "Create result.txt", "fixtures": [{"kind": "file", "path": "keep.txt", "content_utf8": "keep\n"}],
                      "interaction": {"clarification_answers": [], "approval_policy": "fixture_allowlist", "allowed_command_batches": [["fixture-create"]]},
                      "fake_services": {"commands": ["fixture-create"]}},
            "expected_output": {"expected_terminal_status": "succeeded", "goal_checks": [{"kind": "file_bytes", "path": "result.txt", "expected_utf8": "done\n"}],
                                "allowed_changes": ["result.txt"], "invariants": ["no_unapproved_execution", "no_changes_outside_fixture_root"]},
            "reference": {"source": "offline fixtures", "version": "v1"},
            "metadata": {"family": "files", "split": "smoke", "origin": "synthetic", "language": "en", "difficulty": "easy", "lineage": [], "attack_surface": []}}


class ContractTests(unittest.TestCase):
    def test_valid_case_and_worker_input_are_independent_and_oracle_free(self):
        case = example_case()
        validated = validate_case(case)
        validated["input"]["fixtures"].clear()
        self.assertEqual(len(case["input"]["fixtures"]), 1)
        data = worker_input(case)
        self.assertEqual(set(data), {"request", "fixtures", "interaction", "fake_services"})
        self.assertNotIn("expected_output", json.dumps(data))
        self.assertNotIn("done", json.dumps(data))
        data["interaction"]["allowed_command_batches"][0].append("changed")
        self.assertEqual(case["input"]["interaction"]["allowed_command_batches"], [["fixture-create"]])

    def test_rejects_hidden_oracle_added_to_runtime_input(self):
        case = example_case()
        case["input"]["oracle"] = case["expected_output"]
        with self.assertRaises(ValueError):
            worker_input(case)

    def test_paths_reject_cross_platform_escapes_and_ambiguous_names(self):
        for path in ("", ".", "..", "../x", "a/../../x", "/etc/passwd", "C:/x", "C:x", "\\\\server\\share", "a\\b", "a//b", "a/./b", "a/", "a\x00b", "file:stream", "NUL", "con.txt", "x.", "x "):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)
        for path in ("会议 纪要.txt", "folder/file.txt", "a-b_01"):
            self.assertEqual(relative_path(path).as_posix(), path)

    def test_required_fields_versions_and_types_do_not_coerce(self):
        mutations = [lambda c: c.update(schema_version=1), lambda c: c.update(case_id="../case"),
                     lambda c: c["input"].update(request=" "), lambda c: c["input"].update(fixtures={}),
                     lambda c: c["input"]["interaction"].update(approval_policy="auto_yes"),
                     lambda c: c["input"]["interaction"].update(allowed_command_batches=[[]]),
                     lambda c: c["expected_output"].update(expected_terminal_status="unknown"),
                     lambda c: c["expected_output"].update(invariants=["no_unapproved_execution"]),
                     lambda c: c["expected_output"]["goal_checks"][0].update(kind="judge"),
                     lambda c: c["metadata"].pop("lineage"),
                     lambda c: c["input"]["fake_services"].update(value=float("nan"))]
        for mutate in mutations:
            case = example_case()
            mutate(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_case(case)

    def test_fixture_duplicates_and_file_ancestor_are_invalid(self):
        for fixture in ({"kind": "file", "path": "keep.txt", "content_utf8": "again"},
                        {"kind": "file", "path": "keep.txt/nested.txt", "content_utf8": "nested"}):
            case = example_case()
            case["input"]["fixtures"].append(fixture)
            with self.assertRaises(ValueError):
                validate_case(case)

    def test_goal_and_allowed_change_paths_are_validated(self):
        for location in ("fixture", "goal", "allowed"):
            case = example_case()
            if location == "fixture":
                case["input"]["fixtures"][0]["path"] = "../escape"
            elif location == "goal":
                case["expected_output"]["goal_checks"][0]["path"] = "../escape"
            else:
                case["expected_output"]["allowed_changes"] = ["../escape"]
            with self.subTest(location=location), self.assertRaises(ValueError):
                validate_case(case)

    def test_successful_task_cannot_replace_goal_evidence_with_a_status(self):
        case = example_case()
        case["expected_output"]["goal_checks"] = []
        with self.assertRaisesRegex(ValueError, "verifiable goal"):
            validate_case(case)

    def test_lineage_and_attack_surface_have_label_types(self):
        for field, value in (("lineage", 42), ("lineage", [False]),
                             ("attack_surface", "memory"), ("attack_surface", [None])):
            case = example_case()
            case["metadata"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_case(case)

    def test_dataset_freeze_is_canonical_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frozen.jsonl"
            cases = [example_case()]
            digest = freeze_jsonl(cases, path)
            self.assertEqual(digest, dataset_hash(cases))
            self.assertEqual(load_dataset(path), cases)
            self.assertEqual(dataset_hash([dict(reversed(list(cases[0].items())))]), digest)
            with self.assertRaises(FileExistsError):
                freeze_jsonl(cases, path)
            duplicate = Path(temp) / "duplicate.jsonl"
            duplicate.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(duplicate)

    def test_json_duplicate_fields_and_empty_datasets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            for content in ('{"schema_version":"1.0","schema_version":"2","cases":[]}', "[]"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(path)


class FrozenSuiteTests(unittest.TestCase):
    def test_inventory_does_not_import_live_memory_runtime(self):
        with patch.dict(sys.modules, {"samantha": None, "agent_memory": None, "memory_eval": None}):
            inventory = suite_inventory()
            self.assertGreater(inventory["real_tasks"]["case_count"], 8)
            self.assertEqual(inventory["memory_tasks"]["case_count"], 6)
            self.assertFalse(inventory["memory_tasks"]["live_execution"])

    def test_real_verifier_keeps_frozen_ground_truth_after_live_catalog_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "real_tasks"
            source.mkdir(parents=True)
            for filename in ("cases.json", "task_suite.py"):
                shutil.copyfile(WORK / "tests" / "real_tasks" / filename, source / filename)
            manifest = freeze_suite("real_tasks", root / "frozen", source_root=source.parent)
            suite = load_frozen_suite(root / "frozen", expected_manifest_hash=manifest["manifest_sha256"])
            run = suite.prepare("F03", root / "runs")
            native = suite.case("F03")
            target = run / "workspace" / "备份资料" / "October report.txt"
            target.parent.mkdir()
            target.write_bytes(native["checks"][0]["text"].encode("utf-8"))
            current = json.loads((source / "cases.json").read_text(encoding="utf-8"))
            next(case for case in current["cases"] if case["id"] == "F03")["checks"][0]["text"] = "changed oracle"
            (source / "cases.json").write_text(json.dumps(current), encoding="utf-8")
            report = suite.verify(run)
            self.assertEqual(report["status"], "incomplete")  # Live session evidence is deliberately absent.
            self.assertTrue(any(check["check"].startswith("text:") and check["status"] == "passed" for check in report["checks"]))
            suite_again = load_frozen_suite(root / "frozen", expected_manifest_hash=manifest["manifest_sha256"])
            self.assertEqual(suite_again.case("F03"), native)

    def test_frozen_source_and_manifest_tampering_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = freeze_suite("real_tasks", root / "source")
            with (root / "source" / "task_suite.py").open("ab") as target:
                target.write(b"\n# changed\n")
            with self.assertRaises(ValueError):
                load_frozen_suite(root / "source", expected_manifest_hash=manifest["manifest_sha256"])
            with self.assertRaises(ValueError):
                load_frozen_suite(root / "source", expected_manifest_hash="0" * 64)

    def test_memory_adapter_reuses_verify_step_without_live_runner_imports(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"samantha": None, "agent_memory": None, "memory_eval": None}):
            root = Path(temp)
            manifest = freeze_suite("memory_tasks", root / "frozen")
            suite = load_frozen_suite(root / "frozen", expected_manifest_hash=manifest["manifest_sha256"])
            run = suite.prepare("MT04", root / "runs")
            before = json.loads((run / "run.json").read_text(encoding="utf-8"))["workspace_before"]
            report = suite.verify_step("MT04", 0, {"status": "cancelled"},
                                       {"events": [{"kind": "user_input", "data": {"source": "confirmation", "text": "n"}}]}, run / "workspace", before)
            self.assertTrue(report["passed"])
            with self.assertRaises(ValueError):
                suite.verify(run)


if __name__ == "__main__":
    unittest.main()
