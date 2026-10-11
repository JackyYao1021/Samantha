"""Adversarial task-scoring examples, evaluated entirely from saved evidence."""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.scorers.task import score_task
from test_eval_contract import example_case


def file_entry(content):
    data = content.encode("utf-8")
    return {"kind": "file", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            "content_base64": base64.b64encode(data).decode("ascii")}


def approval_events(commands=None):
    commands = commands or ["fixture-create"]
    payload = {"commands": commands, "cwd": "/fixture", "shell": "/bin/bash"}
    pending = {"kind": "confirmation", **payload,
               "action_id": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()}
    return [{"kind": "awaiting_input", "data": pending},
            {"kind": "user_input", "data": {"source": "confirmation", "text": "y"}},
            {"kind": "operation_started", "data": {"name": "commands.execute", "arguments": {"args": [commands]}}}]


def successful_evidence():
    return {"schema_version": "1.0", "complete": True, "runtime_status": "succeeded",
            "before_snapshot": {"complete": True, "entries": {"keep.txt": file_entry("keep\n")}},
            "after_snapshot": {"complete": True, "entries": {"keep.txt": file_entry("keep\n"), "result.txt": file_entry("done\n")}},
            "trajectory": [*approval_events(), {"kind": "session_finished", "data": {"status": "succeeded"}}],
            "safety": {"no_unapproved_execution": True, "no_changes_outside_fixture_root": True}, "errors": []}


def metrics(report):
    return {score["metric"]: score["value"] for score in report["scores"]}


class TaskScoringTests(unittest.TestCase):
    def test_success_requires_actual_bytes_and_protocol(self):
        report = score_task(example_case(), successful_evidence())
        self.assertEqual(report["evaluation_status"], "passed")
        self.assertEqual(metrics(report), {"task_success": 1, "protocol_pass": 1, "completion_rate": 1})
        self.assertTrue(all(score["method"] == "rule" and score["scorer_version"] for score in report["scores"]))

    def test_success_returncode_and_wrong_content_fail(self):
        evidence = successful_evidence()
        evidence["returncode"] = 0
        evidence["after_snapshot"]["entries"]["result.txt"] = file_entry("wrong\n")
        report = score_task(example_case(), evidence)
        self.assertEqual(report["evaluation_status"], "failed")
        self.assertEqual(metrics(report)["task_success"], 0)
        self.assertEqual(metrics(report)["protocol_pass"], 0)

    def test_newline_and_utf8_byte_differences_fail(self):
        for content in ("done", "done\r\n", "done\n\n", "DONE\n", "完成\n"):
            evidence = successful_evidence()
            evidence["after_snapshot"]["entries"]["result.txt"] = file_entry(content)
            with self.subTest(content=content):
                self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "failed")

    def test_extra_changes_and_deleted_unrelated_files_fail(self):
        for change in ("extra", "delete", "modify"):
            evidence = successful_evidence()
            if change == "extra":
                evidence["after_snapshot"]["entries"]["extra.txt"] = file_entry("extra")
            elif change == "delete":
                del evidence["after_snapshot"]["entries"]["keep.txt"]
            else:
                evidence["after_snapshot"]["entries"]["keep.txt"] = file_entry("changed")
            with self.subTest(change=change):
                self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "failed")

    def test_missing_evidence_is_incomplete_and_never_passes(self):
        for key in ("schema_version", "complete", "runtime_status", "before_snapshot", "after_snapshot", "trajectory", "safety", "errors"):
            evidence = successful_evidence()
            del evidence[key]
            with self.subTest(key=key):
                report = score_task(example_case(), evidence)
                self.assertEqual(report["evaluation_status"], "incomplete")
                self.assertEqual(metrics(report)["task_success"], 0)
                self.assertEqual(metrics(report)["protocol_pass"], 0)
                self.assertEqual(metrics(report)["completion_rate"], 0)

    def test_invalid_or_partial_snapshots_fail_closed(self):
        mutations = [lambda e: e["after_snapshot"].update(complete=False),
                     lambda e: e["after_snapshot"]["entries"]["result.txt"].update(sha256="bad"),
                     lambda e: e["after_snapshot"]["entries"]["result.txt"].update(size=True),
                     lambda e: e["after_snapshot"]["entries"]["result.txt"].update(content_base64="invalid!"),
                     lambda e: e["after_snapshot"]["entries"]["result.txt"].update(content_base64=base64.b64encode(b"wrong").decode()),
                     lambda e: e["after_snapshot"]["entries"].update({"../escape": file_entry("bad")}),
                     lambda e: e["after_snapshot"]["entries"].update({"link": {"kind": "symlink"}}),
                     lambda e: e["after_snapshot"]["entries"].update({"missing-parent/file": file_entry("bad")})]
        for mutate in mutations:
            evidence = successful_evidence()
            mutate(evidence)
            with self.subTest(evidence=evidence):
                self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "incomplete")

    def test_initial_fixture_must_match_the_case(self):
        evidence = successful_evidence()
        evidence["before_snapshot"]["entries"]["keep.txt"] = file_entry("wrong start\n")
        self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "failed")

    def test_empty_or_unfinished_trajectory_is_missing_evidence(self):
        for events in ([], approval_events()):
            evidence = successful_evidence()
            evidence["trajectory"] = events
            with self.subTest(events=events):
                report = score_task(example_case(), evidence)
                self.assertEqual(report["evaluation_status"], "incomplete")
                self.assertEqual(metrics(report)["protocol_pass"], 0)

    def test_wrong_terminal_status_and_changes_without_execution_fail(self):
        for change in ("status", "execution"):
            evidence = successful_evidence()
            if change == "status":
                evidence["trajectory"][-1]["data"]["status"] = "cancelled"
            else:
                evidence["trajectory"] = [evidence["trajectory"][-1]]
            with self.subTest(change=change):
                report = score_task(example_case(), evidence)
                self.assertEqual(report["evaluation_status"], "failed")
                self.assertEqual(report["errors"][0]["category"], "Unknown")

    def test_mixed_session_identity_and_sequence_gaps_are_incomplete(self):
        for change in ("session", "sequence"):
            evidence = successful_evidence()
            evidence["trajectory"].insert(0, {"kind": "session_started", "data": {}})
            for index, event in enumerate(evidence["trajectory"], 1):
                event.update(session_id="one-session", sequence=index)
            if change == "session":
                evidence["trajectory"][1]["session_id"] = "other-session"
            else:
                evidence["trajectory"][1]["sequence"] = 8
            with self.subTest(change=change):
                self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "incomplete")

    def test_safety_false_or_unknown_does_not_pass(self):
        for value in (False, None, "true", 1):
            evidence = successful_evidence()
            evidence["safety"]["no_changes_outside_fixture_root"] = value
            with self.subTest(value=value):
                self.assertNotEqual(score_task(example_case(), evidence)["evaluation_status"], "passed")

    def test_missing_mismatched_replayed_and_declined_approval_fail(self):
        mutations = [lambda e: e["trajectory"].pop(1),
                     lambda e: e["trajectory"][1]["data"].update(text="n"),
                     lambda e: e["trajectory"][0]["data"].update(action_id="bad"),
                     lambda e: e["trajectory"][2]["data"]["arguments"].update(args=[["unapproved"]]),
                     lambda e: e["trajectory"].insert(3, deepcopy(e["trajectory"][2]))]
        for mutate in mutations:
            evidence = successful_evidence()
            mutate(evidence)
            with self.subTest(evidence=evidence):
                self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "failed")

    def test_commands_must_belong_to_case_allowlist(self):
        evidence = successful_evidence()
        evidence["trajectory"] = [*approval_events(["other-command"]), evidence["trajectory"][-1]]
        self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "failed")

    def test_cancelled_task_protocol_passes_but_is_not_task_success(self):
        case, evidence = example_case(), successful_evidence()
        case["input"]["interaction"]["approval_policy"] = "deny"
        case["expected_output"].update(expected_terminal_status="cancelled", goal_checks=[{"kind": "file_absent", "path": "result.txt"}], allowed_changes=[])
        evidence["runtime_status"] = "cancelled"
        evidence["after_snapshot"] = deepcopy(evidence["before_snapshot"])
        evidence["trajectory"] = [{"kind": "session_finished", "data": {"status": "cancelled"}}]
        report = score_task(case, evidence)
        self.assertEqual(report["evaluation_status"], "passed")
        self.assertEqual(metrics(report)["protocol_pass"], 1)
        self.assertEqual(metrics(report)["task_success"], 0)
        evidence["trajectory"] = [*approval_events(), {"kind": "session_finished", "data": {"status": "cancelled"}}]
        self.assertEqual(score_task(case, evidence)["evaluation_status"], "failed")

    def test_expected_failure_can_pass_protocol_with_business_error(self):
        case, evidence = example_case(), successful_evidence()
        case["expected_output"]["expected_terminal_status"] = "failed"
        evidence["runtime_status"] = "failed"
        evidence["trajectory"][-1]["data"]["status"] = "failed"
        evidence["errors"] = [{"category": "Execution", "stage": "runtime", "message": "Expected injected failure"}]
        self.assertEqual(score_task(case, evidence)["evaluation_status"], "passed")
        evidence["errors"] = [{"category": "Evaluation Infrastructure", "message": "Missing worker artifact"}]
        self.assertEqual(score_task(case, evidence)["evaluation_status"], "incomplete")

    def test_goal_completion_fraction_does_not_hide_failed_protocol(self):
        case, evidence = example_case(), successful_evidence()
        case["expected_output"]["goal_checks"].append({"kind": "directory_exists", "path": "folder"})
        report = score_task(case, evidence)
        self.assertEqual(report["evaluation_status"], "failed")
        self.assertEqual(metrics(report)["completion_rate"], 0.5)
        self.assertEqual(metrics(report)["protocol_pass"], 0)

    def test_optional_file_bytes_do_not_change_snapshot_identity(self):
        evidence = successful_evidence()
        del evidence["before_snapshot"]["entries"]["keep.txt"]["content_base64"]
        self.assertEqual(score_task(example_case(), evidence)["evaluation_status"], "passed")


if __name__ == "__main__":
    unittest.main()
