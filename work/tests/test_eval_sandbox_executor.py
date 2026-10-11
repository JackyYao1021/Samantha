"""Controller copyback checks using saved worker facts, never live Docker/Bash."""

import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.harness import snapshot
from evals.sandbox import SandboxUnavailable
from evals.sandbox_executor import SandboxExecutor, restore_snapshot
from evals.scorers.trajectory import score_recovery


def factual(files, directories=()):
    entries = {name: {"kind": "directory"} for name in directories}
    for name, data in files.items():
        entries[name] = {"kind": "file", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "content_base64": base64.b64encode(data).decode("ascii")}
    return {"complete": True, "entries": entries}


def outcome(before, after, *, complete=True, success=True):
    return {"schema_version": "1.0", "complete": complete, "success": success,
            "runtime_status": "succeeded" if success else "failed", "output": "saved worker facts",
            "output_truncated": False, "returncode": 0 if success else 1,
            "before_snapshot": before, "after_snapshot": after, "errors": []}


class SandboxExecutorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "fixture"
        self.root.mkdir()
        self.outside = self.parent / "outside.txt"
        self.outside.write_bytes(b"outside canary")
        self.executor = SandboxExecutor({"request": "Fixture task", "fixtures": [], "interaction": {}},
                                        self.root, enabled=True)

    def test_exact_commands_and_binary_file_state_are_sent_only_as_public_bytes(self):
        data = b"\xef\xbb\xbfalpha\r\n\x00\xff"
        (self.root / "notes.txt").write_bytes(data)
        commands = ["printf '%s' '$(container-only)' >> notes.txt"]
        before = snapshot(self.root)
        appended = factual({"notes.txt": data + b"beta\n"})
        renamed = factual({"final.txt": data + b"beta\n"})
        with patch.object(self.executor.sandbox, "run", side_effect=[outcome(before, appended),
                                                                   outcome(appended, renamed)]) as run:
            first = self.executor(commands)
            second = self.executor.execute(["mv -- notes.txt final.txt"])
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertEqual((self.root / "final.txt").read_bytes(), data + b"beta\n")
        self.assertFalse((self.root / "notes.txt").exists())
        self.assertEqual(run.call_args_list[0].args[0]["commands"], commands)
        self.assertEqual(set(run.call_args.args[0]), {"fixtures", "commands"})
        self.assertEqual(run.call_args.args[0]["fixtures"][0]["content_base64"],
                         appended["entries"]["notes.txt"]["content_base64"])
        self.assertEqual(self.executor.tool_calls, 2)
        self.assertTrue(all(batch["copied_back"] for batch in self.executor.batches))
        self.assertEqual(self.outside.read_bytes(), b"outside canary")

    def test_incomplete_result_retains_facts_and_never_copies_back(self):
        before = snapshot(self.root)
        after = factual({"new.txt": b"must not copy"})
        with patch.object(self.executor.sandbox, "run", return_value=outcome(before, after, complete=False)):
            with self.assertRaisesRegex(SandboxUnavailable, "Incomplete"):
                self.executor(["touch new.txt"])
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.executor.batches[0]["copied_back"])
        self.assertEqual(self.executor.batches[0]["sandbox_result"]["after_snapshot"], after)

    def test_complete_failure_preserves_verified_partial_file_effects(self):
        before = snapshot(self.root)
        after = factual({"partial.txt": b"partial"})
        with patch.object(self.executor.sandbox, "run", return_value=outcome(before, after, success=False)):
            result = self.executor(["printf partial > partial.txt; false"])
        self.assertFalse(result["success"])
        self.assertEqual((self.root / "partial.txt").read_bytes(), b"partial")
        self.assertTrue(self.executor.batches[0]["copied_back"])
        # Arbitrary Bash snapshots demonstrate file changes, not completion of
        # independently identified effects. Absence remains unknown to replay
        # scoring; an empty completed_effect_ids list would falsely assert none.
        self.assertNotIn("completed_effect_ids", result)
        self.assertNotEqual(result["before_snapshot"], result["after_snapshot"])
        evidence = {"complete": True, "runtime_status": "succeeded",
                    "before_snapshot": before, "after_snapshot": after,
                    "trajectory": [
                        {"kind": "operation_started", "data": {"name": "commands.execute",
                            "arguments": {"args": [["printf partial > partial.txt; false"]]}}},
                        {"kind": "operation_finished", "data": {"name": "commands.execute",
                            "status": "failed", "result": result}},
                        {"kind": "operation_started", "data": {"name": "commands.execute",
                            "arguments": {"args": [["retry using different Bash syntax"]]}}},
                        {"kind": "operation_finished", "data": {"name": "commands.execute",
                            "status": "succeeded", "result": {"success": True}}},
                        {"kind": "session_finished", "data": {"status": "succeeded"}}]}
        self.assertIsNone(score_recovery(evidence)["no_replay"])
        evidence["trajectory"][2]["data"]["arguments"]["args"] = [["printf partial > partial.txt; false"]]
        self.assertFalse(score_recovery(evidence)["no_replay"])

    def test_invalid_scope_digest_and_initial_state_fail_before_any_copyback(self):
        (self.root / "canary.txt").write_bytes(b"original")
        before = snapshot(self.root)
        candidates = [factual({"../outside.txt": b"overwrite"}), factual({"/absolute": b"overwrite"}),
                      factual({"new.txt": b"new"})]
        candidates[-1]["entries"]["new.txt"]["sha256"] = "0" * 64
        for after in candidates:
            with patch.object(self.executor.sandbox, "run", return_value=outcome(before, after)):
                with self.assertRaises(ValueError):
                    self.executor(["untrusted command text"])
            self.assertEqual(snapshot(self.root), before)
            self.assertEqual(self.outside.read_bytes(), b"outside canary")
        with patch.object(self.executor.sandbox, "run", return_value=outcome(factual({}), factual({}))):
            with self.assertRaisesRegex(SandboxUnavailable, "initial bytes differ"):
                self.executor(["true"])
        self.assertEqual(snapshot(self.root), before)

    def test_concurrent_controller_change_is_not_overwritten(self):
        (self.root / "canary.txt").write_bytes(b"original")
        before = snapshot(self.root)
        def changed(_):
            (self.root / "canary.txt").write_bytes(b"independent controller change")
            return outcome(before, factual({"canary.txt": b"worker result"}))
        with patch.object(self.executor.sandbox, "run", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.executor(["command"])
        self.assertEqual((self.root / "canary.txt").read_bytes(), b"independent controller change")

    def test_nested_directory_and_file_type_changes_restore_complete_evidence(self):
        (self.root / "old").mkdir()
        (self.root / "old" / "child").write_bytes(b"old bytes")
        (self.root / "become-dir").write_bytes(b"previous file")
        before = snapshot(self.root)
        after = factual({"old": b"now a file", "become-dir/new": b"new bytes"}, ["become-dir"])
        restore_snapshot(self.root, before, after)
        self.assertEqual(snapshot(self.root), after)
        self.assertEqual(self.outside.read_bytes(), b"outside canary")
        incomplete = deepcopy(after)
        del incomplete["entries"]["become-dir"]
        with self.assertRaisesRegex(ValueError, "ancestor"):
            restore_snapshot(self.root, after, incomplete)
        self.assertEqual(snapshot(self.root), after)

    def test_disabled_executor_and_oracle_input_cannot_start_host_processes(self):
        executor = SandboxExecutor({"request": "Task"}, self.root)
        with patch("evals.sandbox.subprocess.Popen") as started:
            with self.assertRaisesRegex(SandboxUnavailable, "disabled"):
                executor(["touch file.txt"])
            started.assert_not_called()
        with self.assertRaisesRegex(ValueError, "public case input only"):
            SandboxExecutor({"request": "Task", "expected_output": {}}, self.root)
        self.assertEqual(snapshot(self.root), factual({}))


if __name__ == "__main__":
    unittest.main()
