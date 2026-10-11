"""Read-only artifact checks and durable, scoped object lineage."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent_memory import MemoryScope
from object_tracking import ObjectConflict, ObjectStore, ObjectStoreError, capture_before, verify_after
from object_tracking.verification import observe_path


SCOPE = MemoryScope("user-a", "project-a")


def effect(kind, path, *, content=None, destination=None, entity=None):
    return {"kind": kind, "path": str(path), "content": content,
            "destination": str(destination) if destination else None,
            "entity_id": entity["entity_id"] if entity else None,
            "entity_version": entity["version"] if entity else None}


class ObjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "objects.sqlite3"
        self.store = ObjectStore(self.database)
        self.addCleanup(self.store.close)
        self.sequence = 0

    def commit(self, proposal, before, *, store=None, scope=SCOPE, status="succeeded"):
        self.sequence += 1
        return (store or self.store).apply_verified(
            scope, proposal, verify_after(proposal, before), session_id=f"session-{self.sequence}",
            operation_id=f"operation-{self.sequence}", starting_directory=str(self.root),
            execution_status=status)

    def create(self, name="notes.txt", content="alpha\n", *, scope=SCOPE):
        path = self.root / name
        proposal = effect("create", path, content=content)
        before = capture_before(proposal)
        path.write_bytes(content.encode("utf-8"))
        return self.commit(proposal, before, scope=scope)

    def test_three_independent_calls_keep_identity_and_absolute_paths(self):
        first = self.create("中文 笔记.txt")
        self.store.close()
        reopened = ObjectStore(self.database)
        self.addCleanup(reopened.close)
        loaded = reopened.candidates(SCOPE)[0]
        self.assertEqual(loaded["entity_id"], first["entity_id"])
        destination = self.root / "中文 待办.txt"
        rename = effect("rename", Path(loaded["current_path"]), destination=destination, entity=loaded)
        before = capture_before(rename, entity=loaded)
        Path(loaded["current_path"]).rename(destination)
        second = self.commit(rename, before, store=reopened)
        reopened.close()
        third_store = ObjectStore(self.database)
        self.addCleanup(third_store.close)
        loaded = third_store.lookup(SCOPE, second["entity_id"])
        append = effect("append", destination, content="beta\n", entity=loaded)
        before = capture_before(append, entity=loaded)
        with destination.open("ab") as output:
            output.write(b"beta\n")
        third = self.commit(append, before, store=third_store)
        self.assertEqual(third["entity_id"], first["entity_id"])
        self.assertEqual(third["version"], 3)
        self.assertEqual(destination.read_bytes(), b"alpha\nbeta\n")
        self.assertFalse((self.root / "中文 笔记.txt").exists())
        self.assertEqual([row["operation_kind"] for row in third_store.events(SCOPE)],
                         ["create", "rename", "append"])
        self.assertEqual(third["starting_directory"], str(self.root))
        self.assertEqual(json.loads(third_store.latest_context(SCOPE))["focus"], first["entity_id"])
        self.assertNotIn("approved", third)

    def test_scopes_are_isolated_and_context_is_bounded(self):
        item = self.create()
        for scope in (MemoryScope("other", "project-a"), MemoryScope("user-a", "other")):
            self.assertEqual(self.store.candidates(scope), [])
            self.assertIsNone(self.store.lookup(scope, item["entity_id"]))
        text = self.store.latest_context(SCOPE, max_chars=90)
        self.assertLessEqual(len(text), 90)
        self.assertEqual(json.loads(text)["candidates"], [])
        self.assertTrue(json.loads(text)["truncated"])
        self.assertEqual(self.store.latest_context(SCOPE, max_chars=0), "")

    def test_explicit_path_lookup_has_no_recent_candidate_cutoff(self):
        first = self.create()
        self.create("latest.txt")
        self.assertNotEqual(self.store.candidates(SCOPE, limit=1)[0]["entity_id"], first["entity_id"])
        self.assertEqual(self.store.lookup_path(SCOPE, first["current_path"])["entity_id"],
                         first["entity_id"])
        self.assertIsNone(self.store.lookup_path(MemoryScope("other", "project-a"), first["current_path"]))
        Path(first["current_path"]).unlink()
        self.assertEqual(self.store.lookup_path(SCOPE, first["current_path"])["status"], "tombstone")

    def test_equal_clocks_use_commit_order_for_creation_events_and_candidates(self):
        with patch("object_tracking.store._now", return_value="2026-10-11T00:00:00Z"), \
                patch("object_tracking.store.uuid.uuid4", side_effect=[
                    SimpleNamespace(hex="0" * 32), SimpleNamespace(hex="f" * 32)]):
            first = self.create("first.txt")
            second = self.create("second.txt")
        self.assertEqual(self.store.first_creation(SCOPE)["entity"]["entity_id"], first["entity_id"])
        self.assertEqual(self.store.latest_creation(SCOPE)["entity"]["entity_id"], second["entity_id"])
        self.assertEqual([item["entity_id"] for item in self.store.candidates(SCOPE)],
                         [second["entity_id"], first["entity_id"]])
        self.assertEqual([item["entity_id"] for item in self.store.events(SCOPE)],
                         [first["entity_id"], second["entity_id"]])
        Path(first["current_path"]).unlink()
        self.assertEqual(self.store.first_creation(SCOPE)["entity"]["status"], "tombstone")
        self.assertEqual(self.store.latest_creation(SCOPE)["entity"]["entity_id"], second["entity_id"])
        self.assertIsNone(self.store.latest_creation(MemoryScope("other", "project-a")))

    def test_clock_rollback_and_updates_cannot_reorder_the_ledger(self):
        with patch("object_tracking.store._now", return_value="2026-10-11T00:00:02Z"):
            first = self.create("first.txt")
        with patch("object_tracking.store._now", return_value="2026-10-11T00:00:01Z"):
            second = self.create("second.txt")
        proposal = effect("append", Path(first["current_path"]), content="tail", entity=first)
        before = capture_before(proposal, entity=first)
        with Path(first["current_path"]).open("ab") as stream:
            stream.write(b"tail")
        with patch("object_tracking.store._now", return_value="2026-10-11T00:00:00Z"):
            updated = self.commit(proposal, before)
        self.assertEqual(self.store.candidates(SCOPE)[0]["entity_id"], first["entity_id"])
        self.assertEqual(self.store.latest_creation(SCOPE)["entity"]["entity_id"], second["entity_id"])
        self.assertEqual([item["operation_kind"] for item in self.store.events(SCOPE)],
                         ["create", "create", "append"])
        replay = self.store.apply_verified(SCOPE, proposal, verify_after(proposal, before),
            session_id="session-3", operation_id="operation-3", execution_status="succeeded")
        self.assertEqual(replay["version"], updated["version"])
        self.assertEqual(len(self.store.events(SCOPE)), 3)

    def test_schema_v1_commit_order_survives_independent_processes(self):
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from agent_memory import MemoryScope
from object_tracking import ObjectStore, capture_before, verify_after
import object_tracking.store as module
module._now = lambda: sys.argv[5]
path = Path(sys.argv[3])
effect = {"kind":"create", "path":str(path), "content":"child", "destination":None,
          "entity_id":None, "entity_version":None}
before = capture_before(effect)
path.write_bytes(b"child")
with ObjectStore(sys.argv[2]) as store:
    store.apply_verified(MemoryScope("user-a","project-a"), effect, verify_after(effect,before),
                         session_id=sys.argv[4], operation_id=sys.argv[4])
'''
        for label, timestamp in (("first", "2026-10-11T00:00:02Z"),
                                  ("second", "2026-10-11T00:00:02Z"),
                                  ("third", "2026-10-11T00:00:01Z")):
            subprocess.run([sys.executable, "-c", script, str(SRC), str(self.database),
                            str(self.root / (label + ".txt")), label, timestamp],
                           check=True, capture_output=True, timeout=30)
        self.store.close()
        with ObjectStore(self.database) as reopened:
            self.assertEqual(reopened.connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual([row["operation_id"] for row in reopened.events(SCOPE)],
                             ["first", "second", "third"])
            self.assertEqual(reopened.first_creation(SCOPE)["creation"]["operation_id"], "first")
            self.assertEqual(reopened.latest_creation(SCOPE)["creation"]["operation_id"], "third")
            self.assertEqual([Path(row["current_path"]).name for row in reopened.candidates(SCOPE)],
                             ["third.txt", "second.txt", "first.txt"])

    def test_clear_focus_preserves_verified_candidates_and_is_scoped(self):
        first = self.create()
        self.store.clear_focus(MemoryScope("other", "project-a"))
        self.assertEqual(json.loads(self.store.latest_context(SCOPE))["focus"], first["entity_id"])
        self.store.clear_focus(SCOPE)
        self.assertIsNone(json.loads(self.store.latest_context(SCOPE))["focus"])
        self.assertEqual(self.store.candidates(SCOPE)[0]["entity_id"], first["entity_id"])
        self.assertEqual(len(self.store.events(SCOPE)), 1)

    @unittest.skipUnless(os.name == "nt", "Case-insensitive paths are Windows-specific.")
    def test_case_variant_explicit_path_keeps_windows_object_identity(self):
        first = self.create("MixedCase.txt")
        variant = first["current_path"].lower()
        self.assertEqual(self.store.lookup_path(SCOPE, variant)["entity_id"], first["entity_id"])
        proposal = effect("append", Path(variant), content="beta", entity=first)
        before = capture_before(proposal, entity=first)
        self.assertEqual(before["status"], "ready")
        with Path(variant).open("ab") as output:
            output.write(b"beta")
        updated = self.commit(proposal, before)
        self.assertEqual(updated["entity_id"], first["entity_id"])
        self.assertEqual(updated["version"], 2)
        self.assertEqual(len(self.store.candidates(SCOPE)), 1)

    def test_reimport_is_idempotent_and_changed_effect_is_rejected(self):
        proposal = effect("create", self.root / "notes.txt", content="alpha")
        before = capture_before(proposal)
        Path(proposal["path"]).write_bytes(b"alpha")
        verification = verify_after(proposal, before)
        arguments = {"session_id": "s", "operation_id": "o"}
        first = self.store.apply_verified(SCOPE, proposal, verification, **arguments)
        replay = self.store.apply_verified(SCOPE, proposal, verification, **arguments)
        self.assertEqual(replay["version"], 1)
        self.assertEqual(replay["entity_id"], first["entity_id"])
        self.assertEqual(len(self.store.events(SCOPE)), 1)
        changed = effect("create", self.root / "other.txt", content="beta")
        before = capture_before(changed)
        Path(changed["path"]).write_bytes(b"beta")
        with self.assertRaisesRegex(ObjectConflict, "already imported"):
            self.store.apply_verified(SCOPE, changed, verify_after(changed, before), **arguments)
        self.assertEqual(len(self.store.events(SCOPE)), 1)

    def test_failed_batch_records_only_actual_verified_effect(self):
        proposal = effect("create", self.root / "notes.txt", content="actual")
        before = capture_before(proposal)
        self.assertIsNone(self.commit(proposal, before, status="failed"))
        Path(proposal["path"]).write_bytes(b"actual")
        item = self.commit(proposal, before, status="failed")
        self.assertEqual(item["status"], "active")
        self.assertEqual(self.store.events(SCOPE)[0]["execution_status"], "failed")

    def test_same_path_recreation_keeps_old_tombstone_and_new_id(self):
        first = self.create()
        path = Path(first["current_path"])
        path.unlink()
        self.assertEqual(self.store.candidates(SCOPE), [])
        self.assertEqual(self.store.lookup(SCOPE, first["entity_id"])["status"], "tombstone")
        second = self.create(content="replacement")
        self.assertNotEqual(first["entity_id"], second["entity_id"])
        self.assertEqual(self.store.lookup(SCOPE, first["entity_id"])["version"], 2)
        context = json.loads(self.store.latest_context(SCOPE))
        self.assertEqual(context["focus"], second["entity_id"])
        self.assertEqual(context["stale"][0]["entity_id"], first["entity_id"])

    def test_recreation_without_intermediate_lookup_still_has_new_identity(self):
        first = self.create()
        Path(first["current_path"]).unlink()
        second = self.create(content="replacement")
        self.assertNotEqual(first["entity_id"], second["entity_id"])
        self.assertEqual(self.store.lookup(SCOPE, first["entity_id"])["status"], "tombstone")

    def test_external_content_change_and_replacement_are_stale(self):
        first = self.create()
        path = Path(first["current_path"])
        path.write_bytes(b"externally changed")
        self.assertEqual(self.store.candidates(SCOPE), [])
        stale = self.store.lookup(SCOPE, first["entity_id"])
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["version"], 2)
        self.assertEqual(self.store.lookup(SCOPE, first["entity_id"])["version"], 2)
        another = self.create("another.txt")
        replacement = self.root / "replacement.txt"
        replacement.write_bytes(b"alpha\n")
        replacement.replace(another["current_path"])
        self.assertEqual(self.store.lookup(SCOPE, another["entity_id"])["status"], "stale")

    def test_concurrent_same_version_cannot_overwrite_current_verified_state(self):
        first = self.create()
        other_store = ObjectStore(self.database)
        self.addCleanup(other_store.close)
        proposal = effect("append", Path(first["current_path"]), content="", entity=first)
        before = capture_before(proposal, entity=first)
        evidence = verify_after(proposal, before)
        self.store.apply_verified(SCOPE, proposal, evidence, session_id="s-a", operation_id="a")
        with self.assertRaisesRegex(ObjectConflict, "version changed"):
            other_store.apply_verified(SCOPE, proposal, evidence, session_id="s-b", operation_id="b")
        self.assertEqual(other_store.lookup(SCOPE, first["entity_id"])["version"], 2)
        self.assertEqual(len(other_store.events(SCOPE)), 2)

    def test_persistence_failure_does_not_touch_observed_file_or_commit_partial_state(self):
        proposal = effect("create", self.root / "notes.txt", content="alpha")
        before = capture_before(proposal)
        Path(proposal["path"]).write_bytes(b"alpha")
        self.store.connection.execute("CREATE TRIGGER reject_object BEFORE INSERT ON object_entities "
                                      "BEGIN SELECT RAISE(ABORT, 'test write failure'); END")
        with self.assertRaises(ObjectStoreError):
            self.commit(proposal, before)
        self.assertEqual(Path(proposal["path"]).read_bytes(), b"alpha")
        self.assertEqual(self.store.candidates(SCOPE), [])
        self.assertEqual(self.store.events(SCOPE), [])


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_create_requires_absence_and_exact_utf8_bytes(self):
        path = self.root / "file.txt"
        proposal = effect("create", path, content="中文\n")
        before = capture_before(proposal)
        self.assertEqual(before["status"], "ready")
        path.write_bytes("中文".encode("utf-8"))
        self.assertEqual(verify_after(proposal, before)["status"], "unverified")
        path.write_bytes("中文\n".encode("utf-8"))
        self.assertEqual(verify_after(proposal, before)["status"], "verified")
        self.assertEqual(capture_before(proposal)["status"], "unverified")
        self.assertNotIn("bytes", json.dumps(before))

    def test_append_requires_exact_baseline_suffix_once_and_same_identity(self):
        path = self.root / "file.txt"
        path.write_bytes(b"alpha\n")
        proposal = effect("append", path, content="beta\n")
        before = capture_before(proposal)
        path.write_bytes(b"beta\n")
        self.assertEqual(verify_after(proposal, before)["status"], "unverified")
        path.write_bytes(b"alpha\nbeta\n")
        self.assertEqual(verify_after(proposal, before)["status"], "verified")
        with path.open("ab") as output:
            output.write(b"beta\n")
        self.assertEqual(verify_after(proposal, before)["status"], "unverified")
        replacement = self.root / "replacement.txt"
        replacement.write_bytes(b"alpha\nbeta\n")
        replacement.replace(path)
        self.assertEqual(verify_after(proposal, before)["status"], "unverified")

    def test_rename_requires_identity_destination_absence_and_old_path_absence(self):
        path, destination = self.root / "source.txt", self.root / "destination.txt"
        path.write_bytes(b"alpha")
        proposal = effect("rename", path, destination=destination)
        before = capture_before(proposal)
        shutil.copyfile(path, destination)
        self.assertEqual(verify_after(proposal, before)["status"], "unverified")
        self.assertEqual(capture_before(proposal)["status"], "unverified")
        path.unlink()
        copied = verify_after(proposal, before)
        self.assertEqual(copied["status"], "unverified")
        destination.unlink()
        path.write_bytes(b"alpha")
        before = capture_before(proposal)
        path.rename(destination)
        self.assertEqual(verify_after(proposal, before)["status"], "verified")

    def test_missing_directories_oversized_files_and_expected_append_are_conservative(self):
        directory = effect("append", self.root, content="x")
        self.assertEqual(capture_before(directory)["status"], "unverified")
        path = self.root / "file.txt"
        path.write_bytes(b"abcd")
        self.assertEqual(observe_path(str(path), max_bytes=3)["status"], "unverified")
        self.assertEqual(capture_before(effect("append", path, content="ef"), max_bytes=5)["status"],
                         "unverified")
        self.assertEqual(capture_before(effect("create", self.root / "new", content="abcdef"),
                                       max_bytes=5)["status"], "unverified")
        with self.assertRaisesRegex(ValueError, "absolute"):
            capture_before(effect("create", Path("relative.txt"), content="x"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            capture_before(effect("copy", path, destination=self.root / "copy"))

    def test_leaf_and_parent_symlinks_are_not_followed(self):
        path = self.root / "file.txt"
        path.write_bytes(b"alpha")
        link = self.root / "link"
        try:
            link.symlink_to(path)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symlinks requires platform privileges.")
        self.assertEqual(observe_path(str(link))["status"], "unverified")
        self.assertEqual(capture_before(effect("create", link, content="alpha"))["status"], "unverified")
        parent = self.root / "parent"
        parent.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(observe_path(str(parent / "file.txt"))["status"], "unverified")
        self.assertEqual(observe_path(str(parent / "nonexistent"))["status"], "unverified")

    def test_reparse_point_is_rejected_before_open_without_platform_privileges(self):
        path = self.root / "file.txt"
        path.write_bytes(b"alpha")
        original_lstat = os.lstat
        reparse = SimpleNamespace(st_mode=original_lstat(path).st_mode, st_file_attributes=0x400)
        with patch("object_tracking.verification.os.lstat", side_effect=lambda value:
                   reparse if Path(value) == path else original_lstat(value)), \
                patch("object_tracking.verification.os.open") as opened:
            observed = observe_path(str(path))
            self.assertEqual(observed["status"], "unverified")
            self.assertIn("reparse", observed["reason"])
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
