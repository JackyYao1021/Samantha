"""Read-only entity inspection against real verified SQLite records."""

from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import entities_cli
import samantha
from agent_memory import MemoryScope
from object_tracking.store import ObjectStore
from object_tracking.verification import capture_before, verify_after


SCOPE = MemoryScope("local", "project-a")


class EntitiesCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "objects # 中文.sqlite3"
        self.store = ObjectStore(self.database)
        self.addCleanup(self.store.close)

    def create(self, name, scope=SCOPE):
        path = self.root / name
        effect = {"kind": "create", "path": str(path), "destination": None,
                  "content": "alpha\n", "entity_id": None, "entity_version": None}
        before = capture_before(effect)
        path.write_bytes(b"alpha\n")
        return self.store.apply_verified(
            scope, effect, verify_after(effect, before),
            session_id=uuid.uuid4().hex, operation_id=uuid.uuid4().hex,
            starting_directory=str(self.root), execution_status="succeeded",
        )

    def run_cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = entities_cli.main(["--db-path", str(self.database), *args])
        return status, output.getvalue(), errors.getvalue()

    def read_json(self, *args):
        status, output, errors = self.run_cli(*args, "--json")
        self.assertEqual(status, 0, errors)
        self.assertEqual(errors, "")
        return json.loads(output)

    def snapshot(self):
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)) as db:
            return {table: db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    for table in ("object_entities", "object_events", "object_focus", "object_inspections")}

    def test_all_scopes_and_saved_statuses_are_visible_without_context_limits(self):
        expected = {self.create(f"file-{index}.txt")["entity_id"] for index in range(23)}
        other = self.create("other-user.txt", MemoryScope("other-user", "project-a"))
        expected.add(other["entity_id"])
        stale = self.create("stale.txt", MemoryScope("local", "project-b"))
        Path(stale["current_path"]).write_bytes(b"changed externally")
        self.store.lookup(MemoryScope("local", "project-b"), stale["entity_id"])
        expected.add(stale["entity_id"])
        missing = self.create("missing.txt", MemoryScope("local", "project-b"))
        Path(missing["current_path"]).unlink()
        self.store.lookup(MemoryScope("local", "project-b"), missing["entity_id"])
        expected.add(missing["entity_id"])

        with patch.dict(os.environ, {"SAMANTHA_OBJECT_USER": "irrelevant",
                                     "SAMANTHA_OBJECT_PROJECT": "irrelevant"}):
            data = self.read_json()
        self.assertEqual(data["count"], 26)
        self.assertEqual({row["entity_id"] for row in data["entities"]}, expected)
        self.assertEqual({row["status"] for row in data["entities"]},
                         {"active", "stale", "tombstone"})
        self.assertTrue(data["database_exists"])
        self.assertEqual(data["database"], str(self.database))
        self.assertEqual(data["filters"], {"user_id": None, "project_id": None, "status": None})

    def test_filters_combine_exact_user_project_and_status(self):
        active = self.create("active.txt")
        stale = self.create("stale.txt")
        Path(stale["current_path"]).write_bytes(b"changed")
        self.store.lookup(SCOPE, stale["entity_id"])
        foreign = self.create("foreign.txt", MemoryScope("other", "project-a"))
        elsewhere = self.create("elsewhere.txt", MemoryScope("local", "project-b"))
        cases = [
            (("--user", "local", "--project", "project-a", "--status", "active"), {active["entity_id"]}),
            (("--status", "stale"), {stale["entity_id"]}),
            (("--project", "project-a"), {active["entity_id"], stale["entity_id"], foreign["entity_id"]}),
            (("--user", "local"), {active["entity_id"], stale["entity_id"], elsewhere["entity_id"]}),
            (("--project", "project-a' OR 1=1 --"), set()),
        ]
        for arguments, identifiers in cases:
            with self.subTest(arguments=arguments):
                data = self.read_json(*arguments)
                self.assertEqual({row["entity_id"] for row in data["entities"]}, identifiers)
                self.assertEqual(data["count"], len(identifiers))

    def test_inspection_does_not_revalidate_files_or_mutate_any_records(self):
        missing = self.create("gone.txt")
        changed = self.create("changed.txt")
        Path(missing["current_path"]).unlink()
        Path(changed["current_path"]).write_bytes(b"external edit")
        before = self.snapshot()
        self.store.close()
        data = self.read_json()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual({row["status"] for row in data["entities"]}, {"active"})
        by_id = {row["entity_id"]: row for row in data["entities"]}
        self.assertTrue(by_id[changed["entity_id"]]["is_focus"])
        self.assertFalse(by_id[missing["entity_id"]]["is_focus"])
        self.assertEqual(by_id[changed["entity_id"]]["fingerprint"], changed["fingerprint"])
        self.assertEqual(Path(changed["current_path"]).read_bytes(), b"external edit")

    def test_focus_is_scoped_and_clearing_it_does_not_hide_entities(self):
        first = self.create("first.txt")
        self.create("second.txt", MemoryScope("other", "project-a"))
        self.store.clear_focus(SCOPE)
        data = self.read_json()
        self.assertEqual(data["count"], 2)
        by_id = {row["entity_id"]: row for row in data["entities"]}
        self.assertFalse(by_id[first["entity_id"]]["is_focus"])
        self.assertEqual(sum(row["is_focus"] for row in data["entities"]), 1)

    def test_text_shows_full_identity_scope_focus_and_unicode_path(self):
        entity = self.create("中文 笔记.txt")
        status, output, errors = self.run_cli()
        self.assertEqual((status, errors), (0, ""))
        for value in (str(self.database), "Saved entities: 1", "files not rechecked",
                      entity["entity_id"], '"local"', '"project-a"', "active", "yes", "中文 笔记.txt"):
            self.assertIn(value, output)

    def test_path_precedence_and_default_are_independent_of_working_directory(self):
        self.create("notes.txt")
        output = io.StringIO()
        with patch.dict(os.environ, {"SAMANTHA_OBJECT_PATH": str(self.database)}), redirect_stdout(output):
            self.assertEqual(entities_cli.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["count"], 1)
        with patch.dict(os.environ, {"SAMANTHA_OBJECT_PATH": str(self.root / "wrong.sqlite3")}):
            self.assertEqual(self.read_json()["count"], 1)
        output = io.StringIO()
        with patch.dict(os.environ, {"SAMANTHA_OBJECT_PATH": ""}), \
                patch.object(entities_cli, "DEFAULT_OBJECT_PATH", self.database), redirect_stdout(output):
            self.assertEqual(entities_cli.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["database"], str(self.database))

    def test_missing_database_is_empty_and_does_not_create_parent_or_database(self):
        missing = self.root / "uncreated" / "objects.sqlite3"
        output = io.StringIO()
        with redirect_stdout(output):
            status = entities_cli.main(["--db-path", str(missing), "--json"])
        self.assertEqual(status, 0)
        data = json.loads(output.getvalue())
        self.assertFalse(data["database_exists"])
        self.assertEqual((data["count"], data["entities"]), (0, []))
        self.assertFalse(missing.parent.exists())

    def test_wrong_or_damaged_database_returns_error_without_modifying_it(self):
        wrong = self.root / "wrong.sqlite3"
        with closing(sqlite3.connect(wrong)) as db:
            db.execute("CREATE TABLE unrelated (value TEXT)")
            db.execute("PRAGMA user_version=1")
            db.commit()
        damaged = self.root / "damaged.sqlite3"
        damaged.write_bytes(b"not a sqlite database")
        for path in (wrong, damaged):
            with self.subTest(path=path):
                before = path.read_bytes()
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    status = entities_cli.main(["--db-path", str(path), "--json"])
                self.assertEqual(status, 1)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("Unable to read saved entities", errors.getvalue())
                self.assertEqual(path.read_bytes(), before)

    def test_terminal_entry_point_bypasses_workflow_and_history_and_preserves_directory(self):
        self.create("notes.txt")
        state_file = self.root / "terminal-state.json"
        log_file = self.root / "must-not-be-created.sqlite3"
        before = self.snapshot()
        cwd = os.getcwd()
        output = io.StringIO()
        with patch.object(sys, "argv", ["samantha", "entities", "--db-path", str(self.database), "--json"]), \
                patch.dict(os.environ, {"SAMANTHA_STATE_FILE": str(state_file),
                                        "SAMANTHA_LOG_PATH": str(log_file),
                                        "SAMANTHA_OBJECT_TRACKING": "true",
                                        "SAMANTHA_MEMORY_BACKEND": "invalid-backend"}), \
                patch.object(samantha, "load_dotenv"), \
                patch.object(samantha, "process", side_effect=AssertionError("Workflow must not run")), \
                patch.object(samantha, "InteractionJournal", side_effect=AssertionError("History must not change")), \
                patch.object(samantha, "ObjectStore", side_effect=AssertionError("Writable store must not open")), \
                redirect_stdout(output):
            self.assertEqual(samantha.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["count"], 1)
        self.assertEqual(json.loads(state_file.read_text(encoding="utf-8")), {"current_dir": cwd})
        self.assertEqual(os.getcwd(), cwd)
        self.assertFalse(log_file.exists())
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
