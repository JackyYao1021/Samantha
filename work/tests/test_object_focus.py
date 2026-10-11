"""A completed write with failed state persistence must not reuse older focus."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from object_tracking import ObjectStore
from samantha import process
from test_object_tracking import SCOPE, proposed_effect, tracked_services, working_directory


class ObjectFocusTests(unittest.TestCase):
    def test_store_failure_after_new_file_write_clears_older_focus_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObjectStore(root / "objects.sqlite3") as store, working_directory(root):
                def run(fake, request):
                    return process(request, services=fake, input_fn=lambda _: "y", output_fn=lambda _: None,
                                   log_file=root / "interactions.sqlite3", object_store=store, object_scope=SCOPE)

                older, newest = root / "older.txt", root / "newest.txt"
                first = run(tracked_services(proposed_effect("create", older, content="older\n")),
                            "Create older.txt")
                self.assertEqual(first["status"], "succeeded", first)
                old_entity = store.candidates(SCOPE)[0]
                self.assertEqual(json.loads(store.latest_context(SCOPE))["focus"], old_entity["entity_id"])

                fake = tracked_services(proposed_effect("create", newest, content="newest\n"))
                with patch.object(store, "apply_verified", side_effect=RuntimeError("database full")):
                    state = run(fake, "Create newest.txt")

                self.assertEqual(state["status"], "succeeded", state)
                self.assertTrue(state["result"]["success"])
                self.assertIn("database full", state["tracking_error"])
                self.assertEqual(state["tracked_entities"], [])
                self.assertIsNone(json.loads(store.latest_context(SCOPE))["focus"])
                self.assertEqual(store.candidates(SCOPE)[0]["entity_id"], old_entity["entity_id"])
                self.assertEqual(older.read_bytes(), b"older\n")
                self.assertEqual(newest.read_bytes(), b"newest\n")
                fake.execute_commands.assert_called_once()
                fake.correct_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
