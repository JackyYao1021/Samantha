"""Conservative P0 target hints never use append content or rename destinations."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_memory import MemoryScope
from object_tracking.runtime import ObjectTracker


class ObjectReferenceHintTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.tracker = ObjectTracker(self.store, MemoryScope("hint-user", "hint-project"))
        self.no_focus = json.dumps({"focus": None, "candidates": []})

    def test_append_content_cannot_supply_missing_target(self):
        requests = ["Append the text 'to beta.txt' to it",
                    'Append "to my notes.txt" to the file just created',
                    "Append text to beta.txt to it",
                    "给它追加文本 'to beta.txt'。",
                    "把文本 'beta.txt' 追加到刚才的文件。",
                    "将文字 `to README` 追加到它。"]
        for request in requests:
            with self.subTest(request=request):
                self.assertTrue(self.tracker.reference_question(request, self.no_focus))
        self.store.assert_not_called()

    def test_rename_destination_cannot_supply_missing_source(self):
        for request in ("Rename it to notes.txt", "Rename the file just created to 'my notes.txt'",
                        "把刚才的文件重命名为 notes.txt。", "将它改名为 README。"):
            with self.subTest(request=request):
                self.assertTrue(self.tracker.reference_question(request, self.no_focus))

    def test_clear_explicit_source_with_reference_description_does_not_need_focus(self):
        requests = ["给刚才创建的 README 追加一行 beta。",
                    "给刚才创建的README追加一行 beta。",
                    "把刚才创建的 notes.txt 重命名为 todo.txt。",
                    "给刚才的文件 'my notes.txt' 追加一行 beta。",
                    "把刚才的文件 “会议 notes.txt” 重命名为 todo.txt。",
                    "给‘my notes’追加内容 '刚才'。",
                    "Rename the file just created, 'my notes.txt', to todo.txt",
                    'Rename the file just created named "my notes", to todo.txt',
                    "Append beta to the file just created, 'my notes.txt'.",
                    "Append 'it' to README", "Append 'that file' to /tmp/README",
                    "给 `刚才报告.txt` 追加一行 beta。"]
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(self.tracker.reference_question(request, self.no_focus), "")

    def test_chinese_content_before_append_to_is_not_used_as_target(self):
        for request in ("把 beta.txt 的内容追加到它。", "把 README 追加到刚才的文件。",
                        "将 'README' 追加到那个文件。"):
            with self.subTest(request=request):
                self.assertTrue(self.tracker.reference_question(request, self.no_focus))
        # The named append target remains explicit even when the appended text
        # contains a referent that would otherwise activate the guard.
        self.assertEqual(self.tracker.reference_question("把 '给它' 追加到 README。", self.no_focus), "")

    def test_verified_focus_allows_recent_reference_without_implicit_approval(self):
        context = json.dumps({"focus": "entity-a", "candidates": [{"entity_id": "entity-a"}]})
        self.assertEqual(self.tracker.reference_question("Append beta to it", context), "")
        self.store.assert_not_called()

    def test_request_without_recent_reference_passes_to_normal_clarification(self):
        for request in ("Create notes.txt", "Rename README to TODO", "给 report.txt 追加 beta。"):
            with self.subTest(request=request):
                self.assertEqual(self.tracker.reference_question(request, self.no_focus), "")


if __name__ == "__main__":
    unittest.main()
