"""File-history questions must never recreate a file or lose follow-up focus."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_memory import MemoryScope
from interaction_log import InteractionJournal
from object_tracking.questions import (
    creation_answer_order, creation_question_order, is_first_use_question, is_recent_creation_question,
)
from object_tracking.runtime import ObjectTracker
from object_tracking.store import ObjectStore
from samantha import process
from test_object_tracking import SCOPE, proposed_effect, tracked_services, working_directory
from test_workflow import services
from workflow import build_workflow, initial_state


class FileHistoryQuestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "objects.sqlite3"
        self.log = self.root / "interactions.sqlite3"

    def run_write(self, store, effect, request):
        fake = tracked_services(effect)
        state = process(request, services=fake, input_fn=lambda _: "y", output_fn=lambda _: None,
                        log_file=self.log, object_store=store, object_scope=SCOPE)
        self.assertEqual(state["status"], "succeeded", state)
        return state

    def ask(self, store, request="what file I just created?", *, scope=SCOPE, fake=None):
        fake = fake or services()
        output = []
        state = process(request, services=fake,
                        input_fn=lambda _: self.fail("A file-history answer must not request execution approval."),
                        output_fn=output.append, log_file=self.log,
                        object_store=store, object_scope=scope)
        self.assertEqual(state["status"], "succeeded", state)
        self.assertEqual(state["commands"], [])
        self.assertIsNone(state["result"])
        self.assertFalse(state["approved"])
        for agent in (fake.clarify, fake.parse_intent, fake.generate_commands,
                      fake.review_commands, fake.execute_commands, fake.correct_error):
            agent.assert_not_called()
        self.assertIn(state["answer"], output)
        self.assertNotIn("Commands executed successfully ^-^", output)
        return state

    def test_exact_reported_question_bypasses_erroneous_creation_proposal_and_keeps_focus(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            path = self.root / "notes.txt"
            self.run_write(store, proposed_effect("create", path, content="alpha\n"), "Create notes.txt")
            entity = store.candidates(SCOPE)[0]
            # Before the fix this proposal hits source_absent=False for the existing file.
            bad_proposal = tracked_services(proposed_effect("create", path, content=""))
            state = self.ask(store, fake=bad_proposal)
            self.assertIn(str(path), state["answer"])
            self.assertEqual(path.read_bytes(), b"alpha\n")
            self.assertEqual(json.loads(store.latest_context(SCOPE))["focus"], entity["entity_id"])
            self.run_write(store, proposed_effect("append", path, content="beta\n", entity=entity),
                           "Append beta to it")
            self.assertEqual(path.read_bytes(), b"alpha\nbeta\n")
            with InteractionJournal(self.log, readonly=True) as journal:
                events = journal.read_session(state["log_session_id"])["events"]
            operations = [event["data"].get("name") for event in events
                          if event["kind"] == "operation_started"]
            self.assertIn("objects.answer_recent_creation", operations)
            self.assertNotIn("commands.execute", operations)
            self.assertNotIn("objects.clear_focus", operations)

    def test_reopened_store_answers_original_and_current_names_after_rename(self):
        path = self.root / "notes.txt"
        renamed = self.root / "todo.txt"
        with working_directory(self.root):
            with ObjectStore(self.database) as store:
                self.run_write(store, proposed_effect("create", path, content="alpha\n"), "Create notes.txt")
                entity = store.candidates(SCOPE)[0]
                self.run_write(store, proposed_effect("rename", path, destination=renamed, entity=entity),
                               "Rename it to todo.txt")
            with ObjectStore(self.database) as store:
                state = self.ask(store)
                self.assertIn(str(path), state["answer"])
                self.assertIn(str(renamed), state["answer"])
                self.assertFalse(path.exists())
                self.assertEqual(renamed.read_bytes(), b"alpha\n")
                self.assertEqual(len(store.events(SCOPE)), 2)

    def test_latest_creation_is_independent_of_focus_and_latest_append(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            older, newer = self.root / "older.txt", self.root / "newer.txt"
            self.run_write(store, proposed_effect("create", older, content="old"), "Create older.txt")
            old_entity = store.lookup_path(SCOPE, str(older))
            self.run_write(store, proposed_effect("create", newer, content="new"), "Create newer.txt")
            self.run_write(store, proposed_effect("append", older, content="extra", entity=old_entity),
                           "Append extra to older.txt")
            state = self.ask(store)
            self.assertIn(str(newer), state["answer"])
            self.assertNotIn(str(older), state["answer"])
            self.assertEqual(json.loads(store.latest_context(SCOPE))["focus"], old_entity["entity_id"])
            store.clear_focus(SCOPE)
            self.assertIn(str(newer), self.ask(store)["answer"])
            self.assertIsNone(json.loads(store.latest_context(SCOPE))["focus"])

    def test_missing_or_changed_latest_creation_does_not_fall_back_to_older_file(self):
        for change in ("missing", "changed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with ObjectStore(root / "objects.sqlite3") as store, working_directory(root):
                    older, newer = root / "older.txt", root / "newer.txt"
                    self.run_write(store, proposed_effect("create", older, content="old"), "Create older.txt")
                    self.run_write(store, proposed_effect("create", newer, content="new"), "Create newer.txt")
                    if change == "missing":
                        newer.unlink()
                    else:
                        newer.write_text("external change", encoding="utf-8")
                    state = self.ask(store)
                    self.assertIn(str(newer), state["answer"])
                    self.assertIn("cannot be verified", state["answer"])
                    self.assertNotIn(str(older), state["answer"])

    def test_no_record_or_different_scope_cannot_invent_a_filename(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            self.assertIn("no verified file creation record", self.ask(store)["answer"])
            path = self.root / "private.txt"
            self.run_write(store, proposed_effect("create", path, content="private"), "Create private.txt")
            for scope in (MemoryScope("other-user", SCOPE.project_id),
                          MemoryScope(SCOPE.user_id, "other-project")):
                state = self.ask(store, scope=scope)
                self.assertIn("no verified file creation record", state["answer"])
                self.assertNotIn(str(path), state["answer"])

    def test_disabled_tracking_answers_without_model_or_shell(self):
        state = self.ask(None)
        self.assertIn("tracking is disabled", state["answer"])

    def test_chinese_question_returns_verified_filename(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            path = self.root / "notes.txt"
            self.run_write(store, proposed_effect("create", path, content="alpha"), "Create notes.txt")
            self.assertIn("你刚才创建了 " + str(path), self.ask(store, "我刚才创建了哪个文件？")["answer"])

    def test_first_creation_returns_current_path_after_rename_and_reopen_independently_of_focus(self):
        original, renamed, newer = (self.root / name for name in ("first.txt", "renamed.txt", "newer.txt"))
        with working_directory(self.root):
            with ObjectStore(self.database) as store:
                self.run_write(store, proposed_effect("create", original, content="first"), "Create first.txt")
                first = store.lookup_path(SCOPE, str(original))
                self.run_write(store, proposed_effect("rename", original, destination=renamed, entity=first),
                               "Rename first.txt")
                self.run_write(store, proposed_effect("create", newer, content="newer"), "Create newer.txt")
                focused = json.loads(store.latest_context(SCOPE))["focus"]
            with ObjectStore(self.database) as store:
                state = self.ask(store, "help me show the path of the file I created first")
                self.assertIn(str(original), state["answer"])
                self.assertIn(str(renamed), state["answer"])
                self.assertNotIn(str(newer), state["answer"])
                self.assertEqual(json.loads(store.latest_context(SCOPE))["focus"], focused)
                self.assertEqual(store.first_creation(SCOPE)["entity"]["entity_id"], first["entity_id"])
                self.assertEqual(store.first_creation(SCOPE)["entity"]["version"], 2)
                self.assertEqual(renamed.read_bytes(), b"first")

    def test_reported_used_first_question_resolves_short_creation_reply_without_models_or_focus(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            first, newer = self.root / "first.txt", self.root / "newer.txt"
            self.run_write(store, proposed_effect("create", first, content="first"), "Create first.txt")
            self.run_write(store, proposed_effect("create", newer, content="newer"), "Create newer.txt")
            store.clear_focus(SCOPE)
            fake = services()
            output, prompts = [], []

            def reply(prompt):
                prompts.append(prompt)
                self.assertEqual(len(prompts), 1, "A timestamp comparison must not ask a second question.")
                return "the one I created first"

            state = process("help me show the path of the file I used first", services=fake,
                            input_fn=reply, output_fn=output.append, log_file=self.log,
                            object_store=store, object_scope=SCOPE)
            self.assertEqual(state["status"], "succeeded", state)
            self.assertIn(str(first), state["answer"])
            self.assertNotIn(str(newer), state["answer"])
            self.assertEqual(state["commands"], [])
            self.assertIsNone(state["result"])
            self.assertEqual(state["clarification_answers"][0]["answer"], "the one I created first")
            self.assertTrue(any("Do you mean the file you created first?" in line for line in output))
            for agent in (fake.clarify, fake.parse_intent, fake.generate_commands,
                          fake.review_commands, fake.execute_commands, fake.correct_error):
                agent.assert_not_called()
            self.assertIsNone(json.loads(store.latest_context(SCOPE))["focus"])
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(newer.read_bytes(), b"newer")
            with InteractionJournal(self.log, readonly=True) as journal:
                events = journal.read_session(state["log_session_id"])["events"]
            operations = [event["data"].get("name") for event in events
                          if event["kind"] == "operation_started"]
            self.assertIn("objects.answer_first_creation", operations)
            self.assertNotIn("objects.clear_focus", operations)
            self.assertNotIn("commands.execute", operations)

    def test_first_creation_is_available_outside_recent_candidate_budget(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            first = self.root / "first.txt"
            self.run_write(store, proposed_effect("create", first, content="first"), "Create first.txt")
            for index in range(22):
                path = self.root / f"later-{index}.txt"
                self.run_write(store, proposed_effect("create", path, content="later"), "Create a later file")
            context = json.loads(store.latest_context(SCOPE))
            self.assertNotIn(str(first), [entity["current_path"] for entity in context["candidates"]])
            state = self.ask(store, "show me the path of the first file I created")
            self.assertIn(str(first), state["answer"])

    def test_yes_confirms_first_creation_only_in_the_matching_history_clarification(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            first = self.root / "first.txt"
            self.run_write(store, proposed_effect("create", first, content="first"), "Create first.txt")
            for question, reply in (("help me show the path of the file I used first", "yes"),
                                    ("显示我最先使用的文件路径", "是的")):
                with self.subTest(question=question):
                    fake = services()
                    state = process(question, services=fake, input_fn=lambda _: reply,
                                    output_fn=lambda _: None, log_file=self.log,
                                    object_store=store, object_scope=SCOPE)
                    self.assertEqual(state["status"], "succeeded", state)
                    self.assertIn(str(first), state["answer"])
                    fake.clarify.assert_not_called()
                    fake.execute_commands.assert_not_called()
            self.assertIsNone(creation_answer_order("yes"))

    def test_missing_or_changed_first_creation_is_not_replaced_with_a_later_file(self):
        for change in ("missing", "changed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with ObjectStore(root / "objects.sqlite3") as store, working_directory(root):
                    first, newer = root / "first.txt", root / "newer.txt"
                    self.run_write(store, proposed_effect("create", first, content="first"), "Create first.txt")
                    self.run_write(store, proposed_effect("create", newer, content="newer"), "Create newer.txt")
                    if change == "missing":
                        first.unlink()
                    else:
                        first.write_bytes(b"changed externally")
                    state = self.ask(store, "what file did I create first?")
                    self.assertIn(str(first), state["answer"])
                    self.assertIn("cannot be verified", state["answer"])
                    self.assertNotIn(str(newer), state["answer"])

    def test_first_creation_respects_scope_and_disabled_tracking(self):
        question = "what file did I create first?"
        with ObjectStore(self.database) as store, working_directory(self.root):
            self.assertIn("no verified file creation record", self.ask(store, question)["answer"])
            first = self.root / "private.txt"
            self.run_write(store, proposed_effect("create", first, content="private"), "Create private.txt")
            for scope in (MemoryScope("other-user", SCOPE.project_id),
                          MemoryScope(SCOPE.user_id, "other-project")):
                state = self.ask(store, question, scope=scope)
                self.assertIn("no verified file creation record", state["answer"])
                self.assertNotIn(str(first), state["answer"])
            self.assertIn(str(first), self.ask(store, "我最先创建的文件的路径是什么？")["answer"])
        self.assertIn("tracking is disabled", self.ask(None, question)["answer"])

    def test_first_use_clarification_does_not_swallow_a_chained_action_in_the_reply(self):
        with ObjectStore(self.database) as store, working_directory(self.root):
            fake = services()
            fake.clarify.return_value = {"cancelled": True}
            state = process("help me show the path of the file I used first", services=fake,
                            input_fn=lambda _: "the one I created first and delete it",
                            output_fn=lambda _: None, log_file=self.log,
                            object_store=store, object_scope=SCOPE)
            self.assertEqual(state["status"], "cancelled")
            fake.clarify.assert_called_once()
            self.assertIn("the one I created first and delete it", fake.clarify.call_args.args[0])
            fake.execute_commands.assert_not_called()

    def test_first_use_clarification_respects_the_configured_limit(self):
        with ObjectStore(self.database) as store:
            fake = services()
            graph = build_workflow(fake, object_tracker=ObjectTracker(store, SCOPE))
            state = graph.invoke(initial_state("help me show the path of the file I used first",
                                               max_clarifications=0),
                                 {"configurable": {"thread_id": "no-clarifications"}})
            self.assertEqual(state["status"], "failed")
            self.assertIn("Clarification limit (0) reached", state["error"])
            self.assertNotIn("__interrupt__", state)
            fake.clarify.assert_not_called()

    def test_first_query_recognition_requires_a_complete_read_only_question(self):
        first_questions = ["help me show the path of the file I created first",
                           "Please show me the path of the first file I created.",
                           "what file did I create first?", "what was the first file I created?",
                           "我最先创建的文件的路径是什么？", "显示我最早创建的文件路径"]
        for request in first_questions:
            with self.subTest(request=request):
                self.assertEqual(creation_question_order(request), "first")
                self.assertFalse(is_recent_creation_question(request))
        used = "help me show the path of the file I used first"
        self.assertTrue(is_first_use_question(used))
        self.assertIsNone(creation_question_order(used))
        self.assertEqual(creation_answer_order("the one I created first"), "first")
        self.assertEqual(creation_answer_order("最先创建的那个文件"), "first")
        for request in ("Show me the path of the first file I created and delete it",
                        "Rename the first file I created to first.txt",
                        "Create a file named 'what file did I create first'",
                        "显示我最先创建的文件路径然后删除它"):
            with self.subTest(request=request):
                self.assertIsNone(creation_question_order(request))
                self.assertFalse(is_first_use_question(request))
        self.assertIsNone(creation_answer_order("the one I created first and delete it"))

    def test_question_wording_and_action_requests_remain_distinct(self):
        questions = ["what file I just created?", "What file did I just create?",
                     "Which file did you create?", "Which file was just created?",
                     "What is the name of the file I just created?",
                     "What was the last created file?", "Tell me what file I just created.",
                     "我刚才创建了什么文件？", "刚刚创建的文件叫什么？", "我刚才创建的文件是什么？"]
        actions = ["Create notes.txt", "Create a file named 'what file I just created'",
                   "Rename the file I just created to todo.txt", "Append beta to it",
                   "What file did I just create? Then delete it.",
                   "Tell me what file I just created and rename it to todo.txt",
                   "我刚才创建了哪个文件？把它删除。", "删除刚刚创建的文件"]
        for request in questions:
            with self.subTest(request=request):
                self.assertTrue(is_recent_creation_question(request))
        for request in actions:
            with self.subTest(request=request):
                self.assertFalse(is_recent_creation_question(request))


if __name__ == "__main__":
    unittest.main()
