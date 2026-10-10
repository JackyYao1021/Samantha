"""P0 graph/CLI integration with real files, journal evidence and fresh approvals.

Agents are deterministic fixtures; these checks do not measure Qwen quality.
The subprocess fixture uses the real Bash executor only for literal test-file
commands in its own temporary directory.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent_memory import MemoryScope
from interaction_log import InteractionJournal
from langgraph.types import Command
from object_tracking.effects import render_effects
from object_tracking.runtime import ObjectTracker
from object_tracking.store import ObjectStore
from samantha import main, process
from test_workflow import services
from workflow import build_workflow, initial_state


SCOPE = MemoryScope("object-user", "object-project")


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def proposed_effect(kind, path, *, content=None, destination=None, entity=None):
    return {"kind": kind, "path": str(path),
            "destination": None if destination is None else str(destination),
            "content": content,
            "entity_id": None if entity is None else entity["entity_id"],
            "entity_version": None if entity is None else entity["version"]}


def apply_fixture_effect(effect):
    """Only test-owned paths reach this deterministic executor."""
    path = Path(effect["path"])
    if effect["kind"] == "create":
        path.write_bytes(effect["content"].encode("utf-8"))
    elif effect["kind"] == "rename":
        path.rename(effect["destination"])
    else:
        with path.open("ab") as stream:
            stream.write(effect["content"].encode("utf-8"))
    return {"success": True, "output": "fixture operation completed", "current_dir": os.getcwd()}


def tracked_services(effect, *, executor=None, commands=None):
    fake = services()
    fake.generate_commands.return_value = json.dumps({
        "Commands": render_effects([effect]) if commands is None else commands,
        "Explanation": "Perform the explicitly proposed file effect.", "Effects": [effect],
    }, ensure_ascii=False)
    fake.execute_commands.side_effect = executor or (lambda _: apply_fixture_effect(effect))
    return fake


class ObjectTrackingIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.database = self.root / "objects.sqlite3"
        self.log = self.root / "interactions.sqlite3"

    def run_tracked(self, store, fake, *, answer="y", input_fn=None, request="Create a file", scope=SCOPE):
        return process(request, services=fake, input_fn=input_fn or (lambda _: answer),
                       output_fn=lambda _: None, log_file=self.log,
                       object_store=store, object_scope=scope)

    def read_log(self, state):
        with InteractionJournal(self.log, readonly=True) as journal:
            return journal.read_session(state["log_session_id"])

    def assert_execution_provenance(self, state, entity, store):
        events = self.read_log(state)["events"]
        executed = next(event for event in events if event["kind"] == "operation_finished"
                        and event["data"].get("name") == "commands.execute")
        self.assertEqual(entity["source_operation_id"], executed["data"]["operation_id"])
        self.assertEqual(entity["source_session_id"], state["log_session_id"])
        verification = next(event for event in events if event["kind"] == "operation_finished"
                            and event["data"].get("name") == "objects.verify")
        self.assertLess(executed["sequence"], verification["sequence"])
        commits = [event for event in events if event["kind"] == "operation_started"
                   and event["data"].get("name") in {"objects.commit", "objects.apply"}]
        self.assertTrue(commits, "The object-state commit must be journaled.")
        self.assertLess(verification["sequence"], commits[0]["sequence"])
        object_event = next(event for event in store.events(SCOPE, entity_id=entity["entity_id"])
                            if event["operation_id"] == executed["data"]["operation_id"])
        self.assertEqual(object_event["execution_status"],
                         "succeeded" if state["result"]["success"] else "failed")
        self.assertEqual(object_event["verification_status"], "verified")
        self.assertEqual(object_event["evidence"]["action_id"], state["reviewed_action_id"])

    def test_reopened_store_tracks_create_rename_append_with_fresh_approvals(self):
        with working_directory(self.workspace):
            old_path = self.workspace / "会议 notes.txt"
            new_path = self.workspace / "todo 清单.txt"
            keep = self.workspace / "keep.txt"
            keep.write_bytes(b"unchanged\n")
            states, entities, fakes = [], [], []
            operations = [("create", old_path, "alpha\n", None),
                          ("rename", old_path, None, new_path),
                          ("append", new_path, "beta\n", None)]
            for kind, path, content, destination in operations:
                store = ObjectStore(self.database)
                try:
                    entity = entities[-1] if entities else None
                    effect = proposed_effect(kind, path, content=content,
                                             destination=destination, entity=entity)
                    fake = tracked_services(effect)
                    state = self.run_tracked(store, fake, request=(
                        "Create notes" if kind == "create" else "Update the file just created"))
                    self.assertEqual(state["status"], "succeeded", state)
                    self.assertFalse(state["approved"])
                    self.assertEqual(state["approved_action_id"], "")
                    self.assertFalse(state.get("tracking_error"), state)
                    self.assertEqual(state["object_verifications"][0]["status"], "verified")
                    candidates = store.candidates(SCOPE)
                    self.assertEqual(len(candidates), 1)
                    entities.append(candidates[0])
                    states.append(state)
                    fakes.append(fake)
                    self.assert_execution_provenance(state, candidates[0], store)
                finally:
                    store.close()
            self.assertEqual(len({entity["entity_id"] for entity in entities}), 1)
            self.assertEqual([entity["version"] for entity in entities], [1, 2, 3])
            self.assertEqual(entities[-1]["current_path"], str(new_path))
            self.assertEqual(len({state["log_session_id"] for state in states}), 3)
            self.assertFalse(old_path.exists())
            self.assertEqual(new_path.read_bytes(), b"alpha\nbeta\n")
            self.assertEqual(keep.read_bytes(), b"unchanged\n")
            self.assertIn(json.dumps(str(new_path), ensure_ascii=False), fakes[-1].clarify.call_args.args[0])
            for state, fake in zip(states, fakes):
                fake.execute_commands.assert_called_once()
                evidence = self.read_log(state)
                confirmations = [event for event in evidence["events"]
                                 if event["kind"] == "user_input"
                                 and event["data"].get("source") == "confirmation"]
                self.assertEqual(len(confirmations), 1)

    def test_baseline_without_effects_keeps_the_existing_workflow(self):
        fake = services()
        state = process("Create a file", services=fake, input_fn=lambda _: "y",
                        output_fn=lambda _: None, log_file=self.log)
        self.assertEqual(state["status"], "succeeded")
        fake.execute_commands.assert_called_once_with(["touch test.txt"])
        self.assertFalse(state.get("object_effects"))

    def test_enabled_tracking_requires_explicit_effects_field(self):
        store = ObjectStore(self.database)
        self.addCleanup(store.close)
        fake = services()
        state = self.run_tracked(store, fake, input_fn=lambda _: self.fail("No approval expected"))
        self.assertEqual(state["status"], "failed", state)
        fake.execute_commands.assert_not_called()
        self.assertEqual(store.candidates(SCOPE), [])

    def test_explicit_empty_effects_keeps_unsupported_command_behavior(self):
        store = ObjectStore(self.database)
        self.addCleanup(store.close)
        fake = services()
        fake.generate_commands.return_value = json.dumps({"Commands": ["touch test.txt"],
                                                          "Explanation": "Untracked operation", "Effects": []})
        state = self.run_tracked(store, fake)
        self.assertEqual(state["status"], "succeeded", state)
        fake.execute_commands.assert_called_once_with(["touch test.txt"])
        self.assertEqual(state["object_effects"], [])
        self.assertEqual(state["object_verifications"], [])
        self.assertEqual(store.candidates(SCOPE), [])

    def test_declined_create_does_not_create_completed_object(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            effect = proposed_effect("create", self.workspace / "cancelled.txt", content="alpha\n")
            fake = tracked_services(effect)
            state = self.run_tracked(store, fake, answer="n")
            self.assertEqual(state["status"], "cancelled")
            fake.execute_commands.assert_not_called()
            self.assertEqual(store.candidates(SCOPE), [])
            self.assertFalse(Path(effect["path"]).exists())

    def test_declined_new_target_clears_older_focus_and_followup_requires_clarification(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            old_path = self.workspace / "older.txt"
            created = self.run_tracked(store, tracked_services(
                proposed_effect("create", old_path, content="alpha\n")))
            self.assertEqual(created["status"], "succeeded", created)
            old_entity = store.candidates(SCOPE)[0]
            rejected = self.run_tracked(store, tracked_services(
                proposed_effect("create", self.workspace / "declined.txt", content="other\n")), answer="n")
            self.assertEqual(rejected["status"], "cancelled", rejected)
            self.assertIsNone(json.loads(store.latest_context(SCOPE))["focus"])
            fake = tracked_services(proposed_effect("append", old_path, content="beta\n", entity=old_entity))
            followup = self.run_tracked(store, fake, request="给刚才的文件追加一行 beta。", answer="cancel")
            self.assertEqual(followup["status"], "cancelled", followup)
            self.assertEqual(followup["clarification_count"], 1)
            fake.clarify.assert_not_called()
            fake.generate_commands.assert_not_called()
            fake.execute_commands.assert_not_called()
            self.assertEqual(old_path.read_bytes(), b"alpha\n")

    def test_unsupported_completed_operation_clears_previous_focus(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            created = self.run_tracked(store, tracked_services(
                proposed_effect("create", self.workspace / "notes.txt", content="alpha\n")))
            self.assertEqual(created["status"], "succeeded", created)
            self.assertIsNotNone(json.loads(store.latest_context(SCOPE))["focus"])
            fake = services()
            fake.generate_commands.return_value = json.dumps({"Commands": ["pwd"], "Effects": []})
            completed = self.run_tracked(store, fake, request="Show current directory")
            self.assertEqual(completed["status"], "succeeded", completed)
            self.assertIsNone(json.loads(store.latest_context(SCOPE))["focus"])
            self.assertEqual(len(store.candidates(SCOPE)), 1)

    def test_other_project_recent_reference_asks_before_model_generation(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            created = self.run_tracked(store, tracked_services(
                proposed_effect("create", self.workspace / "notes.txt", content="alpha\n")))
            self.assertEqual(created["status"], "succeeded", created)
            fake = services()
            state = self.run_tracked(store, fake, scope=MemoryScope(SCOPE.user_id, "other-project"),
                                     request="给刚才的文件追加一行 beta。", answer="cancel")
            self.assertEqual(state["status"], "cancelled", state)
            self.assertEqual(state["clarification_count"], 1)
            fake.clarify.assert_not_called()
            fake.generate_commands.assert_not_called()
            fake.execute_commands.assert_not_called()
            self.assertEqual((self.workspace / "notes.txt").read_bytes(), b"alpha\n")

    def test_file_changed_while_waiting_for_approval_blocks_execution(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            target = self.workspace / "notes.txt"
            fake = tracked_services(proposed_effect("create", target, content="alpha\n"))

            def answer(_):
                target.write_bytes(b"external modification\n")
                return "y"

            state = self.run_tracked(store, fake, input_fn=answer)
            self.assertEqual(state["status"], "failed", state)
            fake.execute_commands.assert_not_called()
            self.assertEqual(target.read_bytes(), b"external modification\n")
            self.assertEqual(store.candidates(SCOPE), [])

    def test_mismatched_effect_or_forged_entity_stops_before_approval(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            target = self.workspace / "notes.txt"
            mismatch = proposed_effect("create", target, content="alpha\n")
            fake = tracked_services(mismatch, commands=["printf '%s' 'different' > notes.txt"])
            state = self.run_tracked(store, fake, input_fn=lambda _: self.fail("No approval expected"))
            self.assertEqual(state["status"], "failed", state)
            fake.execute_commands.assert_not_called()
            target.write_bytes(b"alpha\n")
            forged = proposed_effect("append", target, content="beta\n")
            forged.update(entity_id="nonexistent-entity", entity_version=1)
            fake = tracked_services(forged)
            state = self.run_tracked(store, fake, input_fn=lambda _: self.fail("No approval expected"))
            self.assertEqual(state["status"], "failed", state)
            fake.execute_commands.assert_not_called()
            self.assertEqual(store.candidates(SCOPE), [])

    def test_other_scope_and_wrong_version_cannot_reuse_entity(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            target = self.workspace / "notes.txt"
            created = self.run_tracked(store, tracked_services(
                proposed_effect("create", target, content="alpha\n")))
            self.assertEqual(created["status"], "succeeded", created)
            entity = store.candidates(SCOPE)[0]
            for scope in (MemoryScope("other-user", SCOPE.project_id),
                          MemoryScope(SCOPE.user_id, "other-project")):
                with self.subTest(scope=scope):
                    fake = tracked_services(proposed_effect("append", target, content="beta\n", entity=entity))
                    state = self.run_tracked(store, fake, scope=scope,
                                             input_fn=lambda _: self.fail("No approval expected"))
                    self.assertEqual(state["status"], "failed", state)
                    fake.execute_commands.assert_not_called()
                    self.assertEqual(store.candidates(scope), [])
            fake = tracked_services(proposed_effect("append", target, content="beta\n",
                                                     entity={**entity, "version": entity["version"] + 1}))
            state = self.run_tracked(store, fake, input_fn=lambda _: self.fail("No approval expected"))
            self.assertEqual(state["status"], "failed", state)
            fake.execute_commands.assert_not_called()
            self.assertEqual(store.candidates(SCOPE)[0]["version"], 1)
            self.assertEqual(target.read_bytes(), b"alpha\n")

    def test_effect_changed_after_review_cannot_use_old_approval(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            fake = tracked_services(proposed_effect("create", self.workspace / "notes.txt", content="alpha\n"))
            with InteractionJournal(self.log) as journal, journal.start_session("Create notes"):
                graph = build_workflow(fake, object_tracker=ObjectTracker(store, SCOPE))
                config = {"configurable": {"thread_id": uuid.uuid4().hex}}
                pending = graph.invoke(initial_state("Create notes"), config)
                altered = [dict(effect) for effect in pending["object_effects"]]
                altered[0]["content"] = "not reviewed\n"
                graph.update_state(config, {"object_effects": altered})
                final = graph.invoke(Command(resume="y"), config)
            self.assertEqual(final["status"], "failed", final)
            fake.execute_commands.assert_not_called()
            self.assertEqual(store.candidates(SCOPE), [])

    def test_store_failure_preserves_successful_result_without_replay(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            target = self.workspace / "notes.txt"
            fake = tracked_services(proposed_effect("create", target, content="alpha\n"))
            with patch.object(store, "apply_verified", side_effect=RuntimeError("database full")):
                state = self.run_tracked(store, fake)
            self.assertEqual(state["status"], "succeeded", state)
            self.assertTrue(state["result"]["success"])
            self.assertIn("database full", state["tracking_error"])
            self.assertEqual(target.read_bytes(), b"alpha\n")
            fake.execute_commands.assert_called_once()
            fake.correct_error.assert_not_called()

    def test_postcondition_mismatch_preserves_result_and_stops_retries(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            target = self.workspace / "notes.txt"

            def wrong_effect(_):
                target.write_bytes(b"wrong content\n")
                return {"success": True, "output": "shell reported success", "current_dir": os.getcwd()}

            fake = tracked_services(proposed_effect("create", target, content="alpha\n"), executor=wrong_effect)
            state = self.run_tracked(store, fake)
            self.assertEqual(state["status"], "failed", state)
            self.assertTrue(state["result"]["success"])
            self.assertTrue(state["tracking_error"])
            self.assertEqual(state["object_verifications"][0]["status"], "unverified")
            self.assertEqual(store.candidates(SCOPE), [])
            fake.execute_commands.assert_called_once()
            fake.correct_error.assert_not_called()

    def test_failed_execution_with_verified_effect_records_fact_without_retry(self):
        with working_directory(self.workspace):
            store = ObjectStore(self.database)
            self.addCleanup(store.close)
            effect = proposed_effect("create", self.workspace / "notes.txt", content="alpha\n")

            def partial_failure(_):
                result = apply_fixture_effect(effect)
                return {**result, "success": False, "output": "failure after file write"}

            fake = tracked_services(effect, executor=partial_failure)
            state = self.run_tracked(store, fake)
            self.assertEqual(state["status"], "failed", state)
            self.assertFalse(state["result"]["success"])
            self.assertEqual(state["object_verifications"][0]["status"], "verified")
            candidates = store.candidates(SCOPE)
            self.assertEqual(len(candidates), 1)
            self.assert_execution_provenance(state, candidates[0], store)
            fake.execute_commands.assert_called_once()
            fake.correct_error.assert_not_called()


class ObjectTrackingCliTests(unittest.TestCase):
    def test_opt_in_requires_stable_project_before_opening_store_or_executing(self):
        with patch.dict(os.environ, {"SAMANTHA_OBJECT_TRACKING": "true"}, clear=True), \
                patch("samantha.load_dotenv"), patch("samantha.sys.argv", ["samantha", "request"]), \
                patch("samantha.ObjectStore") as constructor, patch("samantha.process") as run, \
                patch("builtins.print") as output:
            self.assertEqual(main(), 1)
            constructor.assert_not_called()
            run.assert_not_called()
            self.assertIn("stable SAMANTHA_OBJECT_PROJECT", output.call_args.args[0])

    def test_explicit_object_scope_and_memory_scope_fallback_are_passed_to_process(self):
        configurations = [({"SAMANTHA_OBJECT_PROJECT": "stable-project", "SAMANTHA_OBJECT_USER": "named-user"},
                           MemoryScope("named-user", "stable-project")),
                          ({"SAMANTHA_MEMORY_PROJECT": "memory-project", "SAMANTHA_MEMORY_USER": "memory-user"},
                           MemoryScope("memory-user", "memory-project"))]
        for configuration, expected in configurations:
            with self.subTest(configuration=configuration), \
                    patch.dict(os.environ, {**configuration, "SAMANTHA_OBJECT_TRACKING": "true",
                                            "SAMANTHA_OBJECT_PATH": "test-objects.sqlite3"}, clear=True), \
                    patch("samantha.load_dotenv"), patch("samantha.sys.argv", ["samantha", "request"]), \
                    patch("samantha.ObjectStore") as constructor, \
                    patch("samantha.process", return_value={"status": "succeeded"}) as run:
                self.assertEqual(main(), 0)
                constructor.assert_called_once_with("test-objects.sqlite3")
                self.assertEqual(run.call_args.kwargs["object_scope"], expected)
                self.assertIs(run.call_args.kwargs["object_store"], constructor.return_value)
                constructor.return_value.close.assert_called_once()

    def test_default_cli_does_not_create_object_storage(self):
        with patch.dict(os.environ, {}, clear=True), patch("samantha.load_dotenv"), \
                patch("samantha.sys.argv", ["samantha", "request"]), patch("samantha.ObjectStore") as constructor, \
                patch("samantha.process", return_value={"status": "succeeded"}) as run:
            self.assertEqual(main(), 0)
            constructor.assert_not_called()
            run.assert_called_once_with("request")


def _fixture_step(phase, root):
    """A fresh Python process per invocation, with real literal Bash commands."""
    from run_commands import run_commands

    root = Path(root)
    workspace = root / "workspace"
    with working_directory(workspace):
        store = ObjectStore(root / "objects.sqlite3")
        try:
            previous = store.candidates(SCOPE)
            entity = previous[0] if previous else None
            if phase == "create":
                effect = proposed_effect("create", workspace / "notes.txt", content="alpha\n")
            elif phase == "rename":
                effect = proposed_effect("rename", entity["current_path"],
                                         destination=workspace / "todo.txt", entity=entity)
            else:
                effect = proposed_effect("append", entity["current_path"], content="beta\n", entity=entity)
            fake = tracked_services(effect, executor=run_commands)
            state = process("Create notes" if phase == "create" else "Update the file just created",
                            services=fake, input_fn=lambda _: "y", output_fn=lambda _: None,
                            log_file=root / "interactions.sqlite3", object_store=store, object_scope=SCOPE)
            candidates = store.candidates(SCOPE)
            print(json.dumps({"state": state, "entities": candidates,
                              "clarification_prompt": fake.clarify.call_args.args[0]}, ensure_ascii=False))
        finally:
            store.close()


class ObjectTrackingSubprocessTests(unittest.TestCase):
    def test_three_independent_processes_with_real_bash_executor(self):
        configured = os.environ.get("SAMANTHA_BASH")
        bash = configured or (str(Path("C:/Program Files/Git/bin/bash.exe")) if sys.platform == "win32"
                              else shutil.which("bash"))
        if not bash or not (Path(bash).is_file() or shutil.which(bash)):
            self.skipTest("A working Bash executable is required for the controlled subprocess fixture.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace").mkdir()
            keep = root / "workspace" / "keep.txt"
            keep.write_bytes(b"unchanged\n")
            outcomes = []
            environment = dict(os.environ, SAMANTHA_BASH=bash, SAMANTHA_AZURE_FALLBACK="false",
                               PYTHONIOENCODING="utf-8")
            for phase in ("create", "rename", "append"):
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                            "--fixture-step", phase, str(root)],
                                           env=environment, capture_output=True, text=True,
                                           encoding="utf-8", timeout=60, shell=False)
                self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
                outcome = json.loads(completed.stdout)
                self.assertEqual(outcome["state"]["status"], "succeeded", outcome)
                self.assertFalse(outcome["state"].get("tracking_error"), outcome)
                self.assertEqual(len(outcome["entities"]), 1)
                outcomes.append(outcome)
            self.assertEqual(len({item["entities"][0]["entity_id"] for item in outcomes}), 1)
            self.assertEqual([item["entities"][0]["version"] for item in outcomes], [1, 2, 3])
            self.assertEqual(len({item["state"]["log_session_id"] for item in outcomes}), 3)
            self.assertIn(json.dumps(str(root / "workspace" / "todo.txt"), ensure_ascii=False),
                          outcomes[-1]["clarification_prompt"])
            self.assertFalse((root / "workspace" / "notes.txt").exists())
            self.assertEqual((root / "workspace" / "todo.txt").read_bytes(), b"alpha\nbeta\n")
            self.assertEqual(keep.read_bytes(), b"unchanged\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixture-step":
        _fixture_step(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
