"""Budget boundaries, measured tool requests and per-instance transport timeouts."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chat import Chat
from model_telemetry import model_request
from task_agent.planner import ModelPlanner
from task_agent.runtime import AgentRuntime
from task_agent.tools import ToolRegistry


def write_decision():
    return {"type": "tool", "tool": "write_file", "args": {"path": "report.txt", "content": "once"}}


class TaskBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.db = self.root / "tasks.sqlite3"
        self.clock = [0.0]

    def change_budget(self, runtime, task, *, maximum=None, active=None):
        state = runtime.store.read(task["id"], events=False)
        if maximum is not None:
            state["max_active_seconds"] = maximum
        if active is not None:
            state["active_seconds"] = active
        runtime.store.save(state, "test_budget")

    def test_over_budget_planning_cannot_propose_or_execute_a_write(self):
        def planner(*_):
            self.clock[0] += 2
            return write_decision()
        runtime = AgentRuntime(self.db, planner)
        task = runtime.create_task("write", self.workspace)
        self.change_budget(runtime, task, maximum=1)
        with patch("task_agent.runtime.time.monotonic", side_effect=lambda: self.clock[0]):
            result = runtime.run(task["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("budget", result["error"])
        self.assertEqual(result["active_seconds"], 2)
        self.assertIsNone(result["pending_action"])
        self.assertFalse((self.workspace / "report.txt").exists())

    def test_approved_mutation_cannot_start_after_persisted_time_budget_is_exhausted(self):
        runtime = AgentRuntime(self.db, lambda *_: write_decision())
        task = runtime.run(runtime.create_task("write", self.workspace)["id"])
        runtime.approve(task["id"], task["pending_action"]["id"])
        self.change_budget(runtime, task, maximum=1, active=1)
        result = AgentRuntime(self.db, lambda *_: self.fail("must not plan")).run(task["id"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.workspace / "report.txt").exists())

    def test_embedding_tool_time_and_physical_requests_are_accounted(self):
        clock = self.clock
        class Registry(ToolRegistry):
            def _search_content(self, query, limit):
                for physical in (True, False):
                    with model_request(provider="test", transport="ollama", model="test-embed",
                                       parameters={}, call_type="embedding", physical_request=physical):
                        clock[0] += 2
                return {"ok": True, "results": []}
        runtime = AgentRuntime(self.db, lambda *_: {"type": "tool", "tool": "search_content",
                                                    "args": {"query": "test"}}, registry_factory=Registry)
        task = runtime.create_task("search", self.workspace)
        self.change_budget(runtime, task, maximum=1)
        with patch("task_agent.runtime.time.monotonic", side_effect=lambda: self.clock[0]):
            result = runtime.run(task["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(result["model_call_records"]), 2)
        self.assertEqual(result["active_seconds"], 4)
        self.assertEqual(result["steps"], 1)

    def test_preparation_execution_and_verification_are_charged_once_each(self):
        clock = self.clock
        class Registry(ToolRegistry):
            def prepare(self, *args):
                clock[0] += 1
                return super().prepare(*args)
            def execute(self, *args):
                clock[0] += 2
                return super().execute(*args)
            def verify(self, *args):
                clock[0] += .5
                return super().verify(*args)
        decisions = iter([write_decision(), {"type": "finish", "answer": "done"}])
        runtime = AgentRuntime(self.db, lambda *_: next(decisions), registry_factory=Registry)
        task = runtime.create_task("write", self.workspace)
        with patch("task_agent.runtime.time.monotonic", side_effect=lambda: self.clock[0]):
            task = runtime.run(task["id"])
            runtime.approve(task["id"], task["pending_action"]["id"])
            result = runtime.run(task["id"])
        self.assertEqual(result["status"], "succeeded")
        # Preparation 1 + execute 2 (plus its internal verification .5)
        # + independent runtime verification .5; no overlapping timer charges.
        self.assertEqual(result["active_seconds"], 4)
        self.assertEqual(sum(event["payload"].get("active_seconds", 0) for event in result["events"]), 4)

    def test_recovery_still_verifies_applied_effect_after_time_budget_exhaustion(self):
        class Crash(BaseException):
            pass
        def crash(phase, state):
            if phase == "after_execute":
                raise Crash()
        runtime = AgentRuntime(self.db, lambda *_: write_decision(), fault_hook=crash)
        task = runtime.run(runtime.create_task("write", self.workspace)["id"])
        runtime.approve(task["id"], task["pending_action"]["id"])
        with self.assertRaises(Crash):
            runtime.run(task["id"])
        self.change_budget(runtime, task, maximum=1, active=1)
        result = AgentRuntime(self.db, lambda *_: self.fail("must not plan")).run(task["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["actions"][0]["status"], "verified")
        self.assertIsNone(result["pending_action"])
        self.assertEqual((self.workspace / "report.txt").read_bytes(), b"once")
        self.assertTrue(any(event["kind"] == "action_recovered" for event in result["events"]))

    def test_cancellation_during_preparation_is_observed_before_approval(self):
        runtime = None
        task = None
        class Registry(ToolRegistry):
            def prepare(self, *args):
                prepared = super().prepare(*args)
                runtime.store.request_cancel(task["id"])
                return prepared
        runtime = AgentRuntime(self.db, lambda *_: write_decision(), registry_factory=Registry)
        task = runtime.create_task("write", self.workspace)
        result = runtime.run(task["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["pending_action"])
        self.assertFalse((self.workspace / "report.txt").exists())

    def test_model_planner_caps_per_instance_timeout_without_mutating_environment(self):
        with patch.dict(os.environ, {"QWEN_TIMEOUT": "120"}), patch("task_agent.planner.Chat") as chat:
            chat.return_value.chat_context.return_value = '{"type":"finish","answer":"done"}'
            ModelPlanner()({"remaining_seconds": .25}, [])
            self.assertEqual(chat.call_args.kwargs["request_timeout"], .25)
            self.assertEqual(os.environ["QWEN_TIMEOUT"], "120")
            chat.return_value.client_qwen.close.assert_called_once()

    def test_native_and_openai_transports_receive_the_same_explicit_timeout(self):
        with patch.dict(os.environ, {"QWEN_API_MODE": "ollama", "QWEN_TIMEOUT": "120"}), \
                patch("chat.OpenAI") as qwen, patch("chat.httpx.Client") as http:
            qwen.return_value.base_url = "http://localhost:11434/v1"
            http.return_value.__enter__.return_value.post.return_value.json.return_value = {
                "done": True, "done_reason": "stop", "message": {"content": "done"}}
            chat = Chat(request_timeout=.5)
            self.assertEqual(chat.chat_context("hi"), "done")
            self.assertEqual(qwen.call_args.kwargs["timeout"], .5)
            http.assert_called_once_with(timeout=.5, trust_env=False)
            self.assertEqual(os.environ["QWEN_TIMEOUT"], "120")


if __name__ == "__main__":
    unittest.main()
