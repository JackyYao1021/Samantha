"""Real native tools, persisted tasks, process crashes and approval binding."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from task_agent.runtime import AgentRuntime
from task_agent.store import process_lock
from task_agent.planner import continuation_messages, decision_schema, strict_json
from task_agent.tools import ToolRegistry


def tool(name, **args):
    return {"type": "tool", "tool": name, "args": args, "reason": "test action"}


def finish():
    return {"type": "finish", "answer": "Completed."}


class Script:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.contexts = []

    def __call__(self, state, tools):
        self.contexts.append(state)
        return next(self.decisions)


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.db = self.root / "tasks.sqlite3"

    def runtime(self, *decisions, **kwargs):
        return AgentRuntime(self.db, Script(*decisions), **kwargs)

    def create(self, runtime, request="test"):
        return runtime.create_task(request, self.workspace)

    def approve(self, runtime, task):
        return runtime.approve(task["id"], task["pending_action"]["id"])

    def test_restart_waiting_for_approval_then_executes_exactly_once(self):
        runtime = self.runtime(tool("write_file", path="notes.txt", content="你好\n"))
        task = runtime.run(self.create(runtime)["id"])
        self.assertEqual(task["status"], "awaiting_approval")
        self.assertFalse((self.workspace / "notes.txt").exists())
        restored = self.runtime(finish())
        self.approve(restored, task)
        task = restored.run(task["id"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual((self.workspace / "notes.txt").read_bytes(), "你好\n".encode())
        self.assertEqual(task["actions"][0]["status"], "verified")
        self.assertEqual(restored.run(task["id"])["steps"], task["steps"])
        with self.assertRaises(ValueError):
            restored.approve(task["id"], task["actions"][0]["id"])

    def test_crash_after_append_reconciles_without_repeating(self):
        (self.workspace / "notes.txt").write_bytes(b"alpha\n")
        runtime = self.runtime(tool("append_file", path="notes.txt", content="beta\n"))
        task = runtime.run(self.create(runtime)["id"])
        self.approve(runtime, task)
        # A genuine process exit tests OS-lock release and SQLite crash durability.
        code = ("import sys,os;sys.path.insert(0,sys.argv[1]);"
                "from task_agent.runtime import AgentRuntime;"
                "r=AgentRuntime(sys.argv[2],planner=lambda *_:None,"
                "fault_hook=lambda phase,state: os._exit(87) if phase=='after_execute' else None);"
                "r.run(sys.argv[3])")
        child = subprocess.run([sys.executable, "-c", code, str(SRC), str(self.db), task["id"]],
                               capture_output=True, timeout=30)
        self.assertEqual(child.returncode, 87, child.stderr.decode(errors="replace"))
        self.assertEqual(runtime.get_task(task["id"])["pending_action"]["status"], "executing")
        restored = self.runtime(finish())
        result = restored.run(task["id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual((self.workspace / "notes.txt").read_bytes(), b"alpha\nbeta\n")
        self.assertTrue(any(e["kind"] == "action_recovered" for e in result["events"]))

    def test_crash_before_effect_requires_fresh_confirmation(self):
        class Crash(BaseException):
            pass
        def fail(phase, state):
            if phase == "before_execute":
                raise Crash()
        runtime = self.runtime(tool("write_file", path="a.txt", content="x"), fault_hook=fail)
        task = runtime.run(self.create(runtime)["id"])
        self.approve(runtime, task)
        with self.assertRaises(Crash):
            runtime.run(task["id"])
        restored = self.runtime(finish())
        result = restored.run(task["id"])
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertFalse((self.workspace / "a.txt").exists())
        self.approve(restored, result)
        self.assertEqual(restored.run(task["id"])["status"], "succeeded")

    def test_changed_file_after_approval_fails_without_overwrite(self):
        target = self.workspace / "notes.txt"
        target.write_bytes(b"old")
        runtime = self.runtime(tool("append_file", path="notes.txt", content="new"))
        task = runtime.run(self.create(runtime)["id"])
        self.approve(runtime, task)
        target.write_bytes(b"external")
        result = runtime.run(task["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(target.read_bytes(), b"external")

    def test_rejection_never_runs_and_cannot_reuse_approval(self):
        runtime = self.runtime(tool("write_file", path="a.txt", content="x"))
        task = runtime.run(self.create(runtime)["id"])
        runtime.approve(task["id"], task["pending_action"]["id"], False)
        self.assertEqual(runtime.run(task["id"])["status"], "cancelled")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_clarification_survives_new_runtime(self):
        runtime = self.runtime({"type": "ask", "question": "文件名？"})
        task = runtime.run(self.create(runtime)["id"])
        planner = Script(tool("write_file", path="a.txt", content="x"), finish())
        runtime = AgentRuntime(self.db, planner)
        runtime.answer(task["id"], "a.txt")
        task = runtime.run(task["id"])
        self.assertEqual(planner.contexts[0]["clarifications"][0]["answer"], "a.txt")
        self.approve(runtime, task)
        self.assertEqual(runtime.run(task["id"])["status"], "succeeded")

    def test_read_result_drives_next_tool(self):
        (self.workspace / "input.txt").write_bytes(b"evidence")
        def planner(state, schemas):
            if not state["observations"]:
                return tool("read_file", path="input.txt")
            if not state["verified_actions"]:
                self.assertTrue(state["observations"][-1]["ok"])
                return tool("write_file", path="report.txt", content="observed")
            return finish()
        runtime = AgentRuntime(self.db, planner)
        task = runtime.run(self.create(runtime)["id"])
        self.assertEqual(task["actions"][0]["status"], "observed")
        self.approve(runtime, task)
        self.assertEqual(runtime.run(task["id"])["status"], "succeeded")

    def test_budget_is_persistent(self):
        runtime = self.runtime(tool("list_files"))
        task = runtime.create_task("inspect", self.workspace, max_steps=1)
        task = runtime.run(task["id"])
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["steps"], 1)
        self.assertIn("budget", task["error"])

    def test_unknown_tool_cannot_become_silent_success(self):
        runtime = self.runtime(tool("shell", command="anything"), finish())
        task = runtime.run(self.create(runtime)["id"])
        self.assertEqual(task["status"], "failed")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_follow_up_scope_and_verified_evidence(self):
        runtime = self.runtime(tool("write_file", path="a.txt", content="hello"), finish())
        task = runtime.run(self.create(runtime)["id"])
        self.approve(runtime, task)
        task = runtime.run(task["id"])
        follow = runtime.create_task("append to that file", self.workspace, parent_id=task["id"])
        self.assertEqual(follow["prior_evidence"][0]["args"]["path"], "a.txt")
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            runtime.create_task("follow", other, parent_id=task["id"])

    def test_concurrent_runner_rejected_and_cancel_survives_active_planner(self):
        started, release = threading.Event(), threading.Event()
        def planner(*_):
            started.set()
            release.wait(5)
            return tool("write_file", path="a.txt", content="x")
        runtime = AgentRuntime(self.db, planner)
        task = self.create(runtime)
        thread = threading.Thread(target=runtime.run, args=(task["id"],))
        thread.start()
        self.assertTrue(started.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                self.runtime().run(task["id"])
            self.assertTrue(runtime.cancel(task["id"])["cancel_requested"])
        finally:
            release.set()
            thread.join(10)
        self.assertEqual(runtime.get_task(task["id"])["status"], "cancelled")
        self.assertFalse((self.workspace / "a.txt").exists())

    def test_strict_json_rejects_duplicate_keys(self):
        with self.assertRaises(ValueError):
            strict_json('{"type":"finish","type":"tool"}')

    def test_missing_workspace_is_persisted_as_failure(self):
        runtime = self.runtime(finish())
        task = self.create(runtime)
        self.workspace.rmdir()
        task = runtime.run(task["id"])
        self.assertEqual(task["status"], "failed")
        self.assertTrue(any(e["kind"] == "task_failed" for e in task["events"]))

    def test_context_packing_preserves_user_constraints(self):
        runtime = self.runtime()
        state = self.create(runtime)
        state["observations"] = [{"text": "x" * 10000} for _ in range(15)]
        context = runtime._context(state)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False)), 24000)
        self.assertEqual(context["request"], "test")
        self.assertGreater(context["omitted_observations"], 0)
        state["clarifications"] = [{"question": "q", "answer": "a" * 12000} for _ in range(3)]
        with self.assertRaises(ValueError):
            runtime._context(state)

    def test_planner_receives_completed_effects_as_conversation(self):
        state = {"request": "append once", "observations": [{"tool": "append_file",
                 "args": {"path": "a.txt", "content": "x"}, "ok": True,
                 "result": {"recovered": True, "verification": {"ok": True,
                            "details": {"a.txt": {"actual": {"inode": 123}}}}}}],
                 "verified_actions": [{"tool": "append_file", "args": {"path": "a.txt", "content": "x"}}]}
        messages, final = continuation_messages(state, [])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        result = json.loads(messages[-1]["content"])["tool_result"]
        self.assertEqual(result["status"], "completed_and_verified")
        self.assertTrue(result["recovered_after_restart"])
        self.assertNotIn("inode", messages[-1]["content"])
        self.assertEqual(json.loads(final)["completed_and_verified"], state["verified_actions"])

    def test_output_schema_limits_selection_to_available_tools(self):
        schema = decision_schema(ToolRegistry.schemas())
        variants = schema["anyOf"]
        tools = {s["properties"]["tool"]["const"] for s in variants if "tool" in s["properties"]}
        self.assertEqual(tools, {s["function"]["name"] for s in ToolRegistry.schemas()})
        self.assertTrue(all(s["additionalProperties"] is False for s in variants))
        self.assertNotIn("shell", tools)


if __name__ == "__main__":
    unittest.main()
