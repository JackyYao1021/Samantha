"""Public CLI/HTTP entry points share the opt-in durable intent runtime."""

from contextlib import redirect_stdout
import http.client
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from task_agent import cli
from task_agent.runtime import AgentRuntime
from task_agent.web import make_server


class IntentInterfaceTests(unittest.TestCase):
    def test_cli_explicit_mode_is_forwarded_for_new_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "AgentRuntime") as runtime:
            runtime.return_value.create_task.return_value = {"id": "task"}
            runtime.return_value.run.return_value = {"id": "task", "status": "succeeded"}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--db", "state.sqlite3", "run", "read note.txt",
                                           "--workspace", directory, "--intent-mode", "hierarchical"]), 0)
            runtime.assert_called_once_with("state.sqlite3", intent_mode="hierarchical")

    def test_cli_server_forwards_default_for_new_web_tasks(self):
        with tempfile.TemporaryDirectory() as directory, patch("task_agent.web.serve") as serve:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--db", "state.sqlite3", "serve", "--workspace", directory,
                                           "--intent-mode", "hierarchical"]), 0)
            self.assertEqual(serve.call_args.kwargs["intent_mode"], "hierarchical")

    def test_cli_correction_uses_saved_task_instead_of_new_request(self):
        with patch.object(cli, "AgentRuntime") as runtime:
            runtime.return_value.revise.return_value = {"id": "task"}
            runtime.return_value.run.return_value = {"id": "task", "status": "awaiting_approval"}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["revise", "task", "copy, preserve the original"]), 0)
            runtime.return_value.revise.assert_called_once_with("task", "copy, preserve the original")
            runtime.return_value.create_task.assert_not_called()

    def test_web_fallback_retains_approval_and_persisted_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            calls = []

            def malformed(context, schemas):
                calls.append(context["request"])
                return {"schema_version": "invalid"}

            decisions = iter([
                {"type": "tool", "tool": "write_file", "args": {"path": "notes.txt", "content": "hello"}},
                {"type": "finish", "answer": "done"},
            ])
            runtime = AgentRuntime(root / "tasks.sqlite3", lambda *_: next(decisions),
                                   intent_mode="hierarchical", intent_classifier=malformed)
            server = make_server(runtime, workspace)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            thread.start()

            def request(method, path, body=None, token=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=4)
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-CSRF-Token"] = token
                try:
                    connection.request(method, path, json.dumps(body) if body is not None else None, headers)
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    connection.close()

            def wait_for(identifier, status):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    code, task = request("GET", "/api/tasks/" + identifier)
                    self.assertEqual(code, 200)
                    if task["status"] == status and not task["runner_active"]:
                        return task
                    time.sleep(.01)
                self.fail("Web worker did not reach " + status)

            try:
                code, config = request("GET", "/api/config")
                self.assertEqual((code, config["intent_mode"]), (200, "hierarchical"))
                code, task = request("POST", "/api/tasks", {"request": "Create notes.txt containing hello"}, config["csrf_token"])
                self.assertEqual(code, 202)
                task = wait_for(task["id"], "awaiting_approval")
                self.assertFalse((workspace / "notes.txt").exists())
                self.assertEqual(task["intent_mode"], "hierarchical")
                code, _ = request("POST", "/api/tasks/" + task["id"] + "/approve",
                                  {"action_id": task["pending_action"]["id"], "approved": True}, config["csrf_token"])
                self.assertEqual(code, 202)
                final = wait_for(task["id"], "succeeded")
                self.assertEqual((workspace / "notes.txt").read_bytes(), b"hello")
                self.assertEqual(len(calls), 1)
                with patch.dict(os.environ, {"SAMANTHA_INTENT_MODE": "baseline"}):
                    restored = AgentRuntime(root / "tasks.sqlite3").get_task(final["id"])
                self.assertEqual(restored["intent_mode"], "hierarchical")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)
                server.worker.join(2)


if __name__ == "__main__":
    unittest.main()
