"""HTTP boundary tests using the actual durable runtime and native file tools."""

import http.client
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from task_agent.runtime import AgentRuntime
from task_agent.web import BODY_LIMIT, make_server


class Script:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)

    def __call__(self, state, tools):
        return next(self.decisions)


def finish(answer="Done"):
    return {"type": "finish", "answer": answer}


def write(path="note.txt", content="hello"):
    return {"type": "tool", "tool": "write_file", "args": {"path": path, "content": content}}


class TaskWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.db = self.root / "tasks.sqlite3"
        self.start(AgentRuntime(self.db, Script(finish())))

    def start(self, runtime, host="127.0.0.1"):
        self.runtime = runtime
        self.server = make_server(runtime, self.workspace, host=host)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        server, thread = self.server, self.thread
        closed = False

        def stop():
            nonlocal closed
            if not closed:
                closed = True
                server.shutdown()
                server.server_close()
                thread.join(2)
                server.worker.join(2)
        self.stop = stop
        self.addCleanup(stop)
        self.host = f"127.0.0.1:{server.server_port}"
        status, config, _ = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.token = config["csrf_token"]

    def request(self, method, path, data=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)
        defaults = {"Host": self.host}
        body = raw
        if data is not None:
            body = json.dumps(data).encode()
            defaults.update({"Content-Type": "application/json", "X-CSRF-Token": self.token})
        defaults.update(headers or {})
        connection.request(method, path, body=body, headers=defaults)
        response = connection.getresponse()
        payload = response.read()
        if response.getheader("Content-Type", "").startswith("application/json"):
            payload = json.loads(payload)
        result = response.status, payload, dict(response.getheaders())
        connection.close()
        return result

    def post(self, path, data):
        return self.request("POST", path, data)

    def wait_for(self, task_id, status):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            code, task, _ = self.request("GET", "/api/tasks/" + task_id)
            self.assertEqual(code, 200)
            if task["status"] == status and not task["runner_active"]:
                return task
            time.sleep(0.01)
        self.fail(f"Task did not reach {status}: {task}")

    def test_create_approve_and_verify_real_file(self):
        self.runtime.planner = Script(write(content="你好\n"), finish())
        code, task, _ = self.post("/api/tasks", {"request": "创建笔记"})
        self.assertEqual(code, 202)
        task = self.wait_for(task["id"], "awaiting_approval")
        self.assertFalse((self.workspace / "note.txt").exists())
        action_id = task["pending_action"]["id"]
        code, _, _ = self.post(f"/api/tasks/{task['id']}/approve", {"action_id": action_id, "approved": True})
        self.assertEqual(code, 202)
        task = self.wait_for(task["id"], "succeeded")
        self.assertEqual((self.workspace / "note.txt").read_text(encoding="utf-8"), "你好\n")
        self.assertTrue(task["actions"][0]["verification"]["ok"])
        code, _, _ = self.post(f"/api/tasks/{task['id']}/approve", {"action_id": action_id})
        self.assertEqual(code, 400)

    def test_denied_approval_never_writes(self):
        self.runtime.planner = Script(write())
        _, task, _ = self.post("/api/tasks", {"request": "Create file"})
        task = self.wait_for(task["id"], "awaiting_approval")
        self.post(f"/api/tasks/{task['id']}/approve", {"action_id": task["pending_action"]["id"], "approved": False})
        self.wait_for(task["id"], "cancelled")
        self.assertFalse((self.workspace / "note.txt").exists())

    def test_clarification_and_followup(self):
        self.runtime.planner = Script({"type": "ask", "question": "哪份文件？"}, finish(), finish())
        _, task, _ = self.post("/api/tasks", {"request": "Read that file"})
        task = self.wait_for(task["id"], "awaiting_input")
        code, _, _ = self.post(f"/api/tasks/{task['id']}/answer", {"text": "不用读取，结束即可"})
        self.assertEqual(code, 202)
        self.wait_for(task["id"], "succeeded")
        code, child, _ = self.post("/api/tasks", {"request": "Continue", "parent_id": task["id"]})
        self.assertEqual(code, 202)
        self.assertEqual(child["parent_id"], task["id"])

    def test_restart_keeps_waiting_approval_and_rotates_csrf(self):
        self.runtime.planner = Script(write())
        _, task, _ = self.post("/api/tasks", {"request": "Save later"})
        task = self.wait_for(task["id"], "awaiting_approval")
        old_token = self.token
        self.stop()
        self.start(AgentRuntime(self.db, Script(finish())))
        _, restored, _ = self.request("GET", f"/api/tasks/{task['id']}")
        self.assertEqual(restored["pending_action"]["id"], task["pending_action"]["id"])
        self.assertNotEqual(self.token, old_token)
        code, _, _ = self.request("POST", f"/api/tasks/{task['id']}/approve", {"action_id": task["pending_action"]["id"]}, headers={"X-CSRF-Token": old_token})
        self.assertEqual(code, 403)
        self.post(f"/api/tasks/{task['id']}/approve", {"action_id": task["pending_action"]["id"]})
        self.wait_for(task["id"], "succeeded")

    def test_runner_is_background_deduplicated_and_cooperatively_cancelled(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def slow(state, tools):
            calls.append(1)
            entered.set()
            release.wait(3)
            return write()
        self.runtime.planner = slow
        self.addCleanup(release.set)
        _, task, _ = self.post("/api/tasks", {"request": "Wait"})
        self.assertTrue(entered.wait(1))
        for _ in range(3):
            self.assertEqual(self.post(f"/api/tasks/{task['id']}/run", {})[0], 202)
        self.assertEqual(self.request("GET", "/api/tasks")[0], 200)
        code, task, _ = self.post(f"/api/tasks/{task['id']}/cancel", {})
        self.assertEqual(code, 200)
        self.assertTrue(task["cancel_requested"])
        release.set()
        self.wait_for(task["id"], "cancelled")
        self.assertEqual(calls, [1])
        self.assertFalse((self.workspace / "note.txt").exists())

    def test_host_origin_and_csrf_are_enforced(self):
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/config", headers={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/config", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request("POST", "/api/tasks", {"request": "No"}, headers={"X-CSRF-Token": ""})[0], 403)
        self.assertEqual(self.request("POST", "/api/tasks", {"request": "No"}, headers={"Origin": "null"})[0], 403)
        self.assertEqual(self.request("POST", "/api/tasks", {"request": "Yes"}, headers={"Origin": "http://" + self.host})[0], 202)

    def test_rejected_request_with_delayed_body_gets_json_instead_of_reset(self):
        # Browsers/HTTP clients may send headers and the body in separate TCP writes.
        # Closing on the rejected headers before consuming that body can reset the
        # connection on Windows and hide the intended JSON 403 from the caller.
        payload = json.dumps({"request": "x" * 4000}).encode()
        for rejected in ({"Origin": "null"}, {"X-CSRF-Token": "invalid"}):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            try:
                connection.putrequest("POST", "/api/tasks")
                headers = {"Content-Type": "application/json", "Content-Length": str(len(payload)),
                           "X-CSRF-Token": self.token, **rejected}
                for key, value in headers.items():
                    connection.putheader(key, value)
                connection.endheaders()
                time.sleep(0.03)
                connection.send(payload)
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                self.assertTrue(json.loads(response.read())["error"])
            finally:
                connection.close()

    def test_rejected_incomplete_body_has_bounded_wait(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.putrequest("POST", "/api/tasks")
            connection.putheader("Content-Length", "4096")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Origin", "null")
            started = time.monotonic()
            connection.endheaders()
            # No body ever arrives: error cleanup must not wait indefinitely.
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            self.assertTrue(json.loads(response.read())["error"])
            self.assertLess(time.monotonic() - started, 2)
        finally:
            connection.close()

    def test_workspace_scope_applies_to_create_list_get_and_mutation(self):
        outside = self.root / "outside"
        outside.mkdir()
        foreign = self.runtime.create_task("Private", outside)
        self.assertEqual(self.post("/api/tasks", {"request": "Escape", "workspace": str(outside)})[0], 403)
        self.assertEqual(self.post("/api/tasks", {"request": "Escape", "workspace": ".."})[0], 403)
        self.assertEqual(self.request("GET", f"/api/tasks/{foreign['id']}")[0], 403)
        self.assertEqual(self.post(f"/api/tasks/{foreign['id']}/run", {})[0], 403)
        self.assertEqual(self.post("/api/tasks", {"request": "Child", "parent_id": foreign["id"]})[0], 403)
        _, tasks, _ = self.request("GET", "/api/tasks")
        self.assertEqual(tasks["tasks"], [])
        child = self.workspace / "child"
        child.mkdir()
        self.assertEqual(self.post("/api/tasks", {"request": "Valid", "workspace": "child"})[0], 202)

    def test_hidden_child_workspace_cannot_bypass_hidden_tool_paths(self):
        hidden = self.workspace / ".git"
        hidden.mkdir()
        ordinary = self.workspace / "project"
        ordinary.mkdir()
        nested_hidden = ordinary / ".private"
        nested_hidden.mkdir()
        nested_child = nested_hidden / "child"
        nested_child.mkdir()
        hidden_task = self.runtime.create_task("Hidden root", hidden)
        for path in (".git", str(hidden), "project/.private/child", ".git/../project"):
            self.assertEqual(self.post("/api/tasks", {"request": "No", "workspace": path})[0], 403)
        self.assertEqual(self.request("GET", f"/api/tasks/{hidden_task['id']}")[0], 403)
        _, data, _ = self.request("GET", "/api/tasks")
        self.assertNotIn(hidden_task["id"], [task["id"] for task in data["tasks"]])
        self.assertEqual(self.post("/api/tasks", {"request": "Allowed", "workspace": "project"})[0], 202)

    def test_explicit_hidden_server_root_remains_valid(self):
        self.stop()
        self.workspace = self.root / ".samantha" / "workspace"
        self.workspace.mkdir(parents=True)
        self.start(AgentRuntime(self.db, Script(finish())))
        code, task, _ = self.post("/api/tasks", {"request": "Allowed"})
        self.assertEqual(code, 202)
        self.wait_for(task["id"], "succeeded")

    def test_child_symlink_or_reparse_root_is_rejected_before_resolving(self):
        # Windows can require privileges for real symlink creation. A mocked
        # lstat reparse flag covers this boundary deterministically on all hosts.
        child = self.workspace / "linked"
        child.mkdir()
        real_lstat = Path.lstat
        def lstat(path):
            if path == child:
                class ReparseInfo:
                    st_mode = 0o40755
                    st_file_attributes = 0x400
                return ReparseInfo()
            return real_lstat(path)
        with patch.object(Path, "lstat", lstat):
            self.assertEqual(self.post("/api/tasks", {"request": "No", "workspace": "linked"})[0], 403)

    def test_invalid_json_limits_and_inputs(self):
        headers = {"X-CSRF-Token": self.token, "Content-Type": "application/json"}
        for raw in (b"{", b"[]", b'{"request":"a","request":"b"}', b'{"request":NaN}'):
            self.assertEqual(self.request("POST", "/api/tasks", raw=raw, headers=headers)[0], 400)
        self.assertEqual(self.request("POST", "/api/tasks", raw=b"{}", headers={**headers, "Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/tasks", raw=b" " * (BODY_LIMIT + 1), headers=headers)[0], 413)
        for data in ({"request": ""}, {"request": "x" * 12001}, {"request": "x", "max_steps": True}, {"request": "x", "max_steps": 33}, {"request": "x", "workspace": []}):
            self.assertEqual(self.post("/api/tasks", data)[0], 400)

    def test_static_security_headers_no_arbitrary_file_server(self):
        code, page, headers = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("本地任务工作台".encode(), page)
        code, script, _ = self.request("GET", "/app.js")
        self.assertEqual(code, 200)
        self.assertNotIn(b"innerHTML", script)
        self.assertIn(b"textContent", script)
        self.assertEqual(self.request("GET", "/../web.py")[0], 404)
        self.assertEqual(self.request("GET", "/%2e%2e/web.py")[0], 404)

    def test_static_dom_references_and_accessible_form_contract(self):
        # Browser automation is unavailable in this environment; this catches drift
        # between the shipped DOM and handlers without pretending to render the UI.
        _, page, _ = self.request("GET", "/")
        _, script, _ = self.request("GET", "/app.js")
        html, javascript = page.decode("utf-8"), script.decode("utf-8")
        ids = re.findall(r'\bid="([^"\s]+)"', html)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(re.findall(r'\$\("([^"]+)"\)', javascript)).issubset(set(ids)))
        self.assertIn('lang="zh-CN"', html)
        self.assertIn('maxlength="12000"', html)
        self.assertIn('max="32"', html)
        self.assertIn('for="request"', html)

    def test_configuration_does_not_claim_connectivity_or_expose_secrets(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "test-model", "QWEN_API_KEY": "never-expose", "QWEN_BASE_URL": "https://secret:password@example.test/v1?key=secret"}):
            _, config, _ = self.request("GET", "/api/config")
        self.assertEqual(config["mode"], "live")
        self.assertEqual(config["model"], "test-model")
        self.assertFalse(config["connection_verified"])
        self.assertNotIn("never-expose", json.dumps(config))
        self.assertNotIn("password", json.dumps(config))

    def test_docker_binding_keeps_local_browser_host_and_csrf_checks(self):
        self.stop()
        self.start(AgentRuntime(self.db, Script(finish())), host="0.0.0.0")
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/api/config", headers={
            "Host": f"localhost:{self.server.server_port}"})[0], 200)
        for host in ("0.0.0.0", "192.168.1.2", "example.com"):
            self.assertEqual(self.request("GET", "/api/config", headers={
                "Host": f"{host}:{self.server.server_port}"})[0], 403)
        self.assertEqual(self.post("/api/tasks", {"request": "Run"})[0], 202)
        self.assertEqual(self.request("POST", "/api/tasks", {"request": "Run"},
                                      headers={"X-CSRF-Token": "invalid"})[0], 403)
        self.assertEqual(self.request("POST", "/api/tasks", {"request": "Run"},
                                      headers={"Origin": "http://example.com"})[0], 403)

    def test_other_non_loopback_binding_refused(self):
        for host in ("192.168.1.2", "example.com", "::1"):
            with self.assertRaises(ValueError):
                make_server(self.runtime, self.workspace, host=host)

    def test_model_failure_is_visible_and_not_fake_success(self):
        def failing(state, tools):
            raise RuntimeError("Model connection failed")
        self.runtime.planner = failing
        _, task, _ = self.post("/api/tasks", {"request": "Run"})
        task = self.wait_for(task["id"], "failed")
        self.assertIn("Model connection failed", task["error"])
        self.assertEqual(task["answer"], "")


if __name__ == "__main__":
    unittest.main()
