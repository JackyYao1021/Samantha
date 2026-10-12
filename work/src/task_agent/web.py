"""Local, dependency-free task console. Model and file operations run off HTTP threads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import secrets
import stat
import threading
import time
from urllib.parse import urlsplit


BODY_LIMIT = 64 * 1024
# A small rejection-only margin lets just-over-limit requests receive their 413.
REJECT_DRAIN_LIMIT = BODY_LIMIT + 8192
STATIC = Path(__file__).with_name("static")
STATIC_ROUTES = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}


class HTTPProblem(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class TaskServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, runtime, workspace, host, port):
        if host not in {"localhost", "0.0.0.0"}:
            try:
                address = ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError("Use an IPv4 loopback address or 0.0.0.0 for Docker") from exc
            if address.version != 4 or not address.is_loopback:
                raise ValueError("Use an IPv4 loopback address or 0.0.0.0 for Docker")
        self.runtime = runtime
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.csrf_token = secrets.token_urlsafe(32)
        self.jobs = queue.Queue()
        self.pending = set()
        self.worker_errors = {}
        self.worker_guard = threading.Lock()
        self.stopping = threading.Event()
        super().__init__((host, port), TaskHandler)
        # Binding all container interfaces does not expand trusted browser hosts.
        browser_hosts = ({"127.0.0.1", "localhost"} if host == "0.0.0.0"
                         else {host, "localhost", self.server_address[0]})
        self.allowed_hosts = {f"{value}:{self.server_port}" for value in browser_hosts}
        self.worker = threading.Thread(target=self._work, name="samantha-task-runner", daemon=True)
        self.worker.start()

    def within_workspace(self, value):
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.workspace / path
        try:
            relative = path.relative_to(self.workspace)
        except ValueError:
            raise HTTPProblem(403, "任务目录必须位于此服务的工作区内。")
        # The configured root can intentionally contain hidden ancestors. Child
        # roots selected through HTTP cannot elevate a hidden path to a tool root.
        cursor = self.workspace
        for part in relative.parts:
            if part.startswith("."):
                raise HTTPProblem(403, "不能选择隐藏目录或包含上级跳转的任务工作区。")
            cursor /= part
            try:
                info = cursor.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise HTTPProblem(403, "不能选择符号链接或重解析点作为任务工作区。")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise HTTPProblem(403, "任务目录必须位于此服务的工作区内。")
        return resolved

    def task(self, task_id):
        state = self.runtime.get_task(task_id)
        self.within_workspace(state["workspace"])
        with self.worker_guard:
            return {**state, "runner_active": task_id in self.pending,
                    "runner_error": self.worker_errors.get(task_id)}

    def schedule(self, task_id):
        with self.worker_guard:
            if task_id not in self.pending:
                self.pending.add(task_id)
                self.worker_errors.pop(task_id, None)
                self.jobs.put(task_id)

    def _work(self):
        while not self.stopping.is_set():
            task_id = self.jobs.get()
            if task_id is None:
                return
            try:
                # Recheck the workspace just before execution, including resolved symlinks.
                self.task(task_id)
                self.runtime.run(task_id)
            except Exception as exc:
                with self.worker_guard:
                    self.worker_errors[task_id] = str(exc)
            finally:
                with self.worker_guard:
                    self.pending.discard(task_id)
                self.jobs.task_done()

    def server_close(self):
        self.stopping.set()
        self.jobs.put(None)
        super().server_close()


class TaskHandler(BaseHTTPRequestHandler):
    server_version = "SamanthaLocal/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # Prompts, task IDs and local paths need not appear in web access logs.
        pass

    def respond(self, status, data, content_type="application/json; charset=utf-8"):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self'; img-src 'self'; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def security(self, *, mutation=False):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in self.server.allowed_hosts:
            raise HTTPProblem(403, "不允许的 Host。请通过控制台显示的本地地址访问。")
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != "http://" + hosts[0]):
            raise HTTPProblem(403, "不允许的跨来源请求。")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise HTTPProblem(403, "不允许的跨站点请求。")
        if mutation:
            token = self.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(token, self.server.csrf_token):
                raise HTTPProblem(403, "会话已失效，请刷新页面后重试。")

    def body(self):
        if self.headers.get("Transfer-Encoding"):
            raise HTTPProblem(400, "不支持分块请求。")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise HTTPProblem(400, "请求需要唯一的 Content-Length。")
        try:
            length = int(lengths[0])
        except ValueError:
            raise HTTPProblem(400, "无效的请求长度。")
        if length < 0 or length > BODY_LIMIT:
            raise HTTPProblem(413, "请求内容过大。")
        if self.headers.get_content_type() != "application/json":
            raise HTTPProblem(415, "请使用 application/json。")
        self._body_consumed = True
        try:
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("duplicate key")
                    result[key] = value
                return result
            payload = json.loads(self.rfile.read(length).decode("utf-8"), object_pairs_hook=pairs,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, ValueError):
            raise HTTPProblem(400, "请求必须是有效的 JSON。")
        if not isinstance(payload, dict):
            raise HTTPProblem(400, "请求必须是 JSON 对象。")
        return payload

    def discard_rejected_body(self):
        """Avoid TCP reset on rejected POSTs without accepting an unbounded drain.

        HTTP clients can send headers and bodies separately. Closing while a normal
        body is still arriving discards the JSON error on Windows. Consume only a
        known, bounded Content-Length, under a total half-second deadline; the
        request is never parsed or authorized by this cleanup.
        """
        if getattr(self, "_body_consumed", False) or self.headers.get("Transfer-Encoding"):
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            return
        try:
            remaining = int(lengths[0])
        except ValueError:
            return
        if not 0 < remaining <= REJECT_DRAIN_LIMIT:
            return
        deadline = time.monotonic() + 0.5
        original_timeout = self.connection.gettimeout()
        try:
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass
        finally:
            self.connection.settimeout(original_timeout)
            self._body_consumed = True

    @staticmethod
    def text_field(body, key, limit=12000):
        value = body.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise HTTPProblem(400, f"{key} 必须是 1–{limit} 字符的文本。")
        return value

    def dispatch(self, *, mutation=False):
        try:
            self.security(mutation=mutation)
            path = urlsplit(self.path).path
            if not mutation:
                self.read_route(path)
            else:
                self.write_route(path, self.body())
        except HTTPProblem as exc:
            self.discard_rejected_body()
            self.respond(exc.status, {"error": exc.message})
        except KeyError:
            self.respond(404, {"error": "找不到此任务。"})
        except (ValueError, TypeError) as exc:
            self.respond(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.respond(409, {"error": str(exc)})
        except Exception:
            self.respond(500, {"error": "本地服务发生错误。请检查服务终端后重试。"})

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch(mutation=True)

    def read_route(self, path):
        if path in STATIC_ROUTES:
            filename, content_type = STATIC_ROUTES[path]
            return self.respond(200, (STATIC / filename).read_bytes(), content_type)
        if path == "/api/config":
            model = os.environ.get("QWEN_MODEL", "qwen3:4b")
            configured = bool(model.strip() and os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:11434/v1").strip())
            return self.respond(200, {"csrf_token": self.server.csrf_token,
                                      "workspace": str(self.server.workspace), "mode": "live",
                                      "model": model, "model_configured": configured,
                                      "connection_verified": False})
        if path == "/api/tasks":
            tasks = []
            for task in self.server.runtime.list_tasks():
                try:
                    self.server.within_workspace(task["workspace"])
                except HTTPProblem:
                    continue
                with self.server.worker_guard:
                    task = {**task, "runner_active": task["id"] in self.server.pending}
                tasks.append(task)
            return self.respond(200, {"tasks": tasks})
        match = re.fullmatch(r"/api/tasks/([A-Za-z0-9_-]{1,128})", path)
        if match:
            return self.respond(200, self.server.task(match[1]))
        raise HTTPProblem(404, "找不到此地址。")

    def write_route(self, path, body):
        runtime = self.server.runtime
        if path == "/api/tasks":
            request = self.text_field(body, "request")
            workspace = body.get("workspace", str(self.server.workspace))
            if not isinstance(workspace, str) or not workspace:
                raise HTTPProblem(400, "workspace 必须是目录路径。")
            workspace = self.server.within_workspace(workspace)
            if not workspace.is_dir():
                raise HTTPProblem(400, "任务工作区目录不存在。")
            max_steps = body.get("max_steps", 12)
            if type(max_steps) is not int or not 1 <= max_steps <= 32:
                raise HTTPProblem(400, "max_steps 必须是 1–32 的整数。")
            parent_id = body.get("parent_id")
            if parent_id is not None:
                if not isinstance(parent_id, str):
                    raise HTTPProblem(400, "parent_id 必须是任务 ID。")
                parent = self.server.task(parent_id)
                if Path(parent["workspace"]).resolve() != workspace:
                    raise HTTPProblem(400, "后续任务必须使用相同工作区。")
            task = runtime.create_task(request, str(workspace), max_steps=max_steps, parent_id=parent_id)
            self.server.schedule(task["id"])
            return self.respond(202, self.server.task(task["id"]))
        match = re.fullmatch(r"/api/tasks/([A-Za-z0-9_-]{1,128})/(run|approve|answer|cancel)", path)
        if not match:
            raise HTTPProblem(404, "找不到此操作。")
        task_id, action = match.groups()
        self.server.task(task_id)
        if action == "approve":
            action_id = self.text_field(body, "action_id", 128)
            approved = body.get("approved", True)
            if type(approved) is not bool:
                raise HTTPProblem(400, "approved 必须是布尔值。")
            runtime.approve(task_id, action_id, approved=approved)
        elif action == "answer":
            runtime.answer(task_id, self.text_field(body, "text"))
        elif action == "cancel":
            runtime.cancel(task_id)
            return self.respond(200, self.server.task(task_id))
        if action != "cancel":
            self.server.schedule(task_id)
        return self.respond(202, self.server.task(task_id))


def make_server(runtime, workspace, host="127.0.0.1", port=0):
    return TaskServer(runtime, workspace, host, port)


def serve(db_path, workspace, host="127.0.0.1", port=8765):
    from .runtime import AgentRuntime
    server = make_server(AgentRuntime(db_path), workspace, host=host, port=port)
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Samantha task console: http://{browser_host}:{server.server_port}", flush=True)
    print(f"Workspace: {server.workspace}", flush=True)
    print("Model configuration is loaded; connectivity has not been verified. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
