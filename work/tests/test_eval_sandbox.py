"""Docker structure and protocol checks; these tests never start a container."""

import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.sandbox import DockerSandbox, SandboxUnavailable
from evals import sandbox_worker


def result():
    return {"schema_version": "1.0", "complete": True, "success": True, "runtime_status": "succeeded",
            "output": "ok", "output_truncated": False, "returncode": 0,
            "before_snapshot": {"complete": True, "entries": {}},
            "after_snapshot": {"complete": True, "entries": {}}, "errors": []}


class InputPipe(io.BytesIO):
    def close(self):
        self.was_closed = True


class FakeDocker:
    def __init__(self, response=None, *, timeout=False):
        self.stdin = InputPipe()
        self.stdout = io.BytesIO(json.dumps(response or result()).encode("utf-8"))
        self.stderr = io.BytesIO(b"local docker diagnostics")
        self.returncode = 0
        self.timeout = timeout
        self.killed = False

    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("docker", timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class DockerSandboxTests(unittest.TestCase):
    def test_default_disabled_never_starts_docker_or_bash(self):
        with patch("evals.sandbox.subprocess.Popen") as started:
            with self.assertRaisesRegex(SandboxUnavailable, "disabled"):
                DockerSandbox().run({"fixtures": [], "commands": ["touch file.txt"]})
            started.assert_not_called()
        with self.assertRaisesRegex(ValueError, "never a host shell"):
            DockerSandbox(docker="/bin/bash")

    def test_readiness_is_read_only_and_missing_daemon_fails_without_execution(self):
        with patch("evals.sandbox.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run, \
                patch("evals.sandbox.subprocess.Popen") as started:
            sandbox = DockerSandbox(enabled=True)
            self.assertFalse(sandbox.readiness()["ready"])
            with self.assertRaisesRegex(SandboxUnavailable, "daemon unavailable"):
                sandbox.run({"fixtures": [], "commands": ["touch file.txt"]})
            self.assertTrue(all(call.args[0][1] == "info" for call in run.call_args_list))
            started.assert_not_called()

    def test_readiness_returns_ascii_immutable_image_identity_and_run_pins_it(self):
        identity = "sha256:" + "a" * 64
        sandbox = DockerSandbox(enabled=True)
        answers = [SimpleNamespace(returncode=0, stdout=b"27.1.0\n"),
                   SimpleNamespace(returncode=0, stdout=(identity + "\n").encode("ascii"))]
        with patch("evals.sandbox.subprocess.run", side_effect=answers) as diagnostics:
            ready = sandbox.readiness()
        self.assertEqual(ready["image_id"], identity)
        self.assertTrue(ready["ready"])
        self.assertEqual(diagnostics.call_args.args[0][1:3], ["image", "inspect"])
        with patch.object(sandbox, "readiness", return_value=ready), \
                patch("evals.sandbox.subprocess.Popen", return_value=FakeDocker()) as started:
            sandbox.run({"fixtures": [], "commands": ["true"]})
        self.assertEqual(started.call_args.args[0][-1], identity)
        for invalid in (b"", b"tag:latest", b"sha256:\xff"):
            with patch("evals.sandbox.subprocess.run", side_effect=[
                    answers[0], SimpleNamespace(returncode=0, stdout=invalid)]):
                self.assertFalse(sandbox.readiness()["ready"])

    def test_run_has_no_host_mounts_credentials_network_or_host_shell_command(self):
        command = "printf '%s' '$(host-canary)' > '文件.txt'"
        fake = FakeDocker()
        sandbox = DockerSandbox(enabled=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-canary", "DOCKER_HOST": "ssh://private-canary"}), \
                patch.object(sandbox, "readiness", return_value={"ready": True}), \
                patch("evals.sandbox.subprocess.Popen", return_value=fake) as started:
            actual = sandbox.run({"fixtures": [], "commands": [command]})
        self.assertTrue(actual["success"])
        arguments = started.call_args.args[0]
        self.assertEqual(arguments[0:2], ["docker", "run"])
        self.assertNotIn(command, arguments)
        self.assertNotIn("--volume", arguments)
        self.assertNotIn("--mount", arguments)
        self.assertFalse(started.call_args.kwargs["shell"])
        self.assertEqual(arguments[arguments.index("--network") + 1], "none")
        self.assertEqual(arguments[arguments.index("--ipc") + 1], "none")
        self.assertIn("--read-only", arguments)
        self.assertEqual(arguments[arguments.index("--cap-drop") + 1], "ALL")
        tmpfs = [arguments[index + 1] for index, item in enumerate(arguments) if item == "--tmpfs"]
        self.assertEqual(len(tmpfs), 1)
        self.assertTrue(tmpfs[0].startswith("/fixture:"))
        self.assertNotIn("secret-canary", repr(started.call_args))
        self.assertNotIn("private-canary", repr(started.call_args))
        self.assertEqual(json.loads(fake.stdin.getvalue())["input"]["commands"], [command])

    def test_timeout_removes_only_its_generated_container_and_preserves_failure(self):
        fake = FakeDocker(timeout=True)
        sandbox = DockerSandbox(enabled=True, timeout_seconds=.01)
        with patch.object(sandbox, "readiness", return_value={"ready": True}), \
                patch("evals.sandbox.subprocess.Popen", return_value=fake), \
                patch("evals.sandbox.subprocess.run", return_value=SimpleNamespace(returncode=0)) as cleanup:
            actual = sandbox.run({"fixtures": [], "commands": ["sleep 100"]})
        self.assertEqual(actual["runtime_status"], "timeout")
        self.assertFalse(actual["complete"])
        cleanup_args = cleanup.call_args.args[0]
        self.assertEqual(cleanup_args[:3], ["docker", "rm", "--force"])
        self.assertTrue(cleanup_args[3].startswith("samantha-eval-"))
        self.assertEqual(len(cleanup_args), 4)
        self.assertTrue(fake.killed)

    def test_protocol_output_limit_and_invalid_json_fail_closed(self):
        sandbox = DockerSandbox(enabled=True, max_output_bytes=1024)
        for data in (b"x" * 2048, b"not json", b'{"schema_version":"1.0"}'):
            fake = FakeDocker()
            fake.stdout = io.BytesIO(data)
            with patch.object(sandbox, "readiness", return_value={"ready": True}), \
                    patch("evals.sandbox.subprocess.Popen", return_value=fake):
                with self.assertRaises(SandboxUnavailable):
                    sandbox.run({"fixtures": [], "commands": ["true"]})

    def test_build_target_does_not_inherit_application_or_oracle_copy(self):
        dockerfile = (WORK.parent / "Dockerfile").read_text(encoding="utf-8")
        evaluation = dockerfile.split("FROM buildbase AS evalsandbox", 1)[1].split("FROM runtime AS default-runtime", 1)[0]
        self.assertIn("COPY work/src/ /opt/samantha/src/", evaluation)
        self.assertIn("COPY work/evals/sandbox_worker.py", evaluation)
        self.assertNotIn("COPY work/ /work/", evaluation)
        self.assertNotIn("work/tests", evaluation)
        self.assertNotIn("datasets", evaluation)
        base = dockerfile.split("FROM buildbase AS runtime", 1)[0]
        self.assertNotIn("COPY work/ /work/", base)
        self.assertIn("FROM runtime AS task-tests", dockerfile)
        self.assertTrue(dockerfile.strip().endswith("FROM runtime AS default-runtime"))
        compose = (WORK.parent / "docker-compose.tests.yaml").read_text(encoding="utf-8")
        service = compose.split("  eval-sandbox:", 1)[1].split("  tests:", 1)[0]
        self.assertNotIn("volumes:", service)
        self.assertIn("network_mode: none", service)
        self.assertIn("read_only: true", service)


class SandboxWorkerTests(unittest.TestCase):
    def test_public_input_accepts_binary_restore_and_rejects_oracles_and_paths(self):
        decoded = sandbox_worker.validate_input({"commands": ["true"], "fixtures": [
            {"kind": "file", "path": "nested/binary", "content_base64": "AP8="}]})
        self.assertEqual(decoded[0][2], b"\x00\xff")
        for path in ("../outside", "/absolute", "a/../b", "a\\b", "a:stream"):
            with self.assertRaises(ValueError):
                sandbox_worker.validate_input({"commands": ["true"], "fixtures": [
                    {"kind": "file", "path": path, "content_utf8": "x"}]})
        with self.assertRaises(ValueError):
            sandbox_worker.validate_input({"commands": ["true"], "fixtures": [], "expected_output": {}})

    def test_host_worker_cannot_launch_commands_even_with_valid_input(self):
        with patch("evals.sandbox_worker.in_container", return_value=False), \
                patch("evals.sandbox_worker.subprocess.Popen") as started:
            with self.assertRaisesRegex(RuntimeError, "dedicated Docker"):
                sandbox_worker.run_input({"commands": ["touch host-canary"], "fixtures": []})
            started.assert_not_called()

    def test_snapshot_is_factual_and_compatible_with_saved_p0_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            (root / "folder" / "binary").write_bytes(b"\x00\xff")
            snapshot = sandbox_worker.snapshot(root)
        self.assertTrue(snapshot["complete"])
        entry = snapshot["entries"]["folder/binary"]
        self.assertEqual(entry["size"], 2)
        self.assertEqual(base64.b64decode(entry["content_base64"]), b"\x00\xff")
        self.assertEqual(set(entry), {"kind", "size", "sha256", "content_base64"})

    def test_output_is_drained_with_a_bounded_prefix(self):
        result = {}
        sandbox_worker.read_bounded(io.BytesIO(b"x" * 100000), 1024, result)
        self.assertEqual(len(result["data"]), 1024)
        self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
