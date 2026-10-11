"""Opt-in Docker execution; generated commands never enter a host shell.

Each call rebuilds public file state in a fresh container. It does not preserve
Shell processes, inode identity, sessions, or other state between batches.
Docker availability and image construction are external prerequisites.
"""

import json
import math
import os
from pathlib import Path
import re
import subprocess
from threading import Thread
import uuid

from .sandbox_worker import MAX_INPUT_BYTES, read_bounded, validate_input


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, *, enabled=False, image="samantha-eval-sandbox:latest",
                 timeout_seconds=30, max_output_bytes=32 * 1024 * 1024, docker="docker"):
        if type(enabled) is not bool:
            raise ValueError("Sandbox enabled must be an explicit boolean")
        if (not isinstance(image, str) or not image or image.startswith("-")
                or any(character.isspace() for character in image)):
            raise ValueError("Expected a Docker image reference")
        if Path(docker).name.lower() not in {"docker", "docker.exe"}:
            raise ValueError("The sandbox executable must be Docker, never a host shell")
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("Sandbox timeout must be positive and finite")
        if type(max_output_bytes) is not int or not 1024 <= max_output_bytes <= 32 * 1024 * 1024:
            raise ValueError("Sandbox protocol output limit must be between 1 KiB and 32 MiB")
        self.enabled, self.image, self.docker = enabled, image, str(docker)
        self.timeout_seconds, self.max_output_bytes = timeout_seconds, max_output_bytes

    @staticmethod
    def environment():
        # Docker communicates with its local daemon. Model keys, repository .env,
        # DOCKER_HOST, DOCKER_CONFIG and arbitrary environment values are omitted.
        return {name: os.environ[name] for name in ("PATH", "SystemRoot", "WINDIR", "TMP", "TEMP")
                if name in os.environ}

    def readiness(self):
        """Read-only diagnostics; never build, pull, create, or start a container."""
        try:
            daemon = subprocess.run([self.docker, "info", "--format", "{{.ServerVersion}}"],
                                    capture_output=True, timeout=10, env=self.environment(), check=False)
            if daemon.returncode:
                return {"ready": False, "reason": "Docker daemon unavailable", "enabled": self.enabled}
            image = subprocess.run([self.docker, "image", "inspect", "--format", "{{.Id}}", self.image],
                                   capture_output=True, timeout=10, env=self.environment(), check=False)
            if image.returncode:
                return {"ready": False, "reason": "Sandbox image missing; an explicit image build is required",
                        "enabled": self.enabled}
            try:
                image_id = image.stdout.decode("ascii").strip()
            except (AttributeError, UnicodeDecodeError):
                image_id = ""
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                return {"ready": False, "reason": "Docker did not return an immutable image ID",
                        "enabled": self.enabled}
            return {"ready": True, "reason": "ready", "enabled": self.enabled, "image_id": image_id}
        except (OSError, subprocess.SubprocessError):
            return {"ready": False, "reason": "Docker CLI or daemon unavailable", "enabled": self.enabled}

    def command(self, name, image=None):
        return [self.docker, "run", "--rm", "--pull", "never", "--name", name, "--init", "--interactive",
                "--network", "none", "--ipc", "none", "--read-only", "--user", "10001:10001", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true", "--pids-limit", "64", "--memory", "256m",
                "--cpus", "1", "--tmpfs", "/fixture:rw,nosuid,nodev,noexec,size=16m,uid=10001,gid=10001,mode=0700",
                "--env", "SAMANTHA_EVAL_CONTAINER=1", "--env", "PYTHONUTF8=1", image or self.image]

    def run(self, public_input):
        """Execute exactly one controller-approved batch using public inputs only."""
        if not self.enabled:
            raise SandboxUnavailable("Docker sandbox execution is disabled; enable it explicitly")
        validate_input(public_input)
        payload = json.dumps({"schema_version": "1.0", "input": public_input}, ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("Sandbox input byte budget exceeded")
        readiness = self.readiness()
        if not readiness["ready"]:
            raise SandboxUnavailable(readiness["reason"])
        name = "samantha-eval-" + uuid.uuid4().hex
        child = subprocess.Popen(self.command(name, readiness.get("image_id", self.image)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=self.environment(), shell=False)
        output, error = {}, {}
        def send_input():
            try:
                child.stdin.write(payload)
                child.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        threads = [Thread(target=read_bounded, args=(child.stdout, self.max_output_bytes, output), daemon=True),
                   Thread(target=read_bounded, args=(child.stderr, 16384, error), daemon=True)]
        for thread in threads:
            thread.start()
        try:
            writer = Thread(target=send_input, daemon=True)
            writer.start()
            child.wait(timeout=self.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt, BrokenPipeError) as exc:
            # Stop only the uniquely named container created by this call. No
            # filesystem mount exists; neither rm nor kill touches host files.
            try:
                subprocess.run([self.docker, "rm", "--force", name], capture_output=True,
                               timeout=10, env=self.environment(), check=False)
            except (OSError, subprocess.SubprocessError):
                pass
            child.kill()
            child.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=2)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return {"schema_version": "1.0", "complete": False, "success": False,
                    "runtime_status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
                    "output": "", "output_truncated": output.get("truncated", False), "returncode": None,
                    "before_snapshot": None, "after_snapshot": None,
                    "errors": [{"category": "Evaluation Infrastructure", "stage": "sandbox",
                                "message": type(exc).__name__}]}
        for thread in threads:
            thread.join(timeout=2)
        writer.join(timeout=2)
        if child.returncode or output.get("truncated") or any(thread.is_alive() for thread in threads):
            raise SandboxUnavailable("Docker worker exited or exceeded the protocol output budget")
        try:
            result = json.loads(output.get("data", b""))
        except (ValueError, UnicodeError) as exc:
            raise SandboxUnavailable("Docker worker did not return one JSON result") from exc
        fields = {"schema_version", "complete", "success", "runtime_status", "output", "output_truncated",
                  "returncode", "before_snapshot", "after_snapshot", "errors"}
        if (not isinstance(result, dict) or set(result) != fields or result.get("schema_version") != "1.0"
                or any(type(result.get(field)) is not bool for field in ("complete", "success", "output_truncated"))
                or result.get("runtime_status") not in {"succeeded", "failed", "timeout"}
                or not isinstance(result.get("output"), str) or not isinstance(result.get("errors"), list)):
            raise SandboxUnavailable("Docker worker returned an invalid result schema")
        return result
