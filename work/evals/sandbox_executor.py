"""Copy public fixture bytes through the independent, opt-in Docker worker.

The production workflow must bind approval before invoking this callable. The
executor passes that exact command text unchanged to Docker; it never invokes a
host shell. Every batch reconstructs file state, not arbitrary process state.
"""

import base64
from copy import deepcopy
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile

from .harness import fixture_path, snapshot
from .sandbox import DockerSandbox, SandboxUnavailable
from .sandbox_worker import MAX_FILE_BYTES, MAX_SNAPSHOT_BYTES, validate_input


def validate_snapshot(value, root):
    """Verify complete factual bytes before allowing any controller filesystem write."""
    if (not isinstance(value, dict) or set(value) != {"complete", "entries"}
            or value["complete"] is not True or not isinstance(value["entries"], dict)
            or len(value["entries"]) > 1000):
        raise ValueError("A complete bounded fixture snapshot is required")
    decoded, total, targets = {}, 0, set()
    for name, entry in value["entries"].items():
        target = os.path.normcase(str(fixture_path(root, name)))
        if target in targets:
            raise ValueError("Snapshot paths alias the same controller file")
        targets.add(target)
        if not isinstance(entry, dict):
            raise ValueError("Snapshot entry must be an object")
        if entry.get("kind") == "directory" and set(entry) == {"kind"}:
            decoded[name] = ("directory", None)
            continue
        if entry.get("kind") != "file" or set(entry) != {"kind", "size", "sha256", "content_base64"}:
            raise ValueError("Snapshot entry must be a regular file or directory")
        if (type(entry["size"]) is not int or not 0 <= entry["size"] <= MAX_FILE_BYTES
                or not isinstance(entry["content_base64"], str)
                or len(entry["content_base64"]) > 4 * ((MAX_FILE_BYTES + 2) // 3)):
            raise ValueError("Snapshot file exceeds its byte budget")
        try:
            data = base64.b64decode(entry["content_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Snapshot file has invalid base64") from exc
        total += len(data)
        if (len(data) != entry["size"] or total > MAX_SNAPSHOT_BYTES
                or hashlib.sha256(data).hexdigest() != entry["sha256"]):
            raise ValueError("Snapshot bytes do not match the declared size or digest")
        decoded[name] = ("file", data)
    # A complete snapshot lists every parent directory; do not silently invent
    # missing evidence or accept paths beneath a file.
    for name in decoded:
        for parent in PurePosixPath(name).parents:
            if parent == PurePosixPath("."):
                continue
            if decoded.get(parent.as_posix(), (None,))[0] != "directory":
                raise ValueError("Snapshot lacks a directory ancestor")
    return decoded


def fixtures_from_snapshot(value, root):
    validate_snapshot(value, root)
    return [{"kind": entry["kind"], "path": name,
             **({"content_base64": entry["content_base64"]} if entry["kind"] == "file" else {})}
            for name, entry in sorted(value["entries"].items())]


def restore_snapshot(root, before, after):
    """Restore only verified fixture entries; reject stale host state and all links.

    Individual file replacements are atomic. A whole multi-file batch is not a
    filesystem transaction; callers retain before/after evidence for failures.
    """
    previous = validate_snapshot(before, root)
    desired = validate_snapshot(after, root)
    if snapshot(root) != before:
        raise ValueError("Controller fixture state changed during sandbox execution")
    for name, (kind, _) in previous.items():
        if kind == "file" and desired.get(name, (None,))[0] != "file":
            fixture_path(root, name).unlink()
    for name, (kind, _) in sorted(previous.items(), key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
        if kind == "directory" and desired.get(name, (None,))[0] != "directory":
            fixture_path(root, name).rmdir()  # Never recursively delete a computed path.
    for name, (kind, _) in sorted(desired.items(), key=lambda item: len(PurePosixPath(item[0]).parts)):
        if kind == "directory":
            fixture_path(root, name).mkdir(exist_ok=True)
    for name, (kind, data) in desired.items():
        if kind != "file":
            continue
        target = fixture_path(root, name)
        descriptor, temporary = tempfile.mkstemp(prefix=".sandbox-copy-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if snapshot(root) != after:
        raise ValueError("Controller fixture restoration did not match sandbox evidence")


class SandboxExecutor:
    """Workflow executor callable; approval remains the workflow's responsibility."""

    def __init__(self, public_input, root, *, enabled=False, image="samantha-eval-sandbox:latest",
                 docker="docker", timeout_seconds=30):
        if not isinstance(public_input, dict) or set(public_input) - {
                "request", "fixtures", "interaction", "fake_services", "faults", "steps"}:
            raise ValueError("Sandbox executor accepts public case input only")
        original = Path(root)
        if (not original.is_dir() or original.is_symlink()
                or getattr(original.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Controller fixture root must be a regular directory")
        self.root = original.resolve(strict=True)
        self.sandbox = DockerSandbox(enabled=enabled, image=image, docker=docker,
                                     timeout_seconds=timeout_seconds)
        self.batches = []
        self.tool_calls = 0

    def execute(self, commands):
        before = snapshot(self.root)
        public = {"fixtures": fixtures_from_snapshot(before, self.root), "commands": deepcopy(commands)}
        validate_input(public)
        self.tool_calls += 1
        batch = {"commands": deepcopy(commands), "before_snapshot": before, "copied_back": False}
        self.batches.append(batch)
        result = self.sandbox.run(public)
        batch["sandbox_result"] = deepcopy(result)
        if result.get("complete") is not True:
            raise SandboxUnavailable("Incomplete sandbox evidence cannot be copied into controller fixtures")
        validate_snapshot(result.get("before_snapshot"), self.root)
        validate_snapshot(result.get("after_snapshot"), self.root)
        if result["before_snapshot"] != before:
            raise SandboxUnavailable("Sandbox initial bytes differ from the controller snapshot")
        restore_snapshot(self.root, before, result["after_snapshot"])
        batch["copied_back"] = True
        return {"success": result["success"], "output": result["output"], "current_dir": str(self.root),
                "before_snapshot": before, "after_snapshot": result["after_snapshot"],
                "sandbox_runtime_status": result["runtime_status"],
                "output_truncated": result["output_truncated"], "sandbox_errors": result["errors"]}

    __call__ = execute
