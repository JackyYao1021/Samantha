"""Oracle-free container protocol for one approved Bash command batch.

This file is deliberately standalone: it never imports evaluation datasets,
scorers, or controller code. Calling it on the host cannot execute Bash.
"""

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from threading import Thread

SCHEMA_VERSION = "1.0"
MAX_FILE_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 8 * MAX_FILE_BYTES
MAX_INPUT_BYTES = 12 * MAX_FILE_BYTES
MAX_OUTPUT_BYTES = MAX_FILE_BYTES
COMMAND_TIMEOUT_SECONDS = 25


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Fixture path must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or value == "."
            or ".." in path.parts or any("\x00" in part or ":" in part for part in path.parts)):
        raise ValueError("Fixture path escapes the workspace")
    return path


def validate_input(value):
    """Allow public fixtures and commands only; never an expected result or oracle."""
    if not isinstance(value, dict) or set(value) != {"fixtures", "commands"}:
        raise ValueError("Sandbox input contains only fixtures and commands")
    if (not isinstance(value["commands"], list) or not value["commands"]
            or len(value["commands"]) > 100
            or any(not isinstance(command, str) or not command.strip() or "\x00" in command
                   for command in value["commands"])):
        raise ValueError("Expected 1 to 100 nonempty Bash commands")
    if sum(len(command.encode("utf-8")) for command in value["commands"]) > MAX_FILE_BYTES:
        raise ValueError("Command input exceeds byte budget")
    if not isinstance(value["fixtures"], list) or len(value["fixtures"]) > 1000:
        raise ValueError("Expected at most 1000 fixture entries")
    paths, total, decoded = {}, 0, []
    for fixture in value["fixtures"]:
        if not isinstance(fixture, dict) or not {"kind", "path"} <= fixture.keys():
            raise ValueError("Fixture needs kind and path")
        path = relative_path(fixture["path"])
        if path.as_posix() in paths or fixture["kind"] not in {"file", "directory"}:
            raise ValueError("Fixture paths must be unique and have a supported kind")
        paths[path.as_posix()] = fixture["kind"]
        data = None
        if fixture["kind"] == "directory":
            if set(fixture) != {"kind", "path"}:
                raise ValueError("Directory fixture cannot contain bytes")
        elif set(fixture) == {"kind", "path", "content_utf8"} and isinstance(fixture["content_utf8"], str):
            data = fixture["content_utf8"].encode("utf-8")
        elif set(fixture) == {"kind", "path", "content_base64"}:
            try:
                data = base64.b64decode(fixture["content_base64"], validate=True)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise ValueError("Invalid fixture base64") from exc
        else:
            raise ValueError("File fixture needs exactly one UTF-8 or base64 content field")
        if data is not None:
            total += len(data)
            if len(data) > MAX_FILE_BYTES or total > MAX_SNAPSHOT_BYTES:
                raise ValueError("Fixture byte budget exceeded")
        decoded.append((path.as_posix(), fixture["kind"], data))
    for path in paths:
        if any(paths.get(parent.as_posix()) == "file" for parent in PurePosixPath(path).parents):
            raise ValueError("File fixture cannot contain another fixture")
    return decoded


def snapshot(root):
    entries, total = {}, 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            key = path.relative_to(root).as_posix()
            relative_path(key)
            if path.is_symlink():
                raise ValueError("Links are not supported snapshot evidence")
            if len(entries) >= 1000:
                raise ValueError("Snapshot entry budget exceeded")
            if path.is_dir():
                entries[key] = {"kind": "directory"}
            elif path.is_file():
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("Snapshot file budget exceeded")
                data = path.read_bytes()
                total += len(data)
                if total > MAX_SNAPSHOT_BYTES:
                    raise ValueError("Snapshot byte budget exceeded")
                entries[key] = {"kind": "file", "size": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "content_base64": base64.b64encode(data).decode("ascii")}
            else:
                raise ValueError("Special files are not supported snapshot evidence")
    return {"complete": True, "entries": entries}


def read_bounded(stream, limit, result):
    """Drain the pipe continuously, retaining only the bounded prefix."""
    chunks, size, truncated = [], 0, False
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        available = max(0, limit - size)
        if available:
            chunks.append(chunk[:available])
        size += min(len(chunk), available)
        truncated = truncated or len(chunk) > available
    result.update(data=b"".join(chunks), truncated=truncated)


def in_container():
    return os.environ.get("SAMANTHA_EVAL_CONTAINER") == "1" and Path("/.dockerenv").is_file()


def run_input(value, root=Path("/fixture")):
    if not in_container():
        raise RuntimeError("Bash execution requires the dedicated Docker sandbox")
    fixtures = validate_input(value)
    root = Path(root)
    if root.resolve() != Path("/fixture") or any(root.iterdir()):
        raise ValueError("Sandbox workspace must be a fresh /fixture tmpfs")
    for path, kind, data in sorted(fixtures, key=lambda item: len(PurePosixPath(item[0]).parts)):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=True) if kind == "directory" else target.write_bytes(data)
    before = snapshot(root)
    output = {}
    # Bash receives a clean, fixed environment, never the Docker client's host
    # credentials. Arbitrary command text is parsed only inside the container.
    child = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-c",
                              " && ".join(value["commands"])], cwd=root,
                             env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/fixture",
                                  "TMPDIR": "/fixture", "LANG": "C.UTF-8"}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, start_new_session=True)
    reader = Thread(target=read_bounded, args=(child.stdout, MAX_OUTPUT_BYTES, output), daemon=True)
    reader.start()
    timed_out = False
    try:
        child.wait(timeout=COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        import signal
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    # A batch owns its entire process group, including background descendants.
    # End their writes before collecting filesystem evidence.
    import signal
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    reader.join(timeout=2)
    if reader.is_alive():
        # Descendant processes may retain a pipe after their parent exits.
        import signal
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        reader.join(timeout=2)
    after, errors = None, []
    try:
        after = snapshot(root)
    except (ValueError, OSError) as exc:
        errors.append({"category": "Evaluation Infrastructure", "stage": "sandbox_snapshot",
                       "message": type(exc).__name__})
    return {"schema_version": SCHEMA_VERSION, "complete": after is not None and not reader.is_alive(),
            "success": not timed_out and child.returncode == 0,
            "runtime_status": "timeout" if timed_out else "succeeded" if child.returncode == 0 else "failed",
            "output": output.get("data", b"").decode("utf-8", errors="replace"),
            "output_truncated": output.get("truncated", True), "returncode": child.returncode,
            "before_snapshot": before, "after_snapshot": after, "errors": errors}


def main():
    try:
        content = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(content) > MAX_INPUT_BYTES:
            raise ValueError("Sandbox input byte budget exceeded")
        payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "input"} or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported sandbox protocol")
        result = run_input(payload["input"])
    except (ValueError, OSError, RuntimeError, UnicodeError, KeyError, TypeError) as exc:
        result = {"schema_version": SCHEMA_VERSION, "complete": False, "success": False,
                  "runtime_status": "failed", "output": "", "output_truncated": False,
                  "returncode": None, "before_snapshot": None, "after_snapshot": None,
                  "errors": [{"category": "Evaluation Infrastructure", "stage": "sandbox_worker",
                              "message": type(exc).__name__}]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
