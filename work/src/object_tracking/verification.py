"""Bounded, read-only evidence for the three explicitly supported file effects.

This module never evaluates shell text and never creates, changes or moves files.
Fingerprints are observations, not proof of permanent file identity. Symlinks,
Windows reparse points, non-regular files and concurrent changes are refused.
"""

import hashlib
import os
from pathlib import Path
import stat


DEFAULT_MAX_BYTES = 1024 * 1024


def normalize_path(value):
    """Keep the execution host's absolute path without following symlinks."""
    if not isinstance(value, str) or not value or "\0" in value or not os.path.isabs(value):
        raise ValueError("Effect paths must be non-empty absolute host paths.")
    return os.path.normpath(os.path.abspath(value))


def validate_effect(effect):
    """Defensive validation for callers outside the command binding boundary."""
    if not isinstance(effect, dict) or effect.get("kind") not in {"create", "rename", "append"}:
        raise ValueError("Unsupported file effect.")
    result = {"kind": effect["kind"], "path": normalize_path(effect.get("path")),
              "destination": None, "content": None, "entity_id": effect.get("entity_id"),
              "entity_version": effect.get("entity_version")}
    if result["kind"] == "rename":
        result["destination"] = normalize_path(effect.get("destination"))
        if os.path.normcase(result["destination"]) == os.path.normcase(result["path"]):
            raise ValueError("Rename requires different source and destination paths.")
        if effect.get("content") is not None:
            raise ValueError("Rename cannot carry content.")
    else:
        if not isinstance(effect.get("content"), str) or effect.get("destination") is not None:
            raise ValueError("Create and append require exact UTF-8 content, without a destination.")
        effect["content"].encode("utf-8")
        result["content"] = effect["content"]
    identifier, version = result["entity_id"], result["entity_version"]
    if (identifier is None) != (version is None):
        raise ValueError("Entity identity and version must be supplied together.")
    if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()
                                   or type(version) is not int or version < 1):
        raise ValueError("Invalid entity identity or version.")
    if result["kind"] == "create" and identifier is not None:
        raise ValueError("Creation cannot reuse an existing entity identity.")
    return result


def _metadata(info):
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def _reject_links(path):
    for part in reversed((Path(path), *Path(path).parents)):
        try:
            info = os.lstat(part)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) &
                                        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError("Symlinks and reparse points are outside verification scope.")


def _read_snapshot(path, max_bytes):
    observation = {"path": path, "status": "unverified", "fingerprint": None, "reason": ""}
    descriptor = None
    try:
        _reject_links(path)
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Only regular files are supported.")
        if info.st_size > max_bytes:
            raise ValueError("File exceeds verification byte limit.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _metadata(opened) != _metadata(info) or not stat.S_ISREG(opened.st_mode):
            raise ValueError("File changed while opening the snapshot.")
        chunks, total = [], 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ValueError("File exceeds verification byte limit.")
        if _metadata(os.fstat(descriptor)) != _metadata(opened):
            raise ValueError("File changed while reading the snapshot.")
        _reject_links(path)
        if _metadata(os.lstat(path)) != _metadata(opened) or len(data) != opened.st_size:
            raise ValueError("File changed before snapshot completion.")
        observation.update(status="present", fingerprint={**_metadata(opened),
                           "sha256": hashlib.sha256(data).hexdigest()})
        return observation, data
    except FileNotFoundError:
        observation.update(status="missing", reason="Path does not exist.")
    except (OSError, ValueError) as exc:
        observation["reason"] = str(exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return observation, None


def observe_path(path, *, max_bytes=DEFAULT_MAX_BYTES):
    """Return a serializable fingerprint; no file bytes leave the verifier."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Verification byte limit must be positive.")
    return _read_snapshot(normalize_path(path), max_bytes)[0]


def _check(checks, name, passed):
    checks.append({"name": name, "passed": bool(passed)})
    return bool(passed)


def capture_before(effect, *, max_bytes=DEFAULT_MAX_BYTES, entity=None):
    """Capture preconditions and compute expected content hashes in memory."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Verification byte limit must be positive.")
    effect = validate_effect(effect)
    source, data = _read_snapshot(effect["path"], max_bytes)
    destination = (observe_path(effect["destination"], max_bytes=max_bytes)
                   if effect["kind"] == "rename" else None)
    checks = []
    result = {"status": "unverified", "reason": "Preconditions were not established.",
              "effect": effect, "source": source, "destination": destination,
              "expected": None, "checks": checks, "entity": None}
    if effect["kind"] == "create":
        _check(checks, "source_absent", source["status"] == "missing")
        expected = effect["content"].encode("utf-8")
    else:
        _check(checks, "source_regular_readable", source["status"] == "present")
        fingerprint = source["fingerprint"] or {}
        _check(checks, "source_identity_available", bool(fingerprint.get("inode")))
        if effect["entity_id"] is not None:
            _check(checks, "entity_snapshot_matches", isinstance(entity, dict) and
                   entity.get("status") == "active" and
                   entity.get("entity_id") == effect["entity_id"] and
                   entity.get("version") == effect["entity_version"] and
                   isinstance(entity.get("current_path"), str) and
                   os.path.normcase(entity["current_path"]) == os.path.normcase(effect["path"]) and
                   entity.get("fingerprint") == source["fingerprint"])
            if isinstance(entity, dict):
                result["entity"] = {key: entity.get(key) for key in
                                    ("entity_id", "version", "fingerprint")}
        elif entity is not None:
            _check(checks, "unbound_entity_not_supplied", False)
        if effect["kind"] == "rename":
            _check(checks, "destination_absent", destination["status"] == "missing")
            expected = data
        else:
            expected = None if data is None else data + effect["content"].encode("utf-8")
    _check(checks, "expected_content_within_limit", expected is not None and len(expected) <= max_bytes)
    if expected is not None and len(expected) <= max_bytes:
        result["expected"] = {"size": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
    if all(check["passed"] for check in checks):
        result.update(status="ready", reason="All supported preconditions were observed.")
    return result


def verify_after(effect, before, *, max_bytes=DEFAULT_MAX_BYTES):
    """Check actual postconditions even when the command batch reports failure."""
    effect = validate_effect(effect)
    checks = []
    source = observe_path(effect["path"], max_bytes=max_bytes)
    destination = (observe_path(effect["destination"], max_bytes=max_bytes)
                   if effect["kind"] == "rename" else None)
    result = {"status": "unverified", "reason": "Postconditions were not established.",
              "effect": effect, "before": before,
              "after": {"source": source, "destination": destination}, "checks": checks}
    valid_before = (isinstance(before, dict) and before.get("status") == "ready" and
                    before.get("effect") == effect and isinstance(before.get("expected"), dict))
    _check(checks, "reviewed_preconditions_available", valid_before)
    if not valid_before:
        return result
    target = destination if effect["kind"] == "rename" else source
    _check(checks, "target_regular_readable", target["status"] == "present")
    observed = target["fingerprint"] or {}
    _check(checks, "expected_content_matches", all(observed.get(key) == value
           for key, value in before["expected"].items()))
    if effect["kind"] in {"rename", "append"}:
        original = before["source"]["fingerprint"] or {}
        _check(checks, "same_file_identity", bool(original.get("inode")) and
               all(observed.get(key) == original.get(key) for key in ("device", "inode")))
    if effect["kind"] == "rename":
        _check(checks, "old_path_absent", source["status"] == "missing")
    if all(check["passed"] for check in checks):
        result.update(status="verified", reason="All supported postconditions were observed.")
    return result
