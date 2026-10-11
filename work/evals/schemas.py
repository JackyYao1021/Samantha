"""Small standard-library contracts; evaluation oracles stay in the controller."""

from copy import deepcopy
import json
from pathlib import PurePosixPath, PureWindowsPath
import re

SCHEMA_VERSION = "1.0"
INVARIANTS = {"no_unapproved_execution", "no_changes_outside_fixture_root"}
TERMINAL_STATUSES = {"succeeded", "cancelled", "failed"}
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


def relative_path(value):
    """Return a canonical POSIX path that has the same meaning on Windows/Linux."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts
            or path.as_posix() != value or value == "."):
        raise ValueError("Path must be canonical and remain inside the fixture root")
    for part in path.parts:
        if (any(ord(char) < 32 or char in '<>:"|?*' for char in part)
                or part.endswith((".", " "))
                or part.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}):
            raise ValueError("Path is not portable across Windows and Linux")
    return path


def _object(value, label, required=(), allowed=None):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    missing = set(required) - value.keys()
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    if allowed is not None and value.keys() - set(allowed):
        raise ValueError(f"{label} contains unsupported fields")


def _string(value, label, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{label} must be {'nonempty ' if nonempty else ''}text")


def _list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")


def validate_case(case):
    """Validate and return an independent JSON-compatible case, without coercion."""
    fields = {"schema_version", "case_id", "input", "expected_output", "reference", "metadata"}
    _object(case, "case", fields, fields)
    if case["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported case schema version")
    if not isinstance(case["case_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case["case_id"]):
        raise ValueError("case_id must be a portable identifier")
    data = case["input"]
    _object(data, "input", ("request", "fixtures", "interaction"),
            ("request", "fixtures", "interaction", "fake_services", "faults", "steps"))
    _string(data["request"], "input.request")
    _list(data["fixtures"], "input.fixtures")
    paths = {}
    total_bytes = 0
    for item in data["fixtures"]:
        _object(item, "fixture", ("kind", "path"), ("kind", "path", "content_utf8"))
        path = relative_path(item["path"]).as_posix()
        if path in paths or item["kind"] not in {"file", "directory"}:
            raise ValueError("Fixtures must have unique paths and a supported kind")
        paths[path] = item["kind"]
        if item["kind"] == "file":
            _string(item.get("content_utf8"), "fixture.content_utf8", nonempty=False)
            total_bytes += len(item["content_utf8"].encode("utf-8"))
        elif "content_utf8" in item:
            raise ValueError("Directory fixtures cannot contain file bytes")
    if total_bytes > MAX_FIXTURE_BYTES:
        raise ValueError("Fixture data exceeds 10 MiB")
    for name in paths:
        if any(paths.get(parent.as_posix()) == "file" for parent in PurePosixPath(name).parents):
            raise ValueError("A file fixture cannot contain another fixture")
    interaction = data["interaction"]
    _object(interaction, "input.interaction", ("clarification_answers", "approval_policy", "allowed_command_batches"),
            ("clarification_answers", "approval_policy", "allowed_command_batches", "allowed_effect_batches"))
    _list(interaction["clarification_answers"], "clarification_answers")
    for answer in interaction["clarification_answers"]:
        _string(answer, "clarification answer", nonempty=False)
    if interaction["approval_policy"] not in {"fixture_allowlist", "deny"}:
        raise ValueError("Unknown approval policy")
    _list(interaction["allowed_command_batches"], "allowed_command_batches")
    for batch in interaction["allowed_command_batches"]:
        _list(batch, "command batch")
        if not batch:
            raise ValueError("Allowed command batches cannot be empty")
        for command in batch:
            _string(command, "command")
    if "allowed_effect_batches" in interaction:
        _list(interaction["allowed_effect_batches"], "allowed_effect_batches")
        for batch in interaction["allowed_effect_batches"]:
            _list(batch, "effect batch")
            if not batch:
                raise ValueError("Allowed effect batches cannot be empty")
            for effect in batch:
                _object(effect, "allowed effect", ("kind", "path"), ("kind", "path", "destination", "content"))
                if effect["kind"] not in {"create", "append", "rename"}:
                    raise ValueError("Unsupported allowed effect")
                relative_path(effect["path"])
                if effect["kind"] == "rename":
                    relative_path(effect.get("destination"))
                else:
                    _string(effect.get("content"), "effect content", nonempty=False)
    if "fake_services" in data:
        _object(data["fake_services"], "fake_services")
    if "faults" in data:
        from .faults import validate_faults
        validate_faults(data["faults"])
    expected = case["expected_output"]
    _object(expected, "expected_output", ("expected_terminal_status", "goal_checks", "allowed_changes", "invariants"),
            ("expected_terminal_status", "goal_checks", "allowed_changes", "invariants", "step_statuses"))
    if expected["expected_terminal_status"] not in TERMINAL_STATUSES:
        raise ValueError("Unknown expected terminal status")
    if "steps" in data:
        _list(data["steps"], "steps")
        if not data["steps"] or len(data["steps"]) > 20:
            raise ValueError("Cross-invocation episodes require 1–20 additional steps")
        statuses = expected.get("step_statuses")
        if (not isinstance(statuses, list) or len(statuses) != len(data["steps"]) + 1
                or any(value not in TERMINAL_STATUSES for value in statuses)
                or statuses[-1] != expected["expected_terminal_status"]):
            raise ValueError("Episodes require every step terminal status, matching the final status")
        for index, step in enumerate(data["steps"]):
            _object(step, "step", ("request", "interaction"), ("request", "interaction", "fake_services", "faults"))
            derived = deepcopy(case)
            derived["input"] = {"fixtures": [], **step}
            derived["expected_output"].pop("step_statuses", None)
            derived["expected_output"]["expected_terminal_status"] = statuses[index + 1]
            # Reuse the same strict public-input contract, without exposing any oracle to a worker.
            validate_case(derived)
    elif "step_statuses" in expected:
        raise ValueError("step_statuses requires additional steps")
    _list(expected["goal_checks"], "goal_checks")
    if expected["expected_terminal_status"] == "succeeded" and not expected["goal_checks"]:
        raise ValueError("Successful tasks must define at least one verifiable goal")
    for check in expected["goal_checks"]:
        _object(check, "goal check", ("kind", "path"), ("kind", "path", "expected_utf8"))
        relative_path(check["path"])
        if check["kind"] not in {"file_bytes", "file_absent", "directory_exists"}:
            raise ValueError("Unknown goal check kind")
        if check["kind"] == "file_bytes":
            _string(check.get("expected_utf8"), "expected_utf8", nonempty=False)
        elif "expected_utf8" in check:
            raise ValueError("Only file_bytes checks accept expected_utf8")
    _list(expected["allowed_changes"], "allowed_changes")
    for path in expected["allowed_changes"]:
        relative_path(path)
    if len(set(expected["allowed_changes"])) != len(expected["allowed_changes"]):
        raise ValueError("allowed_changes contains duplicate paths")
    _list(expected["invariants"], "invariants")
    if (any(not isinstance(value, str) for value in expected["invariants"])
            or set(expected["invariants"]) != INVARIANTS
            or len(expected["invariants"]) != len(INVARIANTS)):
        raise ValueError("Both supported safety invariants must appear exactly once")
    _object(case["reference"], "reference", ("source", "version"))
    for key in ("source", "version"):
        _string(case["reference"][key], "reference." + key)
    _object(case["metadata"], "metadata", ("family", "split", "origin", "language", "difficulty", "lineage", "attack_surface"))
    for key in ("family", "split", "origin", "language", "difficulty"):
        _string(case["metadata"][key], "metadata." + key)
    lineage = case["metadata"]["lineage"]
    if isinstance(lineage, str):
        _string(lineage, "metadata.lineage")
    elif isinstance(lineage, list):
        for source in lineage:
            _string(source, "metadata.lineage source")
    else:
        raise ValueError("metadata.lineage must be text or a list of source labels")
    _list(case["metadata"]["attack_surface"], "metadata.attack_surface")
    for surface in case["metadata"]["attack_surface"]:
        _string(surface, "metadata.attack_surface item")
    # JSON serialization also rejects objects supplied accidentally by Python callers.
    try:
        json.dumps(case, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Case must contain finite JSON values") from exc
    return deepcopy(case)


def worker_input(case):
    """Only allowed runtime inputs cross the worker boundary; never return oracles."""
    return deepcopy(validate_case(case)["input"])
