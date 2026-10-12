"""Independent rules for bounded, declarative routing tasks.

Model input contains only the request and initial file inventory. No generated
code or shell command is interpreted. Actions run only in a fresh temporary
directory, after the entire response and sequential action preconditions pass.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from tempfile import TemporaryDirectory

from .harness import MAX_FILE_BYTES, MAX_SNAPSHOT_BYTES, fixture_path, snapshot


SCHEMA_VERSION = "routing-task-v1"
FAMILIES = {"write", "rename", "append", "select", "multi_step", "clarify"}
MAX_ACTIONS = 12
MAX_CONTENT_BYTES = 64 * 1024
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "routing_tasks_v1.jsonl"
OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "actions", "answer", "question"],
    "properties": {
        "status": {"type": "string", "enum": ["completed", "clarify"]},
        "actions": {"type": "array", "maxItems": MAX_ACTIONS, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["tool", "path", "content", "destination"],
            "properties": {
                "tool": {"type": "string", "enum": ["write", "append", "rename", "mkdir"]},
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string", "maxLength": MAX_CONTENT_BYTES},
                "destination": {"type": "string"},
            },
        }},
        "answer": {"type": "array", "maxItems": 64, "uniqueItems": True,
                   "items": {"type": "string", "minLength": 1}},
        "question": {"type": "string", "maxLength": 16384},
    },
}


class PayloadValidationError(ValueError):
    def __init__(self, message, category="output_schema"):
        super().__init__(message)
        self.category = category


def _keys(value, required, label):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(label + " must contain exactly " + ", ".join(sorted(required)))


def _utf8(value, label, limit):
    if not isinstance(value, str):
        raise ValueError(label + " must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(label + " must be valid UTF-8 text") from exc
    if len(encoded) > limit:
        raise ValueError(label + " exceeds the byte limit")
    return encoded


def _path_text(value):
    """Portable lexical validation, also usable without a fixture or oracle."""
    _utf8(value, "path", 4096)
    if not value or any(ord(char) < 32 or char in '\\:<>"|?*' for char in value):
        raise ValueError("Expected a portable relative POSIX path")
    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} or part.rstrip(" .") != part for part in parts)
            or any(part.split(".")[0].upper() in reserved for part in parts)):
        raise ValueError("Path escapes or aliases the fixture root")
    return value


def _checked_path(root, value):
    return fixture_path(root, _path_text(value))


def _paths(values, label):
    if not isinstance(values, list) or len(values) > 64:
        raise ValueError(label + " must be a bounded list")
    result = [_path_text(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(label + " contains duplicate paths")
    return result


def _inventory(files, directories, root, label):
    if not isinstance(files, dict) or len(files) > 64:
        raise ValueError(label + " files must be a bounded object")
    checked_directories = _paths(directories, label + " directories")
    total = 0
    for path, content in files.items():
        _checked_path(root, path)
        total += len(_utf8(content, label + " file content", MAX_FILE_BYTES))
    for path in checked_directories:
        _checked_path(root, path)
    if total > MAX_SNAPSHOT_BYTES or set(files) & set(checked_directories):
        raise ValueError(label + " inventory exceeds limits or has conflicting paths")
    directory_set = set(checked_directories)
    for path in [*files, *checked_directories]:
        parents = list(PurePosixPath(path).parents)[:-1]
        if any(parent.as_posix() not in directory_set for parent in parents):
            raise ValueError(label + " inventory must include every parent directory")


def _validate_case(case, root):
    _keys(case, {"schema_version", "case_id", "split", "family", "input", "expected"}, "case")
    if case["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported routing task schema")
    identifier = case["case_id"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
        raise ValueError("case_id must be a bounded portable identifier")
    if not isinstance(case["split"], str) or case["split"] not in {"development", "heldout"}:
        raise ValueError("Unknown case split")
    if not isinstance(case["family"], str) or case["family"] not in FAMILIES:
        raise ValueError("Unknown task family")
    public = case["input"]
    _keys(public, {"request", "files", "directories"}, "input")
    _utf8(public["request"], "request", MAX_CONTENT_BYTES)
    if not public["request"].strip():
        raise ValueError("Request must not be blank")
    _inventory(public["files"], public["directories"], root, "input")
    expected = case["expected"]
    _keys(expected, {"status", "files", "directories", "answer", "question_terms_any"}, "expected")
    if not isinstance(expected["status"], str) or expected["status"] not in {"completed", "clarify"}:
        raise ValueError("Invalid expected status")
    _inventory(expected["files"], expected["directories"], root, "expected")
    answers = _paths(expected["answer"], "expected answer")
    if any(answer not in expected["files"] and answer not in expected["directories"] for answer in answers):
        raise ValueError("Expected answers must name final inventory entries")
    terms = expected["question_terms_any"]
    if (not isinstance(terms, list) or len(terms) > 32
            or any(not isinstance(term, str) or not term.strip() for term in terms)):
        raise ValueError("Expected clarification terms must be nonblank strings")
    for term in terms:
        _utf8(term, "clarification term", 1024)
    if expected["status"] == "clarify":
        if (not terms or answers or expected["files"] != public["files"]
                or set(expected["directories"]) != set(public["directories"])):
            raise ValueError("Clarification oracle must preserve the fixture and include question terms")
    elif terms:
        raise ValueError("Completed cases cannot have clarification terms")
    return deepcopy(case)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def load_cases(path=DEFAULT_DATASET):
    """Load strict JSONL cases; reject malformed oracles and duplicate IDs."""
    cases, identifiers = [], set()
    with TemporaryDirectory(prefix="routing-validation-") as root:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = _validate_case(json.loads(line, object_pairs_hook=_unique_json_object), root)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid routing case at line {line_number}: {exc}") from exc
            if case["case_id"] in identifiers:
                raise ValueError("Duplicate case_id: " + case["case_id"])
            identifiers.add(case["case_id"])
            cases.append(case)
    if not cases:
        raise ValueError("Routing dataset must not be empty")
    return cases


def public_input(case):
    """Return only an independent copy of the public request and fixture."""
    return deepcopy(case["input"])


def validate_payload(payload):
    """Validate shape and declarative invariants; never read an oracle or execute.

    The returned dict contains only the four output fields. All action fields
    are required; unused content and destination fields must be empty strings.
    Physical symlink checks occur separately inside evaluate_response.
    """
    try:
        _keys(payload, {"status", "actions", "answer", "question"}, "response")
        if not isinstance(payload["status"], str) or payload["status"] not in {"completed", "clarify"}:
            raise ValueError("Invalid output status")
        actions = payload["actions"]
        if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
            raise ValueError("Expected at most 12 actions")
        for action in actions:
            _keys(action, {"tool", "path", "content", "destination"}, "action")
            if not isinstance(action["tool"], str) or action["tool"] not in {"write", "append", "rename", "mkdir"}:
                raise ValueError("Unsupported declarative tool")
            _utf8(action["content"], "action content", MAX_CONTENT_BYTES)
            if not isinstance(action["destination"], str):
                raise ValueError("Action destination must be a string")
            if action["tool"] in {"rename", "mkdir"} and action["content"] != "":
                raise ValueError("Unused content must be empty")
            if action["tool"] != "rename" and action["destination"] != "":
                raise ValueError("Unused destination must be empty")
        answer = _paths(payload["answer"], "answer")
        _utf8(payload["question"], "question", 16384)
        if payload["status"] == "clarify":
            if actions or answer or not payload["question"].strip():
                raise ValueError("Clarify requires a question and no actions or answers")
        elif payload["question"] != "":
            raise ValueError("Completed responses require an empty question")
    except (ValueError, TypeError) as exc:
        raise PayloadValidationError(str(exc)) from exc
    try:
        for action in actions:
            _path_text(action["path"])
            if action["tool"] == "rename":
                _path_text(action["destination"])
    except ValueError as exc:
        raise PayloadValidationError(str(exc), "path_safety") from exc
    return deepcopy(payload)


def _preflight(root, payload, before):
    """Check every path and simulate the sequence before any fixture mutation."""
    for action in payload["actions"]:
        _checked_path(root, action["path"])
        if action["tool"] == "rename":
            _checked_path(root, action["destination"])
    for answer in payload["answer"]:
        _checked_path(root, answer)
    files = {path: entry["size"] for path, entry in before["entries"].items() if entry["kind"] == "file"}
    directories = {path for path, entry in before["entries"].items() if entry["kind"] == "directory"}
    for action in payload["actions"]:
        tool, path = action["tool"], action["path"]
        parent = PurePosixPath(path).parent.as_posix()
        if tool == "mkdir":
            chain = [path, *(value.as_posix() for value in PurePosixPath(path).parents if value.as_posix() != ".")]
            if any(value in files for value in chain):
                raise ValueError("Directory action conflicts with a file")
            directories.update(chain)
        else:
            if parent != "." and parent not in directories:
                raise ValueError("Action parent directory must exist")
            if tool == "write":
                if path in directories:
                    raise ValueError("Cannot write over a directory")
                files[path] = len(action["content"].encode("utf-8"))
            elif tool == "append":
                if path not in files:
                    raise ValueError("Append source must be an existing regular file")
                files[path] += len(action["content"].encode("utf-8"))
            elif tool == "rename":
                destination = action["destination"]
                destination_parent = PurePosixPath(destination).parent.as_posix()
                if path not in files or destination in files or destination in directories:
                    raise ValueError("Rename needs an existing file and an unused destination")
                if destination_parent != "." and destination_parent not in directories:
                    raise ValueError("Rename destination parent must exist")
                files[destination] = files.pop(path)
        if any(size > MAX_FILE_BYTES for size in files.values()) or sum(files.values()) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Action sequence exceeds snapshot byte limits")


def _execute(root, actions):
    for action in actions:
        path = _checked_path(root, action["path"])
        if action["tool"] == "mkdir":
            path.mkdir(parents=True, exist_ok=True)
        elif action["tool"] == "write":
            path.write_bytes(action["content"].encode("utf-8"))
        elif action["tool"] == "append":
            with path.open("ab") as stream:
                stream.write(action["content"].encode("utf-8"))
        else:
            path.rename(_checked_path(root, action["destination"]))


def evaluate_response(case, payload):
    """Score actual files and answer paths, independently of model self-report.

    Bad cases/oracles raise ValueError. Invalid output or a failed preflight
    returns task_success=0 with unchanged before/after snapshots. Unexpected
    execution errors retain the observed partial effects in the after snapshot.
    """
    with TemporaryDirectory(prefix="routing-fixture-") as temporary:
        root = Path(temporary)
        case = _validate_case(case, root)
        for directory in sorted(case["input"]["directories"], key=lambda value: (value.count("/"), value)):
            _checked_path(root, directory).mkdir(parents=True, exist_ok=True)
        for path, content in case["input"]["files"].items():
            _checked_path(root, path).write_bytes(content.encode("utf-8"))
        before = snapshot(root)
        report = {"schema_valid": False, "task_success": 0, "error_category": None,
                  "checks": [], "before_snapshot": before, "after_snapshot": deepcopy(before)}
        try:
            canonical = validate_payload(payload)
        except PayloadValidationError as exc:
            report["error_category"] = exc.category
            report["checks"].append({"name": "schema", "passed": False, "detail": str(exc)})
            return report
        report["schema_valid"] = True
        report["checks"].append({"name": "schema", "passed": True})
        try:
            _preflight(root, canonical, before)
        except ValueError as exc:
            report["error_category"] = "action_precondition"
            report["checks"].append({"name": "action_preflight", "passed": False, "detail": str(exc)})
            return report
        try:
            _execute(root, canonical["actions"])
        except (OSError, ValueError) as exc:
            report["after_snapshot"] = snapshot(root)
            report["error_category"] = "execution"
            report["checks"].append({"name": "execution", "passed": False, "detail": type(exc).__name__})
            return report
        after = snapshot(root)
        report["after_snapshot"] = after
        expected = case["expected"]
        actual_files = {path: entry for path, entry in after["entries"].items() if entry["kind"] == "file"}
        actual_directories = {path for path, entry in after["entries"].items() if entry["kind"] == "directory"}
        expected_bytes = {path: content.encode("utf-8") for path, content in expected["files"].items()}
        checks = [
            ("status", canonical["status"] == expected["status"], "status_mismatch"),
            ("file_inventory", set(actual_files) == set(expected_bytes), "file_inventory"),
            ("file_bytes", all(path in actual_files and actual_files[path]["size"] == len(data)
                and actual_files[path]["sha256"] == hashlib.sha256(data).hexdigest()
                for path, data in expected_bytes.items()), "file_content"),
            ("directory_inventory", actual_directories == set(expected["directories"]), "directory_inventory"),
            ("answer_paths", set(canonical["answer"]) == set(expected["answer"]), "answer_mismatch"),
            ("clarification_question", expected["status"] != "clarify" or any(
                term.casefold() in canonical["question"].casefold() for term in expected["question_terms_any"]), "clarification"),
        ]
        for name, passed, category in checks:
            report["checks"].append({"name": name, "passed": bool(passed)})
            if not passed and report["error_category"] is None:
                report["error_category"] = category
        report["task_success"] = int(all(check["passed"] for check in report["checks"]))
        return report
