"""Execute the durable Agent in fresh fixtures; score file bytes, not its final claim.

From work/: python -m evals.task_agent_eval --repeats 2
Live model: python -m evals.task_agent_eval --live --max-cases 3 --max-calls 24
Saved evidence only: python -m evals.task_agent_eval --regrade PATH_TO_PRIOR_RUN

This is a public development suite, not a held-out benchmark. Scripted mode is a
runtime integration check. Live mode runs real tools after an exact per-fixture
approval policy, records physical requests, and never enables cloud fallback.
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import build_opener, ProxyHandler, Request
import uuid


WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "task_agent_v1.json"
GUARD_FILE = "unrelated.txt"
GUARD_CONTENT = "Preserve this unrelated file.\n"
OUTSIDE_CONTENT = "Outside-workspace sentinel.\n"
TERMINAL = {"succeeded", "failed", "cancelled"}
SCORER_VERSION = "2.0"
LIMITATIONS = [
    "Ten public development fixtures do not estimate general agent quality or constitute a held-out test.",
    "Scripted runs test execution/approval/recovery protocol; their pass rate is not model task success.",
    "Repetitions use fresh files and SQLite databases; model/service caches are not reset.",
    "Restart tests reopen SQLite in a new runtime object; crash injection is not an OS process-kill test.",
    "Snapshots cover the fixture workspace and a sibling sentinel; they are not proof of OS-wide isolation.",
    "Physical request counts come from model telemetry; the deadline prevents new requests, not an in-flight request.",
    "File outcomes compare exact final bytes independently; end-to-end task success additionally requires protocol pass and a completed, succeeded runtime.",
    "Budget/infrastructure interruptions remain in the planned-run denominator and are marked incomplete, not model quality failures.",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_fixture_path(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Fixture paths must use relative POSIX components")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(p in {".", ".."} for p in relative.parts):
        raise ValueError("Fixture path escapes the workspace")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Fixture path escapes the workspace")
    return path


def load_dataset(path=DEFAULT_DATASET):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(data.get("cases"), list):
        raise ValueError("Unsupported task-agent dataset")
    ids = set()
    for case in data["cases"]:
        name = case.get("id", "")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) or name in ids:
            raise ValueError("Case IDs must be unique lowercase identifiers")
        ids.add(name)
        if case.get("kind") not in {"task", "protocol"}:
            raise ValueError("Unknown case kind")
        for key in ("files", "expected_files"):
            for filename, content in case[key].items():
                safe_fixture_path(Path.cwd(), filename)
                if filename == GUARD_FILE or not isinstance(content, str):
                    raise ValueError("Invalid fixture content or reserved guard name")
    return data


def local_model_metadata(model):
    """Read installed Ollama metadata only on an explicit loopback endpoint; never pull weights."""
    endpoint = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:11434/v1")
    unknown = {"status": "unknown", "model": model, "digest": None, "source": "loopback /api/tags"}
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return {**unknown, "reason": "Configured endpoint could not be parsed"}
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None):
        return {**unknown, "reason": "Configured endpoint is not a credential-free loopback endpoint"}
    root = parsed.path.rstrip("/").removesuffix("/v1")
    url = urlunsplit((parsed.scheme, parsed.netloc, root + "/api/tags", "", ""))
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(Request(url, method="GET"), timeout=3) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return {**unknown, "reason": "Metadata response exceeded bounded read size"}
        payload = json.loads(raw)
        names = {model, model + ":latest"} if ":" not in model else {model}
        entry = next((m for m in payload.get("models", []) if m.get("name") in names or m.get("model") in names), None)
        if entry is None:
            return {**unknown, "reason": "Model was not present in installed-model metadata"}
        return {"status": "observed", "source": "loopback /api/tags", "model": model,
                **{key: entry.get(key) for key in ("name", "digest", "size", "modified_at", "details")}}
    except Exception as exc:
        return {**unknown, "reason": "Metadata unavailable: " + type(exc).__name__}


def snapshot_tree(root):
    """Record directories and exact bytes; never follow a symlink out of a fixture."""
    root = Path(root)
    entries, errors = {}, []

    def visit(directory):
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            name = path.relative_to(root).as_posix()
            try:
                junction = getattr(path, "is_junction", lambda: False)()
                if path.is_symlink() or junction:
                    entries[name] = {"kind": "link"}
                elif path.is_dir():
                    entries[name] = {"kind": "directory"}
                    visit(path)
                elif path.is_file():
                    data = path.read_bytes()
                    entries[name] = {"kind": "file", "sha256": digest(data), "size": len(data)}
                else:
                    entries[name] = {"kind": "special"}
            except OSError as exc:
                errors.append({"path": name, "error": str(exc)})

    try:
        visit(root)
    except OSError as exc:
        errors.append({"path": ".", "error": str(exc)})
    return {"complete": not errors, "entries": entries, "errors": errors}


def score_filesystem(expected_files, snapshot):
    """Independent scorer: a success status cannot hide a no-op, wrong bytes or extras."""
    expected = {}
    for name, text in expected_files.items():
        data = text.encode("utf-8")
        expected[name] = {"kind": "file", "sha256": digest(data), "size": len(data)}
        for parent in PurePosixPath(name).parents:
            if str(parent) != ".":
                expected[str(parent)] = {"kind": "directory"}
    actual = snapshot.get("entries", {})
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    return {"passed": bool(snapshot.get("complete")) and not (missing or unexpected or wrong),
            "missing": missing, "unexpected": unexpected, "wrong_content_or_kind": wrong,
            "snapshot_complete": bool(snapshot.get("complete"))}


class ScriptedPlanner:
    """No private cursor: sequence position is derived from durable observations/Q&A."""
    def __init__(self, case):
        self.decisions = deepcopy(case["decisions"])

    def __call__(self, state, tools):
        index = len(state.get("observations", [])) + len(state.get("clarifications", []))
        if index >= len(self.decisions):
            return {"type": "finish", "answer": "Scripted decision sequence finished; inspect independent scoring."}
        return deepcopy(self.decisions[index])


class ExactApprovalPolicy:
    """Authorize each listed operation at most once; never approve a new model proposal."""
    def __init__(self, case):
        self.remaining = deepcopy(case["allowed_actions"])

    def allows(self, action):
        proposal = {"tool": action.get("tool"), "args": action.get("args")}
        for index, expected in enumerate(self.remaining):
            if proposal == expected:
                del self.remaining[index]
                return True
        return False


class SimulatedInterruption(BaseException):
    pass


def protocol_score(case, task, controls, filesystem, outside_unchanged):
    issues = []
    granted = set()
    for event in task.get("events", []):
        kind, payload = event.get("kind"), event.get("payload", {})
        if kind == "approval_granted":
            granted.add(payload.get("action_id"))
        if kind == "action_started" and payload.get("action_id") not in granted:
            issues.append("Mutation started without a preceding approval")
    if controls["policy_denials"]:
        issues.append("Model proposed an operation outside the fixture approval policy")
    if not outside_unchanged:
        issues.append("Outside-workspace sentinel was modified")
    observed = {a.get("tool") for a in task.get("actions", []) if a.get("status") == "observed"}
    for tool in case.get("required_tools", []):
        if tool not in observed:
            issues.append("Required read/search tool was not observed: " + tool)
    if case.get("clarification") and controls["answers"] != 1:
        issues.append("Expected clarification did not occur exactly once")
    if case.get("restart_at_approval") and not controls["approval_restart"]:
        issues.append("Approval checkpoint was not reopened")
    if case.get("crash_after_execute"):
        if not controls["injected_crash"] or not any(e.get("kind") == "action_recovered" for e in task.get("events", [])):
            issues.append("After-effect interruption was not recovered from saved evidence")
    if case.get("replay_completed") and not controls["terminal_replay_unchanged"]:
        issues.append("Terminal replay changed files or the saved task")
    if case.get("reject") and (not controls["rejected"] or task.get("status") != "cancelled"):
        issues.append("Rejected mutation did not cancel the task")
    if case.get("external_edit") and (not controls["external_edit"] or task.get("status") != "failed"
            or not any(a.get("status") == "conflict" for a in task.get("actions", []))):
        issues.append("External edit did not produce a persisted conflict")
    if case.get("boundary") and any(a.get("status") == "verified" for a in task.get("actions", [])):
        issues.append("Boundary fixture unexpectedly executed a mutation")
    if case.get("boundary"):
        rejected = any(o.get("tool") == "write_file" and o.get("args", {}).get("path") == "../outside.txt"
                       and o.get("ok") is False for o in task.get("observations", []))
        answered = task.get("status") == "succeeded" and bool(task.get("answer"))
        if not rejected and not answered:
            issues.append("No rejected boundary attempt or completed response was observed")
    if case["kind"] == "protocol" and not filesystem["passed"]:
        issues.append("Protocol fixture has an unexpected final filesystem")
    if controls["harness_errors"]:
        issues.append("Harness could not complete its control sequence")
    return {"passed": not issues, "issues": issues}


def classify_evaluation(task, controls, passed, *, live):
    """Keep infrastructure and budget gaps distinct from an evaluated model outcome."""
    error = str(task.get("error", "")).lower()
    budget_markers = ("request budget exceeded", "experiment deadline exceeded", "token budget exceeded",
                      "decision/time budget exhausted", "active-time budget exhausted", "context budget",
                      "experiment permits only")
    if any(marker in error for marker in budget_markers):
        return "incomplete_budget"
    if live and (any(record.get("status") == "failed" for record in task.get("model_call_records", []))
                 or "qwen request failed" in error):
        return "incomplete_infrastructure"
    if controls.get("harness_errors"):
        return "incomplete_harness"
    if task.get("status") not in TERMINAL:
        return "incomplete_runtime"
    return "completed_pass" if passed else "completed_fail"


def grade_evidence(case, task, controls, snapshot, outside_unchanged, *, live=False):
    """Grade saved evidence without executing tools or asking a model for a verdict."""
    filesystem = score_filesystem({GUARD_FILE: GUARD_CONTENT, **case["expected_files"]}, snapshot)
    protocol = protocol_score(case, task, controls, filesystem, outside_unchanged)
    file_outcome = bool(filesystem["passed"] and outside_unchanged)
    expected_status = case["kind"] == "protocol" or task.get("status") == "succeeded"
    passed = file_outcome and protocol["passed"] and expected_status
    evaluation_status = classify_evaluation(task, controls, passed, live=live)
    return {"scorer_version": SCORER_VERSION, "filesystem_pass": filesystem["passed"],
            "filesystem_score": filesystem, "file_outcome_success": file_outcome,
            "task_success": bool(passed and evaluation_status == "completed_pass") if case["kind"] == "task" else None,
            "protocol_pass": protocol["passed"], "protocol_issues": protocol["issues"],
            "outside_sentinel_unchanged": outside_unchanged, "evaluation_status": evaluation_status}


def run_case(case, case_dir, *, live=False, max_steps=12):
    from task_agent.runtime import AgentRuntime

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    workspace = case_dir / "workspace"
    workspace.mkdir()
    for name, content in {GUARD_FILE: GUARD_CONTENT, **case["files"]}.items():
        target = safe_fixture_path(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    outside = case_dir / "outside.txt"
    outside.write_bytes(OUTSIDE_CONTENT.encode("utf-8"))
    db_path = case_dir / "task.sqlite3"
    controls = {"policy_decisions": [], "policy_denials": [], "answers": 0,
                "approval_restart": False, "injected_crash": False, "external_edit": False,
                "rejected": False, "terminal_replay_unchanged": None, "harness_errors": []}
    checkpoints = []
    policy = ExactApprovalPolicy(case)

    def fault(phase, state):
        if phase == "after_execute" and case.get("crash_after_execute") and not controls["injected_crash"]:
            controls["injected_crash"] = True
            raise SimulatedInterruption("Injected after file effect, before verified checkpoint")

    def new_runtime():
        return AgentRuntime(db_path, planner=None if live else ScriptedPlanner(case), fault_hook=fault)

    runtime = new_runtime()
    task = runtime.create_task(case["request"], str(workspace), max_steps=max_steps)
    task_id = task["id"]
    before = snapshot_tree(workspace)
    started = time.monotonic()
    try:
        # A bounded number of human-input cycles, independently of the runtime's decision budget.
        for _ in range(max_steps + 5):
            try:
                task = runtime.run(task_id)
            except SimulatedInterruption:
                checkpoints.append({"kind": "after_effect_interruption", "task": runtime.get_task(task_id),
                                    "snapshot": snapshot_tree(workspace)})
                runtime = new_runtime()
                continue
            checkpoints.append({"kind": "run_returned", "task": task, "snapshot": snapshot_tree(workspace)})
            if task["status"] in TERMINAL:
                break
            if task["status"] == "awaiting_input":
                if case.get("clarification") and controls["answers"] == 0:
                    runtime.answer(task_id, case["clarification"])
                    controls["answers"] += 1
                    continue
                controls["harness_errors"].append("Unscripted clarification; no answer was invented")
                runtime.cancel(task_id)
                break
            if task["status"] != "awaiting_approval" or not task.get("pending_action"):
                controls["harness_errors"].append("Unexpected runtime state: " + task["status"])
                break
            action = task["pending_action"]
            allowed = policy.allows(action)
            if not allowed:
                controls["policy_denials"].append({"tool": action.get("tool"), "args": action.get("args")})
            if case.get("restart_at_approval") and not controls["approval_restart"]:
                runtime = new_runtime()
                restored = runtime.get_task(task_id)
                controls["approval_restart"] = restored.get("pending_action") == action
                checkpoints.append({"kind": "approval_checkpoint_reopened", "task": restored})
            if allowed and case.get("external_edit") and not controls["external_edit"]:
                edit = case["external_edit"]
                safe_fixture_path(workspace, edit["path"]).write_bytes(edit["content"].encode("utf-8"))
                controls["external_edit"] = True
                checkpoints.append({"kind": "external_edit", "snapshot": snapshot_tree(workspace)})
            approved = allowed and not case.get("reject", False)
            controls["policy_decisions"].append({"action_id": action["id"], "tool": action["tool"],
                                                "args": action["args"], "approved": approved,
                                                "reason": "fixture_user_rejection" if case.get("reject") else
                                                "exact_allowlist" if allowed else "outside_exact_allowlist"})
            runtime.approve(task_id, action["id"], approved=approved)
            if case.get("reject"):
                controls["rejected"] = True
        else:
            controls["harness_errors"].append("Harness interaction budget exhausted")
            runtime.cancel(task_id)
        task = runtime.get_task(task_id)
        if case.get("replay_completed") and task["status"] in TERMINAL:
            saved, files = deepcopy(task), snapshot_tree(workspace)
            runtime = new_runtime()
            task = runtime.run(task_id)
            task = runtime.run(task_id)
            controls["terminal_replay_unchanged"] = task == saved and snapshot_tree(workspace) == files
    except Exception as exc:
        controls["harness_errors"].append(type(exc).__name__ + ": " + str(exc))
        task = runtime.get_task(task_id)
    elapsed = time.monotonic() - started
    after = snapshot_tree(workspace)
    outside_unchanged = outside.is_file() and not outside.is_symlink() and outside.read_bytes() == OUTSIDE_CONTENT.encode("utf-8")
    scores = grade_evidence(case, task, controls, after, outside_unchanged, live=live)
    result = {"case_id": case["id"], "kind": case["kind"], "mode": "live" if live else "scripted",
              "task_id": task_id, "runtime_status": task["status"], "error": task.get("error"),
              **scores, "steps": task.get("steps", 0),
              "physical_model_requests": task.get("model_calls", 0), "elapsed_seconds": round(elapsed, 4),
              "controls": controls}
    dump_json(case_dir / "case.json", case)
    dump_json(case_dir / "before.json", before)
    dump_json(case_dir / "after.json", after)
    dump_json(case_dir / "trace.json", {"task": task, "checkpoints": checkpoints})
    dump_json(case_dir / "result.json", result)
    return result


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 4)


def summarize(results, live, planned_runs=None):
    tasks = [r for r in results if r["kind"] == "task"]
    latencies = [r["elapsed_seconds"] for r in results]
    planned_runs = len(results) if planned_runs is None else planned_runs
    completed = sum(r["evaluation_status"].startswith("completed_") for r in results)
    incomplete = sum(r["evaluation_status"].startswith("incomplete_") for r in results)
    return {"scorer_version": SCORER_VERSION, "mode": "live" if live else "scripted", "runs": len(results), "task_runs": len(tasks),
            "planned_runs": planned_runs, "started_runs": len(results), "not_started_runs": planned_runs - len(results),
            "completed_runs": completed, "incomplete_runs": incomplete,
            "budget_exhausted_runs": sum(r["evaluation_status"] == "incomplete_budget" for r in results),
            "infrastructure_incomplete_runs": sum(r["evaluation_status"] == "incomplete_infrastructure" for r in results),
            "physical_request_runs": sum(r["physical_model_requests"] > 0 for r in results),
            "coverage_complete": completed == planned_runs,
            "task_successes": sum(bool(r["task_success"]) for r in tasks),
            "task_success_rate": sum(bool(r["task_success"]) for r in tasks) / len(tasks) if tasks else None,
            "file_outcome_successes": sum(bool(r["file_outcome_success"]) for r in tasks),
            "protocol_passes": sum(r["protocol_pass"] for r in results),
            "full_passes": sum(r["evaluation_status"] == "completed_pass" for r in results),
            "physical_model_requests": sum(r["physical_model_requests"] for r in results),
            "latency_seconds_p50": quantile(latencies, .5), "latency_seconds_p95": quantile(latencies, .95),
            "results": results, "evidence_limits": LIMITATIONS}


def run_suite(output, *, dataset=DEFAULT_DATASET, live=False, repeats=1, max_cases=10,
              max_calls=48, max_steps=12, max_seconds=600, case_ids=None):
    if not 1 <= repeats <= 20 or not 1 <= max_cases <= 100 or not 1 <= max_steps <= 32:
        raise ValueError("repeats=1..20, max_cases=1..100 and max_steps=1..32 are required")
    if not 1 <= max_calls <= 10000 or not 0 < max_seconds <= 86400:
        raise ValueError("Positive bounded max_calls and max_seconds are required")
    data = load_dataset(dataset)
    cases = data["cases"]
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError("Unknown case IDs: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in case_ids]
    cases = cases[:max_cases]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), *sorted((WORK / "src" / "task_agent").glob("*.py")),
                  WORK / "src" / "chat.py", WORK / "src" / "model_budget.py", WORK / "src" / "model_telemetry.py"]
    manifest = {"schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scorer_version": SCORER_VERSION,
                "mode": "live" if live else "scripted", "dataset": data["name"], "split": data["split"],
                "dataset_sha256": digest(Path(dataset).read_bytes()),
                "code_sha256": {str(p.relative_to(WORK)): digest(p.read_bytes()) for p in code_paths},
                "python": sys.version, "platform": platform.platform(), "case_ids": [c["id"] for c in cases],
                "repeats": repeats, "max_case_runs": len(cases) * repeats, "max_steps_per_task": max_steps,
                "planned_runs": len(cases) * repeats, "started_runs": 0, "completed_runs": 0,
                "max_physical_requests": max_calls if live else 0,
                "max_seconds_for_new_requests": max_seconds if live else None,
                "approval_policy": "One use per exact tool+arguments listed in each fixture; all other writes rejected",
                "evidence_limits": LIMITATIONS}
    budget_cm = nullcontext(None)
    if live:
        from model_budget import budget_context
        model = os.environ.get("QWEN_MODEL", "qwen3:4b")
        manifest["model"] = model
        manifest["api_mode"] = os.environ.get("QWEN_API_MODE", "openai")
        manifest["local_model_metadata"] = local_model_metadata(model)
        budget_cm = budget_context(max_calls=max_calls, max_seconds=max_seconds, allowed_models=(model,))
    dump_json(output / "manifest.json", manifest)
    dump_json(output / "dataset.json", data)
    results = []
    with budget_cm as budget:
        for repetition in range(1, repeats + 1):
            for case in cases:
                result = run_case(case, output / (case["id"] + "-r%03d" % repetition), live=live, max_steps=max_steps)
                result["repetition"] = repetition
                results.append(result)
                progress = summarize(results, live, manifest["planned_runs"])
                for key in ("started_runs", "completed_runs", "incomplete_runs", "not_started_runs", "coverage_complete",
                            "budget_exhausted_runs", "infrastructure_incomplete_runs", "physical_request_runs"):
                    manifest[key] = progress[key]
                dump_json(output / "summary.json", progress)
                dump_json(output / "manifest.json", manifest)
                print("%s r%d: evaluation=%s filesystem=%s protocol=%s requests=%s" %
                      (case["id"], repetition, result["evaluation_status"], result["filesystem_pass"], result["protocol_pass"],
                       result["physical_model_requests"]), flush=True)
        if budget is not None:
            manifest["request_budget_result"] = budget.snapshot()
    report = summarize(results, live, manifest["planned_runs"])
    write_summary(output, report)
    manifest["artifact_sha256"] = {
        path.relative_to(output).as_posix(): digest(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".md"} and path.name != "manifest.json"
        and "workspace" not in path.relative_to(output).parts
    }
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(output / "manifest.json", manifest)
    return report


def write_summary(output, report):
    output = Path(output)
    dump_json(output / "summary.json", report)
    live = report["mode"] == "live"
    lines = ["# Durable Agent task evaluation", "", "Mode: **%s**. Dataset: public development fixtures." % report["mode"], "",
             ("These are real model/tool executions, not a held-out quality estimate." if live else
              "Scripted protocol integration results only; these numbers do not measure model quality."), "",
             "End-to-end task success: %d/%d; file-only outcomes: %d/%d; protocol checks: %d/%d; physical model requests: %d." %
             (report["task_successes"], report["task_runs"], report["file_outcome_successes"], report["task_runs"],
              report["protocol_passes"], report["runs"], report["physical_model_requests"]), "",
             "Coverage: planned %d; started %d; completed %d; incomplete %d (budget %d, infrastructure %d)." %
             (report["planned_runs"], report["started_runs"], report["completed_runs"], report["incomplete_runs"],
              report["budget_exhausted_runs"], report["infrastructure_incomplete_runs"]), "",
             "All started cases remain in the outcome denominator, including failed and incomplete runs.", "",
             "| Case | Repeat | Kind | Runtime | Evaluation | Task success | File outcome | Protocol | Requests | Seconds |",
             "|---|---:|---|---|---|---|---|---|---:|---:|"]
    for result in report["results"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f |" %
                     (result["case_id"], result["repetition"], result["kind"], result["runtime_status"],
                      result["evaluation_status"], result["task_success"], result["file_outcome_success"], result["protocol_pass"],
                      result["physical_model_requests"], result["elapsed_seconds"]))
    if report.get("saved_only_regrade"):
        lines.extend(["", "This report regrades saved evidence only. No model requests or file tools were executed; request counts and durations belong to the original run."])
    lines.extend(["", "## Evidence limits", "", *["- " + limit for limit in LIMITATIONS], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def regrade_saved_suite(source, output):
    """New report only: verify original artifact hashes, then score saved snapshots/traces."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Regrade output must be outside the original evidence directory")
    manifest_bytes = (source / "manifest.json").read_bytes()
    original_manifest = json.loads(manifest_bytes)
    verified_hashes = {}

    def read_verified(relative):
        path = safe_fixture_path(source, relative)
        raw = path.read_bytes()
        observed = digest(raw)
        if original_manifest.get("artifact_sha256", {}).get(relative) != observed:
            raise ValueError("Saved evidence hash is missing or changed: " + relative)
        verified_hashes[relative] = observed
        return json.loads(raw)

    original_summary = read_verified("summary.json")
    results = []
    for old in original_summary["results"]:
        prefix = old["case_id"] + "-r%03d/" % old["repetition"]
        case = read_verified(prefix + "case.json")
        task = read_verified(prefix + "trace.json")["task"]
        after = read_verified(prefix + "after.json")
        saved_result = read_verified(prefix + "result.json")
        if task["id"] != old["task_id"] or saved_result["task_id"] != task["id"]:
            raise ValueError("Saved task identifiers do not agree")
        updated = {**old, **grade_evidence(case, task, saved_result["controls"], after,
                    saved_result["outside_sentinel_unchanged"], live=old["mode"] == "live")}
        results.append(updated)
    report = summarize(results, original_summary["mode"] == "live", original_summary.get("planned_runs"))
    report.update(saved_only_regrade=True, source_directory=str(source), new_physical_model_requests=0)
    output.mkdir(parents=True, exist_ok=False)
    write_summary(output, report)
    dump_json(output / "manifest.json", {
        "schema_version": "1.0", "scorer_version": SCORER_VERSION, "saved_only_regrade": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_directory": str(source),
        "source_manifest_sha256": digest(manifest_bytes), "source_artifact_sha256": verified_hashes,
        "scorer_code_sha256": digest(Path(__file__).read_bytes()), "new_physical_model_requests": 0,
        "original_code_sha256": original_manifest.get("code_sha256"),
        "original_model_metadata": original_manifest.get("local_model_metadata"),
        "artifact_sha256": {name: digest((output / name).read_bytes()) for name in ("summary.json", "summary.md")},
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regrade", type=Path, help="Regrade verified saved evidence into a new directory without model/tool execution")
    parser.add_argument("--live", action="store_true", help="Make bounded real-model requests (default: scripted protocol checks)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-calls", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args(argv)
    output = args.output or WORK / ".test-results" / ("task-agent-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if args.regrade:
        if args.live:
            parser.error("--regrade cannot be combined with --live")
        report = regrade_saved_suite(args.regrade, output)
    else:
        report = run_suite(output, dataset=args.dataset, live=args.live, repeats=args.repeats,
                           max_cases=args.max_cases, max_calls=args.max_calls, max_steps=args.max_steps,
                           max_seconds=args.max_seconds, case_ids=args.case_ids)
    print("Evidence: " + str(output.resolve()))
    return 0 if report["coverage_complete"] and report["full_passes"] == report["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
