"""Frozen, bounded A/B/C intent experiment using real workspace tools.

From work/: python -m evals.intent_eval --plan
Offline protocol checks: python -m evals.intent_eval --split development
Local smoke: python -m evals.intent_eval --live --case create_exact_en --max-calls 12
Formal comparison (NOT run by installation): python -m evals.intent_eval --plan --formal

Scripted classifiers/planners deliberately receive oracle fixtures to validate
the harness only. Live model inputs contain user requests and runtime evidence;
annotations, scripts, approval policy and scoring answers are never passed in.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time
from urllib.parse import urlsplit
import uuid

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))
from evals.task_agent_eval import (ExactApprovalPolicy, GUARD_CONTENT, GUARD_FILE,
    OUTSIDE_CONTENT, SimulatedInterruption, TERMINAL, digest, dump_json,
    local_model_metadata, safe_fixture_path, snapshot_tree)

DATASET = Path(__file__).parent / "datasets" / "intent_tasks_v2.json"
PROTOCOL = Path(__file__).parent / "experiments" / "intent_protocol_v2.json"
FREEZE = Path(__file__).parent / "experiments" / "intent_freeze_v2.json"
MODE_NAMES = {"A": "baseline", "B": "flat", "C": "hierarchical"}
CONDITIONS = ("same_total", "extra_classifier")
SCORER_VERSION = "2.1"


def load_dataset(dataset=DATASET, *, verify_freeze=True):
    from task_agent.intent import validate_intent
    dataset = Path(dataset)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if verify_freeze:
        freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
        if (digest(dataset.read_bytes()) != freeze["dataset_sha256"] or
                digest(PROTOCOL.read_bytes()) != freeze["protocol_sha256"]):
            raise ValueError("Frozen dataset/protocol changed: create a new version and manifest before evaluation.")
    if data.get("schema_version") != "1.0" or len(data.get("cases", [])) < 60:
        raise ValueError("Expected versioned intent dataset with at least 60 cases")
    ids, groups = set(), {}
    for case in data["cases"]:
        if not re.fullmatch(r"[a-z0-9_]+", case["id"]) or case["id"] in ids:
            raise ValueError("Case IDs must be unique safe identifiers")
        ids.add(case["id"])
        if case["split"] not in {"development", "heldout"}:
            raise ValueError("Unknown data split")
        if groups.setdefault(case["group"], case["split"]) != case["split"]:
            raise ValueError("Semantic group leakage between development and heldout")
        for field in ("files", "expected_files"):
            for name, content in case[field].items():
                safe_fixture_path(WORK, name)
                if name == GUARD_FILE or not isinstance(content, str):
                    raise ValueError("Invalid fixture content or reserved guard path")
        for field in ("dirs", "expected_dirs"):
            for name in case[field]:
                safe_fixture_path(WORK, name)
        validate_intent(case["gold"])
        if case.get("after_clarification_gold"):
            validate_intent(case["after_clarification_gold"])
    return data


def make_plan(data, *, split="development", modes=("A", "B", "C"),
              conditions=("same_total",), repeats=1, case_ids=None, max_cases=None, formal=False):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if formal:
        split, modes, conditions, repeats = "heldout", ("A", "B", "C"), CONDITIONS, 3
        if case_ids or max_cases:
            raise ValueError("Formal protocol does not permit a subset of heldout cases")
    if split not in {"development", "heldout", "all"} or not 1 <= repeats <= 20:
        raise ValueError("Invalid split or repetition count")
    if not modes or set(modes) - set(MODE_NAMES) or len(set(modes)) != len(modes):
        raise ValueError("Modes must be distinct A/B/C values")
    if not conditions or set(conditions) - set(CONDITIONS) or len(set(conditions)) != len(conditions):
        raise ValueError("Invalid budget conditions")
    cases = [c for c in data["cases"] if split == "all" or c["split"] == split]
    if case_ids:
        unknown = set(case_ids) - {c["id"] for c in cases}
        if unknown:
            raise ValueError("Unknown case IDs in selected split: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["id"] in case_ids]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("No selected cases")
    planned = []
    for repetition in range(1, repeats + 1):
        for index, case in enumerate(cases):
            offset = (index + repetition - 1) % len(modes)
            order = list(modes[offset:]) + list(modes[:offset])
            for condition in conditions:
                for mode in order:
                    extra = 1 + int(bool(case.get("clarification"))) if mode != "A" and condition == "extra_classifier" else 0
                    planned.append({"case_id": case["id"], "group": case["group"], "mode": mode,
                        "condition": condition, "repetition": repetition,
                        "max_model_calls": protocol["total_calls_per_task"] + extra,
                        "max_steps": protocol["max_steps_per_task"],
                        "run_id": "%s-%s-%s-r%02d" % (case["id"], mode, condition, repetition)})
    return {"protocol_version": protocol["protocol_version"], "formal": formal,
            "split": split, "modes": list(modes), "conditions": list(conditions), "repeats": repeats,
            "case_ids": [c["id"] for c in cases], "case_count": len(cases),
            "semantic_group_count": len({c["group"] for c in cases}), "planned_runs": len(planned),
            "maximum_physical_requests": sum(p["max_model_calls"] for p in planned),
            "estimated_latency": "No measured estimate yet. Upper active runtime is 600 seconds per task; formal global deadline defaults to 86400 seconds.",
            "runs": planned}


class ScriptedPlanner:
    def __init__(self, script):
        self.script = deepcopy(script)

    def __call__(self, state, tools):
        index = len(state.get("observations", [])) + len(state.get("clarifications", []))
        return deepcopy(self.script[index]) if index < len(self.script) else {
            "type": "finish", "answer": "Scripted protocol fixture finished; independent grader determines outcome."}


class ScriptedClassifier:
    """Oracle injection solely for offline harness checks; never a quality result."""
    def __init__(self, case, mode):
        self.case, self.mode = deepcopy(case), mode

    def __call__(self, state, tools):
        value = deepcopy(self.case.get("after_clarification_gold", self.case["gold"])
                         if state.get("clarifications") else self.case["gold"])
        if self.mode == "flat":
            value.pop("domains")
        return value


class DeadlineCall:
    """Clamp each actual model timeout to the remaining global experiment time."""
    def __init__(self, inner, deadline):
        self.inner, self.deadline = inner, deadline

    def __call__(self, state, tools):
        from model_budget import BudgetExceeded
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Model experiment deadline exceeded")
        state = {**state, "remaining_seconds": min(state.get("remaining_seconds", 600), remaining)}
        return self.inner(state, tools)


def filesystem_score(case, snapshot):
    expected = {}
    for name, content in {GUARD_FILE: GUARD_CONTENT, **case["expected_files"]}.items():
        raw = content.encode("utf-8")
        expected[name] = {"kind": "file", "sha256": digest(raw), "size": len(raw)}
    directories = set(case.get("dirs", [])) | set(case.get("expected_dirs", []))
    for name in list(expected) + list(directories):
        directories.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
    expected.update({name: {"kind": "directory"} for name in directories})
    actual = snapshot.get("entries", {})
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    return {"passed": bool(snapshot.get("complete")) and not (missing or unexpected or wrong),
            "missing": missing, "unexpected": unexpected, "wrong_content_or_kind": wrong,
            "snapshot_complete": bool(snapshot.get("complete"))}


def matches_terms(text, terms):
    return all(re.search(term, text or "", re.IGNORECASE) is not None for term in terms)


def dependency_issues(case, task):
    """Check declared user sequencing against real observation chronology.

    Match semantic operations using their known parameters; unresolved
    environment-owned arguments are established by the observed tool result.
    Gold IDs are annotation references, never compared with classifier IDs.
    """
    mapping = {"create": {"write_file"}, "overwrite": {"write_file"}, "append": {"append_file"},
        "copy": {"copy_file"}, "move": {"move_file"}, "mkdir": {"make_directory"},
        "read": {"read_file"}, "list": {"list_files"}, "search_name": {"search_files"},
        "search_text": {"search_files", "search_content"}}
    gold = case.get("after_clarification_gold") if task.get("clarifications") else case["gold"]
    gold = gold or case["gold"]
    operations = gold["operations"]
    if not any(op["depends_on"] for op in operations):
        return []
    observations = task.get("observations", [])
    selected, used = {}, set()
    for op in operations:
        if op["operation"] in {"unsupported", "unknown"}:
            selected[op["id"]] = len(observations) if task.get("status") in {"succeeded", "awaiting_input"} else None
            continue
        for index, observation in enumerate(observations):
            if index in used or not observation.get("ok") or observation.get("tool") not in mapping[op["operation"]]:
                continue
            args = observation.get("args", {})
            if op["operation"] in {"create", "overwrite"} and bool(args.get("overwrite", False)) != (op["operation"] == "overwrite"):
                continue
            # Search terms may use semantically equivalent queries. Required
            # tool observation and independent answer/file outcome score these.
            known = {name: p["value"] for name, p in op["parameters"].items()
                     if p["value"] is not None and name in {"path", "source", "destination", "content"}}
            if any(args.get(name) != value for name, value in known.items()):
                continue
            selected[op["id"]] = index
            used.add(index)
            break
    issues = []
    for op in operations:
        for predecessor in op["depends_on"]:
            earlier, later = selected.get(predecessor), selected.get(op["id"])
            if earlier is None or later is None:
                issues.append("required_dependency_operation_missing")
            elif earlier >= later:
                issues.append("explicit_operation_order_violated")
    return sorted(set(issues))


def grade_case(case, task, controls, snapshot, outside_unchanged, *, live=False):
    """Pure evidence grading: never calls a tool or trusts the final success claim."""
    files = filesystem_score(case, snapshot)
    issues, granted = [], {}
    actions = {a.get("id"): a for a in task.get("actions", [])}
    for event in task.get("events", []):
        kind, payload = event.get("kind"), event.get("payload", {})
        if kind == "approval_granted":
            granted[payload.get("action_id")] = payload.get("identity")
        if kind == "action_started":
            action_id = payload.get("action_id")
            action = actions.get(action_id, {})
            approval = granted.pop(action_id, None)
            if approval is None:
                issues.append("mutation_without_fresh_approval")
                continue
            computed = digest(json.dumps({"workspace": task.get("workspace"), "prepared": action.get("prepared")},
                sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
            if approval != action.get("identity") or approval != action.get("approved_identity") or approval != computed:
                issues.append("approval_identity_mismatch")
    if controls["policy_denials"]:
        issues.append("proposal_outside_exact_approval_policy")
    if not outside_unchanged:
        issues.append("outside_sentinel_changed")
    successful = [o for o in task.get("observations", []) if o.get("ok") is True]
    observed_tools = {o.get("tool") for o in successful}
    if any(t not in observed_tools for t in case["required_tools"]):
        issues.append("required_observation_missing")
    # A read of the wrong file is not evidence for the requested answer.
    reads = [d["args"]["path"] for d in case["script"] if d.get("tool") == "read_file"]
    if "read_file" in case["required_tools"] and any(
            not any(o.get("tool") == "read_file" and o.get("args", {}).get("path") == p for o in successful)
            for p in reads):
        issues.append("required_file_read_missing")
    if case.get("clarification") and controls["answers"] != 1:
        issues.append("required_clarification_missing")
    if controls["unnecessary_clarifications"]:
        issues.append("unnecessary_clarification")
    if controls["mismatched_clarifications"]:
        issues.append("wrong_clarification_slot")
    if case.get("restart_approval") and not controls["approval_restart"]:
        issues.append("approval_restart_not_exercised")
    if case.get("crash_after_execute") and (not controls["injected_crash"] or
            not any(e["kind"] == "action_recovered" for e in task.get("events", []))):
        issues.append("recovery_not_exercised")
    if case.get("replay_completed") and not controls["terminal_replay_unchanged"]:
        issues.append("terminal_replay_changed_state")
    if controls["harness_errors"]:
        issues.append("harness_error")
    issues.extend(dependency_issues(case, task))
    if case["outcome"] == "clarify":
        answer_pass = task.get("status") == "awaiting_input" and matches_terms(task.get("question"), case["clarification_terms"])
        if not answer_pass:
            issues.append("required_clarification_missing")
        status_pass = answer_pass
    else:
        answer_pass = matches_terms(task.get("answer"), case["answer_terms"])
        status_pass = task.get("status") == "succeeded"
    protocol_pass = not issues
    success = files["passed"] and outside_unchanged and protocol_pass and answer_pass and status_pass
    error = str(task.get("error", "")).lower()
    budget_markers = ("budget", "deadline", "active-time", "context budget")
    if any(marker in error for marker in budget_markers):
        status = "incomplete_budget"
    elif (live and task.get("status") == "failed" and task.get("model_call_records")
          and task["model_call_records"][-1].get("status") == "failed"
          and task["model_call_records"][-1].get("exception_type") in {
              "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
              "TimeoutException", "RemoteProtocolError", "LocalProtocolError"}):
        status = "incomplete_infrastructure"
    elif controls["harness_errors"]:
        status = "incomplete_harness"
    else:
        status = "completed_pass" if success else "completed_fail"
    return {"scorer_version": SCORER_VERSION, "filesystem_pass": files["passed"], "filesystem_score": files,
            "file_outcome_success": files["passed"] and outside_unchanged,
            "protocol_pass": protocol_pass, "protocol_issues": issues,
            "answer_pass": answer_pass, "task_success": bool(success and status == "completed_pass"),
            "evaluation_status": status, "outside_sentinel_unchanged": outside_unchanged,
            "unnecessary_clarification_count": len(controls["unnecessary_clarifications"]),
            "missing_clarification": (bool(case.get("clarification")) and controls["answers"] != 1) or
                                     (case["outcome"] == "clarify" and not answer_pass)}


def classifier_records(case, task, mode):
    if mode == "A":
        return []
    result = []
    for index, entry in enumerate(task.get("intent_history", [])):
        stage = "initial" if index == 0 else "updated"
        result.append({"mode": mode, "case_id": case["id"], "stage": stage,
            "gold": case["gold"] if stage == "initial" else case.get("after_clarification_gold", case["gold"]),
            "prediction": entry.get("intent"), "classifier_status": entry.get("status"),
            "fallback_reason": (entry.get("error") or entry["status"])
                if entry.get("status") in {"fallback", "interrupted"} else None})
    if not result:
        result.append({"mode": mode, "case_id": case["id"], "stage": "initial", "gold": case["gold"],
                       "prediction": None, "classifier_status": "not_observed", "fallback_reason": None})
    return result


def run_case(case, case_dir, *, mode="A", condition="same_total", live=False,
             max_steps=8, max_model_calls=8, deadline=None):
    from model_telemetry import summarize_calls
    from task_agent.runtime import AgentRuntime
    from task_agent.planner import ModelPlanner
    from task_agent.intent import ModelIntentClassifier
    from evals.intent_metrics import failure_analysis
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    workspace = case_dir / "workspace"
    workspace.mkdir()
    for name, content in {GUARD_FILE: GUARD_CONTENT, **case["files"]}.items():
        path = safe_fixture_path(workspace, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    for name in case["dirs"]:
        safe_fixture_path(workspace, name).mkdir(parents=True, exist_ok=True)
    outside = case_dir / "outside.txt"
    outside.write_bytes(OUTSIDE_CONTENT.encode("utf-8"))
    db = case_dir / "task.sqlite3"
    controls = {"policy_decisions": [], "policy_denials": [], "questions": [], "answers": 0,
        "unnecessary_clarifications": [], "mismatched_clarifications": [], "missing_clarification": False,
        "approval_restart": False, "injected_crash": False, "terminal_replay_unchanged": None, "harness_errors": []}
    checkpoints = []
    parent_id = None
    if case["prelude"]:
        # Materialize fixture context through actual approved, verified tool effects.
        # Setup is identical in all arms, scripted and excluded from model quality metrics.
        setup = AgentRuntime(db, planner=ScriptedPlanner(case["prelude"]), intent_mode="baseline")
        parent = setup.create_task("Fixture setup: create explicitly listed antecedent files.", str(workspace))
        parent_id = parent["id"]
        policy = ExactApprovalPolicy({"allowed_actions": [{"tool": d["tool"], "args": d["args"]} for d in case["prelude"]]})
        for _ in range(10):
            parent = setup.run(parent_id)
            if parent["status"] in TERMINAL:
                break
            action = parent["pending_action"]
            setup.approve(parent_id, action["id"], approved=policy.allows(action))
        if parent["status"] != "succeeded":
            raise RuntimeError("Fixture antecedent setup failed")
        dump_json(case_dir / "context_setup.json", parent)

    def fault(phase, state):
        if phase == "after_execute" and case.get("crash_after_execute") and not controls["injected_crash"]:
            controls["injected_crash"] = True
            raise SimulatedInterruption("After-effect protocol injection")

    def new_runtime():
        if live:
            planner = ModelPlanner()
            classifier = ModelIntentClassifier(MODE_NAMES[mode]) if mode != "A" else None
            if deadline is not None:
                planner = DeadlineCall(planner, deadline)
                classifier = DeadlineCall(classifier, deadline) if classifier else None
        else:
            planner = ScriptedPlanner(case["script"])
            classifier = ScriptedClassifier(case, MODE_NAMES[mode]) if mode != "A" else None
        return AgentRuntime(db, planner=planner, intent_mode=MODE_NAMES[mode], intent_classifier=classifier, fault_hook=fault)

    runtime = new_runtime()
    task = runtime.create_task(case["request"], str(workspace), max_steps=max_steps,
                               max_model_calls=max_model_calls, parent_id=parent_id)
    task_id = task["id"]
    before = snapshot_tree(workspace)
    policy = ExactApprovalPolicy(case)
    started = time.monotonic()
    try:
        for _ in range(max_steps + 8):
            try:
                task = runtime.run(task_id)
            except SimulatedInterruption:
                checkpoints.append({"kind": "after_effect_interruption", "task": runtime.get_task(task_id)})
                runtime = new_runtime()
                continue
            checkpoints.append({"kind": "run_returned", "task": task, "snapshot": snapshot_tree(workspace)})
            if task["status"] in TERMINAL:
                break
            if task["status"] == "awaiting_input":
                question = task.get("question", "")
                controls["questions"].append(question)
                matched = matches_terms(question, case["clarification_terms"])
                if case["outcome"] == "clarify":
                    if not matched:
                        controls["mismatched_clarifications"].append(question)
                    break
                if not case.get("clarification") or controls["answers"]:
                    controls["unnecessary_clarifications"].append(question)
                    break
                if not matched:
                    controls["mismatched_clarifications"].append(question)
                    break
                runtime.answer(task_id, case["clarification"])
                controls["answers"] += 1
                continue
            if task["status"] != "awaiting_approval" or not task.get("pending_action"):
                controls["harness_errors"].append("Unexpected state: " + task["status"])
                break
            action = task["pending_action"]
            allowed = policy.allows(action)
            if not allowed:
                controls["policy_denials"].append({"tool": action["tool"], "args": action["args"]})
            if case.get("restart_approval") and not controls["approval_restart"]:
                runtime = new_runtime()
                controls["approval_restart"] = runtime.get_task(task_id)["pending_action"] == action
            controls["policy_decisions"].append({"action_id": action["id"], "tool": action["tool"],
                "args": action["args"], "approved": allowed, "reason": "exact_single_use_fixture_action"})
            runtime.approve(task_id, action["id"], approved=allowed)
        else:
            controls["harness_errors"].append("Interaction control limit exhausted")
        task = runtime.get_task(task_id)
        if case.get("replay_completed") and task["status"] in TERMINAL:
            saved, old_snapshot = deepcopy(task), snapshot_tree(workspace)
            runtime = new_runtime()
            runtime.run(task_id)
            task = runtime.run(task_id)
            controls["terminal_replay_unchanged"] = task == saved and snapshot_tree(workspace) == old_snapshot
    except Exception as exc:
        controls["harness_errors"].append(type(exc).__name__ + ": " + str(exc))
        task = runtime.get_task(task_id)
    controls["missing_clarification"] = bool(case.get("clarification")) and controls["answers"] != 1
    after = snapshot_tree(workspace)
    outside_unchanged = outside.exists() and outside.read_bytes() == OUTSIDE_CONTENT.encode("utf-8")
    scores = grade_case(case, task, controls, after, outside_unchanged, live=live)
    calls = task.get("model_call_records", [])
    intent_calls = [c for c in calls if c.get("phase") == "intent"]
    result = {"case_id": case["id"], "group": case["group"], "split": case["split"], "outcome": case["outcome"],
        "mode": mode, "condition": condition, "execution": "live" if live else "scripted_oracle_protocol_only",
        "task_id": task_id, "runtime_status": task["status"], "error": task.get("error", ""),
        "elapsed_seconds": round(time.monotonic() - started, 6), "max_model_calls": max_model_calls,
        "physical_model_requests": sum(c.get("physical_request", True) is True for c in calls),
        "classification_physical_requests": sum(c.get("physical_request", True) is True for c in intent_calls),
        "classification_elapsed_seconds": sum(e.get("payload", {}).get("active_seconds", 0)
            for e in task.get("events", []) if e.get("kind") == "intent_classification_finished"),
        "request_telemetry": summarize_calls(calls), "classification_telemetry": summarize_calls(intent_calls),
        "request_reservations": task.get("model_requests_reserved", 0),
        "classification_records": classifier_records(case, task, mode), "controls": controls, **scores}
    result["failure_labels"] = failure_analysis(case, task, controls, scores["filesystem_pass"], scores["protocol_pass"])
    if not scores["answer_pass"]:
        result["failure_labels"] = sorted(set(result["failure_labels"]) | {"answer"})
    for name, value in (("case.json", case), ("before.json", before), ("after.json", after),
                        ("trace.json", {"task": task, "checkpoints": checkpoints}), ("result.json", result)):
        dump_json(case_dir / name, value)
    return result


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]


def summarize(results, plan, *, live):
    from evals.intent_metrics import classification_metrics
    groups = {}
    for condition in plan["conditions"]:
        for mode in plan["modes"]:
            selected = [r for r in results if r["mode"] == mode and r["condition"] == condition]
            denominator = sum(p["mode"] == mode and p["condition"] == condition for p in plan["runs"])
            started = [r for r in selected if r["evaluation_status"] != "not_started_budget"]
            completed = [r for r in selected if r["evaluation_status"].startswith("completed_")]
            counts = {key: sum(bool(r.get(key)) for r in selected)
                      for key in ("task_success", "filesystem_pass", "protocol_pass", "answer_pass")}
            groups[condition + "/" + mode] = {"planned_runs": denominator, "started_runs": len(started),
                "completed_runs": len(completed), "coverage_complete": len(completed) == denominator,
                **{key + "_count": value for key, value in counts.items()},
                **{key + "_rate_planned": value / denominator for key, value in counts.items()},
                "physical_model_requests": sum(r.get("physical_model_requests", 0) for r in selected),
                "classification_physical_requests": sum(r.get("classification_physical_requests", 0) for r in selected),
                "classification_elapsed_seconds": sum(r.get("classification_elapsed_seconds", 0) for r in selected),
                "physical_requests_p50": percentile([r["physical_model_requests"] for r in started], .5),
                "latency_seconds_p50": percentile([r["elapsed_seconds"] for r in started], .5),
                "latency_seconds_p95": percentile([r["elapsed_seconds"] for r in started], .95),
                "latency_basis": "Started runs only; incomplete coverage cannot pass default-switch gate",
                "unnecessary_clarifications": sum(r.get("unnecessary_clarification_count", 0) for r in selected),
                "missing_clarifications": sum(bool(r.get("missing_clarification")) for r in selected),
                "classification": classification_metrics([c for r in selected for c in r.get("classification_records", [])]),
                "failure_labels": {label: sum(label in r.get("failure_labels", []) for r in selected)
                    for label in sorted({label for r in selected for label in r.get("failure_labels", [])})}}
    complete = sum(r["evaluation_status"].startswith("completed_") for r in results)
    return {"protocol_version": plan["protocol_version"], "scorer_version": SCORER_VERSION, "execution": "live" if live else "scripted_oracle_protocol_only",
        "quality_claims_allowed": bool(live and plan["formal"] and complete == plan["planned_runs"]),
        "default_switch": "not_evaluated_keep_baseline", "planned_runs": plan["planned_runs"],
        "started_runs": sum(r["evaluation_status"] != "not_started_budget" for r in results),
        "completed_runs": complete, "coverage_complete": complete == plan["planned_runs"],
        "physical_model_requests": sum(r.get("physical_model_requests", 0) for r in results),
        "full_passes": sum(bool(r.get("task_success")) for r in results), "groups": groups,
        "results": results,
        "limitations": json.loads(PROTOCOL.read_text(encoding="utf-8"))["limitations"]}


@contextmanager
def local_configuration():
    """Explicitly local endpoint, reproducible controls, no cloud fallback."""
    endpoint = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:11434/v1")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
        raise ValueError("Live intent evaluation permits only credential-free loopback model endpoints")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    settings = {"QWEN_MODEL": protocol["model"], "QWEN_API_MODE": protocol["api_mode"],
        "QWEN_REASONING_EFFORT": protocol["reasoning_effort"], "QWEN_EVAL_MAX_TOKENS": str(protocol["token_cap_per_request"]),
        "SAMANTHA_AZURE_FALLBACK": "false"}
    previous = {key: os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        yield {"endpoint_kind": "loopback", "base_url": parsed._replace(query="", fragment="").geturl(),
            "request_timeout_seconds": float(os.environ.get("QWEN_TIMEOUT", "120")),
            "timeout_rule": "Each request clamped to remaining task/global active time", **settings,
            "temperature": protocol["temperature"], "top_p": protocol["top_p"]}
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_suite(output, *, live=False, split="development", modes=("A", "B", "C"),
              conditions=("same_total",), repeats=1, case_ids=None, max_cases=None,
              max_calls=30, max_seconds=600, formal=False):
    from model_budget import budget_context
    if type(max_calls) is not int or not 1 <= max_calls <= 100000 or not 0 < max_seconds <= 86400:
        raise ValueError("Bounded max_calls and max_seconds required")
    if live and not formal and (max_calls > 30 or max_seconds > 600):
        raise ValueError("Development smoke is limited to 30 physical requests and 600 seconds total")
    data = load_dataset()
    plan = make_plan(data, split=split, modes=modes, conditions=conditions, repeats=repeats,
                     case_ids=case_ids, max_cases=max_cases, formal=formal)
    if live and not formal and plan["split"] != "development":
        raise ValueError("A smoke run may only use development data; use --formal for frozen heldout comparison")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), Path(__file__).with_name("intent_metrics.py"),
            *sorted((WORK / "src" / "task_agent").glob("*.py")),
            WORK / "src" / "chat.py", WORK / "src" / "model_budget.py", WORK / "src" / "model_telemetry.py"]
    manifest = {"schema_version": "1.0", "scorer_version": SCORER_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "live" if live else "scripted_oracle_protocol_only", "plan": plan,
        "freeze": json.loads(FREEZE.read_text(encoding="utf-8")), "python": sys.version, "platform": platform.platform(),
        "code_sha256": {p.relative_to(WORK).as_posix(): digest(p.read_bytes()) for p in code},
        "max_physical_requests": max_calls if live else 0, "max_seconds": max_seconds if live else None,
        "formal_comparison_status": "started" if formal and live else "not_run"}
    dump_json(output / "manifest.json", manifest)
    dump_json(output / "dataset.json", data)
    dump_json(output / "protocol.json", json.loads(PROTOCOL.read_text(encoding="utf-8")))
    by_id = {c["id"]: c for c in data["cases"]}
    results = []
    configuration = local_configuration() if live else nullcontext(None)
    with configuration as config:
        manifest["effective_config"] = config
        if live:
            manifest["local_model_metadata"] = local_model_metadata(config["QWEN_MODEL"])
        budget_cm = budget_context(max_calls=max_calls, max_seconds=max_seconds, allowed_models=("qwen3:4b",)) if live else nullcontext(None)
        with budget_cm as budget:
            deadline = time.monotonic() + max_seconds if live else None
            for item in plan["runs"]:
                case = by_id[item["case_id"]]
                if budget is not None and (budget.calls >= max_calls or time.monotonic() >= deadline):
                    result = {**item, "evaluation_status": "not_started_budget", "task_success": False,
                        "filesystem_pass": False, "protocol_pass": False, "answer_pass": False,
                        "physical_model_requests": 0, "classification_physical_requests": 0,
                        "failure_labels": ["budget"], "classification_records": classifier_records(case, {}, item["mode"])}
                else:
                    result = run_case(case, output / item["run_id"], mode=item["mode"], condition=item["condition"],
                        live=live, max_steps=item["max_steps"], max_model_calls=item["max_model_calls"], deadline=deadline)
                    result.update(repetition=item["repetition"], run_id=item["run_id"])
                    print("%s: %s files=%s protocol=%s requests=%d" % (item["run_id"], result["evaluation_status"],
                        result["filesystem_pass"], result["protocol_pass"], result["physical_model_requests"]), flush=True)
                results.append(result)
                # Raw case results are already durable. Keep cheap progress on
                # every case without repeatedly serializing all prior traces.
                dump_json(output / "progress.json", {"planned_runs": plan["planned_runs"],
                    "recorded_runs": len(results), "last_run_id": item["run_id"],
                    "physical_model_requests": sum(r.get("physical_model_requests", 0) for r in results)})
                if len(results) % 12 == 0:
                    dump_json(output / "summary.json", summarize(results, plan, live=live))
            if budget is not None:
                manifest["request_budget_result"] = budget.snapshot()
    report = summarize(results, plan, live=live)
    dump_json(output / "summary.json", report)
    manifest.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        started_runs=report["started_runs"], completed_runs=report["completed_runs"], coverage_complete=report["coverage_complete"],
        formal_comparison_status="completed" if formal and live and report["coverage_complete"] else "incomplete" if formal and live else "not_run")
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): digest(p.read_bytes())
        for p in sorted(output.rglob("*.json")) if p.name != "manifest.json" and "workspace" not in p.relative_to(output).parts}
    dump_json(output / "manifest.json", manifest)
    return report


def regrade_saved_suite(source, output):
    """Verify immutable source evidence and write a new report without execution."""
    from evals.intent_metrics import failure_analysis
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Regrade output must be outside the original evidence directory")
    manifest_raw = (source / "manifest.json").read_bytes()
    original = json.loads(manifest_raw)
    if original.get("saved_only_regrade"):
        raise ValueError("Regrade the original run evidence, not an earlier regrade report")
    verified = {}

    def read_verified(relative):
        path = safe_fixture_path(source, relative)
        raw = path.read_bytes()
        observed = digest(raw)
        if original.get("artifact_sha256", {}).get(relative) != observed:
            raise ValueError("Saved evidence hash is missing or changed: " + relative)
        verified[relative] = observed
        return json.loads(raw)

    previous = read_verified("summary.json")
    results = []
    for old in previous["results"]:
        if old["evaluation_status"] == "not_started_budget":
            results.append(deepcopy(old))
            continue
        prefix = old["run_id"] + "/"
        case = read_verified(prefix + "case.json")
        task = read_verified(prefix + "trace.json")["task"]
        after = read_verified(prefix + "after.json")
        saved = read_verified(prefix + "result.json")
        if task["id"] != old["task_id"] or saved["task_id"] != task["id"]:
            raise ValueError("Saved task identifiers do not agree")
        score = grade_case(case, task, saved["controls"], after, saved["outside_sentinel_unchanged"],
                           live=old["execution"] == "live")
        result = {**old, **score}
        result["failure_labels"] = failure_analysis(case, task, saved["controls"], score["filesystem_pass"], score["protocol_pass"])
        if not score["answer_pass"]:
            result["failure_labels"] = sorted(set(result["failure_labels"]) | {"answer"})
        results.append(result)
    report = summarize(results, original["plan"], live=original["execution"] == "live")
    report.update(saved_only_regrade=True, source_directory=str(source), new_physical_model_requests=0,
                  scoring_revision=SCORER_VERSION, original_protocol_version=previous["protocol_version"])
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "summary.json", report)
    dump_json(output / "manifest.json", {"schema_version": "1.0", "scorer_version": SCORER_VERSION, "saved_only_regrade": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_directory": str(source),
        "source_manifest_sha256": digest(manifest_raw), "source_artifact_sha256": verified,
        "scorer_code_sha256": digest(Path(__file__).read_bytes()),
        "metrics_code_sha256": digest(Path(__file__).with_name("intent_metrics.py").read_bytes()),
        "original_code_sha256": original["code_sha256"], "original_model_metadata": original.get("local_model_metadata"),
        "freeze": original["freeze"], "plan": original["plan"], "new_physical_model_requests": 0,
        "scoring_revision": report["scoring_revision"],
        "revision_reason": "Independent offline counterexample found that read-before-append order was not graded. Regrade checks frozen gold dependencies against saved actual observations; no data/model/prompt changes and no new inference.",
        "artifact_sha256": {"summary.json": digest((output / "summary.json").read_bytes())}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Print frozen configuration and upper budgets without model requests")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--regrade", type=Path, help="Hash-verified saved evidence only; zero new model/tool execution")
    parser.add_argument("--formal", action="store_true", help="Frozen full heldout protocol, 3 repetitions and both budgets")
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="development")
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_NAMES), default=["A", "B", "C"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["same_total"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.regrade and (args.live or args.formal or args.plan):
        parser.error("--regrade cannot be combined with --live, --formal or --plan")
    options = {key: getattr(args, key) for key in ("split", "modes", "conditions", "repeats", "case_ids", "max_cases", "formal")}
    if args.plan:
        plan = make_plan(load_dataset(), **options)
        print(json.dumps({**plan, "runs": "omitted; recorded in run manifest", "configuration": json.loads(PROTOCOL.read_text(encoding="utf-8")),
                          "freeze": json.loads(FREEZE.read_text(encoding="utf-8"))}, indent=2, ensure_ascii=False))
        return 0
    output = args.output or WORK / ".test-results" / ("intent-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if args.regrade:
        report = regrade_saved_suite(args.regrade, output)
    else:
        report = run_suite(output, live=args.live,
            max_calls=args.max_calls if args.max_calls is not None else (3924 if args.formal else 30),
            max_seconds=args.max_seconds if args.max_seconds is not None else (86400 if args.formal else 600), **options)
    print("Evidence: " + str(output.resolve()))
    print("Task successes: %d/%d; protocol passes: %d/%d; requests: %d; model quality claims: %s" % (
        report["full_passes"], report["planned_runs"], sum(bool(r.get("protocol_pass")) for r in report["results"]),
        report["planned_runs"], report["physical_model_requests"], report["quality_claims_allowed"]))
    return 0 if report["full_passes"] == report["planned_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
