"""Interactive multi-invocation system eval: actual models, approval and file checks."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "src"))
sys.path.insert(0, str(WORK / "tests"))

from agent_memory import MemoryScope, create_backend
from interaction_log import InteractionJournal
from memory_eval import code_version, file_hash
from real_tasks.task_suite import approval_violations, fixture_path, relative_path, snapshot, write_fixture
from samantha import process

CATALOG = Path(__file__).with_name("cases.json")


def load_cases():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported memory-task catalog.")
    identifiers = set()
    for case in data["cases"]:
        if not case["id"].isalnum() or case["id"] in identifiers:
            raise ValueError("Invalid or repeated case ID.")
        identifiers.add(case["id"])
        for path in case["allowed_changes"]:
            relative_path(path)
        for item in case["fixtures"]:
            relative_path(item["path"])
        for step in case["steps"]:
            if step["status"] not in {"succeeded", "cancelled", "failed"} or not step["request"].strip():
                raise ValueError("Invalid step.")
            for path in [*step.get("files", {}), *step.get("absent", [])]:
                relative_path(path)
    return data["cases"]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def verify_step(step, state, evidence, workspace, before, allowed_changes):
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    check("expected_status", state["status"] == step["status"])
    check("memory_saved", not state.get("memory_error"))
    check("fresh_approval", not approval_violations(evidence["events"]))
    clarifications = sum(event["kind"] == "awaiting_input" and
                         event["data"].get("kind") == "clarification" for event in evidence["events"])
    check("clarification_bounds", step.get("min_clarifications", 0) <= clarifications <=
          step.get("max_clarifications", 3))
    if step.get("no_execution"):
        check("no_execution", not any(event["kind"] == "operation_started" and
              event["data"].get("name") == "commands.execute" for event in evidence["events"]))
    if "confirmation_answer" in step:
        answers = [event["data"]["text"] for event in evidence["events"] if event["kind"] == "user_input"
                   and event["data"].get("source") == "confirmation"]
        check("specified_confirmation", answers == [step["confirmation_answer"]])
    for name, expected in step.get("files", {}).items():
        path = fixture_path(workspace, name)
        check("text:" + name, path.is_file() and path.read_bytes() == expected.encode("utf-8"))
    for name in step.get("absent", []):
        check("absent:" + name, not fixture_path(workspace, name).exists())
    after = snapshot(workspace)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    check("allowed_changes", changed <= set(allowed_changes))
    return {"passed": all(item["passed"] for item in checks), "checks": checks,
            "clarifications": clarifications, "status": state["status"]}


def run_case(case, output, *, backend="none", context_chars=6000, top_k=5,
             services=None, input_fn=None, output_fn=print):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("Unrelated file: keep unchanged.\n", encoding="utf-8")
    for fixture in case["fixtures"]:
        write_fixture(workspace, fixture)
    before = snapshot(workspace)
    log_path = output / "interactions.sqlite3"
    results = []
    user = "terminal-eval-" + uuid.uuid4().hex
    manifest = {"benchmark": "Samantha Memory Tasks v1", "case_id": case["id"],
                "catalog_sha256": file_hash(CATALOG), "suite_sha256": file_hash(__file__),
                **code_version(),
                "backend": backend, "context_budget_chars": context_chars, "top_k": top_k,
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with working_directory(workspace):
        for index, step in enumerate(case["steps"], 1):
            output_fn(f"\n{case['id']} step {index}: {step['request']}\n{step['instruction']}")
            # Reopen the backend between invocations to test durable cross-call use.
            memory = create_backend(backend, path=output / "memory.sqlite3")
            manifest["backend_config"] = memory.describe()
            started = time.perf_counter()
            try:
                state = process(step["request"], services=services, input_fn=input_fn, output_fn=output_fn,
                                log_file=log_path, memory=memory,
                                memory_scope=MemoryScope(user, step.get("project", "project-default")),
                                memory_budget=context_chars, memory_limit=top_k)
            finally:
                memory.close()
            with InteractionJournal(log_path, readonly=True) as journal:
                evidence = journal.read_session(state["log_session_id"])
            (output / f"step-{index}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
            result = verify_step(step, state, evidence, workspace, before, case["allowed_changes"])
            result.update(step=index, duration_ms=round((time.perf_counter() - started) * 1000, 3),
                          context_chars=len(state["memory_context"]), memory_sources=state["memory_sources"])
            results.append(result)
            if not result["passed"]:
                break  # A failed prerequisite cannot supply valid evidence to later steps.
    summary = {"case_id": case["id"], "passed": len(results) == len(case["steps"]) and
               all(result["passed"] for result in results), "steps_completed": len(results),
               "steps_expected": len(case["steps"]), "steps": results}
    manifest["status"] = "passed" if summary["passed"] else "failed"
    manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Samantha terminal memory benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run", help="Run actual models/tools with interactive human approval")
    run.add_argument("case_id")
    run.add_argument("--backend", default="none")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--context-chars", type=int, default=6000)
    run.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    cases = load_cases()
    if args.command == "list":
        for case in cases:
            print(case["id"], case["title"], f"({len(case['steps'])} calls)")
        return 0
    if sys.platform == "win32":
        parser.error("Live terminal tasks require Linux/Bash; use the existing Docker task lab.")
    from dotenv import load_dotenv
    load_dotenv(WORK.parent / ".env", override=False)
    # Keep experiments on one backend/model provider.
    os.environ["SAMANTHA_AZURE_FALLBACK"] = "false"
    case = next((case for case in cases if case["id"] == args.case_id), None)
    if case is None:
        parser.error("Unknown case ID.")
    result = run_case(case, args.output, backend=args.backend,
                      context_chars=args.context_chars, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
