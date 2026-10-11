"""Explicit real-model proposal/reference probe; model commands never execute.

Run this file separately from offline unit tests. Every model confirmation is
declined. Python fixture code creates and renames the test file between calls,
with its own execution/verification provenance, to supply genuine object state.
This measures the live P0 output contract and reference use, not terminal task
success, model-command execution, or improvement over a baseline.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from unittest.mock import Mock
import uuid

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))

from agent_memory import MemoryScope
from chat import Chat
from interaction_log import InteractionJournal, current_operation_reference, operation, record_operation
from memory_eval import code_version, file_hash
from object_tracking.effects import bind_effects, parse_effects, render_effects
from object_tracking.store import ObjectStore
from object_tracking.verification import capture_before, verify_after
from samantha import process
from workflow import default_services


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def model_metadata():
    """Record only public configuration; the tags request is read-only."""
    import httpx

    chat = Chat(allow_azure_fallback=False)
    result = {"endpoint": str(chat.client_qwen.base_url), "model": chat.model_qwen,
              "api_mode": chat.api_mode, "reasoning_effort": chat.reasoning_effort,
              "max_tokens": chat.max_tokens, "temperature": chat.temperature,
              "top_p": chat.top_p, "azure_fallback": False,
              "timeout_seconds": float(os.environ.get("QWEN_TIMEOUT", "120")),
              "context_length": None, "model_digest": None}
    root = result["endpoint"].rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(root + "/api/tags")
            response.raise_for_status()
            data = response.json()
            desired = chat.model_qwen if ":" in chat.model_qwen else chat.model_qwen + ":latest"
            match = next((item for item in data.get("models", [])
                          if desired in {item.get("name"), item.get("model")}), None)
            if match:
                result.update(model_digest=match.get("digest"), model_details=match.get("details"),
                              model_size_bytes=match.get("size"))
            shown = client.post(root + "/api/show", json={"model": chat.model_qwen})
            shown.raise_for_status()
            information = shown.json().get("model_info", {})
            result["model_context_lengths"] = {key: value for key, value in information.items()
                                                if key.endswith(".context_length")}
            # This is model capacity metadata, not an assertion about runtime num_ctx.
            result["runtime_num_ctx"] = None
    except Exception as exc:
        result["model_metadata_error"] = str(exc)
    return result


def effect(kind, path, *, destination=None, content=None, entity=None):
    return {"kind": kind, "path": str(path),
            "destination": None if destination is None else str(destination), "content": content,
            "entity_id": None if entity is None else entity["entity_id"],
            "entity_version": None if entity is None else entity["version"]}


def seed_fixture(proposal, store_path, log_path, scope):
    """Perform deterministic Python setup, never model-generated shell text."""
    with InteractionJournal(log_path) as journal, journal.start_session(
            "Deterministic fixture setup", kind="object_probe_fixture",
            metadata={"source_type": "deterministic_python_fixture"}) as session:
        before = record_operation("fixture.capture_before", capture_before, proposal,
                                  entity=(None if proposal["entity_id"] is None else
                                          _lookup_entity(store_path, scope, proposal["entity_id"])))
        if before["status"] != "ready":
            raise AssertionError("Fixture precondition failed: " + before["reason"])
        with operation("fixture.python.execute", {"effect": proposal,
                        "source_type": "deterministic_python_fixture"}) as outcome:
            reference = current_operation_reference()
            path = Path(proposal["path"])
            if proposal["kind"] == "create":
                path.write_bytes(proposal["content"].encode("utf-8"))
            elif proposal["kind"] == "rename":
                path.rename(proposal["destination"])
            else:
                raise AssertionError("Only create and rename are fixture setup operations.")
            outcome.update(status="succeeded", result={"success": True, "kind": proposal["kind"]})
        verification = record_operation("fixture.verify", verify_after, proposal, before)
        if verification["status"] != "verified":
            raise AssertionError("Fixture postcondition failed: " + verification["reason"])
        store = ObjectStore(store_path)
        try:
            entity = record_operation("fixture.objects.apply", store.apply_verified, scope, proposal,
                                      verification, session_id=reference["session_id"],
                                      operation_id=reference["operation_id"],
                                      starting_directory=os.getcwd(), execution_status="succeeded")
        finally:
            store.close()
        session.finish("succeeded", details={"entity": entity,
                       "note": "Python fixture setup, not model-command execution."})
        return entity


def _lookup_entity(store_path, scope, identifier):
    store = ObjectStore(store_path)
    try:
        return store.lookup(scope, identifier)
    finally:
        store.close()


def evaluate_step(state, evidence, expected, executor):
    """Assess the exact submitted proposal and count fresh declined confirmations."""
    checks = []

    def check(name, passed, details=None):
        item = {"check": name, "passed": bool(passed)}
        if details is not None:
            item["details"] = details
        checks.append(item)

    events = evidence["events"]
    confirmations = [event["data"] for event in events if event["kind"] == "user_input"
                     and event["data"].get("source") == "confirmation"]
    clarifications = [event["data"] for event in events if event["kind"] == "awaiting_input"
                      and event["data"].get("kind") == "clarification"]
    generated = [event["data"].get("result") for event in events
                 if event["kind"] == "operation_finished"
                 and event["data"].get("name") == "agent.generate_commands"]
    raw_proposals = [event["data"]["arguments"]["args"][0] for event in events
                     if event["kind"] == "operation_started"
                     and event["data"].get("name") == "objects.render_commands"]
    calls = Counter(event["data"]["name"] for event in events if event["kind"] == "operation_started"
                    and event["data"].get("name", "").startswith("agent."))
    check("no_model_command_execution", executor.call_count == 0 and not any(
        event["kind"] == "operation_started" and event["data"].get("name") == "commands.execute"
        for event in events))
    check("confirmation_declined_once", len(confirmations) == 1 and confirmations[0]["text"] == "n")
    check("no_unnecessary_clarification", not clarifications)
    check("declined_status", state.get("status") == "cancelled")
    check("review_present_and_reasonable", bool(state.get("command_review"))
          and state["command_review"].get("verdict") == "reasonable")
    proposed = state.get("object_effects", [])
    check("one_supported_effect", len(proposed) == 1)
    if len(proposed) == 1:
        for name in ("kind", "path", "destination", "content", "entity_id", "entity_version"):
            check("effect_matches:" + name, proposed[0].get(name) == expected.get(name),
                  {"expected": expected.get(name), "actual": proposed[0].get(name)})
        try:
            bind_effects(state["commands"], proposed, state["initial_dir"])
            check("commands_match_effect", True)
        except Exception as exc:
            check("commands_match_effect", False, str(exc))
    else:
        check("commands_match_effect", False)
    return {"passed": all(item["passed"] for item in checks), "checks": checks,
            "confirmation_count": len(confirmations), "clarification_count": len(clarifications),
            "model_call_counts": dict(calls), "generated_command_proposals": generated,
            "raw_model_proposals": raw_proposals,
            "state": state, "evidence": evidence}


def run_probe(report_path):
    report_path = Path(report_path).expanduser().resolve()
    artifacts = report_path.with_suffix(report_path.suffix + ".artifacts")
    if report_path.exists() or artifacts.exists():
        raise FileExistsError("Probe output must be fresh: " + str(report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    log_path = artifacts / "interactions.sqlite3"
    store_path = artifacts / "objects.sqlite3"
    scope = MemoryScope("live-object-probe-" + uuid.uuid4().hex, "temporary-fixture")
    report = {"schema_version": 2, "probe": "P0 live proposal/reference contract v2",
              "passed": False, "status": "running", "steps": [], "fixture_seeds": [],
              "scope": {"user_id": scope.user_id, "project_id": scope.project_id},
              "started_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "script_sha256": file_hash(__file__), **code_version(),
              "artifacts": {"journal": str(log_path), "objects": str(store_path)},
              "limitations": ["Every model confirmation is declined; generated commands never execute.",
                              "Fixture transitions use deterministic Python with separate evidence.",
                              "This is not a live task-success or baseline-improvement measurement."]}
    started = time.monotonic()

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        report["model_config"] = model_metadata()
        save()
        with tempfile.TemporaryDirectory(prefix="object-tracking-probe-") as directory:
            workspace = Path(directory).resolve()
            with working_directory(workspace):
                notes, todo = workspace / "notes.txt", workspace / "todo.txt"
                keep = workspace / "keep.txt"
                keep.write_bytes(b"Unrelated fixture: unchanged.\n")
                entity = None
                requests = [
                    ("create", "在当前目录创建 notes.txt，UTF-8 内容严格为 alpha 加一个换行符。留在当前目录。"),
                    ("rename", "把刚才的文件重命名为当前目录下的 todo.txt，保留内容。留在当前目录。"),
                    ("append", "给刚才的文件追加一行 beta，即追加 UTF-8 文本 beta 和一个换行符。留在当前目录。"),
                ]
                for index, (kind, request) in enumerate(requests, 1):
                    expected = parse_effects({"Effects": [effect(kind, todo if kind == "append" else notes,
                                             destination=todo if kind == "rename" else None,
                                             content="alpha\n" if kind == "create" else
                                                     "beta\n" if kind == "append" else None,
                                             entity=entity)]}, str(workspace))[0]
                    executor = Mock(side_effect=AssertionError("Live probe refuses generated command execution."))
                    live_services = replace(default_services(object_tracking=True), execute_commands=executor)
                    store = ObjectStore(store_path)
                    step_started = time.monotonic()
                    try:
                        state = process(request, services=live_services,
                                        input_fn=lambda prompt: "n" if prompt.startswith("Proceed?") else "cancel",
                                        output_fn=lambda _: None, log_file=log_path,
                                        object_store=store, object_scope=scope)
                    finally:
                        store.close()
                    with InteractionJournal(log_path, readonly=True) as journal:
                        evidence = journal.read_session(state["log_session_id"])
                    result = evaluate_step(state, evidence, expected, executor)
                    result.update(index=index, kind=kind, request=request,
                                  duration_seconds=round(time.monotonic() - step_started, 3))
                    executor.assert_not_called()
                    report["steps"].append(result)
                    print(json.dumps({"step": index, "kind": kind, "passed": result["passed"],
                                      "status": state["status"], "error": state.get("error", ""),
                                      "checks_failed": [item["check"] for item in result["checks"]
                                                        if not item["passed"]]}, ensure_ascii=False), flush=True)
                    save()
                    if kind in {"create", "rename"}:
                        entity = seed_fixture(expected, store_path, log_path, scope)
                        report["fixture_seeds"].append(entity)
                        save()
                report["fixture_checks"] = {"keep_unchanged": keep.read_bytes() == b"Unrelated fixture: unchanged.\n",
                                            "notes_absent": not notes.exists(),
                                            "todo_contains_only_seeded_alpha": todo.read_bytes() == b"alpha\n"}
                report["passed"] = len(report["steps"]) == 3 and all(
                    step["passed"] for step in report["steps"]) and all(report["fixture_checks"].values())
                report["status"] = "passed" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), exception_type=type(exc).__name__,
                      traceback=traceback.format_exc())
    finally:
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Fresh JSON report path.")
    parser.add_argument("--base-url", help="Override local QWEN_BASE_URL.")
    parser.add_argument("--api-mode", choices=("ollama", "openai"), default="ollama")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(WORK.parent / ".env", override=False)
    if args.base_url:
        os.environ["QWEN_BASE_URL"] = args.base_url
    os.environ["QWEN_API_MODE"] = args.api_mode
    os.environ["SAMANTHA_AZURE_FALLBACK"] = "false"
    os.environ["SAMANTHA_LOG_REASONING"] = "false"
    try:
        report = run_probe(args.report)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"passed": report["passed"], "status": report["status"],
                      "report": str(args.report.resolve()), "steps_completed": len(report["steps"])},
                     ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
