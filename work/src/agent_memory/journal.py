"""Project committed journal evidence into memory without inventing completed work."""

import json

from .core import MemoryScope, MemorySession


def journal_session(data: dict, scope: MemoryScope) -> MemorySession:
    messages = []
    submitted = {}
    for event in data["events"]:
        payload = event["data"]
        if event["kind"] == "user_input":
            messages.append({"role": "user", "content": json.dumps(
                {"source": payload["source"], "text": payload["text"]}, ensure_ascii=False)})
        elif event["kind"] == "awaiting_input" and payload.get("kind") == "clarification":
            messages.append({"role": "assistant", "content": payload["message"]})
        elif event["kind"] == "operation_started" and payload.get("name") == "commands.execute":
            submitted[payload["operation_id"]] = payload.get("arguments", {}).get("args", [])
        elif event["kind"] == "operation_finished" and payload.get("name") == "commands.execute":
            arguments = submitted.get(payload["operation_id"], [])
            messages.append({"role": "tool", "content": json.dumps(
                {"operation": "commands.execute", "status": payload["status"],
                 "operation_id": payload["operation_id"],
                 "submitted_commands": arguments[0] if arguments else None,
                 "result": payload.get("result"), "error": payload.get("error")}, ensure_ascii=False)})
        elif event["kind"] == "session_finished":
            state = payload.get("state", {})
            messages.append({"role": "tool", "content": json.dumps({
                "operation": "workflow_outcome", "status": payload["status"],
                "error": payload["error"], "clarified_request": state.get("clarified_request"),
                "planned_commands": state.get("commands", []),
                "note": "Planned commands are not evidence of execution. Shell results cover the entire batch.",
            }, ensure_ascii=False)})
    header = data["session"]
    return MemorySession(header["session_id"], scope, header["ended_at"] or header["started_at"],
                         tuple(messages), {"kind": header["kind"], "status": header["status"],
                                           "initial_dir": header["initial_dir"]})
