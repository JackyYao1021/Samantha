"""CLI for durable tasks; no shell execution or implicit approval."""

import argparse
import json
import os
from pathlib import Path

from .runtime import AgentRuntime


WORK = Path(__file__).resolve().parents[2]


def default_db():
    return os.environ.get("SAMANTHA_TASK_DB") or str(WORK / ".samantha" / "tasks.sqlite3")


def show(state):
    print(json.dumps(state, ensure_ascii=False, indent=2))


def interact(runtime, state):
    try:
        while state["status"] in {"ready", "running", "awaiting_approval", "awaiting_input"}:
            print(f"\nTask {state['id']} · {state['status']} · {state['steps']}/{state['max_steps']}")
            if state["status"] == "awaiting_approval":
                action = state["pending_action"]
                print(json.dumps({"tool": action["tool"], "preview": action["preview"],
                                  "args": action["args"]}, ensure_ascii=False, indent=2))
                reply = input("确认这次文件变更？[y/N]: ").strip().lower()
                state = runtime.approve(state["id"], action["id"], reply in {"y", "yes", "确认"})
            elif state["status"] == "awaiting_input":
                state = runtime.answer(state["id"], input(state["question"] + "\n> "))
            if state["status"] not in {"cancelled", "failed", "succeeded"}:
                state = runtime.run(state["id"])
    except (EOFError, KeyboardInterrupt):
        print("\n任务已保存。使用 agent resume <task_id> 继续。")
        state = runtime.get_task(state["id"])
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description="Samantha durable workspace tasks")
    parser.add_argument("--db", default=default_db())
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("run")
    create.add_argument("request")
    create.add_argument("--workspace", default=os.getcwd())
    create.add_argument("--max-steps", type=int, default=12)
    create.add_argument("--max-model-calls", type=int,
                        help="Total physical request cap including intent classification and tool models")
    create.add_argument("--intent-mode", choices=("baseline", "hierarchical"),
                        help="Intent mode for a new task (default: SAMANTHA_INTENT_MODE or baseline)")
    create.add_argument("--parent-id")
    create.add_argument("--interactive", action="store_true")
    sub.add_parser("list")
    for name in ("show", "resume", "cancel", "approve", "answer", "revise"):
        command = sub.add_parser(name)
        command.add_argument("task_id")
        if name == "resume":
            command.add_argument("--interactive", action="store_true")
        elif name == "approve":
            command.add_argument("action_id")
            command.add_argument("--reject", action="store_true")
        elif name in {"answer", "revise"}:
            command.add_argument("text")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--workspace", default=str(WORK / ".samantha" / "workspace"))
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--intent-mode", choices=("baseline", "hierarchical"),
                              help="Default intent mode for new Web tasks; saved tasks keep their mode")
    serve_parser.add_argument("--host", default="127.0.0.1",
                              help="Bind address; use 0.0.0.0 inside Docker with a loopback-only published port")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .web import serve
        workspace = Path(args.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        serve(args.db, str(workspace), host=args.host, port=args.port, intent_mode=args.intent_mode)
        return 0
    runtime = AgentRuntime(args.db, intent_mode=getattr(args, "intent_mode", None))
    try:
        if args.command == "run":
            state = runtime.create_task(args.request, args.workspace, args.max_steps, args.parent_id,
                                        max_model_calls=args.max_model_calls)
            state = runtime.run(state["id"])
        elif args.command == "list":
            show(runtime.list_tasks())
            return 0
        elif args.command == "show":
            state = runtime.get_task(args.task_id)
        elif args.command == "resume":
            state = runtime.run(args.task_id)
        elif args.command == "cancel":
            state = runtime.cancel(args.task_id)
        elif args.command == "approve":
            state = runtime.approve(args.task_id, args.action_id, not args.reject)
            if not args.reject:
                state = runtime.run(args.task_id)
        elif args.command == "revise":
            state = runtime.revise(args.task_id, args.text)
            state = runtime.run(args.task_id)
        else:
            state = runtime.answer(args.task_id, args.text)
            state = runtime.run(args.task_id)
        if getattr(args, "interactive", False):
            state = interact(runtime, state)
        show(state)
        return 1 if state["status"] == "failed" else 0
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
