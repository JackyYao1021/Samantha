"""Opt-in live Ollama test. Run explicitly; offline unit tests do not invoke it.

Real model roles and the real Bash executor run in a temporary directory.
Only a fixed set of file-fixture commands may be approved by this test.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chat import Chat
from clarify_intent import clarify_intent_once
from langgraph.types import Command
from run_commands import run_commands
from workflow import build_workflow, default_services, initial_state


def approve_fixture_commands(commands, directory):
    targets = {"hello.txt", (directory / "hello.txt").as_posix(), str(directory / "hello.txt")}
    allowed = [["pwd"], ["cd", directory.as_posix()], ["cd", str(directory)]]
    for target in targets:
        allowed.extend([
            ["touch", target],
            ["echo", "hello world", ">", target],
            ["echo", "hello", "world", ">", target],
            ["printf", "%s\\n", "hello world", ">", target],
            ["printf", "hello world\\n", ">", target],
            ["cat", target],
        ])
    for command in commands:
        # Punctuation must match the allowlist too; arbitrary shell syntax is rejected.
        tokens = list(shlex.shlex(command, posix=True, punctuation_chars=";&|<>"))
        # shlex's default wordchars split file names, so use split after guarding operators.
        if any(token in {";", "&", "|", "&&", "||", "<<", ">>"} for token in tokens):
            raise AssertionError(f"Fixture test refuses compound commands: {command}")
        if shlex.split(command) not in allowed:
            raise AssertionError(f"Fixture test refuses unexpected commands: {command}")


def drive(graph, request, answer, report):
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 100}
    graph_input = initial_state(request)
    while True:
        pending = None
        for event in graph.stream(graph_input, config, stream_mode="updates"):
            if "__interrupt__" in event:
                pending = event["__interrupt__"][0].value
            else:
                for node, update in event.items():
                    print(f"  {node}: {update.get('status', 'completed')}", flush=True)
                    if update.get("status") == "failed":
                        print(update.get("error", ""), flush=True)
        if pending is None:
            return dict(graph.get_state(config).values)
        report.setdefault("interrupts", []).append(pending)
        if pending["kind"] != "confirmation":
            raise AssertionError(f"Unexpected clarification for a complete test request: {pending}")
        print("  generated commands:", json.dumps(pending["commands"], ensure_ascii=False), flush=True)
        if answer == "yes":
            approve_fixture_commands(pending["commands"], Path.cwd())
        graph_input = Command(resume=answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Override QWEN_BASE_URL for this test process.")
    parser.add_argument("--bash", help="Bash executable (e.g. Git Bash on Windows).")
    parser.add_argument("--report", type=Path, help="Write JSON results to this file.")
    args = parser.parse_args()
    if args.base_url:
        os.environ["QWEN_BASE_URL"] = args.base_url
    if args.bash:
        os.environ["SAMANTHA_BASH"] = args.bash
    os.environ["SAMANTHA_AZURE_FALLBACK"] = "false"

    chat = Chat(max_tokens=256, response_format={"type": "json_object"})
    report = {"endpoint": str(chat.client_qwen.base_url), "model": chat.model_qwen, "passed": False}
    started = time.monotonic()
    original_dir = Path.cwd()
    try:
        print("Testing real model JSON output...", flush=True)
        assert json.loads(chat.chat_context('Return exactly the JSON object {"ok": true}.')) == {"ok": True}
        report["json_output"] = "passed"

        with tempfile.TemporaryDirectory(prefix="ollama-smoke-", dir=ROOT) as temp:
            directory = Path(temp).resolve()
            executed = []

            def executor(commands):
                approve_fixture_commands(commands, directory)
                executed.append(list(commands))
                return run_commands(commands)

            try:
                os.chdir(directory)
                graph = build_workflow(replace(default_services(), execute_commands=executor))
                print("Testing real graph and approved Bash execution...", flush=True)
                state = drive(graph,
                    "Create hello.txt in the current directory containing hello world, then display its content. "
                    "Use relative paths and stay in the starting directory after completion.",
                    "yes", report)
                assert state["status"] == "succeeded", state["error"]
                assert (directory / "hello.txt").read_text(encoding="utf-8").strip() == "hello world"
                assert "hello world" in state["result"]["output"]
                assert state["current_dir"] == str(directory)
                report["file_execution"] = "passed"
                report["executed_commands"] = executed.copy()

                count = len(executed)
                print("Testing rejection with real model-generated commands...", flush=True)
                state = drive(graph, "Create cancelled.txt in the current directory and stay here.", "no", report)
                assert state["status"] == "cancelled"
                assert len(executed) == count
                assert not (directory / "cancelled.txt").exists()
                report["rejection"] = "passed"

                print("Testing real clarification and follow-up...", flush=True)
                response = clarify_intent_once("Create a file.")
                assert response.get("question"), response
                history = [
                    {"role": "user", "content": "Create a file."},
                    {"role": "assistant", "content": response["question"]},
                ]
                response = clarify_intent_once("Name it clarified.txt, in the current directory, and stay here.", history)
                assert "clarified.txt" in response.get("requirement_summary", ""), response
                assert response["is_jump"] is False
                report["clarification"] = "passed"
            finally:
                os.chdir(original_dir)
        report["passed"] = True
        print("Live Ollama smoke tests passed.", flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        print(f"Live test failed: {exc}", file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
