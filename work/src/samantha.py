"""Terminal adapter for the LangGraph workflow."""

import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from langgraph.types import Command

from model_output import parse_commands  # Keep the previous import location available.
from interaction_log import InteractionJournal, InteractionLogError, record_operation
from workflow import build_workflow, initial_state


PROGRESS = {
    "clarify": "Request clarity checked.",
    "parse_intent": "Request understood.",
    "generate_commands": "Commands generated.",
    "execute": "Approved commands executed.",
    "correct_error": "Error correction prepared.",
}


def write_current_dir(current_dir, path=None):
    target = path or os.environ.get("SAMANTHA_STATE_FILE")
    if target:
        Path(target).write_text(json.dumps({"current_dir": current_dir}), encoding="utf-8")


def process(initial_input, terminal_history=None, *, services=None, input_fn=None,
            output_fn=None, state_file=None, log_file=None):
    """Persist every turn and operation while driving terminal interrupts."""
    with InteractionJournal(log_file) as journal:
        with journal.start_session(initial_input, metadata={
            "terminal_history": list(terminal_history or []),
        }) as session:
            state = _process(initial_input, terminal_history, services=services, input_fn=input_fn,
                             output_fn=output_fn, state_file=state_file, session=session)
            session.finish(state["status"], error=state["error"], details={"state": state})
            state["log_session_id"] = session.session_id
            return state


def _process(initial_input, terminal_history, *, services, input_fn, output_fn, state_file, session):
    input_fn = input_fn or input
    terminal_output = output_fn or print

    def output_fn(text):
        session.output(text + "\n")
        terminal_output(text)

    def read_answer(prompt, source):
        session.output(prompt, prompt=True)
        answer = input_fn(prompt)
        session.user_input(answer, source)
        return answer

    state = initial_state(initial_input, terminal_history)
    # Initialize the per-call output before any model call or user cancellation.
    record_operation("terminal.write_current_dir", write_current_dir, state["initial_dir"], state_file)
    graph = build_workflow(services)
    config = {
        "configurable": {"thread_id": session.session_id},
        "recursion_limit": 100,
    }
    graph_input = state
    output_fn("Checking the clarity of your request...")

    try:
        while True:
            pending = None
            for event in graph.stream(graph_input, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    pending = event["__interrupt__"][0].value
                    session.record("awaiting_input", pending)
                    continue
                for node, update in event.items():
                    session.record("workflow_update", {"node": node, "update": update})
                    if node in PROGRESS and update.get("status") != "failed":
                        output_fn(PROGRESS[node])
                    if node == "execute" and update.get("error"):
                        output_fn(f"Error Message:\n{update['error']}")
            if pending is None:
                state = dict(graph.get_state(config).values)
                break
            output_fn(pending["message"].strip().strip("`").strip())
            if pending["kind"] == "confirmation":
                output_fn("-----Commands-----")
                for command in pending["commands"]:
                    output_fn(command)
                answer = read_answer("Proceed? (y/n): ", "confirmation")
            else:
                answer = read_answer("User: ", "clarification")
            graph_input = Command(resume=answer)
    except InteractionLogError:
        raise
    except (EOFError, KeyboardInterrupt) as exc:
        session.record("interrupted", {"exception_type": type(exc).__name__})
        state = dict(graph.get_state(config).values) or state
        state.update(status="cancelled", approved=False, current_dir=state["initial_dir"])
    except Exception as exc:
        session.record("workflow_error", {"error": str(exc), "exception_type": type(exc).__name__})
        state = dict(graph.get_state(config).values) or state
        state.update(status="failed", approved=False, error=str(exc), current_dir=state["initial_dir"])

    if state["status"] == "succeeded":
        output_fn("Commands executed successfully ^-^")
    elif state["status"] == "cancelled":
        output_fn("Command execution cancelled by user.")
    else:
        output_fn(f"Command execution unsuccessful.\nError Message:\n{state['error']}")
    if state["result"]:
        output_fn(f"-----Output-----\n{state['result']['output']}")
    record_operation("terminal.write_current_dir", write_current_dir, state["current_dir"], state_file)
    return state


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        from history_cli import main as history_main
        write_current_dir(os.getcwd())
        return history_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "content":
        from content_cli import main as content_main
        write_current_dir(os.getcwd())
        return content_main(sys.argv[2:])
    user_command = " ".join(sys.argv[1:])
    if not user_command.strip():
        write_current_dir(os.getcwd())
        message = ("Error: empty command provided.\n"
                   "Usage: samantha <command> | content ... | history ...\n"
                   "Example: samantha create a file named test.txt")
        try:
            with InteractionJournal() as journal, journal.start_session(user_command, kind="usage") as session:
                session.output(message + "\n")
                print(message)
                session.finish("failed", error="Empty command provided.")
        except InteractionLogError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 2
    try:
        state = process(user_command)
    except Exception as exc:
        print(f"Unable to start Samantha: {exc}", file=sys.stderr)
        return 1
    return 1 if state["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
