"""Terminal adapter for the LangGraph workflow."""

import json
import os
from pathlib import Path
import sys
import uuid

from langgraph.types import Command

from model_output import parse_commands  # Keep the previous import location available.
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
            output_fn=None, state_file=None):
    """Drive graph interrupts through terminal input and return the final state."""
    input_fn = input_fn or input
    output_fn = output_fn or print
    state = initial_state(initial_input, terminal_history)
    # Initialize the per-call output before any model call or user cancellation.
    write_current_dir(state["initial_dir"], state_file)
    graph = build_workflow(services)
    config = {
        "configurable": {"thread_id": uuid.uuid4().hex},
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
                    continue
                for node, update in event.items():
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
                answer = input_fn("Proceed? (y/n): ")
            else:
                answer = input_fn("User: ")
            graph_input = Command(resume=answer)
    except (EOFError, KeyboardInterrupt):
        state = dict(graph.get_state(config).values)
        state.update(status="cancelled", approved=False, current_dir=state["initial_dir"])
    except Exception as exc:
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
    write_current_dir(state["current_dir"], state_file)
    return state


def main():
    user_command = " ".join(sys.argv[1:]).strip()
    if not user_command:
        write_current_dir(os.getcwd())
        print("Error: empty command provided.\n"
              "Usage: samantha <command>\n"
              "Example: samantha create a file named test.txt")
        return 2
    try:
        state = process(user_command)
    except Exception as exc:
        print(f"Unable to start Samantha: {exc}", file=sys.stderr)
        return 1
    return 1 if state["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
