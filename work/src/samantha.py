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
from agent_memory import MemoryScope, create_backend
from agent_memory.journal import journal_session
from object_tracking.store import ObjectStore
from object_tracking.runtime import ObjectTracker


PROGRESS = {
    "clarify": "Request clarity checked.",
    "parse_intent": "Request understood.",
    "generate_commands": "Commands generated.",
    "review_commands": "Independent command review completed.",
    "execute": "Approved commands executed.",
    "correct_error": "Error correction prepared.",
}


def write_current_dir(current_dir, path=None):
    target = path or os.environ.get("SAMANTHA_STATE_FILE")
    if target:
        Path(target).write_text(json.dumps({"current_dir": current_dir}), encoding="utf-8")


def process(initial_input, terminal_history=None, *, services=None, input_fn=None,
            output_fn=None, state_file=None, log_file=None, memory=None, memory_scope=None,
            memory_budget=6000, memory_limit=5, object_store=None, object_scope=None):
    """Persist every turn and operation while driving terminal interrupts."""
    tracker = None
    if object_store is not None:
        scope = object_scope or memory_scope
        if scope is None:
            raise ValueError("An explicit user/project scope is required for object tracking.")
        tracker = ObjectTracker(object_store, scope)
    with InteractionJournal(log_file) as journal:
        with journal.start_session(initial_input, metadata={
            "terminal_history": list(terminal_history or []),
            **({"memory": memory.describe(), "memory_scope": {
                "user_id": memory_scope.user_id, "project_id": memory_scope.project_id,
            }} if memory is not None and memory_scope is not None else {}),
            **({"object_tracking": tracker.describe()} if tracker is not None else {}),
        }) as session:
            state = _process(initial_input, terminal_history, services=services, input_fn=input_fn,
                             output_fn=output_fn, state_file=state_file, session=session,
                             memory=memory, memory_scope=memory_scope,
                             memory_budget=memory_budget, memory_limit=memory_limit,
                             object_tracker=tracker)
            session.finish(state["status"], error=state["error"], details={"state": state})
            state["log_session_id"] = session.session_id
            if memory is not None:
                try:
                    evidence = journal_session(journal.read_session(session.session_id), memory_scope)
                    record_operation("memory.ingest", memory.ingest, evidence)
                except InteractionLogError:
                    raise
                except Exception as exc:
                    # Execution already finished. Report failed memory construction
                    # separately; never relabel or replay a completed shell task.
                    state["memory_error"] = str(exc)
                    session.record("memory_error", {"error": str(exc)})
                    message = f"Task finished, but memory could not be saved: {exc}"
                    session.output(message + "\n")
                    (output_fn or print)(message)
            return state


def _process(initial_input, terminal_history, *, services, input_fn, output_fn, state_file, session,
             memory=None, memory_scope=None, memory_budget=6000, memory_limit=5,
             object_tracker=None):
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
    graph = build_workflow(services, memory=memory, memory_scope=memory_scope,
                           memory_budget=memory_budget, memory_limit=memory_limit,
                           object_tracker=object_tracker)
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
                output_fn(f"Working directory: {pending['cwd']}")
                output_fn(f"Shell: {pending['shell']}")
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

    if object_tracker is not None and (
            state["status"] == "cancelled" or
            (state["result"] is not None and not state["object_effects"]) or
            (state.get("tracking_error") and not state["tracked_entities"]) or
            (state["status"] == "failed" and not state["tracked_entities"])):
        try:
            record_operation("objects.clear_focus", object_tracker.clear_focus, state["object_context"])
        except InteractionLogError:
            raise
        except Exception as exc:
            state["tracking_error"] = (state.get("tracking_error", "") + "; " + str(exc)).strip("; ")

    if state["status"] == "succeeded":
        output_fn("Commands executed successfully ^-^")
    elif state["status"] == "cancelled":
        output_fn("Command execution cancelled by user.")
    else:
        output_fn(f"Command execution unsuccessful.\nError Message:\n{state['error']}")
    if state["result"]:
        output_fn(f"-----Output-----\n{state['result']['output']}")
    if state.get("tracking_error"):
        output_fn("Task finished, but object state could not be fully saved or verified: " + state["tracking_error"])
    record_operation("terminal.write_current_dir", write_current_dir, state["current_dir"], state_file)
    return state


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        from history_cli import main as history_main
        write_current_dir(os.getcwd())
        return history_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "entities":
        from entities_cli import main as entities_main
        write_current_dir(os.getcwd())
        return entities_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "content":
        from content_cli import main as content_main
        write_current_dir(os.getcwd())
        return content_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "memory-eval":
        from memory_eval import main as memory_eval_main
        write_current_dir(os.getcwd())
        return memory_eval_main(sys.argv[2:])
    user_command = " ".join(sys.argv[1:])
    if not user_command.strip():
        write_current_dir(os.getcwd())
        message = ("Error: empty command provided.\n"
                   "Usage: samantha <command> | content ... | history ... | entities ...\n"
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
        backend_name = os.environ.get("SAMANTHA_MEMORY_BACKEND", "none")
        tracking_enabled = os.environ.get("SAMANTHA_OBJECT_TRACKING", "false").lower() in {"1", "true", "yes"}
        if backend_name == "none" and not tracking_enabled:
            state = process(user_command)
        else:
            memory = None
            objects = None
            try:
                if backend_name != "none":
                    memory = create_backend(backend_name, path=os.environ.get("SAMANTHA_MEMORY_PATH") or
                                            Path(__file__).resolve().parents[1] / ".samantha" / "memory.sqlite3")
                object_scope = None
                if tracking_enabled:
                    project = os.environ.get("SAMANTHA_OBJECT_PROJECT") or os.environ.get("SAMANTHA_MEMORY_PROJECT")
                    if not project:
                        raise ValueError("Set a stable SAMANTHA_OBJECT_PROJECT (or SAMANTHA_MEMORY_PROJECT) "
                                         "before enabling object tracking; directory changes must keep the same scope.")
                    object_scope = MemoryScope(os.environ.get("SAMANTHA_OBJECT_USER") or
                                               os.environ.get("SAMANTHA_MEMORY_USER", "local"), project)
                    objects = ObjectStore(os.environ.get("SAMANTHA_OBJECT_PATH") or
                                          Path(__file__).resolve().parents[1] / ".samantha" / "objects.sqlite3")
                state = process(user_command, memory=memory, memory_scope=MemoryScope(
                    os.environ.get("SAMANTHA_MEMORY_USER", "local"),
                    os.environ.get("SAMANTHA_MEMORY_PROJECT", str(Path.cwd().resolve()))),
                    memory_budget=int(os.environ.get("SAMANTHA_MEMORY_BUDGET", "6000")),
                    memory_limit=int(os.environ.get("SAMANTHA_MEMORY_LIMIT", "5")),
                    object_store=objects, object_scope=object_scope)
            finally:
                if objects is not None:
                    objects.close()
                if memory is not None:
                    memory.close()
    except Exception as exc:
        print(f"Unable to start Samantha: {exc}", file=sys.stderr)
        return 1
    return 1 if state["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
