"""LangGraph orchestration for Samantha's terminal workflow.

Model calls and human interrupts live in separate nodes: resuming an interrupt
restarts its node, so that node must not call a model or execute commands.
"""

import os
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from model_output import parse_commands
from interaction_log import InteractionLogError, record_operation


class ExecutionResult(TypedDict):
    success: bool
    output: str
    current_dir: str


class SamanthaState(TypedDict):
    user_input: str
    user_turn: str
    clarification_history: list[dict[str, str]]
    question: str
    clarified_request: str
    is_jump: bool
    intent: str
    commands: list[str]
    explanation: str
    confirmation_note: str
    approved: bool
    retries: int
    max_retries: int
    initial_dir: str
    current_dir: str
    result: ExecutionResult | None
    error: str
    status: Literal["running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class WorkflowServices:
    """Inject agents/executor to test graph behavior without models or Bash."""

    clarify: Callable[[str, list[dict[str, str]]], dict]
    parse_intent: Callable[[str], str]
    generate_commands: Callable[[str], str]
    explain_commands: Callable[[list[str]], str]
    correct_error: Callable[[str, list[str], str], str]
    execute_commands: Callable[[list[str]], ExecutionResult]


def default_services() -> WorkflowServices:
    from clarify_intent import clarify_intent_once
    from code_confrimation import code_confrim
    from error_correction import error_correction_agent
    from intent_explain import parse_user_intent
    from n2c import natural_language_to_command_agent
    from run_commands import run_commands

    return WorkflowServices(
        clarify_intent_once,
        parse_user_intent,
        natural_language_to_command_agent,
        code_confrim,
        error_correction_agent,
        run_commands,
    )


def initial_state(user_input: str, terminal_history=None, max_retries: int = 3) -> SamanthaState:
    if not user_input.strip():
        raise ValueError("A non-empty request is required.")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    current_dir = os.getcwd()
    return {
        "user_input": user_input,
        "user_turn": user_input,
        "clarification_history": list(terminal_history or []),
        "question": "",
        "clarified_request": "",
        "is_jump": False,
        "intent": "",
        "commands": [],
        "explanation": "",
        "confirmation_note": "",
        "approved": False,
        "retries": 0,
        "max_retries": max_retries,
        "initial_dir": current_dir,
        "current_dir": current_dir,
        "result": None,
        "error": "",
        "status": "running",
    }


def _failure(stage: str, exc: Exception) -> dict:
    if isinstance(exc, InteractionLogError):
        raise exc
    return {"status": "failed", "approved": False, "error": f"{stage}: {exc}"}


def _require_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The agent returned an empty response.")
    return text


def build_workflow(services: WorkflowServices | None = None, checkpointer=None):
    services = services or default_services()

    def clarify(state: SamanthaState):
        try:
            response = record_operation("agent.clarify", services.clarify,
                                        state["user_turn"], state["clarification_history"])
            if response.get("cancelled"):
                return {"status": "cancelled"}
            if response.get("question"):
                question = _require_text(response["question"])
                history = state["clarification_history"] + [
                    {"role": "user", "content": state["user_turn"]},
                    {"role": "assistant", "content": question},
                ]
                return {"question": question, "clarification_history": history}
            if type(response.get("is_jump")) is not bool:
                raise ValueError("is_jump must be a boolean.")
            return {
                "question": "",
                "is_jump": response["is_jump"],
                "clarified_request": _require_text(response["requirement_summary"]),
            }
        except Exception as exc:
            return _failure("Request clarification failed", exc)

    def route_clarification(state: SamanthaState):
        if state["status"] != "running":
            return END
        return "ask_clarification" if state["question"] else "parse_intent"

    def ask_clarification(state: SamanthaState):
        answer = interrupt({"kind": "clarification", "message": state["question"]})
        if not isinstance(answer, str):
            return _failure("Invalid clarification", ValueError("A text answer is required."))
        if answer.strip().lower() in {"stop", "quit", "exit", "cancel", "取消", "退出", "停止"}:
            return {"status": "cancelled"}
        return {"user_turn": answer.strip()}

    def parse_intent(state: SamanthaState):
        try:
            return {"intent": _require_text(record_operation(
                "agent.parse_intent", services.parse_intent, state["clarified_request"]))}
        except Exception as exc:
            return _failure("Intent parsing failed", exc)

    def generate_commands(state: SamanthaState):
        try:
            commands, explanation = parse_commands(record_operation(
                "agent.generate_commands", services.generate_commands, state["intent"]))
            return {
                "commands": commands,
                "explanation": explanation,
                "approved": False,
                "confirmation_note": "",
            }
        except Exception as exc:
            return _failure("Command generation failed", exc)

    def explain_commands(state: SamanthaState):
        try:
            return {"confirmation_note": _require_text(record_operation(
                "agent.explain_commands", services.explain_commands, state["commands"]))}
        except Exception as exc:
            return _failure("Command explanation failed", exc)

    def confirm(state: SamanthaState):
        answer = interrupt({
            "kind": "confirmation",
            "message": state["confirmation_note"],
            "commands": state["commands"],
            "retries": state["retries"],
        })
        approved = answer is True or (
            isinstance(answer, str) and answer.strip().lower() in {"y", "yes", "sure", "go ahead"}
        )
        return {"approved": approved, "status": "running" if approved else "cancelled"}

    def execute(state: SamanthaState):
        if not state["approved"]:
            return _failure("Execution blocked", ValueError("User confirmation is required."))
        try:
            result = record_operation("commands.execute", services.execute_commands, state["commands"])
            if (type(result.get("success")) is not bool
                    or not isinstance(result.get("output"), str)
                    or not isinstance(result.get("current_dir"), str)
                    or not result["current_dir"]):
                raise ValueError("The executor returned an invalid result.")
            success = result["success"]
            return {
                "result": result,
                "current_dir": result["current_dir"] if success and state["is_jump"] else state["initial_dir"],
                "approved": False,
                "error": "" if success else result["output"],
                "status": "succeeded" if success else (
                    "running" if state["retries"] < state["max_retries"] else "failed"
                ),
            }
        except Exception as exc:
            return _failure("Command execution failed", exc)

    def correct_error(state: SamanthaState):
        try:
            intent = record_operation("agent.correct_error", services.correct_error,
                state["clarified_request"], state["commands"], state["error"]
            )
            return {"intent": _require_text(intent), "retries": state["retries"] + 1}
        except Exception as exc:
            return _failure("Error correction failed", exc)

    def continue_to(node: str):
        return lambda state: node if state["status"] == "running" else END

    builder = StateGraph(SamanthaState)
    for name, node in {
        "clarify": clarify,
        "ask_clarification": ask_clarification,
        "parse_intent": parse_intent,
        "generate_commands": generate_commands,
        "explain_commands": explain_commands,
        "confirm": confirm,
        "execute": execute,
        "correct_error": correct_error,
    }.items():
        builder.add_node(name, node)

    builder.add_edge(START, "clarify")
    builder.add_conditional_edges("clarify", route_clarification, ["ask_clarification", "parse_intent", END])
    for source, target in [
        ("ask_clarification", "clarify"),
        ("parse_intent", "generate_commands"),
        ("generate_commands", "explain_commands"),
        ("explain_commands", "confirm"),
        ("confirm", "execute"),
        ("execute", "correct_error"),
        ("correct_error", "generate_commands"),
    ]:
        builder.add_conditional_edges(source, continue_to(target), [target, END])
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
