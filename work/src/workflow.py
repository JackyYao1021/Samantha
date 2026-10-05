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

from clarification_context import (
    CANCEL_ANSWERS, MAX_CLARIFICATIONS, MAX_EMPTY_ANSWERS,
    build_clarification_input, check_clarification_question,
)
from model_output import parse_commands
from command_review import (
    CommandReview, ReviewRequest, command_action_id, format_review, validate_review,
)
from interaction_log import InteractionLogError, record_operation


class ExecutionResult(TypedDict):
    success: bool
    output: str
    current_dir: str


class SamanthaState(TypedDict):
    user_input: str
    user_turn: str
    clarification_history: list[dict[str, str]]
    clarification_answers: list[dict[str, str]]
    clarification_count: int
    max_clarifications: int
    question: str
    clarified_request: str
    is_jump: bool
    intent: str
    commands: list[str]
    explanation: str
    confirmation_note: str
    command_review: CommandReview | None
    reviewed_action_id: str
    approved_action_id: str
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
    review_commands: Callable[[ReviewRequest], CommandReview]
    correct_error: Callable[[str, list[str], str], str]
    execute_commands: Callable[[list[str]], ExecutionResult]


def default_services() -> WorkflowServices:
    from clarify_intent import clarify_intent_once
    from command_review import review_commands
    from error_correction import error_correction_agent
    from intent_explain import parse_user_intent
    from n2c import natural_language_to_command_agent
    from run_commands import run_commands

    return WorkflowServices(
        clarify_intent_once,
        parse_user_intent,
        natural_language_to_command_agent,
        review_commands,
        error_correction_agent,
        run_commands,
    )


def initial_state(user_input: str, terminal_history=None, max_retries: int = 3,
                  max_clarifications: int = MAX_CLARIFICATIONS) -> SamanthaState:
    if not user_input.strip():
        raise ValueError("A non-empty request is required.")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if type(max_clarifications) is not int or max_clarifications < 0:
        raise ValueError("max_clarifications must be a non-negative integer.")
    current_dir = os.getcwd()
    return {
        "user_input": user_input,
        "user_turn": user_input,
        "clarification_history": list(terminal_history or []),
        "clarification_answers": [],
        "clarification_count": 0,
        "max_clarifications": max_clarifications,
        "question": "",
        "clarified_request": "",
        "is_jump": False,
        "intent": "",
        "commands": [],
        "explanation": "",
        "confirmation_note": "",
        "command_review": None,
        "reviewed_action_id": "",
        "approved_action_id": "",
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
    return {"status": "failed", "approved": False, "approved_action_id": "",
            "error": f"{stage}: {exc}"}


def _require_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The agent returned an empty response.")
    return text


def build_workflow(services: WorkflowServices | None = None, checkpointer=None):
    services = services or default_services()

    def clarify(state: SamanthaState):
        try:
            prompt = build_clarification_input(
                state["user_input"], state["question"], state["user_turn"],
                state["clarification_answers"])
            response = record_operation("agent.clarify", services.clarify,
                                        prompt, state["clarification_history"])
            if response.get("cancelled"):
                return {"status": "cancelled"}
            if response.get("question"):
                question = _require_text(response["question"])
                check_clarification_question(question, state["question"],
                                             state["clarification_count"], state["max_clarifications"])
                history = state["clarification_history"] + [
                    {"role": "user", "content": state["user_turn"]},
                    {"role": "assistant", "content": question},
                ]
                return {"question": question, "clarification_history": history,
                        "clarification_count": state["clarification_count"] + 1}
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
        message = state["question"]
        # Resume replays this node and its prior interrupt answers. Keep model
        # calls and history updates outside this loop.
        for _ in range(MAX_EMPTY_ANSWERS):
            answer = interrupt({"kind": "clarification", "message": message})
            if not isinstance(answer, str):
                return _failure("Invalid clarification", ValueError("A text answer is required."))
            answer = answer.strip()
            if answer.lower() in CANCEL_ANSWERS:
                return {"status": "cancelled"}
            if answer:
                return {"user_turn": answer, "clarification_answers": state["clarification_answers"] + [
                    {"question": state["question"], "answer": answer},
                ]}
            message = "Please provide a non-empty answer, or type 'cancel' to stop.\n" + state["question"]
        return _failure("Invalid clarification", ValueError(
            f"No non-empty answer received after {MAX_EMPTY_ANSWERS} attempts."))

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
                "command_review": None,
                "reviewed_action_id": "",
                "approved_action_id": "",
            }
        except Exception as exc:
            return _failure("Command generation failed", exc)

    def review_commands(state: SamanthaState):
        try:
            shell = os.environ.get("SAMANTHA_BASH", "/bin/bash")
            request: ReviewRequest = {
                "user_request": state["user_input"],
                "clarified_request": state["clarified_request"],
                "plan": state["intent"], "commands": list(state["commands"]),
                "cwd": state["initial_dir"], "shell": shell,
                "previous_error": state["error"],
            }
            review = validate_review(record_operation(
                "agent.review_commands", services.review_commands, request))
            return {
                "command_review": review,
                "confirmation_note": format_review(review, state["user_input"]),
                "reviewed_action_id": command_action_id(
                    state["commands"], state["initial_dir"], shell),
                "approved": False, "approved_action_id": "",
            }
        except Exception as exc:
            return _failure("Command review failed", exc)

    def current_action_id(state: SamanthaState):
        return command_action_id(state["commands"], os.getcwd(),
                                 os.environ.get("SAMANTHA_BASH", "/bin/bash"))

    def confirm(state: SamanthaState):
        if not state["command_review"] or state["reviewed_action_id"] != current_action_id(state):
            return _failure("Confirmation blocked", ValueError("The reviewed action has changed."))
        answer = interrupt({
            "kind": "confirmation",
            "message": state["confirmation_note"],
            "commands": state["commands"],
            "retries": state["retries"],
            "review": state["command_review"],
            "action_id": state["reviewed_action_id"],
            "cwd": state["initial_dir"],
            "shell": os.environ.get("SAMANTHA_BASH", "/bin/bash"),
        })
        approved = answer is True or (
            isinstance(answer, str) and answer.strip().lower() in {
                "y", "yes", "sure", "go ahead", "是", "确认", "执行", "同意",
            }
        )
        return {"approved": approved,
                "approved_action_id": state["reviewed_action_id"] if approved else "",
                "status": "running" if approved else "cancelled"}

    def execute(state: SamanthaState):
        if not state["approved"]:
            return _failure("Execution blocked", ValueError("User confirmation is required."))
        if (not state["command_review"] or not state["approved_action_id"]
                or state["approved_action_id"] != state["reviewed_action_id"]
                or state["approved_action_id"] != current_action_id(state)):
            return _failure("Execution blocked", ValueError("The approved action has changed."))
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
                "approved_action_id": "",
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
        "review_commands": review_commands,
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
        ("generate_commands", "review_commands"),
        ("review_commands", "confirm"),
        ("confirm", "execute"),
        ("execute", "correct_error"),
        ("correct_error", "generate_commands"),
    ]:
        builder.add_conditional_edges(source, continue_to(target), [target, END])
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
