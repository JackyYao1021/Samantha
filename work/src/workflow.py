"""LangGraph orchestration for Samantha's terminal workflow.

Model calls and human interrupts live in separate nodes: resuming an interrupt
restarts its node, so that node must not call a model or execute commands.
"""

import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from clarification_context import (
    CANCEL_ANSWERS, MAX_CLARIFICATIONS, MAX_EMPTY_ANSWERS,
    build_clarification_input, check_clarification_question,
)
from model_output import parse_commands, parse_json_object
from command_review import (
    CommandReview, ReviewRequest, command_action_id, format_review, validate_review,
)
from interaction_log import InteractionLogError, record_operation, operation, current_operation_reference
from agent_memory import MemoryBackend, MemoryQuery, MemoryScope, pack_hits, with_memory
from object_tracking.runtime import ObjectTracker, with_objects, tracking_action_id, format_effects
from object_tracking.questions import creation_answer_order, creation_question_order, is_first_use_question


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
    memory_context: str
    memory_sources: list[dict]
    object_context: str
    object_effects: list[dict]
    object_before: list[dict]
    object_verifications: list[dict]
    tracked_entities: list[dict]
    reviewed_tracking_id: str
    approved_tracking_id: str
    tracking_error: str
    answer: str


@dataclass(frozen=True)
class WorkflowServices:
    """Inject agents/executor to test graph behavior without models or Bash."""

    clarify: Callable[[str, list[dict[str, str]]], dict]
    parse_intent: Callable[[str], str]
    generate_commands: Callable[[str], str]
    review_commands: Callable[[ReviewRequest], CommandReview]
    correct_error: Callable[[str, list[str], str], str]
    execute_commands: Callable[[list[str]], ExecutionResult]


def default_services(*, object_tracking=False) -> WorkflowServices:
    from clarify_intent import clarify_intent_once
    from command_review import review_commands
    from error_correction import error_correction_agent
    from intent_explain import parse_user_intent
    from n2c import natural_language_to_command_agent
    from run_commands import run_commands

    return WorkflowServices(
        clarify_intent_once,
        (partial(parse_user_intent, object_tracking=True) if object_tracking else parse_user_intent),
        (partial(natural_language_to_command_agent, object_tracking=True)
         if object_tracking else natural_language_to_command_agent),
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
        "memory_context": "",
        "memory_sources": [],
        "object_context": "",
        "object_effects": [],
        "object_before": [],
        "object_verifications": [],
        "tracked_entities": [],
        "reviewed_tracking_id": "",
        "approved_tracking_id": "",
        "tracking_error": "",
        "answer": "",
    }


def _failure(stage: str, exc: Exception) -> dict:
    if isinstance(exc, InteractionLogError):
        raise exc
    return {"status": "failed", "approved": False, "approved_action_id": "", "approved_tracking_id": "",
            "error": f"{stage}: {exc}"}


def _require_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The agent returned an empty response.")
    return text


def build_workflow(services: WorkflowServices | None = None, checkpointer=None, *,
                   memory: MemoryBackend | None = None, memory_scope: MemoryScope | None = None,
                   memory_budget: int = 6000, memory_limit: int = 5,
                   object_tracker: ObjectTracker | None = None):
    services = services or default_services(object_tracking=object_tracker is not None)
    if memory is not None and memory_scope is None:
        raise ValueError("A memory scope is required when enabling memory.")

    def contextual_input(state, text):
        prompt = with_memory(text, state.get("memory_context", ""))
        return with_objects(prompt, state["object_context"]) if object_tracker is not None else prompt

    def load_objects(state: SamanthaState):
        try:
            return {"object_context": record_operation("objects.context", object_tracker.context)}
        except Exception as exc:
            return _failure("Object state loading failed", exc)

    def answer_file_question(state: SamanthaState):
        order = creation_question_order(state["user_input"])
        if order is None and is_first_use_question(state["user_input"]):
            question = ("你是指最先创建的文件吗？我可以查询已验证的创建记录；"
                        "如果指其他操作中的文件，请说明操作或文件路径。"
                        if any("\u3400" <= char <= "\u9fff" for char in state["user_input"]) else
                        "Do you mean the file you created first? I can look up verified creation "
                        "records; otherwise, please specify the operation or file path.")
            if state["clarification_answers"]:
                reply = state["clarification_answers"][-1]
                order = creation_answer_order(reply["answer"])
                if reply["question"] == question and reply["answer"].strip().lower() in {
                        "yes", "yes please", "是", "是的", "对", "对的"}:
                    order = "first"
            else:
                try:
                    check_clarification_question(question, state["question"],
                                                 state["clarification_count"], state["max_clarifications"])
                except Exception as exc:
                    return _failure("Request clarification failed", exc)
                history = state["clarification_history"] + [
                    {"role": "user", "content": state["user_turn"]},
                    {"role": "assistant", "content": question},
                ]
                return {"question": question, "clarification_history": history,
                        "clarification_count": state["clarification_count"] + 1}
        if order is None:
            return {}
        try:
            if object_tracker is not None:
                name, lookup = (("objects.answer_first_creation", object_tracker.answer_first_creation)
                                if order == "first" else
                                ("objects.answer_recent_creation", object_tracker.answer_recent_creation))
                answer = record_operation(name, lookup, state["user_input"])
            else:
                answer = ("文件对象跟踪未启用，无法根据已验证记录确定你创建了哪个文件。"
                          if any("\u3400" <= char <= "\u9fff" for char in state["user_input"]) else
                          "File object tracking is disabled, so I cannot identify the file you "
                          "created from verified records.")
            return {"answer": answer, "question": "", "status": "succeeded"}
        except Exception as exc:
            return _failure("File history lookup failed", exc)

    def route_file_question(state: SamanthaState):
        if state["status"] != "running":
            return END
        if (is_first_use_question(state["user_input"]) and state["question"]
                and not state["clarification_answers"]):
            return "ask_clarification"
        return "clarify"

    def route_clarification_answer(state: SamanthaState):
        if state["status"] != "running":
            return END
        return "answer_file_question" if is_first_use_question(state["user_input"]) else "clarify"

    def retrieve_memory(state: SamanthaState):
        try:
            query = MemoryQuery(memory_scope, state["user_input"],
                                datetime.now(timezone.utc).isoformat(), memory_limit, memory_budget)
            hits = record_operation("memory.retrieve", memory.retrieve, query)
            context, sources = pack_hits(hits, query)
            return {"memory_context": context, "memory_sources": sources}
        except Exception as exc:
            return _failure("Memory retrieval failed", exc)

    def clarify(state: SamanthaState):
        try:
            prompt = build_clarification_input(
                state["user_input"], state["question"], state["user_turn"],
                state["clarification_answers"])
            prompt = contextual_input(state, prompt)
            reference_question = (record_operation("objects.resolve_reference", object_tracker.reference_question,
                                                  state["user_input"], state["object_context"])
                                  if object_tracker is not None and not state["clarification_answers"] else "")
            response = ({"question": reference_question} if reference_question else
                        record_operation("agent.clarify", services.clarify,
                                         prompt, state["clarification_history"]))
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
                "agent.parse_intent", services.parse_intent,
                contextual_input(state, state["clarified_request"])))}
        except Exception as exc:
            return _failure("Intent parsing failed", exc)

    def generate_commands(state: SamanthaState):
        try:
            generation_input = state["intent"]
            if object_tracker is not None:
                generation_input = ("Proposed plan:\n" + generation_input +
                                    "\n\nOriginal user request (authoritative):\n" + state["user_input"] +
                                    "\nClarification answers:\n" + json.dumps(
                                        state["clarification_answers"], ensure_ascii=False))
            response = record_operation(
                "agent.generate_commands", services.generate_commands,
                contextual_input(state, generation_input))
            commands, explanation = parse_commands(response)
            prepared = (record_operation("objects.prepare", object_tracker.prepare,
                                         parse_json_object(response), commands, state["initial_dir"])
                        if object_tracker is not None else {"effects": [], "before": []})
            return {
                "commands": commands,
                "explanation": explanation,
                "approved": False,
                "confirmation_note": "",
                "command_review": None,
                "reviewed_action_id": "",
                "approved_action_id": "",
                "object_effects": prepared["effects"], "object_before": prepared["before"],
                "reviewed_tracking_id": "", "approved_tracking_id": "",
                "object_verifications": [], "tracked_entities": [], "tracking_error": "",
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
            if state.get("memory_context"):
                request["memory_context"] = state["memory_context"]
            if object_tracker is not None:
                request["object_context"] = state["object_context"]
                request["expected_effects"] = state["object_effects"]
            review = validate_review(record_operation(
                "agent.review_commands", services.review_commands, request))
            return {
                "command_review": review,
                "confirmation_note": (format_review(review, state["user_input"]) +
                                      format_effects(state["object_effects"], state["user_input"])),
                "reviewed_action_id": command_action_id(
                    state["commands"], state["initial_dir"], shell),
                "approved": False, "approved_action_id": "",
                "reviewed_tracking_id": (tracking_action_id(
                    state["commands"], state["object_effects"], state["object_before"],
                    state["initial_dir"], shell) if object_tracker is not None else ""),
                "approved_tracking_id": "",
            }
        except Exception as exc:
            return _failure("Command review failed", exc)

    def current_action_id(state: SamanthaState):
        return command_action_id(state["commands"], os.getcwd(),
                                 os.environ.get("SAMANTHA_BASH", "/bin/bash"))

    def current_tracking_id(state: SamanthaState):
        return tracking_action_id(state["commands"], state["object_effects"], state["object_before"],
                                  os.getcwd(), os.environ.get("SAMANTHA_BASH", "/bin/bash"))

    def confirm(state: SamanthaState):
        if not state["command_review"] or state["reviewed_action_id"] != current_action_id(state):
            return _failure("Confirmation blocked", ValueError("The reviewed action has changed."))
        if object_tracker is not None and state["reviewed_tracking_id"] != current_tracking_id(state):
            return _failure("Confirmation blocked", ValueError("The reviewed file effects have changed."))
        answer = interrupt({
            "kind": "confirmation",
            "message": state["confirmation_note"],
            "commands": state["commands"],
            "retries": state["retries"],
            "review": state["command_review"],
            "action_id": state["reviewed_action_id"],
            "cwd": state["initial_dir"],
            "shell": os.environ.get("SAMANTHA_BASH", "/bin/bash"),
            **({"expected_effects": state["object_effects"],
                "tracking_action_id": state["reviewed_tracking_id"]} if object_tracker is not None else {}),
        })
        approved = answer is True or (
            isinstance(answer, str) and answer.strip().lower() in {
                "y", "yes", "sure", "go ahead", "是", "确认", "执行", "同意",
            }
        )
        return {"approved": approved,
                "approved_action_id": state["reviewed_action_id"] if approved else "",
                "approved_tracking_id": state["reviewed_tracking_id"] if approved else "",
                "status": "running" if approved else "cancelled"}

    def execute(state: SamanthaState):
        if not state["approved"]:
            return _failure("Execution blocked", ValueError("User confirmation is required."))
        if (not state["command_review"] or not state["approved_action_id"]
                or state["approved_action_id"] != state["reviewed_action_id"]
                or state["approved_action_id"] != current_action_id(state)):
            return _failure("Execution blocked", ValueError("The approved action has changed."))
        if object_tracker is not None and (
                not state["approved_tracking_id"] or
                state["approved_tracking_id"] != state["reviewed_tracking_id"] or
                state["approved_tracking_id"] != current_tracking_id(state)):
            return _failure("Execution blocked", ValueError("The approved file effects have changed."))
        try:
            if object_tracker is None or not state["object_effects"]:
                result = record_operation("commands.execute", services.execute_commands, state["commands"])
            else:
                record_operation("objects.check_before", object_tracker.check_before,
                                 state["object_effects"], state["object_before"],
                                 state["commands"], state["initial_dir"])
                # Capture the outer command operation, not the nested shell operation.
                # Exiting this context durably records execution before object verification.
                with operation("commands.execute", {"args": [state["commands"]], "kwargs": {},
                                "expected_effects": state["object_effects"]}) as outcome:
                    reference = current_operation_reference()
                    result = services.execute_commands(state["commands"])
                    outcome["result"] = result
                    outcome["status"] = "succeeded" if result.get("success") is True else "failed"
            if (type(result.get("success")) is not bool
                    or not isinstance(result.get("output"), str)
                    or not isinstance(result.get("current_dir"), str)
                    or not result["current_dir"]):
                raise ValueError("The executor returned an invalid result.")
            success = result["success"]
            update = {
                "result": result,
                "current_dir": result["current_dir"] if success and state["is_jump"] else state["initial_dir"],
                "approved": False,
                "approved_action_id": "",
                "approved_tracking_id": "",
                "error": "" if success else result["output"],
                "status": "succeeded" if success else (
                    "running" if state["retries"] < state["max_retries"] else "failed"
                ),
            }
        except Exception as exc:
            return _failure("Command execution failed", exc)

        if object_tracker is not None and state["object_effects"]:
            # A tracked write may have partially happened. Never blindly replay it.
            if not success:
                update["status"] = "failed"
            try:
                verifications = record_operation("objects.verify", object_tracker.verify,
                                                 state["object_effects"], state["object_before"])
                update["object_verifications"] = verifications
                if any(item["status"] != "verified" for item in verifications):
                    reasons = "; ".join(item["reason"] for item in verifications
                                        if item["status"] != "verified")
                    update.update(status="failed", tracking_error=reasons,
                                  error="File effects could not be verified: " + reasons)
            except InteractionLogError:
                raise
            except Exception as exc:
                update.update(status="failed", tracking_error=str(exc),
                              error="File effects could not be verified: " + str(exc))
                return update
            try:
                update["tracked_entities"] = record_operation(
                    "objects.apply", object_tracker.commit, state["object_effects"],
                    verifications, reference, state["initial_dir"], state["reviewed_action_id"],
                    "succeeded" if success else "failed")
            except InteractionLogError:
                raise
            except Exception as exc:
                # The shell result is already durable. Save errors are separate from
                # execution; they cannot request another execution or erase its result.
                update["tracking_error"] = str(exc)
        return update

    def correct_error(state: SamanthaState):
        try:
            intent = record_operation("agent.correct_error", services.correct_error,
                contextual_input(state, state["clarified_request"]),
                state["commands"], state["error"]
            )
            return {"intent": _require_text(intent), "retries": state["retries"] + 1}
        except Exception as exc:
            return _failure("Error correction failed", exc)

    def continue_to(node: str):
        return lambda state: node if state["status"] == "running" else END

    builder = StateGraph(SamanthaState)
    for name, node in {
        "answer_file_question": answer_file_question,
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

    first_node = "load_objects" if object_tracker is not None else "answer_file_question"
    if object_tracker is not None:
        builder.add_node("load_objects", load_objects)
        builder.add_conditional_edges("load_objects", continue_to("answer_file_question"),
                                      ["answer_file_question", END])
    if memory is not None:
        builder.add_node("retrieve_memory", retrieve_memory)
        builder.add_edge(START, "retrieve_memory")
        builder.add_conditional_edges("retrieve_memory", continue_to(first_node), [first_node, END])
    else:
        builder.add_edge(START, first_node)
    builder.add_conditional_edges("answer_file_question", route_file_question,
                                  ["ask_clarification", "clarify", END])
    builder.add_conditional_edges("clarify", route_clarification, ["ask_clarification", "parse_intent", END])
    builder.add_conditional_edges("ask_clarification", route_clarification_answer,
                                  ["answer_file_question", "clarify", END])
    for source, target in [
        ("parse_intent", "generate_commands"),
        ("generate_commands", "review_commands"),
        ("review_commands", "confirm"),
        ("confirm", "execute"),
        ("execute", "correct_error"),
        ("correct_error", "generate_commands"),
    ]:
        builder.add_conditional_edges(source, continue_to(target), [target, END])
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
