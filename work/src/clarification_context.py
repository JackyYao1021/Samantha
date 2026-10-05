"""Explicit context and bounded input handling for clarification turns."""

import json
from pathlib import PurePosixPath
import re

MAX_CLARIFICATIONS = 3
MAX_EMPTY_ANSWERS = 3
CANCEL_ANSWERS = {"stop", "quit", "exit", "cancel", "取消", "退出", "停止"}


def _resolved_name(original_request, question, answer):
    # Label only literal names answering an explicit naming question. Sentences,
    # paths, and uncertain replies still go through ordinary clarification.
    if not re.fullmatch(r"[\w.-]+", answer) or answer.casefold() in {
        ".", "..", "yes", "no", "maybe", "unknown", "unsure", "不知道", "不确定",
    }:
        return ""
    if not re.search(r"\b(?:name|named|filename)\b|名称|名字|命名|文件名|叫什么", question, re.I):
        return ""
    if re.search(r"\b(?:directory|folder)\b|目录|文件夹", question, re.I):
        kind = "directory"
    elif re.search(r"\b(?:file|filename)\b|文件", question, re.I):
        kind = "file"
    else:
        return ""
    name = answer
    if kind == "file" and not PurePosixPath(name).suffix and re.search(
        r"\b(?:py|python)\s+(?:file|script)\b|\b(?:file|script)\s+(?:in\s+)?python\b"
        r"|\.py\b|(?:python|py)\s*(?:文件|脚本)", original_request, re.I,
    ):
        name += ".py"
    return (
        f"\nResolved {kind} name supplied by the user: {json.dumps(name, ensure_ascii=False)}. "
        f"Use this name for the original task (answer to {json.dumps(question, ensure_ascii=False)})."
    )


def build_clarification_input(original_request, question, answer, answers=None):
    if not question:
        return answer
    current = {"question": question, "answer": answer}
    earlier = list(answers or [])
    if earlier and earlier[-1] == current:
        earlier = earlier[:-1]
    prompt = (
        f"Original task: {original_request}\n"
        f"Previous clarification question: {question}\n"
        f"User answer to that question: {answer}\n"
        "Evaluate the original task together with this answer and earlier "
        "clarification answers. Interpret the answer as a reply to the previous "
        "question, not as a new standalone task. Ask only for essential "
        "information that is still missing."
    )
    if earlier:
        prompt += "\nEarlier clarification answers:\n" + json.dumps(earlier, ensure_ascii=False)
    for detail in [*earlier, current]:
        prompt += _resolved_name(original_request, detail["question"], detail["answer"])
    return prompt


def check_clarification_question(question, previous_question, count, limit):
    def normalize(text):
        return " ".join(text.casefold().split()).rstrip("?.!。？！").rstrip()

    if previous_question and normalize(question) == normalize(previous_question):
        raise ValueError(
            "The clarification agent repeated a question after receiving an answer: "
            f"{question.strip()} Please restate the task with the requested details."
        )
    if count >= limit:
        raise ValueError(
            f"Clarification limit ({limit}) reached. "
            "Please restate the task with all required details."
        )
