"""Validate structured model output before it enters the workflow."""

import json
import re


def strip_code_fence(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model returned an empty response.")
    return re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
    ).strip()


def parse_json_object(text: str) -> dict:
    try:
        data = json.loads(strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("The model response must be a JSON object.")
    return data


def parse_commands(text: str) -> tuple[list[str], str]:
    data = parse_json_object(text)
    commands = data.get("Commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("Commands must be a non-empty list.")
    if any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError("Each command must be a non-empty string.")
    explanation = data.get("Explanation", "")
    if not isinstance(explanation, str):
        raise ValueError("Explanation must be a string.")
    return commands, explanation
