from chat import Chat
import re
from pathlib import Path
from datetime import datetime
from model_output import parse_json_object, strip_code_fence

# get current path
def get_current_path():
    return Path.cwd().as_posix()

# get current system time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

CLARIFICATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "clarification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "requirement_summary": {"type": "string"},
                "is_jump": {"type": "boolean"},
                "cancelled": {"type": "boolean"},
            },
            "required": ["question", "requirement_summary", "is_jump", "cancelled"],
            "additionalProperties": False,
        },
    },
}


def clarification_prompt():
    return f"""You clarify Linux terminal tasks. Return only a JSON object with four fields:
question (string), requirement_summary (string), is_jump (boolean), cancelled (boolean).

- If essential information is missing, ask one focused question in question.
  Set requirement_summary="", is_jump=false, cancelled=false.
  For creating a file or directory, its name is essential. Do not invent names.
- If the task is complete, set question="", cancelled=false, and describe the goal
  in requirement_summary using precise English and explicit paths or file names.
- If the user asks to stop or cancel, set cancelled=true, both strings empty,
  and is_jump=false.
- For file operations with no path, assume the current directory. Do not ask for a path.
- For pure navigation, stay in the destination (is_jump=true) without asking.
- For a task acting in another directory, ask whether to stay in the starting or
  destination directory only if the user has not already specified the final directory.
- is_jump=false means stay in the starting directory; true means stay at the destination.
- Do not execute tasks, generate commands, or return environment metadata as your answer.

Examples:
User: Create a file.
Answer: {{"question":"What should the file be named?","requirement_summary":"","is_jump":false,"cancelled":false}}
User: Create test.txt here and stay here.
Answer: {{"question":"","requirement_summary":"Create test.txt in the current directory {get_current_path()} and stay there.","is_jump":false,"cancelled":false}}
User: Go to /tmp.
Answer: {{"question":"","requirement_summary":"Change directory to /tmp and stay in /tmp.","is_jump":true,"cancelled":false}}

Current directory: {get_current_path()}
Current time: {get_current_time()}
"""


def parse_clarification_reply(reply):
    text = strip_code_fence(reply)
    if text == "--end of chat--":
        return {"cancelled": True}
    if text.startswith(("{", "[")) or re.search(r'"is_jump"\s*:', text):
        data = parse_json_object(text)
        if "cancelled" in data and type(data["cancelled"]) is not bool:
            raise ValueError("cancelled must be a boolean.")
        if "question" in data and not isinstance(data["question"], str):
            raise ValueError("question must be a string.")
        if data.get("cancelled") is True:
            return {"cancelled": True}
        if data.get("question"):
            question = data["question"]
            if not question.strip():
                raise ValueError("question must be a non-empty string.")
            return {"question": question}
        if type(data.get("is_jump")) is not bool:
            raise ValueError("is_jump must be a boolean.")
        summary = data.get("requirement_summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("requirement_summary must be a non-empty string.")
        return {"is_jump": data["is_jump"], "requirement_summary": summary}
    return {"question": text}


def clarify_intent_once(user_input, history=None):
    """One model turn; LangGraph owns clarification history and user input."""
    chat_agent = Chat(begin_messages=clarification_prompt(), response_format=CLARIFICATION_RESPONSE_FORMAT)
    chat_agent.messages.extend(history or [])
    return parse_clarification_reply(chat_agent.chat_context(user_input))


def clarify_user_intent(user_input):
    """Compatibility helper for running this agent on its own."""
    history = []
    while True:
        response = clarify_intent_once(user_input, history)
        if response.get("cancelled"):
            raise SystemExit(0)
        if "question" not in response:
            return response["is_jump"], response["requirement_summary"]
        print("Agent:", response["question"])
        history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response["question"]},
        ])
        user_input = input("User: ")


if __name__ == "__main__":
    user_input = "I want to copy a file"
    print(clarify_user_intent(user_input))
