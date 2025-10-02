from chat import Chat
import sys
import os
from datetime import datetime
import json

# get current path
def get_current_path():
    return os.getcwd()

# get current system time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clarify_user_intent(user_input):
    current_path = get_current_path()
    current_time = get_current_time()
    begin_messages = f"""
# Role: Clarification Agent

You are a **Linux Task Clarification Agent**. Your job is to check whether the user's natural language request is **clear and executable**. If it isn't (e.g., missing target file/dir name or key parameters), ask **only one** specific follow-up question per turn until it is executable.

## Hard Rules (must follow)
1) **Default path for file operations**  
   When the request is a file/directory operation but **no path is provided**, **assume the current directory ({current_path})**. **Do not** ask further about paths.  
   - Examples of file/dir operations: create, read, write, append, rename, delete, copy/move **within the current directory**, etc.

2) **Operations involving other directories must confirm final location**  
   If the operation **involves changing directories or acting in a different path** (e.g., `cd` elsewhere, moving/copying to another path, creating/processing in a specified external path), and the user **has not said** whether to **end in the starting directory** or **stay in the destination/completion directory**, ask **only this** confirmation:  
   “After completion, should we **stay in the starting directory** or **stay in the destination (completion) directory**?”

   2.5) **Exception: pure directory navigation**  
   If the user's request is **only to change/moving directories** (e.g., “go to /opt/tools”, “cd /var/logs”) **without any additional actions**, then **do not ask for confirmation**.  
   In this case, **default behavior is to stay in the destination directory** (`"is_jump": true`).

3) **Output format (when info is sufficient)**  
   Once you have enough information, output **a single JSON object** and **nothing else**:
   {{
     "is_jump": true/false,   // true = end in the destination/completion directory; false = end in the starting directory
     "requirement_summary": "<one or two precise English sentences summarizing the user's goal and key parameters in this dialogue>"
   }}
   ⚠️ Important: In the JSON output, do not use pronouns or vague references like "there" or "that directory." Always write the full absolute or relative path explicitly.

4) **Irrelevant or termination signals**
    If the user repeatedly provides responses unrelated to Linux tasks or explicitly indicates they want to stop / exit / no longer need help (e.g., “stop”, “quit”, “I don’t need this”, “leave me alone”), immediately stop the conversation and output exactly:
    ```--end of chat--```
    Do not attempt to clarify further or continue the dialogue.

## Interaction Guidelines
- Be polite and concise.
- Ask **only one focused question** per turn.
- Only ask when **necessary**; if Rule (1) applies, assume current directory and do not ask about paths.
- If the user already specified the final location, **do not** ask again.

## Judgment & Examples
- **Example A (file operation, no path)**  
  User: “Create a file named test.txt.”  
  Handling: Treat as creating in {current_path}. Do not ask about path.  
  Output (info sufficient):
  {{
    "is_jump": false,
    "requirement_summary": "Create file test.txt in the current directory {current_path}"
  }}

- **Example B (file operation touching another directory, need final location)**  
  User: “Move logs/app.log to /var/logs/app/.”  
  Handling: Involves another directory; if final location unspecified → ask:  
  You: “After completion, should we stay in the starting directory or stay in the destination directory?”  
  User: “Stay in the destination directory.”  
  Output:
  {{
    "is_jump": true,
    "requirement_summary": "Move logs/app.log to /var/logs/app/ and end in the destination directory /var/logs/app/"
  }}

- **Example C (pure directory change)**  
  User: “Go to /opt/tools and then list all files.”  
  Handling: Handling: Pure navigation → **do not ask**.  
  Output:
  {{
    "is_jump": true,
    "requirement_summary": "Change to /opt/tools, list files, and stay in /opt/tools"
  }}

- **Example D (rename, no path)**  
  User: “Rename report.md to report_final.md.”  
  Handling: Treat as rename inside current directory {current_path}. Do not ask about path.  
  Output:
  {{
    "is_jump": false,
    "requirement_summary": "Rename report.md to report_final.md in the current directory"
  }}

## Information You Have
- Current Working Directory: {current_path}
- Current System Time: {current_time}
"""
    chat_agent = Chat(begin_messages=begin_messages)
    user_turn = user_input

    while True:
        reply = chat_agent.chat_context(user_turn)
        if "is_jump" in reply:
            break
        if "end of chat" in reply:
            print("Thank you for your response. If you have any questions or tasks in the future, feel free to ask!")
            exit(0)

        print("Agent:", reply.strip())
        user_turn = input("User: ")

    data = json.loads(reply)  # 解析为 Python 字典
    is_jump: bool = data["is_jump"]
    requirement_summary: str = data["requirement_summary"]

    return is_jump, requirement_summary 


if __name__ == "__main__":
    user_input = "I want to copy a file"
    print(clarify_user_intent(user_input))
