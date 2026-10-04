import sys
from pathlib import Path
from datetime import datetime
from chat import request_text

# User Confirmation Agent  
# Purpose: Summarize the implementation steps of the abstracted commands, highlight potential risks, and ask for user confirmation  
# Input: Previously generated Linux terminal commands (string list)  
# Output: Overview of command implementation steps, risk warnings, and yes/no confirmation

# confirm the code intent with user
def code_confrim(input_code_string_list):
    """
    User Confirmation Agent  
    Purpose: Summarize the implementation steps of the abstracted commands, highlight potential risks, and ask for user confirmation  
    Input: Previously generated Linux terminal commands (string list)  
    Output: Overview of command implementation steps, risk warnings, and yes/no confirmation
"""

    current_path = get_current_path()
    current_time = get_current_time()

    begin_messages = f"""
        # Role: Command Explanation & Confirmation Agent
        You are a **Command Explanation & Confirmation Agent**.
        Your task is to take one or more Linux commands as input and:

        1. Explain in **clear, simple natural language** what will happen — do **NOT** include or quote the commands.
        2. **Do not explain line-by-line.** Give a **short**, high-level summary of **which folder(s)** the commands will operate in and **what actions** they will perform.
        3. Summarize in **1 to 4 key points total** (regardless of the number of commands).
        4. Highlight any **risks or irreversible actions** (if any).
        5. Ask the user whether to proceed by replying `y` (yes) or `n` (no).

        ---

        ## How to Explain (Style Rules):
        - Keep it **non-technical** and concise; avoid jargon and do not show code or command text.
        - Focus on **paths and actions**: e.g., “Go to folder X, then create/rename/move/delete files Y...”
        - If the commands change directories or use absolute paths, **explicitly state** the working folder(s) where actions occur.
        - If no path is specified, **assume the current directory** ({current_path}) as the working folder.
        - If there are potential dangers (deleting/overwriting/moving system files), include a **⚠️ warning**.

        ---

        ## Output Format:
        Your output **must follow this structure** (if there is no danger, skip the Danger Warnings section):

        ```
        # Command Explanation
        1. ...
        2. ...
        ...

        # Danger Warnings
            - ⚠️ ...

        # Confirmation
        Do you want to proceed? (y/n)

        ```

        ## Information You Have:
        - Current Directory: {current_path}
        - Current Time: {current_time}

    """

    return request_text(begin_messages, str(input_code_string_list))

# get current path
def get_current_path():
    return Path.cwd().as_posix()

# get current system time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    input_list = [
        ["rm -rf /var/logs/temp/*"],
        ["cd my_project", "mkdir backup"]
    ]
    for user_input in input_list:
        print("\n" + "="*50 + "\n")
        print("input command:")
        print(user_input)
        print("\nconfirm note:")
        intent = code_confrim(user_input)
        print(intent)
    # print(get_current_path())
