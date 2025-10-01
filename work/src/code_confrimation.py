import sys
import os
from datetime import datetime
from chat import Chat

# 用户意见征询确认agent
# 目的：总结抽象指令实现步骤，提示风险，征询用户确认
# 输入：前面生成好的linux终端指令(string list)
# 输出：指令大体实现步骤，风险提示，yes/no

# 获得当前所在的路径
def get_current_path():
    return os.getcwd()

# 获得当前系统时间
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 解析用户意图
def code_confrim(input_code_string_list):
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
       3. ...

    # Danger Warnings
         - ⚠️ ...

    # Confirmation
    Do you want to proceed? (y/n)
    ```

    ## Information You Have:
    - Current Directory: {current_path}
    - Current Time: {current_time}
    """

    chat_agent = Chat(begin_messages=begin_messages)
    confirm_note = chat_agent.chat_context(str(input_code_string_list))
    return confirm_note

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
