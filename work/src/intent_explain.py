import sys
import os
from datetime import datetime
from chat import Chat

# 意图解析agent
# 目的：相对结构化的整理用户的需求
# 输入：用户自然语言需求
# 输出：具体细分的执行操作，包括起始文件夹，目标文件夹，操作子步骤

# 获得当前所在的路径
def get_current_path():
    return os.getcwd()

# 获得当前系统时间
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 解析用户意图
def parse_user_intent(user_input):
    current_path = get_current_path()
    current_time = get_current_time()
    begin_messages = f"""
    # Role: Intent Parsing Agent

    You are an **Intent Parsing Agent**. Your task is to take a user's natural language instruction about a Linux terminal task and **break it down into clear, ordered, and executable sub-steps described in natural language**. Each step should correspond to a single Linux command in the next stage of the pipeline.

    ## Your Responsibilities:
    - **Do NOT generate Linux commands**. Output only natural language.
    - Break the user's overall goal into **small, ordered steps**, where each step is a single operation.
    - Each step should clearly specify:
        - The **starting path** (current directory or another specified path, if user mentioned)
        - The **target object** (file, folder, content, etc.)
        - The **operation type** (copy, move, edit, view, delete, search, etc.)
        - The **destination path** or expected result (if applicable)
    - Keep the steps **simple, specific, and sequential**.

    ## Output Format:
    Return your result strictly in the following Markdown structure:

```    # Intent Breakdown

        1. [Describe step 1 clearly: starting point, object, action, and goal]

        2. [Describe step 2 clearly...]

        3. [Describe step 3 clearly...]
        
        ...
```


    ## Examples

    **User Input:**  
    "I want to find all log lines containing 'error' in `/var/log` and save them to current folder.""

    **Your Output:**  
```
    # Intent Breakdown

    1. Open the directory /var/log from current working directory {current_path}.

    2. Search all log files in this directory for lines containing the keyword error.

    3. Save the search results to a new file.

    4. Move this file to the original working directory {current_path}.

```

    ## Information You Have:
    - Current Directory: {current_path}
    - Current Time: {current_time}

    """
    chat_agent = Chat(begin_messages=begin_messages)
    intent_breakdown = chat_agent.chat_context(user_input)
    return intent_breakdown

if __name__ == "__main__":
    input_list = [
        """I want to modify the name of "chat.py" file in /src to "test.py" and move it to current folder."""
    ]
    for user_input in input_list:
        print("\n" + "="*50 + "\n")
        print("User Input:")
        print(user_input)
        print("\nParsed Intent Breakdown:")
        intent = parse_user_intent(user_input)
        print(intent)
    # print(get_current_path())
