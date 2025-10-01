from chat import Chat
import sys
import os
from datetime import datetime

# 获得当前所在的路径
def get_current_path():
    return os.getcwd()

# 获得当前系统时间
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clarify_user_intent(user_input):
    current_path = get_current_path()
    current_time = get_current_time()
    begin_messages = f"""
    # Role: Clarification Agent

    You are a **Clarification Agent**. Your job is to check if the user's natural language request for a Linux task is **clear and complete**.
    If it's vague or missing critical information (like file path, destination folder, or operation details), ask **ONE specific question** to clarify it.
    Keep asking questions until the request is specific enough for execution.

    ## Guidelines:
    - Be polite and concise.
    - Each round, ask ONLY ONE focused question.
    - Continue the conversation until all necessary details are collected.
    - Once the request is fully clear and actionable, respond with:
      "✅ Intent is clear. Ready to parse."

    ## Examples:

    User: "Create a new name.txt file"
    You: "Could you tell me which folder you'd like to create the file in?"

    User: "Rename a file"
    You: "Which file do you want to rename, and what should the new name be?"

    User: "Create a file 'test.txt' in /home/user"
    You: "✅ Intent is clear. Ready to pass to Intent Parsing Agent."

    ## Information You Have:
    - Current Directory: {current_path}
    - Current Time: {current_time}
    """
    chat_agent = Chat(begin_messages=begin_messages)
    user_turn = user_input

    while True:
        reply = chat_agent.chat_context(user_turn)
        print("Agent:", reply.strip())

        if "✅" in reply:
            break
        user_turn = input("User: ")

    # 去掉多轮对话中的最初提示词和最后的确认消息(删去，messages的[0]和[-1])
    messages = chat_agent.get_messages()[1:-1]

    return str(messages)  # 把澄清阶段的上下文返回给后续解析阶段


if __name__ == "__main__":
    # 示例交互循环
    user_input = "I want to copy a file"
    clarify_user_intent(user_input)
