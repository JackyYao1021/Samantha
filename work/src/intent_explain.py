import sys
from pathlib import Path
from datetime import datetime
from chat import request_text

# get current path
def get_current_path():
    return Path.cwd().as_posix()

# get current system time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_user_intent(user_input):
    """
    Intent Parsing Agent  
    Purpose: Organize the user's request into a relatively structured form  
    Input: User's natural language request  
    Output: Detailed execution steps, including the starting folder, target folder, and sub-operations
    """

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
        - Do not use pronouns or vague references like "there" or "that directory." Always write the full absolute or relative path explicitly.

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
    return request_text(begin_messages, user_input)

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
