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
    begin_messages = f"""Explain the supplied shell commands before user approval.
Return a concise final explanation under 120 words. Do not discuss your analysis.
Use 1-4 bullets under '# Command Explanation' to describe the working folders,
file names, and actual actions. Do not quote commands or claim they already ran.
Warn about deleting, overwriting, or changing permissions under '# Danger Warnings'
when those actions occur. End with '# Confirmation' and 'Proceed? (y/n)'.
The current directory already exists: {current_path}.
Relative paths refer to it. 'mkdir -p' ensures a directory exists; it creates no file.
"""
    return request_text(begin_messages, str(input_code_string_list), max_tokens=768)

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
