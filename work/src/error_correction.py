from chat import request_text
from datetime import datetime
from pathlib import Path


def error_correction_agent(user_input, code, error_message):
    """
        Error Correction Agent
        Input: Natural language instructions + Error message from last command
        Output: Corrected Natural language instructions
    """
    
    begin_messages = f"""
        Role: Linux Command Correction Agent

        You are a **Linux Command Correction Agent**.

        Your task is to take original natural language input, the previous code snippet, and the error message from the last executed OpenEuler command, then:
        
        1. Analyze the error message to identify what went wrong with the previous command.
        2. Based on the error analysis, correct the previous natural language input to fix the issue.
        3. Ensure the corrected natural language input is clear, specific, and can be accurately

        ---

        ## Your Responsibilities:
        - Understand the user's natural language input.
        - Analyze the provided error message to determine the root cause of the failure.
        - Modify the original natural language input to address the identified issues.
        - Ensure the corrected input is safe to execute and does not pose any security risks.
        ---

        ## Output Format:
        Return your result strictly in the following Markdown structure:
        ```
                # Intent Breakdown

                1. [Describe step 1 clearly: starting point, object, action, and goal]

                2. [Describe step 2 clearly...]

                3. [Describe step 3 clearly...]
                
                ...
        ```

        ## Information You Have:
        - Current Directory: {get_current_path()}
        - Current Time: {get_current_time()}

    """

    input_message = f"""
        ## Previous User Input:
        {user_input}

        ## Previous Commands:
        {code}

        ## Error Message:
        {error_message}
    
    """
    ## Original User Input:
    
    return request_text(begin_messages, input_message)

# get current path
def get_current_path():
    return Path.cwd().as_posix()

# get current system time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def chat_loop():
    """
        Chat loop for continuous interaction
    """
    
    history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = natural_language_to_command_agent(user_input)
        print(response)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})



def test(user_input=None):
    """
        Test function for the agent
    """
    while True:
        response = natural_language_to_command_agent(user_input)
        print(response)
        if "Commands" in response:
            break
        

if __name__ == "__main__":
    # chat_loop()
    
    input_list = [
        # #Navigation - Change directory
        # "Go to my downloads folder",
        # "Switch to root directory",
        # "Move to the parent folder",
        # "Enter the etc folder",
        # "Go to my home",
        
        # #Navigation - List files
        # "Show me what's in this directory",
        # "Show me all files, even hidden ones",
        # "Give me detailed list here",
        # "Show files sorted by time",   
        # "List all files, even hidden ones",
        # "Give me detailed list here",
        # "Show files sorted by time",
        # "List contents of /var/log"

        # #Creation - Create new empty file
        # "Make a file called test.txt",
        # "Create an empty log file called app.log",
        # "New file notes.md in current folder",
        # "Add a blank file called report.csv",
        # "Create file script.sh here"

        # #Creation - Create new empty directories
        # "Make a folder called Hackathon Project",
        # "Create a folder named src",
        # "Add a new directory output",
        # "New empty directory test_cases",
        # "Make a logs folder inside tmp"

        # #Basic Manipulation - Move file/directory
        # "Move report.pdf to ~/Documents",
        # "Relocate all .log files into logs/",
        # "Move images folder into backup",
        # "Send main.c into src directory",
        # "Move old_data.csv and rename to archive.csv"

        # #Basic Manipulation - Copy file/directory
        # "Copy config.yaml to backup folder",
        # "Duplicate data.txt as data.bak",
        # "Copy everything in docs to archive",
        # "Copy photo.jpg into ~/Pictures",
        # "Copy all .txt files to notes/",


        # #Basic Manipulation - Rename file/directory
        # "Rename old.txt to new.txt",
        # "Change name of logs folder to old_logs",
        # "Rename report.docx as final_report.docx",
        # "Change draft.md to draft_v2.md",
        # "Rename src directory to source",

        # #Simple Search - Find file/directory
        # "Find a file named notes.txt",
        # "Search for folder called logs",
        # "Look for any file named config.json under /etc",
        # "Find script.sh inside current tree",
        # "Locate Makefile"
        
        # #User Feedback
        # "Tell me when the file is created",
        # "Confirm that folder is moved",
        # "Show error if file doesn't exist",
        # "Notify me after copying finishes",
        # "Say success if rename worked"
        
        # "Find all PDF documents in 'Reports'"
        
        # #Critical Feature - Safety
        # "Delete file old.txt",
        # "Remove folder temp_data",
        # "Erase report.docx",
        # "Delete everything in cache",
        # "Remove backup.tar.gz"

    ]
    for input in input_list:
        print(f"User: {input}")
        test(input)
        print("\n")
