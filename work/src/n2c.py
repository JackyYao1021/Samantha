from chat import Chat


def natural_language_to_command_agent(user_input):
    """
        Natural Language to Command Agent
        Input: Natural language instructions
        Output: Corresponding OpenEuler commands && Explanation
    """
    
    begin_messages = """
        Role: Linux Command Generation Agent

        You are a **Linux Command Generation Agent**.

        Your task is to take one or more natural language inputs and:

        1. Translate the natural language input into one or more OpenEuler (Linux) commands.
        (OpenEuler is a Linux distribution developed by Huawei, based on the Linux kernel.)

        ---

        ## Your Responsibilities:
        - Understand the user's natural language input and convert it into **accurate, efficient, and directly executable** OpenEuler commands.
        - Always assume **user confirmation has already been obtained** by the previous agent — you do **not** need to ask again.
        - Ensure the commands are **non-interactive**:
        - Add `-y` or `--yes` to commands that would normally prompt for confirmation.
        - Add `-f` (force) to commands like `rm`, `cp`, `mv`, etc., where appropriate, to avoid interruptions.
        - Avoid interactive flags such as `-i` (e.g., do **not** use `rm -i`).
        - Commands must be **ready to run directly in a Bash terminal** without additional user input.
        - Use **single quotes** for filename wildcards to prevent unwanted shell expansion.
        - Add any other necessary options to ensure the command works as intended and runs successfully in most cases.

        ---

        ## Output Format:
        If the input is clear and complete, respond with the exact OpenEuler command(s) using the following JSON format:

        {
        "Commands": ["command1", "command2", "..."],
        "Explanation": "A brief explanation of what the command(s) do."
        }
    """

    
    chat_agent = Chat(begin_messages, response_format={"type": "json_object"})
    response = chat_agent.chat_context(user_input)
    return response
    

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
        "Copy config.yaml to backup folder",
        "Duplicate data.txt as data.bak",
        "Copy everything in docs to archive",
        "Copy photo.jpg into ~/Pictures",
        "Copy all .txt files to notes/",


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
