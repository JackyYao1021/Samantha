from intent_explain import *
from code_confrimation import *
from clarify_intent import *
from n2c import *
from run_commands import run_commands
from error_correction import *
import json
import sys

def parse_commands(text):
    """
    Parse the response text to extract commands and explanation.
    
    parameters:
        text (str): response text from the agent
    
    returns:
        tuple: (list of commands, explanation string)
    """
    try:
        data = json.loads(text)
        commands = data.get("Commands", [])
        explanation = data.get("Explanation", "")
        return commands, explanation
    except json.JSONDecodeError:
        return [], "Failed to parse commands."

def process(initial_input, terminal_history):

    print("Checking the clarity of your request...")

    change_dir = True

    ######### use agent0 to check clearness ##########
    change_dir, clarified_msg = clarify_user_intent(initial_input)

    # print("agent0: Clarified Message:\n", clarified_msg)
    print("Trying to understand your request...")

    ######### go to agent1 ##########
    intent = parse_user_intent(clarified_msg)

    # print("agent1: Intent Breakdown:\n", intent)

    print("Generating commands...")
    ######### go to agent2 ##########
    # TODO: include both terminal history and inline history
    response = natural_language_to_command_agent(intent)

    # print("agent2: N2C Response:\n", response)

    commands, _ = parse_commands(response) # parse response to get commands
    # print("agent2: Parsed Commands:\n", commands)

    ######### go to agent3 ##########
    if commands:
        intent = code_confrim(commands)
    else:
        intent = "No commands generated."

    # ask for user confirmation
    print(intent.strip().strip("`").strip()) #"Confirmation Note:\n", 

    confirmation_input = input().strip().lower()

    if confirmation_input == 'y':
        ########## execute commands ##########
        # TODO: figure out if need to change the directory after executing commands?
        result = run_commands(commands)
        attempt = 0
        max_attempts = 3
        while not result["success"] and attempt < max_attempts:
            # print(result)
            error_message = result["output"]
            print(f"Error Message:\n{error_message}\n")
            print("Dealing with the error...")
            intent = error_correction_agent(initial_input, commands, error_message)
            # print(intent)
            response = natural_language_to_command_agent(intent)
            # print("agent4: Error Correction Response:\n", commands)
            commands, _ = parse_commands(response)
            if commands:
                intent = code_confrim(commands)
            else:
                intent = "No commands generated."
                
            print(intent.strip().strip("`").strip())
            confirmation_input = input().strip().lower()
            if confirmation_input == 'y':
                result = run_commands(commands) 
            else:
                result = {"success": False, "output": "Command execution cancelled by user.", "current_dir": os.getcwd()}
                break
            attempt += 1
            
        if result and result["success"]:
            print("Commands executed successfully ^-^ ")   
        else:
            print(f"Command execution unsuccessful.\nError Message:\n{result['output']}")
            
        # print("Success:", result["success"])
        print("-----Output-----\n", result["output"])

        if change_dir:
            # save the current directory to a file
            json.dump({"current_dir": result["current_dir"]}, open("/tmp/current_dir.json", "w"))

    else:
        print("Command execution cancelled by user.")

# process("take me to the upper level directory and create a folder named test, then create a file named test.txt in it, write 'hello world' to the file, and finally display the content of the file", [])

if __name__ == "__main__":

    user_command = " ".join(sys.argv[1:]).strip()
    if user_command:  # only execute if user_command is not empty
        process(user_command, [])
    else:
        print("Error: empty command provided.\n"
                "Usage: samantha <command>\n"
                "Example: samantha create a file named test.txt in the current directory and write hello world to it")

# process("delete the directory /work/test", [])