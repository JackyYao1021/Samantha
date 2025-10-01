from intent_explain import *
from code_confrimation import *
from clarify_intent import *
from n2c import *
from run_commands import run_commands
import json

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

    history = []

    history.append({"role": "user", "content": initial_input})

    ######### use agent0 to check clearness ##########
    # clarified_msg = clarify_user_intent(input)

    ######### go to agent1 ##########
    intent = parse_user_intent(initial_input)

    ######### go to agent2 ##########
    # TODO: include both terminal history and inline history
    response = natural_language_to_command_agent(intent)

    commands, _ = parse_commands(response) # parse response to get commands


    ######### go to agent3 ##########
    if commands:
        intent = code_confrim(commands)
    else:
        intent = "No commands generated."

    # ask for user confirmation
    print("Confirmation Note:\n", intent)

    confirmation_input = input().strip().lower()

    if confirmation_input == 'y':
        ########## execute commands ##########
        # TODO: figure out if need to change the directory after executing commands?
        change_dir = True
        result = run_commands(commands)
        print("Success:", result["success"])
        print("Output:\n", result["output"])

        if change_dir:
            # save the current directory to a file
            json.dump({"current_dir": result["current_dir"]}, open("/tmp/current_dir.json", "w"))

    else:
        print("Command execution cancelled by user.")

process("take me to the upper level directory and create a folder named test, then create a file named test.txt in it, write 'hello world' to the file, and finally display the content of the file", [])