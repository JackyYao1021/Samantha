from src.intent_explain import *
from src.code_confrimation import *
from n2c import *
from run.run_commands import run_commands
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
    
def agent0():
    return False  # TODO: implement intent clarity check

def process(input, terminal_history):

    history = []

    history.append({"role": "user", "content": input})

    ######### use agent0 to check clearness ##########
    unclear = True

    while True:
        unclear, message = agent0()  # TODO: determine if intent is unclear
        if unclear == False:
            break

        # TODO: ask for clarification if intent is unclear
        if unclear:
            print(message)
            user_input = input("User: ")
            history.append({"role": "assistant", "content": message})
            history.append({"role": "user", "content": user_input})


    ######### go to agent1 ##########
    # TODO: directly input history?
    intent = parse_user_intent(history)


    ######### go to agent2 ##########
    # TODO: include both terminal history and inline history
    response = natural_language_to_command_agent(client, intent, history)

    # parse response to get commands
    commands, _ = parse_commands(response)


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