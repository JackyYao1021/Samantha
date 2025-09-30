import json
from openai import OpenAI
import requests
from openai import AzureOpenAI

endpoint = "https://sencemaking.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "REMOVED_CREDENTIAL"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


"""
Natural Language to Command Agent
Input: Natural language instructions
Output: Corresponding OpenEuler commands && Explanation
"""

def natural_language_to_command_agent(client, user_input, history=[]):
    messages = [
        {
        "role": "system", 
        "content": """
        Role: Linux Command Generation Agent
         
        You are a **Linux Command Generation Agent**.

        Your task is to take one or more natural language inputs and:
        
        1. Translate the natural language input into one or more OpenEuler commands.
        (OpenEuler is a Linux distribution developed by Huawei, based on the Linux kernel.)

        ---

        ## Your Responsibilities:
        - Understand the user's natural language input.
        - Translate the input into accurate and efficient OpenEuler commands.
        - Ensure the commands are safe to execute and do not pose any security risks.
        - Add command options as necessary to ensure the command works as intended.
        ---

        ## Output Format:
        If the input is clear and complete, respond with the exact OpenEuler command(s) with following JSON format:
        {
            "Commands": ["command1", "command2", "..."],
            "Explanation": "A brief explanation of what the command(s) do."
        }
        """},
    ]
    messages += history
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        model=deployment
    )

    return response.choices[0].message.content
    
def chat_loop():
    history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = natural_language_to_command_agent(client, user_input, history)
        print(response)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

def test(user_input=None):
    while True:
        response = natural_language_to_command_agent(client, user_input)
        print(response)
        if "Commands" in response:
            break
        

if __name__ == "__main__":
    # chat_loop()
    
    # input_list = [# "How to check the disk usage of the root directory?",
    #               # "How to list all files in a directory including hidden files?", 
    #               "How to find all files with a specific extension in a directory and its subdirectories?",]
    #             #   "How to check the status of a service in OpenEuler?",
    #             #   "How to display the current network configuration?"]
    
    input_list = [
        "Go to my downloads folder",
        "Switch to root directory",
        "Move to the parent folder",
        "Enter the etc folder",
        "Go to my home"
    ]
    for input in input_list:
        print(f"User: {input}")
        test(input)
        print("\n")