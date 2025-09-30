from run_commands import run_commands
commands = [
    "cd /tmp",
    "mkdir -p test_agent",
    "cd test_agent",
    #"echo 'hello world' > hello.txt",
    "cat hello.txt"
    #"pwd"
]

change_dir = True

result = run_commands(commands)

print("Success:", result["success"])
print("Output:\n", result["output"])

# print the current directory after executing commands (if we need to stay there)
print("::TARGET_PATH::", result["current_dir"], end="")