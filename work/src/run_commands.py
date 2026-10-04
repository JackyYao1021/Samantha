import subprocess
import os

def run_commands(commands):
    """
    run a list of shell commands sequentially and return their combined output and success status.

    parameters:
        commands (list[str]): terminal commands, e.g., ["cd /tmp", "ls -l"]

    returns:
        dict: {
            "success": bool,  # if all commands executed successfully
            "output": str     # combined stdout and stderr of the commands
        }
    """
    current_dir = os.getcwd()

    try:
        # concatenate commands with '&&' to ensure sequential execution
        script = " && ".join(commands + ["pwd"])

        # use bash to run the commands
        result = subprocess.run(
            [os.environ.get("SAMANTHA_BASH", "/bin/bash"), "-c", script],
            shell=False,
            text=True,
            capture_output=True
        )

        success = (result.returncode == 0)

        if success:
            # extract the current directory
            lines = result.stdout.strip().splitlines()
            current_dir = lines[-1]  # last line is the current directory after 'pwd'

            # remove the last line (current directory) from output
            command_output = "\n".join(lines[:-1]) if len(lines) > 1 else ""
        else:
            command_output = result.stdout

        output = command_output + result.stderr

        return {
            "success": success,
            "current_dir": current_dir,
            "output": output.strip()
        }
    except Exception as e:
        return {
            "success": False,
            "current_dir": current_dir,
            "output": f"Exception occurred: {str(e)}"
        }

