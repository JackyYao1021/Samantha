To directly control the current terminal, we use a shell function to call the python program and handle its outputs.

#### The following steps show how to run the shell function in your terminal:

1. In your docker container, run: ```vi ~/.bashrc```

    It will open the vim editor. Copy and paste the following contents to the end of your bashrc file:

    ```bash
    run_cmd() {
        output=$(python /work/run/test_run.py)

        target=$(echo "$output" | grep '^::TARGET_PATH:: ' | sed 's/^::TARGET_PATH:: //')

        echo "$output" | grep -v '^::TARGET_PATH::'

        if [ "$1" = "true" ]; then
            cd "$target" || return
        fi
    }
    ```
    __Your may need to change `/work/run/test_run.py` to your own path of the python file.__

    _**Tips:** in vim, type `i` to insert contents, type `:wq` to save and exit._

2. Run: `source ~/.bashrc`
3. After that, you can use `run_cmd` in your terminal, which will directly run the python file in the given path and jump to the final path if needed. 

    For example: ```run_cmd true```. The first parameter `true` means you'll stay at the final path of the commands that have been run. Let it be anything else if you don't want to change your current path.