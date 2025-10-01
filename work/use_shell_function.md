To directly control the current terminal, we use a shell function to call the python program and handle its outputs.

#### The following steps show how to run the shell function in your terminal:

1. In your docker container, run: ```vi ~/.bashrc```

    It will open the vim editor. Copy and paste the following contents to the end of your bashrc file:

    ```bash
    samantha() {
        python /work/src/samantha.py "$*"

        # default - change the path according to the executed commands
        if [ -f /tmp/current_dir.json ]; then
            # extract path from json file
            target_dir=$(grep -o '"current_dir": *"[^"]*"' /tmp/current_dir.json | sed 's/.*"current_dir": *"\([^"]*\)".*/\1/')
            
            # jump to the target directory if it exists
            if [ -n "$target_dir" ] && [ -d "$target_dir" ]; then
                cd "$target_dir" || return
            fi
        fi
    }
    ```
    __Your may need to change `/work/run/samantha.py` to your own path of the python file.__

    _**Tips:** in vim, type `i` to insert contents, type `Esc` and `:wq` to save and exit._

2. Run: `source ~/.bashrc`
3. After that, you can use `samantha` in your terminal, which will directly run the python file in the given path and jump to the final path if needed. 

    For example: ```samantha take me to home``` will execute `cd ~` in the terminal and take you to the home directory. 