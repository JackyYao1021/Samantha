samantha() {
    # todo: attach the history in the terminal

    python /work/src/samantha.py "$*"

    # default - change the path according to the executed commands
    if [ -f /tmp/current_dir.json ]; then
        # extract path from json file
        target_dir=$(jq -r '.current_dir' /tmp/current_dir.json 2>/dev/null)

        # target_dir=$(grep -o '"current_dir": *"[^"]*"' /tmp/current_dir.json | sed 's/.*"current_dir": *"\([^"]*\)".*/\1/')

        # jump to the target directory if it exists
        if [ -n "$target_dir" ] && [ -d "$target_dir" ]; then
            cd "$target_dir" || return
        fi
    fi
}