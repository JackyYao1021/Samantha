run_cmd() {
    output=$(python /path/to/yourscript.py)

    target=$(echo "$output" | grep '^::TARGET_PATH:: ' | sed 's/^::TARGET_PATH:: //')

    echo "$output" | grep -v '^::TARGET_PATH::'

    if [ "$1" = "true" ]; then
        cd "$target" || return
    fi
}