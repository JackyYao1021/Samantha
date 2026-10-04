_SAMANTHA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

samantha() {
    local script_dir state_file target_dir status
    script_dir="$_SAMANTHA_SCRIPT_DIR"
    state_file="$(mktemp "${TMPDIR:-/tmp}/samantha.XXXXXX")" || return

    if SAMANTHA_STATE_FILE="$state_file" python3 "$script_dir/src/samantha.py" "$@"; then
        status=0
    else
        status=$?
    fi

    if [ "$status" -eq 0 ] && [ -s "$state_file" ]; then
        target_dir=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["current_dir"])' "$state_file")
        if [ -n "$target_dir" ] && [ -d "$target_dir" ]; then
            cd -- "$target_dir" || status=$?
        fi
    fi
    rm -f -- "$state_file"
    return "$status"
}
