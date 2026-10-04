#!/usr/bin/env bash
# Exercise the real wrapper using a fake CLI; no models or generated commands run.
set -eu

test_python="$(command -v "${1:-python3}")"
work_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(dirname -- "$work_dir")"
test_dir="$(mktemp -d "$work_dir/tests/shell.XXXXXX")"
mkdir -- "$test_dir/folder with spaces" "$test_dir/other"
trap 'cd -- "$repository_dir"; rmdir -- "$test_dir/folder with spaces" "$test_dir/other" "$test_dir"' EXIT

# Source a relative path, then invoke from another directory.
cd -- "$repository_dir"
source "$(basename -- "$work_dir")/samantha.sh"
cd -- "$test_dir/other"
start_dir="$PWD"
mock_target="$test_dir/folder with spaces"
mock_status=0
state_files=()

python3() {
    if [ "$1" = "-c" ]; then
        "$test_python" "$@"
        return
    fi
    [ "$1" = "$work_dir/src/samantha.py" ] || return 99
    state_files+=("$SAMANTHA_STATE_FILE")
    "$test_python" -c 'import json, sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"current_dir": sys.argv[2]}), encoding="utf-8")' "$SAMANTHA_STATE_FILE" "$mock_target"
    return "$mock_status"
}

samantha "go to a folder with spaces"
[ "$PWD" = "$mock_target" ]
[ ! -e "${state_files[0]}" ]

cd -- "$start_dir"
mock_target="$start_dir" # A cancelled CLI reports the starting directory.
samantha "cancel"
[ "$PWD" = "$start_dir" ]
[ ! -e "${state_files[1]}" ]
[ "${state_files[0]}" != "${state_files[1]}" ]

mock_status=1
mock_target="$test_dir/folder with spaces"
if samantha "simulate failure"; then
    echo "Failure status was lost" >&2
    exit 1
fi
[ "$PWD" = "$start_dir" ]
[ ! -e "${state_files[2]}" ]
echo "Shell wrapper smoke tests passed (success, cancellation, failure, cleanup)."
