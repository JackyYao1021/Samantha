# Interactive helpers; model-generated commands still require user confirmation.
source /work/samantha.sh
_TASK_MODEL_BASE_URL="$QWEN_BASE_URL"
_TASK_MODEL_TIMEOUT="$QWEN_TIMEOUT"

task-list() {
    python3 /work/tests/real_tasks/task_suite.py list "$@"
}

task-use() {
    if [ "$#" -ne 1 ]; then
        printf 'Usage: task-use F03\n' >&2
        return 2
    fi
    local task_name task_directory task_request
    task_name="$1-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
    task_directory="$(python3 /work/tests/real_tasks/task_suite.py prepare "$1" \
        --root "$SAMANTHA_TEST_RESULTS/runs" --name "$task_name")" || return
    task_request="$(python3 /work/tests/real_tasks/task_suite.py show "$1" --field request)" || return
    export TASK_RUN_DIR="$task_directory" TASK_REQUEST="$task_request"
    export SAMANTHA_LOG_PATH="$TASK_RUN_DIR/interactions.sqlite3"
    export CONTENT_QDRANT_URL="" CONTENT_QDRANT_API_KEY=""
    export CONTENT_QDRANT_PATH="$TASK_RUN_DIR/content-index"
    export CONTENT_COLLECTION="eval_${task_name//-/_}"
    export QWEN_BASE_URL="$_TASK_MODEL_BASE_URL" QWEN_TIMEOUT="$_TASK_MODEL_TIMEOUT"
    export SAMANTHA_AZURE_FALLBACK=false
    if [ "$1" = R02 ]; then
        export QWEN_BASE_URL=http://127.0.0.1:9/v1 QWEN_TIMEOUT=5
    fi
    cd -- "$TASK_RUN_DIR/workspace" || return
    cat -- "$TASK_RUN_DIR/TASK.txt"
    printf '\nRun the request with: samantha "$TASK_REQUEST"\nFor content tasks, use the content commands printed above.\nResults: %s\n' "$TASK_RUN_DIR"
}

task-check() {
    if [ -z "${TASK_RUN_DIR:-}" ]; then
        printf 'Select a fresh task first with task-use F03.\n' >&2
        return 2
    fi
    if [ "$#" -gt 1 ]; then
        printf 'Usage: task-check [review.json]\n' >&2
        return 2
    fi
    local task_suffix task_export task_report task_status task_review
    local -a task_review_args=()
    task_suffix="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
    task_export="$TASK_RUN_DIR/session-$task_suffix.json"
    task_report="$TASK_RUN_DIR/result-$task_suffix.json"
    printf '%s\n' "$PWD" > "$TASK_RUN_DIR/final-directory-$task_suffix.txt"
    python3 /work/src/samantha.py history export latest --output "$task_export" || return
    task_review="${1:-$TASK_RUN_DIR/review.json}"
    if [ -f "$task_review" ]; then
        task_review_args=(--review "$task_review")
    elif [ "$#" -eq 1 ]; then
        printf 'Review file not found: %s\n' "$task_review" >&2
        return 2
    fi
    if python3 /work/tests/real_tasks/task_suite.py check "$TASK_RUN_DIR" \
        --session-export "$task_export" "${task_review_args[@]}" > "$task_report"; then
        task_status=0
    else
        task_status=$?
    fi
    cat -- "$task_report"
    printf '\nReport: %s\n' "$task_report"
    return "$task_status"
}

printf '\nSamantha real-task lab\n  task-list\n  task-use F03\n  samantha "$TASK_REQUEST"\n  task-check [review.json]\nEvery task-use creates a fresh Linux fixture and independent logs/index.\n\n'
