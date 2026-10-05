#!/usr/bin/env bash
# Check lab helpers and an unavailable-model failure. No approval or model inference.
set -euo pipefail
export QWEN_BASE_URL=http://model-for-smoke.invalid/v1 QWEN_TIMEOUT=5
source /work/tests/real_tasks/lab.bashrc

[ "$(id -u)" != 0 ]
[ ! -w /work/src/samantha.py ]
task-use R02
lab_failure_run="$TASK_RUN_DIR"
[ "$QWEN_BASE_URL" = http://127.0.0.1:9/v1 ]
if samantha "$TASK_REQUEST" > "$TASK_RUN_DIR/terminal.txt" 2>&1; then
    printf 'Unavailable-model task unexpectedly succeeded.\n' >&2
    exit 1
fi
[ ! -e "$TASK_RUN_DIR/workspace/outage-test.txt" ]
if task-check > "$TASK_RUN_DIR/check-output.txt"; then
    printf 'The pending manual review was incorrectly graded as passed.\n' >&2
    exit 1
else
    lab_check_status=$?
fi
[ "$lab_check_status" -eq 2 ]
python3 - "$TASK_RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
report = json.loads(next(run.glob("result-*.json")).read_text(encoding="utf-8"))
assert report["status"] == "incomplete", report
assert all(check["status"] == ("pending" if check["check"].startswith("人工:") else "passed")
           for check in report["checks"]), report
PY
task-use F03
[ "$TASK_RUN_DIR" != "$lab_failure_run" ]
[ "$QWEN_BASE_URL" = http://model-for-smoke.invalid/v1 ]
[ "$SAMANTHA_LOG_PATH" = "$TASK_RUN_DIR/interactions.sqlite3" ]
[ "$CONTENT_QDRANT_PATH" = "$TASK_RUN_DIR/content-index" ]
[ -f "$TASK_RUN_DIR/workspace/客户资料/October report.txt" ]
printf 'Lab helpers passed: non-root, read-only code, fresh tasks, per-task storage, unavailable-model failure and incomplete grading.\n'
