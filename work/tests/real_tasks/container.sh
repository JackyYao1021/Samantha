#!/usr/bin/env bash
# Entry point for the test image. Only fixed offline commands run automatically.
set -euo pipefail

mkdir -p -- "$HOME"
case "${1:-shell}" in
    offline)
        test_batch="$(python3 -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])')"
        test_result_dir="$SAMANTHA_TEST_RESULTS/offline/$test_batch"
        mkdir -p -- "$test_result_dir" /tmp/samantha-offline
        # The existing shell smoke test creates temporary files beside its code.
        # Copy only image code into tmpfs; /work stays read-only throughout.
        cp -R -- /work /tmp/samantha-offline/work
        cd -- /tmp/samantha-offline
        printf 'Running offline unit tests...\n'
        python3 -m unittest discover -s work/tests -q 2>&1 | tee "$test_result_dir/unit-tests.txt"
        printf 'Running Bash wrapper checks...\n'
        bash work/tests/shell_smoke.sh 2>&1 | tee "$test_result_dir/shell-smoke.txt"
        printf 'Running task lab helper checks...\n'
        bash /work/tests/real_tasks/lab_smoke.sh 2>&1 | tee "$test_result_dir/lab-smoke.txt"
        python3 /work/tests/real_tasks/task_suite.py prepare-all \
            --root "$test_result_dir" --name fixtures | tee "$test_result_dir/prepare.txt"
        printf 'Offline checks passed. Fixtures prepared; live tasks have not run.\nResults: %s\n' "$test_result_dir"
        ;;
    prepare)
        shift
        exec python3 /work/tests/real_tasks/task_suite.py prepare-all \
            --root "$SAMANTHA_TEST_RESULTS/prepared" "$@"
        ;;
    suite)
        shift
        exec python3 /work/tests/real_tasks/task_suite.py "$@"
        ;;
    shell)
        exec bash --noprofile --rcfile /work/tests/real_tasks/lab.bashrc -i
        ;;
    *)
        printf 'Usage: offline | prepare [--name batch] | suite <arguments> | shell\n' >&2
        exit 2
        ;;
esac
