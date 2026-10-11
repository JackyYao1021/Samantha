"""Read-only diagnostics of saved runs, without altering their frozen grades."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .dataset import load_dataset
from .harness import write_json
from .manifest import file_hash, verify_frozen
from .scorers.task import score_task, SCORER_VERSION


def audit_runs(paths):
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("Audit run paths must be unique")
    audits, all_calls = [], []
    started_reservations = 0
    for path in paths:
        run = Path(path).resolve(strict=True)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        verify_frozen(run, manifest)
        cases = {case["case_id"]: case for case in load_dataset(run / "frozen/dataset.jsonl")}
        current, original, budget_stops = [], [], []
        calls, executions = [], 0
        results = sorted((run / "attempts").glob("*/result.json"))
        for result_path in results:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            original.append(saved["evaluation_status"])
            artifact = (run / saved["evidence_ref"]).resolve()
            if not artifact.is_relative_to(run):
                raise ValueError("Evidence reference escapes run")
            if artifact.is_file() and saved.get("evidence_sha256"):
                if file_hash(artifact) != saved["evidence_sha256"]:
                    raise ValueError("Saved evidence hash mismatch")
                evidence = json.loads(artifact.read_text(encoding="utf-8"))
                current.append(score_task(cases[saved["case_id"]], evidence)["evaluation_status"])
            else:
                current.append("incomplete")
            journal = result_path.parent / "artifacts/journal.sqlite3"
            if journal.is_file():
                with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    events = [(kind, json.loads(data)) for kind, data in db.execute("SELECT kind,data FROM events ORDER BY rowid")]
                calls.extend(data for kind, data in events if kind == "model_call" and data.get("physical_request", True))
                executions += sum(kind == "operation_started" and data.get("name") == "commands.execute" for kind, data in events)
                if any(data.get("exception_type") == "BudgetExceeded" for _, data in events):
                    budget_stops.append(saved["case_id"])
        reservation = len(results) * manifest["config"].get("budget", {}).get("max_calls_per_attempt", 0)
        started_reservations += reservation
        all_calls.extend(calls)
        audits.append({"run": str(run), "started_attempts": len(results),
                       "original_frozen_status_counts": dict(Counter(original)),
                       "current_rule_diagnostic_status_counts": dict(Counter(current)),
                       "budget_stopped_cases": budget_stops, "committed_physical_requests": len(calls),
                       "started_call_allowance_upper_bound": reservation, "executed_batches": executions})
    usage = {}
    for name in ("input_tokens", "output_tokens"):
        reported = [call.get("usage", {}).get(name) for call in all_calls if type(call.get("usage", {}).get(name)) is int]
        usage[name] = {"reported_sum": sum(reported), "reported_calls": len(reported),
                       "committed_calls": len(all_calls), "coverage_of_committed_calls": len(reported) / len(all_calls) if all_calls else None,
                       "full_experiment_total": None}
    return {"schema_version": "1.0", "mode": "read_only_diagnostic", "runs": audits,
            "current_scorer_version": SCORER_VERSION, "current_scorer_sha256": file_hash(Path(__file__).parent / "scorers/task.py"),
            "current_source_hashes": {name: file_hash(Path(__file__).parent / name) for name in
                                      ("audit_runs.py", "schemas.py", "dataset.py", "faults.py", "scorers/task.py")},
            "committed_physical_requests_lower_bound": len(all_calls), "started_request_allowances_upper_bound": started_reservations,
            "usage": usage, "api_cost_usd": None,
            "limitations": ["Original frozen grades remain unchanged; current rule diagnostics are a separate audit.",
                            "Committed requests exclude any in-flight request killed before telemetry publication.",
                            "Unknown usage remains unknown; reported sums cover only saved provider usage."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output must be a new file")
    data = audit_runs(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
