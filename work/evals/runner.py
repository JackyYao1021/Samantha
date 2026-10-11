"""Run offline experiments or regrade saved evidence with its frozen scorer.

Examples (from work):
  python -m evals.runner run --output .test-results/evaluation/baseline
  python -m evals.runner regrade --run .test-results/evaluation/baseline
"""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
import sys
import uuid
import time

from .dataset import load_dataset, read_json
from .harness import launch_worker, write_json
from .manifest import file_hash, freeze_run, verify_frozen
from .report import write_report
from .schemas import worker_input


HERE = Path(__file__).resolve().parent


def load_config(path):
    config = read_json(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "name", "mode", "variant", "repeats", "timeout_seconds", "max_attempts", "runtime"}
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {"seed", "model", "budget"}):
        raise ValueError("Configuration must contain supported versioned fields")
    if config.get("schema_version") != "1.0" or config.get("mode") not in {"fake", "model_proposal"}:
        raise ValueError("Only fake and explicitly authorized local model_proposal modes are supported")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("Configuration name must be nonempty text")
    if config.get("variant") not in {"faithful", "noop", "observed_safe_stop"}:
        raise ValueError("Unknown Fake Services variant")
    for name in ("repeats", "max_attempts"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    timeout = config.get("timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    runtime = config.get("runtime")
    if (not isinstance(runtime, dict) or set(runtime) != {"memory", "object_tracking", "azure_fallback"}
            or runtime["memory"] not in {"none", "recent", "bm25"}
            or type(runtime["object_tracking"]) is not bool or runtime["azure_fallback"] is not False):
        raise ValueError("Runtime requires an explicit local memory mode, tracking boolean, and Azure fallback false")
    if "seed" in config and type(config["seed"]) is not int:
        raise ValueError("Seed must be an integer")
    if config["mode"] == "model_proposal":
        if runtime != {"memory": "none", "object_tracking": False, "azure_fallback": False}:
            raise ValueError("Proposal mode requires memory none and tracking off")
        model, budget = config.get("model"), config.get("budget")
        if (not isinstance(model, dict) or set(model) != {"provider", "model", "transport", "max_output_tokens", "request_timeout_seconds"}
                or model["provider"] != "ollama" or model["model"] != "qwen3:4b"
                or model["transport"] not in {"ollama", "openai"}
                or type(model["max_output_tokens"]) is not int or not 1 <= model["max_output_tokens"] <= 4096
                or type(model["request_timeout_seconds"]) not in (int, float)
                or not math.isfinite(model["request_timeout_seconds"]) or not 0 < model["request_timeout_seconds"] <= timeout):
            raise ValueError("Model config must explicitly bound the authorized local qwen3:4b request")
        if (not isinstance(budget, dict) or set(budget) != {"max_calls", "max_calls_per_attempt", "max_seconds"}
                or any(type(budget.get(key)) is not int or budget[key] < 1 for key in ("max_calls", "max_calls_per_attempt"))
                or type(budget.get("max_seconds")) not in (int, float)
                or not math.isfinite(budget["max_seconds"]) or budget["max_seconds"] <= 0):
            raise ValueError("Real model evaluation requires explicit call and wall-time budgets")
    elif "model" in config or "budget" in config:
        raise ValueError("Fake configurations cannot declare real model calls")
    return config


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def frozen_modules(run):
    """Load under a fresh package name so current eval code cannot replace it."""
    package_path = Path(run) / "frozen" / "sources" / "work" / "evals"
    package_name = "_samantha_frozen_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(package_name, package_path / "__init__.py",
                                                  submodule_search_locations=[str(package_path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return (importlib.import_module(package_name + ".dataset"),
            importlib.import_module(package_name + ".scorers.task"))


def build_result(manifest, case, attempt_id, evidence, evidence_ref, score):
    return {"schema_version": "1.0", "experiment_id": manifest["experiment_id"],
            "episode_id": case["case_id"], "case_id": case["case_id"], "attempt_id": attempt_id,
            "config_hash": manifest["config_hash"], "family": case["metadata"]["family"],
            "session_ids": evidence.get("session_ids", []),
            "runtime_status": evidence.get("runtime_status", "failed"),
            "evaluation_status": score["evaluation_status"], "scores": score["scores"],
            "checks": score.get("checks", []), "changed_paths": score.get("changed_paths", []),
            "output": evidence.get("output", ""), "evidence_ref": evidence_ref,
            "trajectory_ref": str(Path(evidence_ref).with_name("trajectory.jsonl")).replace("\\", "/"),
            "evidence_sha256": None, "telemetry": evidence.get("telemetry", {}),
            "usage": {**evidence.get("telemetry", {}).get("usage", {}), "api_cost_usd": None},
            "errors": [*evidence.get("errors", []), *score.get("errors", [])]}


def run_experiment(dataset, config_path, output, *, repeats=None, allow_local_model=False, deadline=None):
    cases = load_dataset(dataset)
    admitted_at = time.monotonic()
    config = load_config(config_path)
    if config["mode"] == "model_proposal":
        if not allow_local_model:
            raise ValueError("Local model calls require explicit --allow-local-model admission")
        if any(case["input"]["interaction"]["approval_policy"] != "deny" or "steps" in case["input"] for case in cases):
            raise ValueError("Real proposal cases must deny command execution and use a single invocation")
        from .local_models import model_identity
        identity = model_identity()
    else:
        identity = None
    if repeats is not None:
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be positive")
        config["repeats"] = repeats
    run = Path(output).resolve()
    run.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous experiment.
    manifest = freeze_run(run, dataset, config, cases)
    if identity:
        manifest["model"] = {**identity, **config["model"], "effective_parameters": {
            "reasoning_effort": "none", "think": False if config["model"]["transport"] == "ollama" else None,
            "max_output_tokens_cap": config["model"]["max_output_tokens"], "azure_fallback": False},
            "parameters_per_request": "Saved in each llm_call; role-specific temperature/top_p/response_format"}
        manifest["budget"].update(config["budget"])
        manifest["isolation"].update(network="loopback_ollama_11434_only", kind="proposal_process_no_executor")
        manifest["limitations"] = ["Real local model proposal/denial experiment; not end-to-end task accuracy.",
            "No generated commands execute; successful proposal leads to expected cancellation.",
            "A socket gate and process boundary do not sandbox arbitrary Bash."]
        write_json(run / "manifest.json", manifest)
    verify_frozen(run, manifest)
    dataset_module, scorer_module = frozen_modules(run)
    cases = dataset_module.load_dataset(run / "frozen" / "dataset.jsonl")
    score_task = scorer_module.score_task
    results = []
    stopped = False
    stop_reason = "completed"
    reserved_calls = 0
    started = admitted_at
    evidences = []
    for repeat in range(1, config["repeats"] + 1):
        schedule = list(cases)
        if "seed" in config:
            random.Random(config["seed"] + repeat).shuffle(schedule)
        for case in schedule:
            if len(results) >= config["max_attempts"]:
                stopped = True
                stop_reason = "attempt_budget"
                break
            effective = config
            if identity:
                remaining_calls = config["budget"]["max_calls"] - reserved_calls
                remaining_seconds = config["budget"]["max_seconds"] - (time.monotonic() - started)
                if deadline is not None:
                    remaining_seconds = min(remaining_seconds, deadline - time.monotonic())
                if remaining_calls <= 0 or remaining_seconds <= 0:
                    stopped, stop_reason = True, "model_call_reservation_or_wall_time_budget"
                    break
                # Reserve the full child allowance, even if it crashes or times out.
                allowance = min(remaining_calls, config["budget"]["max_calls_per_attempt"])
                reserved_calls += allowance
                effective = {**config, "timeout_seconds": min(config["timeout_seconds"], remaining_seconds),
                             "budget": {**config["budget"], "max_calls_per_attempt": allowance}}
            attempt_id = f"repeat_{repeat:02d}"
            name = f"{case['case_id']}-{attempt_id}"
            result_path = run / "attempts" / name / "result.json"
            result_path.parent.mkdir(parents=True, exist_ok=False)
            evidence_ref = f"attempts/{name}/artifacts/evidence.json"
            evidence = {"schema_version": "1.0", "complete": False, "runtime_status": "running",
                        "errors": [{"category": "Evaluation Infrastructure", "message": "attempt started"}]}
            stub = build_result(manifest, case, attempt_id, evidence, evidence_ref, score_task(case, evidence))
            write_json(result_path, stub)
            append_jsonl(run / "attempts.jsonl", {"case_id": case["case_id"], "attempt_id": attempt_id,
                         "started_at": datetime.now(timezone.utc).isoformat(),
                         "result_ref": result_path.relative_to(run).as_posix()})
            try:
                evidence = launch_worker(worker_input(case), result_path.parent / "artifacts",
                                        variant=config["variant"], timeout_seconds=effective["timeout_seconds"],
                                        source_root=run / "frozen" / "sources" / "work", worker_settings=effective,
                                        trace={"trace_id": uuid.uuid4().hex, "experiment_id": manifest["experiment_id"],
                                               "episode_id": case["case_id"], "attempt_id": attempt_id})
            except Exception as exc:
                evidence = {"schema_version": "1.0", "complete": False, "runtime_status": "failed",
                            "errors": [{"category": "Evaluation Infrastructure", "stage": "runner",
                                        "message": type(exc).__name__}]}
            except KeyboardInterrupt:
                evidence = {"schema_version": "1.0", "complete": False, "runtime_status": "cancelled",
                            "errors": [{"category": "Evaluation Infrastructure", "stage": "runner",
                                        "message": "interrupted"}]}
            artifact_path = run / evidence_ref
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(artifact_path, evidence)
            score = score_task(case, evidence)
            result = build_result(manifest, case, attempt_id, evidence, evidence_ref, score)
            result["evidence_sha256"] = file_hash(artifact_path)
            write_json(result_path, result)
            append_jsonl(run / "results.jsonl", result)
            for item in result["scores"]:
                append_jsonl(run / "scores.jsonl", {"case_id": case["case_id"], "attempt_id": attempt_id, **item})
            for item in result["errors"]:
                append_jsonl(run / "errors.jsonl", {"case_id": case["case_id"], "attempt_id": attempt_id, **item})
            results.append(result)
            evidences.append(evidence)
            if evidence["runtime_status"] == "cancelled" and not evidence["complete"]:
                stopped = True
                stop_reason = "interrupted"
                break
        if stopped:
            break
    manifest.update(ended_at=datetime.now(timezone.utc).isoformat(), started_attempts=len(results),
                    stop_reason=stop_reason, reserved_model_call_allowances=reserved_calls)
    write_json(run / "manifest.json", manifest)
    from .analysis import summarize_analysis
    write_json(run / "analysis.json", summarize_analysis(results, evidences))
    return write_report(run, manifest, results)


def regrade(run, destination=None):
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    verify_frozen(run, manifest)
    # Ground truth stays frozen, and validation/scoring uses the original source.
    dataset_module, scorer_module = frozen_modules(run)
    score_task = scorer_module.score_task
    cases = {case["case_id"]: case for case in dataset_module.load_dataset(run / "frozen" / "dataset.jsonl")}
    results = []
    for result_path in sorted((run / "attempts").glob("*/result.json")):
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        case = cases[saved["case_id"]]
        evidence_path = run / saved["evidence_ref"]
        if not evidence_path.resolve().is_relative_to(run):
            raise ValueError("Evidence reference escapes experiment")
        if evidence_path.is_file() and saved.get("evidence_sha256"):
            if file_hash(evidence_path) != saved["evidence_sha256"]:
                raise ValueError("Saved evidence integrity mismatch")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        else:
            evidence = {"schema_version": "1.0", "complete": False, "runtime_status": saved["runtime_status"],
                        "errors": [{"category": "Evaluation Infrastructure", "message": "missing saved evidence"}]}
        result = build_result(manifest, case, saved["attempt_id"], evidence, saved["evidence_ref"],
                              score_task(case, evidence))
        result["evidence_sha256"] = saved.get("evidence_sha256")
        results.append(result)
    destination = Path(destination) if destination else run / ("regrade-" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    for result in results:
        for reference in ("evidence_ref", "trajectory_ref"):
            result[reference] = Path(os.path.relpath(run / result[reference], destination)).as_posix()
        append_jsonl(destination / "results.jsonl", result)
    report_manifest = {**manifest, "regraded_from": str(run),
                       "regraded_at": datetime.now(timezone.utc).isoformat()}
    write_json(destination / "manifest.json", report_manifest)
    summary = write_report(destination, report_manifest, results)
    return summary


def run_matrix(dataset, config_paths, output, *, seed=17, allow_local_model=False,
               max_model_calls=80, max_seconds=900):
    """One scheduler; shuffled config order and a conservative global call reservation."""
    configs = [(Path(path), load_config(path)) for path in config_paths]
    if (type(max_model_calls) is not int or max_model_calls < 1
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Matrix global request and wall-time budgets must be positive and finite")
    if len(configs) < 2 or type(seed) is not int:
        raise ValueError("Matrix requires at least two configurations and an integer seed")
    required_calls = sum(config.get("budget", {}).get("max_calls", 0) for _, config in configs)
    if required_calls > max_model_calls:
        raise ValueError("Matrix configuration call reservations exceed its global budget")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    random.Random(seed).shuffle(configs)
    plan = {"schema_version": "1.0", "seed": seed, "mode": "one_shared_runner",
            "max_model_calls": max_model_calls, "reserved_model_calls": required_calls,
            "max_seconds": max_seconds, "config_order": [config["name"] for _, config in configs],
            "limitations": ["Config order is randomized; variants are not interleaved within episodes.",
                            "Declared allowances remain reserved after timeout or missing telemetry."]}
    write_json(destination / "matrix.json", plan)
    started, runs = time.monotonic(), []
    for index, (path, config) in enumerate(configs):
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        if config["mode"] == "model_proposal":
            config["budget"]["max_seconds"] = min(config["budget"]["max_seconds"], remaining)
        config.setdefault("seed", seed)
        effective_path = destination / f"config_{index:02d}.json"
        write_json(effective_path, config)
        run = destination / f"run_{index:02d}"
        run_experiment(dataset, effective_path, run, allow_local_model=allow_local_model,
                       deadline=started + max_seconds)
        runs.append(run)
    plan.update(completed_runs=len(runs), elapsed_seconds=round(time.monotonic() - started, 3))
    write_json(destination / "matrix.json", plan)
    if len(runs) >= 2:
        from .research import compare_runs
        write_json(destination / "comparison.json", compare_runs(runs))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="Offline only; never calls a real model or Bash")
    execute.add_argument("--dataset", type=Path, default=HERE / "datasets" / "smoke_v1.jsonl")
    execute.add_argument("--config", type=Path, default=HERE / "experiments" / "baseline.json")
    execute.add_argument("--output", type=Path)
    execute.add_argument("--repeats", type=int)
    execute.add_argument("--allow-local-model", action="store_true", help="Explicit local qwen3:4b proposal calls only")
    replay = commands.add_parser("regrade", help="Read saved evidence with the original frozen scorer")
    replay.add_argument("--run", required=True, type=Path)
    replay.add_argument("--output", type=Path)
    inventory = commands.add_parser("inventory", help="Inspect existing development suites without runtime imports")
    frozen = commands.add_parser("freeze-suite", help="Freeze an existing suite for controller-only prepare/verify")
    frozen.add_argument("--suite", choices=["real_tasks", "memory_tasks"], required=True)
    frozen.add_argument("--output", type=Path, required=True)
    matrix = commands.add_parser("matrix", help="Run bounded configurations through the shared runner")
    matrix.add_argument("--dataset", type=Path, required=True)
    matrix.add_argument("--configs", type=Path, nargs="+", required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--seed", type=int, default=17)
    matrix.add_argument("--allow-local-model", action="store_true")
    compare = commands.add_parser("compare", help="Compare saved runs; no execution")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--metric", default="task_success")
    compare.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("research-plan", help="Save one explicit partial-effect hypothesis")
    plan.add_argument("--output", type=Path, required=True)
    template = commands.add_parser("calibration-template", help="Create empty human labels without inventing agreement")
    template.add_argument("--artifact-ids", nargs="+", required=True)
    template.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate", help="Analyze supplied double-human/arbitration labels")
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output = args.output or HERE.parent / ".test-results" / "evaluation" / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
            summary = run_experiment(args.dataset, args.config, output, repeats=args.repeats,
                                     allow_local_model=args.allow_local_model)
            print(f"Saved experiment: {output.resolve()}")
        elif args.command == "regrade":
            summary = regrade(args.run, args.output)
        elif args.command in {"compare", "research-plan", "calibration-template", "calibrate", "matrix"}:
            if args.command == "matrix":
                data = run_matrix(args.dataset, args.configs, args.output, seed=args.seed,
                                  allow_local_model=args.allow_local_model)
            elif args.command in {"compare", "research-plan"}:
                from .research import compare_runs, create_preregistration
                data = compare_runs(args.runs, metric=args.metric) if args.command == "compare" else create_preregistration()
            else:
                from .scorers.judge import calibration_template, load_calibration, summarize_calibration
                data = calibration_template(args.artifact_ids) if args.command == "calibration-template" else summarize_calibration(load_calibration(args.input))
            if args.command != "matrix":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if args.output.exists():
                    raise ValueError("Derived artifacts must not overwrite existing files")
                write_json(args.output, data)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        else:
            from .adapters.existing_suites import freeze_suite, suite_inventory
            data = (suite_inventory() if args.command == "inventory" else
                    freeze_suite(args.suite, args.output))
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if (summary["started_attempts"] == summary["planned_attempts"]
                     and summary["protocol_passes"] == summary["started_attempts"]) else 1
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
