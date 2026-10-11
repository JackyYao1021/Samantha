"""Freeze the code, public input, hidden oracle, and effective offline config."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .harness import write_json
from .dataset import freeze_jsonl
from .scorers.task import SCORER_VERSION


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def git_info(repository):
    def git(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=repository, capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(status) if status is not None else None,
            "status": status.splitlines() if status is not None else None}


def freeze_run(run, dataset_path, config, cases):
    run = Path(run)
    repository = Path(__file__).resolve().parents[2]
    frozen = run / "frozen"
    frozen.mkdir()
    freeze_jsonl(cases, frozen / "dataset.jsonl")
    write_json(frozen / "config.json", config)
    hashes = {}
    for directory in (repository / "work" / "src", repository / "work" / "evals"):
        for source in sorted(directory.rglob("*.py")):
            relative = source.relative_to(repository)
            target = frozen / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[relative.as_posix()] = file_hash(target)
    for name in ("work/requirements.txt",):
        target = frozen / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
        hashes[name] = file_hash(target)
    versions = {}
    for name in ("langgraph", "langgraph-checkpoint", "openai", "httpx", "python-dotenv", "pydantic", "inspect-ai", "langfuse"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    prompt_files = {name: digest for name, digest in hashes.items()
                    if name.rsplit("/", 1)[-1] in {"clarify_intent.py", "intent_explain.py", "n2c.py",
                                                 "command_review.py", "error_correction.py"}}
    manifest = {
        "schema_version": "1.0", "experiment_id": run.name,
        "started_at": datetime.now(timezone.utc).isoformat(), "timestamp_timezone": "UTC",
        "display_timezone": "Asia/Shanghai", "git": git_info(repository),
        "config": config, "config_hash": config_hash(config),
        "dataset": {"path": "frozen/dataset.jsonl", "sha256": file_hash(frozen / "dataset.jsonl"),
                    "input_sha256": file_hash(dataset_path),
                    "case_ids": [case["case_id"] for case in cases], "case_count": len(cases)},
        "source_hashes": hashes, "prompt_template_hashes": prompt_files,
        "scorer": {"version": SCORER_VERSION, "path": "work/evals/scorers/task.py",
                   "sha256": hashes["work/evals/scorers/task.py"]},
        "model": {"provider": "fake", "transport": "in_process_script", "model": None,
                  "digest": None, "effective_parameters": {}, "physical_requests_expected": 0},
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": versions},
        "budget": {"timeout_seconds": config["timeout_seconds"], "max_attempts": config["max_attempts"],
                   "repeats": config["repeats"], "model_calls": 0 if config["mode"] == "fake" else None, "api_cost_usd": None},
        "isolation": {"kind": "process_and_bounded_fake_executor", "network": "disabled_in_worker",
                      "arbitrary_shell": False, "oracle_passed_to_worker": False},
        "limitations": ["Offline Fake Services validation; not real model task accuracy.",
                        "Process isolation is not an arbitrary Bash security sandbox.",
                        "No human wait, cold model start, or content pre-index time is measured."],
    }
    write_json(run / "manifest.json", manifest)
    return manifest


def verify_frozen(run, manifest):
    run = Path(run).resolve()
    frozen = run / "frozen"
    for name, expected in manifest["source_hashes"].items():
        path = frozen / "sources" / name
        if not path.resolve().is_relative_to(frozen.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen source integrity mismatch: {name}")
    if file_hash(frozen / "dataset.jsonl") != manifest["dataset"]["sha256"]:
        raise ValueError("Frozen dataset integrity mismatch")
    config = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    if config_hash(config) != manifest["config_hash"]:
        raise ValueError("Frozen config integrity mismatch")
