"""Prepare fixed judge requests and analyze human labels; never call a model."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re

from ..dataset import canonical_json, read_json
from ..statistics import wilson_interval

VERSION = "saved-judge-v1"
CALIBRATION_FIELDS = ("artifact_id", "judge_label", "human_a", "human_b", "adjudicated_label", "evidence",
                      "human_a_id", "human_b_id", "rubric_id", "rubric_version", "judge_config_hash")


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prepare_judge_request(artifact, rubric, model, parameters):
    """Freeze a controller-side request from a saved artifact; no transport exists."""
    artifact_ref = None
    if isinstance(artifact, (str, Path)):
        path = Path(artifact)
        artifact_ref = str(path)
        artifact = read_json(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("Judge input must be a saved JSON artifact")
    if (not isinstance(rubric, dict) or not isinstance(rubric.get("id"), str) or not rubric["id"]
            or not isinstance(rubric.get("version"), str) or not rubric["version"]
            or not isinstance(rubric.get("criteria"), list) or not rubric["criteria"]):
        raise ValueError("Judge rubric needs an id, version and explicit criteria")
    if not isinstance(model, str) or not model.strip() or not isinstance(parameters, dict):
        raise ValueError("Judge model and effective parameter snapshot are required")
    for field in ("temperature", "top_p", "max_tokens"):
        if field not in parameters:
            raise ValueError("Missing fixed judge parameter: " + field)
    if (type(parameters["temperature"]) not in (int, float) or parameters["temperature"] < 0
            or type(parameters["top_p"]) not in (int, float) or not 0 < parameters["top_p"] <= 1
            or type(parameters["max_tokens"]) is not int or parameters["max_tokens"] < 1):
        raise ValueError("Invalid fixed judge generation parameters")
    config = {"rubric": deepcopy(rubric), "model": model, "parameters": deepcopy(parameters)}
    request = {"schema_version": "1.0", "judge_version": VERSION, "artifact_sha256": _digest(artifact),
               "artifact_ref": artifact_ref, "artifact": deepcopy(artifact), "judge_config": config,
               "judge_config_hash": _digest(config), "status": "prepared_not_called",
               "judge_calibrated": False, "execution_authorized": False}
    request["request_hash"] = _digest(request)
    return request


def score_saved_judge(response, request):
    """Import an explicitly saved verdict as a separate uncalibrated judge metric."""
    if not isinstance(request, dict) or request.get("judge_version") != VERSION:
        raise ValueError("Unsupported fixed judge request")
    frozen = {key: value for key, value in request.items() if key != "request_hash"}
    if _digest(frozen) != request.get("request_hash"):
        raise ValueError("Fixed judge request integrity mismatch")
    def saved_reference_exists(reference):
        if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*", reference):
            return False
        value = request["artifact"]
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\[\d+\]", reference):
            if token.startswith("["):
                index = int(token[1:-1])
                if not isinstance(value, list) or index >= len(value):
                    return False
                value = value[index]
            else:
                if not isinstance(value, dict) or token not in value:
                    return False
                value = value[token]
        return True
    if (not isinstance(response, dict) or response.get("request_hash") != request["request_hash"]
            or type(response.get("passed")) is not bool or not isinstance(response.get("evidence_refs"), list)
            or not response["evidence_refs"] or any(not isinstance(value, str) or not value for value in response["evidence_refs"])
            or any(not saved_reference_exists(value) for value in response["evidence_refs"])
            or not isinstance(response.get("rationale"), str) or not response["rationale"].strip()):
        return {"scorer_version": VERSION, "metric": "judge_pass", "value": None, "evaluation_status": "incomplete",
                "method": "saved_judge", "judge_calibrated": False, "reason": "Missing bound saved verdict/evidence"}
    return {"scorer_version": VERSION, "metric": "judge_pass", "value": int(response["passed"]),
            "evaluation_status": "scored", "method": "saved_judge", "judge_calibrated": False,
            "judge_config_hash": request["judge_config_hash"], "artifact_sha256": request["artifact_sha256"],
            "evidence": deepcopy(response["evidence_refs"]), "rationale": response["rationale"],
            "limitations": ["Saved judge verdict is separate from rule-based task success; actual invocation is not verified here."]}


def _label(value):
    if value is None or value == "":
        return None
    if type(value) is bool:
        return int(value)
    if type(value) is int and value in {0, 1}:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "pass", "passed", "yes"}:
            return 1
        if lowered in {"0", "false", "fail", "failed", "no"}:
            return 0
        if not lowered:
            return None
    raise ValueError("Calibration labels must be explicit binary values or blank")


def validate_calibration(rows):
    if not isinstance(rows, list):
        raise ValueError("Calibration data must be rows")
    validated, identities = [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("artifact_id"), str) or not row["artifact_id"].strip():
            raise ValueError("Calibration rows need a nonempty artifact_id")
        if row["artifact_id"] in identities:
            raise ValueError("Duplicate calibration artifact identity")
        identities.add(row["artifact_id"])
        data = deepcopy(row)
        for field in ("judge_label", "human_a", "human_b", "adjudicated_label"):
            data[field] = _label(row.get(field))
        for field in ("evidence", "human_a_id", "human_b_id", "rubric_id", "rubric_version", "judge_config_hash"):
            value = data.get(field, "")
            if not isinstance(value, str):
                raise ValueError("Calibration metadata must be text")
            data[field] = value
        if data["human_a_id"] and data["human_a_id"] == data["human_b_id"]:
            raise ValueError("Two human reviewers must have distinct identities")
        validated.append(data)
    return validated


def load_calibration(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".jsonl":
        rows = [read_json(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        rows = read_json(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict):
            rows = rows.get("rows")
    return validate_calibration(rows)


def calibration_template(artifact_ids):
    if not isinstance(artifact_ids, list) or len(artifact_ids) != len(set(artifact_ids)) or any(not isinstance(value, str) or not value for value in artifact_ids):
        raise ValueError("Template needs unique saved artifact ids")
    return [{field: artifact_id if field == "artifact_id" else "" for field in CALIBRATION_FIELDS} for artifact_id in artifact_ids]


def summarize_calibration(rows, *, min_samples=50):
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be positive")
    rows = validate_calibration(rows)
    paired = [row for row in rows if row["human_a"] is not None and row["human_b"] is not None]
    agreements = sum(row["human_a"] == row["human_b"] for row in paired)
    observed = agreements / len(paired) if paired else None
    pa = sum(row["human_a"] for row in paired) / len(paired) if paired else None
    pb = sum(row["human_b"] for row in paired) / len(paired) if paired else None
    expected = pa * pb + (1 - pa) * (1 - pb) if paired else None
    kappa = (observed - expected) / (1 - expected) if expected is not None and expected < 1 else None
    eligible, unresolved = [], []
    for row in rows:
        if row not in paired or row["judge_label"] is None or not row["evidence"].strip():
            continue
        truth = row["adjudicated_label"] if row["adjudicated_label"] is not None else row["human_a"] if row["human_a"] == row["human_b"] else None
        if truth is None:
            unresolved.append(row["artifact_id"])
        else:
            eligible.append((row, truth))
    matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row, truth in eligible:
        key = "tp" if row["judge_label"] and truth else "fp" if row["judge_label"] else "fn" if truth else "tn"
        matrix[key] += 1
    tp, tn, fp, fn = (matrix[key] for key in ("tp", "tn", "fp", "fn"))
    count = len(eligible)
    configurations = {(row["rubric_id"], row["rubric_version"], row["judge_config_hash"]) for row, _ in eligible}
    fixed_config = len(configurations) == 1 and all(configurations.pop()) if configurations else False
    verified_reviewers = all(row["human_a_id"] and row["human_b_id"] for row, _ in eligible) if eligible else False
    exploratory = count < max(50, min_samples)
    return {"schema_version": "1.0", "judge_version": VERSION, "labels_sha256": _digest(rows),
            "judge_scorer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows": len(rows), "eligible_labeled_artifacts": count,
            "missing_or_unusable_rows": len(rows) - count, "unresolved_disagreements": unresolved,
            "human_agreement": observed, "human_agreement_ci": wilson_interval(agreements, len(paired)), "cohen_kappa": kappa,
            "confusion_matrix": matrix, "accuracy": (tp + tn) / count if count else None,
            "accuracy_ci": wilson_interval(tp + tn, count), "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "false_positive_rate_ci": wilson_interval(fp, fp + tn), "false_negative_rate": fn / (fn + tp) if fn + tp else None,
            "false_negative_rate_ci": wilson_interval(fn, fn + tp), "exploratory": exploratory,
            "fixed_judge_configuration_verified": False, "uniform_judge_configuration_supplied": bool(fixed_config),
            "reviewer_identities_supplied": bool(verified_reviewers),
            "calibration_status": "no_usable_labels" if not count else "exploratory" if exploratory else "ready_for_human_review",
            "judge_validated": False,
            "limitations": ["No labels or model verdicts are generated by this module.",
                            "Fifty labeled artifacts only enables review; sample size alone does not validate a judge.",
                            "Human disagreements require explicit adjudication; intervals use labeled artifacts as independent units.",
                            "Citation existence is not semantic support; rubric/model/parameters must remain fixed."]}
