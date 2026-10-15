"""Selective rule -> embedding -> LLM routing, with development-only calibration.

Classification never receives task gold or chooses an execution outcome. The
legacy single-stage controls remain unchanged. Similarities are not probabilities.
"""

import math
import json
from pathlib import Path
import re
import time

from .routing import FAMILIES, _rule_decision, embedding_route
from .routing_client import EvidenceWriteError
from model_budget import BudgetExceeded


CASCADE_VERSION = "routing-cascade-v1"


def load_prototype_examples(path):
    """Read explicitly labelled development-only semantic groups, never outcomes."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prototype corpus must be a nonempty list")
    seen, texts, groups = set(), set(), {}
    required = {"case_id", "group_id", "split", "family", "request"}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != required or row["split"] != "development"
                or row["family"] not in FAMILIES):
            raise ValueError("Prototype examples must be labelled development requests only")
        for key in ("case_id", "group_id"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row[key]):
                raise ValueError("Prototype identifiers must be bounded portable strings")
        if not isinstance(row["request"], str) or not row["request"].strip() or len(row["request"]) > 4096:
            raise ValueError("Prototype request must be bounded nonempty text")
        text = " ".join(row["request"].casefold().split())
        if row["case_id"] in seen or text in texts:
            raise ValueError("Duplicate prototype ID or request")
        if row["group_id"] in groups and groups[row["group_id"]] != row["family"]:
            raise ValueError("A semantic prototype group cannot have conflicting labels")
        seen.add(row["case_id"])
        texts.add(text)
        groups[row["group_id"]] = row["family"]
    return rows


def validate_settings(settings):
    keys = {"similarity_thresholds", "margin_thresholds", "min_precision", "min_accepted"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError("Cascade calibration settings must be explicit")
    for name, low, high in (("similarity_thresholds", -1, 1), ("margin_thresholds", 0, 2)):
        values = settings[name]
        if (not isinstance(values, list) or not values or len(values) > 32
                or any(type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high
                       for v in values) or len(set(values)) != len(values)):
            raise ValueError("Invalid cascade " + name)
        if name == "margin_thresholds" and min(values) <= 0:
            raise ValueError("Cascade margins must be positive so category ties abstain")
    precision = settings["min_precision"]
    if type(precision) not in (int, float) or not math.isfinite(precision) or not 0 < precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if type(settings["min_accepted"]) is not int or settings["min_accepted"] < 1:
        raise ValueError("min_accepted must be a positive integer")
    return settings


def rule_candidate(public):
    family, reason = _rule_decision(public, abstain=True)
    return {"stage": "rule", "status": "accepted" if family else "abstain",
            "family": family, "reason": reason}


def embedding_evidence(vector, prototypes):
    first = embedding_route(vector, prototypes, {}, "unmapped")
    if first["prototype_id"] is None:
        return {"candidate_family": None, "score": None, "margin": None,
                "prototype_id": None, "runner_up_family": None, "runner_up_score": None}
    other = embedding_route(vector, [p for p in prototypes if p["family"] != first["family"]], {}, "unmapped")
    return {"candidate_family": first["family"], "score": first["score"],
            "prototype_id": first["prototype_id"],
            "runner_up_family": other["family"] if other["prototype_id"] else None,
            "runner_up_score": other["score"],
            "margin": max(0., first["score"] - other["score"]) if other["score"] is not None else None}


def embedding_candidate(vector, prototypes, gate):
    evidence = embedding_evidence(vector, prototypes)
    accepted = (gate["enabled"] and evidence["score"] is not None and evidence["margin"] is not None
                and evidence["score"] >= gate["similarity_threshold"]
                and evidence["margin"] >= gate["margin_threshold"])
    return {"stage": "embedding", "status": "accepted" if accepted else "abstain",
            "family": evidence["candidate_family"] if accepted else None, **evidence,
            "reason": "Similarity and inter-category margin passed." if accepted
                      else "Insufficient similarity, category separation, or prototype coverage."}


def calibrate_embedding(prototypes, settings):
    """Fit thresholds by leave-one-case/group-out on development vectors only.

All dev cases calibrate the embedding classifier, including rule hits, since a
small dev set can have no rule abstentions. This is fitting evidence, not an
unbiased estimate of downstream cascade accuracy. Optional group_id excludes
translations/paraphrases together; v1 benchmark has no group metadata.
"""
    validate_settings(settings)
    rows = []
    if prototypes:
        # Validate every split, ID, family and vector before using any label.
        embedding_evidence(prototypes[0]["vector"], prototypes)
    for prototype in sorted(prototypes, key=lambda item: item["case_id"]):
        group = prototype.get("group_id", prototype["case_id"])
        others = [p for p in prototypes if p.get("group_id", p["case_id"]) != group]
        evidence = embedding_evidence(prototype["vector"], others)
        rows.append({"case_id": prototype["case_id"], "group_id": group,
                     "expected_family": prototype["family"], **evidence})
    candidates = []
    for threshold in sorted(settings["similarity_thresholds"]):
        for margin in sorted(settings["margin_thresholds"]):
            accepted = [r for r in rows if r["score"] is not None and r["margin"] is not None
                        and r["score"] >= threshold and r["margin"] >= margin]
            correct = sum(r["candidate_family"] == r["expected_family"] for r in accepted)
            precision = correct / len(accepted) if accepted else None
            candidates.append({"similarity_threshold": threshold, "margin_threshold": margin,
                               "accepted": len(accepted), "correct": correct, "precision": precision})
    eligible = [c for c in candidates if c["accepted"] >= settings["min_accepted"]
                and c["precision"] >= settings["min_precision"]]
    if eligible:
        chosen = max(eligible, key=lambda c: (c["accepted"], c["precision"],
                                             c["similarity_threshold"], c["margin_threshold"]))
        gate = {"enabled": True, **chosen}
    else:
        gate = {"enabled": False, "similarity_threshold": None, "margin_threshold": None,
                "accepted": 0, "correct": 0, "precision": None}
    return {"version": CASCADE_VERSION, "split": "development", "settings": settings,
            "method": "leave-one-group-out; absent group_id uses case_id",
            "selection_rule": "Maximize accepted dev cases subject to precision/minimum support; ties prefer precision, higher similarity, then larger margin. No eligible gate disables embedding acceptance.",
            "calibration_cases": len(rows), "gate": gate, "rows": rows, "candidates": candidates}


def cascade_route(public, *, selection, prototypes, calibration, embed, classify, trace):
    """Run stages lazily, retaining partial traces even on fatal budget stops.

embed/classify are bounded transport callbacks. Fatal admission/evidence errors
propagate, never triggering another request. Ordinary classifier errors use the
frozen default model; they do not masquerade as a multi_step classification.
"""
    if not calibration or calibration.get("version") != CASCADE_VERSION:
        raise ValueError("Cascade requires a frozen development calibration artifact")

    def attempt(stage, operation):
        started = time.monotonic()
        try:
            value = operation()
        except (BudgetExceeded, EvidenceWriteError, KeyboardInterrupt) as exc:
            trace.append({"stage": stage, "status": "stopped", "family": None,
                          "reason": type(exc).__name__, "duration_ms": (time.monotonic() - started) * 1000})
            raise
        except Exception as exc:
            value = {"stage": stage, "status": "error", "family": None,
                     "reason": type(exc).__name__ + ": " + str(exc)}
        value["duration_ms"] = (time.monotonic() - started) * 1000
        trace.append(value)
        return value

    def finish(value):
        family = value["family"]
        return {"family": family, "model": selection["family_models"].get(family, selection["economical_model"]),
                "stage": value["stage"], "reason": value["reason"], "default_used": family is None}

    value = attempt("rule", lambda: rule_candidate(public))
    if value["status"] == "accepted":
        return finish(value)
    if calibration["gate"]["enabled"] and prototypes:
        value = attempt("embedding", lambda: embedding_candidate(embed(public["request"]), prototypes, calibration["gate"]))
        if value["status"] == "accepted":
            return finish(value)
    else:
        trace.append({"stage": "embedding", "status": "skipped", "family": None, "duration_ms": 0.,
                      "reason": "No supported development gate or prototypes; escalate without an embedding call."})

    def model_decision():
        category = classify(public)
        if (not isinstance(category, dict) or set(category) != {"family"}
                or not isinstance(category["family"], str) or category["family"] not in (*FAMILIES, "unknown")):
            raise ValueError("Invalid cascade model-router family classification")
        family = category["family"]
        return {"stage": "llm", "status": "abstain" if family == "unknown" else "accepted",
                "family": None if family == "unknown" else family,
                "reason": "Structured model family classification." if family != "unknown" else "Model abstained."}

    value = attempt("llm", model_decision)
    if value["status"] == "accepted":
        return finish(value)
    value = {"stage": "default", "status": "default", "family": None, "duration_ms": 0.,
             "reason": "Classifier unavailable or uncertain; use the frozen declared default model."}
    trace.append(value)
    return finish(value)
