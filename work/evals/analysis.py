"""Read-only descriptive analysis of saved attempts and event trajectories."""

from collections import Counter

from .scorers.trajectory import TAXONOMY, earliest_material_error, score_recovery


def summarize_analysis(results, evidences=None):
    """Evidence may be a parallel list or a mapping keyed by (case_id, attempt_id)."""
    if not isinstance(results, list) or any(not isinstance(result, dict) for result in results):
        raise ValueError("Saved results must be a list of objects")
    if isinstance(evidences, list) and len(evidences) != len(results):
        raise ValueError("Parallel evidence and result lists must have the same length")
    if evidences is not None and not isinstance(evidences, (list, dict)):
        raise ValueError("Evidence must be a list or identity mapping")
    categories, failures, recoveries = Counter(), [], []
    for index, result in enumerate(results):
        identity = (result.get("case_id"), result.get("attempt_id"))
        evidence = evidences[index] if isinstance(evidences, list) else evidences.get(identity, evidences.get("/".join(str(value) for value in identity))) if isinstance(evidences, dict) else None
        saved = evidence if isinstance(evidence, dict) else {"complete": result.get("evaluation_status") != "incomplete",
                                                          "runtime_status": result.get("runtime_status"), "errors": result.get("errors", [])}
        error = earliest_material_error(saved)
        if error is None and result.get("evaluation_status") in {"failed", "incomplete"}:
            error = {"category": "Evaluation Infrastructure" if result.get("evaluation_status") == "incomplete" else "Unknown",
                     "stage": "evaluation", "message": "Outcome failed; no saved causal evidence", "confidence": "unknown",
                     "evidence": [result.get("evidence_ref", "result")]}
        if error is not None:
            categories[error["category"]] += 1
            failures.append({"case_id": identity[0], "attempt_id": identity[1], "expected_outcome_passed": result.get("evaluation_status") == "passed", **error})
        if evidence is not None:
            recovery = score_recovery(saved, evaluation_status=result.get("evaluation_status"))
            if recovery["eligible"]:
                recoveries.append({"case_id": identity[0], "attempt_id": identity[1], **recovery})
    return {"schema_version": "1.0", "started_attempts": len(results),
            "episode_count": len({result.get("episode_id", result.get("case_id")) for result in results}),
            "taxonomy": list(TAXONOMY), "primary_error_counts": dict(sorted(categories.items())), "errors": failures,
            "recovery_eligible_attempts": len(recoveries), "recovery": recoveries,
            "limitations": ["The earliest observed error stage is not a proved causal explanation.",
                            "Expected safe failure can pass its protocol while retaining the underlying execution error.",
                            "Attempt counts and repeated episodes are reported separately."]}
