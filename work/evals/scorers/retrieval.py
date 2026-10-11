"""Rule metrics for evidence retrieval, actual context, OCR and closed QA."""

import unicodedata

VERSION = "retrieval-rules-v1"


def _ids(values, label):
    if not isinstance(values, (list, tuple, set, frozenset)) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(label + " must contain nonempty string identities")
    return set(values)


def retrieval_metrics(candidates, relevant_evidence_ids, *, k=5, context_evidence_ids=None, relevant_session_ids=None):
    if type(k) is not int or k < 1 or not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
        raise ValueError("Candidates must be ranked objects and k must be positive")
    gold = _ids(relevant_evidence_ids, "relevant_evidence_ids")
    for item in candidates:
        for field in ("evidence_id", "session_id"):
            if field in item and (not isinstance(item[field], str) or not item[field]):
                raise ValueError("Candidate identity is malformed")
    # Ranking is the supplied ordered list. Duplicate evidence cannot inflate recall.
    selected = candidates[:k]
    found = {item["evidence_id"] for item in selected if "evidence_id" in item}
    hits = gold & found
    rank = next((index for index, item in enumerate(candidates, 1) if item.get("evidence_id") in gold), None)
    sessions = _ids(relevant_session_ids, "relevant_session_ids") if relevant_session_ids is not None else None
    found_sessions = {item["session_id"] for item in selected if "session_id" in item}
    context = _ids(context_evidence_ids, "context_evidence_ids") if context_evidence_ids is not None else None
    return {"scorer_version": VERSION, "k": k, "relevant_evidence_count": len(gold), "evidence_hits_at_k": sorted(hits),
            "evidence_recall_at_k": len(hits) / len(gold) if gold else None,
            "any_evidence_hit_at_k": int(bool(hits)) if gold else None,
            "all_evidence_hit_at_k": int(hits == gold) if gold else None,
            "reciprocal_rank": 1 / rank if rank else 0. if gold else None,
            "reciprocal_rank_at_k": 1 / rank if rank is not None and rank <= k else 0. if gold else None,
            "session_recall_at_k": len(sessions & found_sessions) / len(sessions) if sessions else None,
            "context_evidence_recall": len(gold & context) / len(gold) if gold and context is not None else None,
            "actual_context_evidence_ids": sorted(context) if context is not None else None,
            "eligible_for_evidence_metrics": bool(gold),
            "limitations": ["Evidence ids require corpus-version binding; session-id hits do not prove evidence retrieval.",
                            "Context recall is unavailable unless the actually packed evidence identities are saved."]}


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for index, lchar in enumerate(left, 1):
        current = [index]
        for column, rchar in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (lchar != rchar)))
        previous = current
    return previous[-1]


def ocr_cer(reference, hypothesis):
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("OCR reference and hypothesis must be saved text")
    distance = _edit_distance(reference, hypothesis)
    return {"scorer_version": VERSION, "method": "character_levenshtein", "edits": distance,
            "reference_characters": len(reference), "cer": distance / len(reference) if reference else 0. if not hypothesis else None,
            "normalization": "none; Unicode code points", "eligible": bool(reference) or not hypothesis}


def citation_metrics(citations, valid_evidence_ids, *, required_evidence_ids=None):
    valid = _ids(valid_evidence_ids, "valid_evidence_ids")
    cited = _ids(citations, "citations")
    required = _ids(required_evidence_ids, "required_evidence_ids") if required_evidence_ids is not None else None
    if required is not None and not required <= valid:
        raise ValueError("Required citation identities must belong to the frozen corpus")
    return {"scorer_version": VERSION, "citation_count": len(cited), "valid_citation_count": len(cited & valid),
            "citation_identity_precision": len(cited & valid) / len(cited) if cited else None,
            "required_citation_recall": len(cited & required) / len(required) if required else None,
            "invalid_citations": sorted(cited - valid), "semantic_grounding_verified": False,
            "limitations": ["Valid evidence identities do not prove that the cited text entails the answer."]}


def score_closed_qa(answer, accepted_answers, *, normalization="exact"):
    if not isinstance(answer, str) or not isinstance(accepted_answers, list) or not accepted_answers or any(not isinstance(item, str) for item in accepted_answers):
        raise ValueError("Closed QA requires an explicit nonempty frozen answer set")
    if normalization not in {"exact", "nfkc_casefold_whitespace"}:
        raise ValueError("Unknown closed-QA normalization")
    def normalize(text):
        return text if normalization == "exact" else " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return {"scorer_version": VERSION, "method": "closed_reference_rule", "normalization": normalization,
            "closed_qa_accuracy": int(normalize(answer) in {normalize(item) for item in accepted_answers}),
            "judge_used": False}
