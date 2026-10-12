"""Frozen, development-only model selection and explainable routing policies.

No provider requests or filesystem reads occur here. Public request/fixture data
may route a held-out task; its hidden label or score must never train the policy.
"""

import math
import re


POLICY_VERSION = "routing-policy-v1"
FAMILIES = ("write", "rename", "append", "select", "multi_step", "clarify")
_METRICS = ("duration_ms", "provider_compute_ms", "input_tokens", "output_tokens")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _number(value, label, *, tokens=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative finite number or None")
    if tokens and type(value) is not int:
        raise ValueError(f"{label} must be a nonnegative integer or None")
    try:
        valid = math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a nonnegative finite number or None")
    return value


def _statistics(records, case_ids):
    expected = len(case_ids)
    complete = sum(record["complete"] for record in records)
    successes = sum(record["task_success"] == 1 and record["complete"] for record in records)
    result = {
        "attempts": expected,
        "recorded_attempts": len(records),
        "complete_attempts": complete,
        "successes": successes,
        "success_rate": successes / expected if expected else None,
        "case_coverage": len(records) / expected if expected else None,
        "complete_coverage": complete / expected if expected else None,
        "missing_case_ids": sorted(set(case_ids) - {record["case_id"] for record in records}),
        "incomplete_success_claims": sum(record["task_success"] == 1 and not record["complete"]
                                         for record in records),
    }
    for metric in _METRICS:
        values = [record[metric] for record in records if record[metric] is not None]
        result[metric] = {
            "mean": math.fsum(values) / len(values) if values else None,
            "total": math.fsum(values) if values else None,
            "reported_attempts": len(values),
            "coverage": len(values) / expected if expected else None,
        }
    return result


def build_capabilities(records, models):
    """Summarize only development cases, counting missing/incomplete runs as failures.

Each (case_id, model) can occur once. The union of observed development case IDs
defines the shared denominator for every declared model. Budget stubs must be
included by the caller when no model produced any record for a case.
"""
    if not isinstance(records, (list, tuple)) or not isinstance(models, (list, tuple)):
        raise ValueError("records and models must be lists or tuples")
    names = [_text(model, "model") for model in models]
    if not names or len(set(names)) != len(names):
        raise ValueError("Declare at least one unique model")
    names = sorted(names)
    cases, seen = {}, set()
    validated = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each capability record must be an object")
        if record.get("split") != "development":
            raise ValueError("Capability selection permits development records only; heldout leakage rejected")
        case_id = _text(record.get("case_id"), "case_id")
        model = _text(record.get("model"), "record.model")
        family = record.get("family")
        if model not in names or family not in FAMILIES:
            raise ValueError("Record has an undeclared model or unsupported family")
        if type(record.get("complete")) is not bool:
            raise ValueError("complete must be a boolean")
        if type(record.get("task_success")) is not int or record["task_success"] not in (0, 1):
            raise ValueError("task_success must be integer 0 or 1")
        if (case_id, model) in seen:
            raise ValueError("Duplicate case/model records would bias the selection denominator")
        if case_id in cases and cases[case_id] != family:
            raise ValueError("A development case cannot have conflicting family labels")
        missing = set(_METRICS) - record.keys()
        if missing:
            raise ValueError("Missing metric fields must be explicitly represented as None")
        metrics = {metric: _number(record[metric], metric, tokens=metric.endswith("tokens"))
                   for metric in _METRICS}
        seen.add((case_id, model))
        cases[case_id] = family
        validated.append({**record, **metrics})
    case_ids = sorted(cases)
    return {
        "policy_version": POLICY_VERSION,
        "split": "development",
        "families": list(FAMILIES),
        "case_ids": case_ids,
        "models": {
            model: {
                "overall": _statistics([record for record in validated if record["model"] == model], case_ids),
                "families": {
                    family: _statistics(
                        [record for record in validated if record["model"] == model and record["family"] == family],
                        [case_id for case_id in case_ids if cases[case_id] == family])
                    for family in FAMILIES
                },
            }
            for model in names
        },
    }


def choose_models(capabilities):
    """Freeze models from development success and fully covered timing evidence.

Provider compute time is a local resource proxy, not monetary cost. Duration is
used only when no model has complete provider-compute coverage. Lexical model
IDs break remaining ties without assuming larger models are more capable.
"""
    if (not isinstance(capabilities, dict) or capabilities.get("split") != "development"
            or capabilities.get("policy_version") != POLICY_VERSION):
        raise ValueError("Selection requires this policy's development capability artifact")
    table = capabilities.get("models")
    if not isinstance(table, dict) or not table:
        raise ValueError("Capability artifact must declare models")
    names = sorted(_text(model, "model") for model in table)

    def success(model, family=None):
        stats = table[model]["overall"] if family is None else table[model]["families"][family]
        rate = stats["success_rate"]
        return -1.0 if rate is None else rate

    reference = min(names, key=lambda model: (-success(model), model))
    economical, metric_name = None, None
    for metric in ("provider_compute_ms", "duration_ms"):
        eligible = [model for model in names
                    if table[model]["overall"]["attempts"] > 0
                    and table[model]["overall"]["case_coverage"] == 1
                    and table[model]["overall"][metric]["coverage"] == 1
                    and table[model]["overall"][metric]["mean"] is not None]
        if eligible:
            economical = min(eligible, key=lambda model: (table[model]["overall"][metric]["mean"], model))
            metric_name = metric
            break
    if economical is None:
        economical = names[0]
        economics = ("Economic selection unsupported: no model has complete development timing coverage; "
                     "use the lexically first declared model as an explicit fallback.")
    else:
        economics = (f"Economical model minimizes development mean {metric_name} with complete case/metric coverage; "
                     "this timing proxy is not an API-price estimate.")
    family_models = {}
    for family in FAMILIES:
        family_models[family] = min(names, key=lambda model: (-success(model, family), model != economical, model))
    return {
        "reference_model": reference,
        "economical_model": economical,
        "family_models": family_models,
        "selection_rule": (
            "Development records only; missing/incomplete attempts remain unsuccessful in the shared case denominator. "
            "Reference maximizes overall success; each family maximizes success and prefers the economical model on ties. "
            "Remaining ties use lexical model IDs; model size never determines capability. " + economics),
    }


def _fixture_paths(public_input):
    # Only public fixture shapes are inspected; no generic recursion into labels.
    paths = set()
    files = public_input.get("files")
    if isinstance(files, dict):
        paths.update(path for path in files if isinstance(path, str))
    fixture = public_input.get("fixtures", public_input.get("fixture", []))
    if isinstance(fixture, list):
        paths.update(item["path"] for item in fixture if isinstance(item, dict)
                     and item.get("kind", "file") == "file" and isinstance(item.get("path"), str))
    elif isinstance(fixture, dict):
        if isinstance(fixture.get("files"), dict):
            paths.update(path for path in fixture["files"] if isinstance(path, str))
        elif isinstance(fixture.get("path"), str):
            paths.add(fixture["path"])
    return sorted(paths)


def _rule_decision(public_input):
    if not isinstance(public_input, dict):
        raise ValueError("Routing input must be a public-input object")
    request = _text(public_input.get("request"), "request")
    paths = _fixture_paths(public_input)
    # Quoted content is data: writing 'rename then append' is one write action.
    action_text = re.sub(r"`[^`]*`|'[^']*'|\"[^\"]*\"|“[^”]*”|‘[^’]*’", " <literal> ", request.lower())
    patterns = {
        "rename": r"\brenam(?:e|es|ed|ing)\b|重命名|改名",
        "append": r"\bappend(?:s|ed|ing)?\b|\badd\b.{0,30}\b(?:end|tail)\b|追加|末尾.{0,8}(?:添加|写入)",
        "select": r"\b(?:select|choose|pick|find|identify|search|filter)\b|\blook for\b|\breturn\b.{0,40}\bpaths\b.{0,60}\b(?:whose|matching)\b|选择|挑选|筛出|筛选|找出|查找|找到",
        "write": r"\b(?:write|create|save|overwrite|make)\b|创建|新建|写入|保存|覆盖",
    }
    matches = {family: list(re.finditer(pattern, action_text)) for family, pattern in patterns.items()}
    active = {family for family, values in matches.items() if values}
    # Chinese 'append/write at the end' denotes append, not two operations.
    if "append" in active and re.search(r"末尾.{0,8}写入|追加写入", action_text):
        active.discard("write")
    sequence = bool(re.search(r"\b(?:then|after that|afterwards|followed by|next)\b|然后|接着|随后|再(?:把|将|创建|写|追加|重命名)", action_text))
    repeated = any(len(values) > 1 for values in matches.values())
    if len(active) > 1 and (sequence or ("select" not in active and re.search(r"\band\b|并|及", action_text))):
        return "multi_step", "Multiple explicit operations with an ordering/conjunction cue; multi-step rule has priority."
    if sequence and repeated:
        return "multi_step", "Repeated explicit operations with an ordering cue; multi-step rule has priority."
    if re.search(r"\b(?:ask|clarify)\b.{0,35}\b(?:which|name|filename|file name)\b|请问|需要.{0,8}(?:澄清|确认)|未指定|没有指定", action_text):
        return "clarify", "The request explicitly asks to clarify missing file information."
    named = bool(re.search(r"[\w./ -]+\.[a-z0-9]{1,8}\b|\b(?:named|called|filename|file name)\b|名为|叫做|命名为", request.lower()))
    ambiguous = bool(re.search(r"\b(?:the|this|that|a|some) file\b|这个文件|那个文件|该文件|一个文件|文件里", action_text))
    mentioned = [path for path in paths if path.lower() in request.lower()]
    if ("select" not in active and ambiguous and not mentioned and not named
            and ((active == {"write"}) or len(paths) != 1)):
        return "clarify", "A file operation lacks a unique named target in the public request and fixtures."
    if active == {"rename"} and not re.search(r"\b(?:to|as)\b|重命名为|改名为|改成|改为", action_text):
        return "clarify", "Rename request omits an explicit destination name."
    for family in ("rename", "append", "select", "write"):
        if family in active:
            return family, f"Explicit {family} operation detected in the public request."
    return "multi_step", "No known operation rule matched; conservative multi_step fallback (unknown intent)."


def rule_family(public_input):
    """Classify only public request and fixture data; return one frozen family."""
    return _rule_decision(public_input)[0]


def rule_route(public_input, family_models, default_model):
    family, reason = _rule_decision(public_input)
    _text(default_model, "default_model")
    model = family_models.get(family, default_model)
    _text(model, "routed model")
    if family not in family_models:
        reason += " No model is configured for this family; use the declared default model."
    return {"family": family, "model": model, "score": None, "reason": reason, "prototype_id": None}


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty vector")
    vector = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain finite numbers")
        try:
            number = float(item)
        except (OverflowError, ValueError):
            raise ValueError(f"{label} must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain finite numbers")
        vector.append(number)
    # Scale first to avoid overflow with finite but very large components.
    scale = max(abs(item) for item in vector)
    if scale == 0:
        raise ValueError(f"{label} must have a nonzero norm")
    scaled = [item / scale for item in vector]
    norm = math.sqrt(math.fsum(item * item for item in scaled))
    return [item / norm for item in scaled]


def embedding_route(query_vector, prototypes, family_models, default_model):
    """Nearest development prototype by cosine; ties use lexical case IDs."""
    query = _vector(query_vector, "query_vector")
    _text(default_model, "default_model")
    if not isinstance(prototypes, (list, tuple)):
        raise ValueError("prototypes must be a list or tuple")
    seen, scored = set(), []
    for prototype in prototypes:
        if not isinstance(prototype, dict) or prototype.get("split") != "development":
            raise ValueError("Embedding prototypes must be development-only; heldout leakage rejected")
        case_id = _text(prototype.get("case_id"), "prototype.case_id")
        if case_id in seen:
            raise ValueError("Duplicate prototype case_id")
        family = prototype.get("family")
        if family not in FAMILIES:
            raise ValueError("Unsupported prototype family")
        vector = _vector(prototype.get("vector"), "prototype.vector")
        if len(query) != len(vector):
            raise ValueError("Query and prototype vectors must have the same dimension")
        seen.add(case_id)
        score = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(query, vector))))
        scored.append((score, case_id, family))
    if not scored:
        return {"family": "multi_step", "model": default_model, "score": None,
                "reason": "No development prototypes; use declared default model with unknown multi_step family.",
                "prototype_id": None}
    best_score = max(item[0] for item in scored)
    tied = [item for item in scored if math.isclose(item[0], best_score, rel_tol=0, abs_tol=1e-12)]
    score, case_id, family = min(tied, key=lambda item: item[1])
    model = family_models.get(family, default_model)
    _text(model, "routed model")
    reason = f"Nearest development prototype {case_id} by cosine similarity; lexical prototype ID breaks ties."
    if family not in family_models:
        reason += " Family model unavailable; use the declared default model."
    return {"family": family, "model": model, "score": score, "reason": reason, "prototype_id": case_id}
