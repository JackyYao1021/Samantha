"""Deterministic intent metrics and conservative, overlapping failure diagnostics.

Labels come from the fixture author, never from classifier predictions. Metrics
describe observable mismatches, not the causal reason the model made a mistake.
"""

from collections import Counter
from functools import lru_cache
import json


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _parameters(operation):
    # Descriptions of unsupported/unknown requests are free prose, not exact slots.
    return {name: value for name, value in operation.get("parameters", {}).items()
            if name != "description" and value.get("value") is not None}


def _pairs(gold, prediction):
    """Maximum-weight one-to-one slot matching within each operation label.

    IDs and array positions may differ without changing user intent. Repeated
    operations are not collapsed. The production contract bounds the arrays at 16.
    """
    pairs = []
    names = sorted({op["operation"] for op in gold} | {op["operation"] for op in prediction})
    for name in names:
        left = [op for op in gold if op["operation"] == name]
        right = [op for op in prediction if op["operation"] == name]
        if not left:
            pairs.extend((None, op) for op in right)
            continue
        if not right:
            pairs.extend((op, None) for op in left)
            continue
        if len(left) > 16 or len(right) > 16:
            raise ValueError("Intent metrics accept at most 16 operations of one type")

        def score(a, b):
            aa, bb = _parameters(a), _parameters(b)
            same = sum(key in bb and _value(value["value"]) == _value(bb[key]["value"])
                       for key, value in aa.items())
            source = sum(key in bb and value.get("source") == bb[key].get("source")
                         for key, value in aa.items())
            # Match cardinality first, then exact slot values, then provenance.
            return 100000 + 100 * same + source

        @lru_cache(None)
        def best(index, used):
            if index == len(left):
                return 0, ()
            result_score, result_path = best(index + 1, used)
            result = (result_score, (-1,) + result_path)
            for j, item in enumerate(right):
                if used & (1 << j):
                    continue
                tail_score, tail_path = best(index + 1, used | (1 << j))
                candidate = (score(left[index], item) + tail_score, (j,) + tail_path)
                if candidate[0] > result[0]:
                    result = candidate
            return result

        _, mapping = best(0, 0)
        used = set()
        for item, index in zip(left, mapping):
            pairs.append((item, right[index] if index >= 0 else None))
            if index >= 0:
                used.add(index)
        pairs.extend((None, item) for index, item in enumerate(right) if index not in used)
    return pairs


def _f1(rows, field):
    labels = sorted({value for gold, prediction in rows for value in gold + prediction})
    details = {}
    for label in labels:
        tp = sum(label in gold and label in prediction for gold, prediction in rows)
        fp = sum(label not in gold and label in prediction for gold, prediction in rows)
        fn = sum(label in gold and label not in prediction for gold, prediction in rows)
        details[label] = {"f1": _ratio(2 * tp, 2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
                          "support": sum(label in gold for gold, _ in rows)}
    return {field + "_macro_f1": (sum(d["f1"] for d in details.values()) / len(details) if details else None),
            field + "_per_label": details,
            field + "_macro_labels": labels}


def _constraints(intent):
    # Ordering is represented by depends_on; an additional prose/order constraint
    # is redundant under the v2 annotation policy and is not an extraction error.
    return Counter({(item["kind"], _value(item["value"])): 1 for item in intent.get("constraints", [])
                    if item["kind"] != "order"})


def _score(records, domains):
    domain_rows, operation_rows = [], []
    p_correct = p_gold = p_predicted = p_exact = source_correct = source_count = 0
    c_correct = c_gold = c_predicted = c_exact = 0
    multi_count = multi_complete = all_complete = support_correct = fallback_correct = 0
    unknown_count = unknown_correct = unknown_safe = actual_fallback = runtime_fallback = advised_fallback = 0
    confusions = Counter({"copy_as_move": 0, "move_as_copy": 0,
                         "append_as_overwrite": 0, "overwrite_as_append": 0})
    for record in records:
        gold = record["gold"]
        pred = record.get("prediction") or {}
        actual_fallback += bool(record.get("fallback_reason") or pred.get("fallback") in {"planner", "clarify"})
        runtime_fallback += bool(record.get("fallback_reason"))
        advised_fallback += pred.get("fallback") in {"planner", "clarify"}
        gg, pp = gold.get("operations", []), pred.get("operations", [])
        go, po = Counter(op["operation"] for op in gg), Counter(op["operation"] for op in pp)
        operation_rows.append((list(go), list(po)))
        if domains:
            domain_rows.append((gold.get("domains", []), pred.get("domains", [])))
        complete = not (go - po)
        all_complete += complete
        if sum(go.values()) > 1:
            multi_count += 1
            multi_complete += complete
        for a, b in (("copy", "move"), ("move", "copy"), ("append", "overwrite"), ("overwrite", "append")):
            # A mixed gold task requiring both is not automatically a confusion.
            if go[a] > po[a] and po[b] > go[b]:
                confusions[a + "_as_" + b] += min(go[a] - po[a], po[b] - go[b])
        exact = bool(record.get("prediction"))
        for a, b in _pairs(gg, pp):
            aa, bb = _parameters(a or {}), _parameters(b or {})
            p_gold += len(aa)
            p_predicted += len(bb)
            matched = {key for key in aa.keys() & bb.keys()
                       if _value(aa[key]["value"]) == _value(bb[key]["value"])}
            p_correct += len(matched)
            source_count += len(aa)
            source_correct += sum(aa[key].get("source") == bb[key].get("source") for key in matched)
            if a is None or b is None or len(matched) != len(aa) or len(matched) != len(bb):
                exact = False
        p_exact += exact
        gc, pc = _constraints(gold), _constraints(pred)
        c_correct += sum((gc & pc).values())
        c_gold += sum(gc.values())
        c_predicted += sum(pc.values())
        c_exact += bool(record.get("prediction")) and gc == pc
        support_correct += gold.get("support") == pred.get("support")
        fallback_correct += gold.get("fallback") == pred.get("fallback")
        if "unknown" in go or gold.get("support") == "unknown":
            unknown_count += 1
            unknown_correct += "unknown" in po and pred.get("support") == "unknown"
            unknown_safe += bool(record.get("fallback_reason") or pred.get("fallback") in {"planner", "clarify"})
    count = len(records)
    result = {
        "samples": count, "invalid_or_missing_predictions": sum(r.get("prediction") is None for r in records),
        "operation_completeness": _ratio(all_complete, count),
        "multi_intent_samples": multi_count, "multi_intent_completeness": _ratio(multi_complete, multi_count),
        "parameters": {"precision": _ratio(p_correct, p_predicted), "recall": _ratio(p_correct, p_gold),
                       "exact_match_rate": _ratio(p_exact, count), "correct": p_correct,
                       "gold_slots": p_gold, "predicted_slots": p_predicted},
        "parameter_sources": {"accuracy": _ratio(source_correct, source_count),
                              "correct": source_correct, "gold_slots": source_count},
        "constraints": {"precision": _ratio(c_correct, c_predicted), "recall": _ratio(c_correct, c_gold),
                        "exact_match_rate": _ratio(c_exact, count), "correct": c_correct,
                        "gold_constraints": c_gold, "predicted_constraints": c_predicted},
        "support_accuracy": _ratio(support_correct, count), "fallback_accuracy": _ratio(fallback_correct, count),
        "combined_fallback_count": actual_fallback, "runtime_fallback_count": runtime_fallback,
        "advised_fallback_count": advised_fallback, "critical_confusions": dict(confusions),
        "unknown": {"samples": unknown_count, "recognition_accuracy": _ratio(unknown_correct, unknown_count),
                    "fallback_rate": _ratio(unknown_safe, unknown_count)},
        "macro_definition": "Unweighted label F1 over labels occurring in gold or prediction in this stage; absent-both labels excluded.",
        "slot_definition": "Exact typed values and canonical constraint sets; redundant order constraints and free description/evidence prose excluded; duplicate operations matched one-to-one.",
    }
    result.update(_f1(operation_rows, "operation"))
    result.update(_f1(domain_rows, "domain") if domains else {
        "domain_macro_f1": None, "domain_per_label": {}, "domain_macro_labels": []})
    return result


def classification_metrics(records):
    """Keep pre-tool (primary) and post-user-update classification scores separate."""
    result = {}
    for mode in ("A", "B", "C"):
        if mode == "A":
            result[mode] = {"applicable": False, "reason": "Baseline has no classifier; no inferred labels scored."}
        else:
            selected = [r for r in records if r["mode"] == mode]
            result[mode] = {"applicable": True,
                            **{stage: _score([r for r in selected if r.get("stage", "initial") == stage], mode == "C")
                               for stage in ("initial", "updated")}}
    return result


_TOOLS = {"create": {"write_file"}, "overwrite": {"write_file"}, "append": {"append_file"},
          "copy": {"copy_file"}, "move": {"move_file"}, "read": {"read_file"}, "list": {"list_files"},
          "mkdir": {"make_directory"}, "search_name": {"search_files"},
          "search_text": {"search_files", "search_content"}}
_OBSERVATION = {"list_files", "read_file", "search_files", "search_content"}
_ALL_TOOLS = set().union(*_TOOLS.values())


def failure_analysis(case, task, controls, filesystem_pass, protocol_pass):
    """Non-causal multi-label diagnostics from tool traces and independent outcomes.

    Extra observation tools remain legal. A wrong mutation or arguments can be
    flagged even when fixture approval prevented damage. Unknown causes stay
    unclassified; this function never interprets the model's final success claim.
    """
    labels = set()
    gold = case.get("after_clarification_gold") if task.get("clarifications") else None
    gold = gold or case.get("gold", {})
    ops = gold.get("operations", [])
    allowed_tools = _OBSERVATION | set().union(*(_TOOLS.get(op["operation"], set()) for op in ops))
    expected = case.get("allowed_actions", [])
    actions = task.get("actions", [])
    attempts = actions + [{"tool": observation.get("tool"), "args": observation.get("args", {})}
                           for observation in task.get("observations", []) if observation.get("ok") is False]
    attempts += [item for item in controls.get("policy_denials", []) if isinstance(item, dict)]
    reference = any(p.get("source") == "verified_context" for op in ops
                    for p in op.get("parameters", {}).values()) or bool(case.get("gold", {}).get("unresolved_references"))
    for action in attempts:
        name, args = action.get("tool"), action.get("args", {})
        if name not in allowed_tools or name not in _ALL_TOOLS:
            labels.add("tool_selection")
        matches = [a for a in expected if a["tool"] == name]
        if matches:
            # Registry preparation inserts default overwrite=false.
            def normalized(item):
                value = dict(item)
                if name == "write_file":
                    value.setdefault("overwrite", False)
                return value
            if all(normalized(args) != normalized(a["args"]) for a in matches):
                labels.add("parameters")
                if reference and any(key in args for key in ("source", "path")):
                    labels.add("reference")
    errors = [task.get("error", "")]
    errors.extend(str(r.get("exception_type", "")) for r in task.get("model_call_records", []) if r.get("status") == "failed")
    markers = ("connecterror", "connectionerror", "apiconnectionerror", "readtimeout", "connecttimeout",
               "apitimeouterror", "connection refused", "model not found", "no such model", "permissionerror",
               "sqlite", "disk full", "infrastructure")
    if any(marker in " ".join(errors).lower() for marker in markers):
        labels.add("infrastructure")
    if controls.get("harness_errors"):
        labels.add("infrastructure")
    if not filesystem_pass:
        completed = Counter(a.get("tool") for a in actions if a.get("status") == "verified")
        required = Counter(a["tool"] for a in expected)
        if any(completed[name] < count for name, count in required.items()):
            labels.add("omitted_operation")
    observed = {a.get("tool") for a in actions if a.get("status") in {"observed", "verified"}}
    if set(case.get("required_tools", [])) - observed:
        labels.add("omitted_operation")
    if not protocol_pass:
        labels.add("protocol")
    return sorted(labels)
