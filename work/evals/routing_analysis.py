"""Saved routing evidence only: held-out denominators and case-paired statistics.

Fixed-model results are diagnostic counterfactual evidence, never replacements
for an actual routed outcome. Token and compute counts are resource proxies;
this module does not measure monetary cost or host energy/resource use.
"""

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

from .statistics import paired_episode_comparison, percentile, wilson_interval


TOKEN_FIELDS = ("input_tokens", "output_tokens")
LIMITATIONS = [
    "仅汇总 heldout；dev 仅用于冻结选择规则，不进入报告成绩。",
    "未启动及不完整任务保留在计划成功率分母中，且不计为成功。",
    "延迟分位数仅使用已启动且有测量值的任务；缺失值不按零处理。",
    "配对 bootstrap 以 case 为独立单位，先平均同一 case 的匹配重复；小样本仅作探索。",
    "固定模型对照仅识别可观察的路由机会，不预测某次路由调用的真实反事实结果；模型输出存在波动。",
    "请求、token 和 provider compute 是观测资源代理；未测量实际 API 费用、显存、能耗或占用成本。",
    "不完整证据可能遗漏已发生的请求；已保存请求数和已知部分 token 不等于完整总量。",
]


def _number(value, name, *, integer=False):
    if value is None:
        return None
    accepted = (int,) if integer else (int, float)
    try:
        finite = type(value) in accepted and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite {'integer' if integer else 'number'} or null")
    return value


def _text(value, name, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _identity(row):
    return row["case_id"], row["policy"], row["repeat"]


def _validate(rows, *, planned=False):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Evidence must be a list of records")
    identities, case_metadata = set(), {}
    for row in rows:
        for field in ("case_id", "family", "policy"):
            _text(row.get(field), field)
        if row.get("split") not in {"dev", "development", "heldout"}:
            raise ValueError("split must be development/dev or heldout")
        if type(row.get("repeat")) is not int or row["repeat"] < 0:
            raise ValueError("repeat must be a nonnegative integer")
        key = _identity(row)
        if key in identities:
            raise ValueError("Duplicate case/policy/repeat evidence")
        identities.add(key)
        metadata = (row["family"], row["split"])
        if row["case_id"] in case_metadata and case_metadata[row["case_id"]] != metadata:
            raise ValueError("A case must retain one family and split")
        case_metadata[row["case_id"]] = metadata
        if planned:
            continue
        if type(row.get("complete")) is not bool or type(row.get("started", True)) is not bool:
            raise ValueError("complete and started must be booleans")
        if type(row.get("task_success")) is not int or row["task_success"] not in {0, 1}:
            raise ValueError("task_success must be 0 or 1")
        if not row.get("started", True) and (row["complete"] or row["task_success"]):
            raise ValueError("An unstarted task cannot be complete or successful")
        if type(row.get("fallback", False)) is not bool:
            raise ValueError("fallback must be a boolean")
        if row.get("policy_kind") not in {None, "fixed", "routed"}:
            raise ValueError("Unknown policy_kind")
        for field in ("model", "initial_model", "selected_model", "fallback_reason", "error_category"):
            _text(row.get(field), field, optional=True)
        for field in ("duration_ms", "router_duration_ms", *TOKEN_FIELDS):
            _number(row.get(field), field, integer=field in TOKEN_FIELDS)
        calls = row.get("calls")
        if calls is not None and (not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls)):
            raise ValueError("calls must be a list or null")
        if not row.get("started", True) and calls:
            raise ValueError("An unstarted task cannot contain calls")
        for call in calls or []:
            _text(call.get("model"), "call.model")
            if call.get("call_type") not in {"chat", "embedding"}:
                raise ValueError("call_type must be chat or embedding")
            if type(call.get("physical_request", True)) is not bool:
                raise ValueError("physical_request must be a boolean")
            for field in ("duration_ms", "provider_compute_ms"):
                _number(call.get(field), "call." + field)
            usage = call.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError("call usage must be an object or null")
            for field in TOKEN_FIELDS:
                _number((usage or {}).get(field), "usage." + field, integer=True)
    return rows


def _heldout(rows):
    return [row for row in rows if row["split"] == "heldout"]


def _started(row):
    return row.get("started", True)


def _success(row):
    return int(_started(row) and row["complete"] and row["task_success"] == 1)


def _physical_calls(row):
    return [call for call in row.get("calls") or [] if call.get("physical_request", True)]


def _measurement(values, total):
    known = [value for value in values if value is not None]
    return {"p50": percentile(known, .5), "p95": percentile(known, .95),
            "mean": sum(known) / len(known) if known else None,
            "known_sum": sum(known) if known else None, "reported": len(known),
            "eligible": total, "coverage": len(known) / total if total else None}


def _usage(calls, *, evidence_complete):
    result, coverage = {}, {}
    for field in TOKEN_FIELDS:
        values = [(call.get("usage") or {}).get(field) for call in calls]
        known = [value for value in values if value is not None]
        result[field] = sum(known) if calls and len(known) == len(calls) and evidence_complete else None
        coverage[field] = {"reported_calls": len(known), "total_calls": len(calls),
                           "ratio": len(known) / len(calls) if calls else None,
                           "reported_sum": sum(known) if known else None}
    result["coverage"] = coverage
    result["evidence_complete"] = evidence_complete
    return result


def _group(records, planned):
    started = [row for row in records if _started(row)]
    complete = [row for row in started if row["complete"]]
    successes = sum(_success(row) for row in records)
    total = len(planned)
    calls = [call for row in started for call in _physical_calls(row)]
    evidence_complete = len(complete) == total and all(row.get("calls") is not None for row in complete)
    present = {_identity(row): row for row in records}
    errors = Counter()
    for item in planned:
        row = present.get(_identity(item))
        if row is None or not _started(row):
            errors["not_started"] += 1
        elif row.get("error_category"):
            errors[row["error_category"]] += 1
        elif not row["complete"]:
            errors["incomplete"] += 1
        elif not _success(row):
            errors["unknown_failure"] += 1
    fallback = [row for row in started if row.get("fallback", False)]
    compute = _measurement([call.get("provider_compute_ms") for call in calls], len(calls))
    compute["total_ms"] = compute["known_sum"] if calls and compute["reported"] == len(calls) and evidence_complete else None
    return {
        "planned": total, "started": len(started), "complete": len(complete),
        "not_started": total - len(started), "incomplete": len(started) - len(complete),
        "started_coverage": len(started) / total if total else None,
        "complete_coverage": len(complete) / total if total else None,
        "task_successes": successes, "task_success_rate": successes / total if total else None,
        "success_rate": successes / total if total else None,
        "success_interval": wilson_interval(successes, total),
        "latency_ms": _measurement([row.get("duration_ms") for row in started], len(started)),
        "router_overhead_ms": _measurement([row.get("router_duration_ms") for row in started], len(started)),
        "physical_calls": len(calls), "physical_llm_calls": sum(call["call_type"] == "chat" for call in calls),
        "physical_embedding_calls": sum(call["call_type"] == "embedding" for call in calls),
        "physical_call_count_complete": evidence_complete,
        "calls_by_model": dict(sorted(Counter(call["model"] for call in calls).items())),
        "usage": _usage(calls, evidence_complete=evidence_complete),
        "provider_compute_ms": compute, "api_cost_usd": None,
        "fallback": {"count": len(fallback), "rate": len(fallback) / len(started) if started else None,
                     "reason_counts": dict(sorted(Counter(row.get("fallback_reason") or "unknown" for row in fallback).items()))},
        "error_category_counts": dict(sorted(errors.items())),
    }


def summarize_results(records, planned, *, seed=17):
    """Keep every held-out planned attempt in the outcome denominator.

Records may include explicit started=False stubs. Missing records are also
reported as not started, without asserting that their resource use is known.
Top-level token aggregates are accepted but never replace call-level evidence.
"""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate(records)
    _validate(planned, planned=True)
    held_records, held_plan = _heldout(records), _heldout(planned)
    plan_map = {_identity(row): row for row in held_plan}
    for row in held_records:
        item = plan_map.get(_identity(row))
        if item is None:
            raise ValueError("Heldout evidence has no matching planned attempt")
        if row["family"] != item["family"]:
            raise ValueError("Saved and planned family do not match")
    policies = {}
    for policy in sorted({row["policy"] for row in held_plan}):
        policy_plan = [row for row in held_plan if row["policy"] == policy]
        policy_records = [row for row in held_records if row["policy"] == policy]
        group = _group(policy_records, policy_plan)
        group["families"] = {family: _group([row for row in policy_records if row["family"] == family],
                                           [row for row in policy_plan if row["family"] == family])
                             for family in sorted({row["family"] for row in policy_plan})}
        policies[policy] = group
    return {"schema_version": "1.0", "split": "heldout", "seed": seed,
            "ignored_dev_records": len(records) - len(held_records),
            "ignored_dev_planned": len(planned) - len(held_plan),
            "planned_cases": len({row["case_id"] for row in held_plan}),
            "policies": policies, "routing_diagnostics": routing_diagnostics(held_records),
            "api_cost_usd": None, "limitations": list(LIMITATIONS)}


def compare_policies(records, *, seed=17):
    """Pair the same case AND repeat before averaging repeats within each case."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate(records)
    held = _heldout(records)
    groups = {policy: {(row["case_id"], row["repeat"]): row for row in held if row["policy"] == policy}
              for policy in sorted({row["policy"] for row in held})}
    comparisons = []
    for left_name, right_name in combinations(groups, 2):
        left, right = groups[left_name], groups[right_name]
        matched = sorted(left.keys() & right.keys())
        def adapt(source):
            return [{"case_id": case, "episode_id": case, "attempt_id": repeat,
                     "scores": [{"metric": "task_success", "value": _success(source[(case, repeat)])}]}
                    for case, repeat in matched]
        result = paired_episode_comparison(adapt(left), adapt(right), seed=seed)
        left_only, right_only = sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())
        complete_pairs = [(case, repeat) for case, repeat in matched
                          if _started(left[(case, repeat)]) and _started(right[(case, repeat)])
                          and left[(case, repeat)]["complete"] and right[(case, repeat)]["complete"]]
        def identities(values):
            return [{"case_id": case, "repeat": repeat} for case, repeat in values]
        result.update(left_policy=left_name, right_policy=right_name,
                      paired_cases=result["paired_episodes"], matched_attempts=len(matched),
                      complete_matched_attempts=len(complete_pairs),
                      complete_pair_coverage=len(complete_pairs) / len(matched) if matched else None,
                      identical_attempt_coverage=not left_only and not right_only,
                      left_only_attempts=identities(left_only), right_only_attempts=identities(right_only),
                      complete_case_pairs=sum(all(pair in complete_pairs for pair in matched if pair[0] == case)
                                              for case in {pair[0] for pair in matched}),
                      sampling_unit="case; matched repeats averaged within case",
                      limitations=[LIMITATIONS[1], LIMITATIONS[3],
                                   "只有同 case/repeat 的结果配对；未匹配记录显式列出，不借用其他重复。",
                                   "此接口只能核对传入记录的匹配覆盖；双方都缺失的计划任务需结合 summary 判断。"])
        comparisons.append(result)
    return {"schema_version": "1.0", "split": "heldout", "seed": seed,
            "ignored_dev_records": len(records) - len(held), "comparisons": comparisons,
            "exploratory": True, "sampling_unit": "case", "limitations": [LIMITATIONS[3]]}


def _fixed(row):
    return row.get("policy_kind") == "fixed" or (row.get("policy_kind") is None and row["policy"].startswith("fixed"))


def _resource_proxy(row):
    calls = _physical_calls(row)
    if not _started(row) or not row["complete"] or not calls:
        return None
    if any(call.get("provider_compute_ms") is None or
           any((call.get("usage") or {}).get(field) is None for field in TOKEN_FIELDS) for call in calls):
        return None
    return {"provider_compute_ms": sum(call["provider_compute_ms"] for call in calls),
            "tokens": sum(call["usage"][field] for call in calls for field in TOKEN_FIELDS)}


def routing_diagnostics(records):
    """Identify fixed-model opportunities without changing actual route outcomes."""
    _validate(records)
    held = _heldout(records)
    fixed = defaultdict(lambda: defaultdict(list))
    model_names = set()
    for row in held:
        if _fixed(row):
            model = row.get("model") or row.get("selected_model") or row.get("initial_model")
            if model:
                model_names.add(model)
                fixed[(row["case_id"], row["repeat"])][model].append(row)
    items = []
    for row in held:
        if _fixed(row):
            continue
        initial, selected = row.get("initial_model"), row.get("selected_model")
        candidates = fixed[(row["case_id"], row["repeat"])]
        evidence = {model: values[0] for model, values in candidates.items() if len(values) == 1
                    and _started(values[0]) and values[0]["complete"]}
        item = {"case_id": row["case_id"], "repeat": row["repeat"], "policy": row["policy"],
                "initial_model": initial, "selected_model": selected,
                "actual_task_success": _success(row), "actual_recorded_task_success": row["task_success"],
                "actual_complete": row["complete"],
                "fixed_evidence": {model: {"policy": saved["policy"], "task_success": saved["task_success"]}
                                   for model, saved in sorted(evidence.items())},
                "classification": "unknown", "reason": "missing_complete_fixed_evidence",
                "resource_comparable": False}
        if not _started(row):
            item["reason"] = "route_not_started"
        elif len(model_names) != 2 or len(evidence) != 2 or initial not in evidence:
            item["reason"] = "missing_or_ambiguous_two_model_fixed_evidence"
        else:
            alternative = next(model for model in evidence if model != initial)
            initial_success, other_success = evidence[initial]["task_success"], evidence[alternative]["task_success"]
            item.update(evidence_complete=True, alternative_model=alternative)
            if initial_success == 0 and other_success == 1:
                item.update(classification="identifiable_misroute", reason="initial_fixed_failed_alternative_fixed_succeeded")
            elif initial_success == 0 and other_success == 0:
                item.update(classification="both_fixed_failed", reason="neither_fixed_model_succeeded")
            elif initial_success == 1 and other_success == 0:
                item.update(classification="supported_initial_choice", reason="initial_fixed_succeeded_alternative_fixed_failed")
            else:
                chosen = selected if selected in evidence else None
                item.update(classification="both_fixed_succeeded", reason="missing_selected_model_evidence")
                if chosen is None:
                    items.append(item)
                    continue
                other = next(model for model in evidence if model != chosen)
                chosen_proxy, other_proxy = _resource_proxy(evidence[chosen]), _resource_proxy(evidence[other])
                item.update(classification="both_fixed_succeeded", reason="resource_evidence_not_comparable")
                if chosen_proxy is not None and other_proxy is not None:
                    dominated = all(chosen_proxy[key] >= other_proxy[key] for key in chosen_proxy) and any(
                        chosen_proxy[key] > other_proxy[key] for key in chosen_proxy)
                    item.update(resource_comparable=True, selected_fixed_resource_proxy=chosen_proxy,
                                alternative_fixed_resource_proxy=other_proxy,
                                classification="avoidable_escalation" if dominated else "both_fixed_succeeded",
                                reason="selected_fixed_resource_proxy_dominated" if dominated else "no_observed_resource_dominance")
        item.setdefault("evidence_complete", False)
        items.append(item)
    def aggregate(values):
        known = sum(item["evidence_complete"] for item in values)
        return {"routed_attempts": len(values), "fixed_evidence_complete_attempts": known,
                "fixed_evidence_coverage": known / len(values) if values else None,
                "classification_counts": dict(sorted(Counter(item["classification"] for item in values).items())),
                "identifiable_misroutes": sum(item["classification"] == "identifiable_misroute" for item in values),
                "avoidable_escalations": sum(item["classification"] == "avoidable_escalation" for item in values),
                "unknown_attempts": sum(item["classification"] == "unknown" for item in values)}
    return {**aggregate(items), "candidate_models": sorted(model_names),
            "policies": {policy: aggregate([item for item in items if item["policy"] == policy])
                         for policy in sorted({item["policy"] for item in items})},
            "attempts": items, "limitations": [LIMITATIONS[4], LIMITATIONS[5],
                "avoidable_escalation 仅在固定调用的 compute 与输入/输出 token 均完整且资源代理满足 Pareto 支配时标记。"]}


def write_routing_report(output, manifest, summary, comparison):
    """Write a concise Chinese report from frozen metadata and saved summaries."""
    path = Path(output)
    if path.suffix.lower() != ".md":
        path = path / "routing_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    def display(value, *, ratio=False):
        if value is None:
            return "未知"
        if ratio:
            return f"{value:.1%}"
        return f"{value:.1f}" if type(value) is float else str(value)
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Samantha 本地双模型路由实验", "",
             f"实验：`{cell(manifest.get('experiment_id', manifest.get('name', '未命名')))}`；"
             f"策略版本：`{cell(manifest.get('policy_version', '未知'))}`。", "",
             f"报告仅包含 heldout 的 {summary.get('planned_cases', 0)} 个 case；"
             f"过滤 dev 记录 {summary.get('ignored_dev_records', 0)} 条。成功率分母包括所有计划任务。", "",
             "| 策略 | 成功/计划（Wilson 95%） | 启动/完整 | 完整覆盖 | p50/p95 ms | 路由 p50 ms | 实际请求 | 输入/输出 token | 降级 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    details = []
    for policy, group in summary.get("policies", {}).items():
        interval, latency, usage = group["success_interval"], group["latency_ms"], group["usage"]
        lines.append(f"| {cell(policy)} | {group['task_successes']}/{group['planned']} "
                     f"({display(group['task_success_rate'], ratio=True)}; {display(interval['low'], ratio=True)}–{display(interval['high'], ratio=True)}) "
                     f"| {group['started']}/{group['complete']} | {display(group['complete_coverage'], ratio=True)} "
                     f"| {display(latency['p50'])}/{display(latency['p95'])} "
                     f"| {display(group['router_overhead_ms']['p50'])} | {group['physical_calls']} "
                     f"| {display(usage['input_tokens'])}/{display(usage['output_tokens'])} | {group['fallback']['count']} |")
        details.extend(["", f"`{cell(policy)}` usage 覆盖率：输入 {display(usage['coverage']['input_tokens']['ratio'], ratio=True)}，"
                      f"输出 {display(usage['coverage']['output_tokens']['ratio'], ratio=True)}；"
                      f"已知部分输入/输出 {display(usage['coverage']['input_tokens']['reported_sum'])}/"
                      f"{display(usage['coverage']['output_tokens']['reported_sum'])}。"
                      f"错误分类：`{cell(json.dumps(group['error_category_counts'], ensure_ascii=False, sort_keys=True))}`。", ""])
    lines.extend(details)
    diagnostic = summary.get("routing_diagnostics", {})
    lines.extend(["错路由证据覆盖："
                  f"{diagnostic.get('fixed_evidence_complete_attempts', 0)}/{diagnostic.get('routed_attempts', 0)} "
                  f"({display(diagnostic.get('fixed_evidence_coverage'), ratio=True)})；"
                  f"可识别误选 {diagnostic.get('identifiable_misroutes', 0)}，"
                  f"可避免的资源升级 {diagnostic.get('avoidable_escalations', 0)}，未知 {diagnostic.get('unknown_attempts', 0)}。", "",
                  "| 配对策略（右减左） | matched case/repeat | 完整配对覆盖 | 成功率差 bootstrap 95% |",
                  "| --- | --- | --- | --- |"])
    for pair in comparison.get("comparisons", []):
        delta = pair["difference"]
        lines.append(f"| {cell(pair['left_policy'])} → {cell(pair['right_policy'])} "
                     f"| {pair['paired_cases']}/{pair['matched_attempts']} "
                     f"| {display(pair['complete_pair_coverage'], ratio=True)} "
                     f"| {display(delta['mean'], ratio=True)} [{display(delta['low'], ratio=True)}, {display(delta['high'], ratio=True)}] |")
    unmatched = [f"{pair['left_policy']} / {pair['right_policy']}: "
                 f"左独有 {len(pair['left_only_attempts'])}，右独有 {len(pair['right_only_attempts'])}"
                 for pair in comparison.get("comparisons", []) if not pair["identical_attempt_coverage"]]
    lines.extend(["", "配对覆盖差异：" + ("；".join(unmatched) if unmatched else "传入记录的 case/repeat 均匹配。"), "",
                  "冻结的选择规则：", "", "```json",
                  json.dumps(manifest.get("selection", {}), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
                  "| 模型 | 冻结 digest |", "| --- | --- |"])
    for model, details in sorted(manifest.get("models", {}).items()):
        digest = details.get("digest") if isinstance(details, dict) else None
        lines.append(f"| {cell(model)} | `{cell(digest or '未知')}` |")
    lines.extend(["", "API 成本为 null；本实验未测量真实资源费用。固定对照不覆盖实际路由输出。", ""])
    limitations = list(dict.fromkeys(summary.get("limitations", []) + manifest.get("limitations", []) + [
        "Wilson 区间为计划 attempt 的描述性区间；重复调用相关性由 case 配对 bootstrap 单独处理。",
        f"{summary.get('planned_cases', 0)} 个 heldout case 的结论仅作探索；样本、模型 digest 与冻结规则限定了适用范围。",
        "已保存物理请求数量可能是下界；token、compute 不完整时总量为未知。",
    ]))
    lines.extend("- " + str(note) for note in limitations)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
