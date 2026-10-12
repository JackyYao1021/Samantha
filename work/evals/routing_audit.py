"""Saved-only integrity audit and frozen-rule regrading of routing experiments."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

from .harness import write_json
from .manifest import file_hash
from .statistics import percentile


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _inside(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Saved evidence path escapes the experiment")
    return path


def verify_sources(run, manifest):
    frozen = run / "frozen"
    for name, digest in manifest["frozen_source_hashes"].items():
        path = _inside(frozen / "sources", name)
        if file_hash(path) != digest:
            raise ValueError("Frozen source integrity mismatch: " + name)
    if file_hash(frozen / "dataset.jsonl") != manifest["dataset"]["sha256"]:
        raise ValueError("Frozen dataset integrity mismatch")
    if json.loads((frozen / "config.json").read_text(encoding="utf-8")) != manifest["config"]:
        raise ValueError("Frozen configuration integrity mismatch")
    if file_hash(frozen / "prompts.json") != manifest["prompt_sha256"]:
        raise ValueError("Frozen prompt integrity mismatch")


def _frozen_modules(run):
    package_path = run / "frozen" / "sources" / "work" / "evals"
    name = "_routing_frozen_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, package_path / "__init__.py",
                submodule_search_locations=[str(package_path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return (importlib.import_module(name + ".routing_tasks"), importlib.import_module(name + ".routing"),
            importlib.import_module(name + ".routing_runner"))


def _original_body(payload, event, tasks, frozen_runner):
    """Reconstruct v1 field order from its frozen request builder.

Older request.json artifacts sort keys for readability. Their original SHA
binds the transport body instead; do not change those historical digests.
New clients also persist request_body.bin, so no reconstruction is needed.
"""
    if event["call_type"] == "embedding":
        ordered = {"model": payload["model"], "input": payload["input"], "truncate": payload["truncate"],
                   "keep_alive": payload["keep_alive"]}
    else:
        schema = frozen_runner.CLASSIFIER_SCHEMA if event["role"] == "model_router" else tasks.OUTPUT_SCHEMA
        if payload["format"] != schema:
            raise ValueError("Saved schema does not match its frozen request role")
        ordered = {"model": payload["model"],
            "messages": [{"role": item["role"], "content": item["content"]} for item in payload["messages"]],
            "stream": payload["stream"], "format": schema,
            "options": {key: payload["options"][key] for key in ("temperature", "seed", "num_ctx", "num_predict")},
            "keep_alive": payload["keep_alive"]}
        if "think" in payload:
            ordered["think"] = payload["think"]
    return json.dumps(ordered, ensure_ascii=False).encode("utf-8")


def _phase(calls):
    result = {"physical_requests": len(calls), "chat": sum(call["call_type"] == "chat" for call in calls),
              "embedding": sum(call["call_type"] == "embedding" for call in calls), "usage": {}}
    for field in ("input_tokens", "output_tokens"):
        known = [(call.get("usage") or {}).get(field) for call in calls]
        values = [value for value in known if type(value) is int and value >= 0]
        result["usage"][field] = {"total": sum(values) if calls and len(values) == len(calls) else None,
                                 "reported_sum": sum(values), "reported_calls": len(values), "calls": len(calls)}
    for field in ("duration_ms", "provider_load_ms", "provider_compute_ms"):
        values = [call[field] for call in calls if type(call.get(field)) in (int, float)]
        result[field] = {"reported_calls": len(values), "total": sum(values) if calls and len(values) == len(calls) else None,
                         "known_sum": sum(values), "p50": percentile(values, .5), "p95": percentile(values, .95)}
    return result


def audit(run, output):
    run = Path(run).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    verify_sources(run, manifest)
    tasks, routing, frozen_runner = _frozen_modules(run)
    cases = {case["case_id"]: case for case in tasks.load_cases(run / "frozen" / "dataset.jsonl")}
    records = _jsonl(run / "results.jsonl")
    submitted = [event for event in _jsonl(run / "requests.jsonl") if event["event"] == "submitted"]
    identifiers, evidence_calls, regrades = set(), [], []
    for row in records:
        if "artifact_ref" in row:
            saved = json.loads(_inside(run, row["artifact_ref"]).read_text(encoding="utf-8"))
            if saved != {key: value for key, value in row.items() if key != "artifact_ref"}:
                raise ValueError("Result journal does not match the saved attempt artifact")
        for call in row["calls"]:
            if call["call_id"] in identifiers:
                raise ValueError("Physical model request appears in multiple attempts")
            identifiers.add(call["call_id"])
            evidence_calls.append(call)
        if row["complete"] and "response" in row:
            rescored = tasks.evaluate_response(cases[row["case_id"]], row["response"])
            regrades.append({"case_id": row["case_id"], "policy": row["policy"], "repeat": row["repeat"],
                            "original_success": row["task_success"], "regraded_success": rescored["task_success"],
                            "same": row["task_success"] == rescored["task_success"]})
    setup_calls = manifest.get("policy_setup_calls", [])
    for call in setup_calls:
        if call["call_id"] in identifiers:
            raise ValueError("Setup request is also attributed to an attempt")
        identifiers.add(call["call_id"])
        evidence_calls.append(call)
    # Per-request files and the submitted journal independently retain the
    # transport ledger; missing completion records cannot erase submissions.
    telemetry, output_diagnostics, same_request_outcomes = [], [], {}
    for event in submitted:
        request_number = event["request_number"]
        directory = run / "requests" / f"request_{request_number:04d}"
        payload = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        raw_path = directory / "request_body.bin"
        original = raw_path.read_bytes() if raw_path.is_file() else _original_body(payload, event, tasks, frozen_runner)
        if json.loads(original) != payload or hashlib.sha256(original).hexdigest() != event["request_sha256"]:
            raise ValueError("Submitted request bytes do not match its saved payload")
        saved = json.loads((directory / "telemetry.json").read_text(encoding="utf-8"))
        telemetry.append(saved)
        if saved["call_id"] not in identifiers:
            raise ValueError("Submitted request telemetry is missing from attempt/setup evidence")
        # Strict parser diagnostics do not rewrite the historical scores.
        response_path = directory / "response.json"
        if response_path.is_file():
            raw = json.loads(response_path.read_text(encoding="utf-8"))
            content = (raw.get("message") or {}).get("content")
            if isinstance(content, str):
                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("Duplicate model-output key: " + key)
                        result[key] = value
                    return result
                try:
                    json.loads(content, object_pairs_hook=unique)
                except ValueError as exc:
                    output_diagnostics.append({"request_number": request_number,
                        "kind": "duplicate_keys" if "Duplicate" in str(exc) else "invalid_json",
                        "message": str(exc)})
                if (event["role"] == "solver" and raw.get("done") is True
                        and raw.get("done_reason") != "length"):
                    case_id = (saved.get("trace") or {}).get("episode_id")
                    if case_id in cases and cases[case_id]["split"] == "heldout":
                        try:
                            candidate = json.loads(content, object_pairs_hook=unique)
                            observed = tasks.evaluate_response(cases[case_id], candidate)["task_success"]
                        except (ValueError, TypeError):
                            observed = 0
                        group = same_request_outcomes.setdefault(event["request_sha256"], {
                            "case_id": case_id, "model": event["model"], "request_sha256": event["request_sha256"], "outcomes": []})
                        group["outcomes"].append({"request_number": request_number, "task_success": observed})
    development = [row for row in records if row["split"] == "development"]
    reconstructed = deepcopy(development)
    seen = {(row["case_id"], row["model"]) for row in development}
    for case_id in manifest["dataset"]["case_ids"]:
        case = cases[case_id]
        if case["split"] != "development":
            continue
        for model in manifest["config"]["models"]:
            if (case_id, model) not in seen:
                reconstructed.append({"case_id": case_id, "family": case["family"], "split": "development",
                    "model": model, "task_success": 0, "complete": False, "duration_ms": None,
                    "provider_compute_ms": None, "input_tokens": None, "output_tokens": None})
    capabilities = routing.build_capabilities(reconstructed, manifest["config"]["models"])
    selection = routing.choose_models(capabilities)
    if selection != manifest["selection"]:
        raise ValueError("Frozen model selection cannot be reconstructed from development evidence")
    phases = {"development": _phase([call for row in development for call in row["calls"]]),
              "prototype_setup": _phase(setup_calls),
              "heldout": _phase([call for row in records if row["split"] == "heldout" for call in row["calls"]])}
    submitted_ids = [call["call_id"] for call in telemetry]
    complete_ledger = (len(set(submitted_ids)) == len(submitted_ids) == len(identifiers)
                       == manifest["actual_physical_requests"])
    repeated = [group for group in same_request_outcomes.values() if len(group["outcomes"]) > 1]
    varying = [group for group in repeated if len({item["task_success"] for item in group["outcomes"]}) > 1]
    initial_models = {policy: dict(Counter(row.get("initial_model") for row in records
                     if row["split"] == "heldout" and row["policy"] == policy and row.get("initial_model")))
                      for policy in manifest["config"]["policies"] if not policy.startswith("fixed_")}
    routed_models = {model for counts in initial_models.values() for model in counts}
    result = {"schema_version": "routing-audit-v1", "original_run": str(run),
              "frozen_sources_verified": True, "development_selection_verified": True,
              "strict_output_duplicate_keys": sum(item["kind"] == "duplicate_keys" for item in output_diagnostics),
              "raw_output_diagnostics": output_diagnostics, "submitted_requests": len(submitted),
              "completion_telemetry": len(telemetry), "ledger_complete": complete_ledger,
              "frozen_rule_regrades": regrades, "regrade_agreement": sum(item["same"] for item in regrades),
              "regrade_count": len(regrades), "phases": phases, "api_cost_usd": None,
              "routing_initial_models": initial_models,
              "initial_model_routing_degenerate": len(routed_models) == 1,
              "degenerate_model": next(iter(routed_models)) if len(routed_models) == 1 else None,
              "same_request_outcome_variability": {"repeated_payload_groups": len(repeated),
                 "varying_groups": len(varying), "groups": repeated,
                 "interpretation": "Identical submitted payload hashes across actual solver calls; observed outcome variability, not an independent controlled repeat experiment."},
              "limitations": ["Regrading replays only bounded Python fixture operations, with no model or Bash calls.",
                              "Hashes establish internal evidence consistency, not a signed external attestation.",
                              "Historical scores are not overwritten; failed and incomplete outputs remain in the primary report."]}
    write_json(output / "audit.json", result)
    lines = ["# 路由实验离线审计", "", f"原运行：`{run.name}`", "",
             f"冻结源码、数据与提示词校验通过；开发集选型可从原始结果重建。",
             f"物理请求账：提交 {len(submitted)}，完整遥测 {len(telemetry)}；一致性：{complete_ledger}。",
             f"原版本规则重评分：{result['regrade_agreement']}/{len(regrades)} 一致；未调用模型或 Bash。", "",
             "| 阶段 | chat | embedding | 全部请求 | 已知输入 Token | 已知输出 Token | 已知加载 ms | 已知计算 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, phase in phases.items():
        lines.append(f"| {name} | {phase['chat']} | {phase['embedding']} | {phase['physical_requests']} "
                     f"| {phase['usage']['input_tokens']['reported_sum']} | {phase['usage']['output_tokens']['reported_sum']} "
                     f"| {phase['provider_load_ms']['known_sum']:.1f} | {phase['provider_compute_ms']['known_sum']:.1f} |")
    lines += ["", "Token 列为已报告部分之和；完整总量与 coverage 见 audit.json，Embedding 输出未知不补零。",
              "端到端延迟包含加载、切换和路由；单轮不同 tokenizer 的 token 和计算时间不等于货币成本。",
              "首轮模型加载和 BGE 插入会改变冷热条件，小规模实验不能据此推断生产推理性能。", "",
              f"路由初始选择：`{json.dumps(initial_models, ensure_ascii=False)}`。",
              f"初始选型退化为单一模型：{result['initial_model_routing_degenerate']}；模型：`{result['degenerate_model']}`。",
              f"同一求解请求字节哈希重复组：{len(repeated)}，其中成功结果有波动：{len(varying)} 组。",
              "同参数重复来自实际策略调用，未另做独立重复实验；差异不能归因于智能选型收益。", ""]
    (output / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("submitted_requests", "ledger_complete", "regrade_agreement", "regrade_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
