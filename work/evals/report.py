"""Read-only summaries; no model calls, command execution, or hidden reasoning."""

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

from .harness import write_json


def metric(result, name):
    return next((score["value"] for score in result.get("scores", []) if score["metric"] == name), 0)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(manifest, results):
    started = len(results)
    planned = manifest["dataset"]["case_count"] * manifest["config"]["repeats"]
    passed = sum(result["evaluation_status"] == "passed" for result in results)
    verified = sum(result["evaluation_status"] in {"passed", "failed"} for result in results)
    succeeded = sum(metric(result, "task_success") == 1 for result in results)
    durations = [result["telemetry"]["end_to_end_duration_ms"] for result in results
                 if isinstance(result.get("telemetry", {}).get("end_to_end_duration_ms"), (int, float))]
    grouped = defaultdict(list)
    for result in results:
        grouped[result["case_id"]].append(result)
    stability = {key: {"started_repeats": len(values),
                       "protocol_pass_fraction": sum(v["evaluation_status"] == "passed" for v in values) / len(values),
                       "task_success_fraction": sum(metric(v, "task_success") for v in values) / len(values)}
                 for key, values in grouped.items()}
    verified_episodes = len({r["case_id"] for r in results if r["evaluation_status"] in {"passed", "failed"}})
    families = defaultdict(lambda: {"started": 0, "passed": 0, "task_successes": 0})
    for result in results:
        group = families[result.get("family", "Unknown")]
        group["started"] += 1
        group["passed"] += result["evaluation_status"] == "passed"
        group["task_successes"] += metric(result, "task_success")
    calls = [r.get("telemetry", {}) for r in results]
    token_values = {key: [item.get("usage", {}).get(key) for item in calls] for key in ("input_tokens", "output_tokens")}
    return {
        "schema_version": "1.0", "experiment_id": manifest["experiment_id"],
        "mode": manifest["config"].get("mode", "fake"), "planned_attempts": planned, "started_attempts": started,
        "verified_attempts": verified, "coverage": verified / planned if planned else None,
        "planned_episodes": manifest["dataset"]["case_count"], "started_episodes": len(grouped),
        "verified_episodes": verified_episodes,
        "episode_coverage": verified_episodes / manifest["dataset"]["case_count"],
        "protocol_passes": passed, "protocol_pass_rate": passed / started if started else None,
        "task_successes": succeeded, "attempted_task_success_rate": succeeded / started if started else None,
        "runtime_status_counts": dict(Counter(r["runtime_status"] for r in results)),
        "evaluation_status_counts": dict(Counter(r["evaluation_status"] for r in results)),
        "completion_rate_macro": sum(metric(r, "completion_rate") for r in results) / started if started else None,
        "latency_ms": {"p50": percentile(durations, .5), "p95": percentile(durations, .95),
                       "includes_worker_startup": True, "includes_human_wait": False},
        "physical_llm_calls": sum(r.get("telemetry", {}).get("physical_llm_calls") or 0 for r in results),
        "physical_call_count_complete": all(r.get("evaluation_status") != "incomplete"
                    and r.get("telemetry", {}).get("model_call_count_complete", True) for r in results),
        "tool_calls": sum(r.get("telemetry", {}).get("tool_calls", 0) for r in results),
        "usage": {**{key: sum(values) if values and all(type(v) is int for v in values) else None
                      for key, values in token_values.items()}, "api_cost_usd": None,
                  "attempt_usage_coverage": sum(all(type(item.get("usage", {}).get(key)) is int
                  for key in ("input_tokens", "output_tokens")) for item in calls) / started if started else None},
        "families": dict(families), "stability": stability,
        "limitations": manifest["limitations"] + [
            "Expected cancellation and safe failure cases count toward protocol pass, not task success.",
            "Scripted repeats are deterministic protocol checks; real-model repeats share episode identity.",
            "Incomplete worker evidence may omit finished physical requests; reserved call allowances bound those attempts."],
    }


def write_report(run, manifest, results):
    run = Path(run)
    summary = summarize(manifest, results)
    write_json(run / "summary.json", summary)
    def ratio(value):
        return f"{value:.1%}" if value is not None else "不可计算"
    def count(value):
        return str(value) if type(value) is int and value >= 0 else "null（未知）"
    local_proposal = manifest["config"].get("mode", "fake") == "model_proposal"
    usage = summary["usage"]
    title = "# Samantha 本地模型提案评测" if local_proposal else "# Samantha P0 离线评测"
    introduction = ("这是本地 qwen3:4b 驱动真实 Workflow 的提案与拒绝执行评测。所有生成命令均被拒绝，"
                    "不衡量执行后的端到端任务成功率。" if local_proposal else
                    "这是 Fake Services 驱动真实 Workflow 的规则验证，不代表真实模型端到端成功率。")
    success_note = ("此模式预期取消，不计为任务成功；请用协议通过和证据评估提案流程。" if local_proposal else
                    "预期取消和安全失败不计为任务成功。")
    lines = [title, "", f"实验：`{manifest['experiment_id']}`", "", introduction, "",
             f"计划尝试 {summary['planned_attempts']}；已启动 {summary['started_attempts']}；"
             f"具有完整评分证据 {summary['verified_attempts']}；覆盖率 {ratio(summary['coverage'])}。", "",
             f"协议通过：{summary['protocol_passes']}/{summary['started_attempts']}"
             f"（{ratio(summary['protocol_pass_rate'])}）。",
             f"Attempted Task Success：{summary['task_successes']}/{summary['started_attempts']}"
             f"（{ratio(summary['attempted_task_success_rate'])}）。{success_note}", "",
             f"已保存的物理 LLM 请求：{summary['physical_llm_calls']}；执行批次：{summary['tool_calls']}。",
             f"输入 Token 总计：{count(usage['input_tokens'])}；输出 Token 总计：{count(usage['output_tokens'])}。"
             "仅使用模型返回的 usage；API 成本未测量，记为 null。",
             f"输入与输出 usage 均完整的尝试覆盖率：{ratio(usage['attempt_usage_coverage'])}。"
             "缺失 usage 不按 0 处理，也不以字符数估算。", "",
             "| 案例 | 重复 | 运行状态 | 评测状态 | 证据 |",
             "| --- | --- | --- | --- | --- |"]
    for result in results:
        evidence = result.get("evidence_ref", "")
        lines.append(f"| {result['case_id']} | {result['attempt_id']} | {result['runtime_status']} | "
                     f"{result['evaluation_status']} | [evidence]({evidence}) |")
    if usage["input_tokens"] is None or usage["output_tokens"] is None:
        partial = {}
        for key in ("input_tokens", "output_tokens"):
            values = [result.get("telemetry", {}).get("usage_coverage", {}).get(key, {}).get(
                "reported_sum", result.get("telemetry", {}).get("usage", {}).get(key)) for result in results]
            known = [value for value in values if type(value) is int and value >= 0]
            partial[key] = sum(known) if known else None
        lines.extend(["", f"已知部分 usage：输入 Token {count(partial['input_tokens'])}；"
                      f"输出 Token {count(partial['output_tokens'])}。这不是包含未知请求的总量。"])
    source = Path(manifest["regraded_from"]) / "frozen" if manifest.get("regraded_from") else run / "frozen"
    source_link = Path(os.path.relpath(source, run)).as_posix()
    lines.extend(["", f"所有评分读取已保存快照和日志。源码、数据集、配置和评分器位于 [冻结产物]({source_link}/)，"
                  "重评会校验哈希。失败与未完成尝试均保留。", "",
                  ("隔离范围是独立 worker、仅限本地 Ollama 的网络入口和禁用命令执行；不宣称具备任意 Shell 安全隔离。"
                   if local_proposal else
                   "隔离范围是进程和受限 Fake 文件动作；未执行生成的 Bash，也不宣称具备任意 Shell 安全隔离。"),
                  "未完成 worker 的请求与 usage 可能未写入证据；已保存请求数只是可观测计数，预算仍保留完整调用额度。",
                  "日志含本地合成 fixture 内容，公开报告前仍需审查。", ""])
    (run / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
