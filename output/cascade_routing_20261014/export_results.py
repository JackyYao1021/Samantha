"""Export compact, auditable experiment summaries after all local runs finish."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ("cascade-v1-20261014", "cascade-augmented-v1-20261014", "cascade-balanced-v1-20261014")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    experiments = []
    for name in RUNS:
        folder = ROOT / "work/.test-results/routing" / name
        audit_folder = folder.with_name(name + "-audit")
        manifest, summary, comparison = (read(folder / f"{file}.json") for file in ("manifest", "summary", "comparison"))
        audit = read(audit_folder / "audit.json")
        if manifest.get("stop_reason") != "completed" or not audit["ledger_complete"] or not audit["cascade_calibration_verified"]:
            raise ValueError("Export requires completed, audited runs: " + name)
        attempts = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        fields = ("case_id", "family", "split", "policy", "repeat", "started", "complete", "task_success",
                  "duration_ms", "router_duration_ms", "initial_model", "selected_model", "classification_family",
                  "route_decision", "route_trace", "fallback", "fallback_reason", "error_category")
        experiments.append({"experiment_id": name, "raw_evidence_directory": folder.relative_to(ROOT).as_posix(),
            "manifest": manifest, "summary": summary, "comparison": comparison,
            "audit": {k: v for k, v in audit.items() if k not in ("frozen_rule_regrades", "same_request_outcome_variability", "original_run")},
            "same_request_variability": {k: v for k, v in audit["same_request_outcome_variability"].items() if k != "groups"},
            "attempts": [{key: row.get(key) for key in fields} for row in attempts],
            "evidence_hashes": {file: digest(folder / file) for file in
                                ("manifest.json", "results.jsonl", "requests.jsonl", "summary.json", "comparison.json", "cascade_calibration.json")},
            "audit_sha256": digest(audit_folder / "audit.json")})
    result = {"schema_version": "cascade-routing-results-v1", "experiments": experiments,
              "actual_physical_requests": sum(e["manifest"]["actual_physical_requests"] for e in experiments),
              "offline_validation": {"routing_tests": 115, "passed": 114, "skipped": 1,
                  "command": ".venv/Scripts/python.exe -B -m unittest discover -s work/tests -p test_routing*.py -q",
                  "live_model_calls_in_unit_tests": 0},
              "primary_scope": "exploratory regression on 12 previously inspected heldout file-tool tasks",
              "contains_full_raw_transport_evidence": False,
              "notes": ["Full request/response/telemetry and frozen sources are retained in local raw evidence directories.",
                        "The second protocol expands development prototypes; no rule, threshold grid or heldout label was adjusted in response to heldout scores.",
                        "A single execution-model mapping cannot establish gains from choosing among complementary models."]}
    target = ROOT / "work/evals/results/cascade_routing_20261014.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = experiments[-1]["summary"]["policies"]
    cascade, direct = latest["cascade"], latest["model_driven"]
    stages = cascade["routing_quality"]["cascade_stages"]
    lines = ["# 2026-10-14 级联模型路由实验", "",
             f"三级级联已在真实任务中运行：最后一组 {cascade['planned']} 个任务分别在规则、Embedding、LLM 层结束 "
             f"{stages['rule']['accepted']}、{stages['embedding']['accepted']}、{stages['llm']['accepted']} 次。",
             f"与独立 LLM 路由相比，LLM 分类请求从 {direct['routing_quality']['llm_classifier_calls']} 次降至 "
             f"{cascade['routing_quality']['llm_classifier_calls']} 次，但全部 heldout 请求从 {direct['physical_calls']} 次增至 "
             f"{cascade['physical_calls']} 次；任务成功为 {cascade['task_successes']}/{cascade['planned']}，对照为 "
             f"{direct['task_successes']}/{direct['planned']}。",
             "因此当前证据支持流程实现与调用构成变化，不支持整体质量提升、显著提速或货币降本的主张。", "",
             "实现与复现实验命令见 [三级级联路由](cascade_model_routing.md)。",
             "本次比较使用已查看过的 12 个 heldout 文件任务，属于探索性回归实验。", "",
             "先以原 12 个开发原型运行两轮；发现开发留一校准没有满足门槛的向量 gate 后，",
             "保留该运行，再添加 48 条开发请求（24 个中英文语义组），单独运行一轮完整六策略比较。",
             "第二组保持规则、提示词、阈值网格及接受门槛不变。严格配置仍无可用 gate 后，第三组依据开发集诊断，",
             "将接受准确率下限改为 75%、最低接受数改为 12，增加覆盖优先的四策略对比。该取舍配置不替换严格配置。",
             "各组的模型能力画像均重新测量，不能直接视为单因素消融；没有根据 heldout 成绩修改规则、提示词或阈值。", "",
             f"三次运行实际物理请求合计 **{result['actual_physical_requests']}**；所有任务均保留在计划分母中。", ""]
    for exp in experiments:
        manifest, summary, audit = exp["manifest"], exp["summary"], exp["audit"]
        gate = manifest["cascade_calibration"]["gate"]
        lines.extend([f"## {exp['experiment_id']}", "",
            f"开发校准样本 {manifest['cascade_calibration']['calibration_cases']}；heldout 重复 {manifest['config']['repeats']} 轮；"
            f"物理请求 {manifest['actual_physical_requests']}；运行 {manifest['duration_seconds']:.1f} 秒。", "",
            f"向量 gate：`{json.dumps(gate, ensure_ascii=False)}`。这是开发拟合结果，不是泛化准确率。", "",
            "| 策略 | 任务成功/计划 | 分类正确/已分类 | LLM 分类请求 | query embedding | 全部 heldout 请求 | p50/p95 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |"])
        for policy, group in summary["policies"].items():
            quality, latency = group["routing_quality"], group["latency_ms"]
            classification = "不适用" if policy.startswith("fixed_") else f"{quality['correct']}/{quality['classified']}"
            lines.append(f"| {policy} | {group['task_successes']}/{group['planned']} "
                f"| {classification} | {quality['llm_classifier_calls']} "
                f"| {quality['embedding_query_calls']} | {group['physical_calls']} "
                f"| {latency['p50']:.1f}/{latency['p95']:.1f} |")
        cascade = summary["policies"]["cascade"]
        direct = summary["policies"]["model_driven"]
        lines.extend(["", "级联层次（接受错误按隐藏类别离线计算，不影响执行或触发升级）：", "",
                      "| 层 | 访问 | 接受 | 错误接受 | 状态分布 |", "| --- | ---: | ---: | ---: | --- |"])
        for stage in ("rule", "embedding", "llm", "default"):
            counts = cascade["routing_quality"]["cascade_stages"].get(stage)
            if counts is None:
                continue
            lines.append(f"| {stage} | {counts['visited']} | {counts['accepted']} | {counts['wrong_acceptances']} "
                         f"| {json.dumps(counts['status_counts'], ensure_ascii=False)} |")
        used = cascade["routing_quality"]["llm_classifier_calls"]
        reference = direct["routing_quality"]["llm_classifier_calls"]
        lines.extend(["", f"相对于独立 LLM 路由，LLM **分类请求**由 {reference} 次变为 {used} 次，"
                      f"减少 {(1-used/reference):.1%}。这不等于全部请求、总 Token、端到端延迟或货币成本按同样比例下降。", "",
                      "冻结的类别到模型映射：", "", "```json",
                      json.dumps(manifest["selection"]["family_models"], ensure_ascii=False, indent=2), "```", ""])
        if len(set(manifest["selection"]["family_models"].values())) == 1:
            lines.extend(["所有类别映射到同一执行模型；本轮不能证明模型分流提高任务成功率。", ""])
        pair = next(p for p in exp["comparison"]["comparisons"] if p["left_policy"] == "cascade" and p["right_policy"] == "model_driven")
        delta = pair["difference"]
        lines.extend([f"任务成功率差（model_driven − cascade）的 case 配对 bootstrap："
                      f"{delta['mean']:.1%}，95% 区间 [{delta['low']:.1%}, {delta['high']:.1%}]；"
                      f"独立 case 数 {pair['paired_cases']}。", "",
                      f"离线审计：请求账一致；阈值可从冻结开发原型重建；原评分器重评分 "
                      f"{audit['regrade_agreement']}/{audit['regrade_count']} 一致。", "",
                      f"完整报告：`{exp['raw_evidence_directory']}/routing_report.md`；"
                      f"审计：`{exp['raw_evidence_directory']}-audit/audit.json`。", ""])
    lines.extend(["## 第一组的失败案例", "",
                  "以下来自原配置 cascade 的第一轮保存响应，未据此修改本轮规则、提示词或评分标准：", "",
                  "| case | 观测 | 含义 |", "| --- | --- | --- |",
                  "| RH-write-02 | 正确分类 write，但结果文件缺少最后一个换行 | 类别正确不代表精确字节达标 |",
                  "| RH-select-02 | 正确分类 select，却生成 write/append 动作并改动文件 | 只读约束需要执行层独立验证 |",
                  "| RH-clarify-02 | 分类 rename，执行器自行猜测新的文件名 | 必要信息缺失时仍可能错误完成 |", "",
                  "这些任务只在临时夹具中执行；文件终态评分将其记为失败，没有因此触发另一次模型求解。", "",
                  "## 证据与限制", "",
                  "当前路由离线回归：115 项测试中 114 项通过、1 项跳过；模拟传输测试不算真实模型实验。",
                  "结构化摘要保存在 `work/evals/results/cascade_routing_20261014.json`，包含逐任务路由轨迹、阈值拟合、请求汇总与原文件哈希。",
                  "摘要不包含完整传输字节和冻结源码；完整证据保存在本机 `.test-results`，该目录被 Git 忽略。",
                  "余弦分数不是正确概率；新增开发样本按工具语义合成并固定标签，尚未经过外部人工独立复核，语义覆盖仍有限。",
                  "模型加载、切换、输出波动会影响计时；第一轮开发阶段还并行运行了离线单元测试。",
                  "两次重复不是独立新增任务；小样本结果不代表线上表现。输入/输出 Token、请求和计算时间均为资源代理，未测量实际货币费用。", ""])
    report = ROOT / "readme_help/cascade_model_routing_results_20261014.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(target), "report": str(report), "actual_physical_requests": result["actual_physical_requests"]}, ensure_ascii=False))


if __name__ == "__main__":
    export()
