"""Offline request-budget admission, returned-usage coverage, and context checks."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "src"))

from chat import Chat
from evals.report import write_report
from model_budget import BudgetExceeded, after_request, before_request, budget_context
from model_telemetry import capture_model_calls, model_request


def admit(provider="qwen", transport="openai", model="qwen3:4b"):
    return before_request(provider, transport, model, {"messages": "private input"})


def response(usage=None):
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="fixture"))])


class ModelBudgetTests(unittest.TestCase):
    def test_no_active_budget_leaves_ordinary_runtime_unrestricted(self):
        self.assertIsNone(admit(provider="azure", model="ordinary-cloud-model"))
        after_request(None, {"usage": {"input_tokens": 10, "output_tokens": 20}})

    def test_admission_counts_requests_and_rejection_has_no_side_effect(self):
        with budget_context(max_calls=2, max_seconds=10) as budget:
            self.assertIs(admit(), budget)
            self.assertIs(admit(transport="ollama"), budget)
            with self.assertRaisesRegex(BudgetExceeded, "request budget"):
                admit()
        self.assertEqual(budget.calls, 2)

    def test_cloud_or_undeclared_models_are_rejected_without_consuming_calls(self):
        with budget_context(max_calls=1, max_seconds=10) as budget:
            for provider, model in (("azure", "qwen3:4b"), ("openai", "qwen3:4b"),
                                    ("qwen", "qwen3:8b"), ("ollama", "bge-m3")):
                with self.subTest(provider=provider, model=model), self.assertRaises(BudgetExceeded):
                    admit(provider=provider, model=model)
            self.assertEqual(budget.calls, 0)
            admit(provider="ollama", transport="ollama")
        self.assertEqual(budget.calls, 1)

    def test_deadline_rejects_at_exact_boundary_and_before_request(self):
        with budget_context(max_calls=2, max_seconds=5) as budget:
            budget.started = 100.0
            with patch("model_budget.time.monotonic", return_value=104.999):
                admit()
            with patch("model_budget.time.monotonic", return_value=105.0), \
                    self.assertRaisesRegex(BudgetExceeded, "deadline"):
                admit()
        self.assertEqual(budget.calls, 1)

    def test_deadline_accepts_only_positive_finite_numbers_without_bool(self):
        for value in (True, False, 0, -1, float("nan"), float("inf"), float("-inf"),
                      "1", None, object()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with budget_context(max_calls=1, max_seconds=value):
                    self.fail("invalid deadline admitted")
        for value in (1, 0.001):
            with self.subTest(value=value), budget_context(max_calls=1, max_seconds=value) as budget:
                self.assertEqual(budget.max_seconds, value)

    def test_call_and_token_caps_require_positive_integers_without_coercion(self):
        for value in (True, False, 0, -1, 1.5, "2", None):
            with self.subTest(call_cap=value), self.assertRaises(ValueError):
                with budget_context(max_calls=value, max_seconds=10):
                    pass
        for value in (True, False, 0, -1, 1.5, "2"):
            with self.subTest(token_cap=value), self.assertRaises(ValueError):
                with budget_context(max_calls=1, max_seconds=10, max_reported_tokens=value):
                    pass

    def test_returned_usage_stops_next_admission_and_does_not_estimate_tokens(self):
        with budget_context(max_calls=5, max_seconds=10, max_reported_tokens=10) as budget:
            ticket = admit()
            after_request(ticket, {"usage": {"input_tokens": 6, "output_tokens": 7}})
            with self.assertRaisesRegex(BudgetExceeded, "token budget"):
                admit()
        self.assertEqual(budget.calls, 1)
        self.assertEqual(budget.reported_tokens, 13)
        self.assertEqual(budget.missing_usage_calls, 0)
        self.assertIn("cannot prove a hard token cap", budget.snapshot()["token_limit_kind"])

    def test_missing_and_invalid_usage_are_unknown_even_with_raw_text(self):
        usage_values = (None, {}, {"input_tokens": None, "output_tokens": 4},
                        {"input_tokens": "20", "output_tokens": 4},
                        {"input_tokens": True, "output_tokens": 4},
                        {"input_tokens": -1, "output_tokens": 4},
                        {"input_tokens": 1.5, "output_tokens": 4})
        with budget_context(max_calls=8, max_seconds=10, max_reported_tokens=1) as budget:
            for usage in usage_values:
                ticket = admit()
                after_request(ticket, {"usage": usage, "prompt": "x" * 1000, "output": "y" * 1000})
            after_request(admit(), {"usage": {"input_tokens": 0, "output_tokens": 0}})
        self.assertEqual(budget.reported_tokens, 0)
        self.assertEqual(budget.missing_usage_calls, len(usage_values))
        self.assertEqual(budget.calls, 8)
        self.assertNotIn("prompt", json.dumps(budget.snapshot()))

    def test_context_restores_after_exception_and_nested_budgets_are_independent(self):
        with budget_context(max_calls=2, max_seconds=10) as outer:
            admit()
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with budget_context(max_calls=1, max_seconds=10) as inner:
                    self.assertIs(admit(), inner)
                    raise RuntimeError("fixture")
            self.assertIs(admit(), outer)
        self.assertEqual((outer.calls, inner.calls), (2, 1))
        self.assertIsNone(admit(provider="azure", model="unrestricted"))

    def test_copied_context_shares_lock_and_never_over_admits_concurrently(self):
        with budget_context(max_calls=3, max_seconds=10) as budget, ThreadPoolExecutor(max_workers=8) as pool:
            def attempt():
                try:
                    admit()
                    return True
                except BudgetExceeded:
                    return False
            contexts = [copy_context() for _ in range(20)]
            admitted = list(pool.map(lambda context: context.run(attempt), contexts))
        self.assertEqual(sum(admitted), 3)
        self.assertEqual(budget.calls, 3)

    def test_failed_physical_call_counts_once_and_unknown_usage_is_preserved(self):
        with budget_context(max_calls=1, max_seconds=10) as budget, capture_model_calls() as calls:
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with model_request(provider="qwen", transport="openai", model="qwen3:4b", parameters={}):
                    raise RuntimeError("fixture")
            with self.assertRaises(BudgetExceeded):
                with model_request(provider="qwen", transport="openai", model="qwen3:4b", parameters={}):
                    self.fail("transport must not start")
        self.assertEqual(budget.calls, 1)
        self.assertEqual(budget.missing_usage_calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "failed")
        self.assertIsNone(calls[0]["usage"]["input_tokens"])

    def test_local_embedding_work_does_not_consume_physical_request_cap(self):
        with budget_context(max_calls=1, max_seconds=10) as budget, capture_model_calls() as calls:
            with model_request(provider="local", transport="FlagEmbedding", model="bge-m3",
                               parameters={}, call_type="embedding", physical_request=False):
                pass
            admit()
        self.assertEqual(budget.calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["physical_request"])
        self.assertEqual(budget.missing_usage_calls, 0)

    def test_budget_count_and_deadline_never_trigger_azure_fallback(self):
        for stop in ("count", "deadline"):
            with self.subTest(stop=stop), patch.dict(os.environ, {
                    "SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "fixture"}, clear=True), \
                    patch("chat.OpenAI") as qwen, patch("chat.AzureOpenAI") as azure, capture_model_calls() as calls:
                qwen.return_value.chat.completions.create.return_value = response({"prompt_tokens": 2, "completion_tokens": 1})
                chat = Chat()
                with budget_context(max_calls=1, max_seconds=10) as budget:
                    if stop == "count":
                        chat.send_message([])
                    else:
                        budget.started -= 11
                    with self.assertRaises(BudgetExceeded):
                        chat.send_message([])
                expected = 1 if stop == "count" else 0
                self.assertEqual(qwen.return_value.chat.completions.create.call_count, expected)
                self.assertEqual(len(calls), expected)
                azure.return_value.chat.completions.create.assert_not_called()


class ModelReportTests(unittest.TestCase):
    def report(self, mode, usage, *, partial=None):
        manifest = {"experiment_id": "fixture", "config": {"mode": mode, "repeats": 1},
                    "dataset": {"case_count": 1}, "limitations": []}
        telemetry = {"physical_llm_calls": 2 if mode == "model_proposal" else 0,
                     "tool_calls": 0, "usage": usage}
        if partial is not None:
            telemetry["usage_coverage"] = {key: {"reported_sum": value} for key, value in partial.items()}
        results = [{"case_id": "fixture", "attempt_id": "repeat_01", "runtime_status": "cancelled",
                    "evaluation_status": "passed", "scores": [{"metric": "task_success", "value": 0}],
                    "evidence_ref": "attempts/fixture/evidence.json", "telemetry": telemetry}]
        with tempfile.TemporaryDirectory() as directory:
            summary = write_report(directory, manifest, results)
            return summary, (Path(directory) / "summary.md").read_text(encoding="utf-8")

    def test_local_proposal_report_describes_real_calls_denial_and_actual_usage(self):
        summary, report = self.report("model_proposal", {"input_tokens": 20, "output_tokens": 10})
        self.assertIn("本地模型提案评测", report)
        self.assertIn("所有生成命令均被拒绝", report)
        self.assertNotIn("Fake Services", report)
        self.assertIn("输入 Token 总计：20", report)
        self.assertIn("输出 Token 总计：10", report)
        self.assertIn("覆盖率：100.0%", report)
        self.assertIsNone(summary["usage"]["api_cost_usd"])

    def test_unknown_usage_reports_known_partial_without_claiming_a_total(self):
        summary, report = self.report("model_proposal", {"input_tokens": None, "output_tokens": None},
                                      partial={"input_tokens": 7, "output_tokens": 3})
        self.assertIsNone(summary["usage"]["input_tokens"])
        self.assertIn("输入 Token 总计：null（未知）", report)
        self.assertIn("已知部分 usage：输入 Token 7；输出 Token 3", report)
        self.assertIn("覆盖率：0.0%", report)
        self.assertIn("缺失 usage 不按 0 处理", report)

    def test_fake_report_retains_protocol_scope_and_null_tokens(self):
        summary, report = self.report("fake", {"input_tokens": None, "output_tokens": None})
        self.assertIn("Fake Services", report)
        self.assertIn("受限 Fake 文件动作", report)
        self.assertIn("输入 Token 总计：null（未知）", report)
        self.assertEqual(summary["physical_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
