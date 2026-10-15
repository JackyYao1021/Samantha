"""Intent and planner physical requests share persisted task admission and telemetry."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_budget import budget_context
from model_telemetry import model_request
from task_agent.runtime import AgentRuntime


def boundary():
    return model_request(provider="qwen", transport="ollama", model="qwen3:4b", parameters={})


def classification():
    return {"schema_version": "1.0", "domains": ["file_management"],
            "operations": [{"id": "list", "operation": "list", "parameters": {}, "depends_on": []}],
            "constraints": [], "unresolved_references": [], "missing_information": [], "ambiguities": [],
            "support": "supported", "fallback": "none"}


class IntentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "state.sqlite"

    def test_classifier_and_planner_share_outer_and_persistent_budget(self):
        def classify(*_):
            with boundary():
                return classification()
        def plan(*_):
            with boundary():
                return {"type": "finish", "answer": "done"}
        runtime = AgentRuntime(self.db, plan, intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.create_task("List files", self.root, max_model_calls=2)
        with budget_context(max_calls=2, max_seconds=30) as outer:
            task = runtime.run(task["id"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(outer.calls, 2)
        self.assertEqual(task["model_calls"], 2)
        self.assertEqual(task["model_requests_reserved"], 2)
        self.assertEqual([r["phase"] for r in task["model_call_records"]], ["intent", "planning"])

    def test_transport_failure_is_charged_and_fallback_is_not_retried(self):
        count = []
        def classify(*_):
            count.append(1)
            with boundary():
                raise OSError("offline fixture")
        runtime = AgentRuntime(self.db, lambda *_: {"type": "finish", "answer": "test"},
                               intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.run(runtime.create_task("List files", self.root)["id"])
        self.assertEqual(count, [1])
        self.assertEqual(task["model_calls"], 1)
        self.assertEqual(task["model_requests_reserved"], 1)
        self.assertEqual(task["model_call_records"][0]["status"], "failed")
        self.assertIsNone(task["model_call_records"][0]["usage"]["input_tokens"])
        self.assertEqual(task["intent_history"][0]["status"], "fallback")

    def test_budget_rejection_during_classifier_is_terminal_without_planner(self):
        def classify(*_):
            with boundary():
                return classification()
        runtime = AgentRuntime(self.db, lambda *_: self.fail("budget rejection must stop before planner"),
                               intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.create_task("List files", self.root)
        with budget_context(max_calls=1, max_seconds=30) as outer:
            with boundary():
                pass
            task = runtime.run(task["id"])
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["intent_history"][0]["status"], "budget_exhausted")
        self.assertEqual(task["model_requests_reserved"], 0)
        self.assertEqual(outer.calls, 1)

    def test_classifier_time_counts_and_stops_planner_after_deadline(self):
        clock = [0.0]
        def classify(*_):
            clock[0] += 2
            return classification()
        runtime = AgentRuntime(self.db, lambda *_: self.fail("over-time classification must stop planner"),
                               intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.create_task("List files", self.root)
        state = runtime.store.read(task["id"], events=False)
        state["max_active_seconds"] = 1
        runtime.store.save(state, "fixture")
        with patch("task_agent.runtime.time.monotonic", side_effect=lambda: clock[0]):
            task = runtime.run(task["id"])
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["active_seconds"], 2)
        self.assertEqual(task["steps"], 0)


if __name__ == "__main__":
    unittest.main()
