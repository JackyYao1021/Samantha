"""Offline schema and single-call extraction tests; no model accuracy claims."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_agent.intent import (ModelIntentClassifier, PROMPT_VERSION, SCHEMA_VERSION, SYSTEM,
                               intent_schema, validate_intent, validate_context_sources)
from task_agent.planner import continuation_messages
from task_agent.runtime import AgentRuntime


def parameter(value, source="request"):
    return {"value": value, "source": source, "evidence": "user or observation excerpt"}


def sample():
    return {"schema_version": "1.0", "domains": ["file_management"],
            "operations": [{"id": "find", "operation": "search_name", "parameters": {"query": parameter("contract")},
                            "depends_on": []},
                           {"id": "copy", "operation": "copy", "parameters": {
                               "source": parameter(None, "unresolved"), "destination": parameter("archive/contract.txt")},
                            "depends_on": ["find"]}],
            "constraints": [{"kind": "preserve_source", "value": True, "source": "request", "evidence": "copy"},
                            {"kind": "no_overwrite", "value": True, "source": "request", "evidence": "do not overwrite"}],
            "unresolved_references": [{"text": "the contract", "operation_id": "copy", "parameter": "source", "candidates": []}],
            "missing_information": [{"operation_id": "copy", "parameter": "source", "reason": "Locate matching contract",
                                     "resolution": "environment"}],
            "ambiguities": [], "support": "supported", "fallback": "planner"}


class IntentContractTests(unittest.TestCase):
    def test_prompt_normalization_is_versioned_separately_from_data_contract(self):
        self.assertEqual(SCHEMA_VERSION, "1.0")
        self.assertEqual(PROMPT_VERSION, "1.1")
        for rule in ("exact_content=true", 'scope="read_only"', "destination_directory", "not an order constraint"):
            self.assertIn(rule, SYSTEM)

    def test_multi_operation_order_and_constraints_are_preserved(self):
        source = sample()
        result = validate_intent(source)
        self.assertEqual(result, source)
        result["operations"].clear()
        self.assertEqual(len(source["operations"]), 2)

    def test_flat_schema_omits_only_domain_level(self):
        hierarchy = intent_schema()
        flat = intent_schema("flat")
        hierarchy["properties"].pop("domains")
        hierarchy["required"].remove("domains")
        self.assertEqual(hierarchy, flat)
        value = sample()
        value.pop("domains")
        self.assertEqual(validate_intent(value, "flat"), value)
        with self.assertRaises(ValueError):
            validate_intent(value)

    def test_rejects_malformed_unknown_labels_and_inconsistent_domains(self):
        changes = [lambda x: x.update(schema_version="2.0"), lambda x: x.update(domains=["bash"]),
                   lambda x: x.update(domains=["content_query"]), lambda x: x.update(extra="ignored"),
                   lambda x: x["operations"][0].update(operation="delete"),
                   lambda x: x["operations"][0]["parameters"]["query"].update(source="guess"),
                   lambda x: x["operations"][0]["parameters"]["query"].update(value=True),
                   lambda x: x["operations"][0].update(id="copy"),
                   lambda x: x["operations"][0].update(depends_on=["copy"]),
                   lambda x: x["operations"][1]["parameters"]["source"].update(value="guessed.txt"),
                   lambda x: x.update(fallback="clarify")]
        for change in changes:
            value = sample()
            change(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_intent(value)

    def test_required_parameters_need_explicit_missing_record(self):
        value = sample()
        value["operations"][0]["parameters"] = {}
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_intent(value)
        value["missing_information"].append({"operation_id": "find", "parameter": "query",
                                              "reason": "User must name object", "resolution": "user"})
        value["fallback"] = "clarify"
        validate_intent(value)

    def test_context_provenance_rejects_invented_paths_and_quoted_path_data(self):
        value = sample()
        value["operations"][1]["parameters"]["source"] = parameter("contract.txt", "verified_context")
        with self.assertRaisesRegex(ValueError, "lacks matching"):
            validate_context_sources(value, {"observations": [{"ok": True, "result": {"content": "contract.txt"}}]})
        context = {"verified_actions": [{"tool": "write_file", "args": {"path": "contract.txt"}}]}
        self.assertIs(validate_context_sources(value, context), value)

    def test_extractor_is_one_structured_call_and_closes_transports(self):
        with patch("task_agent.intent.Chat") as chat:
            chat.return_value.chat_context.return_value = json.dumps(sample())
            result = ModelIntentClassifier()({"request": "Find and copy the contract", "remaining_seconds": 0.5}, [])
            self.assertEqual(len(result["operations"]), 2)
            chat.return_value.chat_context.assert_called_once()
            self.assertFalse(chat.call_args.kwargs["allow_azure_fallback"])
            self.assertEqual(chat.call_args.kwargs["request_timeout"], 0.5)
            chat.return_value.client_qwen.close.assert_called_once()

    def test_invalid_model_json_does_not_retry(self):
        with patch("task_agent.intent.Chat") as chat:
            chat.return_value.chat_context.return_value = '{"bad":1,"bad":2}'
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ModelIntentClassifier()({"remaining_seconds": 1}, [])
            chat.return_value.chat_context.assert_called_once()


class IntentRuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "files"
        self.workspace.mkdir()
        self.db = self.root / "task.sqlite"

    def test_baseline_never_classifies_and_prompt_remains_without_intent(self):
        def classify(*_):
            self.fail("baseline must not classify")
        runtime = AgentRuntime(self.db, lambda *_: {"type": "finish", "answer": "ok"},
                               intent_mode="baseline", intent_classifier=classify)
        task = runtime.run(runtime.create_task("keep original", self.workspace)["id"])
        self.assertEqual(task["intent_history"], [])
        initial, _ = continuation_messages(runtime._context(task), [])
        self.assertNotIn("advisory_intent", json.loads(initial[0]["content"]))

    def test_classifier_failure_falls_back_once_with_original_request(self):
        seen = []
        def classify(*_):
            seen.append("classification")
            return {"schema_version": "999"}
        decisions = iter([{"type": "tool", "tool": "list_files", "args": {}}, {"type": "finish", "answer": "ok"}])
        def plan(context, tools):
            self.assertEqual(context["request"], "list files; do not modify anything")
            self.assertEqual(len(tools), 9)
            return next(decisions)
        runtime = AgentRuntime(self.db, plan, intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.run(runtime.create_task("list files; do not modify anything", self.workspace)["id"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(seen, ["classification"])
        self.assertEqual(task["intent_history"][0]["status"], "fallback")
        self.assertEqual(task["model_calls"], 0)

    def test_short_clarification_reclassifies_full_task_after_restart(self):
        seen = []
        def classify(context, _):
            seen.append(copy.deepcopy(context))
            return sample()
        runtime = AgentRuntime(self.db, lambda *_: {"type": "ask", "question": "What name?"},
                               intent_mode="hierarchical", intent_classifier=classify)
        task = runtime.run(runtime.create_task("Create a notes file", self.workspace)["id"])
        runtime.answer(task["id"], "叫 notes.txt")
        restarted = AgentRuntime(self.db, lambda *_: {"type": "finish", "answer": "test"}, intent_classifier=classify)
        task = restarted.run(task["id"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["request"], "Create a notes file")
        self.assertEqual(seen[1]["clarifications"][0]["answer"], "叫 notes.txt")
        self.assertEqual(task["intent_history"][1]["reason"], "clarification_answered")

    def test_revision_revokes_approval_and_keeps_original_and_correction(self):
        runtime = AgentRuntime(self.db, lambda *_: {"type": "tool", "tool": "write_file",
                                                   "args": {"path": "old.txt", "content": "hello"}},
                               intent_mode="hierarchical", intent_classifier=lambda *_: sample())
        task = runtime.run(runtime.create_task("Create old.txt", self.workspace)["id"])
        action_id = task["pending_action"]["id"]
        runtime.approve(task["id"], action_id)
        task = runtime.revise(task["id"], "Actually name it new.txt")
        self.assertIsNone(task["pending_action"])
        self.assertEqual(task["actions"][0]["status"], "superseded")
        self.assertEqual(task["request"], "Create old.txt")
        self.assertTrue(task["intent_dirty"])
        with self.assertRaises(ValueError):
            runtime.approve(task["id"], action_id)
        self.assertFalse((self.workspace / "old.txt").exists())


if __name__ == "__main__":
    unittest.main()
