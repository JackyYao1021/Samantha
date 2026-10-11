"""Memory isolation, evidence, graph integration and benchmark protocol checks."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent_memory import MemoryHit, MemoryQuery, MemoryScope, MemorySession, create_backend, pack_hits
from agent_memory.journal import journal_session
from interaction_log import InteractionJournal, record_operation
from langgraph.types import Command
from memory_eval import history_sessions, load_dataset, retrieval_metrics, run_dataset
from memory_tasks.task_suite import load_cases, run_case
from samantha import process
from test_workflow import services
from workflow import build_workflow, initial_state


SCOPE = MemoryScope("user-a", "project-a")


def episode(identifier="old", date="2025-01-01", text="The report is in /work/report.txt.", scope=SCOPE):
    return MemorySession(identifier, scope, date, ({"role": "user", "content": text},))


def query(text="Where is the report?", **kwargs):
    return MemoryQuery(SCOPE, text, "2026-01-01", **kwargs)


class MemoryTests(unittest.TestCase):
    def test_persistence_idempotency_scope_and_time_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            backend = create_backend("recent", path=path)
            for item in (episode(), episode(scope=MemoryScope("other-user", "project-a")),
                         episode(scope=MemoryScope("user-a", "other-project")),
                         episode("future", "2027-01-01")):
                backend.ingest(item)
            backend.ingest(episode(text="Updated report in /work/final.txt."))
            backend.close()
            reopened = create_backend("recent", path=path)
            try:
                hits = reopened.retrieve(query())
                self.assertEqual([hit.session_id for hit in hits], ["old"])
                self.assertIn("final.txt", hits[0].text)
            finally:
                reopened.close()

    def test_bm25_retrieves_old_relevant_english_and_chinese_history(self):
        backend = create_backend("bm25")
        self.addCleanup(backend.close)
        backend.ingest(episode("evidence", text="会议纪要 report.txt 包含错误 E1042"))
        backend.ingest(episode("distractor", "2025-02-01", "Today the weather is sunny."))
        self.assertEqual(backend.retrieve(query("会议纪要 E1042"))[0].session_id, "evidence")
        self.assertEqual(backend.retrieve(query("completely_unknown_term")), [])

    def test_budget_includes_json_provenance_and_partial_evidence_is_marked(self):
        hit = MemoryHit("m1", "s1", "2025-01-01", '\\"' * 1000)
        context, sources = pack_hits([hit], query(max_chars=300))
        self.assertLessEqual(len(context), 300)
        self.assertTrue(sources[0]["truncated"])
        self.assertEqual(json.loads(context), sources)
        self.assertEqual(pack_hits([hit], query(max_chars=0)), ("", []))
        with self.assertRaisesRegex(ValueError, "future"):
            pack_hits([replace(hit, timestamp="2027-01-01")], query())

    def test_imported_message_labels_are_not_exposed_to_backend(self):
        session = MemorySession("s", SCOPE, "2025/01/01 (Wed) 12:00", (
            {"role": "user", "content": "report.txt", "has_answer": True, "answer": "gold"},))
        self.assertEqual(session.messages, ({"role": "user", "content": "report.txt"},))

    def test_consolidated_memory_keeps_multiple_sources_for_evaluation(self):
        hit = MemoryHit("fact-1", "old", "2025-02-01", "Report moved from draft.txt to final.txt.",
                        source_session_ids=("old", "new"))
        context, sources = pack_hits([hit], query())
        item = benchmark_question()
        item["answer_session_ids"] = ["old", "new"]
        self.assertEqual(retrieval_metrics(item, sources)["session_recall"], 1)
        self.assertEqual(json.loads(context)[0]["source_session_ids"], ["old", "new"])
        with self.assertRaisesRegex(ValueError, "future"):
            pack_hits([replace(hit, timestamp="2027-01-01")], query())

    def test_external_factory_receives_configuration_without_workflow_changes(self):
        module = ModuleType("experimental_memory_adapter")
        memory = create_backend("none")
        module.create = Mock(return_value=memory)
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertIs(create_backend(module.__name__ + ":create", path="adapter.db",
                                         options={"strategy": "facts"}), memory)
        module.create.assert_called_once_with(path="adapter.db", options={"strategy": "facts"})
        with self.assertRaisesRegex(ValueError, "extra options"):
            create_backend("none", options={"unsupported": True})


class GraphMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = create_backend("recent")
        self.addCleanup(self.memory.close)
        self.memory.ingest(episode(text="A previous tool result mentioned /work/report.txt."))
        self.memory.retrieve = Mock(wraps=self.memory.retrieve)

    def test_context_reaches_agents_without_changing_user_input_or_replaying_retrieval(self):
        fake = services()
        graph = build_workflow(fake, memory=self.memory, memory_scope=SCOPE)
        config = {"configurable": {"thread_id": "task-1"}}
        first = graph.invoke(initial_state("Append to that report"), config)
        self.assertEqual(first["user_input"], "Append to that report")
        self.assertIn("report.txt", fake.clarify.call_args.args[0])
        self.assertIn("Historical reference", fake.parse_intent.call_args.args[0])
        self.assertIn("memory_context", fake.review_commands.call_args.args[0])
        fake.execute_commands.assert_not_called()
        self.assertEqual(first["__interrupt__"][0].value["kind"], "confirmation")
        final = graph.invoke(Command(resume="y"), config)
        self.assertEqual(final["status"], "succeeded")
        self.memory.retrieve.assert_called_once()
        fake.execute_commands.assert_called_once()

    def test_retrieval_failure_stops_before_model_or_execution(self):
        self.memory.retrieve.side_effect = RuntimeError("index unavailable")
        fake = services()
        graph = build_workflow(fake, memory=self.memory, memory_scope=SCOPE)
        state = graph.invoke(initial_state("Create file"), {"configurable": {"thread_id": "failed"}})
        self.assertEqual(state["status"], "failed")
        fake.clarify.assert_not_called()
        fake.execute_commands.assert_not_called()

    def test_cancelled_plan_is_distinct_from_execution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log.sqlite3"
            state = process("Create a file", services=services(), input_fn=lambda _: "n",
                            output_fn=lambda _: None, log_file=log,
                            memory=self.memory, memory_scope=SCOPE)
            with InteractionJournal(log, readonly=True) as journal:
                data = journal.read_session(state["log_session_id"])
            memory_session = journal_session(data, SCOPE)
            self.assertIn('"status": "cancelled"', memory_session.text())
            self.assertIn("planned_commands", memory_session.text())
            self.assertNotIn('"operation": "commands.execute"', memory_session.text())

    def test_failed_batch_keeps_submitted_commands_separate_from_final_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            with InteractionJournal(Path(directory) / "log.sqlite3") as journal:
                with journal.start_session("Run a batch") as session:
                    record_operation("commands.execute", lambda commands: {
                        "success": False, "output": "batch stopped", "current_dir": directory},
                        ["mkdir partial", "false"])
                    session.finish("failed", details={"state": {"commands": ["unsubmitted plan"]}})
                    evidence = journal_session(journal.read_session(session.session_id), SCOPE)
        records = [json.loads(message["content"]) for message in evidence.messages
                   if message["role"] == "tool"]
        self.assertEqual(records[0]["submitted_commands"], ["mkdir partial", "false"])
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[1]["planned_commands"], ["unsubmitted plan"])

    def test_ingest_failure_does_not_relabel_or_repeat_successful_execution(self):
        self.memory.ingest = Mock(side_effect=RuntimeError("store full"))
        fake = services()
        with tempfile.TemporaryDirectory() as directory:
            state = process("Create a file", services=fake, input_fn=lambda _: "y",
                            output_fn=lambda _: None, log_file=Path(directory) / "log.sqlite3",
                            memory=self.memory, memory_scope=SCOPE)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["memory_error"], "store full")
        fake.execute_commands.assert_called_once()


def benchmark_question(identifier="q1", abstention=False):
    return {"question_id": identifier + ("_abs" if abstention else ""),
            "question_type": "knowledge-update", "question": "Where is the report?",
            "answer": "GOLD_LABEL_NEVER_EXPOSE", "question_date": "2025/03/01 (Sat) 12:00",
            "haystack_session_ids": ["new", "old"],
            "haystack_dates": ["2025-02-01", "2025-01-01"],
            "haystack_sessions": [[{"role": "user", "content": "New report is final.txt", "has_answer": True}],
                                  [{"role": "user", "content": "Old report is draft.txt"}]],
            "answer_session_ids": [] if abstention else ["new"]}


class BenchmarkTests(unittest.TestCase):
    def test_repeated_official_ids_preserve_dates_and_original_provenance(self):
        item = benchmark_question()
        item["haystack_session_ids"] = ["new", "new"]
        memory = create_backend("recent")
        self.addCleanup(memory.close)
        sessions = history_sessions(item, SCOPE)
        self.assertEqual(len({session.session_id for session in sessions}), 2)
        for session in sessions:
            self.assertEqual(session.source_session_id, "new")
            memory.ingest(session)
        earlier = MemoryQuery(SCOPE, item["question"], "2025-01-15")
        hits = memory.retrieve(earlier)
        self.assertEqual(len(hits), 1)
        self.assertIn("draft.txt", hits[0].text)
        self.assertEqual(hits[0].session_id, "new")
        self.assertEqual(len(memory.retrieve(query())), 2)
        _, sources = pack_hits(hits, earlier)
        self.assertEqual(retrieval_metrics(item, sources)["session_recall"], 1)

    def test_replay_is_chronological_label_free_and_each_question_is_isolated(self):
        created = []

        def factory(name, **kwargs):
            memory = create_backend("recent", path=kwargs["path"])
            memory.ingest = Mock(wraps=memory.ingest)
            created.append(memory)
            return memory

        reader = SimpleNamespace(describe=lambda: {"model": "fake-reader"},
                                 answer=Mock(return_value=("final.txt", {"prompt_tokens": 1})))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.json"
            dataset.write_text(json.dumps([benchmark_question(), benchmark_question("q2", True)]),
                               encoding="utf-8")
            self.assertEqual(len(load_dataset(dataset)), 2)
            sessions = history_sessions(benchmark_question(), SCOPE)
            self.assertEqual([session.session_id for session in sessions], ["old", "new"])
            summary = run_dataset(dataset, root / "run", backend_name="recent", mode="qa",
                                  reader=reader, factory=factory, top_k=1)
            self.assertEqual(summary["overall"]["completed"], 2)
            self.assertEqual(summary["overall"]["recall_questions"], 1)
            self.assertEqual(summary["overall"]["session_recall"], 1)
            self.assertIsNone(summary["qa_accuracy"])
            predictions = [json.loads(line) for line in (root / "run/hypotheses.jsonl").read_text().splitlines()]
            self.assertEqual(predictions[0], {"question_id": "q1", "hypothesis": "final.txt"})
        first = created[0].ingest.call_args_list[0].args[0]
        second = created[1].ingest.call_args_list[0].args[0]
        self.assertNotEqual(first.scope, second.scope)
        for memory in created:
            for call in memory.ingest.call_args_list:
                text = call.args[0].text()
                self.assertNotIn("GOLD_LABEL_NEVER_EXPOSE", text)
                self.assertNotIn("has_answer", text)
        for call in reader.answer.call_args_list:
            self.assertNotIn("GOLD_LABEL_NEVER_EXPOSE", str(call))

    def test_backend_errors_are_retained_and_counted_not_dropped(self):
        def factory(*args, **kwargs):
            raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps([benchmark_question()]), encoding="utf-8")
            summary = run_dataset(data, root / "run", factory=factory)
            self.assertEqual(summary["overall"]["errors"], 1)
            self.assertEqual(summary["overall"]["session_recall"], 0)
            with self.assertRaises(FileExistsError):
                run_dataset(data, root / "run")


class TerminalBenchmarkTests(unittest.TestCase):
    def test_cross_invocation_fixture_uses_actual_graph_and_committed_memory(self):
        case = next(case for case in load_cases() if case["id"] == "MT01")
        fake = services()
        fake.clarify.side_effect = [
            {"is_jump": False, "requirement_summary": "Write alpha to notes.txt."},
            {"is_jump": False, "requirement_summary": "Append beta to notes.txt."},
        ]

        def execute(commands):
            target = Path("notes.txt")
            target.write_bytes(("alpha\n" if not target.exists() else "alpha\nbeta\n").encode("utf-8"))
            return {"success": True, "output": "write completed", "current_dir": str(Path.cwd())}

        fake.execute_commands.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            summary = run_case(case, Path(directory) / "run", backend="recent", services=fake,
                               input_fn=lambda _: "y", output_fn=lambda _: None)
        self.assertTrue(summary["passed"], summary)
        self.assertGreater(summary["steps"][1]["context_chars"], 0)
        second_prompt = fake.clarify.call_args_list[1].args[0]
        self.assertIn("notes.txt", second_prompt)
        self.assertIn("commands.execute", second_prompt)

    def test_cancelled_history_still_requires_clarification_and_no_execution(self):
        case = next(case for case in load_cases() if case["id"] == "MT04")
        fake = services()
        fake.clarify.side_effect = [{"is_jump": False, "requirement_summary": "Create cancelled.txt."},
                                   {"question": "Which existing file do you mean?"}]
        answers = iter(["n", "cancel"])
        with tempfile.TemporaryDirectory() as directory:
            summary = run_case(case, Path(directory) / "run", backend="recent", services=fake,
                               input_fn=lambda _: next(answers), output_fn=lambda _: None)
        self.assertTrue(summary["passed"], summary)
        fake.execute_commands.assert_not_called()


if __name__ == "__main__":
    unittest.main()
