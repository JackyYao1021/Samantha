"""Durability and transcript tests using real SQLite and LangGraph."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from content_cli import main as content_main
from history_cli import main as history_main
from interaction_log import InteractionJournal, InteractionLogError, LogSession, operation
from run_commands import run_commands
from samantha import main, process
from test_workflow import services


class JournalTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "logs" / "interactions.sqlite3"
        environment = patch.dict(os.environ, {"SAMANTHA_LOG_PATH": str(self.path)})
        environment.start()
        self.addCleanup(environment.stop)

    def read(self, session_id="latest"):
        with InteractionJournal(self.path, readonly=True) as journal:
            return journal.read_session(session_id)

    def run_request(self, fake=None, answers=("y",), request="Create a file", **kwargs):
        return process(request, services=fake or services(), input_fn=Mock(side_effect=answers),
                       output_fn=lambda _: None, **kwargs)


class TranscriptTests(JournalTestCase):
    def test_every_turn_prompt_and_output_survives_reopen_verbatim(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "文件名称？"}, {"question": "文件内容？"},
            {"is_jump": False, "requirement_summary": "Create the specified file."},
        ]
        request = " 创建文件\n第二行🙂 "
        replies = [" 测试.txt ", " 内容\n换行🙂 ", " YES "]
        output, prompts = [], []
        def answer(prompt):
            prompts.append(prompt)
            return replies[len(prompts) - 1]
        state = process(request, services=fake, input_fn=answer, output_fn=output.append)
        data = self.read(state["log_session_id"])
        events = data["events"]
        self.assertEqual(data["session"]["status"], "succeeded")
        self.assertIsNotNone(data["session"]["ended_at"])
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        inputs = [e for e in events if e["kind"] == "user_input"]
        self.assertEqual([e["data"]["text"] for e in inputs], [request, *replies])
        self.assertEqual([e["turn"] for e in inputs], [1, 2, 3, 4])
        self.assertEqual([e["data"]["text"] for e in events if e["kind"] == "prompt"], prompts)
        self.assertEqual([e["data"]["text"] for e in events if e["kind"] == "assistant_output"],
                         [text + "\n" for text in output])
        starts = [e["data"]["name"] for e in events if e["kind"] == "operation_started"]
        self.assertEqual(starts.count("agent.clarify"), 3)
        self.assertEqual(starts.count("agent.generate_commands"), 1)
        fake.execute_commands.assert_called_once()
        for event in events:
            self.assertTrue(event["timestamp"].endswith("+00:00"))

    def test_input_and_confirmation_are_committed_before_executor_starts(self):
        fake = services()
        def answer(prompt):
            # Read with another connection while the session is still active.
            data = self.read()
            self.assertEqual(data["session"]["status"], "running")
            self.assertEqual(data["events"][-1]["kind"], "prompt")
            fake.execute_commands.assert_not_called()
            return "yes"
        def execute(commands):
            events = self.read()["events"]
            self.assertIn("yes", [e["data"]["text"] for e in events if e["kind"] == "user_input"])
            self.assertTrue(any(e["kind"] == "workflow_update" and e["data"]["node"] == "confirm"
                                and e["data"]["update"]["approved"] for e in events))
            self.assertEqual(events[-1]["data"]["name"], "commands.execute")
            return {"success": True, "output": "done", "current_dir": os.getcwd()}
        fake.execute_commands.side_effect = execute
        state = process("Create a file", services=fake, input_fn=answer, output_fn=lambda _: None)
        self.assertEqual(state["status"], "succeeded")

    def test_decline_records_plan_and_rejection_without_execution(self):
        fake = services()
        state = self.run_request(fake, answers=("no",))
        events = self.read()["events"]
        self.assertEqual(state["status"], "cancelled")
        fake.execute_commands.assert_not_called()
        self.assertFalse(any(e["kind"] == "operation_started" and
                             e["data"]["name"] in {"commands.execute", "shell.execute"} for e in events))
        self.assertTrue(any(e["kind"] == "assistant_output" and
                            e["data"]["text"] == "touch test.txt\n" for e in events))
        self.assertEqual(self.read()["session"]["status"], "cancelled")

    def test_eof_and_keyboard_interrupt_have_durable_cancelled_outcomes(self):
        for exception in (EOFError, KeyboardInterrupt):
            with self.subTest(exception=exception):
                fake = services()
                state = process("request", services=fake, input_fn=Mock(side_effect=exception),
                                output_fn=lambda _: None)
                events = self.read(state["log_session_id"])["events"]
                self.assertEqual(state["status"], "cancelled")
                fake.execute_commands.assert_not_called()
                self.assertTrue(any(e["kind"] == "interrupted" and
                                    e["data"]["exception_type"] == exception.__name__ for e in events))
                self.assertEqual(len([e for e in events if e["kind"] == "user_input"]), 1)

    def test_retry_keeps_both_command_lists_outputs_and_correction(self):
        fake = services()
        fake.generate_commands.side_effect = ['{"Commands": ["first"]}', '{"Commands": ["second"]}']
        fake.execute_commands.side_effect = [
            {"success": False, "output": "first error\nraw details", "current_dir": "/partial"},
            {"success": True, "output": "second output", "current_dir": "/destination"},
        ]
        self.run_request(fake, answers=("y", "yes"))
        events = self.read()["events"]
        finishes = [e["data"] for e in events if e["kind"] == "operation_finished"]
        executions = [e for e in finishes if e["name"] == "commands.execute"]
        self.assertEqual([e["result"]["output"] for e in executions],
                         ["first error\nraw details", "second output"])
        self.assertEqual(len([e for e in finishes if e["name"] == "agent.correct_error"]), 1)
        self.assertEqual([e["data"]["text"] for e in events if e["kind"] == "user_input"],
                         ["Create a file", "y", "yes"])
        self.assertEqual(events[-1]["data"]["state"]["retries"], 1)

    def test_invalid_agent_result_and_exception_are_recorded(self):
        for response in ("not json", RuntimeError("provider unavailable")):
            with self.subTest(response=response):
                fake = services()
                if isinstance(response, Exception):
                    fake.generate_commands.side_effect = response
                else:
                    fake.generate_commands.return_value = response
                state = self.run_request(fake, answers=())
                data = self.read(state["log_session_id"])
                self.assertEqual(data["session"]["status"], "failed")
                fake.execute_commands.assert_not_called()
                finish = next(e["data"] for e in data["events"] if e["kind"] == "operation_finished"
                              and e["data"]["name"] == "agent.generate_commands")
                if isinstance(response, Exception):
                    self.assertEqual(finish["error"], "provider unavailable")
                    self.assertEqual(finish["status"], "failed")
                else:
                    self.assertEqual(finish["result"], response)
                self.assertIn("Command generation failed", data["session"]["error"])

    def test_shell_log_preserves_raw_stdout_stderr_and_exit_code(self):
        fake = replace(services(), execute_commands=run_commands)
        stdout, stderr = " first line\nsecond line \n", " error\n"
        with patch("run_commands.subprocess.run", return_value=SimpleNamespace(
                returncode=7, stdout=stdout, stderr=stderr)):
            state = self.run_request(fake, answers=("y", "n"))
        shell = next(e["data"] for e in self.read(state["log_session_id"])["events"]
                     if e["kind"] == "operation_finished" and e["data"]["name"] == "shell.execute")
        self.assertEqual(shell["result"], {"returncode": 7, "stdout": stdout, "stderr": stderr})

    def test_log_write_failure_prevents_unrecorded_execution(self):
        fake = services()
        original = LogSession.record
        def fail_confirmation(session, kind, data, **kwargs):
            if kind == "user_input" and data.get("source") == "confirmation":
                raise InteractionLogError("simulated disk failure")
            return original(session, kind, data, **kwargs)
        with patch.object(LogSession, "record", fail_confirmation), \
                self.assertRaisesRegex(InteractionLogError, "disk failure"):
            self.run_request(fake)
        fake.execute_commands.assert_not_called()
        self.assertEqual(self.read()["session"]["status"], "failed")

    def test_unusable_log_path_stops_before_model_or_executor(self):
        self.path.parent.mkdir(parents=True)
        self.path.mkdir()
        fake = services()
        with self.assertRaises(InteractionLogError):
            self.run_request(fake)
        fake.clarify.assert_not_called()
        fake.execute_commands.assert_not_called()

    def test_startup_failure_is_recorded(self):
        with patch("samantha.build_workflow", side_effect=RuntimeError("cannot initialize")), \
                self.assertRaisesRegex(RuntimeError, "cannot initialize"):
            self.run_request()
        self.assertEqual(self.read()["session"]["error"], "cannot initialize")


class StorageTests(JournalTestCase):
    def test_abrupt_process_exit_keeps_committed_events_and_running_status(self):
        code = """
import os
from interaction_log import InteractionJournal, operation
with InteractionJournal() as journal, journal.start_session('before crash') as session:
    session.output('saved output\\n')
    with operation('unfinished.tool', {'path': 'fixture'}):
        os._exit(23)
"""
        environment = {**os.environ, "PYTHONPATH": str(SRC)}
        result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        data = self.read()
        self.assertEqual(data["session"]["status"], "running")
        self.assertIsNone(data["session"]["ended_at"])
        self.assertEqual(data["events"][-1]["kind"], "operation_started")
        self.assertTrue(any(e["data"].get("text") == "saved output\n" for e in data["events"]))

    def test_concurrent_connections_keep_sessions_and_sequences_independent(self):
        with InteractionJournal(self.path):
            pass
        def write(index):
            with InteractionJournal(self.path) as journal, journal.start_session(f"request {index}") as session:
                for item in range(10):
                    session.output(f"{index}:{item}\n")
                session.finish("succeeded")
                return session.session_id
        with ThreadPoolExecutor(max_workers=6) as pool:
            ids = list(pool.map(write, range(12)))
        self.assertEqual(len(set(ids)), 12)
        with InteractionJournal(self.path, readonly=True) as journal:
            self.assertEqual(len(journal.list_sessions()), 12)
            for session_id in ids:
                events = journal.read_session(session_id)["events"]
                self.assertEqual([e["sequence"] for e in events], list(range(1, 14)))

    def test_newer_schema_is_rejected_without_modification(self):
        with InteractionJournal(self.path) as journal:
            journal.connection.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(InteractionLogError, "schema: 99"):
            InteractionJournal(self.path)


class HistoryTests(JournalTestCase):
    def call(self, args):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = history_main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_missing_history_does_not_create_database(self):
        status, output, _ = self.call(["list", "--json"])
        self.assertEqual((status, output.strip()), (0, "[]"))
        self.assertFalse(self.path.exists())

    def test_list_show_and_export_are_complete_and_do_not_create_sessions(self):
        state = self.run_request()
        status, output, _ = self.call(["list", "--json"])
        self.assertEqual(status, 0)
        self.assertEqual(len(json.loads(output)), 1)
        status, output, _ = self.call(["show", "latest", "--json"])
        data = json.loads(output)
        self.assertEqual(data["session"]["session_id"], state["log_session_id"])
        target = self.path.parent / "export.jsonl"
        status, _, error = self.call(["export", "latest", "--format", "jsonl", "--output", str(target)])
        self.assertEqual(status, 0, error)
        lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[0]["session"], data["session"])
        self.assertEqual(lines[1:], data["events"])
        self.assertEqual(len(json.loads(self.call(["list", "--json"])[1])), 1)
        self.assertEqual(self.call(["export", "latest", "--output", str(target)])[0], 1)
        self.assertEqual([json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()], lines)

    def test_json_export_missing_id_and_database_overwrite_guard(self):
        self.run_request()
        status, output, _ = self.call(["export", "latest"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output), self.read())
        self.assertEqual(self.call(["show", "unknown"])[0], 1)
        self.assertEqual(self.call(["list", "--limit", "0"])[0], 1)
        for target in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            self.assertEqual(self.call(["export", "latest", "--output", str(target)])[0], 1)
        self.assertEqual(self.read()["session"]["status"], "succeeded")

    def test_main_dispatch_preserves_argv_and_initial_directory(self):
        with patch("samantha.sys.argv", ["samantha", "history", "show", "latest", "--json"]), \
                patch("history_cli.main", return_value=0) as history, patch("samantha.write_current_dir") as write:
            self.assertEqual(main(), 0)
            history.assert_called_once_with(["show", "latest", "--json"])
            write.assert_called_once_with(os.getcwd())


class ContentLogTests(JournalTestCase):
    def test_content_json_stdout_and_progress_stderr_are_retained(self):
        def index(path, force=False, progress=None):
            progress("Indexing 路径 with spaces.txt")
            return [{"status": "indexed", "path": str(path), "summary": "完整内容🙂", "tags": ["测试"]}]
        stdout, stderr = StringIO(), StringIO()
        with patch("content_cli.ContentPipeline") as pipeline, redirect_stdout(stdout), redirect_stderr(stderr):
            pipeline.return_value.index.side_effect = index
            self.assertEqual(content_main(["index", "路径 with spaces.txt", "--json"]), 0)
        data = self.read()
        self.assertEqual(data["session"]["kind"], "content")
        for name, expected in (("stdout", stdout.getvalue()), ("stderr", stderr.getvalue())):
            actual = "".join(e["data"]["text"] for e in data["events"] if e["kind"] == "assistant_output"
                             and e["data"]["stream"] == name)
            self.assertEqual(actual, expected)
        json.loads(stdout.getvalue())
        self.assertTrue(any(e["kind"] == "operation_finished" and
                            e["data"]["name"] == "content.index" for e in data["events"]))

    def test_content_errors_and_parser_exit_are_logged(self):
        stdout, stderr = StringIO(), StringIO()
        with patch("content_cli.ContentPipeline", side_effect=RuntimeError("store unavailable")), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(content_main(["search", "anything"]), 1)
        self.assertEqual(self.read()["session"]["status"], "failed")
        self.assertIn("store unavailable", stderr.getvalue())
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            content_main(["unknown"])
        self.assertEqual(self.read()["session"]["status"], "failed")

    def test_diagnostics_do_not_collect_configured_api_keys(self):
        with patch.dict(os.environ, {"CONTENT_VL_API_KEY": "do-not-log-this-secret"}), \
                patch("content_cli.diagnose", return_value=[]), redirect_stdout(StringIO()):
            self.assertEqual(content_main(["doctor", "--json"]), 0)
        self.assertNotIn("do-not-log-this-secret", json.dumps(self.read()))


if __name__ == "__main__":
    unittest.main()
