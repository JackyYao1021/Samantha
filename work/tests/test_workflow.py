"""Behavior tests against the real LangGraph runtime, with fake agents/tools."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clarify_intent import parse_clarification_reply
from chat import Chat, request_text, response_text
from types import SimpleNamespace
from langgraph.types import Command
from model_output import parse_commands
from samantha import main, process
from workflow import WorkflowServices, build_workflow, initial_state


def services():
    return WorkflowServices(
        clarify=Mock(return_value={"is_jump": False, "requirement_summary": "Create a test file."}),
        parse_intent=Mock(return_value="Create test.txt in the current directory."),
        generate_commands=Mock(return_value=json.dumps({
            "Commands": ["touch test.txt"], "Explanation": "Create a file."
        })),
        explain_commands=Mock(return_value="Create test.txt. Proceed? (y/n)"),
        correct_error=Mock(return_value="Create the missing parent folder, then the file."),
        execute_commands=Mock(return_value={
            "success": True, "output": "done", "current_dir": "/destination"
        }),
    )


def start(fake_services, max_retries=3, graph=None, thread_id=None):
    graph = graph if graph is not None else build_workflow(fake_services)
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 100}
    return graph, config, graph.invoke(initial_state("Create a file", max_retries=max_retries), config)


class WorkflowTests(unittest.TestCase):
    def test_execution_requires_confirmation_and_resume_does_not_repeat_model_calls(self):
        fake = services()
        graph, config, state = start(fake)
        pending = state["__interrupt__"][0].value
        self.assertEqual(pending["kind"], "confirmation")
        self.assertEqual(pending["commands"], ["touch test.txt"])
        fake.execute_commands.assert_not_called()
        state = graph.invoke(Command(resume="yes"), config)
        self.assertEqual(state["status"], "succeeded")
        self.assertFalse(state["approved"])
        fake.execute_commands.assert_called_once_with(["touch test.txt"])
        fake.clarify.assert_called_once()
        fake.parse_intent.assert_called_once()
        fake.generate_commands.assert_called_once()
        fake.explain_commands.assert_called_once()

    def test_rejection_or_unrecognized_answer_never_executes(self):
        for answer in ("n", "no", "", "maybe", False, 1, {"approved": True}):
            with self.subTest(answer=answer):
                fake = services()
                graph, config, _ = start(fake)
                state = graph.invoke(Command(resume=answer), config)
                self.assertEqual(state["status"], "cancelled")
                fake.execute_commands.assert_not_called()

    def test_multiple_clarifications_preserve_history(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"question": "What content?"},
            {"is_jump": False, "requirement_summary": "Write hello to test.txt."},
        ]
        graph, config, state = start(fake)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "clarification")
        state = graph.invoke(Command(resume="test.txt"), config)
        self.assertEqual(state["__interrupt__"][0].value["message"], "What content?")
        state = graph.invoke(Command(resume="hello"), config)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        self.assertEqual(fake.clarify.call_count, 3)
        self.assertEqual(fake.clarify.call_args.args, ("hello", [
            {"role": "user", "content": "Create a file"},
            {"role": "assistant", "content": "What file name?"},
            {"role": "user", "content": "test.txt"},
            {"role": "assistant", "content": "What content?"},
        ]))
        fake.execute_commands.assert_not_called()

    def test_stop_during_clarification_ends_without_another_model_call(self):
        fake = services()
        fake.clarify.return_value = {"question": "What file name?"}
        graph, config, _ = start(fake)
        state = graph.invoke(Command(resume="stop"), config)
        self.assertEqual(state["status"], "cancelled")
        fake.clarify.assert_called_once()
        fake.parse_intent.assert_not_called()
        fake.execute_commands.assert_not_called()

    def test_model_termination_signal_ends_graph(self):
        fake = services()
        fake.clarify.return_value = {"cancelled": True}
        _, _, state = start(fake)
        self.assertEqual(state["status"], "cancelled")
        self.assertNotIn("__interrupt__", state)
        fake.parse_intent.assert_not_called()

    def test_invalid_command_output_cannot_reach_confirmation_or_execution(self):
        for response in ("not json", "[]", "{}", '{"Commands": []}',
                         '{"Commands": "touch file"}', '{"Commands": [2]}',
                         '{"Commands": [" "]}', '{"Commands": ["pwd"], "Explanation": 1}'):
            with self.subTest(response=response):
                fake = services()
                fake.generate_commands.return_value = response
                _, _, state = start(fake)
                self.assertEqual(state["status"], "failed")
                self.assertIn("Command generation failed", state["error"])
                self.assertNotIn("__interrupt__", state)
                fake.explain_commands.assert_not_called()
                fake.execute_commands.assert_not_called()

    def test_retry_limit_is_three_corrections_and_four_approved_executions(self):
        fake = services()
        fake.execute_commands.return_value = {
            "success": False, "output": "missing folder", "current_dir": "/partial"
        }
        fake.generate_commands.side_effect = [json.dumps({"Commands": [f"echo attempt-{i}"]})
                                               for i in range(4)]
        graph, config, state = start(fake)
        for attempt in range(4):
            self.assertEqual(state["__interrupt__"][0].value["retries"], attempt)
            self.assertEqual(fake.execute_commands.call_count, attempt)
            state = graph.invoke(Command(resume="YES"), config)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["retries"], 3)
        self.assertEqual(state["current_dir"], os.getcwd())
        self.assertNotIn("__interrupt__", state)
        self.assertEqual(fake.correct_error.call_count, 3)
        self.assertEqual(fake.generate_commands.call_count, 4)
        self.assertEqual(fake.explain_commands.call_count, 4)
        self.assertEqual(fake.execute_commands.call_count, 4)

    def test_retry_can_succeed_and_must_be_confirmed_again(self):
        fake = services()
        fake.execute_commands.side_effect = [
            {"success": False, "output": "missing folder", "current_dir": os.getcwd()},
            {"success": True, "output": "done", "current_dir": "/destination"},
        ]
        graph, config, _ = start(fake)
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        self.assertEqual(fake.execute_commands.call_count, 1)
        fake.correct_error.assert_called_once_with("Create a test file.", ["touch test.txt"], "missing folder")
        state = graph.invoke(Command(resume=True), config)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["retries"], 1)
        self.assertEqual(fake.execute_commands.call_count, 2)

    def test_cancelling_retry_prevents_second_execution(self):
        fake = services()
        fake.execute_commands.return_value = {"success": False, "output": "failed", "current_dir": os.getcwd()}
        graph, config, _ = start(fake)
        graph.invoke(Command(resume="y"), config)
        state = graph.invoke(Command(resume="n"), config)
        self.assertEqual(state["status"], "cancelled")
        fake.execute_commands.assert_called_once()

    def test_zero_retries_stops_after_first_failure(self):
        fake = services()
        fake.execute_commands.return_value = {"success": False, "output": "failed", "current_dir": os.getcwd()}
        graph, config, _ = start(fake, max_retries=0)
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["status"], "failed")
        fake.correct_error.assert_not_called()

    def test_agent_exceptions_stop_cleanly(self):
        for agent in ("clarify", "parse_intent", "generate_commands", "explain_commands"):
            with self.subTest(agent=agent):
                fake = services()
                getattr(fake, agent).side_effect = RuntimeError("provider unavailable")
                _, _, state = start(fake)
                self.assertEqual(state["status"], "failed")
                self.assertIn("provider unavailable", state["error"])
                fake.execute_commands.assert_not_called()

    def test_error_correction_exception_stops_cleanly(self):
        fake = services()
        fake.execute_commands.return_value = {"success": False, "output": "failed", "current_dir": os.getcwd()}
        fake.correct_error.side_effect = RuntimeError("provider unavailable")
        graph, config, _ = start(fake)
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("Error correction failed", state["error"])
        fake.execute_commands.assert_called_once()

    def test_executor_exception_stops_cleanly(self):
        fake = services()
        fake.execute_commands.side_effect = RuntimeError("executor unavailable")
        graph, config, _ = start(fake)
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["status"], "failed")
        fake.correct_error.assert_not_called()

    def test_directory_jump_respects_user_choice(self):
        for jump in (False, True):
            with self.subTest(jump=jump):
                fake = services()
                fake.clarify.return_value = {"is_jump": jump, "requirement_summary": "Go to destination."}
                graph, config, _ = start(fake)
                state = graph.invoke(Command(resume="y"), config)
                self.assertEqual(state["current_dir"], "/destination" if jump else os.getcwd())

    def test_threads_have_independent_approvals(self):
        fake = services()
        graph = build_workflow(fake)
        _, first, _ = start(fake, graph=graph, thread_id="first")
        _, second, _ = start(fake, graph=graph, thread_id="second")
        graph.invoke(Command(resume="y"), first)
        self.assertEqual(graph.get_state(second).values["status"], "running")
        self.assertTrue(graph.get_state(second).next)
        state = graph.invoke(Command(resume="n"), second)
        self.assertEqual(state["status"], "cancelled")
        fake.execute_commands.assert_called_once()


class TerminalTests(unittest.TestCase):
    def test_cli_drives_clarification_then_confirmation(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"is_jump": False, "requirement_summary": "Create test.txt."},
        ]
        output = []
        answers = Mock(side_effect=["test.txt", "y"])
        state = process("Create a file", services=fake, input_fn=answers, output_fn=output.append)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(answers.call_count, 2)
        self.assertIn("touch test.txt", output)
        self.assertEqual(fake.clarify.call_count, 2)
        fake.execute_commands.assert_called_once()

    def test_cancel_overwrites_old_directory_state(self):
        fake = services()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text('{"current_dir": "/stale"}', encoding="utf-8")
            state = process("Create a file", services=fake, input_fn=lambda _: "n",
                            output_fn=lambda _: None, state_file=path)
            self.assertEqual(state["status"], "cancelled")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["current_dir"], os.getcwd())
        fake.execute_commands.assert_not_called()

    def test_eof_or_keyboard_interrupt_cancels_without_execution(self):
        for exception in (EOFError, KeyboardInterrupt):
            with self.subTest(exception=exception):
                fake = services()
                state = process("Create a file", services=fake, input_fn=Mock(side_effect=exception),
                                output_fn=lambda _: None)
                self.assertEqual(state["status"], "cancelled")
                fake.execute_commands.assert_not_called()

    def test_success_writes_destination_directory(self):
        fake = services()
        fake.clarify.return_value = {"is_jump": True, "requirement_summary": "Go to destination."}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            process("Go to destination", services=fake, input_fn=lambda _: "yes",
                    output_fn=lambda _: None, state_file=path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["current_dir"], "/destination")

    def test_cli_exit_status(self):
        with patch("samantha.sys.argv", ["samantha"]), patch("builtins.print"):
            self.assertEqual(main(), 2)
        for status, expected in (("succeeded", 0), ("cancelled", 0), ("failed", 1)):
            with self.subTest(status=status), patch("samantha.sys.argv", ["samantha", "request"]), \
                    patch("samantha.process", return_value={"status": status}):
                self.assertEqual(main(), expected)


class ModelOutputTests(unittest.TestCase):
    def test_fenced_commands_are_accepted(self):
        self.assertEqual(parse_commands('```json\n{"Commands": ["pwd"]}\n```'), (["pwd"], ""))

    def test_clarification_reply_kinds(self):
        self.assertEqual(parse_clarification_reply("What file name?"), {"question": "What file name?"})
        self.assertEqual(parse_clarification_reply("```--end of chat--```"), {"cancelled": True})
        self.assertEqual(parse_clarification_reply('{"question": "What file name?"}'),
                         {"question": "What file name?"})
        self.assertEqual(parse_clarification_reply('{"cancelled": true}'), {"cancelled": True})
        response = parse_clarification_reply('```json\n{"is_jump": true, "requirement_summary": "Go home."}\n```')
        self.assertEqual(response, {"is_jump": True, "requirement_summary": "Go home."})

    def test_invalid_clarification_is_rejected(self):
        for response in ('{"is_jump": "false", "requirement_summary": "Go home."}',
                         '{"is_jump": false}', '{"is_jump": false, "requirement_summary": ""}',
                         '{"is_jump": false,', '{"question": ""}', '{"question": 1}',
                         '{"cancelled": "true"}', "[]", ""):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_clarification_reply(response)

    def test_complete_schema_reply_with_empty_question_is_accepted(self):
        response = parse_clarification_reply(json.dumps({
            "question": "", "requirement_summary": "Create hello.txt here.",
            "is_jump": False, "cancelled": False,
        }))
        self.assertEqual(response, {"is_jump": False, "requirement_summary": "Create hello.txt here."})


class ChatConfigurationTests(unittest.TestCase):
    def test_qwen_only_configuration_and_generation_parameters(self):
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure:
            chat = Chat(max_tokens=128, temperature=0.2, top_p=0.8)
            chat.send_message([{"role": "user", "content": "hello"}])
            azure.assert_not_called()
            kwargs = qwen.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["max_tokens"], 128)
            self.assertEqual(kwargs["temperature"], 0.2)
            self.assertEqual(kwargs["top_p"], 0.8)

    def test_qwen_failure_without_azure_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI") as qwen:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
            with self.assertRaisesRegex(RuntimeError, "Azure fallback is not configured"):
                Chat().send_message([])

    def test_qwen_failure_uses_configured_azure_fallback(self):
        environment = {
            "SAMANTHA_AZURE_FALLBACK": "true",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
        }
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
            Chat().send_message([])
            self.assertEqual(azure.call_args.kwargs["azure_endpoint"], environment["AZURE_OPENAI_ENDPOINT"])
            self.assertEqual(azure.return_value.chat.completions.create.call_args.kwargs["model"], "test-deployment")

    def test_defaults_target_local_ollama(self):
        with patch.dict(os.environ, {}, clear=True), patch("chat.OpenAI") as qwen:
            chat = Chat()
            self.assertEqual(chat.model_qwen, "qwen3:4b")
            self.assertEqual(qwen.call_args.kwargs["base_url"], "http://127.0.0.1:11434/v1")
            self.assertEqual(qwen.call_args.kwargs["api_key"], "ollama")
            self.assertEqual(qwen.call_args.kwargs["timeout"], 120)
            chat.send_message([])
            self.assertEqual(qwen.return_value.chat.completions.create.call_args.kwargs["extra_body"],
                             {"reasoning_effort": "none"})

    def test_azure_key_alone_does_not_enable_cloud_fallback(self):
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}, clear=True), \
                patch("chat.OpenAI"), patch("chat.AzureOpenAI") as azure:
            self.assertIsNone(Chat().client)
            azure.assert_not_called()

    def test_model_settings_can_be_overridden_and_json_mode_is_sent(self):
        with patch.dict(os.environ, {
            "QWEN_BASE_URL": "http://host.docker.internal:11434/v1",
            "QWEN_MODEL": "custom-model", "QWEN_TIMEOUT": "30", "QWEN_REASONING_EFFORT": "",
        }, clear=True), patch("chat.OpenAI") as qwen:
            chat = Chat(response_format={"type": "json_object"})
            chat.send_message([])
            kwargs = qwen.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "custom-model")
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})
            self.assertNotIn("extra_body", kwargs)


class ResponseTextTests(unittest.TestCase):
    @staticmethod
    def response(content, finish="stop"):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish, message=SimpleNamespace(content=content)
        )])

    def test_thinking_is_removed_before_parsing_or_saving_history(self):
        for content in ('<think>reasoning</think>\n{"ok": true}',
                        'reasoning without an opening tag</think>\n{"ok": true}'):
            with self.subTest(content=content):
                self.assertEqual(response_text(self.response(content)), '{"ok": true}')

    def test_literal_think_tag_inside_json_is_preserved(self):
        content = '{"Commands": ["echo </think>"]}'
        self.assertEqual(response_text(self.response(content)), content)

    def test_empty_thinking_only_and_truncated_answers_are_rejected(self):
        for content, finish in ((None, "stop"), ("", "stop"), ("<think>unfinished", "stop"),
                                ("reasoning</think>", "stop"), ('{"partial":', "length")):
            with self.subTest(content=content), self.assertRaises(ValueError):
                response_text(self.response(content, finish))

    def test_text_envelope_preserves_markdown_and_uses_json_mode(self):
        with patch("chat.Chat") as chat:
            chat.return_value.chat_context.return_value = '{"text": "# Plan\\n1. Create a file."}'
            self.assertEqual(request_text("Agent instructions", "request"), "# Plan\n1. Create a file.")
            self.assertEqual(chat.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_invalid_text_envelope_is_rejected(self):
        for response in ('{}', '{"text": ""}', '{"text": 1}'):
            with self.subTest(response=response), patch("chat.Chat") as chat:
                chat.return_value.chat_context.return_value = response
                with self.assertRaises(ValueError):
                    request_text("instructions", "request")


class ExecutorConfigurationTests(unittest.TestCase):
    def test_bash_override_uses_an_explicit_argument_list(self):
        from run_commands import run_commands
        result = SimpleNamespace(returncode=0, stdout="hello\n/destination\n", stderr="")
        with patch.dict(os.environ, {"SAMANTHA_BASH": "custom-bash"}), \
                patch("run_commands.subprocess.run", return_value=result) as run:
            output = run_commands(["echo hello"])
            self.assertEqual(run.call_args.args[0], ["custom-bash", "-c", "echo hello && pwd"])
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertEqual(output, {"success": True, "output": "hello", "current_dir": "/destination"})


if __name__ == "__main__":
    unittest.main()
