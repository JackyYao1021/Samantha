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

from clarify_intent import clarify_user_intent, parse_clarification_reply
from chat import Chat, request_text, response_text
from command_review import command_action_id
from types import SimpleNamespace
from langgraph.types import Command
from model_output import parse_commands
from samantha import main, process
from workflow import WorkflowServices, build_workflow, initial_state


def review_fixture():
    return {
        "verdict": "reasonable", "risk": "low",
        "summary": "Create test.txt in the current directory.",
        "effects": ["Creates an empty file, or updates its timestamp if it exists."],
        "issues": [], "uncertainties": ["Whether test.txt already exists is unknown."],
        "recommendation": "Proceed if this is the intended file.",
    }


def services():
    return WorkflowServices(
        clarify=Mock(return_value={"is_jump": False, "requirement_summary": "Create a test file."}),
        parse_intent=Mock(return_value="Create test.txt in the current directory."),
        generate_commands=Mock(return_value=json.dumps({
            "Commands": ["touch test.txt"], "Explanation": "Create a file."
        })),
        review_commands=Mock(return_value=review_fixture()),
        correct_error=Mock(return_value="Create the missing parent folder, then the file."),
        execute_commands=Mock(return_value={
            "success": True, "output": "done", "current_dir": "/destination"
        }),
    )


def start(fake_services, max_retries=3, graph=None, thread_id=None, max_clarifications=3):
    graph = graph if graph is not None else build_workflow(fake_services)
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 100}
    return graph, config, graph.invoke(initial_state(
        "Create a file", max_retries=max_retries, max_clarifications=max_clarifications), config)


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
        fake.review_commands.assert_called_once()
        self.assertEqual(pending["review"], review_fixture())
        self.assertIn("Whether test.txt already exists is unknown.", pending["message"])
        self.assertEqual(pending["cwd"], os.getcwd())
        self.assertEqual(pending["action_id"], command_action_id(
            pending["commands"], pending["cwd"], pending["shell"]))

    def test_reviewer_receives_request_plan_commands_directory_and_previous_error(self):
        fake = services()
        fake.execute_commands.side_effect = [
            {"success": False, "output": "missing folder", "current_dir": os.getcwd()},
            {"success": True, "output": "done", "current_dir": os.getcwd()},
        ]
        fake.generate_commands.side_effect = [
            '{"Commands": ["touch missing/test.txt"]}',
            '{"Commands": ["mkdir -p missing", "touch missing/test.txt"]}',
        ]
        graph, config, first = start(fake)
        request = fake.review_commands.call_args.args[0]
        self.assertEqual(request["user_request"], "Create a file")
        self.assertEqual(request["clarified_request"], "Create a test file.")
        self.assertEqual(request["plan"], "Create test.txt in the current directory.")
        self.assertEqual(request["commands"], ["touch missing/test.txt"])
        self.assertEqual(request["cwd"], os.getcwd())
        self.assertEqual(request["previous_error"], "")
        second = graph.invoke(Command(resume="y"), config)
        self.assertEqual(fake.review_commands.call_count, 2)
        request = fake.review_commands.call_args.args[0]
        self.assertEqual(request["commands"], ["mkdir -p missing", "touch missing/test.txt"])
        self.assertEqual(request["previous_error"], "missing folder")
        self.assertNotEqual(first["reviewed_action_id"], second["reviewed_action_id"])
        self.assertFalse(second["approved"])
        self.assertEqual(second["approved_action_id"], "")
        fake.execute_commands.assert_called_once()

    def test_review_problems_are_displayed_and_human_can_decline(self):
        fake = services()
        fake.review_commands.return_value = {**review_fixture(), "verdict": "issues_found",
            "risk": "high", "issues": ["Deletes a file instead of creating it."],
            "recommendation": "Reject and revise the commands."}
        graph, config, state = start(fake)
        self.assertIn("Deletes a file instead of creating it.", state["__interrupt__"][0].value["message"])
        fake.execute_commands.assert_not_called()
        state = graph.invoke(Command(resume="n"), config)
        self.assertEqual(state["status"], "cancelled")
        fake.execute_commands.assert_not_called()

    def test_invalid_review_never_reaches_confirmation_or_execution(self):
        for review in (None, {}, {**review_fixture(), "summary": ""},
                       {**review_fixture(), "verdict": "approved"},
                       {**review_fixture(), "verdict": "issues_found", "issues": []}):
            with self.subTest(review=review):
                fake = services()
                fake.review_commands.return_value = review
                _, _, state = start(fake)
                self.assertEqual(state["status"], "failed")
                self.assertIn("Command review failed", state["error"])
                self.assertNotIn("__interrupt__", state)
                fake.execute_commands.assert_not_called()

    def test_changed_commands_cannot_use_the_original_confirmation(self):
        fake = services()
        graph, config, _ = start(fake)
        graph.update_state(config, {"commands": ["rm test.txt"]})
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("reviewed action has changed", state["error"])
        fake.execute_commands.assert_not_called()

    def test_changed_directory_or_shell_cannot_use_the_original_confirmation(self):
        for change in ("cwd", "shell"):
            with self.subTest(change=change):
                fake = services()
                graph, config, _ = start(fake)
                context = (patch("workflow.os.getcwd", return_value="/different") if change == "cwd"
                           else patch.dict(os.environ, {"SAMANTHA_BASH": "different-bash"}))
                with context:
                    state = graph.invoke(Command(resume="y"), config)
                self.assertEqual(state["status"], "failed")
                fake.execute_commands.assert_not_called()

    def test_executor_checks_identity_even_after_approval_node_has_completed(self):
        fake = services()
        graph, config, state = start(fake)
        graph.update_state(config, {
            "approved": True, "approved_action_id": state["reviewed_action_id"],
            "commands": ["rm test.txt"],
        }, as_node="confirm")
        state = graph.invoke(None, config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("approved action has changed", state["error"])
        self.assertEqual(state["approved_action_id"], "")
        fake.execute_commands.assert_not_called()

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
        prompt, history = fake.clarify.call_args.args
        self.assertIn("Original task: Create a file", prompt)
        self.assertIn("Previous clarification question: What content?", prompt)
        self.assertIn("User answer to that question: hello", prompt)
        self.assertIn('"question": "What file name?", "answer": "test.txt"', prompt)
        self.assertIn('Resolved file name supplied by the user: "test.txt"', prompt)
        self.assertEqual(history, [
            {"role": "user", "content": "Create a file"},
            {"role": "assistant", "content": "What file name?"},
            {"role": "user", "content": "test.txt"},
            {"role": "assistant", "content": "What content?"},
        ])
        fake.execute_commands.assert_not_called()

    def test_short_answer_is_attached_to_current_task_with_prior_terminal_history(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What should the file be named?"},
            {"is_jump": False, "requirement_summary": "Write binary search in binary_search.py."},
        ]
        prior = [{"role": "user", "content": "List my downloads."}]
        request = "help me create a py file and wrote a binary search code in it"
        graph = build_workflow(fake)
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        graph.invoke(initial_state(request, prior), config)
        state = graph.invoke(Command(resume="binary_search"), config)
        prompt, history = fake.clarify.call_args.args
        self.assertIn("Original task: " + request, prompt)
        self.assertIn("Previous clarification question: What should the file be named?", prompt)
        self.assertIn("User answer to that question: binary_search", prompt)
        self.assertIn('Resolved file name supplied by the user: "binary_search.py"', prompt)
        self.assertNotIn("List my downloads.", prompt)
        self.assertEqual(history, prior + [
            {"role": "user", "content": request},
            {"role": "assistant", "content": "What should the file be named?"},
        ])
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        self.assertEqual(state["clarification_answers"], [
            {"question": "What should the file be named?", "answer": "binary_search"},
        ])
        fake.parse_intent.assert_called_once_with("Write binary search in binary_search.py.")
        fake.execute_commands.assert_not_called()

    def test_empty_answers_reprompt_without_model_calls_or_history_pollution(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"is_jump": False, "requirement_summary": "Create test.txt."},
        ]
        graph, config, state = start(fake)
        original_history = state["clarification_history"]
        for answer in ("", " \t "):
            state = graph.invoke(Command(resume=answer), config)
            self.assertIn("non-empty answer", state["__interrupt__"][0].value["message"])
            self.assertEqual(state["clarification_history"], original_history)
            self.assertEqual(state["clarification_count"], 1)
            self.assertEqual(state["clarification_answers"], [])
            self.assertEqual(fake.clarify.call_count, 1)
        state = graph.invoke(Command(resume="test.txt"), config)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        self.assertEqual(fake.clarify.call_count, 2)
        fake.execute_commands.assert_not_called()

    def test_explicit_file_extension_and_directory_names_are_preserved(self):
        for question, answer, kind in (
            ("What file name?", "notes.txt", "file"),
            ("What folder name?", "scripts", "directory"),
        ):
            with self.subTest(answer=answer):
                fake = services()
                fake.clarify.side_effect = [
                    {"question": question},
                    {"is_jump": False, "requirement_summary": "Create the requested item."},
                ]
                graph, config, _ = start(fake)
                graph.invoke(Command(resume=answer), config)
                prompt = fake.clarify.call_args.args[0]
                self.assertIn(f'Resolved {kind} name supplied by the user: "{answer}"', prompt)
                self.assertNotIn(answer + ".py", prompt)

    def test_sentences_and_uncertain_answers_are_not_labeled_as_literal_names(self):
        for answer in ("I do not know", "unknown", "Please name it hello.py"):
            with self.subTest(answer=answer):
                fake = services()
                fake.clarify.side_effect = [
                    {"question": "What file name?"},
                    {"is_jump": False, "requirement_summary": "Create the requested file."},
                ]
                graph, config, _ = start(fake)
                graph.invoke(Command(resume=answer), config)
                self.assertNotIn("Resolved file name", fake.clarify.call_args.args[0])

    def test_three_empty_answers_end_without_another_model_call(self):
        fake = services()
        fake.clarify.return_value = {"question": "What file name?"}
        graph, config, _ = start(fake)
        for _ in range(3):
            state = graph.invoke(Command(resume=""), config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("3 attempts", state["error"])
        self.assertNotIn("__interrupt__", state)
        fake.clarify.assert_called_once()
        fake.execute_commands.assert_not_called()

    def test_cancel_after_empty_answer_never_calls_model_again(self):
        fake = services()
        fake.clarify.return_value = {"question": "What file name?"}
        graph, config, _ = start(fake)
        graph.invoke(Command(resume=""), config)
        state = graph.invoke(Command(resume="取消"), config)
        self.assertEqual(state["status"], "cancelled")
        fake.clarify.assert_called_once()
        fake.execute_commands.assert_not_called()

    def test_repeated_answered_question_fails_without_execution(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"question": "  WHAT  FILE NAME ! "},
        ]
        graph, config, _ = start(fake)
        state = graph.invoke(Command(resume="binary_search"), config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("repeated a question", state["error"])
        self.assertNotIn("__interrupt__", state)
        self.assertEqual(state["retries"], 0)
        fake.parse_intent.assert_not_called()
        fake.execute_commands.assert_not_called()

    def test_clarification_limit_bounds_different_questions(self):
        fake = services()
        fake.clarify.side_effect = [{"question": f"Missing detail {i}?"} for i in range(4)]
        graph, config, _ = start(fake)
        for i in range(3):
            state = graph.invoke(Command(resume=f"answer {i}"), config)
        self.assertEqual(state["status"], "failed")
        self.assertIn("Clarification limit (3)", state["error"])
        self.assertEqual(state["clarification_count"], 3)
        self.assertEqual(state["retries"], 0)
        self.assertNotIn("__interrupt__", state)
        fake.execute_commands.assert_not_called()

    def test_final_allowed_answer_can_complete_task(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"is_jump": False, "requirement_summary": "Create test.txt."},
        ]
        graph, config, _ = start(fake, max_clarifications=1)
        state = graph.invoke(Command(resume="test.txt"), config)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        state = graph.invoke(Command(resume="y"), config)
        self.assertEqual(state["status"], "succeeded")
        fake.execute_commands.assert_called_once()

    def test_zero_clarifications_still_allows_a_complete_task(self):
        fake = services()
        _, _, state = start(fake, max_clarifications=0)
        self.assertEqual(state["__interrupt__"][0].value["kind"], "confirmation")
        fake = services()
        fake.clarify.return_value = {"question": "What file name?"}
        _, _, state = start(fake, max_clarifications=0)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("__interrupt__", state)
        fake.execute_commands.assert_not_called()

    def test_invalid_clarification_limits_are_rejected(self):
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                initial_state("Create a file", max_clarifications=limit)

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
                fake.review_commands.assert_not_called()
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
        self.assertEqual(fake.review_commands.call_count, 4)
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
        for agent in ("clarify", "parse_intent", "generate_commands", "review_commands"):
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


class StandaloneClarificationTests(unittest.TestCase):
    def test_short_reply_uses_original_task_and_skips_empty_answer(self):
        response = {"is_jump": False, "requirement_summary": "Write binary search in binary_search.py."}
        with patch("clarify_intent.clarify_intent_once", side_effect=[
            {"question": "What file name?"}, response,
        ]) as clarify, patch("builtins.input", side_effect=["", "binary_search"]), patch("builtins.print"):
            result = clarify_user_intent("Write binary search in a Python file.")
        self.assertEqual(result, (False, response["requirement_summary"]))
        self.assertEqual(clarify.call_count, 2)
        self.assertIn("Original task: Write binary search in a Python file.", clarify.call_args.args[0])
        self.assertIn("User answer to that question: binary_search", clarify.call_args.args[0])

    def test_repeated_question_is_bounded(self):
        with patch("clarify_intent.clarify_intent_once", return_value={"question": "What file name?"}), \
                patch("builtins.input", return_value="test.txt"), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "repeated a question"):
                clarify_user_intent("Create a file.")


class TerminalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {
            "SAMANTHA_LOG_PATH": str(Path(directory.name) / "interactions.sqlite3"),
        })
        environment.start()
        self.addCleanup(environment.stop)

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
        self.assertTrue(any("# Command Review" in text for text in output))
        self.assertIn("Working directory: " + os.getcwd(), output)
        self.assertEqual(fake.clarify.call_count, 2)
        fake.execute_commands.assert_called_once()

    def test_cli_reprompts_empty_answer_then_accepts_short_reply(self):
        fake = services()
        fake.clarify.side_effect = [
            {"question": "What file name?"},
            {"is_jump": False, "requirement_summary": "Create test.txt."},
        ]
        output = []
        answers = Mock(side_effect=["", "test.txt", "n"])
        state = process("Create a file", services=fake, input_fn=answers, output_fn=output.append)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(answers.call_count, 3)
        self.assertEqual(fake.clarify.call_count, 2)
        self.assertTrue(any("non-empty answer" in text for text in output))
        fake.execute_commands.assert_not_called()

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
    def test_qwen3_soft_switch_does_not_mutate_history_or_azure_fallback(self):
        messages = [{"role": "system", "content": "instructions"},
                    {"role": "user", "content": "hello"}]
        environment = {"SAMANTHA_AZURE_FALLBACK": "true", "AZURE_OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure:
            qwen.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
            Chat().send_message(messages)
            sent = qwen.return_value.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual(sent[-1]["content"], "hello\n/no_think")
            self.assertEqual(messages[-1]["content"], "hello")
            self.assertEqual(azure.return_value.chat.completions.create.call_args.kwargs["messages"], messages)

    def test_soft_switch_respects_model_and_thinking_configuration(self):
        for model, effort, content in (("qwen3.5:4b", "none", "hello"),
                                       ("qwen3:4b", "high", "hello"),
                                       ("qwen3:4b", "", "hello"),
                                       ("qwen3:4b", "none", "hello\n/no_think")):
            environment = {"QWEN_MODEL": model, "QWEN_REASONING_EFFORT": effort}
            with self.subTest(model=model, effort=effort), \
                    patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen:
                Chat().send_message([{"role": "user", "content": content}])
                sent = qwen.return_value.chat.completions.create.call_args.kwargs["messages"]
                self.assertEqual(sent[-1]["content"], content)

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


class OllamaTransportTests(unittest.TestCase):
    def test_native_chat_separates_thinking_and_keeps_generation_settings(self):
        environment = {"QWEN_API_MODE": "ollama", "QWEN_TIMEOUT": "30"}
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.httpx.Client") as http:
            qwen.return_value.base_url = "http://host.docker.internal:11434/v1/"
            response = http.return_value.__enter__.return_value.post.return_value
            response.json.return_value = {"done": True, "done_reason": "stop",
                                          "message": {"content": '{"ok": true}', "thinking": "private analysis"}}
            chat = Chat(max_tokens=128, temperature=0.2, top_p=0.8,
                        response_format={"type": "json_object"})
            self.assertEqual(chat.chat_context("hello"), '{"ok": true}')
            call = http.return_value.__enter__.return_value.post.call_args
            self.assertEqual(call.args[0], "http://host.docker.internal:11434/api/chat")
            self.assertFalse(call.kwargs["json"]["think"])
            self.assertEqual(call.kwargs["json"]["format"], "json")
            self.assertEqual(call.kwargs["json"]["options"],
                             {"num_predict": 128, "temperature": 0.2, "top_p": 0.8})
            self.assertEqual(chat.messages[-1]["content"], '{"ok": true}')
            http.assert_called_once_with(timeout=30.0, trust_env=False)
            qwen.return_value.chat.completions.create.assert_not_called()

    def test_native_schema_and_truncated_answers_are_not_accepted(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with patch.dict(os.environ, {"QWEN_API_MODE": "ollama"}, clear=True), \
                patch("chat.OpenAI") as qwen, patch("chat.httpx.Client") as http:
            qwen.return_value.base_url = "http://localhost:11434/v1/"
            response = http.return_value.__enter__.return_value.post.return_value
            response.json.return_value = {"done": True, "done_reason": "length",
                                          "message": {"content": '{"ok":'}}
            chat = Chat(response_format={"type": "json_schema", "json_schema": {"schema": schema}})
            with self.assertRaisesRegex(ValueError, "token limit"):
                chat.chat_context("hello")
            payload = http.return_value.__enter__.return_value.post.call_args.kwargs["json"]
            self.assertEqual(payload["format"], schema)
            self.assertEqual(chat.messages[-1]["role"], "user")
            response.json.return_value = {"done": False, "message": {"content": "partial"}}
            with self.assertRaisesRegex(RuntimeError, "incomplete response"):
                chat.chat_context("hello again")

    def test_native_failure_uses_only_explicit_azure_fallback_with_original_request(self):
        environment = {"QWEN_API_MODE": "ollama", "SAMANTHA_AZURE_FALLBACK": "true",
                       "AZURE_OPENAI_API_KEY": "test-key"}
        messages = [{"role": "user", "content": "hello"}]
        with patch.dict(os.environ, environment, clear=True), patch("chat.OpenAI") as qwen, \
                patch("chat.AzureOpenAI") as azure, patch("chat.httpx.Client") as http:
            qwen.return_value.base_url = "http://localhost:11434/v1/"
            http.return_value.__enter__.return_value.post.side_effect = RuntimeError("unavailable")
            Chat().send_message(messages)
            sent = azure.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(sent["messages"], messages)
            self.assertNotIn("extra_body", sent)


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
