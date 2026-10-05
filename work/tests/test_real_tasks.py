"""Validate fixture isolation and result grading, without live model calls."""

import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS / "real_tasks"))
sys.path.insert(0, str(TESTS.parent / "src"))

from interaction_log import InteractionJournal
from samantha import process
from task_suite import fixture_path, load_cases, prepare, prepare_batch, snapshot, verify
from command_review import command_action_id
from workflow import WorkflowServices


class RealTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cases = {case["id"]: case for case in load_cases()}

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, identifier):
        return prepare(self.cases[identifier], self.root)

    def session(self, run, *, output="", sources=None, request=None, executions=1):
        case = self.cases[run.name]
        is_content = case["mode"] == "content"
        if request is None:
            import shlex
            request = shlex.join(["content", *case["commands"][-1]]) if is_content else case["request"]
        original = Path.cwd()
        try:
            os.chdir(run / "workspace")
            with InteractionJournal(run / "test-log.sqlite3") as journal:
                with journal.start_session(request, kind="content" if is_content else "workflow") as session:
                    if not is_content:
                        commands = ["fixture-operation"]
                        pending = {"kind": "confirmation", "commands": commands,
                                   "cwd": str(run / "workspace"), "shell": "/bin/bash",
                                   "action_id": command_action_id(commands, str(run / "workspace"), "/bin/bash")}
                        session.record("awaiting_input", pending)
                        session.user_input("y" if executions else "n", "confirmation")
                        for _ in range(executions):
                            session.record("operation_started", {"name": "commands.execute", "arguments": {"args": [commands]}})
                    else:
                        session.record("operation_finished", {"name": "content.ask", "result": {"answer": output, "sources": sources or []}})
                    session.finish(case["session"]["status"], details={"state": {
                        "current_dir": str(run / "workspace"), "result": {"output": output},
                    }})
                data = journal.read_session(session.session_id)
        finally:
            os.chdir(original)
        export = run / "session.json"
        export.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return export

    def test_catalog_fixtures_include_real_documents_and_exact_boundaries(self):
        for identifier in self.cases:
            self.prepare(identifier)
        size_dir = self.root / "S02" / "workspace" / "data"
        self.assertEqual((size_dir / "boundary.txt").stat().st_size, 102400)
        self.assertEqual((size_dir / "large.txt").stat().st_size, 102401)
        from docx import Document
        document = Document(self.root / "C01" / "workspace" / "documents" / "演示客户合同.docx")
        self.assertIn("人民币120000元", "\n".join(paragraph.text for paragraph in document.paragraphs))
        import pymupdf
        with pymupdf.open(self.root / "C02" / "workspace" / "documents" / "maintenance.pdf") as document:
            self.assertEqual(len(document), 2)
            self.assertIn("E1042", document[1].get_text())
        from PIL import Image
        with Image.open(self.root / "C03" / "workspace" / "screenshots" / "export-error.png") as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 600)

    def test_new_run_never_overwrites_previous_results(self):
        run = self.prepare("F03")
        before = snapshot(run / "workspace")
        with self.assertRaises(FileExistsError):
            self.prepare("F03")
        self.assertEqual(snapshot(run / "workspace"), before)

    def test_batch_prepares_all_tasks_and_does_not_claim_live_success(self):
        manifest = prepare_batch(list(self.cases.values()), self.root, name="linux-batch")
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["live_tasks"], "not_run")
        batch = Path(manifest["batch_dir"])
        self.assertEqual(set(manifest["case_ids"]), set(self.cases))
        for identifier in self.cases:
            self.assertTrue((batch / identifier / "run.json").is_file())
        with self.assertRaises(FileExistsError):
            prepare_batch(list(self.cases.values()), self.root, name="linux-batch")

    def test_paths_cannot_escape_fixture(self):
        run = self.prepare("F01")
        for name in ("../outside.txt", "/etc/passwd", "C:/outside.txt", "folder/../../bad", "folder\\bad"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fixture_path(run / "workspace", name)

    def test_declined_task_needs_session_evidence_even_when_files_unchanged(self):
        run = self.prepare("I02")
        report = verify(run)
        self.assertEqual(report["status"], "incomplete")
        report = verify(run, self.session(run, executions=0))
        self.assertEqual(report["status"], "passed")

    def test_real_workflow_rejection_export_is_compatible(self):
        run = self.prepare("I02")
        case = self.cases["I02"]
        review = {"verdict": "reasonable", "risk": "high", "summary": "删除测试合同。",
                  "effects": ["删除旧版合同.txt"], "issues": [], "uncertainties": [], "recommendation": "由用户确认。"}
        services = WorkflowServices(
            Mock(return_value={"is_jump": False, "requirement_summary": case["request"]}),
            Mock(return_value=case["request"]),
            Mock(return_value=json.dumps({"Commands": ["rm -- 旧版合同.txt"]})),
            Mock(return_value=review), Mock(), Mock(),
        )
        original = Path.cwd()
        try:
            os.chdir(run / "workspace")
            state = process(case["request"], services=services, input_fn=lambda _: "n",
                            output_fn=lambda _: None, log_file=run / "interaction.sqlite3")
        finally:
            os.chdir(original)
        services.execute_commands.assert_not_called()
        with InteractionJournal(run / "interaction.sqlite3", readonly=True) as journal:
            export = journal.read_session(state["log_session_id"])
        target = run / "real-session.json"
        target.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(verify(run, target)["status"], "passed")

    def test_real_workflow_approval_and_execution_export_is_compatible(self):
        run = self.prepare("F03")
        case = self.cases["F03"]
        workspace = run / "workspace"
        commands = ["mkdir -p -- 备份资料", "cp -- '客户资料/October report.txt' '备份资料/October report.txt'"]
        review = {"verdict": "reasonable", "risk": "low", "summary": "复制测试资料。",
                  "effects": ["创建备份，保留原文件"], "issues": [], "uncertainties": [], "recommendation": "由用户确认。"}

        def execute(_):
            (workspace / "备份资料").mkdir()
            shutil.copy2(workspace / "客户资料" / "October report.txt", workspace / "备份资料" / "October report.txt")
            return {"success": True, "output": "", "current_dir": str(workspace)}

        services = WorkflowServices(
            Mock(return_value={"is_jump": False, "requirement_summary": case["request"]}),
            Mock(return_value=case["request"]), Mock(return_value=json.dumps({"Commands": commands})),
            Mock(return_value=review), Mock(), execute,
        )
        original = Path.cwd()
        try:
            os.chdir(workspace)
            state = process(case["request"], services=services, input_fn=lambda _: "y",
                            output_fn=lambda _: None, log_file=run / "interaction.sqlite3")
        finally:
            os.chdir(original)
        with InteractionJournal(run / "interaction.sqlite3", readonly=True) as journal:
            export = journal.read_session(state["log_session_id"])
        target = run / "real-session.json"
        target.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(verify(run, target)["status"], "passed")

    def test_wrong_task_session_cannot_pass(self):
        run = self.prepare("I02")
        report = verify(run, self.session(run, request="另一个任务", executions=0))
        self.assertEqual(report["status"], "failed")

    def test_cancelled_status_does_not_hide_actual_execution(self):
        run = self.prepare("I02")
        self.assertEqual(verify(run, self.session(run, executions=1))["status"], "failed")

    def test_success_requires_explicit_approval_of_actual_commands(self):
        run = self.prepare("F02")
        (run / "workspace" / "会议纪要.txt").write_bytes("项目：Samantha\n下一步：完成生产测试\n".encode("utf-8"))
        for change in ("missing_answer", "changed_commands", "invalid_action_id"):
            with self.subTest(change=change):
                export = self.session(run)
                data = json.loads(export.read_text(encoding="utf-8"))
                if change == "missing_answer":
                    data["events"] = [event for event in data["events"] if not (event["kind"] == "user_input" and event["data"].get("source") == "confirmation")]
                elif change == "changed_commands":
                    for event in data["events"]:
                        if event["kind"] == "operation_started":
                            event["data"]["arguments"]["args"] = [["unexpected-operation"]]
                else:
                    for event in data["events"]:
                        if event["kind"] == "awaiting_input":
                            event["data"]["action_id"] = "wrong"
                export.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                self.assertEqual(verify(run, export)["status"], "failed")

    def test_file_writes_and_timestamp_changes_outside_scope_fail(self):
        for change in ("content", "timestamp", "extra", "guard"):
            with self.subTest(change=change):
                run = prepare(self.cases["I02"], self.root, name="I02-" + change)
                if change == "guard":
                    (run / "guard" / "outside-workspace.txt").write_text("changed", encoding="utf-8")
                elif change == "extra":
                    (run / "workspace" / "extra.txt").write_text("changed", encoding="utf-8")
                else:
                    path = run / "workspace" / "保留文件.txt"
                    if change == "content":
                        path.write_text("changed", encoding="utf-8")
                    else:
                        stat = path.stat()
                        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                self.assertEqual(verify(run)["status"], "failed")

    def test_expected_file_contents_must_match(self):
        run = self.prepare("F02")
        target = run / "workspace" / "会议纪要.txt"
        target.write_bytes("项目：Samantha\n下一步：完成生产测试".encode("utf-8"))
        export = self.session(run)
        self.assertEqual(verify(run, export)["status"], "passed")
        target.write_bytes("项目：Samantha\n错误内容\n".encode("utf-8"))
        self.assertEqual(verify(run, export)["status"], "failed")

    def test_manual_checks_need_conclusion_and_actual_evidence(self):
        run = self.prepare("I04")
        export = self.session(run, executions=0)
        self.assertEqual(verify(run, export)["status"], "incomplete")
        review = run / "review.json"
        review.write_text(json.dumps({"checks": {"overwrite_notice": {"passed": True, "evidence": ""}}}), encoding="utf-8")
        self.assertEqual(verify(run, export, review)["status"], "incomplete")
        review.write_text(json.dumps({"checks": {"overwrite_notice": {"passed": True, "evidence": "审阅说明明确指出覆盖 config.txt；会话拒绝，没有执行记录。"}}}), encoding="utf-8")
        self.assertEqual(verify(run, export, review)["status"], "passed")

    def test_content_source_must_belong_to_this_fixture_and_page(self):
        run = self.prepare("C02")
        export = self.session(run, sources=[{"path": str(self.root / "maintenance.pdf"), "page": 2}])
        self.assertEqual(verify(run, export)["status"], "failed")
        export = self.session(run, sources=[{"path": str(run / "workspace" / "documents" / "maintenance.pdf"), "page": 1}])
        self.assertEqual(verify(run, export)["status"], "failed")
        export = self.session(run, sources=[{"path": str(run / "workspace" / "documents" / "maintenance.pdf"), "page": 2}])
        self.assertEqual(verify(run, export)["status"], "incomplete")  # Semantic review still pending.

    def test_command_preview_is_not_execution_output(self):
        run = self.prepare("S01")
        export = self.session(run, output="")
        data = json.loads(export.read_text(encoding="utf-8"))
        data["events"].append({"kind": "assistant_output", "data": {"text": "README.md docs/部署说明.md"}})
        export.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(verify(run, export)["status"], "failed")

    def test_archive_checks_bytes_and_rejects_extra_files_and_traversal(self):
        run = self.prepare("M02")
        target = run / "workspace" / "release.tar.gz"
        export = self.session(run)
        with tarfile.open(target, "w:gz") as archive:
            archive.add(run / "workspace" / "deliverables", arcname="deliverables")
        self.assertEqual(verify(run, export)["status"], "passed")
        for name in ("extra.txt", "../escaped.txt"):
            with self.subTest(name=name):
                with tarfile.open(target, "w:gz") as archive:
                    archive.add(run / "workspace" / "deliverables", arcname="deliverables")
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))
                self.assertEqual(verify(run, export)["status"], "failed")
        self.assertFalse((run / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
