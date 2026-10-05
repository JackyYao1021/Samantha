"""Prepare real-task fixtures and verify outcomes; never run generated commands."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shlex
import sys
import tarfile
import uuid


CATALOG = Path(__file__).with_name("cases.json")
DEFAULT_RUNS = Path(__file__).resolve().parents[2] / ".test-results" / "real-tasks"
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Expected a relative POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Path escapes the fixture: {value!r}")
    return path


def fixture_path(root, value):
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative_path(value).parts)
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"Symlinks are not accepted as fixture results: {value}")
        cursor = cursor.parent
    path.resolve().relative_to(root)
    return path


def load_cases(path=CATALOG):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported task catalog schema")
    cases = catalog["cases"]
    identifiers = set()
    for case in cases:
        identifier = case["id"]
        if identifier in identifiers or not identifier.isalnum():
            raise ValueError(f"Duplicate or invalid case ID: {identifier}")
        identifiers.add(identifier)
        if case["mode"] not in {"shell", "content", "fault"}:
            raise ValueError(f"Invalid case mode: {identifier}")
        if case["level"] not in {"baseline", "production_goal"}:
            raise ValueError(f"Invalid case level: {identifier}")
        if not case["request"].strip():
            raise ValueError(f"Empty request: {identifier}")
        for item in case["fixtures"]:
            if relative_path(item["path"]) == PurePosixPath("."):
                raise ValueError("A fixture cannot replace the workspace root")
            if item.get("type", "text") not in {"text", "directory", "docx", "pdf", "png"}:
                raise ValueError(f"Unsupported fixture type: {item}")
        for value in case["allowed_changes"]:
            relative_path(value)
        manual_ids = [item["id"] for item in case["manual_checks"]]
        if len(manual_ids) != len(set(manual_ids)):
            raise ValueError(f"Duplicate review ID: {identifier}")
        for check in case["checks"]:
            if check["kind"] not in {"directory", "text", "absent", "archive", "source", "output_contains", "output_excludes"}:
                raise ValueError(f"Unsupported result check: {check}")
            if "path" in check:
                relative_path(check["path"])
            for name in check.get("files", {}):
                relative_path(name)
        if case["mode"] == "content" and not case.get("commands"):
            raise ValueError(f"Content task has no commands: {identifier}")
        if case["session"] is None and not case["manual_checks"]:
            raise ValueError(f"Task has no execution evidence requirement: {identifier}")
    return cases


def snapshot(root):
    """Hash files without following links; detect writes, removals and extra paths."""
    root = Path(root)
    entries = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            key = path.relative_to(root).as_posix()
            stat = path.lstat()
            if path.is_symlink():
                entries[key] = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_dir():
                entries[key] = {"kind": "directory", "mode": stat.st_mode}
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(65536), b""):
                        digest.update(block)
                entries[key] = {"kind": "file", "sha256": digest.hexdigest(),
                                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "mode": stat.st_mode}
            else:
                entries[key] = {"kind": "special", "mode": stat.st_mode}
    return entries


def write_fixture(root, item):
    path = fixture_path(root, item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = item.get("type", "text")
    if kind == "directory":
        path.mkdir()
    elif kind == "text":
        repeat = item.get("repeat", 1)
        if type(repeat) is not int or repeat < 1:
            raise ValueError("Fixture repeat must be a positive integer")
        encoded = item["text"].encode("utf-8")
        if len(encoded) * repeat > MAX_FIXTURE_BYTES:
            raise ValueError("Fixture exceeds the size limit")
        path.write_bytes(encoded * repeat)
    elif kind == "docx":
        from docx import Document
        document = Document()
        for paragraph in item["paragraphs"]:
            document.add_paragraph(paragraph)
        document.save(path)
    elif kind == "pdf":
        import pymupdf
        with pymupdf.open() as document:
            for text in item["pages"]:
                page = document.new_page()
                if page.insert_textbox((40, 40, 550, 780), text, fontsize=13) < 0:
                    raise ValueError("PDF fixture text does not fit on its page")
            document.save(path)
    elif kind == "png":
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB", (1000, 220), "white")
        ImageDraw.Draw(image).multiline_text((28, 28), "\n".join(item["lines"]),
                                             fill="black", font=ImageFont.load_default(size=30), spacing=16)
        image.save(path)
    if "mtime" in item:
        timestamp = datetime.fromisoformat(item["mtime"]).timestamp()
        os.utime(path, (timestamp, timestamp))


def prepare(case, root=DEFAULT_RUNS, name=None):
    root = Path(root).expanduser().resolve()
    name = name or case["id"]
    if not name or relative_path(name).name != name or name in {".", ".."}:
        raise ValueError("Run name must be one directory name")
    run = root / name
    run.mkdir(parents=True, exist_ok=False)  # Never reset or overwrite an existing run.
    workspace = run / "workspace"
    guard = run / "guard"
    workspace.mkdir()
    guard.mkdir()
    (workspace / "保留文件.txt").write_text("This unrelated file must remain unchanged.\n", encoding="utf-8")
    (guard / "outside-workspace.txt").write_text("Sibling guard: never modify.\n", encoding="utf-8")
    for item in case["fixtures"]:
        write_fixture(workspace, item)
    data = {"schema_version": 1, "case_id": case["id"],
            "workspace_before": snapshot(workspace), "guard_before": snapshot(guard)}
    (run / "run.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    instructions = [f"{case['id']} {case['title']}", f"工作目录：{workspace}",
                    f"用户请求：{case['request']}", ""]
    instructions.extend(case.get("setup_instructions", []))
    instructions.extend(case.get("responses", []))
    if case["mode"] == "content":
        instructions.extend("samantha content " + shlex.join(command) for command in case["commands"])
    instructions.extend(["", "人工验收：", *[f"{item['id']}: {item['description']}" for item in case["manual_checks"]]])
    if case.get("known_gap"):
        instructions.append("已知缺口：" + case["known_gap"])
    (run / "TASK.txt").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    return run


def prepare_batch(cases, root=DEFAULT_RUNS, name=None):
    name = name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not name or relative_path(name).name != name or name in {".", ".."}:
        raise ValueError("Batch name must be one directory name")
    batch = Path(root).expanduser().resolve() / name
    batch.mkdir(parents=True, exist_ok=False)
    for case in cases:
        prepare(case, batch)
    manifest = {"schema_version": 1, "status": "prepared", "live_tasks": "not_run",
                "prepared_count": len(cases), "case_ids": [case["id"] for case in cases],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "python": platform.python_version(),
                "batch_dir": str(batch)}
    (batch / "batch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_FIXTURE_BYTES:
        raise ValueError("Evidence file exceeds 10 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def check_archive(path, expected):
    if path.stat().st_size > MAX_FIXTURE_BYTES:
        return False
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        total = 0
        for count, member in enumerate(archive, 1):
            total += member.size
            if count > 100 or total > MAX_FIXTURE_BYTES:
                return False
            name = relative_path(member.name).as_posix()
            if member.isdir():
                if not any(key.startswith(name + "/") for key in expected):
                    return False
                continue
            if not member.isfile() or name not in expected or name in files:
                return False
            wanted = expected[name].encode("utf-8")
            if member.size != len(wanted):
                return False
            with archive.extractfile(member) as source:
                files[name] = source.read(len(wanted) + 1)
    return files == {name: text.encode("utf-8") for name, text in expected.items()}


def approval_violations(events):
    """Require a fresh explicit approval bound to the actual command list."""
    pending, approved = None, False
    violations = []
    answers = {"y", "yes", "sure", "go ahead", "是", "确认", "执行", "同意"}
    for event in events:
        data = event["data"]
        if event["kind"] == "awaiting_input" and data.get("kind") == "confirmation":
            pending, approved = data, False
        elif event["kind"] == "user_input" and data.get("source") == "confirmation":
            approved = bool(pending) and str(data.get("text", "")).strip().lower() in answers
        elif event["kind"] == "operation_started" and data.get("name") == "commands.execute":
            if not pending or not approved:
                violations.append("执行前缺少对应的明确批准")
            else:
                payload = {key: pending.get(key) for key in ("commands", "cwd", "shell")}
                action_id = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                arguments = data.get("arguments", {}).get("args", [])
                if (pending.get("action_id") != action_id or not arguments
                        or arguments[0] != pending.get("commands")):
                    violations.append("执行命令与确认内容或动作标识不一致")
            approved = False
    return violations


def verify(run, session_export=None, review_file=None):
    run = Path(run).resolve(strict=True)
    metadata = read_json(run / "run.json")
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported prepared run schema")
    case = next((item for item in load_cases() if item["id"] == metadata["case_id"]), None)
    if case is None:
        raise ValueError("Unknown prepared case")
    workspace = fixture_path(run, "workspace")
    guard = fixture_path(run, "guard")
    checks = []

    def record(label, passed, detail=""):
        checks.append({"check": label, "status": "pending" if passed is None else "passed" if passed else "failed", "detail": detail})

    before = metadata["workspace_before"]
    after = snapshot(workspace)
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    unexpected = [key for key in changed if key not in case["allowed_changes"]]
    record("操作范围", not unexpected, ", ".join(unexpected))
    record("工作目录外的保护文件", guard.is_dir() and snapshot(guard) == metadata["guard_before"])

    output, sources = "", []
    if case["session"] is not None:
        if session_export is None:
            record("真实会话证据", None, "请传入 samantha history export 导出的最终调用 JSON")
        else:
            session = read_json(session_export)
            header, events = session["session"], session["events"]
            record("日志版本", session.get("schema_version") == 1)
            expected_request = (shlex.join(["content", *case["commands"][-1]])
                                if case["mode"] == "content" else case["request"])
            identity = (header["request"] == expected_request
                        and Path(header["initial_dir"]).resolve() == workspace
                        and header["kind"] == ("content" if case["mode"] == "content" else "workflow"))
            record("会话属于本任务", identity)
            wanted = case["session"]
            record("最终状态", header["status"] == wanted["status"], header["status"])
            finished = [event["data"] for event in events if event["kind"] == "session_finished"]
            record("会话已结束", bool(finished))
            if case["mode"] != "content":
                violations = approval_violations(events)
                record("每次执行均获得对应批准", not violations, "; ".join(violations))
            state = finished[-1].get("state", {}) if finished else {}
            if identity:
                if case["mode"] == "content":
                    results = [event["data"].get("result", {}) for event in events
                               if event["kind"] == "operation_finished" and event["data"].get("name") == "content.ask"]
                    result = results[-1] if results else {}
                    output, sources = result.get("answer", ""), result.get("sources", [])
                else:
                    output = (state.get("result") or {}).get("output", "")
            counts = {
                "execution": sum(event["kind"] == "operation_started" and event["data"].get("name") == "commands.execute" for event in events),
                "confirmation": sum(event["kind"] == "awaiting_input" and event["data"].get("kind") == "confirmation" for event in events),
                "clarification": sum(event["kind"] == "awaiting_input" and event["data"].get("kind") == "clarification" for event in events),
            }
            for key, count in counts.items():
                if key + "_min" in wanted:
                    record(key + " 次数下限", count >= wanted[key + "_min"], str(count))
                if key + "_max" in wanted:
                    record(key + " 次数上限", count <= wanted[key + "_max"], str(count))
            if "current_dir" in wanted:
                actual = state.get("current_dir")
                record("最终目录", bool(actual) and Path(actual).resolve() == fixture_path(workspace, wanted["current_dir"]))

    for check in case["checks"]:
        kind = check["kind"]
        label = kind + ": " + check.get("path", ", ".join(check.get("values", [])))
        try:
            if kind in {"output_contains", "output_excludes", "source"} and session_export is None:
                record(label, None, "缺少真实输出或检索来源证据")
                continue
            path = fixture_path(workspace, check["path"]) if "path" in check else None
            if kind == "directory":
                passed = path.is_dir()
            elif kind == "absent":
                passed = not path.exists()
            elif kind == "text":
                actual = path.read_bytes().decode("utf-8")
                expected = check["text"]
                passed = actual in {expected, expected.removesuffix("\n")} if check.get("final_newline_optional") else actual == expected
            elif kind == "archive":
                passed = check_archive(path, check["files"])
            elif kind == "output_contains":
                passed = all(value in output for value in check["values"])
            elif kind == "output_excludes":
                passed = all(value not in output for value in check["values"])
            elif kind == "source":
                passed = any(Path(source["path"]).resolve() == path
                             and ("page" not in check or source.get("page") == check["page"]) for source in sources)
            record(label, passed)
        except (OSError, UnicodeError, ValueError, tarfile.TarError) as exc:
            record(label, False, str(exc))

    reviews = read_json(review_file).get("checks", {}) if review_file else {}
    expected_ids = {item["id"] for item in case["manual_checks"]}
    if set(reviews) - expected_ids:
        raise ValueError("Review file contains IDs not defined by this task")
    for item in case["manual_checks"]:
        review = reviews.get(item["id"])
        if review is None:
            record("人工: " + item["id"], None, item["description"])
        elif type(review.get("passed")) is not bool or not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
            record("人工: " + item["id"], None, "需要布尔结论和非空 evidence，填写实际观察及日志位置")
        else:
            record("人工: " + item["id"], review["passed"], review["evidence"])
    status = "failed" if any(item["status"] == "failed" for item in checks) else "incomplete" if any(item["status"] == "pending" for item in checks) else "passed"
    return {"schema_version": 1, "case_id": case["id"], "level": case["level"],
            "status": status, "changed_paths": changed, "checks": checks,
            "scope": "文件和会话结果验证；人工语义判断需审阅记录，本工具不提供操作系统沙箱"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List real tasks without invoking any model")
    listing.add_argument("--category", choices=("files", "search", "interaction", "content", "reliability"))
    listing.add_argument("--level", choices=("baseline", "production_goal"))
    show = commands.add_parser("show", help="Show a task definition or its request")
    show.add_argument("case_id")
    show.add_argument("--field", choices=("request", "title", "mode"))
    setup = commands.add_parser("prepare", help="Prepare a new fixture directory")
    setup.add_argument("case_id")
    setup.add_argument("--root", type=Path, default=DEFAULT_RUNS)
    setup.add_argument("--name", help="Fresh run directory name, e.g. F03-qwen-run2")
    batch = commands.add_parser("prepare-all", help="Prepare every task in a fresh batch")
    batch.add_argument("--root", type=Path, default=DEFAULT_RUNS)
    batch.add_argument("--name", help="Fresh batch directory name; otherwise UTC time plus random ID")
    check = commands.add_parser("check", help="Read-only outcome checks")
    check.add_argument("run", type=Path)
    check.add_argument("--session-export", type=Path)
    check.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
        if args.command == "list":
            for case in cases:
                if args.category and case["category"] != args.category:
                    continue
                if args.level and case["level"] != args.level:
                    continue
                print(f"{case['id']} [{case['category']}/{case['level']}] {case['title']}\n  {case['request']}")
            return 0
        if args.command == "prepare-all":
            print(json.dumps(prepare_batch(cases, args.root, args.name), ensure_ascii=False, indent=2))
            return 0
        if args.command in {"prepare", "show"}:
            case = next((item for item in cases if item["id"] == args.case_id), None)
            if case is None:
                raise ValueError(f"Unknown task: {args.case_id}")
            if args.command == "show":
                print(case[args.field] if args.field else json.dumps(case, ensure_ascii=False, indent=2))
            else:
                print(prepare(case, args.root, args.name))
            return 0
        report = verify(args.run, args.session_export, args.review)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Task evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Piped Windows stdout otherwise uses the system code page, corrupting
    # Chinese task requests and JSON when consumed by UTF-8 tools.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
