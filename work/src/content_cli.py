"""CLI: python work/src/content_cli.py index|search|ask ..."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import shlex
import sys

from dotenv import load_dotenv

from content_search.config import Settings, WORK_DIR
from content_search.diagnostics import diagnose
from content_search.pipeline import ContentPipeline
from interaction_log import InteractionJournal, InteractionLogError, LoggedStream, operation, record_operation


def parser():
    result = argparse.ArgumentParser(description="Samantha file and image content search")
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check services and installed models without downloading weights")
    doctor.add_argument("--json", action="store_true")
    index = commands.add_parser("index", help="Index a file or recursively index a directory")
    index.add_argument("path", type=Path)
    index.add_argument("--force", action="store_true")
    index.add_argument("--json", action="store_true")
    for name in ("search", "ask"):
        command = commands.add_parser(name)
        command.add_argument("query", nargs="+")
        command.add_argument("--limit", type=int, default=5 if name == "ask" else 10)
        command.add_argument("--tag", action="append", default=[], help="Require this tag; repeat for AND")
        command.add_argument("--ext", help="File extension, e.g. pdf or .png")
        command.add_argument("--json", action="store_true")
    return result


def main(argv=None):
    load_dotenv(WORK_DIR.parent / ".env", override=False)
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        with InteractionJournal() as journal, journal.start_session(
            shlex.join(["content", *argv]), kind="content", metadata={"argv": argv}
        ) as session:
            with redirect_stdout(LoggedStream(sys.stdout, session, "stdout")), \
                    redirect_stderr(LoggedStream(sys.stderr, session, "stderr")):
                status = _main(argv)
            session.finish("succeeded" if status == 0 else "failed",
                           error="" if status == 0 else f"Content command exited with status {status}.",
                           details={"exit_code": status})
            return status
    except InteractionLogError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _main(argv):
    args = parser().parse_args(argv)
    pipeline = None
    try:
        settings = Settings.from_env()
        if args.command == "doctor":
            checks = record_operation("content.doctor", diagnose, settings)
            if args.json:
                print(json.dumps(checks, ensure_ascii=False, indent=2))
            else:
                for check in checks:
                    print(f"{'OK' if check['ready'] else 'MISSING'} {check['component']}: {check['detail']}")
            return 0 if all(check["ready"] for check in checks) else 1
        with operation("content.initialize", {"settings": settings}):
            pipeline = ContentPipeline(settings)
        if args.command == "index":
            # The callback itself is not serialized; it tees its output to the log.
            data = record_operation("content.index", lambda path, force: pipeline.index(
                path, force=force, progress=lambda message: print(message, file=sys.stderr)),
                args.path, args.force)
            if not data:
                raise ValueError("No supported files found.")
            status = 1 if any(item["status"] == "failed" for item in data) else 0
        else:
            query = " ".join(args.query)
            method = pipeline.search if args.command == "search" else pipeline.ask
            data = record_operation("content." + args.command, method, query,
                                    limit=args.limit, tags=args.tag, extension=args.ext)
            status = 0
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.command == "index":
            for item in data:
                print(f"{item['status']}: {item['path']}")
                if item.get("error"):
                    print("  " + item["error"])
                elif item.get("summary"):
                    print("  " + item["summary"])
                    print("  tags: " + ", ".join(item["tags"]))
        elif args.command == "search":
            if not data:
                print("No matching files.")
            for index, file in enumerate(data, 1):
                print(f"[{index}] {file['path']} (score={file['score']:.4f})")
                print("  " + file["summary"])
                print("  tags: " + ", ".join(file["tags"]))
                for hit in file["matches"]:
                    location = f"page {hit['page']}" if hit["page"] is not None else f"chunk {hit['chunk']}"
                    print(f"  {location}, {hit['kind']}: {' '.join(hit['text'].split())[:300]}")
        else:
            print(data["answer"])
            for index, source in enumerate(data["sources"], 1):
                print(f"[{index}] {source['path']} page={source['page']} kind={source['kind']}")
        return status
    except InteractionLogError:
        raise
    except Exception as exc:
        print(f"Content search failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            with operation("content.close", {}):
                pipeline.close()


if __name__ == "__main__":
    sys.exit(main())
