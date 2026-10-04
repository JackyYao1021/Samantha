"""CLI: python work/src/content_cli.py index|search|ask ..."""

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

from content_search.config import Settings, WORK_DIR
from content_search.diagnostics import diagnose
from content_search.pipeline import ContentPipeline


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
    args = parser().parse_args(argv)
    load_dotenv(WORK_DIR.parent / ".env", override=False)
    pipeline = None
    try:
        settings = Settings.from_env()
        if args.command == "doctor":
            checks = diagnose(settings)
            if args.json:
                print(json.dumps(checks, ensure_ascii=False, indent=2))
            else:
                for check in checks:
                    print(f"{'OK' if check['ready'] else 'MISSING'} {check['component']}: {check['detail']}")
            return 0 if all(check["ready"] for check in checks) else 1
        pipeline = ContentPipeline(settings)
        if args.command == "index":
            data = pipeline.index(args.path, force=args.force,
                                  progress=lambda message: print(message, file=sys.stderr))
            if not data:
                raise ValueError("No supported files found.")
            status = 1 if any(item["status"] == "failed" for item in data) else 0
        else:
            query = " ".join(args.query)
            method = pipeline.search if args.command == "search" else pipeline.ask
            data = method(query, limit=args.limit, tags=args.tag, extension=args.ext)
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
    except Exception as exc:
        print(f"Content search failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    sys.exit(main())
