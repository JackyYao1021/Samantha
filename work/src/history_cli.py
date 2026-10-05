"""Read and export interaction logs without changing the latest session."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from interaction_log import InteractionJournal, InteractionLogError, log_path


def parser():
    result = argparse.ArgumentParser(description="Samantha persistent interaction history")
    result.add_argument("--log-path", type=Path, help="Override SAMANTHA_LOG_PATH")
    commands = result.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List recent invocations")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show", help="Show all turns and operations in order")
    show.add_argument("session_id", help="Full session ID or latest")
    show.add_argument("--json", action="store_true")
    export = commands.add_parser("export", help="Export a complete session as JSON or JSONL")
    export.add_argument("session_id", help="Full session ID or latest")
    export.add_argument("--format", choices=("json", "jsonl"), default="json")
    export.add_argument("--output", type=Path, help="Save to a new file; otherwise write to stdout")
    return result


def _display(data):
    session = data["session"]
    print(f"{session['session_id']}  {session['status']}  {session['started_at']}")
    print(f"Directory: {session['initial_dir']}")
    for event in data["events"]:
        payload = event["data"]
        print(f"\n[{event['sequence']} / turn {event['turn']}] {event['timestamp']} "
              f"{event['role']} {event['kind']}")
        if event["kind"] in {"user_input", "assistant_output", "prompt"}:
            print(payload["text"], end="" if payload["text"].endswith("\n") else "\n")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    path = log_path(args.log_path)
    if not path.exists():
        if args.command == "list":
            print("[]" if args.json else f"No interaction history yet ({path}).")
            return 0
        print(f"No interaction history yet ({path}).", file=sys.stderr)
        return 1
    try:
        with InteractionJournal(path, readonly=True) as journal:
            if args.command == "list":
                data = journal.list_sessions(args.limit)
                if args.json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    for session in data:
                        print(f"{session['session_id']}  {session['status']}  {session['started_at']}  "
                              f"{json.dumps(session['request'], ensure_ascii=False)}")
                return 0
            data = journal.read_session(args.session_id)
        if args.command == "show":
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                _display(data)
            return 0
        if args.format == "jsonl":
            lines = [{"schema_version": data["schema_version"], "session": data["session"]}, *data["events"]]
            text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
        else:
            text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            # Never overwrite a prior export, or the live database and its sidecars.
            target = args.output.expanduser().resolve()
            if str(target) in {str(path), str(path) + "-wal", str(path) + "-shm"}:
                raise ValueError("Export output cannot be the interaction database or its sidecars.")
            with target.open("x", encoding="utf-8", newline="\n") as output:
                output.write(text)
            print(f"Exported {data['session']['session_id']} to {target}")
        else:
            print(text, end="")
        return 0
    except (InteractionLogError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"Unable to read/export interaction history: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
