"""Inspect saved file objects without models, filesystem checks, or store writes."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys

from object_tracking.store import DEFAULT_OBJECT_PATH


def parser():
    result = argparse.ArgumentParser(
        description="List all saved file entities; stored file status is not rechecked."
    )
    result.add_argument("--db-path", type=Path, help="Override SAMANTHA_OBJECT_PATH")
    result.add_argument("--user", help="Only this user ID; default includes all users")
    result.add_argument("--project", help="Only this project ID; default includes all projects")
    result.add_argument("--status", choices=("active", "stale", "tombstone"),
                        help="Only this saved status; default includes all statuses")
    result.add_argument("--json", action="store_true", help="Print the complete records as JSON")
    return result


def _read_entities(path, filters):
    # Do not construct ObjectStore: opening a writable store initializes its
    # schema, and its candidate APIs inspect files and can invalidate records.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Expected an object database with schema version 1.")
        where = []
        parameters = []
        for column, value in filters.items():
            if value is not None:
                where.append(f"e.{column} = ?")
                parameters.append(value)
        query = """
            SELECT e.user_id, e.project_id, e.entity_id, e.kind, e.current_path,
                   e.status, e.version, e.fingerprint, e.created_at, e.updated_at,
                   e.source_session_id, e.source_operation_id,
                   e.starting_directory, e.stale_reason,
                   CASE WHEN f.entity_id IS NULL THEN 0 ELSE 1 END AS is_focus
            FROM object_entities AS e
            LEFT JOIN object_focus AS f
                ON f.user_id=e.user_id AND f.project_id=e.project_id
                AND f.entity_id=e.entity_id
        """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY e.updated_at DESC, e.user_id, e.project_id, e.entity_id"
        entities = []
        for row in db.execute(query, parameters):
            entity = dict(row)
            entity["fingerprint"] = json.loads(entity["fingerprint"])
            entity["is_focus"] = bool(entity["is_focus"])
            entity["source_type"] = "verified_artifact"
            entities.append(entity)
        return entities


def _display(data):
    print(f"Database: {data['database']}")
    if not data["database_exists"]:
        print("No object database found. No database was created.")
        return
    filters = {key: value for key, value in data["filters"].items() if value is not None}
    if filters:
        print("Filters: " + json.dumps(filters, ensure_ascii=False))
    print(f"Saved entities: {data['count']} (stored status; files not rechecked)")
    if not data["entities"]:
        return
    print("USER\tPROJECT\tENTITY_ID\tSTATUS\tVERSION\tFOCUS\tPATH")
    for entity in data["entities"]:
        print("\t".join((
            json.dumps(entity["user_id"], ensure_ascii=False),
            json.dumps(entity["project_id"], ensure_ascii=False),
            entity["entity_id"], entity["status"], str(entity["version"]),
            "yes" if entity["is_focus"] else "-",
            json.dumps(entity["current_path"], ensure_ascii=False),
        )))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        path = Path(args.db_path or os.environ.get("SAMANTHA_OBJECT_PATH") or
                    DEFAULT_OBJECT_PATH).expanduser().resolve()
        filters = {"user_id": args.user, "project_id": args.project, "status": args.status}
        exists = path.exists()
        entities = _read_entities(path, filters) if exists else []
        data = {"database": str(path), "database_exists": exists, "filters": filters,
                "count": len(entities), "entities": entities}
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            _display(data)
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Unable to read saved entities: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
