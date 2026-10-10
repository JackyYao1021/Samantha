"""Scoped SQLite object lineage, updated only from verified execution evidence."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
import uuid

from agent_memory.core import MemoryScope
from .verification import DEFAULT_MAX_BYTES, normalize_path, observe_path, validate_effect, verify_after


DEFAULT_OBJECT_PATH = Path(__file__).resolve().parents[2] / ".samantha" / "objects.sqlite3"


class ObjectStoreError(RuntimeError):
    """Object persistence failed; callers must never replay filesystem commands."""


class ObjectConflict(ObjectStoreError):
    """The proposal used stale identity/version or reused different provenance."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scope(scope):
    if not isinstance(scope, MemoryScope):
        raise ValueError("Object scope must be a MemoryScope.")
    return scope.user_id, scope.project_id


class ObjectStore:
    """The caller owns lifetime; object state never includes workflow approval."""

    def __init__(self, path=None, *, max_bytes=DEFAULT_MAX_BYTES):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Verification byte limit must be positive.")
        self.max_bytes = max_bytes
        self.path = str(path if path is not None else
                        os.environ.get("SAMANTHA_OBJECT_PATH") or DEFAULT_OBJECT_PATH)
        self.connection = None
        self._lock = RLock()
        try:
            if self.path != ":memory:":
                self.path = str(Path(self.path).expanduser().absolute())
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"Unsupported object store schema: {version}")
            if version == 1 and not self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='object_entities'").fetchone():
                raise ValueError("Database is not an object store.")
            if version == 1 and "path_key" not in {row[1] for row in self.connection.execute(
                    "PRAGMA table_info(object_entities)").fetchall()}:
                raise ValueError("Incompatible preliminary object store schema (missing path_key).")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS object_entities (
                    user_id TEXT NOT NULL, project_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind='file'), current_path TEXT NOT NULL, path_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','stale','tombstone')),
                    version INTEGER NOT NULL CHECK(version>0), fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    source_session_id TEXT NOT NULL, source_operation_id TEXT NOT NULL,
                    starting_directory TEXT NOT NULL, stale_reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(user_id, project_id, entity_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS object_live_path
                    ON object_entities(user_id, project_id, path_key) WHERE status='active';
                CREATE TABLE IF NOT EXISTS object_events (
                    user_id TEXT NOT NULL, project_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL, session_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    effect_index INTEGER NOT NULL, operation_kind TEXT NOT NULL,
                    old_path TEXT, new_path TEXT NOT NULL, timestamp TEXT NOT NULL,
                    execution_status TEXT NOT NULL, verification_status TEXT NOT NULL,
                    request_hash TEXT NOT NULL, evidence TEXT NOT NULL, entity_version INTEGER NOT NULL,
                    PRIMARY KEY(user_id, project_id, session_id, operation_id, effect_index),
                    FOREIGN KEY(user_id, project_id, entity_id)
                        REFERENCES object_entities(user_id, project_id, entity_id)
                );
                CREATE TABLE IF NOT EXISTS object_focus (
                    user_id TEXT NOT NULL, project_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    PRIMARY KEY(user_id, project_id)
                );
                CREATE TABLE IF NOT EXISTS object_inspections (
                    inspection_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL, timestamp TEXT NOT NULL, previous_version INTEGER NOT NULL,
                    status TEXT NOT NULL, reason TEXT NOT NULL, observation TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)
        except (OSError, sqlite3.Error, ValueError) as exc:
            self.close()
            raise ObjectStoreError(f"Cannot open object store {self.path}: {exc}") from exc

    def close(self):
        with self._lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield
                self.connection.commit()
            except BaseException as exc:
                self.connection.rollback()
                if isinstance(exc, (sqlite3.Error, OSError)):
                    raise ObjectStoreError(f"Cannot write object store {self.path}: {exc}") from exc
                raise

    @staticmethod
    def _entity(row):
        if row is None:
            return None
        result = dict(row)
        result.pop("path_key", None)
        result["fingerprint"] = json.loads(result["fingerprint"])
        result["source_type"] = "verified_artifact"
        return result

    def _lookup(self, scope, entity_id):
        with self._lock:
            return self._entity(self.connection.execute(
                "SELECT * FROM object_entities WHERE user_id=? AND project_id=? AND entity_id=?",
                (*_scope(scope), entity_id)).fetchone())

    def _invalidate(self, entity, observation, reason=None):
        status = "tombstone" if observation["status"] == "missing" else "stale"
        reason = reason or observation.get("reason") or "File fingerprint changed outside a verified operation."
        parameters = (entity["user_id"], entity["project_id"], entity["entity_id"])
        updated = self.connection.execute(
            "UPDATE object_entities SET status=?,version=version+1,updated_at=?,stale_reason=? "
            "WHERE user_id=? AND project_id=? AND entity_id=? AND version=? AND status='active'",
            (status, _now(), reason, *parameters, entity["version"]))
        if updated.rowcount:
            self.connection.execute("INSERT INTO object_inspections VALUES (?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, *parameters, _now(), entity["version"], status, reason, _json(observation)))
        return updated.rowcount

    def _inspect(self, scope, entity):
        if entity is None or entity["status"] != "active":
            return entity
        observation = observe_path(entity["current_path"], max_bytes=self.max_bytes)
        if observation["status"] == "present" and observation["fingerprint"] == entity["fingerprint"]:
            return entity
        with self._transaction():
            self._invalidate(entity, observation)
        # A concurrent verified update may have won the CAS. Do not return the old
        # candidate; inspect the new version once before exposing it to a caller.
        current = self._lookup(scope, entity["entity_id"])
        if current and current["status"] == "active":
            latest = observe_path(current["current_path"], max_bytes=self.max_bytes)
            if latest["status"] != "present" or latest["fingerprint"] != current["fingerprint"]:
                with self._transaction():
                    self._invalidate(current, latest)
                current = self._lookup(scope, entity["entity_id"])
        return current

    def lookup(self, scope, entity_id, *, check_stale=True):
        entity = self._lookup(scope, entity_id)
        return self._inspect(scope, entity) if check_stale else entity

    def lookup_path(self, scope, path, *, check_stale=True):
        """Resolve an explicit current path without a recent-candidate cutoff."""
        path = normalize_path(path)
        with self._lock:
            entity = self._entity(self.connection.execute(
                "SELECT * FROM object_entities WHERE user_id=? AND project_id=? AND path_key=? "
                "AND status='active'", (*_scope(scope), os.path.normcase(path))).fetchone())
        return self._inspect(scope, entity) if check_stale else entity

    def clear_focus(self, scope, *, expected_operation_id=None):
        """An unsupported or cancelled task cannot assert which file was just used."""
        with self._transaction():
            sql = "DELETE FROM object_focus WHERE user_id=? AND project_id=?"
            parameters = _scope(scope)
            if expected_operation_id is not None:
                sql += " AND operation_id=?"
                parameters += (expected_operation_id,)
            return self.connection.execute(sql, parameters).rowcount

    def candidates(self, scope, *, limit=20, check_stale=True):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Object candidate limit must be between 1 and 1000.")
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM object_entities WHERE user_id=? AND project_id=? AND status='active' "
                "ORDER BY updated_at DESC,entity_id LIMIT ?", (*_scope(scope), limit)).fetchall()
        records = []
        for row in rows:
            entity = self._entity(row)
            entity = self._inspect(scope, entity) if check_stale else entity
            if entity and entity["status"] == "active":
                records.append(entity)
        return records

    def latest_context(self, scope, *, max_chars=6000, limit=20):
        if type(max_chars) is not int or max_chars < 0:
            raise ValueError("Object context character budget must be nonnegative.")
        candidates = self.candidates(scope, limit=limit)
        with self._lock:
            focus = self.connection.execute(
                "SELECT entity_id FROM object_focus WHERE user_id=? AND project_id=?", _scope(scope)).fetchone()
            stale = self.connection.execute(
                "SELECT entity_id,current_path,status,version,stale_reason FROM object_entities "
                "WHERE user_id=? AND project_id=? AND status!='active' ORDER BY updated_at DESC LIMIT ?",
                (*_scope(scope), limit)).fetchall()
        context = {"focus": focus[0] if focus and any(item["entity_id"] == focus[0]
                   for item in candidates) else None, "candidates": candidates,
                   "stale": [dict(row) for row in stale], "truncated": False}
        while len(_json(context)) > max_chars:
            context["truncated"] = True
            if context["stale"]:
                context["stale"].pop()
            elif context["candidates"]:
                removed = context["candidates"].pop()
                if removed["entity_id"] == context["focus"]:
                    context["focus"] = None
            else:
                return ""
        return _json(context)

    context = latest_context

    def apply_verified(self, scope, effect, verification, *, session_id, operation_id,
                       starting_directory="", effect_index=0, execution_status="unknown"):
        """Commit once after the caller durably recorded actual execution.

        A failed batch may still contain a verified effect; batch success is never
        used as artifact proof. Re-importing the same provenance is idempotent.
        Conflicts are diagnostic only and never trigger command execution.
        """
        scope_values = _scope(scope)
        if not all(isinstance(value, str) and value.strip() for value in (session_id, operation_id)):
            raise ValueError("Actual source session and operation IDs are required.")
        if type(effect_index) is not int or effect_index < 0:
            raise ValueError("Effect index must be a nonnegative integer.")
        if execution_status not in {"unknown", "succeeded", "failed"}:
            raise ValueError("Unsupported execution status.")
        effect = validate_effect(effect)
        if not isinstance(verification, dict) or verification.get("status") != "verified":
            return None
        if verification.get("effect") != effect:
            raise ObjectConflict("Verification does not match the proposed effect.")
        if starting_directory:
            starting_directory = normalize_path(starting_directory)
        request_hash = hashlib.sha256(_json(effect).encode("utf-8")).hexdigest()
        provenance = (*scope_values, session_id, operation_id, effect_index)
        with self._transaction():
            prior = self.connection.execute(
                "SELECT entity_id,request_hash FROM object_events WHERE user_id=? AND project_id=? "
                "AND session_id=? AND operation_id=? AND effect_index=?", provenance).fetchone()
            if prior:
                if prior["request_hash"] != request_hash:
                    raise ObjectConflict("Source operation was already imported with a different effect.")
                return self._lookup(scope, prior["entity_id"])
            # Recheck read-only evidence at commit so stale observations cannot
            # overwrite a concurrent verified path update or an external change.
            current = verify_after(effect, verification.get("before"), max_bytes=self.max_bytes)
            if current["status"] != "verified" or current["after"] != verification.get("after"):
                raise ObjectConflict("Artifact evidence changed before object state commit.")
            target = (current["after"]["destination"] if effect["kind"] == "rename"
                      else current["after"]["source"])
            new_path, fingerprint = target["path"], target["fingerprint"]
            entity = self._lookup(scope, effect["entity_id"]) if effect["entity_id"] else None
            if effect["entity_id"] and (not entity or entity["status"] != "active" or
                    entity["version"] != effect["entity_version"] or
                    os.path.normcase(entity["current_path"]) != os.path.normcase(effect["path"]) or
                    entity["fingerprint"] != current["before"]["source"]["fingerprint"]):
                raise ObjectConflict("Object identity or version changed after the proposal.")
            if entity is None and effect["kind"] != "create":
                known = self.connection.execute(
                    "SELECT entity_id FROM object_entities WHERE user_id=? AND project_id=? "
                    "AND path_key=? AND status='active'", (*scope_values, os.path.normcase(effect["path"]))).fetchone()
                if known:
                    raise ObjectConflict("A tracked source requires its current identity and version.")
            occupied = self.connection.execute(
                "SELECT * FROM object_entities WHERE user_id=? AND project_id=? AND path_key=? "
                "AND status='active'", (*scope_values, os.path.normcase(new_path))).fetchone()
            if occupied and (entity is None or occupied["entity_id"] != entity["entity_id"]):
                if effect["kind"] != "create":
                    raise ObjectConflict("Destination is already owned by another tracked object.")
                self._invalidate(self._entity(occupied), current["before"]["source"],
                                 "Previous object was absent before verified creation at the same path.")
            timestamp = _now()
            identifier = entity["entity_id"] if entity else uuid.uuid4().hex
            version = entity["version"] + 1 if entity else 1
            if entity:
                update = self.connection.execute(
                    "UPDATE object_entities SET current_path=?,path_key=?,version=?,fingerprint=?,updated_at=?,"
                    "source_session_id=?,source_operation_id=?,starting_directory=?,stale_reason='' "
                    "WHERE user_id=? AND project_id=? AND entity_id=? AND version=? AND status='active'",
                    (new_path, os.path.normcase(new_path), version, _json(fingerprint), timestamp, session_id, operation_id,
                     starting_directory, *scope_values, identifier, entity["version"]))
                if update.rowcount != 1:
                    raise ObjectConflict("Concurrent object update won the version check.")
            else:
                self.connection.execute("INSERT INTO object_entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*scope_values, identifier, "file", new_path, os.path.normcase(new_path), "active", version, _json(fingerprint),
                     timestamp, timestamp, session_id, operation_id, starting_directory, ""))
            evidence = {"before_source": current["before"]["source"],
                        "before_destination": current["before"]["destination"],
                        "expected": current["before"]["expected"], "after": current["after"],
                        "checks": current["checks"]}
            if verification.get("action_id") is not None:
                evidence["action_id"] = verification["action_id"]
            event_id = hashlib.sha256(_json(provenance).encode("utf-8")).hexdigest()
            self.connection.execute("INSERT INTO object_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (*scope_values, event_id, identifier, session_id, operation_id, effect_index,
                 effect["kind"], effect["path"] if effect["kind"] != "create" else None,
                 new_path, timestamp, execution_status, "verified", request_hash, _json(evidence), version))
            self.connection.execute(
                "INSERT INTO object_focus VALUES (?,?,?,?,?) ON CONFLICT(user_id,project_id) "
                "DO UPDATE SET entity_id=excluded.entity_id,session_id=excluded.session_id,"
                "operation_id=excluded.operation_id", (*scope_values, identifier, session_id, operation_id))
            return self._lookup(scope, identifier)

    def events(self, scope, *, entity_id=None):
        """Read auditable lineage without reconstructing executable actions."""
        sql = "SELECT * FROM object_events WHERE user_id=? AND project_id=?"
        parameters = _scope(scope)
        if entity_id is not None:
            sql += " AND entity_id=?"
            parameters += (entity_id,)
        with self._lock:
            rows = self.connection.execute(sql + " ORDER BY timestamp,event_id", parameters).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence"])} for row in rows]
