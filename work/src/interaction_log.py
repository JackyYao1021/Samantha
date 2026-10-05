"""Durable, ordered interaction journal shared by the terminal and tools.

Every write is committed before returning. User text, tool output, and model
reasoning returned by the provider are retained separately.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / ".samantha" / "interactions.sqlite3"
_active_session = ContextVar("samantha_log_session", default=None)
_active_operation = ContextVar("samantha_log_operation", default=None)


class InteractionLogError(RuntimeError):
    """Logging failed; stop rather than perform unrecorded operations."""


def log_path(path=None):
    return Path(path or os.environ.get("SAMANTHA_LOG_PATH") or DEFAULT_LOG_PATH).expanduser().resolve()


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(data):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            # Settings can carry credentials. Only retain non-secret configuration.
            return {key: item for key, item in asdict(value).items()
                    if not any(word in key.lower() for word in ("key", "token", "password", "secret"))}
        raise TypeError(f"Cannot log {type(value).__name__}")
    return json.dumps(data, ensure_ascii=False, default=convert)


class InteractionJournal:
    def __init__(self, path=None, *, readonly=False):
        self.path = log_path(path)
        self.connection = None
        try:
            if readonly:
                self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(str(self.path), timeout=30)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (readonly and version != 1):
                raise ValueError(f"Unsupported interaction log schema: {version}")
            if not readonly:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        kind TEXT NOT NULL,
                        request TEXT NOT NULL,
                        initial_dir TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed','cancelled')),
                        error TEXT NOT NULL DEFAULT ''
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        session_id TEXT NOT NULL REFERENCES sessions(session_id),
                        sequence INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        turn INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        role TEXT NOT NULL,
                        data TEXT NOT NULL,
                        PRIMARY KEY (session_id, sequence)
                    );
                    CREATE INDEX IF NOT EXISTS sessions_started ON sessions(started_at DESC);
                    PRAGMA user_version=1;
                """)
        except (OSError, sqlite3.Error, ValueError) as exc:
            self.close()
            raise InteractionLogError(f"Cannot open interaction log {self.path}: {exc}") from exc

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def _transaction(self):
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield
            self.connection.commit()
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self.connection.rollback()
            raise InteractionLogError(f"Cannot write interaction log {self.path}: {exc}") from exc
        except BaseException:
            self.connection.rollback()
            raise

    def _append(self, session_id, turn, kind, role, data):
        sequence = self.connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, sequence, _timestamp(), turn, kind, role, _json(data)),
        )

    def start_session(self, request, *, kind="workflow", metadata=None):
        session_id = uuid.uuid4().hex
        cwd = os.getcwd()
        with self._transaction():
            self.connection.execute(
                "INSERT INTO sessions(session_id, started_at, kind, request, initial_dir, status) "
                "VALUES (?, ?, ?, ?, ?, 'running')", (session_id, _timestamp(), kind, request, cwd),
            )
            self._append(session_id, 1, "session_started", "system", {"cwd": cwd, **(metadata or {})})
            self._append(session_id, 1, "user_input", "user", {"text": request, "source": "request"})
        return LogSession(self, session_id)

    def list_sessions(self, limit=20):
        if not 1 <= limit <= 1000:
            raise ValueError("History limit must be between 1 and 1000.")
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC, session_id DESC LIMIT ?", (limit,)
        )]

    def read_session(self, session_id):
        # One read transaction makes the header and event list a consistent snapshot.
        with self.connection:
            self.connection.execute("BEGIN")
            if session_id == "latest":
                row = self.connection.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC, session_id DESC LIMIT 1"
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT * FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            if row is None:
                raise ValueError(f"Interaction session not found: {session_id}")
            events = [dict(event) for event in self.connection.execute(
                "SELECT * FROM events WHERE session_id=? ORDER BY sequence", (row["session_id"],)
            )]
            for event in events:
                event["data"] = json.loads(event["data"])
            return {"schema_version": 1, "session": dict(row), "events": events}


class LogSession:
    def __init__(self, journal, session_id):
        self.journal = journal
        self.session_id = session_id
        self.turn = 1
        self.finished = False

    def __enter__(self):
        self._token = _active_session.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if not self.finished:
                cancelled = isinstance(exc, (EOFError, KeyboardInterrupt))
                success_exit = isinstance(exc, SystemExit) and exc.code in (None, 0)
                self.finish("cancelled" if cancelled else "succeeded" if success_exit else "failed",
                            error=str(exc) if exc is not None else "Session ended without a result.",
                            details={"exception_type": exc_type.__name__ if exc_type else None})
        finally:
            _active_session.reset(self._token)

    def record(self, kind, data, *, role="samantha"):
        with self.journal._transaction():
            self.journal._append(self.session_id, self.turn, kind, role, data)

    def user_input(self, text, source):
        self.turn += 1
        self.record("user_input", {"text": text, "source": source}, role="user")

    def output(self, text, *, stream="stdout", prompt=False):
        self.record("prompt" if prompt else "assistant_output",
                    {"text": text, "stream": stream}, role="assistant")

    def finish(self, status, *, error="", details=None):
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("A terminal session status is required.")
        with self.journal._transaction():
            self.journal._append(self.session_id, self.turn, "session_finished", "system",
                                 {"status": status, "error": error, **(details or {})})
            self.journal.connection.execute(
                "UPDATE sessions SET status=?, error=?, ended_at=? WHERE session_id=?",
                (status, error, _timestamp(), self.session_id),
            )
        self.finished = True


@contextmanager
def operation(name, arguments):
    """Record an actual operation before it starts and its outcome afterwards."""
    session = _active_session.get()
    outcome = {}
    if session is None:
        yield outcome
        return
    operation_id = uuid.uuid4().hex
    session.record("operation_started", {"operation_id": operation_id, "name": name,
                                         "arguments": arguments})
    started = time.monotonic()
    token = _active_operation.set({"operation_id": operation_id, "operation_name": name})
    try:
        yield outcome
    except BaseException as exc:
        session.record("operation_finished", {"operation_id": operation_id, "name": name,
                       "status": "cancelled" if isinstance(exc, (EOFError, KeyboardInterrupt)) else "failed",
                       "error": str(exc), "exception_type": type(exc).__name__,
                       "duration_ms": round((time.monotonic() - started) * 1000, 3), **outcome})
        raise
    else:
        session.record("operation_finished", {"operation_id": operation_id, "name": name,
                       "status": "completed", "duration_ms": round((time.monotonic() - started) * 1000, 3),
                       **outcome})
    finally:
        _active_operation.reset(token)


def record_model_reasoning(data):
    """Attach provider-returned reasoning to the active agent/tool operation."""
    session = _active_session.get()
    enabled = os.environ.get("SAMANTHA_LOG_REASONING", "true").lower() in {"1", "true", "yes"}
    if session is not None and enabled:
        session.record("model_reasoning", {**(_active_operation.get() or {}), **data}, role="model")


def record_operation(name, function, *args, **kwargs):
    with operation(name, {"args": args, "kwargs": kwargs}) as outcome:
        result = function(*args, **kwargs)
        outcome["result"] = result
        if isinstance(result, dict) and type(result.get("success")) is bool:
            outcome["status"] = "succeeded" if result["success"] else "failed"
        elif isinstance(result, dict) and result.get("status") == "failed":
            outcome["status"] = "failed"
        return result


class LoggedStream:
    """Tee content CLI output without altering whitespace or JSON output."""
    def __init__(self, stream, session, name):
        self.stream, self.session, self.name = stream, session, name

    def write(self, text):
        if text:
            self.session.output(text, stream=self.name)
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)
