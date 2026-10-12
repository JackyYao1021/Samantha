"""Atomic task snapshots/events, with an OS lock released even after process death."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading


def now():
    return datetime.now(timezone.utc).isoformat()


_locks = {}
_guard = threading.Lock()


@contextmanager
def process_lock(path):
    """Serialize runners and decisions across threads AND independent processes."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _guard:
        mutex = _locks.setdefault(str(path), threading.Lock())
    if not mutex.acquire(blocking=False):
        raise RuntimeError("Another task operation is active; retry shortly.")
    stream = None
    locked = False
    try:
        stream = path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("Another task process is active; retry shortly.") from exc
        yield
    finally:
        if stream is not None:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
        mutex.release()


class TaskStore:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS agent_tasks (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS agent_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS agent_events_task ON agent_events(task_id,id)")
            db.execute("CREATE TABLE IF NOT EXISTS agent_cancellation (task_id TEXT PRIMARY KEY)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def request_cancel(self, task_id):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO agent_cancellation VALUES (?)", (task_id,))

    def cancelled(self, task_id):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM agent_cancellation WHERE task_id=?", (task_id,)).fetchone() is not None

    def save(self, state, kind, payload=None, *, create=False):
        # State and its event always commit together; tools are outside SQLite's transaction.
        state["updated_at"] = now()
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            if create:
                db.execute("INSERT INTO agent_tasks VALUES (?, ?)", (state["id"], encoded))
            else:
                cursor = db.execute("UPDATE agent_tasks SET state=? WHERE id=?", (encoded, state["id"]))
                if cursor.rowcount != 1:
                    raise KeyError("Unknown task")
            db.execute("INSERT INTO agent_events(task_id,kind,payload,created_at) VALUES (?,?,?,?)",
                       (state["id"], kind, json.dumps(payload or {}, ensure_ascii=False, allow_nan=False), now()))

    def read(self, task_id, *, events=True):
        with self.connect() as db:
            row = db.execute("SELECT state FROM agent_tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown task")
            state = json.loads(row[0])
            if events:
                state["events"] = [{"id": r[0], "kind": r[1], "payload": json.loads(r[2]), "created_at": r[3]}
                                   for r in db.execute("SELECT id,kind,payload,created_at FROM agent_events "
                                                       "WHERE task_id=? ORDER BY id", (task_id,))]
            return state

    def list(self):
        with self.connect() as db:
            states = [json.loads(row[0]) for row in db.execute("SELECT state FROM agent_tasks")]
        keys = ("id", "request", "workspace", "status", "steps", "max_steps", "answer", "error",
                "created_at", "updated_at", "parent_id", "model_calls")
        return [{key: state.get(key) for key in keys}
                for state in sorted(states, key=lambda s: s["created_at"], reverse=True)]
