"""Dependency-free raw-session baselines and an explicit adapter factory boundary."""

from collections import Counter
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import re
import sqlite3

from .core import MemoryHit, MemoryQuery, MemorySession


def terms(text):
    # Preserve Latin identifiers/paths; Chinese characters and bigrams allow local
    # Chinese queries without a tokenizer dependency. This is a lexical baseline.
    result = re.findall(r"[a-z0-9_./-]+", text.casefold())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        result.extend(run)
        result.extend(run[index:index + 2] for index in range(len(run) - 1))
    return result


class NoMemory:
    def ingest(self, session):
        pass

    def retrieve(self, query):
        return []

    def describe(self):
        return {"backend": "none", "version": 1}

    def close(self):
        pass


class SessionMemory:
    """Persistent raw-history baseline, not an extracted-fact memory system.

    SQLite is the source store; BM25 is computed over the scoped corpus per query.
    Adequate for a baseline, with no claim of a production-scale search index.
    """

    def __init__(self, path=None, *, strategy="recent"):
        if strategy not in {"recent", "bm25"}:
            raise ValueError("Unknown raw-history strategy.")
        self.strategy = strategy
        if path is not None:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            path = str(target)
        self.connection = sqlite3.connect(path or ":memory:", timeout=30)
        self.connection.row_factory = sqlite3.Row
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.close()
            raise ValueError("Unsupported memory database version.")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS memory_sessions (
                user_id TEXT NOT NULL, project_id TEXT NOT NULL,
                session_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                messages TEXT NOT NULL, metadata TEXT NOT NULL,
                PRIMARY KEY(user_id, project_id, session_id)
            );
            CREATE INDEX IF NOT EXISTS memory_scope_time
                ON memory_sessions(user_id, project_id, timestamp);
            PRAGMA user_version=1;
        """)

    def ingest(self, session: MemorySession):
        with self.connection:
            self.connection.execute("INSERT INTO memory_sessions VALUES (?,?,?,?,?,?) "
                                    "ON CONFLICT(user_id,project_id,session_id) DO UPDATE SET "
                                    "timestamp=excluded.timestamp,messages=excluded.messages,"
                                    "metadata=excluded.metadata", (
                session.scope.user_id, session.scope.project_id, session.session_id,
                session.timestamp, json.dumps(session.messages, ensure_ascii=False),
                json.dumps(session.metadata, ensure_ascii=False),
            ))

    def retrieve(self, query: MemoryQuery):
        statement = ("SELECT * FROM memory_sessions WHERE user_id=? AND project_id=? AND timestamp<=? "
                     "ORDER BY timestamp DESC, session_id DESC")
        parameters = [query.scope.user_id, query.scope.project_id, query.as_of]
        if self.strategy == "recent":
            statement += " LIMIT ?"
            parameters.append(query.limit)
        rows = list(self.connection.execute(statement, parameters))
        hits = [MemoryHit(row["session_id"], json.loads(row["metadata"]).get("source_session_id", row["session_id"]),
                          row["timestamp"],
                          json.dumps({"messages": json.loads(row["messages"]),
                                      "metadata": json.loads(row["metadata"])}, ensure_ascii=False))
                for row in rows]
        if self.strategy == "recent":
            return hits[:query.limit]
        documents = [Counter(terms(hit.text)) for hit in hits]
        query_terms = set(terms(query.text))
        average = sum(sum(doc.values()) for doc in documents) / max(1, len(documents))
        frequencies = Counter(term for doc in documents for term in doc)
        ranked = []
        for hit, doc in zip(hits, documents):
            length = sum(doc.values())
            score = 0.0
            for term in query_terms & doc.keys():
                inverse = math.log(1 + (len(documents) - frequencies[term] + 0.5) /
                                   (frequencies[term] + 0.5))
                count = doc[term]
                score += inverse * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / average))
            if score > 0:
                ranked.append(MemoryHit(hit.memory_id, hit.session_id, hit.timestamp, hit.text, score))
        return sorted(ranked, key=lambda hit: (-hit.score,
                      -datetime.fromisoformat(hit.timestamp).timestamp(), hit.memory_id))[:query.limit]

    def describe(self):
        return {"backend": self.strategy, "version": 1, "granularity": "session",
                "storage": "sqlite", "tokenizer": "latin-identifiers+han-unigrams-bigrams"}

    def close(self):
        self.connection.close()


def create_backend(name="none", *, path=None, options=None):
    """External adapters use an explicit module:factory; no vendor imports in agents.

    Factory signature: factory(path=..., options={...}) -> MemoryBackend.
    Adapter code and credentials/configuration are owned by the caller.
    """
    options = dict(options or {})
    if name in {"none", "recent", "bm25"} and options:
        raise ValueError("Built-in backends do not accept extra options.")
    if name == "none":
        backend = NoMemory()
    elif name in {"recent", "bm25"}:
        backend = SessionMemory(path, strategy=name)
    elif ":" in name:
        module, factory = name.split(":", 1)
        backend = getattr(importlib.import_module(module), factory)(path=path, options=options)
    else:
        raise ValueError(f"Unknown memory backend: {name}")
    if not all(callable(getattr(backend, method, None))
               for method in ("ingest", "retrieve", "describe", "close")):
        raise TypeError("Memory adapter does not implement MemoryBackend.")
    return backend
