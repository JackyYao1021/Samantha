"""Context propagation and versioned events; no external exporter by default."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import copy
import hashlib
import math
from pathlib import PurePath
import uuid


TRACE_FIELDS = ("trace_id", "experiment_id", "episode_id", "attempt_id", "session_id",
                "operation_id", "parent_operation_id", "call_id")
_trace = ContextVar("samantha_trace_context", default={})
_captures = ContextVar("samantha_event_captures", default=())
_exporters = ContextVar("samantha_event_exporters", default=())


def _json_safe(value, *, seen=None, depth=0):
    """Copy local event facts without invoking opaque object repr/serialization.

    Local paths remain available for audit; the external projection removes
    them. Bytes retain only a digest and size. Cycles and unknown objects are
    explicit markers, so collecting telemetry cannot break evidence writing.
    """
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if not isinstance(value, (dict, list, tuple)):
        return {"opaque_type": type(value).__name__}
    seen = set() if seen is None else seen
    if depth >= 24 or id(value) in seen:
        return {"opaque_type": "recursive_or_deep_payload"}
    seen.add(id(value))
    try:
        if isinstance(value, dict):
            result = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 4096:
                    result["capture_truncated"] = True
                    break
                if isinstance(key, str):
                    result[key] = _json_safe(item, seen=seen, depth=depth + 1)
            return result
        return [_json_safe(item, seen=seen, depth=depth + 1) for item in value[:4096]]
    finally:
        seen.remove(id(value))


def current_trace(**overrides):
    values = {name: _trace.get().get(name) for name in TRACE_FIELDS}
    values.update({name: value for name, value in overrides.items() if name in TRACE_FIELDS})
    return values


@contextmanager
def trace_context(**identifiers):
    """Bind supplied IDs without mutating outer context; copy across processes explicitly.

An experiment/episode/attempt tuple gives a deterministic trace ID so controller
and worker can bind the same tuple independently. Other roots use a random ID.
    """
    if set(identifiers) - set(TRACE_FIELDS):
        raise ValueError("Unknown trace context field")
    if any(value is not None and (not isinstance(value, str) or not value or len(value) > 256)
           for value in identifiers.values()):
        raise ValueError("Trace identifiers must be nonempty strings or null")
    values = current_trace(**identifiers)
    if not values["trace_id"]:
        parts = [values[name] for name in ("experiment_id", "episode_id", "attempt_id")]
        values["trace_id"] = (hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]
                              if all(parts) else uuid.uuid4().hex)
    token = _trace.set(values)
    try:
        yield dict(values)
    finally:
        _trace.reset(token)


@contextmanager
def capture_events():
    events = []
    token = _captures.set((*_captures.get(), events))
    try:
        yield events
    finally:
        _captures.reset(token)


@contextmanager
def use_exporter(exporter):
    """Attach an explicit submit(event) sink; sink errors cannot replay business work."""
    token = _exporters.set((*_exporters.get(), exporter))
    try:
        yield exporter
    finally:
        _exporters.reset(token)


def emit_event(kind, data, *, role="samantha", trace=None):
    event = {"schema_version": "1.1", "event_id": uuid.uuid4().hex,
             "timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind,
             "role": role, "trace": dict(trace) if trace is not None else current_trace(),
             "data": data}
    try:
        event = _json_safe(event)
    except Exception:
        event["data"] = {"capture_error": "opaque_payload"}
    for events in _captures.get():
        try:
            events.append(copy.deepcopy(event))
        except Exception:
            events.append({**event, "data": {"capture_error": "opaque_payload"}})
    for exporter in _exporters.get():
        try:
            exporter.submit(event)
        except Exception:
            # Export is a projection of committed local facts, not business work.
            pass
    return event
