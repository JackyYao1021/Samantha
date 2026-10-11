"""Recursive privacy filtering and a strict external telemetry projection.

Local journals retain their existing audit semantics. External export accepts
only typed telemetry fields; arbitrary metadata/freeform is never exported.
"""

import math
import re
from datetime import datetime


REDACTED = "[redacted]"
_SECRET_KEYS = re.compile(r"(?:^|_)(?:authorization|api_key|apikey|token|access_token|refresh_token|secret|password|passwd|credential|cookie|private_key)(?:$|_)", re.I)
_PRIVATE_KEYS = re.compile(r"(?:prompt$|messages$|document|image|base64|reasoning(?:_content)?$|thinking$|freeform$|source_text|content_utf8|(?:^|_)(?:input|output|text|content)$)", re.I)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{6,}|Bearer\s+\S+|Basic\s+[A-Za-z0-9+/=]+)", re.I)
_ASSIGNMENT = re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret|password|authorization)\s*[:=]\s*[^\s,;]+", re.I)
_URL = re.compile(r"(?:https?|ftp)://[^\s\"'<>]+", re.I)
_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\r\n\"'<>]+|\\\\[^\r\n\"'<>]+|(?<![\w:])/(?:[^\s\"'<>]+))")
_DATA = re.compile(r"data:[^\s,]+;base64,[A-Za-z0-9+/=]+", re.I)
_BASE64 = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9])")
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.I | re.S)


def _key(key):
    if not isinstance(key, str):
        return ""
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def redact(value, *, secrets=(), max_depth=12, max_items=512, max_text=4096):
    """Return a bounded JSON-compatible copy, never stringify opaque objects.

Credential values found under secret keys are also removed from sibling text.
Unknown freeform secrets require the external field allowlist below.
    """
    found = {secret for secret in secrets if isinstance(secret, str) and secret}
    visited = set()

    def collect(item, depth):
        if depth > max_depth or id(item) in visited:
            return
        if isinstance(item, (dict, list, tuple)):
            visited.add(id(item))
        if isinstance(item, dict):
            for name, child in list(item.items())[:max_items]:
                if _SECRET_KEYS.search(_key(name)) and isinstance(child, str) and child:
                    found.add(child)
                else:
                    collect(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item[:max_items]:
                collect(child, depth + 1)
    collect(value, 0)
    visited.clear()

    def clean(item, depth):
        if depth > max_depth:
            return REDACTED
        if isinstance(item, str):
            text = item
            for secret in sorted(found, key=len, reverse=True):
                text = text.replace(secret, REDACTED)
            for pattern in (_THINK, _DATA, _URL, _TOKEN, _ASSIGNMENT, _PATH, _BASE64):
                text = pattern.sub(REDACTED, text)
            return text[:max_text]
        if item is None or type(item) in (int, bool):
            return item
        if type(item) is float:
            return item if math.isfinite(item) else None
        if isinstance(item, (dict, list, tuple)):
            if id(item) in visited:
                return REDACTED
            visited.add(id(item))
            if isinstance(item, dict):
                result = {}
                for name, child in list(item.items())[:max_items]:
                    name = str(name) if isinstance(name, (str, int)) else REDACTED
                    normalized = _key(name)
                    safe_name = clean(name, depth + 1)
                    result[safe_name] = (REDACTED if _SECRET_KEYS.search(normalized) or
                                        _PRIVATE_KEYS.search(normalized) else clean(child, depth + 1))
            else:
                result = [clean(child, depth + 1) for child in item[:max_items]]
            visited.remove(id(item))
            return result
        return REDACTED
    return clean(value, 0)


def _label(value):
    if (isinstance(value, str) and len(value) <= 256 and
            re.fullmatch(r"[A-Za-z0-9_.:\-]+", value) and not _TOKEN.search(value)):
        return value
    return None


def _hash(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) else None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _timestamp(value):
    try:
        return datetime.fromisoformat(value).isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def external_event(event):
    """Drop private text and arbitrary metadata instead of guessing its contents."""
    event = event if isinstance(event, dict) else {}
    source = event.get("data") if isinstance(event.get("data"), dict) else {}
    data = {}
    for name in ("provider", "transport", "model", "call_type", "status", "exception_type", "name",
                 "operation_name", "operation_id", "call_id", "provider_call_id", "finish_reason", "source", "stream"):
        if name in source:
            data[name] = _label(source[name])
    for name in ("duration_ms", "physical_llm_calls", "physical_embedding_calls", "tool_calls"):
        if name in source:
            data[name] = _number(source[name])
    if "physical_request" in source and type(source["physical_request"]) is bool:
        data["physical_request"] = source["physical_request"]
    for name in ("api_cost", "api_cost_usd"):
        if name in source:
            data[name] = None  # No price snapshot is supplied by this exporter.
    if isinstance(source.get("usage"), dict):
        data["usage"] = {name: _number(source["usage"].get(name)) for name in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")}
    if isinstance(source.get("effective_parameters"), dict):
        params = source["effective_parameters"]
        data["effective_parameters"] = {name: _hash(value) for name, value in params.items()
                                        if name.endswith("_sha256")}
        for name in ("max_tokens", "temperature", "top_p", "num_predict", "stream", "think", "truncate", "batch_size"):
            if name in params:
                data["effective_parameters"][name] = params[name] if type(params[name]) is bool else _number(params[name])
        if params.get("reasoning_effort") in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            data["effective_parameters"]["reasoning_effort"] = params["reasoning_effort"]
    if isinstance(source.get("prompt_templates"), list):
        data["prompt_templates"] = [{"template_id": _label(item.get("template_id")),
                                     "version": _label(item.get("version")),
                                     "template_kind": _label(item.get("template_kind")),
                                     "template_sha256": _hash(item.get("template_sha256")),
                                     "rendered_sha256": _hash(item.get("rendered_sha256"))}
                                    for item in source["prompt_templates"][:16] if isinstance(item, dict)]
    trace = event.get("trace") if isinstance(event.get("trace"), dict) else {}
    result = {"schema_version": "1.1", "event_id": _label(event.get("event_id")),
              "timestamp": _timestamp(event.get("timestamp")), "kind": _label(event.get("kind")),
              "role": _label(event.get("role")), "trace": {name: _label(trace.get(name)) for name in (
                  "trace_id", "experiment_id", "episode_id", "attempt_id", "session_id", "operation_id",
                  "parent_operation_id", "call_id")}, "data": data}
    return redact(result)
