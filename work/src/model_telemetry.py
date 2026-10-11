"""Local telemetry for actual model requests, with no prompt or credential text.

SDK retries must be disabled at the caller so one recorded request corresponds
to one transport attempt. Missing provider usage is unknown, never zero.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import json
import math
import re
import time
import uuid

from observability.events import current_trace, emit_event, trace_context
from model_budget import before_request, after_request


USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens",
)
_captures = ContextVar("samantha_model_call_captures", default=())


@contextmanager
def capture_model_calls():
    """Yield request records; nested captures also belong to their outer capture.

Context variables isolate independent episodes and follow the context copied by
LangGraph workers. The returned list remains readable after the context exits.
"""
    calls = []
    token = _captures.set((*_captures.get(), calls))
    try:
        yield calls
    finally:
        _captures.reset(token)


def _digest(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=lambda obj: {
                                 "unsupported_type": type(obj).__name__,
                             }).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value):
    # Endpoints, URL userinfo, and typical API keys are not model identifiers.
    if (not isinstance(value, str) or len(value) > 200 or
            not re.fullmatch(r"[A-Za-z0-9_./:\-]+", value) or "://" in value or
            value.lower().startswith(("sk-", "sk_", "bearer"))):
        return "[redacted]"
    return value


def _effective_parameters(parameters):
    # Allowlist request controls. Hash messages and schemas instead of saving any
    # free text; never include request headers, base URLs, API keys, or responses.
    result = {"prompt_sha256": _digest(parameters.get("messages", parameters.get("input")))}
    for key in ("max_tokens", "temperature", "top_p", "num_predict", "stream", "truncate", "batch_size"):
        if key in parameters:
            value = parameters[key]
            result[key] = (value if type(value) in (int, bool) or
                           (type(value) is float and math.isfinite(value)) else None)
    for key in ("reasoning_effort", "think"):
        if key in parameters:
            value = parameters[key]
            if type(value) is bool or value in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                result[key] = value
            else:
                result[key + "_sha256"] = _digest(value)
    for key in ("response_format", "format"):
        if key in parameters:
            result[key + "_sha256"] = _digest(parameters[key])
    return result


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _tokens(value):
    return value if type(value) is int and value >= 0 else None


def _first_tokens(*values):
    return next((value for value in values if _tokens(value) is not None), None)


def _usage(response, transport, call_type):
    empty = dict.fromkeys(USAGE_FIELDS)
    try:
        if transport == "ollama":
            return {**empty,
                    "input_tokens": _tokens(_field(response, "prompt_eval_count")),
                    "output_tokens": None if call_type == "embedding" else _tokens(_field(response, "eval_count"))}
        usage = _field(response, "usage")
        inputs = _field(usage, "prompt_tokens_details") or _field(usage, "input_tokens_details")
        outputs = _field(usage, "completion_tokens_details") or _field(usage, "output_tokens_details")
        return {
            "input_tokens": _first_tokens(_field(usage, "prompt_tokens"), _field(usage, "input_tokens"),
                                           _field(usage, "total_tokens") if call_type == "embedding" else None),
            "output_tokens": None if call_type == "embedding" else _first_tokens(_field(usage, "completion_tokens"), _field(usage, "output_tokens")),
            "cache_read_tokens": _first_tokens(_field(inputs, "cached_tokens"),
                                                _field(usage, "cache_read_tokens")),
            "cache_write_tokens": _first_tokens(_field(inputs, "cache_write_tokens"),
                                                 _field(usage, "cache_write_tokens")),
            "reasoning_tokens": _first_tokens(_field(outputs, "reasoning_tokens"),
                                               _field(usage, "reasoning_tokens")),
        }
    except Exception:
        # A nonstandard usage object must not change a completed model request.
        return empty


def _publish(record):
    for calls in _captures.get():
        calls.append(copy.deepcopy(record))
    # Reuse the journal's context without opening a new session or storing raw
    # request data. Journal failures retain the existing fail-closed behavior.
    from interaction_log import _active_operation, _active_session
    session = _active_session.get()
    if session is not None:
        session.record("model_call", {**(_active_operation.get() or {}), **record}, role="model")
    else:
        emit_event("model_call", record, role="model")


@contextmanager
def model_request(*, provider, transport, model, parameters, call_type="chat",
                  prompt_templates=None, physical_request=True):
    """Record one request boundary; assign the provider response to result.

Construct and validate payloads before entering this context. Its clock starts
immediately before create/post; response parsing may occur inside the boundary.
    """
    # Budget admission happens before the transport and outside request failure
    # recording. It must never be mistaken for a failed physical model request.
    ticket = (before_request(provider=provider, transport=transport, model=model, parameters=parameters)
              if physical_request else None)
    record = {
        "call_id": uuid.uuid4().hex,
        "provider": provider,
        "transport": transport,
        "model": _identifier(model),
        "effective_parameters": _effective_parameters(parameters),
        "status": "succeeded",
        "exception_type": None,
        "usage": dict.fromkeys(USAGE_FIELDS),
        "api_cost": None,
        "call_type": call_type,
        "physical_request": physical_request,
        "prompt_templates": list(prompt_templates or []),
    }
    trace_scope = trace_context(call_id=record["call_id"])
    trace_scope.__enter__()
    record["trace"] = current_trace()
    outcome = {}
    started = time.monotonic()
    try:
        yield outcome
    except BaseException as exc:
        record.update(status="failed", exception_type=type(exc).__name__)
        raise
    finally:
        record["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        response = outcome.get("response")
        record["usage"] = _usage(response, transport, call_type)
        try:
            if transport == "ollama":
                finish_reason = _field(response, "done_reason")
                provider_call_id = None
            else:
                choices = _field(response, "choices")
                finish_reason = _field(choices[0], "finish_reason") if isinstance(choices, list) and choices else None
                provider_call_id = _field(response, "id")
        except Exception:
            finish_reason, provider_call_id = None, None
        record["finish_reason"] = _identifier(finish_reason) if isinstance(finish_reason, str) else None
        record["provider_call_id"] = _identifier(provider_call_id) if isinstance(provider_call_id, str) else None
        try:
            try:
                _publish(record)
            finally:
                if ticket is not None:
                    after_request(ticket, record)
        finally:
            trace_scope.__exit__(None, None, None)


def summarize_calls(calls):
    """Aggregate complete usage and expose known partial totals in coverage.

If even one attempted call lacks a token component, that aggregate is null.
Zero calls also has unknown usage; no pricing snapshot means unknown API cost.
    """
    calls = list(calls)
    usage = {}
    coverage = {}
    for key in USAGE_FIELDS:
        values = [_tokens((call.get("usage") or {}).get(key)) for call in calls]
        reported = [value for value in values if value is not None]
        usage[key] = sum(reported) if calls and len(reported) == len(calls) else None
        coverage[key] = {
            "reported_calls": len(reported), "total_calls": len(calls),
            "ratio": len(reported) / len(calls) if calls else None,
            "reported_sum": sum(reported) if reported else None,
        }
    physical = [call for call in calls if call.get("physical_request", True)]
    embeddings = [call for call in physical if call.get("call_type") == "embedding"]
    return {
        "physical_llm_calls": len(physical) - len(embeddings),
        "physical_embedding_calls": len(embeddings),
        "physical_model_calls": len(physical),
        "local_embedding_calls": sum(call.get("call_type") == "embedding" and not call.get("physical_request", True)
                                     for call in calls),
        "duration_ms": round(sum(call["duration_ms"] for call in calls), 3),
        "usage": usage,
        "usage_coverage": coverage,
        "api_cost": None,
    }
