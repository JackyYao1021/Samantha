"""Opt-in bounds for physical requests. Ordinary CLI calls have no budget hook."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
import math
import time


class BudgetExceeded(RuntimeError):
    pass


_active = ContextVar("model_request_budget", default=None)


@dataclass
class RequestBudget:
    max_calls: int
    max_seconds: float
    max_reported_tokens: int | None = None
    allowed_models: tuple[str, ...] = ("qwen3:4b",)
    calls: int = 0
    reported_tokens: int = 0
    missing_usage_calls: int = 0
    started: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock, repr=False)

    def snapshot(self):
        return {"max_calls": self.max_calls, "calls": self.calls,
                "max_seconds": self.max_seconds, "elapsed_seconds": round(time.monotonic() - self.started, 3),
                "max_reported_tokens": self.max_reported_tokens, "reported_tokens": self.reported_tokens,
                "missing_usage_calls": self.missing_usage_calls,
                "token_limit_kind": "returned_usage_stop; missing usage cannot prove a hard token cap"}


@contextmanager
def budget_context(*, max_calls, max_seconds, max_reported_tokens=None, allowed_models=("qwen3:4b",)):
    if (type(max_calls) is not int or max_calls < 1
            or type(max_seconds) not in (int, float)
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("A positive request count and deadline are required")
    if max_reported_tokens is not None and (type(max_reported_tokens) is not int or max_reported_tokens < 1):
        raise ValueError("Reported token budget must be positive")
    budget = RequestBudget(max_calls, max_seconds, max_reported_tokens, tuple(allowed_models))
    token = _active.set(budget)
    try:
        yield budget
    finally:
        _active.reset(token)


def before_request(provider, transport, model, parameters, *, admission=None):
    budget = _active.get()
    if budget is None:
        if admission is not None:
            admission()
        return None
    with budget.lock:
        if provider not in {"qwen", "ollama"} or model not in budget.allowed_models:
            raise BudgetExceeded("Experiment permits only its declared local model")
        if time.monotonic() - budget.started >= budget.max_seconds:
            raise BudgetExceeded("Model experiment deadline exceeded")
        if budget.calls >= budget.max_calls:
            raise BudgetExceeded("Physical model request budget exceeded")
        if budget.max_reported_tokens is not None and budget.reported_tokens >= budget.max_reported_tokens:
            raise BudgetExceeded("Reported model token budget exceeded")
        # Durable task admission runs only after the outer experiment has passed
        # validation, and before its counter changes. A rejected admission is not
        # a physical request in either budget.
        if admission is not None:
            admission()
        budget.calls += 1
    return budget


def after_request(ticket, record):
    if ticket is None:
        return
    usage = record.get("usage") or {}
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    with ticket.lock:
        if all(type(value) is int and value >= 0 for value in values):
            ticket.reported_tokens += sum(values)
        else:
            ticket.missing_usage_calls += 1
