"""Optional bounded projection to the current Langfuse Python SDK API.

Verified against official sources on 2026-10-11:
https://python.reference.langfuse.com/langfuse (Langfuse constructor timeout,
start_observation with trace_context/as_type/model_parameters/usage_details,
observation.update/end, and flush() without a timeout argument).
https://langfuse.com/docs/observability/sdk/advanced-features (isolated provider).
https://langfuse.com/docs/observability/features/masking (export-stage masking).
https://langfuse.com/docs/observability/features/token-and-cost-tracking
(inclusive provider counts must become mutually exclusive usage buckets).

No SDK is imported, initialized, or contacted unless enabled=True. Application
calls never wait for external delivery. flush has a local deadline; an already
running SDK call cannot be forcibly cancelled, so at most one daemon flush is
retained. Neither delivery nor flush retries business/model operations.

Queued observations are SDK delivery spans. Their native span timestamps are
not backdated to model requests: the current start_observation API has no
start_time argument. Use the saved event timestamp/duration_ms for performance
analysis, not the delivery span's native latency display.
"""

from collections import OrderedDict
import hashlib
import math
import queue
import threading
import time

from .redaction import external_event


def _exclusive_usage(usage):
    """Convert inclusive OpenAI/Ollama totals without double counting details.

Unknown details remain inside the inclusive base bucket; never estimate them.
Raw nullable provider components remain in the projected event metadata.
    """
    usage = usage or {}
    result = {}
    for total_key, base_key, details in (
        ("input_tokens", "input", (("cache_read_tokens", "input_cached_tokens"),
                                   ("cache_write_tokens", "input_cache_write_tokens"))),
        ("output_tokens", "output", (("reasoning_tokens", "output_reasoning_tokens"),)),
    ):
        known = [(name, usage.get(key)) for key, name in details if type(usage.get(key)) is int]
        total = usage.get(total_key)
        if type(total) is int:
            subtotal = sum(value for _, value in known)
            if subtotal > total:
                return {}  # Inconsistent provider usage must not produce false totals.
            result[base_key] = total - subtotal
        result.update(dict(known))
    if type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int:
        result["total"] = usage["input_tokens"] + usage["output_tokens"]
    return result


class LangfuseExporter:
    def __init__(self, *, enabled=False, client=None, max_queue=128, request_timeout_seconds=5):
        if type(max_queue) is not int or max_queue < 1:
            raise ValueError("max_queue must be a positive integer")
        if (type(request_timeout_seconds) not in (int, float) or
                not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0):
            raise ValueError("request_timeout_seconds must be positive and finite")
        self.enabled = bool(enabled)
        self.client = client
        self.timeout = request_timeout_seconds
        self.queue = queue.Queue(maxsize=max_queue)
        self._observations = OrderedDict()
        self._limit = max_queue
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._closed = False
        self._flush_thread = None
        self._flush_done = threading.Event()
        self._stats = {"accepted": 0, "exported": 0, "dropped": 0, "failures": 0,
                       "pending": 0, "last_error_type": None}
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._work, daemon=True, name="samantha-langfuse-export")
            try:
                self._thread.start()
            except Exception as exc:
                self._failure(exc)
                self._thread = None
                self.enabled = False

    def submit(self, event):
        if not self.enabled or self._closed:
            return False
        try:
            safe = external_event(event)
            with self._condition:
                self.queue.put_nowait(safe)
                self._stats["accepted"] += 1
                self._stats["pending"] += 1
                self._condition.notify_all()
            return True
        except queue.Full:
            with self._condition:
                self._stats["dropped"] += 1
            return False
        except Exception as exc:
            self._failure(exc)
            return False

    def stats(self):
        with self._condition:
            return dict(self._stats)

    def _failure(self, exc):
        with self._condition:
            self._stats["failures"] += 1
            self._stats["last_error_type"] = type(exc).__name__  # Never exception text.
            self._condition.notify_all()

    def _client(self):
        if self.client is None:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider
            self.client = Langfuse(timeout=math.ceil(self.timeout), debug=False,
                                   tracer_provider=TracerProvider(),
                                   should_export_span=lambda span: span.name.startswith("samantha."))
        return self.client

    def _deliver(self, event):
        client = self._client()
        trace, data = event["trace"], event["data"]
        kind = event["kind"] or "unknown"
        key = (trace.get("trace_id"), trace.get("operation_id"))
        if kind == "operation_finished" and key in self._observations:
            observation = self._observations.pop(key)
            observation.update(metadata={"event": event})
            observation.end()
            return
        trace_id = trace.get("trace_id") or hashlib.sha256(str(trace.get("session_id")).encode()).hexdigest()[:32]
        # OTel requires a 32-character hex trace ID; preserve our opaque IDs in metadata.
        if len(trace_id) != 32 or any(character not in "0123456789abcdef" for character in trace_id):
            trace_id = hashlib.sha256(trace_id.encode()).hexdigest()[:32]
        context = {"trace_id": trace_id}
        parent_key = (trace.get("trace_id"), trace.get("parent_operation_id") if kind == "operation_started" else trace.get("operation_id"))
        parent = self._observations.get(parent_key)
        if parent is not None:
            parent_id = getattr(parent, "id", None)
            if isinstance(parent_id, str):
                context["parent_span_id"] = parent_id
        options = {"name": "samantha." + kind, "trace_context": context,
                   "as_type": "span", "metadata": {"event": event}}
        if kind == "model_call":
            options.update(as_type="embedding" if data.get("call_type") == "embedding" else "generation",
                           model=data.get("model"), model_parameters=data.get("effective_parameters", {}),
                           usage_details=_exclusive_usage(data.get("usage")),
                           level="ERROR" if data.get("status") == "failed" else "DEFAULT")
        observation = client.start_observation(**options)
        if trace.get("session_id"):
            observation.update_trace(session_id=trace["session_id"])
        if kind == "operation_started" and key[1]:
            self._observations[key] = observation
            while len(self._observations) > self._limit:
                _, expired = self._observations.popitem(last=False)
                expired.end()
        else:
            observation.end()

    def _work(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._deliver(event)
                with self._condition:
                    self._stats["exported"] += 1
            except Exception as exc:
                self._failure(exc)
            finally:
                self.queue.task_done()
                with self._condition:
                    self._stats["pending"] -= 1
                    self._condition.notify_all()

    def flush(self, timeout_seconds=2):
        if not self.enabled:
            return self.stats()["failures"] == 0
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            return False
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._stats["pending"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        if self.client is None:
            return self.stats()["failures"] == 0
        with self._condition:
            if self._flush_thread is None or not self._flush_thread.is_alive():
                self._flush_done.clear()

                def flush_sdk():
                    try:
                        self.client.flush()  # Current API has no timeout keyword.
                    except Exception as exc:
                        self._failure(exc)
                    finally:
                        self._flush_done.set()
                self._flush_thread = threading.Thread(target=flush_sdk, daemon=True, name="samantha-langfuse-flush")
                try:
                    self._flush_thread.start()
                except Exception as exc:
                    self._failure(exc)
                    return False
        finished = self._flush_done.wait(max(0, deadline - time.monotonic()))
        return finished and self.stats()["failures"] == 0

    def close(self, timeout_seconds=2):
        self._closed = True
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds < 0):
            self._stop.set()
            return False
        started = time.monotonic()
        flushed = self.flush(timeout_seconds)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(max(0, timeout_seconds - (time.monotonic() - started)))
        return flushed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
