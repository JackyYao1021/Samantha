"""Bounded, explicit local Ollama calls for the routing experiment.

No ambient proxy, credentials, remote fallback, model pulls, or shell execution.
Every submitted request is journaled before transport and retains raw evidence.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .harness import write_json

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_budget import BudgetExceeded
from model_telemetry import model_request


ENDPOINT = "http://127.0.0.1:11434"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class EvidenceWriteError(RuntimeError):
    """Durability failed; this must never initiate another model request."""


def _save(path, value):
    try:
        write_json(path, value)
    except OSError as exc:
        raise EvidenceWriteError("Model evidence could not be persisted") from exc


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError("Local experiment requests must not redirect")


def local_opener():
    return build_opener(ProxyHandler({}), NoRedirect())


def strict_model_json(content):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate model-output key: " + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(content, object_pairs_hook=unique, parse_constant=invalid_constant)


def read_inventory(opener=None):
    opener = opener or local_opener()
    with opener.open(ENDPOINT + "/api/tags", timeout=5) as response:
        data = json.load(response)
    models = {}
    for item in data.get("models", []):
        if isinstance(item.get("name"), str) and isinstance(item.get("digest"), str):
            models[item["name"]] = {key: item.get(key) for key in ("name", "digest", "size", "details")}
    return models


def _append(path, value):
    try:
        with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except OSError as exc:
        raise EvidenceWriteError("Model request journal could not be persisted") from exc


class OllamaClient:
    def __init__(self, output, *, allowed_models, request_timeout=35, deadline=None,
                 max_output_tokens=768, seed=17, opener=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.allowed_models = tuple(allowed_models)
        self.request_timeout = request_timeout
        self.deadline = deadline
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.opener = opener or local_opener()
        self.calls = []

    def _post(self, path, payload, *, call_type="chat", role="solver"):
        model = payload.get("model")
        if model not in self.allowed_models or path not in {"/api/chat", "/api/embed"}:
            raise ValueError("Request is outside the declared local model experiment")
        remaining = self.deadline - time.monotonic() if self.deadline is not None else self.request_timeout
        if remaining <= 0:
            raise BudgetExceeded("Routing experiment wall-time budget exhausted")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ticket_number = len(self.calls) + 1
        request_dir = self.output / f"request_{ticket_number:04d}"
        try:
            request_dir.mkdir(exist_ok=False)
        except OSError as exc:
            raise EvidenceWriteError("Model request evidence directory could not be created") from exc
        _save(request_dir / "request.json", payload)
        try:
            (request_dir / "request_body.bin").write_bytes(body)
        except OSError as exc:
            raise EvidenceWriteError("Exact request bytes could not be persisted") from exc
        metadata = {"request_number": ticket_number, "model": model, "role": role,
                    "call_type": call_type, "request_sha256": hashlib.sha256(body).hexdigest(),
                    "request_ref": request_dir.relative_to(self.output.parent).as_posix() + "/request.json"}
        started = time.monotonic()
        from observability.events import capture_events
        try:
            with capture_events() as events:
                with model_request(provider="ollama", transport="ollama", model=model,
                                   parameters=payload, call_type=call_type) as outcome:
                    # Admission has happened. This record conservatively counts a
                    # request even if the process exits before completion evidence.
                    _append(self.output.parent / "requests.jsonl", {**metadata, "event": "submitted"})
                    request = Request(ENDPOINT + path, data=body,
                                      headers={"Content-Type": "application/json"}, method="POST")
                    with self.opener.open(request, timeout=min(self.request_timeout, remaining)) as response:
                        raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ValueError("Model response exceeds evidence byte bound")
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError("Ollama response must be a JSON object")
                    outcome["response"] = data
                    _save(request_dir / "response.json", data)
            call = next(event["data"] for event in reversed(events) if event["kind"] == "model_call")
            compute = [data.get("prompt_eval_duration"), data.get("eval_duration")]
            call["provider_compute_ms"] = (sum(compute) / 1e6
                if all(type(value) is int and value >= 0 for value in compute) else None)
            load = data.get("load_duration")
            call["provider_load_ms"] = load / 1e6 if type(load) is int and load >= 0 else None
            call.update(metadata)
            self.calls.append(call)
            _save(request_dir / "telemetry.json", call)
            _append(self.output.parent / "requests.jsonl", {**metadata, "event": "finished", "status": call["status"]})
            return data
        except BaseException as exc:
            # A rejected budget admission has no physical request event.
            found = [event["data"] for event in locals().get("events", []) if event["kind"] == "model_call"]
            persistence_error = exc if isinstance(exc, EvidenceWriteError) else None
            if found:
                existing = next((item for item in self.calls if item["call_id"] == found[-1]["call_id"]), None)
                call = existing or {**found[-1], **metadata, "provider_compute_ms": None, "provider_load_ms": None}
                if existing is None:
                    self.calls.append(call)
                if persistence_error is not None:
                    call["evidence_error"] = "persistence_failed"
                try:
                    _save(request_dir / "telemetry.json", call)
                    _append(self.output.parent / "requests.jsonl", {**metadata, "event": "evidence_error" if persistence_error else "finished",
                                                                  "status": call["status"]})
                except EvidenceWriteError as saving:
                    persistence_error = saving
                    call["evidence_error"] = "persistence_failed"
            try:
                _save(request_dir / "error.json", {"type": type(exc).__name__,
                      "message": str(exc), "duration_ms": (time.monotonic() - started) * 1000})
            except EvidenceWriteError as saving:
                persistence_error = saving
            if persistence_error is not None:
                raise persistence_error from exc
            raise

    def chat(self, messages, schema, *, model, role="solver", max_output_tokens=None):
        payload = {"model": model, "messages": messages, "stream": False, "format": schema,
                   "options": {"temperature": 0, "seed": self.seed, "num_ctx": 4096,
                               "num_predict": max_output_tokens or self.max_output_tokens},
                   "keep_alive": "5m"}
        # Qwen2.5-VL has no thinking mode. Do not send unsupported settings.
        if model.split(":")[0] == "qwen3":
            payload["think"] = False
        data = self._post("/api/chat", payload, role=role)
        if data.get("done") is not True or data.get("done_reason") == "length":
            raise ValueError("Model output is incomplete or hit its output-token bound")
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Model response is missing text content")
        return strict_model_json(content)

    def embed(self, texts, *, model):
        data = self._post("/api/embed", {"model": model, "input": texts, "truncate": False,
                          "keep_alive": "5m"}, call_type="embedding", role="embedding_router")
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Embedding response does not match the submitted input count")
        return vectors
