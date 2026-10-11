"""Explicit, bounded local-Qwen proposal evaluation; no generated host commands."""

from contextlib import contextmanager
import ipaddress
import json
import os
import socket
from urllib.request import ProxyHandler, HTTPRedirectHandler, build_opener

ENDPOINT = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"


def model_identity():
    """Read only the authorized local server inventory, with no ambient proxy."""
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            raise ValueError("Local model inventory must not redirect outside its declared endpoint")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(ENDPOINT + "/api/tags", timeout=5) as response:
        models = json.load(response).get("models", [])
    match = next((item for item in models if item.get("name") == MODEL), None)
    if match is None or not isinstance(match.get("digest"), str):
        raise ValueError("Authorized local qwen3:4b model is unavailable")
    return {"provider": "ollama", "model": MODEL, "digest": match["digest"], "endpoint": ENDPOINT}


@contextmanager
def local_worker(settings):
    """Admission policy plus a socket gate, installed only in a disposable worker."""
    from model_budget import budget_context
    original_socket = socket.socket

    class LocalSocket(original_socket):
        def connect(self, address):
            if (not isinstance(address, tuple) or len(address) < 2 or address[1] != 11434
                    or not ipaddress.ip_address(address[0]).is_loopback):
                raise RuntimeError("Evaluation network permits only loopback Ollama port 11434")
            return super().connect(address)

        def connect_ex(self, address):
            try:
                self.connect(address)
                return 0
            except OSError as exc:
                return exc.errno

    environment = dict(QWEN_MODEL=MODEL, QWEN_BASE_URL=ENDPOINT + "/v1",
                      QWEN_API_MODE=settings["model"]["transport"], QWEN_REASONING_EFFORT="none",
                      QWEN_EVAL_MAX_TOKENS=str(settings["model"]["max_output_tokens"]),
                      QWEN_TIMEOUT=str(settings["model"]["request_timeout_seconds"]),
                      SAMANTHA_AZURE_FALLBACK="false")
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    socket.socket = LocalSocket
    try:
        with budget_context(max_calls=settings["budget"]["max_calls_per_attempt"],
                            max_seconds=settings["timeout_seconds"], allowed_models=(MODEL,)) as budget:
            yield budget
    finally:
        socket.socket = original_socket
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def proposal_services(public_input, root):
    from workflow import WorkflowServices, default_services
    base = default_services(object_tracking=False)
    answers = iter(public_input["interaction"]["clarification_answers"])

    def answer(prompt):
        return "n" if prompt == "Proceed? (y/n): " else next(answers, "cancel")

    def reject_execution(commands):
        raise RuntimeError("Proposal evaluation forbids all generated command execution")

    return WorkflowServices(base.clarify, base.parse_intent, base.generate_commands,
                            base.review_commands, base.correct_error, reject_execution), answer
