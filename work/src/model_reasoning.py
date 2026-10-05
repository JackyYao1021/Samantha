"""Capture reasoning explicitly returned by a model API before answer parsing."""

import uuid

from interaction_log import record_model_reasoning


def _field(message, name):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def log_message_reasoning(message, *, model, provider, transport, finish_reason=None,
                          response_complete=True):
    blocks = []
    for source in ("thinking", "reasoning_content", "reasoning"):
        text = _field(message, source)
        if isinstance(text, str) and text.strip():
            blocks.append((source, text, True))
    content = _field(message, "content")
    if isinstance(content, str):
        # Match the final-answer parser's envelope rules. Literal tags inside
        # JSON or a code fence belong to the answer, not to model reasoning.
        text = content.lstrip()
        if text.startswith("<think>"):
            thought, closing, _ = text[len("<think>"):].partition("</think>")
            if thought.strip():
                blocks.append(("think_tag", thought, bool(closing)))
        elif "</think>" in text and not text.startswith(("{", "[", "```")):
            thought, _, _ = text.partition("</think>")
            if thought.strip():
                blocks.append(("think_tag", thought, True))
    metadata = {"call_id": uuid.uuid4().hex, "model": model, "provider": provider,
                "transport": transport, "finish_reason": finish_reason}
    complete = response_complete and finish_reason not in {"length", "content_filter"}
    if not blocks:
        record_model_reasoning({**metadata, "available": False, "source": None,
                                "text": None, "complete": complete})
    for source, text, closed in blocks:
        record_model_reasoning({**metadata, "available": True, "source": source,
                                "text": text, "complete": complete and closed})


def log_response_reasoning(response, *, model, provider, transport="openai"):
    choices = getattr(response, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        choice = choices[0]
        log_message_reasoning(choice.message, model=model, provider=provider,
                              transport=transport, finish_reason=choice.finish_reason)
