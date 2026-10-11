"""Immutable ID/version bindings and hashes of template and rendered system text.

Legacy Python f-string templates retain their exact rendering. For those prompts
template_kind='python_source' explicitly identifies the rendering function source
as the versioned template representation; rendered_sha256 hashes its actual text.
"""

from dataclasses import dataclass
import hashlib
import inspect
import re
from threading import RLock


_templates = {}
_lock = RLock()


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    version: str
    text: str
    template_kind: str = "text"

    def metadata(self):
        return {"template_id": self.template_id, "version": self.version,
                "template_kind": self.template_kind, "template_sha256": _sha(self.text)}


class RenderedPrompt(str):
    def __new__(cls, text, metadata):
        value = super().__new__(cls, text)
        value.prompt_metadata = [dict(item) for item in metadata]
        return value


def register_template(template_id, text, *, version="1", template_kind="text"):
    if (not isinstance(template_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", template_id)
            or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", version)
            or not isinstance(text, str)):
        raise ValueError("Prompt ID/version must be stable identifiers and template must be text")
    template = PromptTemplate(template_id, version, text, template_kind)
    with _lock:
        previous = _templates.get((template_id, version))
        if previous is not None and previous != template:
            raise ValueError("A changed prompt template requires a new version")
        _templates[(template_id, version)] = template
    return template


def bind_prompt(template_id, rendered, *, template_text=None, version="1", template_kind="text"):
    template = register_template(template_id, rendered if template_text is None else template_text,
                                 version=version, template_kind=template_kind)
    return RenderedPrompt(rendered, [{**template.metadata(), "rendered_sha256": _sha(rendered)}])


def bind_source_prompt(template_id, rendered, source_function, *, version="1"):
    return bind_prompt(template_id, rendered, template_text=inspect.getsource(source_function),
                       version=version, template_kind="python_source")


def render_prompt(template_id, template_text, *, version="1", variables=None):
    rendered = template_text.format_map(dict(variables or {}))
    return bind_prompt(template_id, rendered, template_text=template_text, version=version)


def compose_prompt(prompt, suffix, *, template_id="terminal.text_envelope", version="1"):
    suffix_template = register_template(template_id, suffix, version=version)
    text = str(prompt) + suffix
    metadata = list(getattr(prompt, "prompt_metadata", []))
    metadata.append({**suffix_template.metadata(), "rendered_sha256": _sha(text)})
    return RenderedPrompt(text, metadata)


def describe_registry():
    """Hashes and IDs only; no local template contents are exported."""
    with _lock:
        return [template.metadata() for _, template in sorted(_templates.items())]
