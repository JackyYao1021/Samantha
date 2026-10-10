"""Validate proposed file effects and bind them to a small, literal Bash grammar.

Effects describe expected outcomes, never completed work.  This module does not
run commands or inspect files.  Unsupported command batches can use ``Effects=[]``
and retain the existing command workflow without claiming object tracking.
"""

import os
from copy import deepcopy
from pathlib import Path
import shlex
from typing import Any


MAX_EFFECTS = 1
MAX_PATH_CHARS = 4096
MAX_CONTENT_BYTES = 64 * 1024
MAX_COMMAND_CHARS = MAX_CONTENT_BYTES * 5 + MAX_PATH_CHARS * 5
MAX_ENTITY_ID_CHARS = 128

EFFECTS_SCHEMA = {
    "type": "array",
    "maxItems": MAX_EFFECTS,
    "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["create", "rename", "append"]},
            # Keep the generation schema structural; parse_effects enforces
            # string and UTF-8 byte bounds before review or execution.
            "path": {"type": "string", "minLength": 1},
            "destination": {"type": ["string", "null"]},
            "content": {"type": ["string", "null"]},
            "entity_id": {"type": ["string", "null"]},
            "entity_version": {"type": ["integer", "null"], "minimum": 1},
        },
        "required": ["kind", "path", "destination", "content", "entity_id", "entity_version"],
        "additionalProperties": False,
    },
}

PROPOSAL_EFFECTS_SCHEMA = deepcopy(EFFECTS_SCHEMA)
PROPOSAL_EFFECTS_SCHEMA["items"]["properties"]["content"] = {
    "type": ["object", "null"],
    "properties": {
        "lines": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "ends_with_newline": {"type": "boolean"},
    },
    "required": ["lines", "ends_with_newline"],
    "additionalProperties": False,
}

TRACKING_GENERATION_INSTRUCTIONS = r"""
Object tracking is enabled for this request. Return Explanation and Effects only.
Effects is a proposal, not a claim that an operation has completed.
For a single supported file operation, return exactly one effect:
  {"kind":"create|rename|append", "path":"exact file path",
   "destination":null, "content":null, "entity_id":null, "entity_version":null}
create/append: content is {"lines":["first line","second line"],
"ends_with_newline":true|false}. Each line holds literal text WITHOUT newline
characters. The application joins lines with a newline and adds one final newline
only when ends_with_newline is true. Preserve blank lines as empty line strings.
For an empty file use lines:[""] and ends_with_newline:false.
destination must be null. create must have null entity_id and entity_version.
rename: destination is the exact new file path; content must be null.
Use the supplied current absolute path for a referenced object. Copy its entity_id
and version into entity_id/entity_version for rename/append, or use null for both
when no supplied tracked object is referenced. Do not invent object IDs or versions.
The application will render a literal Bash command from the validated effect,
then independently review it and request confirmation. Never generate commands
in this response. A requested newline is represented by the line structure or
ends_with_newline, never by putting a backslash followed by n inside a line.
Only use literal backslash-n text when the user explicitly requests those characters.
Do not add directory preparation or other operations to a single file effect.
For a file in the current directory, prefer its explicitly requested relative
filename over copying the full current-directory string from a plan.
The original request and clarification answers take precedence over the plan.
For other operations or a batch of operations, return Effects: []. The application
will call its original command generator; these operations have no tracked effect.
""".strip()


class EffectValidationError(ValueError):
    """The proposal is malformed or cannot be bound to its commands."""


def _text(value: Any, name: str, *, limit: int, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise EffectValidationError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}string")
    if len(value) > limit or "\x00" in value:
        raise EffectValidationError(f"{name} exceeds its limit or contains NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EffectValidationError(f"{name} must be valid UTF-8 text") from exc
    return value


def normalize_path(value: Any, cwd: str) -> str:
    """Keep actual absolute paths without following symlinks or shell expansion."""
    value = _text(value, "path", limit=MAX_PATH_CHARS)
    if any(ord(character) < 32 for character in value):
        raise EffectValidationError("path must not contain control characters")
    if not isinstance(cwd, str) or not cwd or not os.path.isabs(cwd):
        raise EffectValidationError("cwd must be an absolute path")
    return os.path.normpath(os.path.abspath(os.path.join(cwd, value)))


def parse_effects(data: dict, cwd: str) -> list[dict]:
    """Parse the enabled generation response; missing Effects is a format error."""
    if not isinstance(data, dict) or "Effects" not in data:
        raise EffectValidationError("object tracking response requires Effects")
    raw_effects = data["Effects"]
    if not isinstance(raw_effects, list) or len(raw_effects) > MAX_EFFECTS:
        raise EffectValidationError("Effects must be a list with at most one effect")
    normalized = []
    fields = {"kind", "path", "destination", "content", "entity_id", "entity_version"}
    for raw in raw_effects:
        if not isinstance(raw, dict) or set(raw) - fields:
            raise EffectValidationError("effect must be an object containing only effect fields")
        kind = raw.get("kind")
        if kind not in ("create", "rename", "append"):
            raise EffectValidationError("effect kind must be create, rename, or append")
        effect = {"kind": kind, "path": normalize_path(raw.get("path"), cwd),
                  "destination": None, "content": None,
                  "entity_id": raw.get("entity_id"), "entity_version": raw.get("entity_version")}
        if kind == "rename":
            if raw.get("content") is not None:
                raise EffectValidationError("rename cannot contain content")
            effect["destination"] = normalize_path(raw.get("destination"), cwd)
            if effect["path"] == effect["destination"]:
                raise EffectValidationError("rename source and destination must differ")
        else:
            if raw.get("destination") is not None:
                raise EffectValidationError(f"{kind} cannot contain destination")
            content = _text(raw.get("content"), "content", limit=MAX_CONTENT_BYTES, empty=True)
            if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
                raise EffectValidationError("content exceeds its UTF-8 byte limit")
            if kind == "append" and not content:
                raise EffectValidationError("append content must not be empty")
            effect["content"] = content
        entity_id, version = effect["entity_id"], effect["entity_version"]
        if (entity_id is None) != (version is None):
            raise EffectValidationError("entity_id and entity_version must be provided together")
        if entity_id is not None:
            _text(entity_id, "entity_id", limit=MAX_ENTITY_ID_CHARS)
            if type(version) is not int or version < 1:
                raise EffectValidationError("entity_version must be a positive integer")
            if kind == "create":
                raise EffectValidationError("create must propose a new object without an entity reference")
        normalized.append(effect)
    return normalized


def _redirects(command: str) -> list[tuple[int, int, str]]:
    """Validate literal shell words and locate real, unquoted redirections.

    shlex alone discards quote context: a quoted '>' must not be mistaken for
    an operator, and an unquoted '$VAR' must not be mistaken for a literal path.
    This scan keeps that context before shlex decodes the word values.
    """
    redirects = []
    quote = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote == "'":
            if character == "'":
                quote = None
        elif quote == '"':
            if character == '"':
                quote = None
            elif character in "$`":
                raise EffectValidationError("shell expansions are not supported in tracked commands")
            elif character == "\\":
                index += 1
                if index >= len(command) or command[index] not in ('"', "\\"):
                    raise EffectValidationError("unsupported double-quoted shell escape")
        elif character in ("'", '"'):
            quote = character
        elif character == "\\":
            index += 1
            if index >= len(command) or command[index] in "\r\n":
                raise EffectValidationError("unsupported shell escape")
        elif character == ">":
            # A directly adjacent numeric word becomes an IO_NUMBER in Bash,
            # not a printf data argument. Require a separated operator rather
            # than silently interpreting ``printf '%s' 1>file`` as content '1'.
            if index == 0 or command[index - 1] not in " \t":
                raise EffectValidationError("output redirection must be separated by whitespace")
            end = index + 1
            if end < len(command) and command[end] == ">":
                end += 1
            redirects.append((index, end, command[index:end]))
            index = end - 1
        elif character in "$`|&;()<>*?[]{}~#!\r\n":
            raise EffectValidationError("unsupported shell syntax in tracked command")
        elif ord(character) < 32 and character != "\t":
            raise EffectValidationError("unsupported control character in tracked command")
        index += 1
    if quote is not None:
        raise EffectValidationError("unterminated shell quote")
    return redirects


def _words(fragment: str) -> list[str]:
    try:
        return shlex.split(fragment, comments=False, posix=True)
    except ValueError as exc:
        raise EffectValidationError("invalid literal shell command") from exc


def _command_path(value: str, cwd: str) -> str:
    # normpath is lexical. Following ``link/../file`` in Bash can reach a
    # different file than checking the normalized path, if link is a symlink.
    # Use canonical paths instead of accepting that ambiguous command spelling.
    components = value.replace("\\", "/").split("/") if os.name == "nt" else value.split("/")
    if ".." in components:
        raise EffectValidationError("tracked command paths cannot contain parent traversal")
    if os.name == "nt" and "\\" in value:
        raise EffectValidationError("tracked Bash paths on Windows require forward slashes")
    return normalize_path(value, cwd)


def _command_effect(command: str, cwd: str) -> dict:
    _text(command, "command", limit=MAX_COMMAND_CHARS)
    redirects = _redirects(command)
    if redirects:
        if len(redirects) != 1:
            raise EffectValidationError("tracked writes require exactly one output redirection")
        start, end, operator = redirects[0]
        prefix, suffix = _words(command[:start]), _words(command[end:])
        if len(suffix) != 1:
            raise EffectValidationError("tracked writes require exactly one literal target path")
        if len(prefix) == 3 and prefix[0] == "printf" and prefix[1] in ("%s", "%s\\n"):
            content = prefix[2] + ("\n" if prefix[1] == "%s\\n" else "")
        elif len(prefix) == 2 and prefix[0] == "echo" and not prefix[1].startswith("-") and "\\" not in prefix[1]:
            # echo flags and backslash handling vary by Bash configuration.
            content = prefix[1] + "\n"
        else:
            raise EffectValidationError("tracked writes require literal printf or a simple literal echo")
        return {"kind": "append" if operator == ">>" else "create",
                "path": _command_path(suffix[0], cwd), "destination": None, "content": content}
    words = _words(command)
    if words[:2] == ["mv", "--"] and len(words) == 4:
        source, destination = words[2:]
    elif len(words) == 3 and words[0] == "mv" and not any(word.startswith("-") for word in words[1:]):
        source, destination = words[1:]
    else:
        raise EffectValidationError("tracked rename requires mv [--] SOURCE DESTINATION without options")
    return {"kind": "rename", "path": _command_path(source, cwd),
            "destination": _command_path(destination, cwd), "content": None}


def bind_effects(commands: list[str], effects: list[dict], cwd: str) -> list[dict]:
    """Require the exact command sequence to produce the proposed literal effect.

    Empty effects intentionally impose no restrictions on the legacy command
    grammar.  A nonempty proposal cannot accompany any additional commands.
    """
    effects = parse_effects({"Effects": effects}, cwd)
    if not effects:
        return []
    if not isinstance(commands, list) or len(commands) != 1:
        raise EffectValidationError("one tracked effect requires exactly one supported command")
    actual = _command_effect(commands[0], cwd)
    expected = effects[0]
    for name in ("kind", "path", "destination", "content"):
        if actual[name] != expected[name]:
            raise EffectValidationError(f"command does not match proposed effect {name}")
    return effects


def render_effects(effects: list[dict]) -> list[str]:
    """Render already normalized effects for generation examples or callers.

    Any replacement of reviewed commands must invalidate review and approval in
    the workflow.  The renderer is not an authorization or execution interface.
    """
    effects = parse_effects({"Effects": effects}, os.getcwd())
    commands = []
    for effect in effects:
        if effect["kind"] == "rename":
            source = Path(effect["path"]).as_posix()
            destination = Path(effect["destination"]).as_posix()
            commands.append(f"mv -- {shlex.quote(source)} {shlex.quote(destination)}")
        else:
            operator = ">>" if effect["kind"] == "append" else ">"
            path = Path(effect["path"]).as_posix()
            commands.append(f"printf '%s' {shlex.quote(effect['content'])} {operator} {shlex.quote(path)}")
    return commands
