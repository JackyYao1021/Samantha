"""Dataset loading and canonical freezing, with no model or runtime imports."""

import hashlib
import json
from pathlib import Path

from .schemas import SCHEMA_VERSION, validate_case


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))


def validate_dataset(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a nonempty list")
    result = [validate_case(case) for case in cases]
    identifiers = [case["case_id"] for case in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Dataset contains duplicate case_id values")
    return result


def load_dataset(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        cases = [read_json(line) for line in text.splitlines() if line.strip()]
    else:
        cases = read_json(text)
        if isinstance(cases, dict):
            if cases.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported dataset schema version")
            cases = cases.get("cases")
    return validate_dataset(cases)


def dataset_bytes(cases):
    return ("\n".join(canonical_json(case) for case in validate_dataset(cases)) + "\n").encode("utf-8")


def dataset_hash(cases):
    return hashlib.sha256(dataset_bytes(cases)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_jsonl(cases, path):
    """Exclusively create a canonical snapshot; an old run is never overwritten."""
    content = dataset_bytes(cases)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(content)
    return hashlib.sha256(content).hexdigest()
