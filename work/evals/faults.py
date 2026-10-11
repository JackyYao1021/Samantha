"""Bounded fault descriptions for Fake Services; no commands or network calls."""

from copy import deepcopy

FAULT_VERSION = "bounded-faults-v1"
FAULT_KINDS = {"execution_failure", "malformed_executor", "timeout", "persistence_error", "tool_output_injection", "memory_injection"}


def validate_faults(faults):
    if not isinstance(faults, list):
        raise ValueError("Faults must be a list")
    validated, identities = [], set()
    for spec in faults:
        if not isinstance(spec, dict) or set(spec) - {"kind", "call_index", "after_actions", "recoverable", "message", "payload"}:
            raise ValueError("Fault spec contains unsupported fields")
        if spec.get("kind") not in FAULT_KINDS or type(spec.get("call_index")) is not int or spec["call_index"] < 0:
            raise ValueError("Fault kind and zero-based call_index are required")
        data = {"after_actions": 0, "recoverable": False, **deepcopy(spec)}
        if type(data["after_actions"]) is not int or data["after_actions"] < 0 or type(data["recoverable"]) is not bool:
            raise ValueError("Invalid fault action count or recoverability")
        for field in ("message", "payload"):
            if field in data and not isinstance(data[field], str):
                raise ValueError(field + " must be text")
        if data["after_actions"] and data["kind"] != "execution_failure":
            raise ValueError("Only execution_failure supports partial action application")
        if data["kind"] in {"tool_output_injection", "memory_injection"} and "payload" not in data:
            raise ValueError("Injection fixtures need an explicit saved payload")
        identity = data["kind"], data["call_index"]
        if identity in identities:
            raise ValueError("Duplicate fault hook")
        identities.add(identity)
        validated.append(data)
    return validated


def faults_for_call(faults, call_index, *, kind=None):
    if type(call_index) is not int or call_index < 0:
        raise ValueError("call_index must be a nonnegative integer")
    return [spec for spec in validate_faults(faults) if spec["call_index"] == call_index and (kind is None or spec["kind"] == kind)]


def fault_inventory():
    return {"version": FAULT_VERSION, "kinds": sorted(FAULT_KINDS), "mode": "offline_bounded_fake_services",
            "coverage": {"execution_failure": "failure before or after a declared number of fixture actions",
                         "malformed_executor": "saved executor output contract violation",
                         "timeout": "explicit timeout signal; no waiting required",
                         "persistence_error": "explicit local persistence hook failure",
                         "tool_output_injection": "untrusted output payload; does not authorize commands",
                         "memory_injection": "untrusted context payload; does not authorize commands"},
            "limitations": ["Fault fixtures exercise known service boundaries, not arbitrary tools or provider outage realism.",
                            "Injection resilience requires checking saved actions; merely supplying an attack is not a passing result."]}
