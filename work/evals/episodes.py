"""Cross-invocation offline episodes using the existing persistent backends.

One isolated harness worker owns the episode. Each step calls ``process`` afresh,
reopens its SQLite backends, and starts a distinct journal session. Only fixture
file state, verified objects, and scoped journal-derived history are carried over.
Scripted outputs validate runtime behavior; they do not measure model quality.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .harness import ScriptedServices, approval_evidence, fixture_path, snapshot, write_json
from .faults import validate_faults


def replace_root(value, root):
    if isinstance(value, str):
        return value.replace("${ROOT}", Path(root).as_posix())
    if isinstance(value, list):
        return [replace_root(item, root) for item in value]
    if isinstance(value, dict):
        return {key: replace_root(item, root) for key, item in value.items()}
    return value


def semantic_effects(effects, root):
    """Normalize human allowlisted semantics without inventing entity identities."""
    from n2c import _render_tracked_commands
    from object_tracking.effects import parse_effects
    if any(isinstance(effect.get("content"), dict) for effect in effects if isinstance(effect, dict)):
        rendered = _render_tracked_commands(json.dumps({"Explanation": "Fixture semantic scope", "Effects": effects}), str(root))
        normalized = json.loads(rendered)["Effects"] if rendered is not None else []
    else:
        normalized = parse_effects({"Effects": effects}, str(root))
    result = []
    for effect in normalized:
        semantic = {name: effect[name] for name in ("kind", "path", "destination", "content")}
        for name in ("path", "destination"):
            if semantic[name] is None:
                continue
            path = Path(semantic[name]).resolve()
            try:
                relative = path.relative_to(Path(root).resolve()).as_posix()
            except ValueError as exc:
                raise ValueError("Tracked effect escapes the fixture root") from exc
            fixture_path(root, relative)
            semantic[name] = os.path.normcase(str(path))
        result.append(semantic)
    return result


def allowed_effects(effects, interaction, root):
    if not effects:
        return True
    actual = semantic_effects(effects, root)
    return any(actual == semantic_effects(batch, root)
               for batch in interaction.get("allowed_effect_batches", []))


class EpisodeServices(ScriptedServices):
    def __init__(self, public_input, root, variant, *, object_tracking=False):
        super().__init__(replace_root(deepcopy(public_input), root), root, variant)
        self.object_tracking = object_tracking
        self.service_inputs = []

    def record_input(self, role, value):
        self.service_inputs.append({"role": role, "input": deepcopy(value)})

    def clarify(self, request, history):
        self.record_input("clarify", request)
        return super().clarify(request, history)

    def parse_intent(self, request):
        self.record_input("parse_intent", request)
        return super().parse_intent(request)

    def generate_commands(self, request):
        self.record_input("generate_commands", request)
        value = super().generate_commands(request)
        if not self.object_tracking:
            return value
        from model_output import parse_json_object
        from n2c import _render_tracked_commands
        data = parse_json_object(value)
        # Normalize the same structured proposal shape as the production N2C
        # boundary before review; never rewrite commands after approval.
        if set(data) == {"Explanation", "Effects"}:
            rendered = _render_tracked_commands(value, str(self.root))
            if rendered is None:
                raise ValueError("An empty tracked proposal needs explicit untracked commands")
            data = parse_json_object(rendered)
            value = rendered
        if data.get("Effects"):
            semantic_effects(data["Effects"], self.root)  # Check scope before filesystem inspection.
        return value

    def review_commands(self, request):
        self.record_input("review_commands", request)
        return super().review_commands(request)

    def correct_error(self, request, commands, error):
        self.record_input("correct_error", {"request": request, "commands": commands, "error": error})
        return super().correct_error(request, commands, error)

    def answer(self, prompt):
        answer = super().answer(prompt)
        if (prompt == "Proceed? (y/n): " and answer == "y" and self.object_tracking
                and not allowed_effects((self.last_review or {}).get("expected_effects", []),
                                        self.input["interaction"], self.root)):
            self.approved_id = ""
            return "n"
        return answer


def episode_approval_evidence(events, root, interaction, *, object_tracking=False):
    """Independent replay of the command and semantic effect approval protocol."""
    if not approval_evidence(events, root):
        return False
    pending = None
    approved = False
    for event in events:
        data = event.get("data", {})
        if event.get("kind") == "awaiting_input" and data.get("kind") == "confirmation":
            pending, approved = data, False
        elif event.get("kind") == "user_input" and data.get("source") == "confirmation":
            approved = data.get("text", "").strip().lower() in {"y", "yes"}
        elif event.get("kind") == "operation_started" and data.get("name") == "commands.execute":
            if (interaction["approval_policy"] != "fixture_allowlist" or not pending or not approved
                    or pending.get("commands") not in interaction["allowed_command_batches"]):
                return False
            if object_tracking and not allowed_effects(pending.get("expected_effects", []), interaction, root):
                return False
            approved = False
    return True


def execute_episode(public_input, root, artifact_dir, variant="faithful", runtime=None, output=None, calls=None):
    """Return ``(states, journal_events, stats)`` inside an existing isolated worker.

    The top-level request is step 1; optional ``steps`` are later invocations.
    The caller prepares fixtures, captures physical model telemetry, disables
    network, and owns trace context and scoring. Expected outputs never enter here.
    """
    from agent_memory import MemoryScope, MemorySession, create_backend
    from interaction_log import InteractionJournal
    from object_tracking.store import ObjectStore
    from samantha import process
    from workflow import WorkflowServices

    runtime = dict(runtime or {"memory": "none", "object_tracking": False, "azure_fallback": False})
    if runtime.get("memory") not in {"none", "recent", "bm25"}:
        raise ValueError("Episode memory must use a built-in offline backend")
    if type(runtime.get("object_tracking")) is not bool or runtime.get("azure_fallback") is not False:
        raise ValueError("Episode runtime needs explicit tracking and disabled cloud fallback")
    root = Path(root).resolve(strict=True)
    artifact_dir = Path(artifact_dir).resolve(strict=True)
    if artifact_dir == root or artifact_dir.is_relative_to(root):
        raise ValueError("Episode controller artifacts must stay outside the fixture root")
    steps = [{name: deepcopy(public_input[name]) for name in ("request", "interaction", "fake_services", "faults")
              if name in public_input}, *deepcopy(public_input.get("steps", []))]
    if not 1 <= len(steps) <= 50:
        raise ValueError("Episode must contain 1 to 50 invocations")
    scope = MemoryScope("evaluation-fixture", "episode:" + str(root))
    output = output if output is not None else []
    states, events, step_evidence = [], [], []
    role_calls = Counter()
    tool_calls = scope_rejections = 0
    partial_effects = False
    previous = Path.cwd()
    os.chdir(root)
    try:
        for index, raw_step in enumerate(steps, 1):
            if (not isinstance(raw_step, dict) or set(raw_step) - {"request", "interaction", "fake_services", "faults"}
                    or not isinstance(raw_step.get("request"), str) or not raw_step["request"].strip()
                    or not isinstance(raw_step.get("interaction"), dict)):
                raise ValueError("Episode step needs public request, interaction and optional fake services")
            step = replace_root(raw_step, root)
            faults = validate_faults(step.get("faults", []))
            if any(fault["kind"] == "memory_injection" and fault["call_index"] != 0 for fault in faults):
                raise ValueError("Memory injection call_index must be 0 before the invocation's first retrieval")
            folder = artifact_dir / "steps" / f"step_{index:02d}"
            folder.mkdir(parents=True, exist_ok=False)
            before = snapshot(root)
            write_json(folder / "before.json", before)
            fake = EpisodeServices(step, root, variant, object_tracking=runtime["object_tracking"])
            services = WorkflowServices(fake.clarify, fake.parse_intent, fake.generate_commands,
                                        fake.review_commands, fake.correct_error, fake.execute_commands)
            memory = create_backend(runtime["memory"], path=artifact_dir / "memory.sqlite3") \
                if runtime["memory"] != "none" else None
            objects = None
            step_output = []
            injections = []
            try:
                objects = ObjectStore(artifact_dir / "objects.sqlite3") if runtime["object_tracking"] else None
                for fault_index, fault in enumerate(faults):
                    if fault["kind"] != "memory_injection":
                        continue
                    source_id = f"untrusted-fault-memory-step-{index:02d}-{fault_index:02d}"
                    injections.append({"source_session_id": source_id, "applied": memory is not None,
                                       "source_type": "untrusted_fault_fixture"})
                    if memory is not None:
                        # A tool message is untrusted historical evidence. It is
                        # never a system instruction, approval, or a new policy.
                        memory.ingest(MemorySession(source_id, scope, datetime.now(timezone.utc).isoformat(),
                            ({"role": "tool", "content": fault["payload"]},),
                            {"source_session_id": source_id, "source_type": "untrusted_fault_fixture",
                             "fixture_fault": "memory_injection", "step_index": index}))
                state = process(step["request"], services=services, input_fn=fake.answer,
                    output_fn=step_output.append, state_file=folder / "state.json",
                    log_file=artifact_dir / "journal.sqlite3", memory=memory, memory_scope=scope,
                    object_store=objects, object_scope=scope)
                # Save persisted factual lineage before closing this invocation's
                # connection. The next invocation must reopen it independently.
                object_events = objects.events(scope) if objects is not None else []
            finally:
                if objects is not None:
                    objects.close()
                if memory is not None:
                    memory.close()
            with InteractionJournal(artifact_dir / "journal.sqlite3", readonly=True) as journal:
                session = journal.read_session(state["log_session_id"])
            step_events = session["events"]
            after = snapshot(root)
            evidence = {"schema_version": "1.0", "complete": True, "step_index": index,
                "request": step["request"], "runtime_status": state["status"],
                "before_snapshot": before, "after_snapshot": after, "trajectory": step_events,
                "session_ids": [state["log_session_id"]], "output": "\n".join(step_output),
                "safety": {"no_unapproved_execution": episode_approval_evidence(step_events, root,
                    step["interaction"], object_tracking=runtime["object_tracking"]),
                    "no_changes_outside_fixture_root": fake.scope_rejections == 0},
                "memory_sources": state.get("memory_sources", []),
                "memory_context": state.get("memory_context", ""),
                "object_context": state.get("object_context", ""),
                "tracked_entities": state.get("tracked_entities", []),
                "object_verifications": state.get("object_verifications", []),
                "object_events": object_events, "service_inputs": fake.service_inputs,
                "faults": faults, "memory_injections": injections,
                "partial_effects": getattr(fake, "partial_effects", False),
                "errors": ([{"stage": "runtime", "category": "Unknown", "message": state["error"]}]
                           if state.get("error") else [])}
            write_json(folder / "after.json", after)
            write_json(folder / "evidence.json", evidence)
            states.append(state)
            events.extend(step_events)
            step_evidence.append(evidence)
            output.extend(step_output)
            role_calls.update(fake.role_calls)
            tool_calls += fake.tool_calls
            scope_rejections += fake.scope_rejections
            partial_effects |= getattr(fake, "partial_effects", False)
    finally:
        os.chdir(previous)
    return states, events, {"session_ids": [state["log_session_id"] for state in states],
                           "step_evidence": step_evidence, "fake_role_calls": dict(role_calls),
                           "tool_calls": tool_calls, "scope_rejections": scope_rejections,
                           "runtime_status": states[-1]["status"],
                           "runtime": runtime, "invocations": len(states),
                           "partial_effects": partial_effects,
                           "observed_safe_stop": variant == "observed_safe_stop" and partial_effects}
