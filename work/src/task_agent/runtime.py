"""Durable, bounded tool-selection loop shared by CLI, HTTP and task evaluations.

SQLite is the authoritative application checkpoint. An executing action is reconciled
against its saved pre/postconditions after a crash; an uncertain effect is never replayed.
The legacy Bash LangGraph workflow remains a separate compatibility entry point.
"""

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import uuid

from model_telemetry import capture_model_calls
from .planner import ModelPlanner
from .store import TaskStore, now, process_lock
from .tools import ToolRegistry


TERMINAL = {"succeeded", "failed", "cancelled"}


def identity(workspace, prepared):
    return hashlib.sha256(json.dumps({"workspace": workspace, "prepared": prepared},
                                    sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class AgentRuntime:
    def __init__(self, db_path, planner=None, *, registry_factory=ToolRegistry, fault_hook=None):
        self.store = TaskStore(db_path)
        self.planner = planner if planner is not None else ModelPlanner()
        self.registry_factory = registry_factory
        self.fault_hook = fault_hook

    def create_task(self, request, workspace, max_steps=12, parent_id=None):
        if not isinstance(request, str) or not request.strip() or len(request) > 12000:
            raise ValueError("Request must contain 1–12000 characters.")
        if type(max_steps) is not int or not 1 <= max_steps <= 32:
            raise ValueError("max_steps must be between 1 and 32.")
        root = Path(workspace).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Workspace must be an existing directory.")
        prior = []
        if parent_id:
            parent = self.store.read(parent_id, events=False)
            if parent["workspace"] != str(root):
                raise ValueError("Follow-up tasks must use the same workspace.")
            if parent["status"] not in TERMINAL:
                raise ValueError("Finish or cancel the parent task before following up.")
            prior = [{"tool": a["tool"], "args": a["args"], "result": a.get("result"),
                      "verification": a.get("verification")}
                     for a in parent["actions"] if a["status"] == "verified"][-8:]
        state = {"version": 1, "id": uuid.uuid4().hex, "request": request.strip(),
                 "workspace": str(root), "parent_id": parent_id, "prior_evidence": prior,
                 "status": "ready", "steps": 0, "max_steps": max_steps,
                 "model_calls": 0, "model_call_records": [], "active_seconds": 0,
                 "max_active_seconds": 600, "answer": "", "error": "", "question": "",
                 "clarifications": [], "observations": [], "actions": [], "pending_action": None,
                 "created_at": now(), "updated_at": now()}
        self.store.save(state, "task_created", {"request": state["request"], "workspace": str(root)}, create=True)
        return self.get_task(state["id"])

    def get_task(self, task_id):
        state = self.store.read(task_id)
        state["cancel_requested"] = self.store.cancelled(task_id)
        return state

    def list_tasks(self):
        return self.store.list()

    def _save(self, state, kind, payload=None):
        self.store.save(state, kind, payload)

    @contextmanager
    def _account(self, state, phase, payload=None):
        """Charge each non-overlapping unit of active work and its physical requests.

        Human approval/input wait time is excluded. A killed process cannot report
        the duration or telemetry of its unfinished unit; persisted decision counts
        still prevent a restart from resetting the decision budget.
        """
        started = time.monotonic()
        calls = []
        try:
            with capture_model_calls() as calls:
                yield
        finally:
            elapsed = time.monotonic() - started
            physical = sum(call.get("physical_request", True) is True for call in calls)
            state["active_seconds"] += elapsed
            state["model_calls"] += physical
            state["model_call_records"].extend(calls)
            self._save(state, phase, {**(payload or {}), "active_seconds": elapsed,
                                     "physical_requests": physical})

    @staticmethod
    def _time_exhausted(state):
        return state["active_seconds"] >= state["max_active_seconds"]

    def _fail(self, state, message, kind="task_failed"):
        state.update(status="failed", error=message)
        self._save(state, kind, {"error": message})

    def approve(self, task_id, action_id, approved=True):
        if type(approved) is not bool:
            raise ValueError("approved must be boolean.")
        with process_lock(self.store.lock_path):
            state = self.store.read(task_id, events=False)
            action = state["pending_action"]
            if self.store.cancelled(task_id):
                raise ValueError("Task cancellation was requested; approval refused.")
            if (state["status"] != "awaiting_approval" or not action or action["id"] != action_id
                    or action["status"] != "proposed"):
                raise ValueError("This approval is stale or the task is not awaiting approval.")
            if action["identity"] != identity(state["workspace"], action["prepared"]):
                raise ValueError("Action identity changed; approval refused.")
            if not approved:
                action["status"] = "rejected"
                self._sync_action(state, action)
                state.update(status="cancelled", pending_action=None)
                self._save(state, "approval_rejected", {"action_id": action_id})
            else:
                action.update(status="approved", approved_identity=action["identity"], approved_at=now())
                self._sync_action(state, action)
                state["status"] = "ready"
                self._save(state, "approval_granted", {"action_id": action_id, "identity": action["identity"]})
        return self.get_task(task_id)

    def answer(self, task_id, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError("Clarification must contain 1–12000 characters.")
        with process_lock(self.store.lock_path):
            state = self.store.read(task_id, events=False)
            if state["status"] != "awaiting_input":
                raise ValueError("Task is not awaiting clarification.")
            state["clarifications"].append({"question": state["question"], "answer": text.strip()})
            state.update(question="", status="ready")
            self._save(state, "clarification_answered", state["clarifications"][-1])
        return self.get_task(task_id)

    def cancel(self, task_id):
        current = self.store.read(task_id, events=False)
        if current["status"] in TERMINAL:
            return self.get_task(task_id)
        self.store.request_cancel(task_id)
        try:
            with process_lock(self.store.lock_path):
                return self._cancel_locked(task_id)
        except RuntimeError:
            # The worker observes this independent durable flag at its next safe boundary.
            return self.get_task(task_id)

    def _cancel_locked(self, task_id):
        state = self.store.read(task_id, events=False)
        if state["status"] not in TERMINAL:
            # Executing can mean a crashed process. Reconcile before discarding its evidence.
            if state["pending_action"] and state["pending_action"]["status"] == "executing":
                self._recover(state, self.registry_factory(state["workspace"]))
                if state["status"] == "failed":
                    return self.get_task(task_id)
            if state["pending_action"]:
                state["pending_action"]["status"] = "cancelled"
                self._sync_action(state, state["pending_action"])
            state.update(status="cancelled", pending_action=None)
            self._save(state, "task_cancelled", {"note": "Previously verified file changes remain."})
        return self.get_task(task_id)

    @staticmethod
    def _sync_action(state, action):
        for index, previous in enumerate(state["actions"]):
            if previous["id"] == action["id"]:
                state["actions"][index] = copy.deepcopy(action)
                return
        state["actions"].append(copy.deepcopy(action))

    def _observe(self, state, tool, args, result, ok=True):
        state["observations"].append({"tool": tool, "args": args, "ok": ok, "result": result})
        state["error"] = "" if ok else str(result.get("error", "Tool failed"))

    def _verified(self, state, action, registry, *, recovered=False):
        with self._account(state, "verification_finished", {"action_id": action["id"]}):
            verification = registry.verify(action["prepared"])
        action["verification"] = verification
        if verification.get("ok") is not True:
            action["status"] = "conflict"
            self._sync_action(state, action)
            self._fail(state, "File outcome verification failed; action will not be replayed.", "verification_failed")
            return False
        action["status"] = "verified"
        action["verified_at"] = now()
        if recovered:
            action["result"] = {"recovered": True, "verification": verification}
        self._sync_action(state, action)
        self._observe(state, action["tool"], action["args"], action.get("result") or verification)
        state.update(pending_action=None, status="running")
        self._save(state, "action_recovered" if recovered else "action_verified",
                   {"action_id": action["id"], "tool": action["tool"], "verification": verification})
        return True

    def _recover(self, state, registry):
        action = state["pending_action"]
        if not action or action["status"] != "executing":
            return
        if action.get("approved_identity") != identity(state["workspace"], action["prepared"]):
            self._fail(state, "Saved approval does not match the interrupted action.")
            return
        with self._account(state, "reconciliation_finished", {"action_id": action["id"]}):
            situation = registry.reconcile(action["prepared"])
        if situation == "applied":
            self._verified(state, action, registry, recovered=True)
        elif situation == "not_applied":
            # A fresh confirmation makes retry explicit when execution outcome was interrupted.
            action.update(status="proposed", approved_identity=None)
            self._sync_action(state, action)
            state["status"] = "awaiting_approval"
            self._save(state, "recovery_needs_approval", {"action_id": action["id"]})
        else:
            action["status"] = "conflict"
            self._sync_action(state, action)
            self._fail(state, "Interrupted action has conflicting file state; inspect it before starting a new task.",
                       "recovery_conflict")

    def _execute_pending(self, state, registry):
        action = state["pending_action"]
        if self._time_exhausted(state):
            self._fail(state, "Task active-time budget exhausted before file execution.", "budget_exhausted")
            return
        if (action["status"] != "approved" or
                action.get("approved_identity") != identity(state["workspace"], action["prepared"])):
            self._fail(state, "Execution requires approval for this exact action.")
            return
        # This commit precedes every side effect. Crash tests interrupt on both sides of execution.
        action["status"] = "executing"
        self._sync_action(state, action)
        state["status"] = "running"
        self._save(state, "action_started", {"action_id": action["id"], "tool": action["tool"]})
        if self.fault_hook:
            self.fault_hook("before_execute", state)
        try:
            with self._account(state, "tool_execution_finished", {"action_id": action["id"]}):
                action["result"] = registry.execute(action["prepared"])
        except Exception as exc:
            with self._account(state, "reconciliation_finished", {"action_id": action["id"]}):
                situation = registry.reconcile(action["prepared"])
            if situation == "applied":
                self._verified(state, action, registry, recovered=True)
            elif situation == "not_applied":
                action.update(status="failed", error=str(exc))
                self._sync_action(state, action)
                self._observe(state, action["tool"], action["args"], {"error": str(exc)}, ok=False)
                state["pending_action"] = None
                self._save(state, "action_failed", {"action_id": action["id"], "error": str(exc)})
            else:
                action.update(status="conflict", error=str(exc))
                self._sync_action(state, action)
                self._fail(state, "File state changed or execution was partial: " + str(exc), "execution_conflict")
            return
        if self.fault_hook:
            self.fault_hook("after_execute", state)
        self._verified(state, action, registry)

    @staticmethod
    def _context(state):
        # Clip individual observations, never the current request or clarification answers.
        def clipped(value, limit=9000):
            encoded = json.dumps(value, ensure_ascii=False)
            return value if len(encoded) <= limit else {"truncated": True, "excerpt": encoded[:limit]}
        context = {"request": state["request"], "clarifications": state["clarifications"],
                "remaining_steps": state["max_steps"] - state["steps"],
                "remaining_seconds": max(0, state["max_active_seconds"] - state["active_seconds"]),
                "previous_task_verified_evidence": [clipped(v, 2500) for v in state["prior_evidence"]],
                "observations": [clipped(o, 5000) for o in state["observations"][-10:]],
                "verified_actions": [{"tool": a["tool"], "args": clipped(a["args"], 1200)}
                                     for a in state["actions"] if a["status"] == "verified"]}
        # Trim old retrieved observations before any explicit user constraints.
        # This is a character bound (not a tokenizer claim). The full evidence remains on disk.
        context["omitted_observations"] = max(0, len(state["observations"]) - len(context["observations"]))
        context["omitted_prior_evidence"] = 0
        for field, counter in (("observations", "omitted_observations"),
                               ("previous_task_verified_evidence", "omitted_prior_evidence")):
            while len(json.dumps(context, ensure_ascii=False)) > 24000 and context[field]:
                context[field].pop(0)
                context[counter] += 1
        if len(json.dumps(context, ensure_ascii=False)) > 24000:
            raise ValueError("Explicit task constraints and verified actions exceed the context budget; "
                             "start a smaller follow-up task. No user constraints were silently truncated.")
        return context

    @staticmethod
    def _validate_decision(decision):
        if not isinstance(decision, dict):
            raise ValueError("Planner must return an object.")
        kind = decision.get("type")
        allowed = {"tool": {"type", "tool", "args", "reason"},
                   "ask": {"type", "question"}, "finish": {"type", "answer"}}
        if kind not in allowed or set(decision) - allowed[kind]:
            raise ValueError("Invalid planner decision schema.")
        if kind == "tool":
            if not isinstance(decision.get("tool"), str) or not isinstance(decision.get("args"), dict):
                raise ValueError("A tool decision needs a tool name and argument object.")
            if not isinstance(decision.get("reason", ""), str):
                raise ValueError("Tool reason must be text.")
        else:
            value = decision.get("question" if kind == "ask" else "answer")
            if not isinstance(value, str) or not value.strip() or len(value) > 16000:
                raise ValueError("Planner text must contain 1–16000 characters.")
        return kind

    def run(self, task_id):
        with process_lock(self.store.lock_path):
            state = self.store.read(task_id, events=False)
            if self.store.cancelled(task_id) and state["status"] not in TERMINAL:
                return self._cancel_locked(task_id)
            if state["status"] in TERMINAL | {"awaiting_input", "awaiting_approval"}:
                return self.get_task(task_id)
            try:
                registry = self.registry_factory(state["workspace"])
                self._recover(state, registry)
                while state["status"] not in TERMINAL | {"awaiting_input", "awaiting_approval"}:
                    if self.store.cancelled(task_id):
                        return self._cancel_locked(task_id)
                    if state["pending_action"]:
                        self._execute_pending(state, registry)
                        continue
                    if state["steps"] >= state["max_steps"] or self._time_exhausted(state):
                        self._fail(state, "Task decision/time budget exhausted.", "budget_exhausted")
                        break
                    state.update(status="running", steps=state["steps"] + 1)
                    self._save(state, "planning_started", {"step": state["steps"]})
                    with self._account(state, "planning_finished", {"step": state["steps"]}):
                        decision = self.planner(self._context(state), registry.schemas())
                        kind = self._validate_decision(decision)
                    if self.store.cancelled(task_id):
                        return self._cancel_locked(task_id)
                    if self._time_exhausted(state):
                        self._fail(state, "Task active-time budget exhausted during planning.", "budget_exhausted")
                        break
                    if kind == "finish":
                        state["answer"] = decision["answer"].strip()
                        if state["error"]:
                            self._fail(state, "Task ended with an unresolved tool error: " + state["error"])
                        else:
                            state["status"] = "succeeded"
                            self._save(state, "task_finished", {"answer": state["answer"],
                                       "note": "Model reports completion; task-level goals require independent evaluation."})
                        break
                    if kind == "ask":
                        if len(state["clarifications"]) >= 3:
                            self._fail(state, "Clarification limit reached.")
                        else:
                            state.update(status="awaiting_input", question=decision["question"].strip())
                            self._save(state, "clarification_requested", {"question": state["question"]})
                        break
                    name, args = decision["tool"], decision["args"]
                    try:
                        with self._account(state, "tool_preparation_finished", {"tool": name}):
                            prepared = registry.prepare(name, args)
                    except Exception as exc:
                        self._observe(state, name, args, {"error": str(exc)}, ok=False)
                        self._save(state, "tool_rejected", {"tool": name, "error": str(exc)})
                        continue
                    if self.store.cancelled(task_id):
                        return self._cancel_locked(task_id)
                    if self._time_exhausted(state):
                        self._fail(state, "Task active-time budget exhausted during tool preparation.", "budget_exhausted")
                        break
                    action = {"id": uuid.uuid4().hex, "tool": name, "args": prepared["args"],
                              "prepared": prepared, "preview": prepared["preview"],
                              "before": prepared.get("before"), "expected": prepared.get("expected"),
                              "reason": decision.get("reason", ""),
                              "identity": identity(state["workspace"], prepared), "status": "proposed",
                              "result": None, "verification": None, "created_at": now()}
                    if prepared["mutating"]:
                        self._sync_action(state, action)
                        state.update(pending_action=action, status="awaiting_approval")
                        self._save(state, "approval_requested", {"action_id": action["id"], "tool": name,
                                                                "preview": action["preview"]})
                        break
                    try:
                        with self._account(state, "tool_execution_finished", {"action_id": action["id"]}):
                            result = registry.execute(prepared)
                        ok = not isinstance(result, dict) or result.get("ok") is not False
                        action.update(status="observed" if ok else "failed", result=result)
                        self._observe(state, name, args, result, ok=ok)
                    except Exception as exc:
                        action.update(status="failed", error=str(exc))
                        self._observe(state, name, args, {"error": str(exc)}, ok=False)
                    self._sync_action(state, action)
                    self._save(state, "tool_observed", {"action_id": action["id"], "tool": name,
                                                       "status": action["status"], "result": action["result"]})
            except Exception as exc:
                self._fail(state, str(exc))
            # BaseException (process kill, simulated crash) deliberately leaves the durable checkpoint intact.
        return self.get_task(task_id)
