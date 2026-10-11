"""Freeze and reuse existing suites without importing their live task runners.

These helpers run in the evaluation controller. Only the prepared ``workspace``
is a task input; catalogs, source code, TASK.txt and verification results are
private controller artifacts. They do not implement an operating-system sandbox.
"""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types

from ..dataset import canonical_json, file_hash, read_json
from ..schemas import relative_path

ADAPTER_VERSION = "existing-suites-v1"
_WORK = Path(__file__).resolve().parents[2]
_SUITES = {"real_tasks", "memory_tasks"}


def _suite_paths(suite, source_root=None):
    if suite not in _SUITES:
        raise ValueError("Unknown existing suite")
    tests = Path(source_root) if source_root is not None else _WORK / "tests"
    folder = tests / suite
    return folder / "cases.json", folder / "task_suite.py", tests / "real_tasks" / "task_suite.py"


def _catalog(content, suite):
    data = read_json(content.decode("utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("Unsupported existing catalog")
    identifiers = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].isalnum() or case["id"] in identifiers:
            raise ValueError("Invalid or duplicated existing case ID")
        identifiers.add(case["id"])
        for item in case.get("fixtures", []):
            relative_path(item["path"])
        for path in case.get("allowed_changes", []):
            relative_path(path)
        if suite == "memory_tasks":
            for step in case.get("steps", []):
                for path in [*step.get("files", {}), *step.get("absent", [])]:
                    relative_path(path)
        else:
            for check in case.get("checks", []):
                if "path" in check:
                    relative_path(check["path"])
                for path in check.get("files", {}):
                    relative_path(path)
    return data


def suite_inventory(suite=None, *, source_root=None):
    """Read native catalogs, without importing Samantha or memory-task modules."""
    if suite is None:
        return {name: suite_inventory(name, source_root=source_root) for name in sorted(_SUITES)}
    catalog, source, helper = _suite_paths(suite, source_root)
    data = _catalog(catalog.read_bytes(), suite)
    inventory = {"adapter_version": ADAPTER_VERSION, "suite": suite,
                 "case_count": len(data["cases"]), "case_ids": [case["id"] for case in data["cases"]],
                 "catalog_sha256": file_hash(catalog), "suite_sha256": file_hash(source),
                 "capabilities": ["prepare_fixtures", "verify" if suite == "real_tasks" else "verify_step"],
                 "live_execution": False}
    if suite == "memory_tasks":
        inventory["helper_sha256"] = file_hash(helper)
        inventory["step_count"] = sum(len(case["steps"]) for case in data["cases"])
    return inventory


def freeze_suite(suite, destination, *, source_root=None):
    """Freeze catalog and all reused scorer source into a fresh private folder."""
    catalog, source, helper = _suite_paths(suite, source_root)
    catalog_bytes, source_bytes = catalog.read_bytes(), source.read_bytes()
    data = _catalog(catalog_bytes, suite)
    files = {"cases.json": catalog_bytes, "task_suite.py": source_bytes}
    if suite == "memory_tasks":
        files["real_task_suite.py"] = helper.read_bytes()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, content in files.items():
        with (destination / name).open("xb") as target:
            target.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": "1.0", "adapter_version": ADAPTER_VERSION, "suite": suite,
                "case_count": len(data["cases"]), "case_ids": [case["id"] for case in data["cases"]],
                "files": hashes, "live_execution": False,
                "oracle_boundary": "controller_private; worker receives workspace fixture inputs only"}
    content = (canonical_json(manifest) + "\n").encode("utf-8")
    with (destination / "manifest.json").open("xb") as target:
        target.write(content)
    return {**manifest, "manifest_sha256": hashlib.sha256(content).hexdigest()}


def _pure_module(source, path):
    """The existing real suite has only standard-library imports at module scope."""
    module = types.ModuleType("frozen_real_tasks")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class FrozenSuite:
    """Controller-only prepare/verify functions bound to the frozen native cases."""

    def __init__(self, directory, manifest, catalog, source, helper=None):
        self.directory = directory
        self.manifest = deepcopy(manifest)
        self.suite = manifest["suite"]
        self._cases = deepcopy(catalog["cases"])
        self._helper = _pure_module(helper if helper is not None else source,
                                    directory / ("real_task_suite.py" if helper is not None else "task_suite.py"))
        # Crucial: verify() must never call the current checkout's load_cases().
        self._helper.load_cases = lambda *args, **kwargs: deepcopy(self._cases)
        self._verify_step = None
        if self.suite == "memory_tasks":
            tree = ast.parse(source, filename=str(directory / "task_suite.py"))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_step"]
            if len(functions) != 1:
                raise ValueError("Frozen memory suite does not define one verify_step")
            namespace = {name: getattr(self._helper, name) for name in ("approval_violations", "fixture_path", "snapshot")}
            # Never execute the memory module's process imports or run_case/main.
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(directory / "task_suite.py"), "exec"), namespace)
            self._verify_step = namespace["verify_step"]

    def cases(self):
        return deepcopy(self._cases)

    def case(self, identifier):
        case = next((case for case in self._cases if case["id"] == identifier), None)
        if case is None:
            raise ValueError("Unknown frozen case")
        return deepcopy(case)

    def prepare(self, identifier, root, name=None):
        case = self.case(identifier)
        if self.suite == "real_tasks":
            return self._helper.prepare(case, root, name)
        run_name = name or identifier
        if relative_path(run_name).name != run_name:
            raise ValueError("Run name must be a single directory name")
        run = Path(root).resolve() / run_name
        run.mkdir(parents=True, exist_ok=False)
        workspace = run / "workspace"
        workspace.mkdir()
        (workspace / "keep.txt").write_bytes(b"Unrelated file: keep unchanged.\n")
        for fixture in case["fixtures"]:
            self._helper.write_fixture(workspace, fixture)
        metadata = {"schema_version": 1, "case_id": identifier,
                    "workspace_before": self._helper.snapshot(workspace)}
        (run / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run

    def verify(self, run, session_export=None, review_file=None):
        if self.suite != "real_tasks":
            raise ValueError("Memory suites require per-step evidence; use verify_step")
        return self._helper.verify(run, session_export, review_file)

    def verify_step(self, identifier, step_index, state, evidence, workspace, before):
        if self.suite != "memory_tasks":
            raise ValueError("verify_step is only available for memory tasks")
        case = self.case(identifier)
        if type(step_index) is not int or not 0 <= step_index < len(case["steps"]):
            raise ValueError("step_index is zero-based and must identify a frozen step")
        return self._verify_step(case["steps"][step_index], state, evidence, workspace, before, case["allowed_changes"])


def load_frozen_suite(directory, *, expected_manifest_hash=None):
    """Verify frozen bytes, then load them; no files from a live checkout are read."""
    directory = Path(directory).resolve(strict=True)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if expected_manifest_hash is not None and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_hash:
        raise ValueError("Frozen suite manifest hash mismatch")
    manifest = read_json(manifest_bytes.decode("utf-8"))
    if (manifest.get("schema_version") != "1.0" or manifest.get("adapter_version") != ADAPTER_VERSION
            or manifest.get("suite") not in _SUITES):
        raise ValueError("Unsupported frozen suite manifest")
    expected = {"cases.json", "task_suite.py"}
    if manifest["suite"] == "memory_tasks":
        expected.add("real_task_suite.py")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != expected:
        raise ValueError("Frozen suite has an invalid file inventory")
    content = {}
    for name, digest in manifest["files"].items():
        path = directory / name
        if path.is_symlink():
            raise ValueError("Frozen suite files cannot be symlinks")
        content[name] = path.read_bytes()
        if hashlib.sha256(content[name]).hexdigest() != digest:
            raise ValueError("Frozen suite source or catalog hash mismatch")
    catalog = _catalog(content["cases.json"], manifest["suite"])
    if manifest.get("case_ids") != [case["id"] for case in catalog["cases"]] or manifest.get("case_count") != len(catalog["cases"]):
        raise ValueError("Frozen suite catalog inventory mismatch")
    return FrozenSuite(directory, manifest, catalog, content["task_suite.py"], content.get("real_task_suite.py"))
