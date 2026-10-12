"""Saved-only integrity and historical-score audits; no model/network calls."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.harness import write_json
from evals.manifest import file_hash
from evals.routing_audit import audit, verify_sources
from evals.routing_tasks import OUTPUT_SCHEMA


MODELS = ["qwen3:4b", "qwen2.5vl:7b"]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")


class RoutingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "saved-run"
        frozen = self.run / "frozen"
        source = frozen / "sources" / "work" / "evals" / "routing_tasks.py"
        source.parent.mkdir(parents=True)
        source.write_text("# Frozen original scoring module.\n", encoding="utf-8")
        self.cases = [
            {"case_id": "RD-one", "split": "development", "family": "write"},
            {"case_id": "RD-missing", "split": "development", "family": "rename"},
            {"case_id": "RH-one", "split": "heldout", "family": "select"},
        ]
        write_jsonl(frozen / "dataset.jsonl", self.cases)
        config = {"schema_version": "routing-run-v1", "models": MODELS,
                  "embedding_model": "bge-m3", "policies": ["fixed_a", "fixed_b"],
                  "repeats": 1, "seed": 17, "max_calls": 20, "max_seconds": 60,
                  "request_timeout_seconds": 10, "max_output_tokens": 768, "fallback": False}
        write_json(frozen / "config.json", config)
        write_json(frozen / "prompts.json", {"solver": "frozen solver", "classifier": "frozen classifier"})
        self.selection = {"economical_model": MODELS[0],
                          "family_models": {"write": MODELS[0], "rename": MODELS[1], "select": MODELS[1]}}
        self.manifest = {
            "schema_version": "routing-run-v1", "config": config,
            "frozen_source_hashes": {"work/evals/routing_tasks.py": file_hash(source)},
            "dataset": {"sha256": file_hash(frozen / "dataset.jsonl"),
                        "case_ids": [case["case_id"] for case in self.cases]},
            "prompt_sha256": file_hash(frozen / "prompts.json"),
            "selection": deepcopy(self.selection), "policy_setup_calls": [], "actual_physical_requests": 2,
        }
        self.tasks = SimpleNamespace(
            OUTPUT_SCHEMA=deepcopy(OUTPUT_SCHEMA),
            load_cases=Mock(return_value=deepcopy(self.cases)),
            evaluate_response=Mock(side_effect=lambda case, payload: {
                "task_success": 1 if case["case_id"] == "RD-one" else 0}),
        )
        self.routing = SimpleNamespace(build_capabilities=Mock(return_value={"frozen": "capabilities"}),
                                       choose_models=Mock(return_value=deepcopy(self.selection)))
        self.frozen_runner = SimpleNamespace(CLASSIFIER_SCHEMA={
            "type": "object", "properties": {"family": {"type": "string", "enum": ["write", "select"]}},
            "required": ["family"], "additionalProperties": False})
        self.records, self.events, self.bodies = [], [], {}
        for number, (case_id, split, family, model, score) in enumerate((
                ("RD-one", "development", "write", MODELS[0], 1),
                ("RH-one", "heldout", "select", MODELS[1], 0)), 1):
            # Construct the original v1 transport order; write_json then sorts
            # the readable JSON artifact, including nested options and schema.
            payload = {"model": model,
                "messages": [{"role": "system", "content": "frozen solver"},
                             {"role": "user", "content": "public fixture α"}],
                "stream": False, "format": deepcopy(self.tasks.OUTPUT_SCHEMA),
                "options": {"temperature": 0, "seed": 17, "num_ctx": 4096, "num_predict": 768},
                "keep_alive": "5m"}
            if model == "qwen3:4b":
                payload["think"] = False
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.bodies[number] = body
            directory = self.run / "requests" / f"request_{number:04d}"
            directory.mkdir(parents=True)
            write_json(directory / "request.json", payload)
            output = {"status": "completed", "actions": [], "answer": [], "question": ""}
            write_json(directory / "response.json", {"message": {"content": json.dumps(output)}})
            metadata = {"request_number": number, "model": model, "role": "solver", "call_type": "chat",
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "request_ref": f"requests/request_{number:04d}/request.json"}
            physical = {**metadata, "call_id": f"physical-{number}", "status": "succeeded",
                        "usage": {"input_tokens": 16, "output_tokens": 8}, "duration_ms": 10.,
                        "provider_load_ms": 1., "provider_compute_ms": 8.}
            write_json(directory / "telemetry.json", physical)
            self.events.extend([{**metadata, "event": "submitted"}, {**metadata, "event": "finished"}])
            self.records.append({"case_id": case_id, "split": split, "family": family, "model": model,
                "policy": "fixed_a" if split == "development" else "fixed_b", "repeat": 1,
                "complete": True, "task_success": score, "response": output, "calls": [physical],
                "duration_ms": 10., "provider_compute_ms": 8., "input_tokens": 16, "output_tokens": 8,
                "artifact_ref": f"attempts/{case_id}.json"})
        self._save_records()
        write_jsonl(self.run / "requests.jsonl", self.events)
        write_json(self.run / "manifest.json", self.manifest)

    def _save_records(self):
        (self.run / "attempts").mkdir(exist_ok=True)
        for row in self.records:
            write_json(self.run / "attempts" / f"{row['case_id']}.json",
                       {key: value for key, value in row.items() if key != "artifact_ref"})
        write_jsonl(self.run / "results.jsonl", self.records)

    def _audit(self):
        with patch("evals.routing_audit._frozen_modules", return_value=(self.tasks, self.routing, self.frozen_runner)):
            return audit(self.run, self.root / "audit-output")

    def test_verify_sources_rejects_source_dataset_config_and_prompt_tampering(self):
        verify_sources(self.run, self.manifest)
        source = self.run / "frozen" / "sources" / "work" / "evals" / "routing_tasks.py"
        targets = [source, self.run / "frozen" / "dataset.jsonl", self.run / "frozen" / "config.json",
                   self.run / "frozen" / "prompts.json"]
        for target in targets:
            original = target.read_bytes()
            with self.subTest(file=target.name):
                if target.name == "config.json":
                    changed = deepcopy(self.manifest["config"])
                    changed["max_calls"] += 1
                    write_json(target, changed)
                else:
                    target.write_bytes(original + b"\n# tampered\n")
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    verify_sources(self.run, self.manifest)
            target.write_bytes(original)

    def test_result_evidence_reference_cannot_escape_the_saved_run(self):
        outside = self.root / "outside.json"
        write_json(outside, {key: value for key, value in self.records[0].items() if key != "artifact_ref"})
        self.records[0]["artifact_ref"] = "../outside.json"
        write_jsonl(self.run / "results.jsonl", self.records)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self._audit()

    def test_result_artifact_and_journal_must_agree(self):
        artifact = self.run / "attempts" / "RD-one.json"
        altered = json.loads(artifact.read_text(encoding="utf-8"))
        altered["task_success"] = 0
        write_json(artifact, altered)
        with self.assertRaisesRegex(ValueError, "journal does not match"):
            self._audit()

    def test_original_frozen_rules_regrade_both_success_and_failure_consistently(self):
        original_results = (self.run / "results.jsonl").read_bytes()
        with patch("evals.routing_tasks.evaluate_response", side_effect=AssertionError("live scorer must not run")):
            result = self._audit()
        self.assertEqual(result["regrade_count"], 2)
        self.assertEqual(result["regrade_agreement"], 2)
        self.assertEqual([row["regraded_success"] for row in result["frozen_rule_regrades"]], [1, 0])
        self.tasks.evaluate_response.assert_has_calls([
            call(self.cases[0], self.records[0]["response"]),
            call(self.cases[2], self.records[1]["response"]),
        ])
        self.assertEqual(original_results, (self.run / "results.jsonl").read_bytes())
        self.assertTrue(result["ledger_complete"])

    def test_duplicate_physical_calls_across_attempts_are_rejected(self):
        self.records[1]["calls"] = deepcopy(self.records[0]["calls"])
        self._save_records()
        with self.assertRaisesRegex(ValueError, "Physical model request appears"):
            self._audit()

    def test_setup_call_cannot_also_be_attributed_to_an_attempt(self):
        self.manifest["policy_setup_calls"] = deepcopy(self.records[0]["calls"])
        write_json(self.run / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Setup request is also attributed"):
            self._audit()

    def test_partial_development_reconstructs_every_missing_model_case_as_zero(self):
        result = self._audit()
        self.assertTrue(result["development_selection_verified"])
        reconstructed, models = self.routing.build_capabilities.call_args.args
        self.assertEqual(models, MODELS)
        self.assertEqual({(row["case_id"], row["model"]) for row in reconstructed},
                         {(case_id, model) for case_id in ("RD-one", "RD-missing") for model in MODELS})
        stubs = [row for row in reconstructed if not row["complete"]]
        self.assertEqual(len(stubs), 3)
        for row in stubs:
            self.assertEqual(row["task_success"], 0)
            self.assertIsNone(row["duration_ms"])
            self.assertIsNone(row["provider_compute_ms"])
            self.assertIsNone(row["input_tokens"])
            self.assertIsNone(row["output_tokens"])
        self.assertNotIn("RH-one", {row["case_id"] for row in reconstructed})

    def test_selection_drift_is_rejected_instead_of_reusing_manifest_unchecked(self):
        self.routing.choose_models.return_value = {"economical_model": MODELS[1]}
        with self.assertRaisesRegex(ValueError, "selection cannot be reconstructed"):
            self._audit()

    def test_invalid_and_duplicate_raw_json_are_diagnosed_without_changing_old_scores(self):
        contents = ['{"status":"clarify","status":"completed"}', '{invalid json']
        for number, content in enumerate(contents, 1):
            write_json(self.run / "requests" / f"request_{number:04d}" / "response.json",
                       {"message": {"content": content}})
        before = {path.relative_to(self.run).as_posix(): path.read_bytes()
                  for path in self.run.rglob("*") if path.is_file()}
        result = self._audit()
        after = {path.relative_to(self.run).as_posix(): path.read_bytes()
                 for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertEqual(result["strict_output_duplicate_keys"], 1)
        self.assertEqual({item["kind"] for item in result["raw_output_diagnostics"]}, {"duplicate_keys", "invalid_json"})
        self.assertEqual(result["regrade_agreement"], 2)
        self.assertEqual([row["original_success"] for row in result["frozen_rule_regrades"]], [1, 0])

    def test_legacy_sorted_request_artifact_reconstructs_the_original_transport_digest(self):
        readable = json.loads((self.run / "requests" / "request_0001" / "request.json").read_text(encoding="utf-8"))
        naive = json.dumps(readable, ensure_ascii=False).encode("utf-8")
        self.assertNotEqual(hashlib.sha256(naive).hexdigest(), self.events[0]["request_sha256"])
        self.assertTrue(self._audit()["ledger_complete"])

    def test_saved_request_body_is_verified_byte_for_byte(self):
        for number, body in self.bodies.items():
            (self.run / "requests" / f"request_{number:04d}" / "request_body.bin").write_bytes(body)
        body_path = self.run / "requests" / "request_0001" / "request_body.bin"
        body_path.write_bytes(self.bodies[1] + b" ")
        with self.assertRaisesRegex(ValueError, "Submitted request bytes"):
            self._audit()

    def test_changed_legacy_request_parameters_invalidate_the_submitted_digest(self):
        request_path = self.run / "requests" / "request_0001" / "request.json"
        altered = json.loads(request_path.read_text(encoding="utf-8"))
        altered["options"]["num_predict"] += 1
        write_json(request_path, altered)
        with self.assertRaisesRegex(ValueError, "Submitted request bytes"):
            self._audit()


if __name__ == "__main__":
    unittest.main()
