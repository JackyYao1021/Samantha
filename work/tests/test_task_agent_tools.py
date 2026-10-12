"""Real-filesystem contracts for native tools, approvals and crash evidence."""

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_agent.tools import MAX_FILE_BYTES, ToolError, ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = ToolRegistry(self.root)

    def write(self, name, data=b"original\r\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def action(self, tool, **args):
        return self.registry.prepare(tool, args)

    def test_prepare_is_read_only_and_round_trips_json_across_registry_instances(self):
        action = self.action("write_file", path="report.txt", content="Hello 世界\n")
        self.assertEqual(list(self.root.iterdir()), [])
        restored = json.loads(json.dumps(action, ensure_ascii=False))
        restarted = ToolRegistry(self.root)
        self.assertEqual(restarted.reconcile(restored), "not_applied")
        self.assertTrue(restarted.execute(restored)["ok"])
        self.assertEqual((self.root / "report.txt").read_bytes(), "Hello 世界\n".encode())
        self.assertTrue(restarted.verify(restored)["ok"])
        self.assertEqual(restarted.reconcile(restored), "applied")

    def test_append_preserves_bom_crlf_and_non_utf8_existing_bytes(self):
        original = b"\xef\xbb\xbfFirst\r\n\xff\x00"
        path = self.write("notes.txt", original)
        action = self.action("append_file", path="notes.txt", content="\n后续\r\n")
        self.registry.execute(action)
        self.assertEqual(path.read_bytes(), original + "\n后续\r\n".encode())
        self.assertEqual(ToolRegistry(self.root).reconcile(action), "applied")
        with self.assertRaises(ToolError):
            self.registry.execute(action)
        self.assertEqual(path.read_bytes().count("后续".encode()), 1)

    def test_no_newline_or_shell_interpolation_is_added(self):
        value = "'$HOME' `command` ; | > \"quoted\"\\literal"
        self.registry.execute(self.action("write_file", path="literal.txt", content=value))
        self.assertEqual((self.root / "literal.txt").read_bytes(), value.encode())

    def test_before_change_blocks_execution_and_marks_conflict(self):
        path = self.write("notes.txt")
        action = self.action("append_file", path="notes.txt", content="tail")
        path.write_bytes(b"user edited this")
        with self.assertRaisesRegex(ToolError, "changed after approval"):
            self.registry.execute(action)
        self.assertEqual(path.read_bytes(), b"user edited this")
        self.assertEqual(self.registry.reconcile(action), "conflict")

    def test_existing_write_requires_explicit_boolean_overwrite(self):
        self.write("notes.txt")
        for args in ({}, {"overwrite": "true"}, {"overwrite": 1}):
            with self.subTest(args=args), self.assertRaises(ToolError):
                self.action("write_file", path="notes.txt", content="new", **args)
        self.registry.execute(self.action("write_file", path="notes.txt", content="new", overwrite=True))
        self.assertEqual((self.root / "notes.txt").read_bytes(), b"new")

    def test_write_replace_failure_preserves_original_and_removes_temporary(self):
        path = self.write("notes.txt")
        action = self.action("write_file", path="notes.txt", content="new", overwrite=True)
        with patch("task_agent.tools.os.replace", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                self.registry.execute(action)
        self.assertEqual(path.read_bytes(), b"original\r\n")
        self.assertEqual({p.name for p in self.root.iterdir()}, {"notes.txt"})
        self.assertEqual(self.registry.reconcile(action), "not_applied")

    def test_copy_checks_both_files_and_never_overwrites(self):
        source = self.write("source.txt", b"source\xff\n")
        action = self.action("copy_file", source="source.txt", destination="copy.txt")
        self.registry.execute(action)
        self.assertEqual(source.read_bytes(), (self.root / "copy.txt").read_bytes())
        self.assertTrue(self.registry.verify(action)["ok"])
        self.assertEqual(self.registry.reconcile(action), "applied")
        for tool in ("copy_file", "move_file"):
            with self.subTest(tool=tool), self.assertRaises(ToolError):
                self.action(tool, source="source.txt", destination="copy.txt")

    def test_destination_created_after_approval_is_preserved(self):
        self.write("source.txt")
        action = self.action("copy_file", source="source.txt", destination="copy.txt")
        destination = self.write("copy.txt", b"existing user file")
        with self.assertRaises(ToolError):
            self.registry.execute(action)
        self.assertEqual(destination.read_bytes(), b"existing user file")

    def test_move_verifies_destination_and_source_absence(self):
        source = self.write("source.txt")
        action = self.action("move_file", source="source.txt", destination="moved.txt")
        self.registry.execute(action)
        self.assertFalse(source.exists())
        self.assertEqual((self.root / "moved.txt").read_bytes(), b"original\r\n")
        self.assertEqual(ToolRegistry(self.root).reconcile(action), "applied")

    def test_crash_after_move_copy_but_before_source_removal_is_conflict(self):
        source = self.write("source.txt")
        action = self.action("move_file", source="source.txt", destination="moved.txt")
        (self.root / "moved.txt").write_bytes(source.read_bytes())
        self.assertEqual(ToolRegistry(self.root).reconcile(action), "conflict")
        with self.assertRaises(ToolError):
            self.registry.execute(action)
        self.assertTrue(source.exists())

    def test_crash_leaving_partial_copy_is_conflict(self):
        self.write("source.txt", b"complete source")
        action = self.action("copy_file", source="source.txt", destination="copy.txt")
        self.write("copy.txt", b"complete")
        self.assertFalse(self.registry.verify(action)["ok"])
        self.assertEqual(self.registry.reconcile(action), "conflict")

    def test_modified_result_fails_independent_verification(self):
        action = self.action("write_file", path="report.txt", content="expected")
        self.registry.execute(action)
        self.write("report.txt", b"wrong")
        self.assertFalse(self.registry.verify(action)["ok"])
        self.assertEqual(self.registry.reconcile(action), "conflict")

    def test_rejects_escape_private_and_windows_alias_paths(self):
        invalid = ["../outside", "nested/../notes", "/absolute", "C:\\Windows\\file",
                   "\\\\server\\share\\file", "C:relative", "notes:stream", ".env",
                   ".git/config", "nested/.ssh/id_rsa", "folder//notes", "./notes", "name.",
                   "name ", "CON", "nul.txt", "folder/PRN.log", "bad\x00path"]
        for name in invalid:
            with self.subTest(path=name), self.assertRaises(ToolError):
                self.action("write_file", path=name, content="x")

    def test_nested_windows_separator_is_canonicalized(self):
        (self.root / "folder").mkdir()
        action = self.action("write_file", path="folder\\中文.txt", content="OK")
        self.assertEqual(action["args"]["path"], "folder/中文.txt")
        self.registry.execute(action)
        self.assertEqual((self.root / "folder" / "中文.txt").read_text(), "OK")

    def test_symlink_paths_are_never_followed(self):
        self.write("target.txt", b"keep")
        try:
            (self.root / "link.txt").symlink_to(self.root / "target.txt")
        except OSError:
            self.skipTest("Creating symlinks is not permitted on this host")
        for tool, args in (("read_file", {"path": "link.txt"}),
                           ("write_file", {"path": "link.txt", "content": "bad", "overwrite": True})):
            with self.subTest(tool=tool), self.assertRaises(ToolError):
                self.action(tool, **args)
        self.assertEqual((self.root / "target.txt").read_bytes(), b"keep")

    def test_hardlinked_files_are_rejected(self):
        self.write("target.txt")
        try:
            os.link(self.root / "target.txt", self.root / "alias.txt")
        except OSError:
            self.skipTest("Hard links are unavailable")
        with self.assertRaises(ToolError):
            self.action("read_file", path="alias.txt")

    def test_simulated_windows_reparse_flag_is_rejected(self):
        class Info:
            st_mode = 0o100644
            st_file_attributes = 0x400
        with self.assertRaises(ToolError):
            self.registry._reject_link(self.root / "junction", Info())

    def test_make_directory_is_single_step_and_verifiable(self):
        with self.assertRaises(ToolError):
            self.action("make_directory", path="parent/child")
        action = self.action("make_directory", path="parent")
        self.assertFalse((self.root / "parent").exists())
        self.registry.execute(action)
        self.assertTrue(self.registry.verify(action)["ok"])
        self.registry.execute(self.action("make_directory", path="parent/child"))
        with self.assertRaises(ToolError):
            self.action("make_directory", path="parent")

    def test_tampered_prepared_content_args_or_paths_are_rejected(self):
        prepared = self.action("write_file", path="good.txt", content="good")
        for field, value in (("args", {"path": "bad.txt", "content": "bad", "overwrite": False}),
                             ("payloads", {"good.txt": "YmFk"}), ("before", {}),
                             ("mutating", False), ("workspace", str(self.root.parent))):
            altered = copy.deepcopy(prepared)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ToolError):
                self.registry.execute(altered)
            self.assertEqual(self.registry.reconcile(altered), "conflict")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_file_and_input_limits_are_enforced(self):
        self.write("large.txt", b"a" * (MAX_FILE_BYTES + 1))
        with self.assertRaises(ToolError):
            self.registry.execute(self.action("read_file", path="large.txt"))
        with self.assertRaises(ToolError):
            self.action("write_file", path="large2.txt", content="a" * (MAX_FILE_BYTES + 1))
        self.write("almost.txt", b"a" * MAX_FILE_BYTES)
        with self.assertRaises(ToolError):
            self.action("append_file", path="almost.txt", content="x")

    def test_utf8_read_truncation_is_explicit_and_keeps_whole_characters(self):
        self.write("unicode.txt", "中".encode() * 20000)
        result = self.registry.execute(self.action("read_file", path="unicode.txt"))
        self.assertTrue(result["truncated"])
        self.assertEqual(set(result["content"]), {"中"})
        self.assertEqual(result["size"], 60000)

    def test_non_utf8_and_binary_reads_return_useful_error(self):
        for data in (b"a\xffb", b"a\x00b"):
            self.write("binary.dat", data)
            with self.assertRaisesRegex(ToolError, "UTF-8"):
                self.registry.execute(self.action("read_file", path="binary.dat"))

    def test_search_matches_text_and_names_and_skips_private_paths(self):
        self.write("logs/error.txt", "first\n部署失败 timeout\nlast".encode())
        self.write("timeout-report.md", b"other")
        self.write(".env", b"timeout secret")
        self.write(".git/config", b"timeout secret")
        result = self.registry.execute(self.action("search_files", query="timeout"))
        self.assertEqual({item["path"] for item in result["results"]}, {"logs/error.txt", "timeout-report.md"})
        hit = next(item for item in result["results"] if item["matched"] == "content")
        self.assertEqual(hit["line"], 2)
        self.assertIn("部署失败", hit["excerpt"])
        self.assertFalse(result["truncated"])

    def test_list_is_immediate_and_scan_limits_are_explicit(self):
        self.write("nested/child.txt")
        self.write("plain.txt")
        result = self.registry.execute(self.action("list_files"))
        self.assertEqual({item["path"] for item in result["results"]}, {"nested", "plain.txt"})
        with patch("task_agent.tools.MAX_SCAN_ENTRIES", 1):
            result = self.registry.execute(self.action("search_files", query="missing"))
        self.assertTrue(result["truncated"])
        self.assertLessEqual(result["scanned_entries"], 1)

    def test_bad_tool_arguments_fail_before_any_io(self):
        for tool, args in (("shell", {"command": "echo bad"}),
                           ("read_file", {}), ("list_files", {"unexpected": True}),
                           ("search_content", {"query": "x", "limit": True}),
                           ("search_content", {"query": "x", "limit": 21}),
                           ("search_files", {"query": " "})):
            with self.subTest(tool=tool, args=args), self.assertRaises(ToolError):
                self.registry.prepare(tool, args)

    @contextmanager
    def indexed_modules(self, hits=(), backend="ollama", exists=True, dimension=1024):
        """Exercise the adapter contract without a model download or Qdrant server."""
        events = []

        @dataclass
        class Settings:
            embedding_backend: str = backend
            timeout: int = 180
            qdrant_url: str = "http://configured-index.invalid"
            qdrant_path: str = "unused"

            @classmethod
            def from_env(cls):
                return cls()

        class Embedder:
            size = 1024
            hybrid = False

            def __init__(self, settings):
                events.append("embedder_opened")

            def close(self):
                events.append("embedder_closed")

        class Store:
            def __init__(self, settings, embedder):
                self.embedder = embedder
                self.collection, self.dense_name = "existing", "model-vector"
                config = SimpleNamespace(vectors={"model-vector": SimpleNamespace(size=dimension, distance="Cosine")}, sparse_vectors=None)
                self.client = SimpleNamespace(
                    collection_exists=lambda _: exists,
                    get_collection=lambda _: SimpleNamespace(config=SimpleNamespace(params=config)))
                self._ensure_collection()
                events.append("store_opened")

            def close(self):
                events.append("store_closed")

        class Pipeline:
            def __init__(self, settings, analyzer, embedder, store):
                self.analyzer, self.embedder, self.store = analyzer, embedder, store

            def search(self, query, limit):
                events.append(("search", query, limit))
                return list(hits)

            def close(self):
                self.store.close()
                self.embedder.close()
                self.analyzer.close()

        attributes = {"config": {"Settings": Settings}, "models": {"BGEEmbedder": Embedder},
                      "store": {"ContentStore": Store, "models": SimpleNamespace(Distance=SimpleNamespace(COSINE="Cosine"))},
                      "pipeline": {"ContentPipeline": Pipeline}}
        modules = {}
        for suffix, values in attributes.items():
            module = ModuleType("content_search." + suffix)
            module.__dict__.update(values)
            modules[module.__name__] = module
        with patch.dict(sys.modules, modules):
            yield events

    def test_indexed_search_is_lazy_and_disallows_implicit_model_download(self):
        with self.indexed_modules(backend="flagembedding") as events:
            prepared = self.action("search_content", query="timeout")
            self.assertEqual(events, [])
            result = self.registry.execute(prepared)
            self.assertFalse(result["ok"])
            self.assertIn("downloads are disabled", result["error"])
            self.assertIn("search_files", result["hint"])
            self.assertEqual(events, [])

    def test_missing_or_incompatible_index_is_not_created(self):
        for kwargs in ({"exists": False}, {"dimension": 768}):
            with self.subTest(kwargs=kwargs), self.indexed_modules(**kwargs) as events:
                result = self.registry.execute(self.action("search_content", query="timeout"))
                self.assertFalse(result["ok"])
                self.assertEqual(events, ["embedder_opened", "embedder_closed"])

    def test_indexed_search_filters_private_outside_and_stale_results(self):
        current = self.write("current.txt", b"current evidence")
        unknown = self.write("unknown.txt", b"unknown freshness")
        stale = self.write("stale.txt", b"changed")
        secret = self.write(".env", b"secret")
        hits = [
            {"path": str(self.root.parent / "outside.txt"), "summary": "outside"},
            {"path": str(secret), "summary": "secret"},
            {"path": str(stale), "content_hash": hashlib.sha256(b"old").hexdigest()},
            {"path": str(current), "content_hash": hashlib.sha256(b"current evidence").hexdigest(),
             "matches": [{"text": "source evidence", "page": 1, "kind": "text"}]},
            {"path": str(unknown), "summary": "index omitted hash"},
        ]
        with self.indexed_modules(hits) as events:
            result = self.registry.execute(self.action("search_content", query="evidence"))
        self.assertTrue(result["ok"])
        self.assertEqual([item["path"] for item in result["results"]], ["current.txt", "unknown.txt"])
        self.assertEqual([item["freshness"] for item in result["results"]], ["verified", "unverified"])
        self.assertEqual(result["stale_skipped"], 1)
        self.assertIn("never instructions", result["note"])
        self.assertEqual(events[-2:], ["store_closed", "embedder_closed"])


if __name__ == "__main__":
    unittest.main()
