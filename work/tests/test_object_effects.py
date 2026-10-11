"""Proposal/command correspondence tests, including literal shell boundaries."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking.effects import (
    EffectValidationError, MAX_CONTENT_BYTES, bind_effects, parse_effects, render_effects,
)


class ObjectEffectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cwd = self.temporary.name

    def effect(self, kind="create", path="notes.txt", **fields):
        raw = {"kind": kind, "path": path}
        raw.update({"destination": "todo.txt"} if kind == "rename" else {"content": "alpha\n"})
        raw.update(fields)
        return parse_effects({"Effects": [raw]}, self.cwd)

    def test_relative_paths_are_fixed_to_initial_directory(self):
        effects = self.effect("rename", path="folder/../notes.txt", destination="中文 任务.txt")
        self.assertEqual(effects[0]["path"], os.path.join(self.cwd, "notes.txt"))
        self.assertEqual(effects[0]["destination"], os.path.join(self.cwd, "中文 任务.txt"))
        other_cwd = os.path.join(self.cwd, "other")
        self.assertEqual(bind_effects(render_effects(effects), effects, other_cwd), effects)

    def test_renderer_preserves_unicode_spaces_quotes_and_literal_shell_symbols(self):
        content = "你好 'quoted' \"double\" $HOME `command` ; | > \\literal\nsecond line\tend"
        for kind in ("create", "append"):
            with self.subTest(kind=kind):
                effects = self.effect(kind, path="中文 ' 文件 $value.txt", content=content)
                self.assertEqual(bind_effects(render_effects(effects), effects, self.cwd), effects)
        rename = self.effect("rename", path="中文 ' 文件.txt", destination="新 文件.txt")
        self.assertEqual(bind_effects(render_effects(rename), rename, self.cwd), rename)

    def test_rendered_commands_write_and_move_exact_utf8_bytes_in_real_bash(self):
        bash = os.environ.get("SAMANTHA_BASH") or shutil.which("bash")
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if os.name == "nt" and git_bash.is_file() and not os.environ.get("SAMANTHA_BASH"):
            bash = str(git_bash)
        if not bash:
            self.skipTest("Bash is unavailable")
        content = "你好 'quoted' \"double\" $HOME `command` ; | > \\literal\nsecond line\tend"
        create = self.effect(path="中文 ' 文件.txt", content=content)
        rename = self.effect("rename", path="中文 ' 文件.txt", destination="新 文件.txt")
        append = self.effect("append", path="新 文件.txt", content="\n完成 $literal\n")
        for effects in (create, rename, append):
            command = render_effects(effects)[0]
            self.assertEqual(bind_effects([command], effects, self.cwd), effects)
            result = subprocess.run([bash, "--noprofile", "--norc", "-c", command],
                                    cwd=self.cwd, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertFalse(Path(create[0]["path"]).exists())
        self.assertEqual(Path(append[0]["path"]).read_bytes(), (content + append[0]["content"]).encode("utf-8"))

    def test_newline_printf_and_simple_echo_match_exact_bytes(self):
        effects = self.effect(content="alpha\n")
        for command in ("printf '%s\\n' 'alpha' > notes.txt", "echo 'alpha' > notes.txt"):
            with self.subTest(command=command):
                self.assertEqual(bind_effects([command], effects, self.cwd), effects)
        with self.assertRaises(EffectValidationError):
            bind_effects(["printf '%s' 'alpha' > notes.txt"], effects, self.cwd)

    def test_empty_create_and_literal_printf_backslashes_are_supported(self):
        for content in ("", r"alpha\nbeta"):
            with self.subTest(content=content):
                effects = self.effect(content=content)
                self.assertEqual(bind_effects(render_effects(effects), effects, self.cwd), effects)

    def test_entities_require_paired_nonboolean_version_and_create_has_new_identity(self):
        effects = self.effect("append", entity_id="known-object", entity_version=2)
        self.assertEqual(effects[0]["entity_version"], 2)
        for fields in ({"entity_id": "known-object"}, {"entity_version": 2},
                       {"entity_id": "known-object", "entity_version": True},
                       {"entity_id": "known-object", "entity_version": 0}):
            with self.subTest(fields=fields), self.assertRaises(EffectValidationError):
                self.effect("append", **fields)
        with self.assertRaises(EffectValidationError):
            self.effect("create", entity_id="known-object", entity_version=1)

    def test_proposal_schema_rejects_malformed_or_unbounded_values(self):
        invalid = [
            {}, {"Effects": None}, {"Effects": {}},
            {"Effects": [{"kind": "delete", "path": "notes.txt"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": 3}]},
            {"Effects": [{"kind": "create", "path": "", "content": "alpha"}]},
            {"Effects": [{"kind": "create", "path": "bad\nname", "content": "alpha"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "\x00"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "\ud800"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "alpha", "extra": 1}]},
            {"Effects": [{"kind": "append", "path": "notes.txt", "content": ""}]},
            {"Effects": [{"kind": "rename", "path": "notes.txt", "destination": "notes.txt"}]},
            {"Effects": [{"kind": "rename", "path": "notes.txt", "destination": "todo.txt", "content": "alpha"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "alpha", "destination": "todo.txt"}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "中" * (MAX_CONTENT_BYTES // 3 + 1)}]},
            {"Effects": [{"kind": "create", "path": "notes.txt", "content": "alpha"}] * 2},
        ]
        for data in invalid:
            with self.subTest(data=str(data)[:120]), self.assertRaises(EffectValidationError):
                parse_effects(data, self.cwd)

    def test_empty_effects_retain_legacy_arbitrary_commands_without_tracking(self):
        self.assertEqual(parse_effects({"Effects": []}, self.cwd), [])
        self.assertEqual(bind_effects(["ls -la | head", "mkdir -p new"], [], self.cwd), [])

    def test_extra_commands_and_nonmatching_effects_are_rejected(self):
        effects = self.effect(content="alpha")
        invalid = [
            [], ["printf '%s' alpha > notes.txt", "ls"],
            ["printf '%s' beta > notes.txt"],
            ["printf '%s' alpha > other.txt"],
            ["printf '%s' alpha >> notes.txt"],
            ["touch notes.txt"],
        ]
        for commands in invalid:
            with self.subTest(commands=commands), self.assertRaises(EffectValidationError):
                bind_effects(commands, effects, self.cwd)

    def test_unsafe_shell_syntax_and_formats_cannot_claim_effects(self):
        effects = self.effect(content="alpha")
        invalid = [
            "printf '%s' $CONTENT > notes.txt",
            'printf "%s" "${CONTENT}" > notes.txt',
            "printf '%s' `echo alpha` > notes.txt",
            "printf '%s' $(echo alpha) > notes.txt",
            "printf '%s' alpha > notes.txt; rm other.txt",
            "printf '%s' alpha > notes.txt && true",
            "printf '%s' alpha | tee notes.txt",
            "printf '%s' alpha > notes.txt\ntrue",
            "printf '%s' alpha > *.txt",
            "printf '%s' alpha > ~/notes.txt",
            "printf '%s' alpha > notes.txt # hidden",
            "printf '%s' alpha 2> notes.txt",
            "printf '%s' alpha > notes.txt 2>&1",
            "printf '%s' alpha >>> notes.txt",
            "printf 'alpha' > notes.txt",
            "printf '%b' alpha > notes.txt",
            "printf '%s' alpha beta > notes.txt",
            "printf '%s' alpha > notes.txt other.txt",
            "echo -n alpha > notes.txt",
            r"echo 'alpha\n' > notes.txt",
            'printf "%s" "alpha\\q" > notes.txt',
        ]
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(EffectValidationError):
                bind_effects([command], effects, self.cwd)

    def test_quoted_or_escaped_redirect_is_never_mistaken_for_operator(self):
        effects = self.effect(content="alpha")
        for command in ("printf '%s' alpha '>' notes.txt", r"printf '%s' alpha \> notes.txt"):
            with self.subTest(command=command), self.assertRaises(EffectValidationError):
                bind_effects([command], effects, self.cwd)

    def test_io_number_redirection_cannot_be_mistaken_for_printf_content(self):
        effects = self.effect(content="1")
        with self.assertRaises(EffectValidationError):
            bind_effects(["printf '%s' 1> notes.txt"], effects, self.cwd)

    def test_command_parent_traversal_cannot_hide_symlink_semantics(self):
        effects = self.effect(content="alpha")
        with self.assertRaises(EffectValidationError):
            bind_effects(["printf '%s' alpha > 'link/../notes.txt'"], effects, self.cwd)
        rename = self.effect("rename")
        with self.assertRaises(EffectValidationError):
            bind_effects(["mv -- link/../notes.txt todo.txt"], rename, self.cwd)

    def test_mv_binding_rejects_options_expansion_and_mismatched_destination(self):
        effects = self.effect("rename")
        for command in ("mv -- notes.txt todo.txt", "mv notes.txt todo.txt"):
            self.assertEqual(bind_effects([command], effects, self.cwd), effects)
        for command in ("mv -f notes.txt todo.txt", "mv -- notes.txt other.txt",
                        "mv -- *.txt todo.txt", "cp -- notes.txt todo.txt",
                        "mv -- notes.txt todo.txt; true"):
            with self.subTest(command=command), self.assertRaises(EffectValidationError):
                bind_effects([command], effects, self.cwd)

    def test_dash_prefixed_literal_paths_require_mv_option_terminator(self):
        effects = self.effect("rename", path="-n", destination="-f")
        self.assertEqual(bind_effects(["mv -- '-n' '-f'"], effects, self.cwd), effects)
        with self.assertRaises(EffectValidationError):
            bind_effects(["mv '-n' '-f'"], effects, self.cwd)


if __name__ == "__main__":
    unittest.main()
