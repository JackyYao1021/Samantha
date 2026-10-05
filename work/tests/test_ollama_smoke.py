"""Verify that live-test approvals stay within the file fixture."""

from pathlib import Path
import shlex
import unittest

from ollama_smoke import approve_fixture_commands


class FixtureApprovalTests(unittest.TestCase):
    def test_redundant_mkdir_is_allowed_only_for_the_fixture_directory(self):
        directory = Path('/tmp/samantha-fixture')
        approve_fixture_commands([
            'mkdir -p ' + shlex.quote(directory.as_posix()),
            "echo 'hello world' > hello.txt",
            'cat hello.txt',
        ], directory)
        with self.assertRaises(AssertionError):
            approve_fixture_commands(['mkdir -p /tmp/another-directory'], directory)

    def test_compound_commands_and_substitutions_cannot_bypass_approval(self):
        directory = Path('/tmp/samantha-fixture')
        for command in (
            "mkdir -p /tmp/samantha-fixture; rm -rf /tmp/another-directory",
            "echo 'hello world' > hello.txt && cat /etc/passwd",
            'echo "$(touch unexpected.txt)" > hello.txt',
            'mkdir -p /tmp/samantha-fixture/../another-directory',
        ):
            with self.subTest(command=command), self.assertRaises(AssertionError):
                approve_fixture_commands([command], directory)


if __name__ == '__main__':
    unittest.main()
