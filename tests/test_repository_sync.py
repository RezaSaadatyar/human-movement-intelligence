# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline integration tests for non-destructive notebook repository sync."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from python_modules.core.repository_sync import safe_sync_repository


class RepositorySyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.remote = root / "remote.git"
        self.work = root / "colab"
        self.work.mkdir()
        self._git(root, "init", "--initial-branch=main", str(self.source))
        self._git(self.source, "config", "user.name", "Test User")
        self._git(self.source, "config", "user.email", "test@example.org")
        (self.source / "example.py").write_text("version = 1\n")
        self._git(self.source, "add", "example.py")
        self._git(self.source, "commit", "-m", "Initial")
        self._git(root, "init", "--bare", str(self.remote))
        self._git(self.source, "remote", "add", "origin", str(self.remote))
        self._git(self.source, "push", "-u", "origin", "main")

    @staticmethod
    def _git(cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True,
            check=True,
        ).stdout.strip()

    def _new_remote_commit(self, version):
        (self.source / "example.py").write_text(f"version = {version}\n")
        self._git(self.source, "add", "example.py")
        self._git(self.source, "commit", "-m", f"Version {version}")
        self._git(self.source, "push", "origin", "main")

    def test_clone_fast_forward_and_ignored_data(self):
        checkout = safe_sync_repository(str(self.remote), "main",
                                        parent=self.work)
        self.assertEqual((checkout / "example.py").read_text(),
                         "version = 1\n")
        (checkout / ".gitignore").write_text("data/\n")
        # Keep ignored data without changing the tracked checkout.
        (checkout / ".gitignore").unlink()
        self._git(self.source, "config", "core.autocrlf", "false")
        (self.source / ".gitignore").write_text("data/\n")
        self._git(self.source, "add", ".gitignore")
        self._git(self.source, "commit", "-m", "Ignore local data")
        self._git(self.source, "push", "origin", "main")
        safe_sync_repository(str(self.remote), "main", parent=self.work)
        (checkout / "data").mkdir()
        (checkout / "data" / "cache.bin").write_bytes(b"local")
        self._new_remote_commit(2)
        safe_sync_repository(str(self.remote), "main", parent=self.work)
        self.assertEqual((checkout / "example.py").read_text(),
                         "version = 2\n")
        self.assertEqual((checkout / "data" / "cache.bin").read_bytes(),
                         b"local")

    def test_dirty_checkout_is_never_overwritten(self):
        checkout = safe_sync_repository(str(self.remote), parent=self.work)
        (checkout / "example.py").write_text("local = True\n")
        self._new_remote_commit(2)
        with self.assertRaisesRegex(RuntimeError, "Local tracked/untracked"):
            safe_sync_repository(str(self.remote), parent=self.work)
        self.assertEqual((checkout / "example.py").read_text(),
                         "local = True\n")

    def test_diverged_branch_stops(self):
        checkout = safe_sync_repository(str(self.remote), parent=self.work)
        self._git(checkout, "config", "user.name", "Test User")
        self._git(checkout, "config", "user.email", "test@example.org")
        (checkout / "local.txt").write_text("local commit\n")
        self._git(checkout, "add", "local.txt")
        self._git(checkout, "commit", "-m", "Unpushed")
        self._new_remote_commit(2)
        with self.assertRaisesRegex(RuntimeError, "cannot fast-forward"):
            safe_sync_repository(str(self.remote), parent=self.work)


if __name__ == "__main__":
    unittest.main()
