# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline Git + real IPython tests; no research data or models are executed."""
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell
from python_modules.core.colab_setup import setup_colab
from python_modules.core.repository_sync import load_managed_data, sync_repository
from tests import test_repository_sync as git_tests


class ColabSetupTests(unittest.TestCase):
    _git = staticmethod(git_tests.RepositorySyncTests._git)
    def setUp(self):
        git_tests.RepositorySyncTests.setUp(self)
        self.study = "biomechanical_load_estimation"
        self.src = self.source / self.study / "code" / "src"
        self.src.mkdir(parents=True)
        (self.src / "hmi_sync_demo.py").write_text(
            "VALUE = 1\ndef score():\n    return VALUE\n")
        (self.source / "requirements.txt").write_text("")
        core = self.source / "python_modules" / "core"
        core.mkdir(parents=True)
        (core / "__init__.py").write_text("")
        (self.source / "report.json").write_text('{"value": 1}')
        (self.source / ".gitignore").write_text("data/\n__pycache__/\n")
        self.publish("Study fixture")
        self.checkout = sync_repository(str(self.remote), parent=self.work)
        self.shell = InteractiveShell.instance()
        self.original_paths = sys.path[:]
        self.addCleanup(self.cleanup_shell)
        self.shell_patch = patch(
            "python_modules.core.repository_sync._get_shell",
            return_value=self.shell)
        self.shell_patch.start()
        self.addCleanup(self.shell_patch.stop)
        setup_colab(self.study, root=self.checkout)

    def cleanup_shell(self):
        active = getattr(self.shell, "_hmi_notebook_sync", None)
        if active is not None:
            self.shell.ast_transformers.remove(active)
            self.shell.events.unregister("post_run_cell", active.record_completed_cell)
            del self.shell._hmi_notebook_sync
        for name in list(sys.modules):
            if name.startswith("hmi_sync_"):
                sys.modules.pop(name)
        for name in ["score", "VALUE", "report", "marker", "observed", "alias"]:
            self.shell.user_ns.pop(name, None)
        sys.path[:] = self.original_paths
        importlib.invalidate_caches()

    def publish(self, message="Update"):
        self._git(self.source, "add", ".")
        self._git(self.source, "commit", "-m", message)
        self._git(self.source, "push", "origin", "main")

    def cell(self, code, ok=True):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = self.shell.run_cell(code)
        if ok:
            self.assertTrue(result.success, result.error_before_exec or result.error_in_exec)
        else:
            self.assertFalse(result.success)
        return result

    def test_function_and_from_import_constant_refresh_same_size(self):
        self.cell("from hmi_sync_demo import score, VALUE")
        (self.src / "hmi_sync_demo.py").write_text(
            "VALUE = 2\ndef score():\n    return VALUE\n")
        self.publish()
        self.cell("observed = (score(), VALUE)")
        self.assertEqual(self.shell.user_ns["observed"], (2, 2))
        self.assertEqual(len(self.shell._hmi_notebook_sync.history), 2)

    def test_managed_json_refresh_leaves_alias_unchanged(self):
        load_managed_data("report", self.checkout / "report.json",
                          lambda p: json.loads(p.read_text()))
        self.cell("alias = report")
        (self.source / "report.json").write_text('{"value": 2}')
        self.publish()
        self.cell("observed = report['value']")
        self.assertEqual(self.shell.user_ns["observed"], 2)
        self.assertEqual(self.shell.user_ns["alias"]["value"], 1)

    def test_dirty_checkout_without_upstream_update_can_continue(self):
        (self.checkout / "local.txt").write_text("user file")
        self.cell("marker = True")
        self.assertTrue(self.shell.user_ns["marker"])

    def test_dirty_incoming_update_continues_with_local_file(self):
        (self.checkout / "report.json").write_text("local result")
        (self.source / "report.json").write_text('{"value": 2}')
        self.publish()
        before = self._git(self.checkout, "rev-parse", "HEAD")
        setup_colab(self.study, root=self.checkout)
        self.cell("marker = True")
        self.assertTrue(self.shell.user_ns["marker"])
        self.assertEqual((self.checkout / "report.json").read_text(), "local result")
        self.assertEqual(before, self._git(self.checkout, "rev-parse", "HEAD"))

    def test_local_outputs_allow_unrelated_code_refresh(self):
        self.cell("from hmi_sync_demo import score")
        (self.checkout / "report.json").write_text("new local output")
        (self.checkout / "new_output.csv").write_text("result\n1\n")
        self._git(self.checkout, "add", "report.json")
        (self.src / "hmi_sync_demo.py").write_text(
            "VALUE = 2\ndef score():\n    return VALUE\n")
        self.publish()
        self.cell("observed = score()")
        self.assertEqual(self.shell.user_ns["observed"], 2)
        self.assertEqual((self.checkout / "report.json").read_text(), "new local output")
        self.assertEqual((self.checkout / "new_output.csv").read_text(), "result\n1\n")
        self.assertEqual(self._git(self.checkout, "show", ":report.json"), "new local output")

    def test_untracked_incoming_collision_defers_without_overwriting(self):
        (self.checkout / "new_output.csv").write_text("local output")
        (self.source / "new_output.csv").write_text("remote output")
        self.publish()
        self.cell("marker = True")
        self.assertEqual((self.checkout / "new_output.csv").read_text(), "local output")

    def test_deferred_update_resumes_after_local_change_is_saved(self):
        (self.checkout / "report.json").write_text("local output")
        (self.source / "report.json").write_text('{"value": 2}')
        self.publish()
        self.cell("marker = True")
        # Restore this temporary fixture to simulate a clean working tree.
        (self.checkout / "report.json").write_text('{"value": 1}')
        self.cell("marker = True")
        self.assertEqual((self.checkout / "report.json").read_text(), '{"value": 2}')

    def test_class_change_rejected_before_merge_and_execution(self):
        self.cell("from hmi_sync_demo import score")
        (self.src / "hmi_sync_demo.py").write_text(
            "class Model:\n    pass\ndef score():\n    return 2\n")
        self.publish()
        before = self._git(self.checkout, "rev-parse", "HEAD")
        self.cell("marker = True", ok=False)
        self.assertNotIn("marker", self.shell.user_ns)
        self.assertEqual(before, self._git(self.checkout, "rev-parse", "HEAD"))

    def test_current_final_outputs_update_before_next_cell(self):
        output = self.source / self.study / "output"
        output.mkdir()
        (output / "08_current_results.csv").write_text('{"changed": true}')
        self.publish()
        self.cell("marker = True")
        self.assertTrue((self.checkout / self.study / "output" / "08_current_results.csv").is_file())

    def test_reload_failure_latches_and_recovery_command_works(self):
        self.cell("from hmi_sync_demo import score")
        (self.src / "hmi_sync_demo.py").write_text(
            "raise RuntimeError('reload failed')\ndef score():\n    return 2\n")
        self.publish()
        self.cell("marker = True", ok=False)
        self.cell("marker = True", ok=False)
        self.assertNotIn("marker", self.shell.user_ns)
        self.cell("_hmi_stop_sync()")
        self.assertFalse(hasattr(self.shell, "_hmi_notebook_sync"))

    def test_repeated_setup_and_off_do_not_leave_duplicate_hooks(self):
        setup_colab(self.study, root=self.checkout)
        self.assertEqual(sum(type(t).__name__ == "_NotebookSync"
                             for t in self.shell.ast_transformers), 1)
        setup_colab(self.study, root=self.checkout, sync_mode="off")
        self.assertFalse(hasattr(self.shell, "_hmi_notebook_sync"))
        with patch("python_modules.core.repository_sync._git") as git:
            self.cell("marker = True")
            git.assert_not_called()

    def test_check_mode_does_not_merge(self):
        setup_colab(self.study, root=self.checkout, sync_mode="check")
        (self.source / "report.json").write_text('{"value": 2}')
        self.publish()
        self.cell("marker = True")
        self.assertEqual(json.loads((self.checkout / "report.json").read_text()),
                         {"value": 1})

    def test_ignored_collision_is_not_overwritten(self):
        (self.checkout / "data").mkdir()
        (self.checkout / "data" / "raw.txt").write_text("local raw data")
        (self.source / "data").mkdir()
        (self.source / "data" / "raw.txt").write_text("remote raw data")
        self._git(self.source, "add", "-f", "data/raw.txt")
        self.publish()
        self.cell("marker = True", ok=False)
        self.assertEqual((self.checkout / "data" / "raw.txt").read_text(),
                         "local raw data")

    def test_setup_and_imports_in_same_cell_refresh_constants(self):
        self.shell.user_ns["setup_colab"] = setup_colab
        self.shell.user_ns["checkout"] = self.checkout
        self.cell("setup_colab('biomechanical_load_estimation', root=checkout)\n"
                  "from hmi_sync_demo import VALUE")
        (self.src / "hmi_sync_demo.py").write_text(
            "VALUE = 2\ndef score():\n    return VALUE\n")
        self.publish()
        self.cell("observed = VALUE")
        self.assertEqual(self.shell.user_ns["observed"], 2)

    def test_renamed_checkout_is_supported(self):
        setup_colab(self.study, root=self.checkout, sync_mode="off")
        renamed = self.checkout.with_name("renamed-project")
        self.checkout.rename(renamed)
        self.checkout = renamed
        setup_colab(self.study, root=renamed)
        self.cell("marker = True")
        self.assertTrue(self.shell.user_ns["marker"])

    def test_dependency_constant_and_aliased_function_refresh(self):
        (self.src / "hmi_sync_dependent.py").write_text(
            "from hmi_sync_demo import VALUE\ndef dependent():\n    return VALUE\n")
        self.publish()
        self.cell("from hmi_sync_dependent import dependent as alias")
        (self.src / "hmi_sync_demo.py").write_text(
            "VALUE = 2\ndef score():\n    return VALUE\n")
        self.publish()
        self.cell("observed = alias()")
        self.assertEqual(self.shell.user_ns["observed"], 2)


if __name__ == "__main__":
    unittest.main()
