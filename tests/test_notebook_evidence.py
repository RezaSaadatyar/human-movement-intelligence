"""Fast offline tests for source-grounded notebook evidence inspection."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_modules"))
from core.notebook_evidence import (
    collect_notebook_inventory, extract_saved_output_text,
    validate_evidence_register,
)


class TestNotebookEvidence(unittest.TestCase):
    def _write_notebook(self, folder, filename, code_executed=True):
        path = Path(folder) / filename
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [{
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1 if code_executed else None,
                "source": ["print('PASSED')"],
                "outputs": [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["Quality gate: PASSED\n"],
                }],
            }],
        }
        path.write_text(json.dumps(notebook), encoding="utf-8")
        return notebook

    def test_saved_outputs_and_missing_execution_count(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_notebook(folder, "01_first.ipynb", code_executed=False)
            inventory = collect_notebook_inventory(folder)
            self.assertEqual(inventory.loc[0, "unexecuted_code_cells"], 1)
            self.assertEqual(inventory.loc[0, "cells_with_outputs"], 1)
            self.assertTrue(bool(inventory.loc[0, "saved_pass_marker"]))

    def test_explicit_error_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            notebook = self._write_notebook(folder, "02_bad.ipynb")
            notebook["cells"][0]["outputs"].append({
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "Sample error",
                "traceback": [],
            })
            (Path(folder) / "02_bad.ipynb").write_text(
                json.dumps(notebook), encoding="utf-8"
            )
            inventory = collect_notebook_inventory(folder)
            self.assertEqual(inventory.loc[0, "saved_error_count"], 1)
            self.assertEqual(inventory.loc[0, "saved_errors"], "ValueError")

    def test_register_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_notebook(folder, "03_valid.ipynb")
            inventory = collect_notebook_inventory(folder)
            record = pd.DataFrame([{
                "stage": 3, "notebook": "03_valid.ipynb",
                "research_question": "Source evidence",
                "protocol": "Offline inspection",
                "key_evidence": "Stored output",
                "limitations": "Not a fresh rerun",
            }])
            merged = validate_evidence_register(inventory, record)
            self.assertEqual(merged["stage"].tolist(), [3])
            record.loc[0, "notebook"] = "03_wrong.ipynb"
            with self.assertRaisesRegex(ValueError, "not found"):
                validate_evidence_register(inventory, record)

    def test_extract_plain_text_and_stream(self):
        content = {"cells": [{"cell_type": "code", "outputs": [
            {"output_type": "stream", "text": ["example\n"]},
            {"output_type": "display_data", "data": {"text/plain": ["42"]}},
        ]}]}
        self.assertIn("example", extract_saved_output_text(content))
        self.assertIn("42", extract_saved_output_text(content))


if __name__ == "__main__":
    unittest.main()
