"""Offline schema tests with synthetic MATLAB files only."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.inspect_mat_nested import inspect_mat_file, structure_summary


class TestMatNestedInspection(unittest.TestCase):
    def test_fields_shapes_and_no_numeric_sample_values(self):
        data = {
            "IMU": {"x": np.arange(24, dtype=float).reshape(3, 8)},
            "ID": {"knee": np.array([[22.456, 35.789]])}
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subj04.mat"
            savemat(path, {"Data": data})
            report = inspect_mat_file(path)
        top = report["data"]
        self.assertEqual(top["kind"], "object_array")
        self.assertEqual(top["shape"], [1, 1])
        child = top["first_element"]
        self.assertEqual(child["kind"], "mat_struct")
        self.assertIn("IMU", child["fields"])
        self.assertEqual(child["fields"]["IMU"]["shape"], [1, 1])
        self.assertNotIn("22.456", str(report))
        self.assertNotIn("35.789", str(report))

    def test_missing_data_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.mat"
            savemat(path, {"Other": np.ones((2, 3))})
            with self.assertRaisesRegex(ValueError, "Expected 'Data'"):
                inspect_mat_file(path)

    def test_bounded_metadata(self):
        values = np.empty((8,), dtype=object)
        for idx in range(8):
            values[idx] = {"signal": np.ones((2, 101))}
        summary = structure_summary(
            values, max_depth=2, max_fields=2, max_nodes=4
        )
        self.assertEqual(summary["kind"], "object_array")
        self.assertEqual(summary["shape"], [8])
        self.assertIn("first_element", summary)
        with self.assertRaises(ValueError):
            structure_summary(values, max_nodes=0)


if __name__ == "__main__":
    unittest.main()
