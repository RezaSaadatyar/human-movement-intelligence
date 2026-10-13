"""Offline synthetic tests of the Python-only BoneNet workflow."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.bonenet_python_pilot import (
    assemble_reference, inspect_source, numeric_matrix, run_baseline
)


class TestBoneNetPilot(unittest.TestCase):
    def test_numeric_assembly(self):
        n, t = 10, 15
        data = {
            "inputs": {
                "x": {"a": np.ones((n, t)), "b": np.ones((n, t))*2},
                "y": {"a": np.ones((n, t))*3},
                "z": {"a": np.ones((n, t))*4},
            },
            "outputs": {"e1": np.ones((n, 11)), "e3": -np.ones((n, 11))},
        }
        x, y, summary = assemble_reference(data, n)
        self.assertEqual(x.shape, (10, 4, 15))
        self.assertEqual(y.shape, (10, 11, 2))
        self.assertEqual(summary["target_names"], ["e1", "e3"])

    def test_reject_opaque_objects(self):
        with self.assertRaisesRegex(ValueError, "expected 2D numeric"):
            numeric_matrix({"_Class": "table"}, "IMU")

    def test_disjoint_subject_baseline(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(48, 4, 20))
        y = rng.normal(size=(48, 11, 2))
        labels = [{"SubjID": f"subj{i//4:02d}", "Task": "Walking", "TrialNum": "1"}
                  for i in range(48)]
        report = run_baseline(x, y, labels)
        self.assertEqual(report["train_subjects"] + report["test_subjects"], 12)
        self.assertIn("ridge_descriptors", report["metrics"])

    def test_mat_ingestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 12
            data = {
                "inputs": {axis: {"imu": np.ones((n, 10))}
                           for axis in ("x", "y", "z")},
                "outputs": {"e1": np.ones((n, 11)), "e3": np.ones((n, 11))*2},
            }
            savemat(root / "Data.mat", data)
            with (root / "labels.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["SubjID", "Task", "TrialNum"])
                for i in range(n):
                    writer.writerow([f"subj{i//2:02d}", "Running", i%2+1])
            audit, rows, x, y = inspect_source(root)
            self.assertTrue(audit["training_ready"])
            self.assertEqual(x.shape, (12, 3, 10))
            self.assertEqual(y.shape, (12, 11, 2))
            self.assertEqual(len(rows), 12)

    def test_reject_unaligned_samples(self):
        n, t = 12, 20
        data = {
            "inputs": {a: {"imu": np.ones((n, t))} for a in ("x", "y", "z")},
            "outputs": {"e1": np.ones((n, t))}
        }
        with self.assertRaisesRegex(ValueError, "Sample count mismatch"):
            assemble_reference(data, n+1)


if __name__ == "__main__":
    unittest.main()
