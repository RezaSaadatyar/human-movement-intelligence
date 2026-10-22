"""Offline synthetic checks for strict matched HARTH benchmark and cache provenance."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python_modules"))
try:
    from core import temporal_benchmark as bench
except ImportError:
    bench = None


@unittest.skipIf(bench is None, "Optional PyTorch unavailable in standard CI")
class TestMatchedBenchmark(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(32)
        self.windows = generator.normal(size=(72, 3, 20)).astype(np.float32)
        self.labels = np.tile([1, 2, 3], 24)
        self.subjects = np.repeat([f"S{i:02d}" for i in range(6)], 12)
        self.plan = bench.make_benchmark_plan(
            self.windows, self.labels, self.subjects, [1, 2, 3], n_splits=3,
            forbidden_subjects=["S98", "S99"]
        )

    def test_grouped_source_coverage_and_fingerprint(self):
        self.assertEqual(len(self.plan["folds"]), 3)
        original = sorted(np.concatenate(
            [va for _, _, va in self.plan["folds"]]).tolist())
        self.assertEqual(original, list(range(72)))
        altered = self.windows.copy()
        altered[0, 0, 0] += 1.
        different = bench.make_benchmark_plan(
            altered, self.labels, self.subjects, [1, 2, 3], n_splits=3,
            forbidden_subjects=[]
        )
        self.assertNotEqual(self.plan["fingerprint"], different["fingerprint"])

    def test_protected_subject_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reserved Test"):
            bench.make_benchmark_plan(
                self.windows, self.labels, self.subjects, [1, 2, 3],
                forbidden_subjects=["s01"]
            )

    def test_resumable_jobs_and_invalid_cache_rejected(self):
        # No actual RF or neural training is required for integrity testing.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(bench, "_fit_and_predict",
                              side_effect=lambda plan, model, tr, va, **kw:
                                  (plan["labels"][va].copy(), 0.1, 0.01, 0)) as fit:
                first = bench.run_benchmark_job(
                    self.plan, 1, "Random Forest", tmp,
                    feature_matrix=np.zeros((72, 5)),
                )
                second = bench.run_benchmark_job(
                    self.plan, 1, "Random Forest", tmp,
                    feature_matrix=np.zeros((72, 5)),
                )
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(first["cache_key"], second["cache_key"])
                self.assertEqual(first["source_rows"], second["source_rows"])

    def test_reject_incomplete_job_matrix(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            bench.summarise_benchmark_results(self.plan, [])

if __name__ == "__main__":
    unittest.main()
