# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline group-CV tests with lightweight mocked model training."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_modules"))
try:
    from core import temporal_comparison
except ImportError:
    temporal_comparison = None


# ======================= 01. Temporal Comparison Tests =======================
@unittest.skipIf(temporal_comparison is None, "Optional PyTorch unavailable")
class TestTemporalComparison(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(72, 3, 20)).astype(np.float32)
        self.y = np.tile([1, 2, 3], 24)
        self.groups = np.repeat([f"S{i:02d}" for i in range(6)], 12)

    def mock_train(self, X_train, y_train, X_validation, y_validation, **kwargs):
        # Test split and evaluation logic, not neural-network optimisation.
        return None, None, np.asarray(y_validation).copy(), {}

    def test_complete_out_of_fold_coverage(self):
        with patch.object(temporal_comparison, "train_temporal_cnn", side_effect=self.mock_train) as train:
            folds, subjects, classes, predictions = temporal_comparison.compare_models_group_cv(
                self.X, self.y, self.groups, [1, 2, 3],
                ["a", "b", "c"], n_splits=3,
                forbidden_subject_ids=["S99"], epochs=1,
            )
        self.assertEqual(train.call_count, 6)
        self.assertEqual(len(folds), 9)
        self.assertEqual(len(subjects), 18)
        self.assertEqual(len(classes), 9)
        self.assertEqual(len(predictions), 72 * 3)
        self.assertEqual(subjects["subject_id"].nunique(), 6)
        for model, frame in predictions.groupby("model"):
            self.assertEqual(frame["source_row"].nunique(), 72)
        for row in folds.itertuples(index=False):
            self.assertFalse(
                set(row.train_subjects.split(", ")) &
                set(row.validation_subjects.split(", "))
            )

    def test_forbidden_subject_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reserved participant"):
            temporal_comparison.compare_models_group_cv(
                self.X, self.y, self.groups, [1, 2, 3],
                ["a", "b", "c"], n_splits=3,
                forbidden_subject_ids=["s01"],
            )

    def test_invalid_label_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            temporal_comparison.compare_models_group_cv(
                self.X, self.y, self.groups, [1, 2],
                ["a", "b", "c"], n_splits=3,
            )


if __name__ == "__main__":
    unittest.main()
