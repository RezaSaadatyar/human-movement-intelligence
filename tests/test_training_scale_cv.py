"""Fast source-row CV checks; neural training mocked for offline testing."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_modules"))
try:
    from core import training_scale_cv
except ImportError:
    training_scale_cv = None


@unittest.skipIf(training_scale_cv is None, "Optional PyTorch unavailable")
class TestTrainingScaleCV(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(144, 3, 24)).astype(np.float32)
        subjects = np.repeat([f"S{i:02d}" for i in range(6)], 24)
        labels = np.tile([1, 2, 3], 48)
        base = np.tile([True] * 12 + [False] * 12, 6)
        self.meta = pd.DataFrame({
            "subject_id": subjects, "label": labels, "in_base_500": base
        })

    @staticmethod
    def fake_training(X_train, y_train, X_validation, y_validation, **kwargs):
        return None, None, np.asarray(y_validation).copy(), {}

    def test_same_validation_source_rows_under_four_models(self):
        with patch.object(training_scale_cv, "train_temporal_cnn",
                          side_effect=self.fake_training) as mocked:
            folds, individuals, classes, predictions = (
                training_scale_cv.evaluate_nested_training_group_cv(
                    self.x, self.meta, [1, 2, 3],
                    ["x", "y", "z"], n_splits=3, epochs=1,
                    forbidden_subject_ids=["S99"],
                )
            )
        self.assertEqual(mocked.call_count, 6)
        self.assertEqual(len(folds), 12)
        self.assertEqual(len(individuals), 24)
        self.assertEqual(len(classes), 12)
        base_rows = np.flatnonzero(self.meta.in_base_500)
        for name, group in predictions.groupby("model"):
            np.testing.assert_array_equal(
                np.sort(group.source_row), base_rows
            )
        for fold, group in folds.groupby("fold"):
            self.assertEqual(group.validation_windows.nunique(), 1)
            self.assertGreater(
                group.loc[group.model == "RF 1000", "train_windows"].iloc[0],
                group.loc[group.model == "RF 500", "train_windows"].iloc[0],
            )

    def test_protected_test_cannot_enter_development(self):
        with self.assertRaisesRegex(ValueError, "Reserved Test"):
            training_scale_cv.evaluate_nested_training_group_cv(
                self.x, self.meta, [1, 2, 3],
                ["x", "y", "z"], forbidden_subject_ids=["s01"],
            )


if __name__ == "__main__":
    unittest.main()
