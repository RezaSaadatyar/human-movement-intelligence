"""Small synthetic tests for group-level CV and participant diagnostics."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/python/src"))
from core.group_cross_validation import evaluate_group_cross_validation


class TestGroupCrossValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(72, 5))
        self.y = np.tile([1, 2, 3], 24)
        self.groups = np.repeat([f"S{i:02d}" for i in range(6)], 12)

    def test_subject_level_folds_and_reports(self):
        folds, subjects, summary = evaluate_group_cross_validation(
            self.x, self.y, self.groups, [1, 2, 3], n_splits=3,
        )
        self.assertEqual(len(folds), 9)
        self.assertEqual(len(subjects), 18)
        self.assertEqual(len(summary), 3)
        self.assertEqual(subjects["subject_id"].nunique(), 6)
        for row in folds.itertuples(index=False):
            train = set(row.train_subjects.split(", "))
            held_out = set(row.validation_subjects.split(", "))
            self.assertFalse(train & held_out)
        self.assertTrue(subjects["macro_f1_fixed_task"].between(0, 1).all())

    def test_original_case_preserved_in_reports(self):
        # Input IDs are kept human-readable; group isolation is case-insensitive.
        folds, subjects, _ = evaluate_group_cross_validation(
            self.x, self.y, self.groups, [1, 2, 3], n_splits=3,
        )
        self.assertEqual(set(subjects["subject_id"]), set(self.groups))
        for row in folds.itertuples(index=False):
            self.assertTrue(all(
                member.startswith("S") for member in row.validation_subjects.split(", ")
            ))

    def test_bad_number_of_groups(self):
        with self.assertRaisesRegex(ValueError, "n_splits"):
            evaluate_group_cross_validation(
                self.x, self.y, self.groups, [1, 2, 3], n_splits=7
            )

    def test_unseen_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate_group_cross_validation(
                self.x, self.y, self.groups, [1, 2], n_splits=3
            )

    def test_nonfinite_features_rejected(self):
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            evaluate_group_cross_validation(
                bad, self.y, self.groups, [1, 2, 3], n_splits=3
            )

if __name__ == "__main__":
    unittest.main()
