# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline contract checks for the Essex trial-wise regression utilities."""

import unittest

import numpy as np

from biomechanical_load_estimation.code.src.essex_regression_baselines import (
    flatten_targets, make_regressor, participant_metrics, regression_metrics,
    subject_balanced_mae, training_group_folds, trial_features,
)


class EssexRegressionBaselineTests(unittest.TestCase):
    def test_feature_schema_and_trial_targets(self):
        x = np.ones((3, 101, 42), dtype=float)
        features, names = trial_features(x, [f"ch{i}" for i in range(42)])
        self.assertEqual(features.shape, (3, 210))
        self.assertEqual(len(names), 210)
        self.assertTrue(np.all(features[:, :42] == 1))
        y = np.ones((3, 11, 2), dtype=float)
        self.assertEqual(flatten_targets(y).shape, (3, 22))
        with self.assertRaises(ValueError):
            flatten_targets(np.zeros((3, 101, 2)))

    def test_grouped_cv_and_metrics(self):
        groups = np.repeat([f"subject{x}" for x in range(10)], 3)
        folds, record = training_group_folds(groups, n_splits=5)
        self.assertEqual(len(folds), 5)
        self.assertEqual(set(record), {1, 2, 3, 4, 5})
        for tr, va in folds:
            self.assertFalse(set(groups[tr]) & set(groups[va]))
        y = np.zeros((len(groups), 11, 2))
        p = np.ones_like(y)
        result = regression_metrics(y, p)
        self.assertAlmostEqual(result[0]["mae"], 1)
        self.assertAlmostEqual(subject_balanced_mae(y, p, groups), 1)
        detail = participant_metrics(y, p, groups, ["Walking"] * len(y),
                                     model="Baseline", phase="TRAIN_CV")
        self.assertEqual(len(detail), 20)

    def test_estimators_are_multioutput_capable(self):
        x = np.arange(210 * 18, dtype=float).reshape(18, 210) / 100
        y = np.arange(18 * 22, dtype=float).reshape(18, 22) / 10
        for name in ("Training Mean", "Ridge", "Random Forest"):
            model = make_regressor(name)
            model.fit(x, y)
            self.assertEqual(model.predict(x[:2]).shape, (2, 22))


if __name__ == "__main__":
    unittest.main()
