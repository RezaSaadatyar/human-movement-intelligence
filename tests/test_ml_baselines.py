"""Offline tests for leakage-aware classical ML training utilities."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/python/src"))
from core.ml_baselines import (
    train_validation_classifiers, classification_class_report,
    classification_confusion,
)

class TestMLBaselines(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.xtrain = rng.normal(size=(90, 5))
        self.ytrain = np.tile(np.array([1, 2, 3]), 30)
        self.xval = rng.normal(size=(24, 5))
        self.yval = np.tile(np.array([1, 2, 3]), 8)

    def test_baseline_shapes_and_metrics(self):
        models, scores, predictions = train_validation_classifiers(
            self.xtrain, self.ytrain, self.xval, self.yval, classes=[1, 2, 3]
        )
        self.assertEqual(len(models), 3)
        self.assertEqual(len(scores), 3)
        self.assertTrue(((scores["macro_f1"] >= 0) &
                         (scores["macro_f1"] <= 1)).all())
        for values in predictions.values():
            self.assertEqual(len(values), len(self.yval))
        self.assertEqual(models["Logistic Regression"].named_steps[
            "standardscaler"].n_samples_seen_, len(self.xtrain))

    def test_class_report_and_confusion(self):
        ytrue = np.array([1, 1, 2, 3])
        ypred = np.array([1, 2, 2, 1])
        report = classification_class_report(
            ytrue, ypred, classes=[1, 2, 3],
            class_names=["Walk", "Run", "Stand"]
        )
        self.assertEqual(report["support"].tolist(), [2, 1, 1])
        self.assertEqual(classification_confusion(
            ytrue, ypred, [1, 2, 3]).shape, (3, 3))

    def test_reject_missing_training_class(self):
        with self.assertRaisesRegex(ValueError, "represented in Train"):
            train_validation_classifiers(
                self.xtrain, self.ytrain, self.xval, self.yval, [1, 2, 3, 4]
            )

    def test_reject_nonfinite_input(self):
        broken = self.xtrain.copy()
        broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            train_validation_classifiers(
                broken, self.ytrain, self.xval, self.yval, [1, 2, 3]
            )

    def test_reject_label_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "counts must match"):
            train_validation_classifiers(
                self.xtrain, self.ytrain[:-1], self.xval, self.yval,
                [1, 2, 3]
            )

if __name__ == "__main__":
    unittest.main()
