"""Offline tests for final HARTH held-out reporting; never access Test recordings."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_modules"))
from core.harth_qualified_test_report import (
    validate_final_test_inputs, summarise_qualified_heldout,
)


class TestQualifiedHeldoutReport(unittest.TestCase):
    def setUp(self):
        self.truth = [1, 2, 1, 2, 1, 2]
        self.people = ["S023"] * 3 + ["S032"] * 3
        self.names = {i: f"Activity {i}" for i in range(1, 9)}
        self.pred = {
            "1D-ResNet Weighted CE": [1, 2, 1, 2, 1, 2],
            "Random Forest": [1, 1, 1, 2, 1, 2],
        }

    def test_fixed_macro_denominator_and_actual_support(self):
        report = summarise_qualified_heldout(
            self.truth, self.people, self.pred, self.names
        )
        main = report["summary"].set_index("model").loc["1D-ResNet Weighted CE"]
        self.assertAlmostEqual(main["fixed_eight_class_macro_f1"], 0.25)
        self.assertAlmostEqual(main["accuracy"], 1)
        self.assertEqual(int(main["true_classes_present"]), 2)
        self.assertEqual(len(report["per_class"]), 16)
        self.assertEqual(int(report["confusion"]["Random Forest"].sum()), 6)
        self.assertEqual(
            report["per_class"].query("source_code == 8")["support"].tolist(),
            [0, 0],
        )

    def test_missing_or_changed_reserved_subjects_blocked(self):
        with self.assertRaises(ValueError):
            validate_final_test_inputs(self.truth, ["S023"] * 6, self.pred)
        with self.assertRaises(ValueError):
            validate_final_test_inputs(
                self.truth, ["S023"] * 3 + ["S030"] * 3, self.pred
            )

    def test_unknown_labels_or_incomplete_model_set_blocked(self):
        with self.assertRaises(ValueError):
            validate_final_test_inputs([9] + self.truth[1:], self.people, self.pred)
        with self.assertRaises(ValueError):
            validate_final_test_inputs(
                self.truth, self.people,
                {"1D-ResNet Weighted CE": self.truth},
            )

    def test_misaligned_predictions_blocked(self):
        corrupted = {**self.pred, "Random Forest": [1, 2]}
        with self.assertRaises(ValueError):
            validate_final_test_inputs(self.truth, self.people, corrupted)


if __name__ == "__main__":
    unittest.main()
