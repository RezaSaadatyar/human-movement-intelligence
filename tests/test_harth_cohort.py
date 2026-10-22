"""Offline checks for the HARTH multi-person cohort adapter."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_modules"))
sys.path.insert(0, str(ROOT / "wearable_activity_recognition/src"))
import harth_data


class TestHarthCohort(unittest.TestCase):
    def test_reject_duplicate_people(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            harth_data.prepare_harth_feature_cohort(["S023", "S023"], ".")

    def test_reject_undocumented_task_labels(self):
        with self.assertRaisesRegex(ValueError, "officially documented"):
            harth_data.prepare_harth_feature_cohort(["S023"], ".", [1, 10])

    def test_alignment_and_exclusion_without_network(self):
        # Replace source download/loading, but exercise real windowing/features.
        times = pd.date_range("2025-01-01", periods=350, freq="20ms")
        def load_mock(path, allow_unmapped_labels=False):
            subject = Path(path).stem
            values = pd.DataFrame({"timestamp": times})
            for channel in harth_data.SENSOR_COLUMNS:
                values[channel] = np.arange(350, dtype=float) / 100
            # Last 100 samples are excluded only AFTER windowing.
            values["label"] = [1] * 200 + [2] * 50 + [13] * 100
            return values, harth_data.SENSOR_COLUMNS.copy()

        with patch.object(harth_data, "ensure_harth_sample",
                          side_effect=lambda root, subject: Path(root) / f"{subject}.csv"):
            with patch.object(harth_data, "load_harth_subject", side_effect=load_mock):
                X, meta, overview, excluded = harth_data.prepare_harth_feature_cohort(
                    ["S023", "S024"], ".", task_classes=[1, 2]
                )
        self.assertEqual(len(X), len(meta))
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(meta["subject_id"].nunique(), 2)
        self.assertEqual(set(meta["label"]), {1})
        self.assertEqual(overview["excluded_windows"].sum(),
                         excluded["excluded_windows"].sum())
        self.assertTrue((excluded["source_code"] == 13).all())

    def test_raw_window_cohort_executes_with_mocked_source(self):
        # Exercise the new temporal cohort function, including NumPy sampling.
        timestamps = pd.date_range("2025-01-01", periods=350, freq="20ms")
        def load_mock(path, allow_unmapped_labels=False):
            data = pd.DataFrame({"timestamp": timestamps})
            for channel in harth_data.SENSOR_COLUMNS:
                data[channel] = np.arange(350, dtype=float) / 100
            data["label"] = [1] * 200 + [2] * 50 + [13] * 100
            return data, harth_data.SENSOR_COLUMNS.copy()

        with patch.object(
            harth_data, "ensure_harth_sample",
            side_effect=lambda root, subject: Path(root) / f"{subject}.csv",
        ):
            with patch.object(harth_data, "load_harth_subject", side_effect=load_mock):
                windows, labels, metadata, overview = (
                    harth_data.prepare_harth_window_cohort(
                        ["S023", "S024"], ".", task_classes=[1, 2],
                        max_windows_per_subject=2,
                    )
                )
        self.assertEqual(windows.shape, (4, 6, 100))
        self.assertEqual(len(labels), len(metadata))
        self.assertEqual(metadata["subject_id"].nunique(), 2)
        self.assertTrue(np.isfinite(windows).all())
        self.assertTrue((overview["sampled_windows"] == 2).all())


    def test_nested_raw_cohort_preserves_original_base_windows(self):
        # Exercise source adapter end to end, with mocked local CSV contents.
        times = pd.date_range("2025-01-01", periods=600, freq="20ms")
        def load_mock(path, allow_unmapped_labels=False):
            values = pd.DataFrame({"timestamp": times})
            for column in harth_data.SENSOR_COLUMNS:
                values[column] = np.arange(600, dtype=float) / 100
            values["label"] = [1] * 400 + [2] * 100 + [13] * 100
            return values, harth_data.SENSOR_COLUMNS.copy()

        with patch.object(
            harth_data, "ensure_harth_sample",
            side_effect=lambda root, subject: Path(root) / f"{subject}.csv",
        ):
            with patch.object(harth_data, "load_harth_subject", side_effect=load_mock):
                X, metadata, overview = harth_data.prepare_harth_nested_window_cohort(
                    ["S023", "S024"], ".", task_classes=[1, 2],
                    base_cap=2, expanded_cap=4,
                )
        self.assertEqual(len(X), len(metadata))
        self.assertEqual(metadata["in_base_500"].sum(), 4)
        self.assertEqual(len(X), 8)
        self.assertEqual(overview["base_windows"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
