# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline validation of reusable, leakage-aware signal windowing."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "python" / "src"))

from core.signal_preprocessing import create_signal_windows


# ====================== 01. Signal Preprocessing Tests =======================
class TestSignalWindowing(unittest.TestCase):
    def test_channel_first_windows_and_overlap(self):
        signal = np.arange(30, dtype=float).reshape(10, 3)
        labels = np.ones(10, dtype=int)
        X, y, metadata = create_signal_windows(
            signal, labels, window_size=4, step_size=2, subject_id="S023"
        )
        self.assertEqual(X.shape, (4, 3, 4))
        np.testing.assert_array_equal(X[0], signal[:4].T)
        self.assertEqual(metadata["start_index"].tolist(), [0, 2, 4, 6])
        self.assertEqual(set(metadata["subject_id"]), {"S023"})
        np.testing.assert_array_equal(y, np.ones(4, dtype=int))

    def test_transitions_rejected_with_full_purity(self):
        signal = np.ones((10, 2))
        labels = np.array([1] * 5 + [2] * 5)
        X, y, metadata = create_signal_windows(signal, labels, 4, 2)
        self.assertEqual(X.shape, (2, 2, 4))
        self.assertEqual(y.tolist(), [1, 2])
        self.assertTrue((metadata["label_purity"] == 1.0).all())

    def test_tolerant_majority_label_explicit(self):
        signal = np.ones((6, 2))
        labels = np.array([1, 1, 1, 2, 2, 2])
        X, y, metadata = create_signal_windows(
            signal, labels, 4, 2, min_label_purity=0.75
        )
        self.assertEqual(len(X), 2)
        self.assertEqual(y.tolist(), [1, 2])
        self.assertTrue((metadata["label_purity"] == 0.75).all())

    def test_timestamp_gap_not_bridged(self):
        signal = np.ones((8, 2))
        labels = np.ones(8, dtype=int)
        t = pd.to_datetime([
            "2026-01-01 00:00:00.000", "2026-01-01 00:00:00.020",
            "2026-01-01 00:00:00.040", "2026-01-01 00:00:00.060",
            "2026-01-01 00:00:01.000", "2026-01-01 00:00:01.020",
            "2026-01-01 00:00:01.040", "2026-01-01 00:00:01.060",
        ])
        X, _, m = create_signal_windows(
            signal, labels, 4, 2, timestamps=t, sampling_rate=50
        )
        self.assertEqual(X.shape, (2, 2, 4))
        self.assertEqual(m["start_index"].tolist(), [0, 4])

    def test_nonmonotonic_timestamps_are_not_bridged(self):
        signal = np.ones((6, 2))
        labels = np.zeros(6, dtype=int)
        timestamps = pd.to_datetime([
            "2026-01-01 00:00:00.000", "2026-01-01 00:00:00.020",
            "2026-01-01 00:00:00.040", "2026-01-01 00:00:00.040",
            "2026-01-01 00:00:00.060", "2026-01-01 00:00:00.080",
        ])
        X, _, m = create_signal_windows(
            signal, labels, 3, 1, timestamps=timestamps, sampling_rate=50
        )
        self.assertEqual(m["start_index"].tolist(), [0, 3])
        self.assertEqual(len(X), 2)

    def test_invalid_shapes_labels_and_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            create_signal_windows(np.ones((10, 2, 2)), np.ones(10), 4, 2)
        with self.assertRaisesRegex(ValueError, "one value"):
            create_signal_windows(np.ones((10, 2)), np.ones(9), 4, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            signal = np.ones((10, 2))
            signal[1, 1] = np.nan
            create_signal_windows(signal, np.ones(10), 4, 2)

    def test_no_eligible_windows_has_stable_shape(self):
        X, y, metadata = create_signal_windows(
            np.ones((3, 4)), np.ones(3, dtype=int), 10, 5
        )
        self.assertEqual(X.shape, (0, 4, 10))
        self.assertEqual(y.shape, (0,))
        self.assertEqual(len(metadata), 0)

    def test_sampling_rate_required_for_timestamps(self):
        with self.assertRaisesRegex(ValueError, "sampling_rate"):
            create_signal_windows(
                np.ones((6, 2)), np.ones(6), 4, 2,
                timestamps=pd.date_range("2026-01-01", periods=6, freq="20ms")
            )


if __name__ == "__main__":
    unittest.main()
