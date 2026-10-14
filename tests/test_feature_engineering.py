"""Tests of shared feature extraction for biomedical signal windows."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "python" / "src"))

from core.feature_engineering import AVAILABLE_FEATURES, extract_signal_features


class TestSignalFeatures(unittest.TestCase):
    def test_shape_names_and_preserved_order(self):
        data = np.arange(24, dtype=float).reshape(2, 3, 4)
        features = extract_signal_features(data, channel_names=["ax", "ay", "az"])
        self.assertEqual(features.shape, (2, 15))
        self.assertEqual(features.columns[0], "ax__mean")
        self.assertEqual(features.columns[1], "ax__std")
        self.assertEqual(features.columns[-1], "az__waveform_length")

    def test_values_for_known_signal(self):
        x = np.array([[[0.0, 2.0, 4.0, 6.0]]])
        f = extract_signal_features(x, ["acc"])
        self.assertAlmostEqual(f.loc[0, "acc__mean"], 3.0)
        self.assertAlmostEqual(f.loc[0, "acc__std"], np.sqrt(5.0))
        self.assertAlmostEqual(f.loc[0, "acc__rms"], np.sqrt(14.0))
        self.assertAlmostEqual(f.loc[0, "acc__peak_to_peak"], 6.0)
        self.assertAlmostEqual(f.loc[0, "acc__waveform_length"], 2.0)

    def test_constant_channels_and_selected_methods(self):
        x = np.full((5, 2, 10), -3.0)
        f = extract_signal_features(x, ["x", "y"], methods=["std", "rms"])
        self.assertEqual(f.shape, (5, 4))
        self.assertTrue((f.filter(like="__std") == 0).all().all())
        self.assertTrue((f.filter(like="__rms") == 3).all().all())

    def test_one_channel_and_42_channels(self):
        for n_channels in (1, 42):
            with self.subTest(channels=n_channels):
                x = np.ones((3, n_channels, 101))
                f = extract_signal_features(x)
                self.assertEqual(f.shape, (3, n_channels * len(AVAILABLE_FEATURES)))

    def test_reject_nonfinite_and_bad_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            extract_signal_features(np.ones((10, 100)))
        with self.assertRaisesRegex(ValueError, "at least one"):
            extract_signal_features(np.empty((0, 6, 100)))
        x = np.ones((2, 3, 10))
        x[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            extract_signal_features(x)

    def test_reject_invalid_channel_names(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            extract_signal_features(np.ones((2, 2, 10)), ["x", "x"])
        with self.assertRaisesRegex(ValueError, "per channel"):
            extract_signal_features(np.ones((2, 3, 10)), ["x", "y"])

    def test_reject_invalid_feature_selection(self):
        x = np.ones((2, 3, 10))
        for methods in (["mean", "mean"], [], "mean"):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                extract_signal_features(x, methods=methods)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            extract_signal_features(x, methods=["fourier_magic"])


if __name__ == "__main__":
    unittest.main()
