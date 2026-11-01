# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline source-time and leakage-contract checks for temporal regressors."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is an optional dependency")
class EssexTemporalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            return
        from biomechanical_load_estimation.code.src import (
            essex_temporal_regression as temporal,
        )
        cls.temporal = temporal

    def setUp(self):
        self.indices = np.tile(
            np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]),
            (8, 1),
        )
        rng = np.random.default_rng(2026)
        self.x = rng.normal(size=(8, 101, 42))
        self.y = rng.normal(size=(8, 11, 2)) * 0.001

    def test_native_coordinate_constraints(self):
        x, idx = self.temporal.check_temporal_arrays(
            self.x, self.indices, self.y,
        )
        self.assertEqual(x.shape, (8, 101, 42))
        self.assertEqual(idx.shape, (8, 11))
        with self.assertRaises(ValueError):
            self.temporal.check_temporal_arrays(
                self.x, self.indices.astype(float), self.y,
            )
        wrong = self.indices.copy()
        wrong[:, 4] = wrong[:, 3]
        with self.assertRaises(ValueError):
            self.temporal.check_temporal_arrays(self.x, wrong)

    def test_both_models_predict_only_original_fe_positions(self):
        for name in ("Temporal CNN", "Dilated TCN"):
            model = self.temporal.make_temporal_model(name)
            with torch.inference_mode():
                estimated = model(
                    torch.from_numpy(self.x[:2].astype("float32")),
                    torch.from_numpy(self.indices[:2]),
                )
            self.assertEqual(tuple(estimated.shape), (2, 11, 2))

    def test_training_scalers_ignore_unseen_participants(self):
        prior = self.temporal.training_fold_statistics(
            self.x[:6], self.y[:6],
        )
        new_eval = self.x.copy()
        new_eval[6:] = 1e6
        after = self.temporal.training_fold_statistics(
            new_eval[:6], self.y[:6],
        )
        np.testing.assert_array_equal(prior.input_mean, after.input_mean)

    def test_one_epoch_training_smoke_cpu(self):
        estimate, history, details = self.temporal.fit_temporal_model(
            "Temporal CNN",
            self.x[:6], self.y[:6], self.indices[:6],
            self.x[6:], self.indices[6:],
            seed=42, epochs=1, batch_size=3, device="cpu",
        )
        self.assertEqual(estimate.shape, (2, 11, 2))
        self.assertEqual(len(history), 1)
        self.assertEqual(details["scaling"], "fit_rows_only")
        self.assertTrue(np.isfinite(estimate).all())


if __name__ == "__main__":
    unittest.main()
