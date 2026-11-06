# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 08 model settings, source-grid predictions and participant metrics."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is an optional dependency")
class FrozenFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            return
        from biomechanical_load_estimation.code.src import essex_stage08_final as m
        cls.m = m

    def test_final_settings_keep_the_original_model_recipes(self):
        first = self.m.final_model_settings()
        second = self.m.final_model_settings()
        self.assertEqual(first["primary_model"]["n_estimators"], 200)
        self.assertEqual(first["secondary_model"]["seed"], 2128)
        self.assertEqual(first["secondary_model"]["epochs"], 35)
        first["secondary_model"]["epochs"] = 1
        self.assertEqual(second["secondary_model"]["epochs"], 35)

    def test_cluster_bootstrap_is_participant_not_trial_resampling(self):
        y = np.zeros((12, 11, 2))
        p = np.ones_like(y)
        ids = np.repeat(["a", "b", "c", "d"], 3)
        result = self.m.participant_bootstrap(y, p, ids, draws=50)
        self.assertEqual(result["participants"], 4)
        self.assertAlmostEqual(result["estimate"], 1)
        self.assertAlmostEqual(result["bootstrap_low_95"], 1)

    def test_primary_feature_and_output_shapes(self):
        model = self.m
        x = np.ones((3, 101, 42))
        channel_names = [f"ch{i}" for i in range(42)]
        from sklearn.dummy import DummyRegressor
        estimator = DummyRegressor(strategy="mean")
        estimator.fit(np.ones((5, 210)), np.zeros((5, 22)))
        p = model.predict_frozen_primary(
            estimator, x, channel_names,
            [f"{stat}_{c}" for stat in (
                "mean", "std", "minimum", "maximum", "rms"
            ) for c in channel_names],
        )
        self.assertEqual(p.shape, (3, 11, 2))

    def test_primary_fit_preserves_targets_and_rejects_unaligned_trials(self):
        x = np.ones((3, 101, 42))
        y = np.full((3, 11, 2), 0.25)
        channels = [f"ch{i}" for i in range(42)]
        settings = self.m.final_model_settings()
        model, names = self.m.fit_frozen_primary(x, y, channels, settings)
        predicted = self.m.predict_frozen_primary(model, x, channels, names)
        np.testing.assert_allclose(predicted, y)
        with self.assertRaisesRegex(ValueError, "same trials"):
            self.m.fit_frozen_primary(x, y[:2], channels, settings)

    def test_frozen_tcn_native_grid(self):
        from biomechanical_load_estimation.code.src.essex_temporal_regression import (
            make_temporal_model, training_fold_statistics,
        )
        model = make_temporal_model("Dilated TCN")
        x = np.zeros((4, 101, 42))
        y = np.zeros((4, 11, 2))
        original = np.tile(
            [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
            (4, 1),
        )
        scaler = training_fold_statistics(x, y)
        result = self.m.predict_frozen_secondary(
            model, scaler, x[:2], original[:2], device="cpu"
        )
        self.assertEqual(result.shape, (2, 11, 2))


if __name__ == "__main__":
    unittest.main()
