# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 08 frozen-model and explicit release safeguards."""

import json
import tempfile
import unittest
from pathlib import Path

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

    def test_test_release_requires_explicit_phrase_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.joinpath("output").mkdir()
            protocol = {"manual_release": {"token": self.m.RELEASE_PHRASE}}
            settings = dict(
                protocol=protocol, protocol_sha256="f" * 64,
                output_dir=root / "output", ledger_dir=root / "ignored",
            )
            with self.assertRaises(PermissionError):
                self.m.release_one_time_test(approval="", **settings)
            receipt = self.m.release_one_time_test(
                approval=self.m.RELEASE_PHRASE, **settings
            )
            self.assertEqual(receipt["status"], "TEST_RELEASE_CONSUMED")
            with self.assertRaises(RuntimeError):
                self.m.release_one_time_test(
                    approval=self.m.RELEASE_PHRASE, **settings
                )

    def test_release_rejects_prior_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "output"
            out.mkdir()
            (out / "08_test_trial_predictions.csv").write_text("exists")
            with self.assertRaises(FileExistsError):
                self.m.release_one_time_test(
                    approval=self.m.RELEASE_PHRASE,
                    protocol={"manual_release": {"token": self.m.RELEASE_PHRASE}},
                    protocol_sha256="a" * 64, output_dir=out,
                    ledger_dir=root / "cache",
                )

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
