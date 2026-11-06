# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 07 model shapes, source sensor grouping and statistical safeguards."""

import unittest
import tempfile
from unittest.mock import patch
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Optional PyTorch not installed")
class EssexStage07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            return
        from biomechanical_load_estimation.code.src import (
            essex_stage07_robustness as stage,
        )

        cls.s = stage

    def test_all_four_models_source_native_output(self):
        x = torch.zeros(2, 101, 42)
        idx = torch.tensor(
            [[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]] * 2,
        )
        for name in self.s.ARCHITECTURES:
            with torch.inference_mode():
                y = self.s.make_stage07_model(name)(x, idx)
            self.assertEqual(tuple(y.shape), (2, 11, 2))

    def test_genuine_bidir_and_peak_ablation(self):
        self.assertTrue(self.s.make_stage07_model("BiGRU").bidirectional)
        self.assertFalse(self.s.make_stage07_model("GRU").bidirectional)
        self.assertEqual(self.s.config_peak_weight(self.s.NO_PEAK), 0.0)
        self.assertEqual(self.s.config_peak_weight("Dilated TCN"), 0.25)
        with self.assertRaises(ValueError):
            self.s.make_stage07_model("Transformer")

    def test_physical_sensor_schema_and_coverage(self):
        suffix = self.s.SENSOR_SUFFIXES
        channels = [
            f"{a}_{s}"
            for a in (
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            )
            for s in suffix
        ]
        groups = self.s.sensor_channel_groups(channels)
        self.assertEqual(len(groups), 7)
        self.assertEqual(len(groups["PEL"]), 6)
        self.assertEqual(
            set(sum([list(v) for v in groups.values()], [])), set(range(42))
        )

    def test_paired_participant_bootstrap(self):
        result = self.s.paired_subject_bootstrap(
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            draws=100,
            seed=9,
        )
        self.assertAlmostEqual(result["difference_mean"], 1)
        self.assertEqual(result["bootstrap_participants"], 4)
        self.assertAlmostEqual(result["bootstrap_low_95"], 1)
        self.assertAlmostEqual(result["bootstrap_high_95"], 1)

    def test_training_rejects_participant_overlap(self):
        n = 4
        x = np.zeros((n, 101, 42))
        y = np.zeros((n, 11, 2))
        idx = np.tile(
            [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
            (n, 1),
        )
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            self.s.run_training_fold(
                "/tmp/unused_test_cache",
                config="GRU",
                seed=2026,
                fold=1,
                x=x,
                y=y,
                positions=idx,
                fit_rows=[0, 1],
                held_rows=[2, 3],
                record_keys=["a", "b", "c", "d"],
                participant_ids=["p1", "p2", "p1", "p3"],
                device="cpu",
            )

    def test_rerunning_fold_trains_again_and_replaces_current_output(self):
        x = np.zeros((4, 101, 42))
        y = np.zeros((4, 11, 2))
        positions = np.tile(np.arange(0, 101, 10), (4, 1))
        result = {
            "prediction": np.zeros((2, 11, 2)),
            "history": np.array([1.0]),
            "perturbed": {},
            "parameter_count": 10,
            "fit_trials": 2,
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(self.s, "train_stage07_fold", return_value=result) as train,
        ):
            for value in (1.0, 2.0):
                result["prediction"] = np.full((2, 11, 2), value)
                outcome = self.s.run_training_fold(
                    folder,
                    config="GRU",
                    seed=2026,
                    fold=1,
                    x=x,
                    y=y,
                    positions=positions,
                    fit_rows=[0, 1],
                    held_rows=[2, 3],
                    record_keys=["a", "b", "c", "d"],
                    participant_ids=["p1", "p1", "p2", "p2"],
                    device="cpu",
                )
                with np.load(outcome["cache_file"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(
                        saved["prediction"], result["prediction"]
                    )
                    np.testing.assert_array_equal(saved["held_keys"], ["c", "d"])
            self.assertEqual(train.call_count, 2)


if __name__ == "__main__":
    unittest.main()
