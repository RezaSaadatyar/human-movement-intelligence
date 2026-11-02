# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 07 model shapes, source sensor grouping and statistical safeguards."""

import unittest
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
        channels = [f"{a}_{s}" for a in (
            "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
        ) for s in suffix]
        groups = self.s.sensor_channel_groups(channels)
        self.assertEqual(len(groups), 7)
        self.assertEqual(len(groups["PEL"]), 6)
        self.assertEqual(set(sum([list(v) for v in groups.values()], [])),
                         set(range(42)))

    def test_paired_participant_bootstrap(self):
        result = self.s.paired_subject_bootstrap(
            [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0],
            draws=100, seed=9,
        )
        self.assertAlmostEqual(result["difference_mean"], 1)
        self.assertEqual(result["bootstrap_participants"], 4)
        self.assertAlmostEqual(result["bootstrap_low_95"], 1)
        self.assertAlmostEqual(result["bootstrap_high_95"], 1)

    def test_caching_rejects_participant_overlap(self):
        n = 4
        x = np.zeros((n, 101, 42))
        y = np.zeros((n, 11, 2))
        idx = np.tile(
            [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], (n, 1),
        )
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            self.s.run_cached_fold(
                "/tmp/unused_test_cache", config="GRU", seed=2026,
                fold=1, x=x, y=y, positions=idx,
                fit_rows=[0, 1], held_rows=[2, 3],
                record_keys=["a", "b", "c", "d"],
                participant_ids=["p1", "p2", "p1", "p3"],
                provenance={"source": "test"}, device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
