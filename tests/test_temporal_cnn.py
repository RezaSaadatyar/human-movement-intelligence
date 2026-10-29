# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline PyTorch validation of shape, train-only scaling and CNN inference."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
try:
    import torch
    from core.temporal_cnn import (
        TemporalCNN, standardise_imu_train_only, train_temporal_cnn
    )
except ImportError:
    torch = None


# ========================== 01. Temporal CNN Tests ===========================
@unittest.skipIf(torch is None, "Optional PyTorch dependency not installed")
class TestTemporalCNN(unittest.TestCase):
    def test_forward_shape(self):
        net = TemporalCNN(in_channels=6, n_classes=8)
        logits = net(torch.zeros(4, 6, 100))
        self.assertEqual(tuple(logits.shape), (4, 8))

    def test_train_only_scaler(self):
        tr = np.ones((5, 2, 10), dtype=np.float32) * 2
        va = np.ones((2, 2, 10), dtype=np.float32) * 10
        x, v, stats = standardise_imu_train_only(tr, va)
        self.assertEqual(x.shape, tr.shape)
        self.assertTrue(np.allclose(x, 0))
        self.assertTrue(np.all(stats["channel_mean"] == 2))
        self.assertTrue(np.all(v > 1e6))

    def test_tiny_training_returns_metrics(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(40, 3, 20)).astype(np.float32)
        labels = np.tile([1, 2], 20)
        net, history, predicted, stats = train_temporal_cnn(
            X[:30], labels[:30], X[30:], labels[30:],
            classes=[1, 2], epochs=1, batch_size=10, seed=12,
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(len(predicted), 10)
        self.assertTrue(set(predicted).issubset({1, 2}))
        self.assertEqual(len(stats["channel_mean"]), 3)

    def test_reject_unlisted_label(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(12, 2, 10)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "outside"):
            train_temporal_cnn(X[:8], np.array([1, 2]*4),
                               X[8:], np.array([1, 3, 1, 2]),
                               classes=[1, 2], epochs=1)

if __name__=="__main__":
    unittest.main()
