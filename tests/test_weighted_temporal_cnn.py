"""Offline tests for training-only class-weighted temporal CNN experiments."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/python/src"))
try:
    from core.temporal_cnn import compute_training_class_weights, train_temporal_cnn
except ImportError:
    compute_training_class_weights = None

@unittest.skipIf(compute_training_class_weights is None, "Optional PyTorch unavailable")
class TestWeightedCNN(unittest.TestCase):
    def test_weight_order_and_minority_priority(self):
        labels = np.array([1]*18 + [2]*2)
        weights = compute_training_class_weights(labels, [1,2])
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(weights.mean()),1.0,places=6)

    def test_uniform_unweighted(self):
        weights = compute_training_class_weights([1,1,2],[1,2],method="none")
        np.testing.assert_array_equal(weights,[1,1])

    def test_missing_class_rejected(self):
        with self.assertRaisesRegex(ValueError,"present in Train"):
            compute_training_class_weights([1,1,1],[1,2])

    def test_deterministic_one_epoch_weighted(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(32,3,24)).astype("float32")
        y = np.array([1]*12+[2]*12+[1]*4+[2]*4)
        _, h1, p1, _ = train_temporal_cnn(
            X[:24],y[:24],X[24:],y[24:],
            classes=[1,2],epochs=1,batch_size=8,seed=7,
            class_weighting="inverse_sqrt"
        )
        _, h2, p2, _ = train_temporal_cnn(
            X[:24],y[:24],X[24:],y[24:],
            classes=[1,2],epochs=1,batch_size=8,seed=7,
            class_weighting="inverse_sqrt"
        )
        np.testing.assert_array_equal(p1,p2)
        self.assertAlmostEqual(float(h1.iloc[0]["train_loss"]),
                               float(h2.iloc[0]["train_loss"]),places=5)

if __name__=="__main__":
    unittest.main()
