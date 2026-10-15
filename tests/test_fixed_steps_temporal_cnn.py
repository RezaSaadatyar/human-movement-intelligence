"""Offline validation for fixed-optimizer-step CNN training."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code/python/src"))
try:
    from core.temporal_cnn import train_temporal_cnn_fixed_steps
except ImportError:
    train_temporal_cnn_fixed_steps = None

@unittest.skipIf(train_temporal_cnn_fixed_steps is None,"Optional PyTorch unavailable")
class TestFixedSteps(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(23)
        self.x=rng.normal(size=(48,3,24)).astype(np.float32)
        self.y=np.tile([1,2,3],16)

    def test_exact_steps_and_checkpoint_reporting(self):
        model,history,pred,stats=train_temporal_cnn_fixed_steps(
            self.x[:36],self.y[:36],self.x[36:],self.y[36:],
            classes=[1,2,3],max_steps=5,eval_every=2,batch_size=8,seed=17)
        self.assertEqual(history["step"].tolist(),[2,4,5])
        self.assertEqual(len(pred),12)
        self.assertEqual(len(stats["channel_mean"]),3)

    def test_same_seed_reproducible(self):
        arguments=dict(classes=[1,2,3],max_steps=3,eval_every=3,batch_size=8,seed=7)
        _,h1,p1,_=train_temporal_cnn_fixed_steps(
            self.x[:36],self.y[:36],self.x[36:],self.y[36:],**arguments)
        _,h2,p2,_=train_temporal_cnn_fixed_steps(
            self.x[:36],self.y[:36],self.x[36:],self.y[36:],**arguments)
        np.testing.assert_array_equal(p1,p2)
        self.assertAlmostEqual(float(h1.iloc[-1]["train_loss_recent"]),
                               float(h2.iloc[-1]["train_loss_recent"]),places=5)

    def test_invalid_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,"max_steps"):
            train_temporal_cnn_fixed_steps(
                self.x[:36],self.y[:36],self.x[36:],self.y[36:],
                classes=[1,2,3],max_steps=0)

if __name__=="__main__": unittest.main()
