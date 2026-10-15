"""Synthetic integrity tests for Stage 24 subject-matched robustness analysis."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/python/src"))
try:
    import torch
    from core import robustness_experiments as rob
    from core.temporal_benchmark import make_benchmark_plan
except ImportError:
    rob=None

@unittest.skipIf(rob is None,"Optional PyTorch is not available in standard CI")
class TestStage24Robustness(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.x=rng.normal(size=(72,3,20)).astype(np.float32)
        self.y=np.tile([1,2,3],24)
        self.groups=np.repeat([f"S{i:02d}" for i in range(6)],12)
        self.plan=make_benchmark_plan(
            self.x,self.y,self.groups,classes=[1,2,3],
            n_splits=3,forbidden_subjects=["S98","S99"]
        )

    def test_loss_weights_and_gradients(self):
        weights=rob.calculate_loss_weights(self.y,[1,2,3],"class_balanced")
        self.assertTrue(np.all(np.isfinite(weights)))
        for method in rob.LOSS_METHODS:
            with self.subTest(method=method):
                logits=torch.randn(6,3,requires_grad=True)
                target=torch.tensor([0,1,2,0,1,2])
                w=torch.ones(3)
                loss=rob.calculate_minibatch_loss(logits,target,w,method)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_focal_zero_gamma_equal_cross_entropy(self):
        logits=torch.randn(8,3)
        target=torch.tensor([0,1,2,0,1,2,0,2])
        weights=torch.tensor([0.5,1.,2.])
        focal=rob.calculate_minibatch_loss(
            logits,target,weights,method="focal",focal_gamma=0.0)
        ce=rob.calculate_minibatch_loss(
            logits,target,weights,method="weighted_ce")
        self.assertAlmostEqual(float(focal),float(ce),places=6)

    def test_resumable_results_fingerprint(self):
        def fake_fit(plan, fold, architecture, seed, method, **kwargs):
            validation=next(va for f,_,va in plan["folds"] if f==fold)
            return plan["labels"][validation].copy(),.2,.02,99
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(rob,"_fit_predict_one",side_effect=fake_fit) as mocked:
                first=rob.run_robustness_job(self.plan,"TCN",1,42,"weighted_ce",
                                             directory,device="cpu")
                second=rob.run_robustness_job(self.plan,"TCN",1,42,"weighted_ce",
                                              directory,device="cpu")
                self.assertEqual(mocked.call_count,1)
                self.assertEqual(first["cache_key"],second["cache_key"])
                self.assertEqual(first["source_rows"],second["source_rows"])

    def test_incomplete_results_rejected(self):
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            rob.summarise_robustness_results(
                self.plan,[],[("TCN","weighted_ce",42)]
            )

if __name__=="__main__":
    unittest.main()
