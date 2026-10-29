# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Tests for held-out softmax inference and descriptive confidence audit."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.prediction_reliability import audit_prediction_confidence
try:
    import torch
    from core.temporal_cnn import TemporalCNN,predict_temporal_cnn_probabilities
except ImportError:
    torch=None

# ===================== 01. Prediction Reliability Tests ======================
class TestReliability(unittest.TestCase):
    def test_probabilities_and_metrics(self):
        p=np.array([[.9,.1],[.2,.8],[.95,.05],[.3,.7]])
        d,b,s,m=audit_prediction_confidence([1,2,2,2],p,["S1","S1","S2","S2"],[1,2])
        self.assertEqual(len(d),4)
        self.assertEqual(s["errors"].sum(),1)
        self.assertEqual(m["high_confidence_errors_0_90"],1)
        self.assertEqual(b["windows"].sum(),4)
    def test_reject_unaligned_probability(self):
        with self.assertRaisesRegex(ValueError,"sum to one"):
            audit_prediction_confidence([1,2],[[.8,.8],[.2,.8]],["S1","S2"],[1,2])
    @unittest.skipIf(torch is None,"PyTorch optional")
    def test_inference_shapes(self):
        model=TemporalCNN(in_channels=3,n_classes=2)
        x=np.ones((5,3,20),dtype=np.float32)
        scores=predict_temporal_cnn_probabilities(model,x,{
            "channel_mean":np.zeros(3),"channel_std":np.ones(3)
        },[1,2],batch_size=2)
        self.assertEqual(scores.shape,(5,2))
        np.testing.assert_allclose(scores.sum(axis=1),1,atol=1e-6)
if __name__=="__main__":unittest.main()
