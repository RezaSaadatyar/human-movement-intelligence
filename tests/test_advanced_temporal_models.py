"""Offline tensor-shape, gradient and validation tests for four temporal models."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code/python/src"))
from core.advanced_temporal_models import (
    ARCHITECTURES, build_temporal_model, count_trainable_parameters,
    TimeSeriesTransformer,
)

class TestAdvancedTemporalModels(unittest.TestCase):
    def test_shapes_finite_logits_and_gradients(self):
        torch.manual_seed(42)
        for name in ARCHITECTURES:
            with self.subTest(architecture=name):
                model=build_temporal_model(name, in_channels=6, n_classes=8)
                x=torch.randn(3,6,100)
                logits=model(x)
                self.assertEqual(tuple(logits.shape),(3,8))
                self.assertTrue(torch.isfinite(logits).all())
                loss=nn.CrossEntropyLoss()(logits,torch.tensor([0,3,7]))
                loss.backward()
                self.assertTrue(any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                ))
                self.assertGreater(count_trainable_parameters(model),0)

    def test_shorter_valid_window_and_batch_one(self):
        for name in ARCHITECTURES:
            with self.subTest(architecture=name):
                model=build_temporal_model(name,in_channels=3,n_classes=2)
                model.eval()
                with torch.no_grad():
                    out=model(torch.randn(1,3,32))
                self.assertEqual(tuple(out.shape),(1,2))

    def test_reject_wrong_input_contract(self):
        model=build_temporal_model("TCN")
        with self.assertRaisesRegex(ValueError,"Expected"):
            model(torch.randn(2,5,100))
        with self.assertRaisesRegex(ValueError,"Unknown model"):
            build_temporal_model("not-a-model")

    def test_transformer_length_guard(self):
        model=TimeSeriesTransformer(max_length=32)
        with self.assertRaisesRegex(ValueError,"exceeds"):
            model(torch.randn(2,6,33))
        with self.assertRaisesRegex(ValueError,"divisible"):
            TimeSeriesTransformer(d_model=30,n_heads=4)

    def test_one_optimizer_step_for_every_architecture(self):
        # This checks learning mechanics, not scientific classification quality.
        torch.manual_seed(42)
        for name in ARCHITECTURES:
            with self.subTest(architecture=name):
                model=build_temporal_model(name,in_channels=3,n_classes=3)
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
                x=torch.randn(4,3,32)
                y=torch.tensor([0,1,2,1])
                before=[p.detach().clone() for p in model.parameters()]
                optimizer.zero_grad(set_to_none=True)
                nn.CrossEntropyLoss()(model(x),y).backward()
                optimizer.step()
                self.assertTrue(any(
                    not torch.equal(old,p.detach())
                    for old,p in zip(before,model.parameters())
                ))

if __name__=="__main__": unittest.main()
