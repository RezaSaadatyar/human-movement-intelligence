# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Synthetic tests of class-aligned temperature scaling."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.temperature_calibration import apply_temperature, fit_temperature, multiclass_nll

# ===================== 01. Temperature Calibration Tests =====================
class TestTemperatureCalibration(unittest.TestCase):
    def test_identity_and_argmax(self):
        p=np.array([[.8,.1,.1],[.1,.7,.2],[.2,.2,.6]])
        np.testing.assert_allclose(apply_temperature(p,1),p,atol=1e-10)
        for T in (0.4,2.5):
            q=apply_temperature(p,T)
            np.testing.assert_array_equal(p.argmax(axis=1),q.argmax(axis=1))
            np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-10)

    def test_fit_calibration_only(self):
        probabilities=np.tile([[.85,.15],[.1,.9],[.7,.3],[.2,.8]],(8,1))
        labels=np.tile([1,2,2,1],8)
        fit=fit_temperature(probabilities,labels,[1,2])
        self.assertTrue(.25<=fit["temperature"]<=4)
        self.assertLessEqual(fit["calibration_nll_after"],fit["calibration_nll_before"]+1e-7)
        self.assertEqual(fit["calibration_windows"],32)

    def test_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError,"sum to one"):
            apply_temperature([[.7,.7]],2)
        with self.assertRaisesRegex(ValueError,"positive"):
            apply_temperature([[.6,.4]],0)
        with self.assertRaisesRegex(ValueError,"align"):
            multiclass_nll([1],[[.5,.5],[.4,.6]],[1,2])

if __name__=="__main__":
    unittest.main()
