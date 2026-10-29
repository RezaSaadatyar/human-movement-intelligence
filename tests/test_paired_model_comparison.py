# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline tests for paired participant-level model comparison."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.group_cross_validation import compare_models_by_subject

# ===================== 01. Paired Model Comparison Tests =====================
class TestPairedSubjectComparison(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([
            {"model":"A","subject_id":"S01","macro_f1_fixed_task":0.8},
            {"model":"A","subject_id":"S02","macro_f1_fixed_task":0.4},
            {"model":"B","subject_id":"S01","macro_f1_fixed_task":0.6},
            {"model":"B","subject_id":"S02","macro_f1_fixed_task":0.5},
        ])
    def test_pairing_and_difference(self):
        paired,summary=compare_models_by_subject(self.frame,"A","B")
        self.assertEqual(len(paired),2)
        self.assertAlmostEqual(summary["mean_difference"],0.05)
        self.assertEqual(summary["model_a_wins"],1)
        self.assertEqual(summary["model_b_wins"],1)
    def test_reject_different_subjects(self):
        bad=self.frame.copy()
        bad.loc[3,"subject_id"]="S03"
        with self.assertRaisesRegex(ValueError,"identical subjects"):
            compare_models_by_subject(bad,"A","B")
    def test_reject_duplicate_subject_scores(self):
        bad=pd.concat([self.frame,self.frame.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"one score"):
            compare_models_by_subject(bad,"A","B")
    def test_reject_same_model(self):
        with self.assertRaises(ValueError):
            compare_models_by_subject(self.frame,"A","A")
if __name__=="__main__": unittest.main()
