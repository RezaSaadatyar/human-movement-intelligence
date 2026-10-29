# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Tests for matched participant-level classification evaluation."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.ml_baselines import evaluate_predictions_by_subject
from core.group_cross_validation import compare_models_by_subject

# ======================= 01. Matched Evaluation Tests ========================
class TestMatchedEvaluation(unittest.TestCase):
    def test_subject_alignment_and_comparison(self):
        y=np.array([1,2,1,2,1,2,1,2])
        subjects=np.array(["S01"]*4+["S02"]*4)
        cnn=evaluate_predictions_by_subject(
            y,np.array([1,2,1,1,1,2,2,2]),subjects,[1,2],"CNN")
        rf=evaluate_predictions_by_subject(
            y,np.array([1,1,1,2,1,2,1,2]),subjects,[1,2],"Random Forest")
        self.assertEqual(set(cnn.subject_id),{"S01","S02"})
        self.assertEqual(cnn.windows.sum(),8)
        paired,summary=compare_models_by_subject(
            __import__("pandas").concat([cnn,rf],ignore_index=True),
            "CNN","Random Forest"
        )
        self.assertEqual(len(paired),2)
        self.assertEqual(summary["participants"],2)

    def test_case_insensitive_subject_ids(self):
        x=evaluate_predictions_by_subject(
            [1,2,1,2],[1,2,1,2],["S01","s01","S02","s02"],
            [1,2],"Model"
        )
        self.assertEqual(x.shape[0],2)

    def test_reject_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError,"aligned"):
            evaluate_predictions_by_subject([1,2],[1],["S01","S01"],[1,2],"X")

    def test_reject_undeclared_prediction(self):
        with self.assertRaisesRegex(ValueError,"declared"):
            evaluate_predictions_by_subject([1,2],[1,99],["S01","S02"],[1,2],"X")
if __name__=="__main__": unittest.main()
