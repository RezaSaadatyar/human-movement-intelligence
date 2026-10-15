"""Source-safe offline tests for HARTH Stage 26 frozen inference gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/python/src"))
try:
    from core import harth_final_inference as final
except ImportError:
    final=None


@unittest.skipIf(final is None,"Optional stage dependencies unavailable")
class TestHarthFinalInference(unittest.TestCase):
    def test_development_scaler_no_test_fit(self):
        training=np.ones((5,6,100),dtype=np.float32)*3.0
        scaler=final.fit_development_scaler(training)
        held_out=np.ones((2,6,100),dtype=np.float32)*8.0
        transformed=final.apply_frozen_scaler(held_out,scaler)
        self.assertEqual(transformed.shape,held_out.shape)
        self.assertAlmostEqual(float(transformed.mean()),5.0,places=4)
        with self.assertRaises(ValueError):
            final.apply_frozen_scaler(np.ones((2,5,100)),scaler)

    def test_explicit_approval_gate(self):
        with self.assertRaises(PermissionError):
            final.enforce_qualified_test_approval(
                {"status":"PENDING_USER_REVIEW"},
                final.TEST_CONFIRMATION
            )
        with self.assertRaises(PermissionError):
            final.enforce_qualified_test_approval(
                {"status":"APPROVED"},""
            )
        self.assertTrue(final.enforce_qualified_test_approval(
            {"status":"APPROVED"},final.TEST_CONFIRMATION
        ))

    def test_current_proposal_remains_unapproved(self):
        root=Path(__file__).resolve().parents[1]
        proposal=root/"studies/wearable_activity_recognition/experiments/stage_25_protocol_proposal.json"
        self.assertTrue(proposal.is_file())
        pending=final.load_final_protocol(proposal,root,require_approval=False)
        self.assertEqual(pending["status"],"PENDING_USER_REVIEW")
        with self.assertRaises(PermissionError):
            final.load_final_protocol(proposal,root,require_approval=True)

    def test_qualified_scoring_validates_original_subjects(self):
        with self.assertRaises(ValueError):
            final.score_qualified_heldout(
                [1,2,1,2],[1,1,1,2],["S001"]*4,range(1,9)
            )


if __name__=="__main__":
    unittest.main()
