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
        training=np.broadcast_to(np.arange(100,dtype=np.float32),(5,6,100)).copy()
        scaler=final.fit_development_scaler(training)
        held_out=np.ones((2,6,100),dtype=np.float32)*8.0
        transformed=final.apply_frozen_scaler(held_out,scaler)
        self.assertEqual(transformed.shape,held_out.shape)
        self.assertAlmostEqual(float(transformed.mean()),float((8.0-scaler["mean"][0])/scaler["std"][0]),places=4)
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

    def test_original_and_reconciled_proposals(self):
        root=Path(__file__).resolve().parents[1]
        directory=root/"studies/wearable_activity_recognition/experiments"
        original=directory/"stage_25_protocol_proposal.json"
        revised=directory/"stage_25_protocol_proposal_reconciled.json"
        assert original.is_file() and revised.is_file()
        old=json.loads(original.read_text(encoding="utf-8"))
        new=json.loads(revised.read_text(encoding="utf-8"))
        self.assertEqual(old["status"],"PENDING_USER_REVIEW")
        self.assertEqual(new["status"],"PENDING_USER_REVIEW")
        # Scientific configuration must be identical; only recorded hashes
        # and an explicit explanatory provenance entry may be added.
        self.assertEqual(
            {k:v for k,v in old.items() if k!="module_sha256"},
            {k:v for k,v in new.items() if k not in ("module_sha256","reconciliation")}
        )
        self.assertNotEqual(old["module_sha256"],new["module_sha256"])
        with self.assertRaisesRegex(ValueError,"Frozen code was modified"):
            final.load_final_protocol(original,root,require_approval=False)
        accepted=final.load_final_protocol(revised,root,require_approval=False)
        self.assertEqual(accepted["status"],"PENDING_USER_REVIEW")
        with self.assertRaises(PermissionError):
            final.load_final_protocol(revised,root,require_approval=True)

    def test_stage26_notebook_uses_reconciled_manifest(self):
        root=Path(__file__).resolve().parents[1]
        notebook=root/"studies/wearable_activity_recognition/notebooks/26_final_development_training_and_inference.ipynb"
        nb=json.loads(notebook.read_text(encoding="utf-8"))
        code="\\n".join(
            "".join(cell["source"]) if isinstance(cell.get("source"),list)
            else cell.get("source","")
            for cell in nb["cells"] if cell["cell_type"]=="code"
        )
        self.assertIn("stage_25_protocol_proposal_reconciled.json",code)
        self.assertNotIn('stage_25_protocol_proposal.json"',code)

    def test_qualified_scoring_validates_original_subjects(self):
        with self.assertRaises(ValueError):
            final.score_qualified_heldout(
                [1,2,1,2],[1,1,1,2],["S001"]*4,range(1,9)
            )


if __name__=="__main__":
    unittest.main()
