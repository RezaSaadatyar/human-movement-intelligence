"""Offline report-integrity tests: never load HARTH source Test recordings."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/python/src"))
from core.harth_selection_evidence import load_selection_evidence
from core.harth_study_report import load_final_harth_evidence, _load_executed_notebook

class TestHarthStage27Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=ROOT/"studies/wearable_activity_recognition"
        cls.dir=cls.study/"notebooks"

    def test_original_executed_and_approved_chain(self):
        development=load_selection_evidence(
            self.dir/"23_unified_matched_model_benchmark.ipynb",
            self.dir/"24_multiseed_and_imbalance_robustness.ipynb",
        )
        final=load_final_harth_evidence(
            self.dir/"26_final_development_training_and_inference.ipynb",
            self.study/"experiments/stage_25_protocol_proposal_reconciled.json",
            self.study/"experiments/stage_25_protocol_approved_qualified_test.json",
            development["development_sha256"],
        )
        self.assertEqual(len(development["stage23"]),7)
        self.assertEqual(len(development["stage24_folds"]),36)
        self.assertEqual(len(final["per_class"]),16)
        self.assertEqual(len(final["per_subject"]),4)
        self.assertEqual(final["coverage"]["sampled_windows"].sum(),1000)
        self.assertEqual(set(final["coverage"]["subject_id"]),{"S023","S032"})
        self.assertEqual(final["approved_manifest"]["stage23_saved_notebook_sha256"],
                         development["notebook23_sha256"])
        self.assertEqual(final["approved_manifest"]["stage24_saved_notebook_sha256"],
                         development["notebook24_sha256"])

    def test_stage25_quality_gate(self):
        notebook,digest=_load_executed_notebook(
            self.dir/"25_model_selection_and_protocol_freeze.ipynb",
            "Stage 25 development selection audit: PASSED",
        )
        self.assertEqual(len(digest),64)
        self.assertTrue(notebook["cells"])

    def test_read_only_reporting_notebook_is_unexecuted_template(self):
        import json
        nb=json.loads(
            (self.dir/"27_final_research_evidence_and_report.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(nb["nbformat"],4)
        codes=[c for c in nb["cells"] if c["cell_type"]=="code"]
        self.assertEqual(len(codes),10)
        self.assertTrue(all(not c["outputs"] for c in codes))
        self.assertNotIn("prepare_harth_window_cohort(",
                         "\n".join("".join(c["source"]) for c in codes))

if __name__=="__main__":
    unittest.main()
