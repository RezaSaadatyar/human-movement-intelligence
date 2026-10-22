"""Offline report-integrity tests: never load HARTH source Test recordings."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python_modules"))
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
            self.study/"evidence/stage_25_protocol_proposal_reconciled.json",
            self.study/"evidence/stage_25_protocol_approved_qualified_test.json",
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

    def test_stage27_saved_report_is_executed_and_read_only(self):
        """Executed notebook outputs are evidence; a draft-only assertion is obsolete."""
        import json
        path = self.dir / "27_final_research_evidence_and_report.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        codes = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(codes), 11)  # one safe Colab setup + ten research cells
        bootstrap = "".join(codes[0]["source"])
        self.assertIn('["git", "-C", str(repo), "pull", "--ff-only", "origin", "main"]', bootstrap)
        research_codes = codes[1:]
        self.assertEqual(len(research_codes), 10)
        missing_counts = [
            i + 1 for i, cell in enumerate(codes)
            if cell.get("execution_count") is None
        ]
        # Preserve missing metadata; do not invent execution counters.
        self.assertTrue(all(cell.get("outputs") for cell in codes))
        self.assertTrue(all(codes[i - 1].get("outputs") for i in missing_counts))
        self.assertFalse(any(
            out.get("output_type") == "error"
            for cell in codes for out in cell.get("outputs", [])
        ))
        last_stream = "\\n".join(
            "".join(out.get("text", []))
            if isinstance(out.get("text"), list) else out.get("text", "")
            for out in codes[-1].get("outputs", [])
            if out.get("output_type") == "stream"
        )
        self.assertIn(
            "STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED",
            last_stream,
        )
        source = "\\n".join(
            "".join(cell["source"]) if isinstance(cell.get("source"), list)
            else cell.get("source", "")
            for cell in codes
        )
        for prohibited in (
            "prepare_harth_window_cohort(",
            "ensure_harth_sample(",
            "fit_final_resnet(",
            "score_qualified_heldout(",
            "infer_source_windows(",
        ):
            self.assertNotIn(prohibited, source)
        # This test validates saved research evidence, never re-executes Test data.

if __name__=="__main__":
    unittest.main()
