"""Offline safety regression tests for read-only HARTH pre-release audit."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_harth_release_readiness.py"
spec = importlib.util.spec_from_file_location("harth_release_preflight", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TestHarthReleaseReadiness(unittest.TestCase):
    def test_frozen_approval_source_and_executed_evidence(self):
        record = audit.verify_recorded_evidence()
        self.assertEqual(record["provenance"], "VERIFIED")
        self.assertEqual(record["frozen_module_hashes"], 6)
        self.assertEqual(set(record["saved_notebooks"]), {"23", "24", "25", "26", "27"})
        self.assertFalse(record["original_test_reopened_by_this_audit"])

    def test_local_model_artifacts_are_optional_without_deserialisation(self):
        record = audit.inspect_optional_artifacts(None, "unused_sha256")
        self.assertEqual(record["status"], "NOT_CHECKED")

    def test_stage27_quality_gate_must_exist_in_executed_notebook(self):
        p = ROOT / "studies/wearable_activity_recognition/notebooks/27_final_research_evidence_and_report.ipynb"
        saved = audit.notebook_execution_check(
            p, 10,
            "STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED",
        )
        self.assertEqual(saved["cells"], 10)
        self.assertEqual(len(saved["notebook_sha256"]), 64)

    def test_approval_lineage_cannot_be_silently_mutated(self):
        path = ROOT / "studies/wearable_activity_recognition/experiments/stage_25_protocol_approved_qualified_test.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "APPROVED")
        self.assertFalse(manifest["evaluation"]["use_test_for_tuning"])


if __name__ == "__main__":
    unittest.main()
