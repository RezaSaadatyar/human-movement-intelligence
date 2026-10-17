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

    def test_in_place_stage23_formatting_preserves_approved_evidence(self):
        """The readable Notebook 23 must match the archived historic execution."""
        import hashlib
        import sys
        notebooks = ROOT / "studies/wearable_activity_recognition/notebooks"
        study = notebooks.parent
        active = notebooks / "23_unified_matched_model_benchmark.ipynb"
        archive = (study / "experiments/frozen_notebook_evidence" /
                   active.name)
        manifest = json.loads(
            (study / "experiments/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        expected = manifest["stage23_saved_notebook_sha256"]
        self.assertEqual(
            audit.verify_approved_text_bytes(archive, expected), "EXACT"
        )
        self.assertTrue(audit.verify_stage23_visual_format(active, archive))
        self.assertNotEqual(
            hashlib.sha256(active.read_bytes()).hexdigest(), expected
        )
        # Stage 27 reads the approved historic evidence through this loader.
        sys.path.insert(0, str(ROOT / "code/python/src"))
        from core.harth_selection_evidence import load_selection_evidence
        evidence = load_selection_evidence(
            active, notebooks / "24_multiseed_and_imbalance_robustness.ipynb"
        )
        self.assertEqual(evidence["notebook23_sha256"], expected)
        self.assertEqual(len(evidence["stage23"]), 7)

    def test_local_model_artifacts_are_optional_without_deserialisation(self):
        record = audit.inspect_optional_artifacts(None, "unused_sha256")
        self.assertEqual(record["status"], "NOT_CHECKED")

    def test_windows_crlf_equivalence_is_strict_and_read_only(self):
        """CRLF checkout may match the frozen LF content, but code edits must fail."""
        import hashlib
        import tempfile
        canonical = b"def example():\n    return 42\n"
        expected = hashlib.sha256(canonical).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.py"
            path.write_bytes(canonical.replace(b"\n", b"\r\n"))
            before = path.read_bytes()
            self.assertEqual(
                audit.verify_approved_text_bytes(path, expected), "CRLF_TO_LF_ONLY"
            )
            self.assertEqual(path.read_bytes(), before)
            path.write_bytes(canonical)
            self.assertEqual(
                audit.verify_approved_text_bytes(path, expected), "EXACT"
            )
            path.write_bytes(b"def example():\r\n    return 43\r\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit.verify_approved_text_bytes(path, expected)

    def test_stage27_quality_gate_must_exist_in_executed_notebook(self):
        p = ROOT / "studies/wearable_activity_recognition/notebooks/27_final_research_evidence_and_report.ipynb"
        saved = audit.notebook_execution_check(
            p, 10,
            "STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED",
            allow_missing_counts_with_outputs=True,
        )
        self.assertEqual(saved["cells"], 10)
        self.assertEqual(saved["saved_output_cells"], 10)
        self.assertIsInstance(saved["missing_execution_counts"], list)
        self.assertEqual(len(saved["notebook_sha256"]), 64)

    def test_approval_lineage_cannot_be_silently_mutated(self):
        path = ROOT / "studies/wearable_activity_recognition/experiments/stage_25_protocol_approved_qualified_test.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "APPROVED")
        self.assertFalse(manifest["evaluation"]["use_test_for_tuning"])


if __name__ == "__main__":
    unittest.main()
