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
        """Readable Stage 23 must match hash-approved Git history."""
        import hashlib
        import sys
        notebooks = ROOT / "wearable_activity_recognition/notebooks"
        study = notebooks.parent
        active = notebooks / "23_unified_matched_model_benchmark.ipynb"
        manifest = json.loads(
            (study / "evidence/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        expected = manifest["stage23_saved_notebook_sha256"]
        source_bytes = audit.read_historical_stage23_bytes(ROOT, expected)
        self.assertEqual(hashlib.sha256(source_bytes).hexdigest(), expected)
        self.assertTrue(audit.verify_stage23_visual_format(active, source_bytes))
        self.assertNotEqual(
            hashlib.sha256(active.read_bytes()).hexdigest(), expected
        )
        # Stage 27 reads the approved historic evidence through this loader.
        sys.path.insert(0, str(ROOT / "python_modules"))
        from core.harth_selection_evidence import load_selection_evidence
        evidence = load_selection_evidence(
            active, notebooks / "24_multiseed_and_imbalance_robustness.ipynb"
        )
        self.assertEqual(evidence["notebook23_sha256"], expected)
        self.assertEqual(len(evidence["stage23"]), 7)

    def test_stage23_allows_blank_stream_lines_only(self):
        """Presentation-only empty lines cannot invalidate saved Stage 23."""
        import tempfile
        from core.harth_provenance import verify_stage23_visual_format
        directory = ROOT / "studies/wearable_activity_recognition"
        active = directory / "notebooks/23_unified_matched_model_benchmark.ipynb"
        approved = json.loads(
            (directory / "evidence/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        historical = audit.read_historical_stage23_bytes(
            ROOT, approved["stage23_saved_notebook_sha256"]
        )
        current = json.loads(active.read_text(encoding="utf-8"))
        cells = [c for c in current["cells"] if c.get("cell_type") == "code"]
        printed = next(
            output for output in cells[4]["outputs"]
            if output.get("output_type") == "stream"
        )
        # Add another empty line without changing any scored or cohort content.
        original = printed["text"]
        text = "".join(original) if isinstance(original, list) else original
        printed["text"] = text.replace(
            "ALL 21 SOURCE-MATCHED EXPERIMENT JOBS COMPLETED",
            "\nALL 21 SOURCE-MATCHED EXPERIMENT JOBS COMPLETED",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checked = Path(tmp) / "stage23_formatting_only.ipynb"
            checked.write_text(json.dumps(current), encoding="utf-8")
            self.assertTrue(verify_stage23_visual_format(checked, historical))

    def test_stage23_rejects_changed_predictive_metric(self):
        """Replays may vary in runtime, but not in frozen scientific scores."""
        import tempfile
        from core.harth_provenance import verify_stage23_visual_format
        folder = ROOT/"studies/wearable_activity_recognition"
        current = folder/"notebooks/23_unified_matched_model_benchmark.ipynb"
        approved = json.loads(
            (folder/"evidence/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        historical = audit.read_historical_stage23_bytes(
            ROOT, approved["stage23_saved_notebook_sha256"]
        )
        notebook = json.loads(current.read_text(encoding="utf-8"))
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        displayed = code[5]["outputs"][1]
        html_part = displayed["data"]["text/html"]
        markup = "".join(html_part) if isinstance(html_part,list) else html_part
        self.assertIn("0.8328", markup)
        displayed["data"]["text/html"] = [markup.replace("0.8328","0.9999",1)]
        with tempfile.TemporaryDirectory() as tmp:
            corrupted = Path(tmp)/"stage23_corrupted.ipynb"
            corrupted.write_text(json.dumps(notebook),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"predictive displayed values"):
                verify_stage23_visual_format(corrupted,historical)

    def test_stage24_current_presentation_matches_approved_history(self):
        """Compare 12 readable study cells with 13 original Colab code cells."""
        root = ROOT
        from core.harth_provenance import (
            read_historical_stage24_bytes, verify_stage24_visual_format,
        )
        current = (
            root / "wearable_activity_recognition/code/notebooks/"
            "24_multiseed_and_imbalance_robustness.ipynb"
        )
        manifest = json.loads(
            (root / "wearable_activity_recognition/evidence/"
             "stage_25_protocol_approved_qualified_test.json").read_text(
                encoding="utf-8"
            )
        )
        original = read_historical_stage24_bytes(
            root, manifest["stage24_saved_notebook_sha256"]
        )
        result = verify_stage24_visual_format(current, original)
        self.assertEqual(result["current_code_cells"], 12)
        self.assertEqual(result["historical_code_cells"], 13)
        # Current GitHub Stage 24 is a real replay with all 12 counts restored.
        # An older pre-replay checkout may retain the two documented gaps.
        self.assertEqual(result["missing_execution_counts"], [])
        self.assertTrue(result["execution_metadata_complete"])
        self.assertTrue(result["historical_execution_verified"])

    def test_stage24_rejects_modified_code_outputs_or_metadata(self):
        """Never mask changes to original executed scientific results."""
        import tempfile
        from core.harth_provenance import (
            read_historical_stage24_bytes, verify_stage24_visual_format,
        )
        current = (
            ROOT / "wearable_activity_recognition/code/notebooks/"
            "24_multiseed_and_imbalance_robustness.ipynb"
        )
        approved = json.loads(
            (ROOT / "wearable_activity_recognition/evidence/"
             "stage_25_protocol_approved_qualified_test.json").read_text(
                encoding="utf-8"
            )
        )
        historical = read_historical_stage24_bytes(
            ROOT, approved["stage24_saved_notebook_sha256"]
        )
        baseline = json.loads(current.read_text(encoding="utf-8"))
        positions = [
            i for i, cell in enumerate(baseline["cells"])
            if cell.get("cell_type") == "code"
        ]
        for change in ("code", "outputs", "metadata"):
            notebook = json.loads(json.dumps(baseline))
            first = notebook["cells"][positions[0]]
            if change == "code":
                first["source"].append("print('altered executable statement')\n")
            elif change == "outputs":
                first["outputs"] = []
            else:
                first["execution_count"] = 999
            with tempfile.TemporaryDirectory() as tmp:
                trial = Path(tmp) / "stage24_check.ipynb"
                trial.write_text(json.dumps(notebook), encoding="utf-8")
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        verify_stage24_visual_format(trial, historical)

    def test_stage24_rejects_modified_scientific_table(self):
        """Runtime variation is allowed, a changed predictive metric is not."""
        import tempfile
        from core.harth_provenance import (
            read_historical_stage24_bytes, verify_stage24_visual_format,
        )
        current = (
            ROOT / "wearable_activity_recognition/code/notebooks/"
            "24_multiseed_and_imbalance_robustness.ipynb"
        )
        approved = json.loads(
            (ROOT / "wearable_activity_recognition/evidence/"
             "stage_25_protocol_approved_qualified_test.json").read_text(
                encoding="utf-8"
            )
        )
        original = read_historical_stage24_bytes(
            ROOT, approved["stage24_saved_notebook_sha256"]
        )
        current_nb = json.loads(current.read_text(encoding="utf-8"))
        cells = [c for c in current_nb["cells"] if c.get("cell_type") == "code"]
        result = cells[10]["outputs"][0]
        html_rows = result["data"]["text/html"]
        html_text = "".join(html_rows) if isinstance(html_rows, list) else html_rows
        assert "0.8286" in html_text
        modified = html_text.replace("0.8286", "0.9999", 1)
        result["data"]["text/html"] = [modified]
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder)/"stage24_modified.ipynb"
            bad.write_text(json.dumps(current_nb), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, r"cell 11, output 1: row 1, column "
                            r"'mean_across_seed_fold_macro_f1'"
            ):
                verify_stage24_visual_format(bad, original)

    def test_stage24_multiindex_rowspan_metric_change_is_rejected(self):
        """A folded MultiIndex seed column must not hide a changed score."""
        import tempfile
        from core.harth_provenance import (
            read_historical_stage24_bytes, verify_stage24_visual_format,
        )
        folder = ROOT / "studies/wearable_activity_recognition"
        current = folder / "notebooks/24_multiseed_and_imbalance_robustness.ipynb"
        approved = json.loads(
            (folder / "evidence/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        historical = read_historical_stage24_bytes(
            ROOT, approved["stage24_saved_notebook_sha256"]
        )
        notebook = json.loads(current.read_text(encoding="utf-8"))
        code = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
        pivot = code[6]["outputs"][0]
        original_html = pivot["data"]["text/html"]
        text = "".join(original_html) if isinstance(original_html, list) else original_html
        self.assertIn('rowspan="3"', text)
        self.assertIn(">0.8360<", text)
        pivot["data"]["text/html"] = [
            text.replace(">0.8360<", ">0.9999<", 1)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            modified = Path(tmp) / "stage24_pivot_changed.ipynb"
            modified.write_text(json.dumps(notebook), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, r"cell 7, output 1: row 2, column"
            ):
                verify_stage24_visual_format(modified, historical)

    def test_stage24_restored_cache_logs_are_checked_against_original_jobs(self):
        """Allow only cache restoration of original fully scored experiments."""
        import tempfile
        from core.harth_provenance import (
            read_historical_stage24_bytes, verify_stage24_visual_format,
        )
        study = ROOT / "studies/wearable_activity_recognition"
        current = study / "notebooks/24_multiseed_and_imbalance_robustness.ipynb"
        manifest = json.loads(
            (study / "evidence/stage_25_protocol_approved_qualified_test.json")
            .read_text(encoding="utf-8")
        )
        original = read_historical_stage24_bytes(
            ROOT, manifest["stage24_saved_notebook_sha256"]
        )
        # Passing verifies all original source statements, metric tables
        # and the actual RESTORED job identifiers, not just the log format.
        self.assertTrue(verify_stage24_visual_format(current, original)[
            "historical_execution_verified"
        ])
        baseline = json.loads(current.read_text(encoding="utf-8"))
        cells = [c for c in baseline["cells"] if c["cell_type"] == "code"]
        stream = cells[4]["outputs"][0]
        raw = stream["text"]
        visible = "".join(raw) if isinstance(raw, list) else raw
        self.assertIn("RESTORED: 1D-ResNet | weighted_ce | seed=42 | fold=1", visible)
        self.assertIn("COMPLETED:", visible)
        for label, corrupted_text in (
            ("unknown_cached_fold", visible.replace(
                "RESTORED: 1D-ResNet | weighted_ce | seed=42 | fold=1",
                "RESTORED: 1D-ResNet | weighted_ce | seed=42 | fold=9", 1
            )),
            ("changed_scored_job", visible.replace(
                "Macro F1=0.7687", "Macro F1=0.9999", 1
            )),
        ):
            notebook = json.loads(json.dumps(baseline))
            code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            code[4]["outputs"][0]["text"] = [corrupted_text]
            with tempfile.TemporaryDirectory() as tmp:
                modified = Path(tmp) / "stage24_modified_cache_log.ipynb"
                modified.write_text(json.dumps(notebook), encoding="utf-8")
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        ValueError, "Stage 24 saved stream differs in cell 5"
                    ):
                        verify_stage24_visual_format(modified, original)

    def test_stage27_has_one_colab_setup_and_ten_research_code_cells(self):
        """Structural check that does not require Stage 27 to have run yet."""
        import json
        notebook = (
            ROOT / "wearable_activity_recognition/code/notebooks/"
            "27_final_research_evidence_and_report.ipynb"
        )
        record = json.loads(notebook.read_text(encoding="utf-8"))
        cells = [
            c for c in record["cells"] if c.get("cell_type") == "code"
        ]
        self.assertEqual(len(cells), 11)
        setup = "".join(cells[0].get("source", []))
        self.assertIn('"pull", "--ff-only", "origin", "main"', setup)
        sections = [
            "".join(c.get("source", []))
            for c in cells[1:]
        ]
        self.assertEqual(len(sections), 10)
        for index, source in enumerate(sections, start=1):
            self.assertIn(f"{index:02d}.", source.splitlines()[0])

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
        """All 11 saved code cells must be genuinely executed and error-free."""
        p = (
            ROOT / "wearable_activity_recognition/code/notebooks/"
            "27_final_research_evidence_and_report.ipynb"
        )
        saved = audit.notebook_execution_check(
            p, 11,
            "STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED",
            allow_missing_counts_with_outputs=False,
        )
        self.assertEqual(saved["cells"], 11)
        self.assertEqual(saved["saved_output_cells"], 11)
        self.assertEqual(saved["missing_execution_counts"], [])
        self.assertTrue(saved["execution_metadata_complete"])
        self.assertEqual(len(saved["notebook_sha256"]), 64)

    def test_approval_lineage_cannot_be_silently_mutated(self):
        path = ROOT / "wearable_activity_recognition/evidence/stage_25_protocol_approved_qualified_test.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "APPROVED")
        self.assertFalse(manifest["evaluation"]["use_test_for_tuning"])


if __name__ == "__main__":
    unittest.main()
