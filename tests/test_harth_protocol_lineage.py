# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline safeguards for replaying the already frozen HARTH Stage 25 protocol."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"python_modules"))
from core.harth_protocol_lineage import (
    validate_stage25_lineage, verified_module_sha256,
    resolve_frozen_source_path,
    verify_preexisting_stage25_lineage,
)

EVIDENCE = ROOT/"wearable_activity_recognition/evidence"


# ===================== 01. HARTH Protocol Lineage Tests ======================
def records():
    original = json.loads(
        (EVIDENCE/"stage_25_protocol_proposal.json").read_text(encoding="utf-8")
    )
    reconciled = json.loads(
        (EVIDENCE/"stage_25_protocol_proposal_reconciled.json").read_text(
            encoding="utf-8"
        )
    )
    approved = json.loads(
        (EVIDENCE/"stage_25_protocol_approved_qualified_test.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = {
        key: value for key, value in reconciled.items()
        if key != "reconciliation"
    }
    return candidate, original, reconciled, approved


class TestStage25HistoricalReplay(unittest.TestCase):
    def test_read_only_existing_history_is_consistent(self):
        candidate, _, _, _ = records()
        result = verify_preexisting_stage25_lineage(candidate, EVIDENCE)
        self.assertEqual(result["reconciled_proposal"], "MATCHED")
        self.assertEqual(result["existing_approval"], "VERIFIED_EXISTING_ONLY")
        self.assertEqual(result["files_written"], 0)
        self.assertFalse(result["new_approval_granted"])

    def test_refuse_changed_model_or_cohort(self):
        candidate, original, reconciled, approved = records()
        for section, field, value in (
            ("primary_model", "optimizer_steps", 401),
            ("source", "development_participants", 30),
            ("evaluation", "use_test_for_tuning", True),
        ):
            altered = copy.deepcopy(candidate)
            altered[section][field] = value
            with self.subTest(section=section, field=field):
                with self.assertRaisesRegex(ValueError, "Replayed Stage 25"):
                    validate_stage25_lineage(
                        altered, original, reconciled, approved
                    )

    def test_refuse_broken_approval_or_original_provenance(self):
        candidate, original, reconciled, approved = records()
        bad_approved = copy.deepcopy(approved)
        bad_approved["approval_record"] = {}
        with self.assertRaisesRegex(ValueError, "approved evaluation provenance is missing"):
            validate_stage25_lineage(
                candidate, original, reconciled, bad_approved
            )
        bad_original = copy.deepcopy(original)
        bad_original["source"]["reserved_test_people"] = ["S999"]
        with self.assertRaisesRegex(ValueError, "scientific protocol"):
            validate_stage25_lineage(
                candidate, bad_original, reconciled, approved
            )

    def test_strict_sha256_with_crlf_only(self):
        canonical = b"def sample():\n    return 2\n"
        sha = hashlib.sha256(canonical).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"module.py"
            path.write_bytes(canonical.replace(b"\n", b"\r\n"))
            self.assertEqual(verified_module_sha256(path, sha), sha)
            path.write_bytes(b"def sample():\n    return 3\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verified_module_sha256(path, sha)

    def test_frozen_main_sources_match_reconciled_hashes(self):
        _, _, reconciled, _ = records()
        for relative, sha in reconciled["module_sha256"].items():
            with self.subTest(path=relative):
                self.assertEqual(
                    verified_module_sha256(resolve_frozen_source_path(ROOT, relative), sha), sha
                )


if __name__ == "__main__":
    unittest.main()
