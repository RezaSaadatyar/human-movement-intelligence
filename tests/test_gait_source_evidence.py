"""Synthetic tests for the Camargo source-contract inspection extension."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "gait_analysis" / "code" / "src"
sys.path.insert(0, str(SRC))

from gait_contract_evidence import (
    audit_candidate_pairs, inspect_source_pair, inspect_timestamp,
    validate_explicit_contract,
)


class GaitSourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        ti = np.arange(0, 8, 0.005)
        tf = np.arange(0, 8, 0.001)
        self.imu = pd.DataFrame({
            "Header": ti, "foot_Gyro_X": np.sin(ti),
            "foot_Gyro_Y": np.cos(ti), "foot_Gyro_Z": np.zeros_like(ti),
        })
        pulse = np.where((tf % 1.0) < 0.6, 550.0, 0.0)
        self.fp = pd.DataFrame({
            "Header": tf, "Treadmill_R_vx": np.zeros_like(tf),
            "Treadmill_R_vy": pulse, "Treadmill_R_vz": 20 * np.sin(tf),
            "Treadmill_L_vx": np.zeros_like(tf),
            "Treadmill_L_vy": np.roll(pulse, 500),
            "Treadmill_L_vz": np.zeros_like(tf),
        })

    def test_original_header_audit_does_not_claim_clock_proof(self):
        evidence, components = inspect_source_pair(
            self.imu, self.fp, self.imu.Header, self.fp.Header
        )
        self.assertAlmostEqual(evidence["imu_samples_per_header_unit"], 200)
        self.assertAlmostEqual(evidence["fp_samples_per_header_unit"], 1000)
        self.assertTrue(evidence["time_overlap_not_clock_proof"])
        self.assertEqual(len(components), 6)
        self.assertFalse(any("event_type" in c for c in components))
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            inspect_timestamp([0, 0.1, 0.1, 0.3], "force")

    def test_contract_requires_explicit_units_clock_and_source(self):
        fields = dict(
            gyro_column="foot_Gyro_Y", force_column="Treadmill_R_vy",
            foot_side="right", force_upward_sign=1,
            time_unit_in_seconds=1.0, force_scale_to_newtons=1.0,
            imu_force_timebase_confirmed=False,
            verified_original_source_note=None,
            observed_imu_columns=set(self.imu.columns),
            observed_force_columns=set(self.fp.columns),
        )
        with self.assertRaisesRegex(ValueError, "IMU_FORCE_TIMEBASE_CONFIRMED"):
            validate_explicit_contract(**fields)
        fields["imu_force_timebase_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "ORIGINAL_SOURCE_NOTE"):
            validate_explicit_contract(**fields)
        fields["verified_original_source_note"] = (
            "Synthetic demonstration, not evidence about Camargo"
        )
        self.assertEqual(
            validate_explicit_contract(**fields)["FORCE_SCALE_TO_NEWTONS"], 1.0
        )
        fields["force_scale_to_newtons"] = None
        with self.assertRaisesRegex(ValueError, "FORCE_SCALE_TO_NEWTONS"):
            validate_explicit_contract(**fields)

    def test_inspection_reports_missing_raw_source_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "i.mat").write_bytes(b"synthetic")
            (root / "f.mat").write_bytes(b"synthetic")
            data = {"i.mat": self.imu, "f.mat": self.fp}
            registry = pd.DataFrame([
                dict(subject_id="S01", session_id="a", condition="treadmill",
                     trial_id="01", imu_path="i.mat", fp_path="f.mat",
                     candidate_pair=True),
                dict(subject_id="S02", session_id="a", condition="treadmill",
                     trial_id="02", imu_path="missing.mat", fp_path="f.mat",
                     candidate_pair=True),
            ])
            inspected, components = audit_candidate_pairs(
                registry, root, lambda p: data[p.name],
                lambda frame, label: frame.Header.to_numpy(float),
            )
            self.assertEqual(inspected.inspection_status.tolist(),
                             ["inspected", "not_inspected"])
            self.assertEqual(len(components), 6)

    def test_notebook_extension_preserves_deferred_scientific_boundary(self):
        notebook = REPO / "gait_analysis/code/notebooks" / (
            "02_alignment_and_event_reference.ipynb"
        )
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        texts = ["".join(c["source"]) for c in cells]
        self.assertTrue(any("02A. Cross-Trial Physical Source Evidence"
                            in t for t in texts))
        self.assertTrue(any("02B. Explicit Physical Source Contract"
                            in t for t in texts))
        self.assertTrue(any("FORCE_SCALE_TO_NEWTONS" in t
                            and "Construct Force References" in t
                            for t in texts))
        self.assertTrue(any("force_scale_to_newtons\": None" in t
                            for t in texts))
        self.assertTrue(any("IMU_FORCE_TIMEBASE_CONFIRMED = False"
                            in t for t in texts))
        self.assertFalse(any("reference_ready = True" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
