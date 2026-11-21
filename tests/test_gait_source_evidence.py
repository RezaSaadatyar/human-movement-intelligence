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
    validate_explicit_contract, recorded_unit_contact_candidates,
    audit_recorded_force_candidates,
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

    def test_screening_uses_source_units_and_keeps_physical_status_false(self):
        """Reproduce the new source-only screening cell with synthetic data."""
        notebook = REPO / "gait_analysis/code/notebooks" / (
            "02_alignment_and_event_reference.ipynb"
        )
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        source = next(
            "".join(cell["source"]) for cell in cells
            if cell.get("id") == "gait-02a-source-code"
        )
        heading = "# --------------- 3. Summarise Observed Force-Channel Evidence"
        self.assertIn(heading, source)
        snippet = source[source.index(heading):]
        with tempfile.TemporaryDirectory() as folder:
            values = pd.DataFrame([
                {"channel": "Treadmill_R_vy", "trial_id": "a",
                 "p01_source": 0.0, "p99_source": 800.0,
                 "median_source": 350.0, "near_zero_fraction_relative": 0.3},
                {"channel": "Treadmill_R_vy", "trial_id": "b",
                 "p01_source": 0.0, "p99_source": 950.0,
                 "median_source": 400.0, "near_zero_fraction_relative": 0.4},
                {"channel": "Treadmill_R_vx", "trial_id": "a",
                 "p01_source": -80.0, "p99_source": 30.0,
                 "median_source": -10.0, "near_zero_fraction_relative": 0.2},
            ])
            namespace = {
                "component_evidence": values, "pd": pd,
                "output_dir": Path(folder), "display": lambda *args: None,
            }
            exec(compile(snippet, "screening", "exec"), namespace)
            saved = pd.read_csv(Path(folder) / "02_force_channel_screening.csv")
            y = saved.loc[saved.channel == "Treadmill_R_vy"].iloc[0]
            self.assertEqual(int(y.n_inspected_trials), 2)
            self.assertAlmostEqual(float(y.median_p99_source), 875)
            self.assertFalse(bool(y.physical_axis_verified))
            self.assertFalse(bool(y.units_verified_newtons))
            self.assertFalse(bool(y.foot_side_verified))


    def test_original_unit_contact_candidates_are_not_reference_labels(self):
        """Real-unit pulse candidates remain distinct from verified IC/TC."""
        t = np.arange(0, 6, 0.001)
        f = np.zeros_like(t)
        for onset in (1.0, 2.0, 3.0, 4.0):
            f[(t >= onset) & (t < onset + 0.55)] = 810.0
        detected = recorded_unit_contact_candidates(t, f)
        self.assertEqual(len(detected), 4)
        self.assertEqual(detected.candidate_onset_header.iloc[0], 1.0)
        self.assertTrue(
            set(detected.columns).isdisjoint(
                {"t_reference_s", "event_type", "is_verified"}
            )
        )
        self.assertTrue(
            recorded_unit_contact_candidates(t, -f).empty
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            recorded_unit_contact_candidates(t[::-1], f[::-1])

    def test_cross_trial_candidate_audit_is_original_data_only(self):
        """Six trial/side/threshold rows, never a scientifically valid reference."""
        t = np.arange(0, 6, 0.001)
        f = np.zeros_like(t)
        for onset in (1.0, 2.0, 3.0, 4.0):
            f[(t >= onset) & (t < onset + 0.55)] = 810.0
        fp = pd.DataFrame({
            "Header": t, "Treadmill_R_vy": f,
            "Treadmill_L_vy": np.roll(f, 500),
        })
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "fp.mat").write_bytes(b"synthetic")
            registry = pd.DataFrame([{
                "subject_id": "S01", "session_id": "X",
                "condition": "treadmill", "trial_id": "t01",
                "fp_path": "fp.mat", "candidate_pair": True,
            }])
            diag, candidates = audit_recorded_force_candidates(
                registry, Path(folder), lambda p: fp,
                lambda x, label: x.Header.to_numpy(float),
            )
            self.assertEqual(len(diag), 6)
            self.assertEqual(len(candidates), 8)
            self.assertTrue(diag.inspection_status.eq("inspected").all())
            self.assertFalse(diag.physical_reference_validated.any())
            self.assertFalse(candidates.physical_reference_validated.any())
            self.assertTrue(
                set(candidates).isdisjoint(
                    {"t_reference_s", "event_type"}
                )
            )
            self.assertTrue(
                set(diag.threshold_recorded_units) == {10, 20, 30}
            )

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
