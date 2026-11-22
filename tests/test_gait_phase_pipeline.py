"""Offline end-to-end smoke test for the strictly exploratory Camargo pathway.

All signals and phase annotations here are synthetic fixtures. This test
does not claim execution on original Camargo MAT data, validated force
events, or reproduced gait timing scores.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python_modules"))
sys.path.insert(0, str(REPO / "gait_analysis/code/src"))

from gait_alignment import mapped_series
from gait_experiments import save_prepared_trial
from gait_evaluation import participant_roles, development_folds
from camargo_source import validate_timebase


class GaitPhasePipelineSmokeTests(unittest.TestCase):
    """Exercise the actual numbered notebook cells without network access."""

    @staticmethod
    def run_cell(number: str, cell_id: str, namespace: dict):
        notebooks = {
            "02": "02_alignment_and_event_reference.ipynb",
            "03": "03_stride_measurement_and_statistics.ipynb",
            "04": "04_imu_event_baselines_and_grouped_validation.ipynb",
            "05": "05_frozen_test_and_research_synthesis.ipynb",
        }
        path = REPO / "gait_analysis/code/notebooks" / notebooks[number]
        nb = json.loads(path.read_text(encoding="utf8"))
        cell = next(c for c in nb["cells"] if c.get("id") == cell_id)
        code = "".join(cell["source"])
        exec(compile(code, path.name + "#" + cell_id, "exec"), namespace)

    def test_phase_annotation_development_and_frozen_test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "gait_analysis/output"
            output.mkdir(parents=True)
            original = root / "data/raw/camargo"
            original.mkdir(parents=True)

            # Twenty-seven synthetic source-paired trials from nine people.
            # Subject-level roles must be assigned before any IMU tuning.
            pairs, phases, force_qc = [], [], []
            for number in range(6, 15):
                subject = f"AB{number:02d}"
                for trial in range(1, 4):
                    trial_id = f"treadmill_{trial:02d}_01"
                    key = {
                        "subject_id": subject,
                        "session_id": "session01",
                        "condition": "treadmill",
                        "trial_id": trial_id,
                    }
                    pairs.append({
                        **key, "candidate_pair": True,
                        "imu_path": f"{subject}/session01/treadmill/imu/{trial_id}.mat",
                    })
                    for moment in np.arange(1, 19, 1.0):
                        phases.append({
                            **key, "phase_wrap_header": float(moment),
                        })
                    quality = 0.55 if (
                        (subject == "AB06" and trial == 1)
                        or (subject == "AB10" and trial == 2)
                    ) else 0.97
                    force_qc.append({
                        **key, "candidate_channel": "Treadmill_R_vy",
                        "max_internal_gap_s": 0.06,
                        "phase_near_match_fraction": quality,
                    })
            timestamps = np.arange(0, 20, 0.005)
            signal = np.sin(2 * np.pi * timestamps)
            imu_table = pd.DataFrame({
                "Header": timestamps, "foot_Gyro_Y": signal,
            })

            def read_imu(_):
                return imu_table

            ns02 = {
                "pd": pd, "np": np, "re": re,
                "root": root, "source_root": original, "output_dir": output,
                "pairs": pd.DataFrame(pairs),
                "original_gc_resets": pd.DataFrame(phases),
                "gap_phase": pd.DataFrame(force_qc),
                "read_sensor_table": read_imu,
                "validate_timebase": validate_timebase,
                "mapped_series": mapped_series,
                "save_prepared_trial": save_prepared_trial,
                "participant_roles": participant_roles,
                "development_folds": development_folds,
                "display": lambda *args, **kwargs: None,
            }
            self.run_cell("02", "gait-02a5-phase-benchmark-code", ns02)
            trial_manifest = pd.read_csv(
                output / "02_phase_annotation_trials.csv"
            )
            self.assertEqual(len(trial_manifest), 27)
            self.assertEqual(int(trial_manifest.review_source_quality.sum()), 2)
            roles = pd.read_csv(
                output / "02_phase_participant_roles.csv"
            )
            self.assertEqual(len(roles), 9)
            self.assertEqual(int(roles.role.eq("Test").sum()), 2)
            self.assertEqual(int(roles.role.eq("Development").sum()), 7)
            self.assertFalse((output / "02_reference_events.csv").exists())
            self.assertFalse((output / "02_eligible_trials.csv").exists())

            # Each subsequent numbered notebook runs its own real code cells.
            # Stub only the repository synchronisation shell, never the
            # scientific calculations, model evaluation or file readers.
            with patch(
                "python_modules.core.colab_setup.setup_colab",
                return_value=root,
            ):
                ns03 = {}
                for name in ("c-03-02", "c-03-03", "c-03-04",
                             "c-03-05", "c-03-06"):
                    self.run_cell("03", name, ns03)
                self.assertTrue(
                    (output / "03_phase_subject_condition_metrics.csv").is_file()
                )
                ns04 = {}
                for name in ("c-04-02", "c-04-03", "c-04-04",
                             "c-04-05", "c-04-06", "c-04-07"):
                    self.run_cell("04", name, ns04)
                self.assertTrue(
                    (output / "04_phase_selected_configuration.csv").is_file()
                )
                self.assertTrue(
                    (output / "04_phase_source_quality_sensitivity.csv").is_file()
                )
                ns05 = {}
                for name in ("c-05-02", "c-05-03", "c-05-04",
                             "c-05-05", "c-05-06", "c-05-07",
                             "c-05-08"):
                    self.run_cell("05", name, ns05)
                summary = pd.read_csv(
                    output / "05_phase_test_participant_metrics.csv"
                )
                self.assertEqual(summary.subject_id.nunique(), 2)
                self.assertTrue(
                    (output / "05_phase_source_quality_sensitivity.csv").is_file()
                )
                self.assertFalse(
                    (output / "05_test_participant_metrics.csv").exists()
                )
                self.assertFalse(
                    (output / "04_selected_configuration.csv").exists()
                )


if __name__ == "__main__":
    unittest.main()
