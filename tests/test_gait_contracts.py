"""Offline tests for the Camargo gait-event scientific contracts."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "gait_analysis" / "code" / "src"
sys.path.insert(0, str(SOURCE))

from camargo_source import inventory_files, paired_trials, validate_timebase
from gait_alignment import mapped_series, validated_overlap
from gait_reference import force_contact_events, reference_strides, stride_summary
from gait_event_detectors import peak_events, candidate_features
from gait_evaluation import (
    match_events, event_metrics, participant_roles, development_folds,
)
from gait_experiments import (
    save_prepared_trial, read_prepared_trial,
    evaluate_trial_collection, participant_selection_score, FIXED_BASELINE,
)


class GaitContractTests(unittest.TestCase):
    def test_source_inventory_and_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for sensor in ("imu", "fp"):
                d = root / "AB06" / "2021-01-01" / "treadmill" / sensor
                d.mkdir(parents=True)
                pd.DataFrame({"Header": [0, 0.01, 0.02],
                              "channel": [0.0, 1.0, 0.0]}).to_csv(
                                  d / "trialA.csv", index=False)
            inventory = inventory_files(root)
            self.assertEqual(len(inventory), 2)
            pairs = paired_trials(inventory)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(bool(pairs.candidate_pair.iloc[0]))
            self.assertEqual(pairs.subject_id.iloc[0], "AB06")

    def test_time_contract_requires_monotonicity_and_actual_overlap(self):
        frame = pd.DataFrame({"Header": [0.0, 0.1, 0.2],
                              "foot_Gyro_X": [1.0, 2.0, 3.0]})
        self.assertEqual(len(validate_timebase(frame)), 3)
        np.testing.assert_array_equal(
            mapped_series(frame, "foot_Gyro_X", "gyro"), [1., 2., 3.]
        )
        with self.assertRaises(ValueError):
            validate_timebase(pd.DataFrame({"Header": [0, 0, 1]}))
        with self.assertRaises(ValueError):
            mapped_series(frame, "unreviewed", "gyro")
        with self.assertRaises(ValueError):
            validated_overlap(np.arange(0, 5), np.arange(10, 15))
        self.assertEqual(validated_overlap(np.arange(0, 10),
                                           np.arange(2, 9)), (2., 8.))

    def test_force_events_and_stride_units(self):
        t = np.arange(0, 6, .001)
        force = np.zeros(len(t))
        for start in (1., 2., 3., 4.):
            force[(t >= start) & (t < start + .5)] = 750.
        events = force_contact_events(t, force)
        self.assertEqual(len(events.loc[events.event_type == "IC"]), 4)
        self.assertEqual(len(events.loc[events.event_type == "TC"]), 4)
        stride = reference_strides(events)
        self.assertEqual(len(stride), 3)
        self.assertAlmostEqual(stride.stride_time_s.mean(), 1.0, places=3)
        self.assertAlmostEqual(stride.cadence_steps_per_min.mean(), 120.0)
        self.assertAlmostEqual(stride_summary(stride)["stride_time_cv"], 0.0)

    def test_one_to_one_matching_preserves_misses_and_extras(self):
        events, missed, extra = match_events(
            [1.0, 2.0, 3.0], [1.02, 1.04, 3.01], .15
        )
        self.assertEqual((len(events), missed, extra), (2, 1, 1))
        metrics = event_metrics([1, 2, 3], [1.02, 1.04, 3.01])
        self.assertEqual(metrics["n_matched"], 2)
        self.assertEqual(metrics["n_missed"], 1)
        self.assertEqual(metrics["n_extra"], 1)
        self.assertAlmostEqual(metrics["sensitivity"], 2/3)
        empty = event_metrics([1, 2], [])
        self.assertTrue(np.isnan(empty["timing_mae_ms"]))
        self.assertEqual(empty["n_missed"], 2)

    def test_grouped_split_is_deterministic_and_disjoint(self):
        people = [f"AB{i:02d}" for i in range(1, 13)]
        a = participant_roles(people)
        b = participant_roles(reversed(people))
        pd.testing.assert_frame_equal(a, b)
        dev = set(a.loc[a.role == "Development", "subject_id"])
        test = set(a.loc[a.role == "Test", "subject_id"])
        self.assertFalse(dev & test)
        folds = development_folds(a)
        self.assertEqual(set(folds.subject_id), dev)
        self.assertEqual(folds.validation_fold.nunique(), 5)

    def test_prepared_trial_and_detector_with_synthetic_wave(self):
        t = np.arange(0, 10, .005)
        gyro = np.sin(2 * np.pi * t)
        with tempfile.TemporaryDirectory() as folder:
            save_prepared_trial(Path(folder), "AB06_trialA", t, gyro,
                                np.array([1., 2., 3.]))
            t2, gyro2, reference = read_prepared_trial(folder, "AB06_trialA")
            self.assertEqual(len(t), len(t2))
            self.assertEqual(len(reference), 3)
        detections = peak_events(t, gyro, polarity=1,
                                 prominence=.5, min_interval_s=.45)
        self.assertGreater(len(detections), 5)
        times, features = candidate_features(t, gyro)
        self.assertEqual(len(times), len(features))
        rows = evaluate_trial_collection(
            [({"subject_id": "AB06", "trial_key": "AB06_trialA"}, (
                t2, gyro2, reference
            ))],
            FIXED_BASELINE, "B0_fixed",
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isfinite(participant_selection_score(rows)))

    def test_notebook_code_static_syntax_and_scientific_stages(self):
        import ast
        paths = sorted((REPO / "gait_analysis/code/notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            nb = json.loads(path.read_text(encoding="utf8"))
            self.assertEqual(nb["nbformat"], 4)
            self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
            code = [c for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertTrue(code)
            for cell in code:
                source = "".join(cell["source"])
                # Replace Colab's valid IPython magic for standard Python AST.
                source = "\n".join(
                    "pass  # IPython magic"
                    if line.lstrip().startswith("%") else line
                    for line in source.split("\n")
                )
                ast.parse(source, filename=path.name)


if __name__ == "__main__":
    unittest.main()
