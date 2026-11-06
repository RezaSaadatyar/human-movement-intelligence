"""Current-file data flow, numerical pairing and saved research results."""

import ast
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "biomechanical_load_estimation"
sys.path.insert(0, str(STUDY / "code/src"))
import essex_figshare as source
import essex_paired_dataset as pairs
from essex_record_audit import load_participant_ids
from essex_temporal_mapping import audit_temporal_mapping
from essex_timebase_audit import EXPECTED_CHANNELS, audit_original_source


class CurrentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_download_reuses_current_file_and_rejects_incomplete_transfer(self):
        item = {
            "name": "source.txt",
            "size": 3,
            "download_url": "https://example.org/source.txt",
        }
        with patch.object(source, "urlopen", return_value=io.BytesIO(b"new")):
            path = source.ensure_source_file(item, self.root)
        self.assertEqual(path.read_bytes(), b"new")
        with patch.object(
            source, "urlopen", side_effect=AssertionError("network not needed")
        ):
            self.assertEqual(source.ensure_source_file(item, self.root), path)
        path.unlink()
        with patch.object(source, "urlopen", return_value=io.BytesIO(b"x")):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                source.ensure_source_file(item, self.root)
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_archive_replaces_same_size_extracted_file_and_rejects_traversal(self):
        archive = self.root / "MATLAB.zip"
        destination = self.root / "mat"
        for payload in (b"old", b"new"):
            with ZipFile(archive, "w") as z:
                z.writestr("folder/subj01.mat", payload)
            extracted = source.extract_matlab_files(
                archive, destination, source.inspect_matlab_archive(archive)
            )
            self.assertEqual(extracted[0].read_bytes(), payload)
        with ZipFile(archive, "w") as z:
            z.writestr("../escape.mat", b"bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            source.inspect_matlab_archive(archive)

    def test_workbook_summary_does_not_require_forty_people(self):
        table = pd.DataFrame({"Participent ID": ["Subj01", "Subj01", "Subj44", "Mean"]})
        with patch("essex_record_audit.pd.read_excel", return_value=table):
            summary, people = load_participant_ids(self.root / "workbook.xlsx")
        self.assertEqual(summary["participants"], 2)
        self.assertEqual(summary["duplicate_ids"], 1)
        self.assertEqual(people["subject_id"].tolist(), ["subj01", "subj44"])

    def synthetic_source(self):
        imu = pd.DataFrame(
            np.arange(101 * 42).reshape(101, 42), columns=EXPECTED_CHANNELS
        )
        imu["timestep"] = np.arange(101, dtype=float)
        positions = np.arange(0, 101, 10)
        fe = pd.DataFrame(
            {
                "timestep": positions.astype(float),
                "e1": positions / 1000,
                "e3": -positions / 1000,
            }
        )
        data = {
            "SubjID": "Subj01",
            "Task": "Walking",
            "UpdatedTrialID": "walk_1",
            "IMU": imu,
            "e1e3": fe,
        }
        registry = pd.DataFrame(
            [
                {
                    "subject_id": "subj01",
                    "task": "Walking",
                    "updated_trial_id": "walk_1",
                    "local_record_index": 0,
                    "local_record_key": "subj01:0000",
                    "source_file": "subj01.mat",
                    "fe_representation": "matlab_object_unresolved",
                }
            ]
        )
        return lambda *a, **k: {"Data": [data]}, registry, imu, fe

    def test_numeric_audit_mapping_and_pairing_with_small_current_cohort(self):
        reader, registry, imu, fe = self.synthetic_source()
        audit, errors, _ = audit_original_source(reader, self.root, registry)
        self.assertTrue(errors.empty)
        self.assertEqual(audit["imu_status"].tolist(), ["VALID_CHANNELS"])
        mapping, _, mapping_errors, _ = audit_temporal_mapping(
            reader, self.root, registry
        )
        self.assertTrue(mapping_errors.empty)
        arrays, eligible, excluded = pairs.build_paired_arrays(
            reader, self.root, registry, mapping
        )
        self.assertEqual(len(eligible), 1)
        self.assertTrue(excluded.empty)
        np.testing.assert_array_equal(
            arrays["X_at_fe"][0], imu.loc[::10, list(EXPECTED_CHANNELS)]
        )
        np.testing.assert_array_equal(arrays["y_fe"][0], fe[["e1", "e3"]])
        wrong = mapping.copy()
        wrong["local_record_key"] = "different_trial"
        with self.assertRaisesRegex(ValueError, "trial keys mismatch"):
            pairs.build_paired_arrays(reader, self.root, registry, wrong)

    def test_original_participant_allocation_is_preserved(self):
        output = STUDY / "output"
        registry = pd.read_csv(output / "01_original_record_registry.csv")
        eligible = pd.read_csv(output / "04_eligible_trial_registry.csv")
        people, assigned, _, _ = pairs.fixed_subject_splits(eligible, registry)
        old_people = pd.read_csv(output / "04_subject_split_assignments.csv")
        old_trials = pd.read_csv(output / "04_trial_split_assignments.csv")
        pd.testing.assert_series_equal(
            people.set_index("subject_id")["split"].sort_index(),
            old_people.set_index("subject_id")["split"].sort_index(),
        )
        pd.testing.assert_series_equal(
            assigned.set_index("local_record_key")["split"].sort_index(),
            old_trials.set_index("local_record_key")["split"].sort_index(),
        )

    def test_arrays_are_replaced_and_loaded_in_trial_table_order(self):
        output, raw, processed = [self.root / n for n in ("output", "raw", "processed")]
        for p in (output, raw, processed):
            p.mkdir()
        for name in (
            "01_original_record_registry.csv",
            "03_original_time_coordinate_mapping.csv",
            "04_trial_split_assignments.csv",
        ):
            (output / name).write_text("input\n")
        assigned = pd.DataFrame({"local_record_key": ["a", "b"]})
        arrays = {
            name: np.array([[1], [2]])
            for name in ("X_full", "X_at_fe", "y_fe", "imu_positions")
        }
        path = processed / "essex_imu_fe_pairs_v13.npz"
        pairs.save_paired_data(path, arrays, assigned)
        arrays["y_fe"] = np.array([[3], [4]])
        pairs.save_paired_data(path, arrays, assigned)
        with patch.object(
            pairs,
            "ensure_matlab_archive",
            side_effect=AssertionError("must reuse current arrays"),
        ):
            loaded = pairs.load_paired_data(
                output, processed, raw, None, assigned.iloc[::-1]
            )
        np.testing.assert_array_equal(loaded["y_fe"], [[4], [3]])
        self.assertEqual(len(list(processed.iterdir())), 1)

    def test_newer_upstream_table_rebuilds_arrays(self):
        output, raw, processed = [self.root / n for n in ("output", "raw", "processed")]
        for p in (output, raw, processed):
            p.mkdir()
        for name in (
            "01_original_record_registry.csv",
            "03_original_time_coordinate_mapping.csv",
            "04_trial_split_assignments.csv",
        ):
            (output / name).write_text("local_record_key\na\n")
        assigned = pd.DataFrame({"local_record_key": ["a"]})
        arrays = {
            name: np.zeros((1, 1))
            for name in ("X_full", "X_at_fe", "y_fe", "imu_positions")
        }
        cache = pairs.save_paired_data(
            processed / "essex_imu_fe_pairs_v13.npz", arrays, assigned
        )
        os.utime(cache, (1, 1))
        updated = {name: np.ones((1, 1)) for name in arrays}
        with (
            patch.object(
                pairs, "ensure_matlab_archive", return_value=raw / "MATLAB.zip"
            ),
            patch.object(pairs, "inspect_matlab_archive", return_value=[]),
            patch.object(pairs, "extract_matlab_files"),
            patch.object(
                pairs,
                "build_paired_arrays",
                return_value=(updated, assigned, pd.DataFrame()),
            ) as build,
        ):
            loaded = pairs.load_paired_data(output, processed, raw, None, assigned)
        build.assert_called_once()
        np.testing.assert_array_equal(loaded["y_fe"], [[1]])

    def test_split_reader_rejects_participant_leakage(self):
        pd.DataFrame(
            {
                "local_record_key": ["a", "b"],
                "subject_id": ["same", "same"],
                "split": ["TRAIN", "TEST"],
            }
        ).to_csv(self.root / "04_trial_split_assignments.csv", index=False)
        with self.assertRaisesRegex(ValueError, "more than one"):
            pairs.read_trial_splits(self.root)

    def test_training_folds_align_by_key_and_keep_participants_together(self):
        trials = pd.DataFrame(
            {"local_record_key": ["a", "b", "c"], "subject_id": ["p1", "p1", "p2"]}
        )
        folds = trials.assign(validation_fold=[1, 1, 2])
        path = self.root / "05_training_fold_registry.csv"
        folds.iloc[::-1].to_csv(path, index=False)
        aligned = pairs.read_training_folds(self.root, trials)
        self.assertEqual(aligned["validation_fold"].tolist(), [1, 1, 2])
        folds.loc[1, "validation_fold"] = 2
        folds.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "one fold"):
            pairs.read_training_folds(self.root, trials)

    def test_notebook_syntax_and_no_administrative_prerequisites(self):
        notebooks = sorted((STUDY / "code/notebooks").glob("0[1-8]_*.ipynb"))
        self.assertEqual(len(notebooks), 8)
        for path in notebooks:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                tree = ast.parse(TransformerManager().transform_cell(cell.source))
                self.assertFalse(
                    any(isinstance(n, ast.Assert) for n in ast.walk(tree)), path
                )
                for obsolete in (
                    "quality_gate",
                    "source_gate",
                    "archive_sha256",
                    "_consolidation_drafts",
                    "release_one_time_test",
                ):
                    self.assertNotIn(obsolete, cell.source, path)
                self.assertFalse(
                    any(o.output_type == "error" for o in cell.get("outputs", []))
                )

    def test_saved_test_predictions_reproduce_participant_weighted_mae(self):
        output = STUDY / "output"
        predictions = pd.read_csv(output / "08_test_trial_predictions.csv")
        metrics = pd.read_csv(output / "08_test_primary_secondary_metrics.csv")
        keys = [
            c
            for c in ("model", "local_record_key", "fe_position", "target")
            if c in predictions.columns
        ]
        self.assertEqual(len(keys), 4)
        self.assertFalse(predictions.duplicated(keys).any())
        self.assertTrue(np.isfinite(predictions[["observed", "predicted"]]).all().all())
        predictions["error"] = (
            predictions["observed"] - predictions["predicted"]
        ).abs()
        balanced = (
            predictions.groupby(["model", "subject_id"])["error"]
            .mean()
            .groupby("model")
            .mean()
        )
        for name, observed in balanced.items():
            reported = metrics.loc[metrics["model"].eq(name), "subject_balanced_mae"]
            np.testing.assert_allclose(reported, observed, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
