"""Offline tests for record-level MATLAB metadata audit (synthetic files)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from scripts.audit_mat_records import audit_records, classify_field, inspect_subject


class MatAuditTests(unittest.TestCase):
    def test_opaque_and_empty_classification(self):
        fake = np.zeros(1, dtype=[
            ("_TypeSystem", object), ("_Class", object),
            ("_ObjectMetadata", object),
        ])
        self.assertEqual(classify_field(fake), "matlab_object_reference")
        self.assertEqual(classify_field(np.empty((0, 0))), "empty")

    def test_all_records_and_duplicate_ids(self):
        field_names = (
            'SubjID', 'Task', 'SubTask', 'TrialNum', 'TrialNumUpdated',
            'UpdatedTrialID', 'Cycle', 'IMU', 'ID', 'FP', 'e1e3', 'JCF', 'MuscleF'
        )
        records = np.empty((2, 1), dtype=[(f, object) for f in field_names])
        for i in range(2):
            fields = {
                "SubjID": "subj04", "Task": "Walking", "SubTask": "low",
                "TrialNum": str(i + 1), "TrialNumUpdated": 1,
                "UpdatedTrialID": "duplicate", "Cycle": 1,
                "IMU": np.ones((3, 101)), "ID": np.ones((3, 101)),
                "FP": np.empty((0, 0)), "e1e3": np.empty((0, 0)),
                "JCF": np.ones((2, 101)), "MuscleF": np.empty((0, 0))
            }
            for name, value in fields.items():
                records[name][i, 0] = value
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subj04.mat"
            savemat(path, {"Data": records})
            audit = inspect_subject(path)
        self.assertEqual(audit["records"], 2)
        self.assertEqual(audit["fields"]["e1e3"]["empty"], 2)
        self.assertEqual(audit["fields"]["IMU"]["numeric_array"], 2)
        self.assertEqual(audit["updated_trial_id_duplicates"], 1)
        self.assertTrue(audit["filename_subject_matches"])
        self.assertEqual(audit["tasks"], {"Walking": 2})

    def test_reject_incorrect_data_shape(self):
        with self.assertRaises(ValueError):
            audit_records(np.ones((2, 2)), "invalid.mat")


if __name__ == "__main__":
    unittest.main()
