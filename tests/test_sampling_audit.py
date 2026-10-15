"""Offline tests for label-blind temporal sampling audits."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/python/src"))
from core.sampling_audit import audit_temporal_sampling, select_evenly_spaced_indices, select_nested_window_indices

class TestSamplingAudit(unittest.TestCase):
    def setUp(self):
        self.meta=pd.DataFrame({
            "subject_id":["S01"]*12+["S02"]*8,
            "label":[1]*9+[2]*3+[1]*4+[2]*4,
        })
    def test_exact_stage_ten_positions(self):
        np.testing.assert_array_equal(
            select_evenly_spaced_indices(12,5),
            np.linspace(0,11,num=5,dtype=np.int64),
        )
    def test_nested_training_caps(self):
        small, big = select_nested_window_indices(2400, 500, 1000)
        self.assertEqual(len(small), 500)
        self.assertEqual(len(big), 1000)
        self.assertTrue(np.isin(small, big).all())
        np.testing.assert_array_equal(
            small, select_evenly_spaced_indices(2400, 500)
        )
        self.assertTrue(np.all(np.diff(big) > 0))

    def test_nested_caps_with_short_recording(self):
        small, big = select_nested_window_indices(201, 500, 1000)
        np.testing.assert_array_equal(small, big)
        self.assertEqual(len(big), 201)

    def test_full_and_capped_provenance(self):
        selected,per_person,classes=audit_temporal_sampling(
            self.meta,caps=[3,5],classes=[1,2],
            forbidden_subjects=["S99"],
        )
        self.assertEqual(len(selected[selected.policy=="full"]),20)
        self.assertEqual(len(selected[selected.policy=="cap_3"]),6)
        self.assertEqual(len(selected[selected.policy=="cap_5"]),10)
        self.assertEqual(len(classes),6)
        self.assertTrue(
            selected.groupby(["policy","subject_id"])["source_row"]
            .apply(lambda x:x.is_unique).all()
        )
    def test_protected_subject_rejected_case_insensitively(self):
        with self.assertRaisesRegex(ValueError,"Protected Test"):
            audit_temporal_sampling(
                self.meta,caps=[3],classes=[1,2],
                forbidden_subjects=["s02"],
            )
    def test_invalid_caps_and_labels(self):
        with self.assertRaises(ValueError):
            audit_temporal_sampling(self.meta,caps=[0],classes=[1,2])
        with self.assertRaisesRegex(ValueError,"undocumented"):
            audit_temporal_sampling(self.meta,caps=[3],classes=[1,3])

if __name__=="__main__":unittest.main()
