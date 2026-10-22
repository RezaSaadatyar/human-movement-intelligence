"""Offline tests of shared subject/class coverage auditing."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.data_splitting import audit_group_class_coverage

class TestGroupCoverage(unittest.TestCase):
    def setUp(self):
        self.subjects=np.repeat(["S001","S002","S003","S004","S005","S006"], 12)
        self.labels=np.tile([1,2,3,1,2,3,1,2,3,1,2,3],6)

    def test_subject_counts_and_fold_leakage(self):
        coverage,folds=audit_group_class_coverage(
            self.labels,self.subjects,classes=[1,2,3],n_splits=3
        )
        self.assertEqual(coverage.shape,(6,3))
        self.assertTrue((coverage.sum(axis=1)==12).all())
        self.assertEqual(len(folds),3)
        self.assertFalse(folds["missing_train_classes"].map(bool).any())
        for row in folds.itertuples(index=False):
            self.assertFalse(set(row.train_subjects.split(", ")) &
                             set(row.validation_subjects.split(", ")))

    def test_missing_validation_classes_are_reported(self):
        labels=np.repeat([1,1,2,2,3,3],12)
        coverage,folds=audit_group_class_coverage(
            labels,self.subjects,classes=[1,2,3],n_splits=3
        )
        self.assertTrue(folds["missing_validation_classes"].map(bool).any())
        self.assertEqual(coverage.to_numpy().sum(),len(labels))

    def test_casefolded_ids_display_original(self):
        ids=self.subjects.copy()
        ids[0]="s001"
        coverage,_=audit_group_class_coverage(
            self.labels,ids,classes=[1,2,3],n_splits=3
        )
        self.assertEqual(coverage.shape[0],6)

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            audit_group_class_coverage(self.labels,self.subjects[:-1],[1,2,3])
        with self.assertRaises(ValueError):
            audit_group_class_coverage(self.labels,self.subjects,[1,2,3],n_splits=9)

if __name__=="__main__": unittest.main()
