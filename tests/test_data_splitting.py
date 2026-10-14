"""Tests for reusable subject-independent splitting."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/python/src"))
from core.data_splitting import split_group_indices

class TestGroupedSplit(unittest.TestCase):
    def setUp(self):
        self.ids=np.repeat([f"S{i:03d}" for i in range(8)],[4,8,3,7,10,5,9,6])
    def test_no_leakage_and_full_coverage(self):
        split=split_group_indices(self.ids)
        self.assertEqual([len(set(self.ids[ix])) for ix in split.values()],[4,2,2])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))),np.arange(len(self.ids)))
        groups=[set(self.ids[ix]) for ix in split.values()]
        self.assertFalse(groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
    def test_deterministic(self):
        a=split_group_indices(self.ids,random_state=12)
        b=split_group_indices(self.ids,random_state=12)
        for k in a: np.testing.assert_array_equal(a[k],b[k])
    def test_small_groups(self):
        d=split_group_indices(["a","b","c"],0.8,0.1,0.1)
        self.assertEqual([len(v) for v in d.values()],[1,1,1])
    def test_reject_one_person(self):
        with self.assertRaises(ValueError): split_group_indices(["S023"]*8)
    def test_invalid_fraction(self):
        with self.assertRaises(ValueError): split_group_indices(self.ids,0.6,0.4,0.4)
    def test_empty_ids(self):
        with self.assertRaises(ValueError): split_group_indices(["a",None,"b","c"])
if __name__=="__main__": unittest.main()
