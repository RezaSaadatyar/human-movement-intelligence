"""Offline regression tests for source HARTH timing and annotation audits."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/python/src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"studies/wearable_activity_recognition/src"))
from harth_quality import (window_rejection_reasons,
    characterise_annotation_bouts,inspect_raw_sensor_channels)

class TestHarthQuality(unittest.TestCase):
    def test_window_reasons_agree_with_time_safe_pipeline(self):
        # 1. Two clean 4-point windows, one label transition and one gap.
        times=pd.date_range("2026-01-01",periods=16,freq="20ms").to_series().reset_index(drop=True)
        times.iloc[10:]+=pd.Timedelta(seconds=.2)
        y=np.array([1]*7+[2]*9)
        result=window_rejection_reasons(y,times,4,2,50)
        self.assertEqual(result["candidate_windows"],7)
        self.assertEqual(sum(result[k] for k in (
            "retained_pure_time_safe","rejected_label_only",
            "rejected_time_only","rejected_both")),7)
        self.assertGreater(result["rejected_label_only"],0)
        self.assertGreater(result["rejected_time_only"],0)

    def test_bouts_split_when_time_discontinuous(self):
        # 1. Distinguish separate chronological episodes under same label.
        times=pd.date_range("2026-01-01",periods=8,freq="20ms").to_series().reset_index(drop=True)
        times.iloc[4:]+=pd.Timedelta(seconds=.5)
        bouts=characterise_annotation_bouts([3]*8,times,"S006",50,4)
        self.assertEqual(len(bouts),2)
        self.assertEqual(bouts["samples"].tolist(),[4,4])

    def test_sensor_stats_are_source_only(self):
        source=pd.DataFrame({f"axis_{k}":np.arange(10,dtype=float)
                             for k in range(6)})
        report=inspect_raw_sensor_channels(source,"S006",source.columns)
        self.assertEqual(len(report),6)
        self.assertEqual(report["nonfinite_samples"].sum(),0)
        self.assertTrue(report["std"].gt(0).all())
        
    def test_reject_invalid_timestamp_alignment(self):
        with self.assertRaisesRegex(ValueError,"identical row counts"):
            window_rejection_reasons([1,2],[pd.Timestamp("2026-01-01")],2,1,50)

if __name__=="__main__":unittest.main()
