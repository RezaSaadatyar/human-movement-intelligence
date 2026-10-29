"""Offline regression checks for executed HARTH selection evidence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python_modules"))
from core.harth_selection_evidence import read_html_table,load_selection_evidence

class TestSelectionEvidence(unittest.TestCase):
    def test_parse_flat_pandas_table(self):
        html="<table><thead><tr><th></th><th>model</th><th>macro_f1</th></tr></thead><tbody><tr><th>0</th><td>ResNet</td><td>0.83</td></tr></tbody></table>"
        frame=read_html_table(html)
        self.assertEqual(frame["model"].tolist(),["ResNet"])
        self.assertAlmostEqual(float(frame.loc[0,"macro_f1"]),0.83)

    def test_refuse_incomplete_table(self):
        html="<table><thead><tr><th>model</th><th>score</th></tr></thead><tbody><tr><td>ResNet</td></tr></tbody></table>"
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            read_html_table(html)

    def test_read_real_saved_research_notebooks_when_available(self):
        root=Path(__file__).resolve().parents[1]
        p=root/"wearable_activity_recognition/code/notebooks"
        n23=p/"22_unified_matched_model_benchmark.ipynb"
        n24=p/"23_multiseed_and_imbalance_robustness.ipynb"
        evidence=load_selection_evidence(n23,n24)
        self.assertEqual(len(evidence["stage23"]),7)
        self.assertEqual(len(evidence["stage24_folds"]),36)
        self.assertEqual(len(evidence["stage24_summary"]),4)
        self.assertEqual(len(evidence["development_sha256"]),64)

if __name__=="__main__":unittest.main()
