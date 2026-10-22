"""Check reusable file loaders and the study-specific BoneNet reader."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python_modules'))
sys.path.insert(0, str(ROOT / 'biomechanical_load_estimation' / 'code' / 'src'))

from core.data_loading import load_data
from core.data_inspection import inspect_data, plot_signal
from bonenet_data import load_bonenet_data


class TestCommonDataFunctions(unittest.TestCase):
    def test_csv_and_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = pd.DataFrame({'subject': ['s1', 's2'], 'angle': [10.0, np.nan]})
            table.to_csv(root / 'test.csv', index=False)
            table.to_excel(root / 'test.xlsx', index=False)
            for file in ('test.csv', 'test.xlsx'):
                loaded = load_data(root / file)
                self.assertEqual(loaded.shape, (2, 2))
                self.assertEqual(inspect_data(loaded, file)['missing_values'], 1)

    def test_text_and_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'test.dat').write_text('1 2 3\n4 5 6\n', encoding='utf-8')
            savemat(root / 'test.mat', {'signal': np.ones((3, 8))})
            self.assertEqual(load_data(root / 'test.dat').shape, (2, 3))
            mat = load_data(root / 'test.mat')
            self.assertEqual(mat['signal'].shape, (3, 8))

    def test_generic_array_inspection_and_signal_plot(self):
        from matplotlib import pyplot as plt
        arr = np.ones((3, 4, 15))
        arr[0, 0, 0] = np.nan
        self.assertEqual(inspect_data(arr)['nonfinite_values'], 1)
        fig = plot_signal(np.ones((2, 15)), channel_names=['x', 'y'])
        self.assertEqual(len(fig.axes[0].lines), 2)
        plt.close(fig)

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'data.unknown'
            file.write_text('1 2')
            with self.assertRaisesRegex(ValueError, 'Unsupported format'):
                load_data(file)

    def test_bonenet_adapter_uses_common_mat_and_csv_readers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n, t = 4, 9
            inputs = {axis: {'foot_acc': np.ones((n, t))} for axis in 'xyz'}
            outputs = {'e1': np.ones((n, t)), 'e3': -np.ones((n, t))}
            savemat(root / 'Data.mat', {'inputs': inputs, 'outputs': outputs})
            pd.DataFrame({'SubjID': ['s1','s1','s2','s2'],
                          'Task': ['Walk']*n, 'TrialNum': [1,2,1,2]}).to_csv(
                root / 'labels.csv', index=False)
            X, y, labels, channels, targets = load_bonenet_data(root)
            self.assertEqual(X.shape, (4, 3, 9))
            self.assertEqual(y.shape, (4, 9, 2))
            self.assertEqual(channels, ['x.foot_acc', 'y.foot_acc', 'z.foot_acc'])
            self.assertEqual(targets, ['e1', 'e3'])
            self.assertEqual(len(labels), 4)


if __name__ == '__main__':
    unittest.main()
