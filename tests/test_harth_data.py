"""Offline tests: HARTH adapter reuses the common CSV reader and inspector."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "python" / "src"))
sys.path.insert(0, str(ROOT / "studies" / "wearable_activity_recognition" / "src"))

from core.data_inspection import inspect_data, plot_signal
from harth_data import SENSOR_COLUMNS, load_harth_subject


class TestHarthData(unittest.TestCase):
    def source_table(self, n=20):
        times = pd.date_range("2025-01-01", periods=n, freq="20ms")
        table = pd.DataFrame({"timestamp": times})
        for index, name in enumerate(SENSOR_COLUMNS):
            table[name] = np.linspace(0, 1, n) + index
        table["label"] = [1] * (n // 2) + [6] * (n - n // 2)
        return table

    def test_read_harth_csv_with_generic_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S023.csv"
            self.source_table().to_csv(path, index=False)
            data, channels = load_harth_subject(path)
            self.assertEqual(data.shape, (20, 8))
            self.assertEqual(channels, SENSOR_COLUMNS)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(data.timestamp))
            self.assertEqual(inspect_data(data)["missing_values"], 0)

    def test_signal_plot(self):
        from matplotlib import pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S023.csv"
            self.source_table().to_csv(path, index=False)
            data, channels = load_harth_subject(path)
            fig = plot_signal(data.loc[:9, channels[:3]].to_numpy().T,
                              channel_names=channels[:3])
            self.assertEqual(len(fig.axes[0].lines), 3)
            plt.close(fig)

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S023.csv"
            self.source_table().drop(columns=["thigh_z"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_harth_subject(path)

    def test_unknown_activity_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S023.csv"
            table = self.source_table()
            table.loc[0, "label"] = 999
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected HARTH activity"):
                load_harth_subject(path)


if __name__ == "__main__":
    unittest.main()
