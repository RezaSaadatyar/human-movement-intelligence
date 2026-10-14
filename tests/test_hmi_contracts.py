"""Offline tests of the reusable HMI data contract."""
import unittest

import numpy as np

from hmi.data import DatasetBundle


class TestDatasetBundle(unittest.TestCase):
    def make_bundle(self) -> DatasetBundle:
        return DatasetBundle(
            dataset_id="synthetic_wearables",
            inputs={"imu": np.ones((6, 3, 101))},
            targets={"movement": np.array([0, 1, 0, 1, 0, 1])},
            subject_ids=["a", "a", "b", "b", "c", "c"],
            sample_ids=[f"record_{i}" for i in range(6)],
            input_axes={"imu": ("sample", "channel", "time")},
            target_axes={"movement": ("sample",)},
            units={"imu": "not_verified"},
        )

    def test_multidimensional_dataset(self):
        bundle = self.make_bundle()
        self.assertEqual(bundle.n_samples, 6)
        self.assertEqual(bundle.n_subjects, 3)
        profile = bundle.summary()
        self.assertEqual(profile["inputs"]["imu"]["shape"], [6, 3, 101])
        self.assertEqual(profile["inputs"]["imu"]["nonfinite_count"], 0)

    def test_bad_first_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 6 observations"):
            DatasetBundle("bad", {"imu": np.ones((5, 3))}, ["a"] * 6,
                          [f"row_{i}" for i in range(6)])

    def test_duplicate_sample_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            DatasetBundle("bad", {"imu": np.ones((2, 3))},
                          ["a", "a"], ["one", "one"])

    def test_missing_subject_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            DatasetBundle("bad", {"imu": np.ones((2, 3))},
                          ["a", ""], ["one", "two"])

    def test_invalid_axis_declaration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "start with 'sample'"):
            DatasetBundle("bad", {"imu": np.ones((2, 3, 5))},
                          ["a", "b"], ["one", "two"],
                          input_axes={"imu": ("channel", "sample", "time")})

    def test_multiple_modalities_and_missing_value_count(self):
        bundle = DatasetBundle(
            dataset_id="fusion_demo",
            inputs={"imu": np.ones((3, 6, 100)),
                    "emg": np.full((3, 4, 100), np.nan)},
            subject_ids=["a", "b", "c"],
            sample_ids=["one", "two", "three"],
        )
        self.assertEqual(bundle.summary()["inputs"]["emg"]["nonfinite_count"], 1200)

    def test_tabular_inputs_supported(self):
        bundle = DatasetBundle(
            dataset_id="tabular_demo",
            inputs={"clinical_features": np.ones((5, 10))},
            subject_ids=[f"s{i}" for i in range(5)],
            sample_ids=[f"row{i}" for i in range(5)],
        )
        self.assertEqual(bundle.summary()["inputs"]["clinical_features"]["shape"], [5, 10])


if __name__ == "__main__":
    unittest.main()
