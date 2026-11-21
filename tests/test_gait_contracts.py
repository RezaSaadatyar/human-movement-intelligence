"""Offline tests for the Camargo gait-event scientific contracts."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import io
import hashlib
import zipfile
import unittest
import types
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "gait_analysis" / "code" / "src"
sys.path.insert(0, str(SOURCE))

from camargo_source import (inventory_files, paired_trials, validate_timebase,
    read_sensor_table,
    extract_pilot_trial_from_zip, locate_pilot_source_archive,
    download_mendeley_pilot_archive)
from gait_alignment import mapped_series, validated_overlap, candidate_channels, preview_channels
from gait_reference import force_contact_events, reference_strides, stride_summary
from gait_event_detectors import peak_events, candidate_features
from gait_evaluation import (
    match_events, event_metrics, participant_roles, development_folds,
)
from gait_experiments import (
    save_prepared_trial, read_prepared_trial,
    evaluate_trial_collection, participant_selection_score, FIXED_BASELINE,
)


class GaitContractTests(unittest.TestCase):
    def test_colab_setup_imports_public_package_and_registers_study(self):
        """Ensure the repaired Notebook 01 import resolves in a checkout."""
        from python_modules.core.colab_setup import setup_colab

        root = setup_colab(
            "gait_analysis", root=REPO, sync_mode="off",
            module_refresh="off",
        )
        self.assertEqual(root, REPO.resolve())
        from core.plot_style import plot_style
        self.assertTrue(callable(plot_style))

    def test_source_inventory_and_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for sensor in ("imu", "fp"):
                d = root / "AB06" / "2021-01-01" / "treadmill" / sensor
                d.mkdir(parents=True)
                pd.DataFrame({"Header": [0, 0.01, 0.02],
                              "channel": [0.0, 1.0, 0.0]}).to_csv(
                                  d / "trialA.csv", index=False)
            inventory = inventory_files(root)
            self.assertEqual(len(inventory), 2)
            pairs = paired_trials(inventory)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(bool(pairs.candidate_pair.iloc[0]))
            self.assertEqual(pairs.subject_id.iloc[0], "AB06")

    def test_extract_single_original_pilot_trial_from_zip(self):
        """Preserve Camargo paths and extract one matched treadmill pair only."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            zip_path = base / "AB06.zip"
            src = "AB06/10_21_2018/treadmill"
            with zipfile.ZipFile(zip_path, "w") as archive:
                for trial in ("treadmill_01_01", "treadmill_01_02"):
                    for sensor in ("imu", "fp"):
                        archive.writestr(
                            f"{src}/{sensor}/{trial}.csv",
                            "Header,signal\\n0,1\\n0.1,2\\n0.2,3\\n",
                        )
            target = base / "extracted"
            self.assertEqual(locate_pilot_source_archive(target, "AB06",
                                                          zip_path), zip_path)
            result = extract_pilot_trial_from_zip(zip_path, target)
            self.assertEqual(result["subject_id"], "AB06")
            files = sorted(target.rglob("*.csv"))
            self.assertEqual(len(files), 2)
            self.assertEqual(
                sorted(x.parent.name for x in files), ["fp", "imu"]
            )
            pairs = paired_trials(inventory_files(target))
            self.assertEqual(int(pairs.candidate_pair.sum()), 1)
            self.assertEqual(pairs.trial_id.iloc[0], "treadmill_01_01")

    def test_pilot_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            z = base / "AB06.zip"
            with zipfile.ZipFile(z, "w") as archive:
                archive.writestr("../unsafe.csv", "a")
            with self.assertRaises(ValueError):
                extract_pilot_trial_from_zip(z, base / "out")

    def test_official_pilot_download_uses_version_one_and_checks_archive(self):
        """Exercise metadata lookup, single-file streaming and SHA256 offline."""
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("AB06/session/treadmill/imu/trial.csv",
                             "Header,signal\n0,1\n0.1,2\n0.2,3\n")
            archive.writestr("AB06/session/treadmill/fp/trial.csv",
                             "Header,force\n0,0\n0.1,750\n0.2,0\n")
        payload = raw.getvalue()
        metadata = [{
            "id": "12345678-1234-1234-1234-123456789abc",
            "filename": "AB06.zip", "size": len(payload),
            "content_details": {
                "sha256_hash": hashlib.sha256(payload).hexdigest(),
                "download_url": "https://files.mendeley.example/AB06.zip",
            },
        }]

        class FakeResponse(io.BytesIO):
            def __init__(self, value):
                super().__init__(value)
                self.headers = {}

            def getcode(self):
                return 200

        seen = []
        def opener(request, timeout):
            url = request.full_url
            seen.append(url)
            if url.endswith("/files?folder_id=root&version=1"):
                return FakeResponse(json.dumps(metadata).encode())
            if url == "https://files.mendeley.example/AB06.zip":
                return FakeResponse(payload)
            raise AssertionError("Unexpected network path: " + url)

        with tempfile.TemporaryDirectory() as tmp:
            target = download_mendeley_pilot_archive(
                Path(tmp), max_bytes=100_000, opener=opener
            )
            self.assertEqual(target.name, "AB06.zip")
            self.assertTrue(zipfile.is_zipfile(target))
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(len(seen), 2)
            # Rerun reuses the verified archive without further HTTP requests.
            target2 = download_mendeley_pilot_archive(
                Path(tmp), max_bytes=100_000, opener=opener
            )
            self.assertEqual(target, target2)
            self.assertEqual(len(seen), 2)

    def test_mendeley_metadata_fallback_to_official_public_api(self):
        """If anonymous listing is blocked, try the documented files API."""
        from camargo_source import _mendeley_public_file_metadata

        class FakeResponse(io.BytesIO):
            def __init__(self, payload):
                super().__init__(payload)
                self.headers = {}

        seen = []

        def opener(request, timeout):
            seen.append(request.full_url)
            if "/public-api/" in request.full_url:
                raise OSError("Public listing temporarily unavailable")
            if "api.data.mendeley.com/datasets/publics/" in request.full_url:
                return FakeResponse(json.dumps([{
                    "filename": "AB06.zip", "id": "uuid-from-official-api",
                    "size": 1024,
                }]).encode())
            raise AssertionError("Unexpected URL: " + request.full_url)

        found = _mendeley_public_file_metadata(opener=opener)
        self.assertEqual(found["filename"], "AB06.zip")
        self.assertEqual(len(seen), 2)
        self.assertIn("version=1", seen[0])
        self.assertIn("version=1", seen[1])

    def test_download_falls_back_when_first_public_file_url_fails(self):
        """An expired direct URL must not prevent the official redirect."""
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("AB06/s/treadmill/imu/t.csv", "Header,x\n1,2\n")
            archive.writestr("AB06/s/treadmill/fp/t.csv", "Header,x\n1,2\n")
        payload = raw.getvalue()
        info = [{"filename": "AB06.zip", "id": "valid-file-uuid",
                 "size": len(payload),
                 "content_details": {
                     "download_url": "https://files.mendeley.example/expired"
                 }}]

        class FakeResponse(io.BytesIO):
            def __init__(self, payload):
                super().__init__(payload)
                self.headers = {}

            def getcode(self):
                return 200

        seen = []
        def opener(req, timeout):
            seen.append(req.full_url)
            if "folder_id=root&version=1" in req.full_url:
                return FakeResponse(json.dumps(info).encode())
            if req.full_url.endswith("/expired"):
                raise OSError("Signed URL expired")
            if "file_downloaded?version=1" in req.full_url:
                return FakeResponse(payload)
            raise AssertionError(req.full_url)

        with tempfile.TemporaryDirectory() as tmp:
            archive = download_mendeley_pilot_archive(
                Path(tmp), max_bytes=10_000, opener=opener
            )
            self.assertTrue(zipfile.is_zipfile(archive))
            self.assertEqual(archive.read_bytes(), payload)
            self.assertEqual(len(seen), 3)

    def test_carmargo_matlab_table_mcos_fallback_preserves_original_names(self):
        """A MATLAB table is decoded without inventing time or IMU axes."""
        original = pd.DataFrame({
            "Header": [0.0, 0.005, 0.010, 0.015],
            "RightFoot_Gyro_X": [0.1, 0.2, 0.5, 0.3],
            "Shank_accelZ": [-1.0, -0.9, -1.2, -0.7],
        })
        fake = types.ModuleType("matio")
        fake.load_from_mat = lambda path, raw_data=False: {
            "treadmillData": original
        }
        with mock.patch("scipy.io.loadmat", return_value={
            "None": np.array([1, 2, 3], dtype=np.uint32)
        }), mock.patch.dict(sys.modules, {"matio": fake}):
            decoded = read_sensor_table(Path("original_imu.mat"))
        pd.testing.assert_frame_equal(decoded, original)
        np.testing.assert_allclose(
            validate_timebase(decoded), [0.0, 0.005, 0.010, 0.015]
        )

    def test_matlab_numeric_struct_still_uses_scipy(self):
        """Existing simple source dictionaries remain readable."""
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "numeric_trial.mat"
            savemat(path, {"trial": {
                "Header": np.array([0, 0.01, 0.02, 0.03]),
                "force_z": np.array([0, 450, 600, 0]),
            }})
            frame = read_sensor_table(path)
        self.assertEqual(frame.columns.tolist(), ["Header", "force_z"])
        np.testing.assert_allclose(frame["Header"], [0, 0.01, 0.02, 0.03])

    def test_matlab_table_without_source_header_is_rejected(self):
        """No timestamps may be inferred from an arbitrary sample index."""
        fake = types.ModuleType("matio")
        fake.load_from_mat = lambda path, raw_data=False: {
            "table": pd.DataFrame({"gyro": [1.0, 2.0, 3.0]})
        }
        with mock.patch("scipy.io.loadmat", side_effect=ValueError(
            "Opaque MATLAB table"
        )), mock.patch.dict(sys.modules, {"matio": fake}):
            with self.assertRaisesRegex(ValueError, "no named numeric time table"):
                read_sensor_table(Path("invalid_imu.mat"))

    def test_multiple_matlab_time_tables_are_not_chosen_arbitrarily(self):
        """Ambiguous MATLAB source objects must fail rather than mix channels."""
        t = pd.DataFrame({"Header": [0, 1, 2], "signal": [3, 4, 5]})
        fake = types.ModuleType("matio")
        fake.load_from_mat = lambda path, raw_data=False: {
            "a": t, "b": t
        }
        with mock.patch("scipy.io.loadmat", side_effect=ValueError(
            "Opaque MATLAB table"
        )), mock.patch.dict(sys.modules, {"matio": fake}):
            with self.assertRaisesRegex(ValueError, "ambiguous multiple"):
                read_sensor_table(Path("ambiguous_imu.mat"))

    def test_treadmill_source_preview_channel_names(self):
        """Find real AB06 v-component names without inventing vertical GRF."""
        force = pd.DataFrame({
            "Header": [11.99, 11.991, 11.992],
            "Treadmill_R_vy": [0.0, 410.0, 615.0],
            "Treadmill_L_vy": [430.0, 80.0, 0.0],
        })
        imu = pd.DataFrame({
            "Header": [11.99, 11.995, 12.0],
            "foot_Gyro_Y": [0.0, 1.0, 0.0],
            "shank_Gyro_Y": [0.0, 0.5, 0.0],
        })
        self.assertIn("Treadmill_R_vy", candidate_channels(force, "fp"))
        self.assertEqual(
            preview_channels(force, "fp"), ["Treadmill_R_vy", "Treadmill_L_vy"]
        )
        self.assertEqual(
            preview_channels(imu, "imu"), ["foot_Gyro_Y", "shank_Gyro_Y"]
        )
        with self.assertRaises(ValueError):
            preview_channels(imu, "invalid")

    def test_time_contract_requires_monotonicity_and_actual_overlap(self):
        frame = pd.DataFrame({"Header": [0.0, 0.1, 0.2],
                              "foot_Gyro_X": [1.0, 2.0, 3.0]})
        self.assertEqual(len(validate_timebase(frame)), 3)
        np.testing.assert_array_equal(
            mapped_series(frame, "foot_Gyro_X", "gyro"), [1., 2., 3.]
        )
        with self.assertRaises(ValueError):
            validate_timebase(pd.DataFrame({"Header": [0, 0, 1]}))
        with self.assertRaises(ValueError):
            mapped_series(frame, "unreviewed", "gyro")
        with self.assertRaises(ValueError):
            validated_overlap(np.arange(0, 5), np.arange(10, 15))
        self.assertEqual(validated_overlap(np.arange(0, 10),
                                           np.arange(2, 9)), (2., 8.))

    def test_force_events_and_stride_units(self):
        t = np.arange(0, 6, .001)
        force = np.zeros(len(t))
        for start in (1., 2., 3., 4.):
            force[(t >= start) & (t < start + .5)] = 750.
        events = force_contact_events(t, force)
        self.assertEqual(len(events.loc[events.event_type == "IC"]), 4)
        self.assertEqual(len(events.loc[events.event_type == "TC"]), 4)
        stride = reference_strides(events)
        self.assertEqual(len(stride), 3)
        self.assertAlmostEqual(stride.stride_time_s.mean(), 1.0, places=3)
        self.assertAlmostEqual(stride.cadence_steps_per_min.mean(), 120.0)
        self.assertAlmostEqual(stride_summary(stride)["stride_time_cv"], 0.0)

    def test_one_to_one_matching_preserves_misses_and_extras(self):
        events, missed, extra = match_events(
            [1.0, 2.0, 3.0], [1.02, 1.04, 3.01], .15
        )
        self.assertEqual((len(events), missed, extra), (2, 1, 1))
        metrics = event_metrics([1, 2, 3], [1.02, 1.04, 3.01])
        self.assertEqual(metrics["n_matched"], 2)
        self.assertEqual(metrics["n_missed"], 1)
        self.assertEqual(metrics["n_extra"], 1)
        self.assertAlmostEqual(metrics["sensitivity"], 2/3)
        empty = event_metrics([1, 2], [])
        self.assertTrue(np.isnan(empty["timing_mae_ms"]))
        self.assertEqual(empty["n_missed"], 2)

    def test_grouped_split_is_deterministic_and_disjoint(self):
        people = [f"AB{i:02d}" for i in range(1, 13)]
        a = participant_roles(people)
        b = participant_roles(reversed(people))
        pd.testing.assert_frame_equal(a, b)
        dev = set(a.loc[a.role == "Development", "subject_id"])
        test = set(a.loc[a.role == "Test", "subject_id"])
        self.assertFalse(dev & test)
        folds = development_folds(a)
        self.assertEqual(set(folds.subject_id), dev)
        self.assertEqual(folds.validation_fold.nunique(), 5)

    def test_prepared_trial_and_detector_with_synthetic_wave(self):
        t = np.arange(0, 10, .005)
        gyro = np.sin(2 * np.pi * t)
        with tempfile.TemporaryDirectory() as folder:
            save_prepared_trial(Path(folder), "AB06_trialA", t, gyro,
                                np.array([1., 2., 3.]))
            t2, gyro2, reference = read_prepared_trial(folder, "AB06_trialA")
            self.assertEqual(len(t), len(t2))
            self.assertEqual(len(reference), 3)
        detections = peak_events(t, gyro, polarity=1,
                                 prominence=.5, min_interval_s=.45)
        self.assertGreater(len(detections), 5)
        times, features = candidate_features(t, gyro)
        self.assertEqual(len(times), len(features))
        rows = evaluate_trial_collection(
            [({"subject_id": "AB06", "trial_key": "AB06_trialA"}, (
                t2, gyro2, reference
            ))],
            FIXED_BASELINE, "B0_fixed",
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isfinite(participant_selection_score(rows)))

    def test_source_summary_and_download_share_notebook_final_section(self):
        """Gait Notebook 01 ends in one numbered scientific summary step."""
        path = REPO / "gait_analysis/code/notebooks" / (
            "01_source_and_reference_eligibility.ipynb"
        )
        nb = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(len(nb["cells"]), 17)
        self.assertEqual(nb["cells"][-2]["id"], "m-01-07")
        self.assertEqual(nb["cells"][-1]["id"], "c-01-07")
        self.assertIn(
            "**07. Research Summary and Output Download**",
            "".join(nb["cells"][-2]["source"]),
        )
        code = "".join(nb["cells"][-1]["source"])
        self.assertIn("01_source_eligibility_summary.csv", code)
        self.assertIn("download_outputs_from_colab(", code)
        self.assertNotIn("Next: review", code)
        self.assertFalse(any(
            "Research Summary and Output Download" in "".join(c["source"])
            for c in nb["cells"][:-2]
        ))

    def test_gait_final_cells_package_their_own_outputs(self):
        """Each numbered notebook ends with the approved download helper."""
        folder = REPO / "gait_analysis" / "code" / "notebooks"
        for number, path in enumerate(sorted(folder.glob("*.ipynb")), 1):
            nb = json.loads(path.read_text(encoding="utf8"))
            imports = "".join([
                c for c in nb["cells"] if c["cell_type"] == "code"
            ][1]["source"])
            self.assertIn("from core.output_download import "
                          "download_outputs_from_colab", imports)
            self.assertIn(f"NOTEBOOK_NUMBER = {number}", imports)
            self.assertIn("Research Summary and Output Download",
                          "".join(nb["cells"][-2]["source"]))
            last = "".join(nb["cells"][-1]["source"])
            self.assertIn('folder="gait_analysis/output"', last)
            self.assertIn('stage=f"{NOTEBOOK_NUMBER:02d}"', last)

    def test_gait_figures_have_editable_signal_and_event_panels(self):
        """Six conceptual SVGs retain fixed five-band design and annotations."""
        from xml.etree import ElementTree as ET

        images = REPO / "gait_analysis" / "images"
        figures = sorted(images.glob("*.svg"))
        self.assertEqual(len(figures), 6)
        for figure in figures:
            element = ET.parse(figure).getroot()
            self.assertEqual(element.attrib.get("viewBox"), "0 0 1700 816")
            ids = [child.attrib["id"] for child in element.iter()
                   if "id" in child.attrib]
            self.assertEqual(len(ids), len(set(ids)))
            for number in range(5):
                self.assertTrue(
                    any(value.startswith(f"band-{number}-")
                        for value in ids)
                )
            for suffix in ("imu-wave", "force-wave"):
                self.assertIn(f"panel-0-{suffix}", ids)
            self.assertIn("panel-1-pulses", ids)
            self.assertIn("panel-2-ref-line", ids)
            text = " ".join(child.text or "" for child in element.iter())
            self.assertIn("Schematic curves only", text)
            self.assertIn("unverified", text)

    def test_source_plot_produces_two_numeric_panels_offline(self):
        """Execute just the Notebook 01 plot cell with synthetic source rows."""
        import types
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        source = REPO / "gait_analysis/code/notebooks"
        path = source / "01_source_and_reference_eligibility.ipynb"
        nb = json.loads(path.read_text(encoding="utf8"))
        cell = next(c for c in nb["cells"] if c.get("id") == "c-01-06")
        imu_t = np.arange(12., 17., 0.005)
        fp_t = np.arange(12., 17., 0.001)
        imu = pd.DataFrame({
            "Header": imu_t,
            "foot_Gyro_Y": np.sin(imu_t),
            "shank_Gyro_Y": np.cos(imu_t),
        })
        force = pd.DataFrame({
            "Header": fp_t,
            "Treadmill_R_vy": 400 + 200 * np.sin(fp_t),
            "Treadmill_L_vy": 450 - 200 * np.sin(fp_t),
        })
        saved = []
        variables = {
            "imu_table": imu, "fp_table": force,
            "imu_time": imu_t, "fp_time": fp_t,
            "np": np, "pd": pd, "plt": plt,
            "sample": types.SimpleNamespace(
                subject_id="AB06", trial_id="treadmill_01_01"
            ),
            "preview_channels": preview_channels,
            "output_dir": Path("/not/used"),
            "save_figure": lambda fig, output: saved.append(str(output)),
        }
        try:
            exec(compile("".join(cell["source"]), path.name, "exec"), variables)
            self.assertEqual(len(variables["axes"][0].lines), 2)
            self.assertEqual(len(variables["axes"][1].lines), 2)
            self.assertTrue(saved[0].endswith("01_sensor_source_preview.png"))
        finally:
            if "fig" in variables:
                plt.close(variables["fig"])

    def test_notebook_code_static_syntax_and_scientific_stages(self):
        import ast
        paths = sorted((REPO / "gait_analysis/code/notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            nb = json.loads(path.read_text(encoding="utf8"))
            self.assertEqual(nb["nbformat"], 4)
            self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
            code = [c for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertTrue(code)
            environment = "".join(code[1]["source"])
            self.assertIn(
                "from python_modules.core.colab_setup import setup_colab",
                environment,
            )
            self.assertNotIn("from core.colab_setup import setup_colab", environment)
            if path.name.startswith(("01_", "02_")):
                self.assertIn("mat-io==1.0.2", environment)
                self.assertIn('find_spec("matio")', environment)
            for cell in code:
                headings = [
                    line for line in "".join(cell["source"]).splitlines()
                    if line.startswith("# ") and
                    (line[2:4] == "==" or line[2:4] == "--")
                ]
                self.assertTrue(headings)
                self.assertTrue(all(len(line) == 79 for line in headings))
            for cell in code:
                source = "".join(cell["source"])
                # Replace Colab's valid IPython magic for standard Python AST.
                source = "\n".join(
                    "pass  # IPython magic"
                    if line.lstrip().startswith("%") else line
                    for line in source.split("\n")
                )
                ast.parse(source, filename=path.name)


if __name__ == "__main__":
    unittest.main()
