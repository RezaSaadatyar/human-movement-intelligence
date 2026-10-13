"""Python-only ingestion and subject-held-out pilot for the authors' BoneNet derivative.

Presented by: Reza Saadatyar (2026)
This is not the complete six-task Figshare dataset. Training requires explicit
acceptance of the authors' row-order labels, whose alignment is unverified.
No MATLAB executable or engine is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat, whosmat

REFERENCE_REPO = (
    "https://raw.githubusercontent.com/ZainabAltai/"
    "BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-"
    "Femoral-Neck-strains-from-Exercise-Data/main/"
)
SOURCE_FILES = {"Data.mat", "labels.csv"}
DEFAULT_ROOT = Path("data/raw/bonenet_reference")
DEFAULT_REPORT = Path("outputs/bonenet_python_pilot_audit.json")


def sha256_file(path: Path) -> str:
    """Return the local file digest for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(name: str, root: Path, *, size_limit_mb: int = 64) -> dict[str, Any]:
    """Download one known source file, with size cap and no silent overwrite."""
    if name not in SOURCE_FILES:
        raise ValueError(f"Unknown remote file: {name}")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    if dest.exists():
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ValueError(f"Invalid cached file: {dest}")
        return {"filename": name, "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest), "retrieval": "cached"}
    partial = dest.with_name(dest.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Check existing partial file: {partial}")
    req = Request(REFERENCE_REPO + name, headers={
        "User-Agent": "human-movement-intelligence/0.1"})
    count = 0
    digest = hashlib.sha256()
    try:
        with urlopen(req, timeout=120) as response, partial.open("xb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > size_limit_mb * 1024 * 1024:
                    raise ValueError(f"Download exceeds {size_limit_mb} MB cap")
                output.write(block)
                digest.update(block)
        if count == 0:
            raise ValueError("Remote file is empty")
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {"filename": name, "bytes": count,
            "sha256": digest.hexdigest(), "retrieval": "downloaded"}


def read_labels(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read official derivative labels and audit non-unique subject/trial keys."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if fields != ["SubjID", "Task", "TrialNum"]:
            raise ValueError(f"Unexpected label columns: {fields}")
        rows = list(reader)
    if not rows or any(not all(row.get(k) for k in fields) for row in rows):
        raise ValueError("Incomplete or empty labels")
    keys = [(r["SubjID"].strip().casefold(), r["Task"].strip(),
             r["TrialNum"].strip()) for r in rows]
    return rows, {
        "label_rows": len(rows),
        "subjects": len(set(k[0] for k in keys)),
        "activities": dict(sorted(Counter(r["Task"] for r in rows).items())),
        "duplicate_SubjID_Task_TrialNum": len(keys) - len(set(keys)),
        "source_row_order_independently_verified": False,
    }


def numeric_matrix(obj: Any, key: str) -> np.ndarray:
    """Reject opaque MATLAB objects and accept numeric 2D arrays only."""
    array = np.asarray(obj)
    if array.ndim != 2 or array.dtype.kind not in "biuf":
        raise ValueError(
            f"{key}: expected 2D numeric array; got {array.shape} / {array.dtype}")
    if min(array.shape) < 2:
        raise ValueError(f"{key}: unexpectedly small {array.shape}")
    return array.astype(np.float64, copy=False)


def stack_components(obj: Any, section: str) -> tuple[np.ndarray, list[str]]:
    """Stack fields in stored MATLAB struct order (records, time, channels)."""
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{section} is not a plain MATLAB numeric struct")
    arrays, names = [], []
    for name, values in obj.items():
        arr = numeric_matrix(values, f"{section}.{name}")
        if arrays and arr.shape != arrays[0].shape:
            raise ValueError(f"{section}.{name}: mismatched shape {arr.shape}")
        arrays.append(arr)
        names.append(str(name))
    return np.stack(arrays, axis=-1), names


def assemble_reference(
    data: dict[str, Any], label_count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the authors' axis-stacking recipe with explicit shape checks."""
    if not isinstance(data.get("inputs"), dict) or not isinstance(data.get("outputs"), dict):
        raise ValueError("Data.mat lacks numeric inputs/outputs MATLAB structs")
    axes, feature_names = [], []
    for axis in ("x", "y", "z"):
        block, names = stack_components(data["inputs"].get(axis), f"inputs.{axis}")
        axes.append(block)
        feature_names.extend(f"{axis}.{name}" for name in names)
    if any(a.shape[:2] != axes[0].shape[:2] for a in axes):
        raise ValueError("Different IMU axes have mismatched record/time dimensions")
    raw_x = np.concatenate(axes, axis=-1)
    y, targets = stack_components(data["outputs"], "outputs")
    if raw_x.shape[0] != label_count or y.shape[0] != label_count:
        raise ValueError(
            f"Sample count mismatch: IMU={raw_x.shape[0]}, "
            f"targets={y.shape[0]}, labels={label_count}; cannot infer pairing.")
    x = np.moveaxis(raw_x, -1, 1)
    return x, y, {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "feature_names": feature_names,
        "target_names": targets,
        "input_nonfinite": int(np.count_nonzero(~np.isfinite(x))),
        "target_nonfinite": int(np.count_nonzero(~np.isfinite(y))),
    }


def inspect_source(
    root: Path
) -> tuple[dict[str, Any], list[dict[str, str]], np.ndarray | None, np.ndarray | None]:
    """Audit the derivative before any model training."""
    rows, label_info = read_labels(root / "labels.csv")
    mat = root / "Data.mat"
    fields = [{"name": name, "shape": list(shape), "type": dtype}
              for name, shape, dtype in whosmat(str(mat))]
    report: dict[str, Any] = {
        "source": "Authors' BoneNet derivative, not the full six-task Figshare archive",
        "source_url": REFERENCE_REPO.rsplit("/", 1)[0],
        "data_mat_sha256": sha256_file(mat),
        "labels_sha256": sha256_file(root / "labels.csv"),
        "mat_fields": fields,
        "labels": label_info,
        "training_ready": False,
        "blockers": [],
    }
    try:
        data = loadmat(
            str(mat), simplify_cells=True,
            variable_names=["inputs", "outputs"],
            verify_compressed_data_integrity=True,
        )
        x, y, info = assemble_reference(data, len(rows))
        report["numeric_arrays"] = info
        if info["input_nonfinite"]:
            report["blockers"].append("Non-finite inputs require data-quality review")
        if info["target_nonfinite"]:
            report["blockers"].append("Non-finite targets require data-quality review")
        report["training_ready"] = not report["blockers"]
    except (ValueError, TypeError, KeyError, NotImplementedError) as error:
        x, y = None, None
        report["blockers"].append(str(error))
    report["limitations"] = [
        "Derivative represents the authors' walking/running subset only.",
        "Condition/intensity is not uniquely encoded by labels.csv.",
        "Row-order association is used in source notebooks but not independently verified.",
        "Numeric tensors and target units must be verified before scientific claims.",
    ]
    return report, rows, x, y


def run_baseline(
    x: np.ndarray, y: np.ndarray, labels: list[dict[str, str]],
    *, seed: int = 42, test_fraction: float = 0.25
) -> dict[str, Any]:
    """Train-only standardisation and subject-held-out mean/Ridge baselines."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if x.shape[0] != len(labels) or y.shape[0] != len(labels):
        raise ValueError("Record count does not match the source labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NaN/Inf present; refusing to train")
    groups = np.array([r["SubjID"].strip().casefold() for r in labels])
    if len(set(groups)) < 5:
        raise ValueError("Too few independent participants")
    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train, test = next(split.split(x, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise RuntimeError("Participant leakage")
    features = np.concatenate((x.mean(axis=2), x.std(axis=2)), axis=1)
    mean_prediction = np.broadcast_to(y[train].mean(axis=0), y[test].shape).copy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(features[train], y[train].reshape(len(train), -1))
    ridge_prediction = model.predict(features[test]).reshape(y[test].shape)
    results = {}
    for name, pred in {
        "training_mean": mean_prediction, "ridge_descriptors": ridge_prediction
    }.items():
        results[name] = {
            "MAE": float(mean_absolute_error(y[test].ravel(), pred.ravel())),
            "RMSE": float(math.sqrt(mean_squared_error(y[test].ravel(), pred.ravel()))),
        }
    return {
        "status": "exploratory_subject_held_out_replication",
        "group_split": "subject-level",
        "seed": seed,
        "train_samples": len(train),
        "test_samples": len(test),
        "train_subjects": len(set(groups[train])),
        "test_subjects": len(set(groups[test])),
        "metrics": results,
        "caveat": "Row-order label alignment is author-assumed, not independently proven.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--download", action="store_true",
                        help="Download public authors' Data.mat and labels.csv (~14 MB).")
    parser.add_argument("--train", action="store_true",
                        help="Attempt a leakage-aware exploratory baseline.")
    parser.add_argument("--accept-order-based-label-alignment", action="store_true",
                        help="Acknowledge unverified sample-to-label row-order pairing.")
    args = parser.parse_args()
    if args.train and not args.accept_order_based_label_alignment:
        parser.error("--train requires --accept-order-based-label-alignment")
    try:
        downloads = []
        if args.download:
            for name in ("Data.mat", "labels.csv"):
                item = fetch_file(name, args.root)
                downloads.append(item)
                print(f"{name}: {item['retrieval']} ({item['bytes']} bytes)")
        for name in SOURCE_FILES:
            if not (args.root / name).is_file():
                parser.error(f"Missing {name}; rerun with --download.")
        report, rows, x, y = inspect_source(args.root)
        report["downloads"] = downloads
        print(f"Labels: {report['labels']['label_rows']}; "
              f"subjects: {report['labels']['subjects']}")
        print(f"MAT fields: {[f['name'] for f in report['mat_fields']]}")
        if "numeric_arrays" in report:
            print(f"X={report['numeric_arrays']['input_shape']} "
                  f"Y={report['numeric_arrays']['output_shape']}")
        if report["blockers"]:
            print("Numeric training blocked: " + "; ".join(report["blockers"]))
        if args.train:
            if not report["training_ready"] or x is None or y is None:
                raise ValueError("Source numeric contract not satisfied: refusing training")
            report["exploratory_baseline"] = run_baseline(x, y, rows)
            print(f"Subject-held-out results: {report['exploratory_baseline']['metrics']}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Audit report: {args.report}")
    except (HTTPError, URLError, OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Python pilot error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
