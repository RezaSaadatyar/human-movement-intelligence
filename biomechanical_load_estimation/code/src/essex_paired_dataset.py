# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Verified original Essex numerical pairing and participant-level partitioning.

No source FE target imputation, numerical unit conversion, interpolation,
model fitting, cross-participant leakage or original-file editing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from essex_figshare import (
    ensure_matlab_archive,
    inspect_matlab_archive,
    extract_matlab_files,
)
from essex_python_first import _original_text, original_records, source_field
from essex_temporal_mapping import describe_source_empty, match_original_coordinates
from essex_timebase_audit import EXPECTED_CHANNELS, validate_imu, validate_fe


# ================== 01. Original Data and Mapping Integrity ==================
def build_paired_arrays(reader, mat_dir, registry, mapping):
    """Build source-exact 101-row IMU and 11-row IMU/FE numerical pairs."""
    required = {
        "local_record_key",
        "source_file",
        "subject_id",
        "task",
        "updated_trial_id",
        "local_record_index",
        "fe_representation",
    }
    if not required.issubset(registry.columns):
        raise ValueError("Missing original record-registry fields.")
    if not {
        "local_record_key",
        "mapping_status",
        "imu_position_indices",
        "originally_empty_fe",
    }.issubset(mapping.columns):
        raise ValueError("Incomplete original FE time-mapping evidence.")
    if (
        len(registry) != len(mapping)
        or not registry["local_record_key"].is_unique
        or not mapping["local_record_key"].is_unique
        or set(registry["local_record_key"]) != set(mapping["local_record_key"])
    ):
        raise ValueError("Stage 01 and Stage 03 original trial keys mismatch.")
    indexed = mapping.set_index("local_record_key")
    x_full, x_at_fe, y_fe, indexes = [], [], [], []
    eligible, excluded = [], []
    for filename, group in registry.groupby("source_file", sort=True):
        path = Path(mat_dir) / str(filename)
        raw = reader(str(path), variable_names=["Data"], raw_data=False)
        if not isinstance(raw, dict) or "Data" not in raw:
            raise ValueError("MAT file contains no original Data variable.")
        records = original_records(raw["Data"])
        if len(records) != len(group):
            raise ValueError("Original MAT record count changed.")
        for row in group.sort_values("local_record_index").itertuples(index=False):
            key = str(row.local_record_key)
            data = records[int(row.local_record_index)]
            identity = (
                _original_text(source_field(data, "SubjID")),
                _original_text(source_field(data, "Task")),
                _original_text(source_field(data, "UpdatedTrialID")),
            )
            if (
                identity[0] is None
                or identity[0].lower() != row.subject_id
                or identity[1] != row.task
                or identity[2] != row.updated_trial_id
            ):
                raise ValueError("Original identity mismatch: " + key)
            m = indexed.loc[key]
            is_empty = row.fe_representation == "empty"
            if bool(m["originally_empty_fe"]) != is_empty:
                raise ValueError("Original FE missingness mismatch: " + key)
            imu = source_field(data, "IMU")
            fe = source_field(data, "e1e3")
            imu_status, imu_time = validate_imu(imu)
            if imu_status["imu_status"] != "VALID_CHANNELS" or imu_time is None:
                raise ValueError("Original IMU numeric audit failed: " + key)
            if is_empty:
                empty = describe_source_empty(fe)
                if (
                    m["mapping_status"] != "NO_SOURCE_FE_TARGET"
                    or empty["empty_representation_status"]
                    != "READER_EMPTY_REPRESENTATION"
                ):
                    raise ValueError("Source-empty FE representation unresolved.")
                excluded.append(
                    {
                        "local_record_key": key,
                        "subject_id": row.subject_id,
                        "task": row.task,
                        "source_file": filename,
                        "reason": "SOURCE_FE_TARGET_EMPTY",
                        "reader_class": empty["python_class"],
                    }
                )
                continue
            fe_status, fe_time = validate_fe(fe, False)
            if fe_status["fe_status"] != "VALID_STRAIN_COLUMNS" or fe_time is None:
                raise ValueError("Original FE numeric audit failed: " + key)
            observed, values = match_original_coordinates(imu_time, fe_time)
            if (
                observed["mapping_status"] != "EXACT_TIME_COORDINATE_MAPPING"
                or m["mapping_status"] != observed["mapping_status"]
                or not bool(m["source_identity_matches"])
            ):
                raise ValueError("Original FE–IMU coordinate mapping mismatch.")
            positions = np.asarray(values["imu_indices"], dtype=np.int64)
            if positions.tolist() != json.loads(m["imu_position_indices"]):
                raise ValueError("Original mapped time indices were modified.")
            feature = imu.loc[:, list(EXPECTED_CHANNELS)].to_numpy(
                dtype=np.float64, copy=True
            )
            target = fe.loc[:, ["e1", "e3"]].to_numpy(dtype=np.float64, copy=True)
            x_full.append(feature)
            x_at_fe.append(feature[positions, :])
            y_fe.append(target)
            indexes.append(positions)
            eligible.append(
                {
                    "local_record_key": key,
                    "subject_id": row.subject_id,
                    "source_file": filename,
                    "local_record_index": int(row.local_record_index),
                    "task": row.task,
                    "updated_trial_id": row.updated_trial_id,
                    "time_mapping": "EXACT_TIME_COORDINATE_MAPPING",
                }
            )
    if not eligible:
        raise ValueError("No usable IMU–FE pairs were found in the current data.")
    arrays = {
        "X_full": np.stack(x_full),
        "X_at_fe": np.stack(x_at_fe),
        "y_fe": np.stack(y_fe),
        "imu_positions": np.stack(indexes),
    }
    return (
        arrays,
        pd.DataFrame(eligible),
        pd.DataFrame(
            excluded,
            columns=[
                "local_record_key",
                "subject_id",
                "task",
                "source_file",
                "reason",
                "reader_class",
            ],
        ),
    )


# ================= 02. Participant-Only Deterministic Splits =================
def fixed_subject_splits(eligible, registry):
    """Use the study's existing participant allocation without hashing IDs."""
    all_subjects = sorted(registry["subject_id"].unique())
    good_subjects = set(eligible["subject_id"])
    test = {"subj08", "subj17", "subj20", "subj33"}
    validation = {"subj10", "subj21", "subj24", "subj41"}

    def label(name):
        if name not in good_subjects:
            return "EXCLUDED_NO_FE_TARGET"
        return (
            "TEST" if name in test else "VALIDATION" if name in validation else "TRAIN"
        )

    subjects = pd.DataFrame(
        {"subject_id": all_subjects, "split": [label(name) for name in all_subjects]}
    )
    assigned = eligible.copy()
    assigned["split"] = assigned["subject_id"].map(label)
    summary = (
        assigned.groupby("split")
        .agg(
            participants=("subject_id", "nunique"), trials=("local_record_key", "size")
        )
        .reset_index()
    )
    activities = (
        assigned.groupby(["split", "task"], dropna=False)
        .size()
        .rename("trials")
        .reset_index()
    )
    return subjects, assigned, summary, activities


# =========== 03. Training-Only Statistics and Ignored Array Cache ============
def training_channel_stats(arrays, assigned):
    """Compute only training-subject statistics; leave source arrays unchanged."""
    mask = assigned["split"].to_numpy() == "TRAIN"
    if not mask.any():
        raise ValueError("Training partition contains no source trials.")
    values = arrays["X_at_fe"][mask].reshape(-1, 42)
    mean, std = values.mean(axis=0), values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Nonfinite training-only IMU statistics.")
    return pd.DataFrame(
        {
            "channel": EXPECTED_CHANNELS,
            "train_mean": mean,
            "train_std": std,
            "zero_std": std == 0,
            "safe_denominator": np.where(std == 0, 1.0, std),
            "train_sample_rows": len(values),
        }
    )


def save_paired_data(path, arrays, assigned):
    """Replace the current local NPZ with this run's arrays and trial keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **arrays,
                record_keys=assigned["local_record_key"].to_numpy(dtype=str),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_trial_splits(output_dir):
    """Read current assignments and reject ambiguous participant allocation."""
    assigned = pd.read_csv(
        Path(output_dir) / "04_trial_split_assignments.csv", dtype=str
    )
    if not assigned["local_record_key"].is_unique:
        raise ValueError("Trial keys must be unique to align arrays and metadata.")
    if assigned.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("A participant appears in more than one data split.")
    if assigned["split"].isna().any() or not set(assigned["split"]) <= {
        "TRAIN",
        "VALIDATION",
        "TEST",
    }:
        raise ValueError("Unknown or missing participant split.")
    return assigned


def read_training_folds(output_dir, training_trials):
    """Align current fold labels to trials while keeping each participant together."""
    folds = pd.read_csv(
        Path(output_dir) / "05_training_fold_registry.csv",
        dtype={"local_record_key": str, "subject_id": str, "task": str},
    )
    if not folds["local_record_key"].is_unique:
        raise ValueError("Training-fold trial keys must be unique.")
    aligned = (
        folds.set_index("local_record_key")
        .loc[training_trials["local_record_key"]]
        .reset_index()
    )
    if not np.array_equal(aligned["subject_id"], training_trials["subject_id"]):
        raise ValueError("Training-fold participants do not match the current trials.")
    if (
        aligned["validation_fold"].isna().any()
        or aligned.groupby("subject_id")["validation_fold"].nunique().gt(1).any()
    ):
        raise ValueError("Each Training participant must belong to one fold.")
    return aligned


def load_paired_data(output_dir, processed_dir, raw_dir, reader, assigned):
    """Use current arrays; rebuild when absent or older than upstream tables.

    No historical versions or hashes are read. Arrays are aligned to the
    current trial table by key, not by an assumed row order.
    """
    output_dir, processed_dir, raw_dir = map(Path, (output_dir, processed_dir, raw_dir))
    cache = processed_dir / "essex_imu_fe_pairs_v13.npz"
    inputs = [
        output_dir / name
        for name in (
            "01_original_record_registry.csv",
            "03_original_time_coordinate_mapping.csv",
            "04_trial_split_assignments.csv",
        )
    ]
    # A newly downloaded/replaced source archive also makes the arrays stale.
    archive = raw_dir / "MATLAB.zip"
    if archive.is_file():
        inputs.append(archive)
    rebuild = not cache.is_file() or any(
        p.stat().st_mtime_ns > cache.stat().st_mtime_ns for p in inputs
    )
    if rebuild:
        archive = ensure_matlab_archive(raw_dir)
        mat_dir = raw_dir / "matlab_records"
        extract_matlab_files(archive, mat_dir, inspect_matlab_archive(archive))
        registry = pd.read_csv(
            inputs[0],
            dtype={
                "subject_id": str,
                "source_file": str,
                "local_record_key": str,
                "task": str,
                "updated_trial_id": str,
            },
        )
        mapping = pd.read_csv(
            inputs[1], dtype={"local_record_key": str, "imu_position_indices": str}
        )
        arrays, eligible, _ = build_paired_arrays(reader, mat_dir, registry, mapping)
        save_paired_data(cache, arrays, eligible)
    with np.load(cache, allow_pickle=False) as saved:
        keys = pd.Index(saved["record_keys"].astype(str))
        if not keys.is_unique:
            raise ValueError(
                "Duplicate trial keys in paired arrays; rerun Notebook 04."
            )
        order = keys.get_indexer(assigned["local_record_key"])
        if (order < 0).any() or len(keys) != len(assigned):
            raise ValueError(
                "Current arrays and trial table differ; rerun Notebook 04."
            )
        return {
            name: saved[name][order].copy()
            for name in ("X_full", "X_at_fe", "y_fe", "imu_positions")
        }
