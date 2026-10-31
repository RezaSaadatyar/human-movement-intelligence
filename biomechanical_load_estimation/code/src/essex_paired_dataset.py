# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Verified original Essex numerical pairing and participant-level partitioning.

No source FE target imputation, numerical unit conversion, interpolation,
model fitting, cross-participant leakage or original-file editing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from essex_figshare import file_hashes
from essex_python_first import _original_text, original_records, source_field
from essex_temporal_mapping import describe_source_empty, match_original_coordinates
from essex_timebase_audit import EXPECTED_CHANNELS, validate_imu, validate_fe


# ================== 01. Original Data and Mapping Integrity ==================
def build_paired_arrays(reader, mat_dir, registry, mapping, source_gate):
    """Build source-exact 101-row IMU and 11-row IMU/FE numerical pairs."""
    required = {
        "local_record_key", "source_file", "subject_id", "task",
        "updated_trial_id", "local_record_index", "fe_representation",
    }
    if not required.issubset(registry.columns):
        raise ValueError("Missing original record-registry fields.")
    if not {"local_record_key", "mapping_status", "imu_position_indices",
            "originally_empty_fe"}.issubset(mapping.columns):
        raise ValueError("Incomplete original FE time-mapping evidence.")
    if (len(registry) != len(mapping) or len(registry) != 1897
            or not registry["local_record_key"].is_unique
            or not mapping["local_record_key"].is_unique
            or set(registry["local_record_key"]) != set(mapping["local_record_key"])):
        raise ValueError("Stage 01 and Stage 03 original trial keys mismatch.")
    indexed = mapping.set_index("local_record_key")
    x_full, x_at_fe, y_fe, indexes = [], [], [], []
    eligible, excluded = [], []
    for filename, group in registry.groupby("source_file", sort=True):
        path = Path(mat_dir) / str(filename)
        if path.name not in source_gate["per_file_sha256"]:
            raise ValueError("Original source file absent from approved manifest.")
        if file_hashes(path)["sha256"] != source_gate["per_file_sha256"][path.name]:
            raise ValueError("Original source-file SHA256 mismatch: " + path.name)
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
            if (identity[0] is None or identity[0].lower() != row.subject_id
                    or identity[1] != row.task
                    or identity[2] != row.updated_trial_id):
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
                if (m["mapping_status"] != "NO_SOURCE_FE_TARGET"
                        or empty["empty_representation_status"]
                        != "READER_EMPTY_REPRESENTATION"):
                    raise ValueError("Source-empty FE representation unresolved.")
                excluded.append({
                    "local_record_key": key, "subject_id": row.subject_id,
                    "task": row.task, "source_file": filename,
                    "reason": "SOURCE_FE_TARGET_EMPTY",
                    "reader_class": empty["python_class"],
                })
                continue
            fe_status, fe_time = validate_fe(fe, False)
            if fe_status["fe_status"] != "VALID_STRAIN_COLUMNS" or fe_time is None:
                raise ValueError("Original FE numeric audit failed: " + key)
            observed, values = match_original_coordinates(imu_time, fe_time)
            if (observed["mapping_status"] != "EXACT_TIME_COORDINATE_MAPPING"
                    or m["mapping_status"] != observed["mapping_status"]
                    or not bool(m["source_identity_matches"])):
                raise ValueError("Original FE–IMU coordinate mapping mismatch.")
            positions = np.asarray(values["imu_indices"], dtype=np.int64)
            if positions.tolist() != json.loads(m["imu_position_indices"]):
                raise ValueError("Original mapped time indices were modified.")
            feature = imu.loc[:, list(EXPECTED_CHANNELS)].to_numpy(
                dtype=np.float64, copy=True
            )
            target = fe.loc[:, ["e1", "e3"]].to_numpy(
                dtype=np.float64, copy=True
            )
            if (feature.shape != (101, 42) or target.shape != (11, 2)
                    or positions.shape != (11,) or not np.isfinite(feature).all()
                    or not np.isfinite(target).all()):
                raise ValueError("Nonfinite or malformed original numeric pair.")
            x_full.append(feature)
            x_at_fe.append(feature[positions, :])
            y_fe.append(target)
            indexes.append(positions)
            eligible.append({
                "local_record_key": key, "subject_id": row.subject_id,
                "source_file": filename, "local_record_index": int(row.local_record_index),
                "task": row.task, "updated_trial_id": row.updated_trial_id,
                "time_mapping": "EXACT_TIME_COORDINATE_MAPPING",
            })
    if len(eligible) != 1729 or len(excluded) != 168:
        raise ValueError("Unexpected original numerical eligibility counts.")
    arrays = {
        "X_full": np.stack(x_full),
        "X_at_fe": np.stack(x_at_fe),
        "y_fe": np.stack(y_fe),
        "imu_positions": np.stack(indexes),
    }
    assert arrays["X_full"].shape == (1729, 101, 42)
    assert arrays["X_at_fe"].shape == (1729, 11, 42)
    assert arrays["y_fe"].shape == (1729, 11, 2)
    assert arrays["imu_positions"].shape == (1729, 11)
    check = np.take_along_axis(
        arrays["X_full"], arrays["imu_positions"][:, :, None], axis=1
    )
    if not np.array_equal(check, arrays["X_at_fe"]):
        raise ValueError("Paired IMU indices do not match original signal.")
    return arrays, pd.DataFrame(eligible), pd.DataFrame(excluded)


# ================= 02. Participant-Only Deterministic Splits =================
def fixed_subject_splits(eligible, registry, seed=2026):
    """Partition eligible subjects by stable SHA256 ID order, never FE values."""
    all_subjects = sorted(registry["subject_id"].unique().tolist())
    good_subjects = sorted(eligible["subject_id"].unique().tolist())
    if len(all_subjects) != 40 or len(good_subjects) != 37:
        raise ValueError("Incorrect original or eligible subject counts.")
    order = sorted(good_subjects, key=lambda name: hashlib.sha256(
        f"essex-v13:{seed}:{name}".encode("utf-8")
    ).hexdigest())
    test, val, train = set(order[:4]), set(order[4:8]), set(order[8:])
    if not (len(test) == len(val) == 4 and len(train) == 29):
        raise ValueError("Unexpected fixed split sizes.")
    def label(name):
        return ("TRAIN" if name in train else
                "VALIDATION" if name in val else
                "TEST" if name in test else "EXCLUDED_NO_FE_TARGET")
    subjects = pd.DataFrame({
        "subject_id": all_subjects,
        "split": [label(s) for s in all_subjects],
    })
    assigned = eligible.copy()
    assigned["split"] = assigned["subject_id"].map(
        dict(zip(subjects["subject_id"], subjects["split"]))
    )
    if (assigned["split"].isna().any()
            or (assigned.groupby("subject_id")["split"].nunique() != 1).any()):
        raise ValueError("Participant-level partition leakage.")
    summary = assigned.groupby("split").agg(
        participants=("subject_id", "nunique"),
        trials=("local_record_key", "size"),
    ).reset_index()
    activities = assigned.groupby(["split", "task"]).size().rename(
        "trials"
    ).reset_index()
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
    return pd.DataFrame({
        "channel": EXPECTED_CHANNELS, "train_mean": mean,
        "train_std": std, "zero_std": std == 0,
        "safe_denominator": np.where(std == 0, 1.0, std),
        "train_sample_rows": len(values),
    })


def paired_source_digest(arrays, assigned):
    digest = hashlib.sha256()
    for key in ("X_full", "X_at_fe", "y_fe", "imu_positions"):
        a = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}:{a.shape}:{a.dtype}".encode("ascii"))
        digest.update(a.tobytes())
    digest.update(json.dumps(
        assigned["local_record_key"].tolist(), separators=(",", ":")
    ).encode("utf-8"))
    return digest.hexdigest()


def save_ignored_cache(path, arrays, assigned):
    """Write local data/processed NPZ; refuse to overwrite conflicting cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["record_keys"] = assigned["local_record_key"].to_numpy(dtype=str)
    if path.is_file():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(payload) or not all(
                np.array_equal(previous[k], a) for k, a in payload.items()
            ):
                raise ValueError("Existing local cache differs from original data.")
        return "VERIFIED_EXISTING"
    tmp = path.with_name(path.name + ".part")
    if tmp.exists():
        raise FileExistsError("Interrupted local array cache already exists.")
    try:
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return "CREATED_LOCAL_ONLY"
