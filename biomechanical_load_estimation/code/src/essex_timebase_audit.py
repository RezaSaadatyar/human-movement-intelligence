# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Source-linked IMU and FE timebase audit for original Essex MAT records.

The source file remains original. IMU time and FE time are checked independently.
Numeric sensor and FE channels are not interpreted as clinically validated
measurements or a machine-learning training dataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from essex_python_first import (
    _original_text, original_records, source_field,
)

SENSORS = ("PEL", "RTH", "LTH", "RSK", "LSK", "RFT", "LFT")
COMPONENTS = ("accel_x", "accel_y", "accel_z",
              "gyro_x", "gyro_y", "gyro_z")
EXPECTED_CHANNELS = tuple(
    f"{component}_{sensor}" for component in COMPONENTS for sensor in SENSORS
)
EXPECTED_FE_COLUMNS = ("timestep", "e1", "e3")



# ============ 01. Original Labels and Numerical Channel Contracts ============
def validate_imu(table):
    """Validate original labels and finite source data; separate the time column."""
    result = {
        "imu_status": "INVALID", "imu_numeric_channels": 0,
        "imu_timestep_status": "UNAVAILABLE", "imu_t0": None,
        "imu_tlast": None, "imu_tstep": None,
    }
    if not isinstance(table, pd.DataFrame):
        result["imu_status"] = "NOT_A_TABLE"
        return result, None
    if table.shape != (101, 43) or not table.columns.is_unique:
        result["imu_status"] = "UNEXPECTED_SHAPE_OR_LABELS"
        return result, None
    if set(table.columns) != set(EXPECTED_CHANNELS) | {"timestep"}:
        result["imu_status"] = "UNEXPECTED_SOURCE_COLUMNS"
        return result, None
    signal = table.loc[:, list(EXPECTED_CHANNELS)]
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in signal.dtypes):
        result["imu_status"] = "NONNUMERIC_CHANNEL"
        return result, None
    values = signal.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        result["imu_status"] = "NONFINITE_CHANNEL"
        return result, None
    result["imu_numeric_channels"] = 42
    times, time_status = read_time_column(table["timestep"])
    result["imu_timestep_status"] = time_status
    result["imu_status"] = "VALID_CHANNELS"
    if times is not None:
        result["imu_t0"] = float(times[0])
        result["imu_tlast"] = float(times[-1])
        result["imu_tstep"] = float(np.median(np.diff(times)))
    return result, times


def validate_fe(table, original_fe_empty):
    """Check the FE e1/e3 columns and preserve original empty representation."""
    result = {
        "fe_status": "INVALID", "fe_numeric_channels": 0,
        "fe_timestep_status": "UNAVAILABLE",
        "fe_t0": None, "fe_tlast": None, "fe_tstep": None,
        "fe_time_entry_class": "",
    }
    if type(table).__name__ == "MatlabCanonicalEmpty" or (
        isinstance(table, pd.DataFrame) and table.empty
    ):
        result["fe_status"] = (
            "EXPECTED_EMPTY" if original_fe_empty else "UNEXPECTED_EMPTY"
        )
        return result, None
    if original_fe_empty:
        result["fe_status"] = "UNEXPECTED_NONEMPTY"
        return result, None
    if not isinstance(table, pd.DataFrame):
        result["fe_status"] = "NOT_A_TABLE"
        return result, None
    if (table.shape != (11, 3) or not table.columns.is_unique
            or set(table.columns) != set(EXPECTED_FE_COLUMNS)):
        result["fe_status"] = "UNEXPECTED_SHAPE_OR_LABELS"
        return result, None
    if not all(
        pd.api.types.is_numeric_dtype(table[col].dtype)
        for col in ("e1", "e3")
    ):
        result["fe_status"] = "NONNUMERIC_STRAIN_COLUMNS"
        return result, None
    values = table.loc[:, ["e1", "e3"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        result["fe_status"] = "NONFINITE_STRAIN_COLUMNS"
        return result, None
    result["fe_numeric_channels"] = 2
    raw_entry = table["timestep"].iloc[0]
    result["fe_time_entry_class"] = type(raw_entry).__name__
    times, time_status = read_time_column(table["timestep"])
    result["fe_timestep_status"] = time_status
    result["fe_status"] = "VALID_STRAIN_COLUMNS"
    if times is not None:
        result["fe_t0"] = float(times[0])
        result["fe_tlast"] = float(times[-1])
        result["fe_tstep"] = float(np.median(np.diff(times)))
    return result, times



# ============== 02. Conservative Source Time Vector Extraction ===============
def _scalar_time(value):
    """Unwrap scalar MATLAB object cells without interpreting arrays as times."""
    for _ in range(5):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None, "NONSCALAR_OBJECT"
            value = value.reshape(-1)[0]
            continue
        break
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return float(pd.Timedelta(value).total_seconds()), "DURATION_SECONDS"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return float(pd.Timestamp(value).timestamp()), "DATETIME_EPOCH_SECONDS"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, (bool, np.bool_)):
            return None, "BOOLEAN_NOT_TIME"
        number = float(value)
        if np.isfinite(number):
            return number, "ORIGINAL_NUMERIC"
        return None, "NONFINITE_TIME"
    return None, "UNSUPPORTED_TIME_OBJECT"


def read_time_column(series):
    """Only accept scalar, finite, strictly increasing original time values."""
    parsed = [_scalar_time(item) for item in series.tolist()]
    codes = {status for _, status in parsed}
    if len(codes) != 1:
        return None, "MIXED_TIME_CLASSES"
    status = next(iter(codes))
    if any(value is None for value, _ in parsed):
        return None, status
    values = np.asarray([v for v, _ in parsed], dtype="float64")
    if (not np.isfinite(values).all() or len(values) < 2
            or not np.all(np.diff(values) > 0)):
        return None, "NONMONOTONIC_TIME"
    return values, "VALID_" + status


def compare_time_grids(imu, fe):
    """Compare 11 FE positions against every tenth original IMU time point.

    An equal numerical grid is evidence of temporal consistency, not of the
    physical meaning of the FE outputs or full clinical model readiness.
    """
    if imu is None or fe is None:
        return "TIME_NOT_RECOVERED", None
    if len(imu) != 101 or len(fe) != 11:
        return "UNEXPECTED_TIME_LENGTH", None
    expected = imu[::10]
    error = float(np.max(np.abs(expected - fe)))
    span = float(np.max(imu) - np.min(imu))
    threshold = max(1e-7, span * 1e-6)
    if np.allclose(expected, fe, rtol=0.0, atol=threshold):
        return "NUMERIC_TIME_GRID_MATCH", error
    return "NUMERIC_TIME_GRID_DIFFERS", error



# ========== 03. Complete Original Record Audit without Raw Exports ===========
def audit_original_source(reader, mat_dir, registry):
    """Inspect all source records with original IDs and immutable FE accounting."""
    required = {
        "source_file", "subject_id", "local_record_index",
        "local_record_key", "task", "updated_trial_id", "fe_representation",
    }
    if not required.issubset(registry.columns):
        raise ValueError("Original registry missing required fields.")
    if len(registry) != 1897 or registry["subject_id"].nunique() != 40:
        raise ValueError("Original source registry count mismatch.")
    rows, errors = [], []
    first_example = None
    for filename, group in registry.groupby("source_file", sort=True):
        try:
            decoded = reader(
                str(Path(mat_dir) / filename),
                variable_names=["Data"], raw_data=False,
            )
            if not isinstance(decoded, dict) or "Data" not in decoded:
                raise ValueError("MAT reader returned no original Data.")
            records = original_records(decoded["Data"])
            if len(records) != len(group):
                raise ValueError("Original source record count changed.")
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:250]}"
            errors.append({
                "source_file": filename, "error_kind": type(exc).__name__,
                "error_message": str(exc)[:250],
            })
            for row in group.itertuples(index=False):
                rows.append({
                    "local_record_key": row.local_record_key,
                    "subject_id": row.subject_id, "task": row.task,
                    "source_file": filename,
                    "updated_trial_id": row.updated_trial_id,
                    "identity_valid": False,
                    "fe_originally_empty": row.fe_representation == "empty",
                    "imu_status": "READER_ERROR",
                    "imu_numeric_channels": 0,
                    "fe_status": "READER_ERROR",
                    "fe_numeric_channels": 0,
                    "time_alignment": "READER_ERROR",
                    "time_error": None,
                })
            continue
        for row in group.itertuples(index=False):
            idx = int(row.local_record_index)
            if idx < 0 or idx >= len(records):
                raise ValueError("Original record index outside source array.")
            original = records[idx]
            subject = _original_text(source_field(original, "SubjID"))
            task = _original_text(source_field(original, "Task"))
            trial = _original_text(source_field(original, "UpdatedTrialID"))
            match = (
                subject is not None and subject.lower() == row.subject_id
                and task == row.task and trial == row.updated_trial_id
            )
            imu = source_field(original, "IMU")
            fe = source_field(original, "e1e3")
            if match:
                imu_result, imu_time = validate_imu(imu)
                fe_result, fe_time = validate_fe(
                    fe, row.fe_representation == "empty"
                )
                if fe_result["fe_status"] == "EXPECTED_EMPTY":
                    alignment, max_error = "NO_FE_TARGET", None
                elif imu_result["imu_status"] == "VALID_CHANNELS" and (
                    fe_result["fe_status"] == "VALID_STRAIN_COLUMNS"
                ):
                    alignment, max_error = compare_time_grids(imu_time, fe_time)
                else:
                    alignment, max_error = "INVALID_CHANNEL_OR_TARGET", None
            else:
                imu_result = {"imu_status": "SOURCE_ID_MISMATCH",
                              "imu_numeric_channels": 0}
                fe_result = {"fe_status": "SOURCE_ID_MISMATCH",
                             "fe_numeric_channels": 0}
                alignment, max_error = "SOURCE_ID_MISMATCH", None
            record_row = {
                "local_record_key": row.local_record_key,
                "subject_id": row.subject_id,
                "source_file": filename,
                "local_record_index": idx,
                "task": row.task, "updated_trial_id": row.updated_trial_id,
                "identity_valid": bool(match),
                "fe_originally_empty": row.fe_representation == "empty",
                **imu_result, **fe_result,
                "time_alignment": alignment,
                "time_error": max_error,
            }
            rows.append(record_row)
            if (first_example is None and match
                    and imu_result["imu_status"] == "VALID_CHANNELS"
                    and fe_result["fe_status"] == "VALID_STRAIN_COLUMNS"
                    and isinstance(imu, pd.DataFrame)
                    and isinstance(fe, pd.DataFrame)):
                first_example = {
                    "key": row.local_record_key,
                    "imu_time": imu_time,
                    "fe_time": fe_time,
                    "imu_signal": imu[EXPECTED_CHANNELS[0]].to_numpy(
                        dtype="float64"
                    ),
                    "e1": fe["e1"].to_numpy(dtype="float64"),
                    "e3": fe["e3"].to_numpy(dtype="float64"),
                    "imu_name": EXPECTED_CHANNELS[0],
                }
    out = pd.DataFrame(rows)
    return (
        out,
        pd.DataFrame(errors, columns=[
            "source_file", "error_kind", "error_message"
        ]),
        first_example,
    )



# ===================== 04. Scientific Quality Summaries ======================
def summarise_alignment(audit, errors):
    """Keep source, numeric, and temporal validation as separate decisions."""
    if len(audit) != 1897:
        raise ValueError("Incomplete original source census.")
    good_imu = audit["imu_status"].eq("VALID_CHANNELS")
    good_fe = audit["fe_status"].eq("VALID_STRAIN_COLUMNS")
    empty = audit["fe_status"].eq("EXPECTED_EMPTY")
    matched = audit["time_alignment"].eq("NUMERIC_TIME_GRID_MATCH")
    return {
        "source_records": int(len(audit)),
        "participants": int(audit["subject_id"].nunique()),
        "source_identical": bool(audit["identity_valid"].all()),
        "reader_error_files": int(len(errors)),
        "imu_42_channel_records": int(good_imu.sum()),
        "fe_e1_e3_records": int(good_fe.sum()),
        "source_empty_fe_records": int(empty.sum()),
        "paired_numeric_records": int((good_imu & good_fe).sum()),
        "numerically_matching_time_grids": int(matched.sum()),
        "time_grid_unresolved_records": int(
            audit["time_alignment"].eq("TIME_NOT_RECOVERED").sum()
        ),
        "time_grid_mismatched_records": int(
            audit["time_alignment"].eq("NUMERIC_TIME_GRID_DIFFERS").sum()
        ),
        "numeric_full_export_ready": bool(
            audit["identity_valid"].all()
            and len(errors) == 0
            and good_imu.all()
            and (good_fe | empty).all()
            and int(empty.sum()) == 168
            and (audit.loc[good_fe, "time_alignment"]
                 == "NUMERIC_TIME_GRID_MATCH").all()
        ),
        "physical_units_independently_verified": False,
        "model_ready": False,
    }

