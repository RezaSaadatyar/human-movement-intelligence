# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Conservative Python-first pilot for original Essex MATLAB object tables.

Pure Python utility: no automatic reading of local GitHub state, no model
training, no physical-unit inference and no opaque uint32-as-signal claims.
Only original, finite, source-labelled pandas DataFrames may be exported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PILOT_SUBJECTS = ("subj04", "subj06", "subj09", "subj44")
REQUIRED_FIELDS = ("SubjID", "Task", "UpdatedTrialID", "IMU", "e1e3")


# ============== 01. Source Identity and MATLAB Data Navigation ===============
def original_records(data):
    """Flatten the original one-dimensional MATLAB struct column in F order."""
    if isinstance(data, np.ndarray):
        if data.ndim > 2 or (data.ndim == 2 and min(data.shape) != 1):
            raise ValueError("Unexpected multidimensional original Data ordering.")
        return list(np.ravel(data, order="F"))
    if isinstance(data, (list, tuple)):
        return list(data)
    raise ValueError(f"Unsupported Data container: {type(data).__name__}")


def source_field(record, name):
    """Read an original exact field name, never a position-based surrogate."""
    if isinstance(record, dict):
        return record.get(name)
    if hasattr(record, "_fieldnames") and name in record._fieldnames:
        return getattr(record, name)
    if isinstance(record, np.void) and record.dtype.names:
        return record[name] if name in record.dtype.names else None
    if hasattr(record, name) and not name.startswith("_"):
        return getattr(record, name)
    return None


def _original_text(value):
    """Convert a genuine scalar MATLAB string, not arbitrary array metadata."""
    if isinstance(value, (str, np.str_)):
        return str(value).strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _original_text(value.item())
        if (
            value.dtype.kind in "US"
            and 0 < value.size <= 128
            and value.ndim <= 2
            and 1 in value.shape
        ):
            chars = [str(char) for char in value.ravel(order="F")]
            if all(len(char) == 1 for char in chars):
                return "".join(chars).strip()
    return None


def describe_original(value):
    """Return bounded metadata only; never serialise raw sensor values."""
    return {
        "python_class": type(value).__name__,
        "shape": str(tuple(value.shape)) if hasattr(value, "shape") else "",
        "dtype": str(value.dtype) if isinstance(value, np.ndarray) else "",
        "is_dataframe": isinstance(value, pd.DataFrame),
    }


def classify_numeric(value):
    """Accept only original-label, finite and real DataFrame numeric columns.

    An unlabelled ndarray may be a valid observation but cannot be used as a
    source-labelled sensor or FE target without independent evidence.
    """
    if value is None:
        return ("missing", None, None)
    if type(value).__name__ == "MatlabCanonicalEmpty":
        return ("empty", None, None)
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return ("empty", None, None)
        if (
            not value.columns.is_unique
            or isinstance(value.columns, pd.RangeIndex)
            or not all(isinstance(x, str) and x.strip() for x in value.columns)
        ):
            return ("unverified_column_names", None, None)
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in value.dtypes):
            return ("non_numeric_table", None, None)
        values = value.to_numpy(copy=True)
        if not np.issubdtype(values.dtype, np.number) or not np.isrealobj(values):
            return ("unsupported_numeric_type", None, None)
        if not np.isfinite(values).all():
            return ("nonfinite_table", None, None)
        return (
            "labelled_numeric_table",
            values.astype("float64"),
            [str(x) for x in value.columns],
        )
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ("empty", None, None)
        if value.dtype.kind in "iuf" and value.ndim == 2:
            return ("unlabelled_numeric_array", None, None)
        return ("unresolved_matlab_object", None, None)
    return ("unsupported_class", None, None)


def select_representative_rows(registry, subjects=PILOT_SUBJECTS, max_each=8):
    """Preserve activities and FE-availability states without random sampling."""
    if max_each < 1 or max_each > 100:
        raise ValueError("Bounded pilot sample must contain 1 to 100 rows.")
    selected = []
    for subject in subjects:
        rows = registry.loc[registry["subject_id"] == subject].sort_values(
            "local_record_index", kind="stable"
        )
        if rows.empty:
            raise ValueError(f"Original source participant missing: {subject}")
        selection = (
            pd.concat(
                [
                    rows.drop_duplicates(["task", "fe_representation"]),
                    rows.head(1),
                    rows.tail(1),
                ]
            )
            .drop_duplicates("local_record_key")
            .head(max_each)
        )
        selected.append(selection)
    result = pd.concat(selected, ignore_index=True)
    if not result["local_record_key"].is_unique:
        raise ValueError("Pilot contains duplicate original record keys.")
    return result


# ========== 02. Bounded Reader Pilot and Source Registry Agreement ===========
def probe_pilot(reader, mat_dir, sampled):
    """Run matio's real decoded mode on representative original MAT files.

    Reader failures are findings, not fabricated numerical predictions. One
    file-level exception must not conceal successful decoding in other files.
    """
    mat_dir = Path(mat_dir)
    reports = []
    file_errors = []
    decoded_arrays = {}
    for filename, group in sampled.groupby("source_file", sort=True):
        try:
            payload = reader(
                str(mat_dir / filename),
                variable_names=["Data"],
                raw_data=False,
            )
            if not isinstance(payload, dict) or "Data" not in payload:
                raise ValueError("matio returned no original Data variable.")
            records = original_records(payload["Data"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:350]}"
            file_errors.append(
                {
                    "source_file": filename,
                    "error_kind": type(exc).__name__,
                    "error_message": str(exc)[:350],
                }
            )
            for row in group.itertuples(index=False):
                for modality in ("IMU", "FE"):
                    reports.append(
                        _row(row, modality, "reader_error", False, None, error)
                    )
            continue
        for row in group.itertuples(index=False):
            idx = int(row.local_record_index)
            if idx >= len(records) or idx < 0:
                raise ValueError("Source Data record index outside decoded array.")
            record = records[idx]
            actual_id = _original_text(source_field(record, "SubjID"))
            actual_task = _original_text(source_field(record, "Task"))
            actual_trial = _original_text(source_field(record, "UpdatedTrialID"))
            matched = (
                actual_id is not None
                and actual_id.lower() == row.subject_id
                and actual_task == row.task
                and actual_trial == row.updated_trial_id
            )
            for modality, field in (("IMU", "IMU"), ("FE", "e1e3")):
                value = source_field(record, field)
                status, matrix, column_names = classify_numeric(value)
                if not matched:
                    status = "identity_mismatch"
                    matrix, column_names = None, None
                evidence = _row(row, modality, status, matched, value, "")
                if matrix is not None:
                    evidence["n_time"] = int(matrix.shape[0])
                    evidence["n_channels"] = int(matrix.shape[1])
                    evidence["source_columns"] = json.dumps(column_names)
                    evidence["time_101_candidate"] = bool(
                        modality == "IMU" and matrix.shape[0] == 101
                    )
                    evidence["imu_42_candidate"] = bool(
                        modality == "IMU" and matrix.shape[1] == 42
                    )
                    evidence["fe_11_candidate"] = bool(
                        modality == "FE" and matrix.shape[0] == 11
                    )
                    evidence["named_fe_pair_candidate"] = bool(
                        modality == "FE"
                        and matrix.shape[1] == 2
                        and {s.casefold() for s in column_names} == {"e1", "e3"}
                    )
                    decoded_arrays[(row.local_record_key, modality)] = (
                        matrix,
                        column_names,
                    )
                evidence["source_fe_empty"] = bool(
                    modality == "FE" and row.fe_representation == "empty"
                )
                reports.append(evidence)
    return (
        pd.DataFrame(reports),
        pd.DataFrame(
            file_errors, columns=["source_file", "error_kind", "error_message"]
        ),
        decoded_arrays,
    )


def _row(reg, modality, status, matched, value, error):
    desc = describe_original(value)
    return {
        "local_record_key": reg.local_record_key,
        "subject_id": reg.subject_id,
        "source_file": reg.source_file,
        "local_record_index": int(reg.local_record_index),
        "task": reg.task,
        "updated_trial_id": reg.updated_trial_id,
        "modality": modality,
        "decode_status": status,
        "source_identity_match": bool(matched),
        "source_fe_empty": False,
        "n_time": 0,
        "n_channels": 0,
        "source_columns": "[]",
        "time_101_candidate": False,
        "imu_42_candidate": False,
        "fe_11_candidate": False,
        "named_fe_pair_candidate": False,
        "error_message": error,
        **desc,
    }


def evaluate_pilot(rows, errors):
    """Summarise decoded source matrices and observed reader outcomes."""
    states = rows.groupby(["modality", "decode_status"]).size()
    return {
        "sampled_original_records": int(rows["local_record_key"].nunique()),
        "modality_probes": int(len(rows)),
        "reader_error_files": int(len(errors)),
        "identity_matches": int(
            rows.groupby("local_record_key")["source_identity_match"].all().sum()
        ),
        "labelled_numeric_imu": int(
            (
                (rows["modality"] == "IMU")
                & (rows["decode_status"] == "labelled_numeric_table")
            ).sum()
        ),
        "labelled_numeric_fe": int(
            (
                (rows["modality"] == "FE")
                & (rows["decode_status"] == "labelled_numeric_table")
            ).sum()
        ),
        "imu_101_42_pairs": int(
            (rows["time_101_candidate"] & rows["imu_42_candidate"]).sum()
        ),
        "fe_11_e1e3_pairs": int(
            (rows["fe_11_candidate"] & rows["named_fe_pair_candidate"]).sum()
        ),
        "states": {f"{m}/{s}": int(n) for (m, s), n in states.items()},
        "physical_units_verified": False,
        "fe_temporal_alignment_verified": False,
        "model_ready": False,
    }


# ================== 04. Source-Preserved MATLAB Table Schema ==================
def inspect_native_table_schema(reader, mat_dir, selected):
    """Audit original DataFrame column types and indexes without raw values.

    A 101-by-43 or 11-by-3 table is NOT reduced automatically. Any time or
    index column must be identified from original names and source semantics.
    One schema record per original source column; no measurements are exported.
    """
    columns = []
    index_rows = []
    for filename, group in selected.groupby("source_file", sort=True):
        payload = reader(
            str(Path(mat_dir) / filename),
            variable_names=["Data"],
            raw_data=False,
        )
        if not isinstance(payload, dict) or "Data" not in payload:
            raise ValueError("Original Data missing in Python table schema audit.")
        records = original_records(payload["Data"])
        for entry in group.itertuples(index=False):
            record = records[int(entry.local_record_index)]
            subject = _original_text(source_field(record, "SubjID"))
            task = _original_text(source_field(record, "Task"))
            trial = _original_text(source_field(record, "UpdatedTrialID"))
            if subject is None or (
                subject.lower() != entry.subject_id
                or task != entry.task
                or trial != entry.updated_trial_id
            ):
                raise ValueError("Original record identity changed in schema audit.")
            for modality, field in (("IMU", "IMU"), ("FE", "e1e3")):
                table = source_field(record, field)
                if not isinstance(table, pd.DataFrame):
                    index_rows.append(
                        {
                            "local_record_key": entry.local_record_key,
                            "modality": modality,
                            "table_class": type(table).__name__,
                            "n_rows": 0,
                            "n_columns": 0,
                            "index_class": "",
                            "index_dtype": "",
                            "index_name": "",
                            "index_unique": False,
                        }
                    )
                    continue
                index_rows.append(
                    {
                        "local_record_key": entry.local_record_key,
                        "modality": modality,
                        "table_class": "DataFrame",
                        "n_rows": int(len(table)),
                        "n_columns": int(len(table.columns)),
                        "index_class": type(table.index).__name__,
                        "index_dtype": str(table.index.dtype),
                        "index_name": str(table.index.name or ""),
                        "index_unique": bool(table.index.is_unique),
                    }
                )
                for i in range(table.shape[1]):
                    series = table.iloc[:, i]
                    name = str(table.columns[i])
                    valid = series.dropna()
                    type_name = type(valid.iloc[0]).__name__ if not valid.empty else ""
                    numeric = bool(pd.api.types.is_numeric_dtype(series.dtype))
                    real_finite = False
                    if numeric:
                        arr = series.to_numpy()
                        real_finite = bool(np.isrealobj(arr) and np.isfinite(arr).all())
                    columns.append(
                        {
                            "local_record_key": entry.local_record_key,
                            "subject_id": entry.subject_id,
                            "task": entry.task,
                            "modality": modality,
                            "n_rows": int(len(table)),
                            "column_position": int(i),
                            "original_column_name": name,
                            "pandas_dtype": str(series.dtype),
                            "first_nonnull_python_type": type_name,
                            "null_count": int(series.isna().sum()),
                            "is_numeric_dtype": numeric,
                            "is_real_finite_numeric": real_finite,
                            "is_time_label_candidate": name.casefold().strip()
                            in {
                                "time",
                                "t",
                                "timestamp",
                                "rowtimes",
                                "sample",
                                "sample_index",
                                "index",
                            },
                            "scientifically_verified_time_axis": False,
                        }
                    )
    return pd.DataFrame(columns), pd.DataFrame(index_rows)
