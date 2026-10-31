# ======= Original Essex Temporal Mapping and FE Source Reconciliation ========
"""Trace original FE time coordinates to observed IMU sample indices.

No interpolation, source-value rewriting, automatic strain-unit inference,
or model training is performed. Missingness from the original registry is
audited separately from the Python reader's decoded object representation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from essex_python_first import _original_text, original_records, source_field
from essex_timebase_audit import validate_imu, validate_fe


# ================ 01. Original FE Empty Field Representation =================
def describe_source_empty(value):
    """Describe source-declared FE missingness without guessing its payload."""
    cls = type(value).__name__
    shape = (
        str(tuple(value.shape))
        if isinstance(value, (np.ndarray, pd.DataFrame)) else ""
    )
    if value is None or cls == "MatlabCanonicalEmpty":
        state = "READER_EMPTY_REPRESENTATION"
    elif isinstance(value, np.ndarray) and value.size == 0:
        state = "READER_EMPTY_REPRESENTATION"
    elif isinstance(value, pd.DataFrame) and value.empty:
        state = "READER_EMPTY_REPRESENTATION"
    elif isinstance(value, (list, tuple, dict)) and len(value) == 0:
        state = "READER_EMPTY_REPRESENTATION"
    else:
        state = "SOURCE_EMPTY_REPRESENTATION_UNRESOLVED"
    return {
        "python_class": cls, "python_shape": shape,
        "size_if_array": (
            int(value.size) if isinstance(value, np.ndarray) else None
        ),
        "empty_representation_status": state,
        "original_registry_fe_empty": True,
    }



# ============== 02. Exact Matching of Original Time Coordinates ==============
def match_original_coordinates(imu_time, fe_time):
    """Map each FE coordinate to one unique original IMU position.

    Equality is tested in the native numeric coordinate system. This is
    not an assertion that FE observations are physically validated targets.
    """
    if imu_time is None or fe_time is None:
        return {"mapping_status": "TIME_VALUES_NOT_AVAILABLE",
                "imu_position_indices": "", "maximum_difference": None,
                "consecutive_index_steps": "", "exact_matches": 0}, None
    if len(imu_time) != 101 or len(fe_time) != 11:
        return {"mapping_status": "UNEXPECTED_TIME_AXIS_LENGTH",
                "imu_position_indices": "", "maximum_difference": None,
                "consecutive_index_steps": "", "exact_matches": 0}, None
    imu_time = np.asarray(imu_time, dtype=float)
    fe_time = np.asarray(fe_time, dtype=float)
    if (not np.isfinite(imu_time).all() or not np.isfinite(fe_time).all()
            or not np.all(np.diff(imu_time) > 0)
            or not np.all(np.diff(fe_time) > 0)):
        return {"mapping_status": "INVALID_TIME_SEQUENCE",
                "imu_position_indices": "", "maximum_difference": None,
                "consecutive_index_steps": "", "exact_matches": 0}, None

    # 10^-8 of one original IMU interval is far smaller than a sample.
    tolerance = max(1e-10, float(np.min(np.diff(imu_time))) * 1e-8)
    found = []
    differences = []
    exact = 0
    for target in fe_time:
        residuals = np.abs(imu_time - target)
        hits = np.flatnonzero(residuals <= tolerance)
        if len(hits) != 1:
            status = ("AMBIGUOUS_TIME_MATCH" if len(hits) > 1
                      else "FE_COORDINATE_NOT_IN_IMU")
            return {"mapping_status": status, "imu_position_indices": "",
                    "maximum_difference": float(np.min(residuals)),
                    "consecutive_index_steps": "",
                    "exact_matches": exact}, None
        index = int(hits[0])
        found.append(index)
        error = float(residuals[index])
        differences.append(error)
        exact += bool(error == 0.0)
    if len(set(found)) != len(found) or not np.all(np.diff(found) > 0):
        return {"mapping_status": "NONUNIQUE_INDEX_MAPPING",
                "imu_position_indices": "", "maximum_difference": None,
                "consecutive_index_steps": "", "exact_matches": exact}, None
    status = ("EXACT_TIME_COORDINATE_MAPPING" if exact == len(found)
              else "TOLERANT_TIME_COORDINATE_MAPPING")
    return {
        "mapping_status": status,
        "imu_position_indices": json.dumps(found),
        "maximum_difference": float(max(differences)),
        "consecutive_index_steps": json.dumps(np.diff(found).tolist()),
        "exact_matches": int(exact),
    }, {"imu_time": imu_time, "fe_time": fe_time, "imu_indices": found}



# ============ 03. Complete Source Trial and Time Coordinate Audit ============
def audit_temporal_mapping(reader, mat_dir, registry):
    """Audit original source record IDs and FE missingness for all 40 files."""
    if len(registry) != 1897 or registry["subject_id"].nunique() != 40:
        raise ValueError("Source record registry is incomplete.")
    rows, empty_rows, file_errors = [], [], []
    first_mapping = None
    for filename, group in registry.groupby("source_file", sort=True):
        try:
            decoded = reader(
                str(Path(mat_dir) / filename),
                variable_names=["Data"], raw_data=False,
            )
            if not isinstance(decoded, dict) or "Data" not in decoded:
                raise ValueError("Original MAT Data not returned.")
            original = original_records(decoded["Data"])
            if len(original) != len(group):
                raise ValueError("Source record count differs.")
        except Exception as exc:
            file_errors.append({
                "source_file": filename,
                "error_class": type(exc).__name__,
                "error": str(exc)[:240],
            })
            for record in group.itertuples(index=False):
                rows.append({
                    "local_record_key": record.local_record_key,
                    "subject_id": record.subject_id,
                    "source_file": filename,
                    "task": record.task,
                    "originally_empty_fe": (
                        record.fe_representation == "empty"
                    ),
                    "source_identity_matches": False,
                    "imu_channel_status": "READER_ERROR",
                    "fe_channel_status": "READER_ERROR",
                    "mapping_status": "READER_ERROR",
                    "imu_position_indices": "",
                    "maximum_difference": None,
                    "consecutive_index_steps": "",
                    "exact_matches": 0,
                    "imu_t0": None, "imu_tlast": None,
                    "fe_t0": None, "fe_tlast": None,
                })
            continue
        for row in group.itertuples(index=False):
            idx = int(row.local_record_index)
            if idx < 0 or idx >= len(original):
                raise ValueError("Original source index outside Data array.")
            source = original[idx]
            actual_id = _original_text(source_field(source, "SubjID"))
            actual_task = _original_text(source_field(source, "Task"))
            actual_trial = _original_text(source_field(source, "UpdatedTrialID"))
            identity_ok = (
                actual_id is not None
                and actual_id.lower() == row.subject_id
                and actual_task == row.task
                and actual_trial == row.updated_trial_id
            )
            if not identity_ok:
                raise ValueError("Original trial identity mismatch: "
                                 + row.local_record_key)
            imu_data = source_field(source, "IMU")
            fe_data = source_field(source, "e1e3")
            imu_meta, imu_time = validate_imu(imu_data)
            empty = row.fe_representation == "empty"
            if empty:
                fe_desc = describe_source_empty(fe_data)
                empty_rows.append({
                    "local_record_key": row.local_record_key,
                    "subject_id": row.subject_id,
                    "task": row.task, **fe_desc,
                })
                fe_status = "SOURCE_DECLARED_EMPTY"
                fe_time = None
                match = {
                    "mapping_status": "NO_SOURCE_FE_TARGET",
                    "imu_position_indices": "",
                    "maximum_difference": None,
                    "consecutive_index_steps": "",
                    "exact_matches": 0,
                }
            else:
                fe_meta, fe_time = validate_fe(fe_data, False)
                fe_status = fe_meta["fe_status"]
                if (imu_meta["imu_status"] == "VALID_CHANNELS"
                        and fe_status == "VALID_STRAIN_COLUMNS"):
                    match, example = match_original_coordinates(
                        imu_time, fe_time
                    )
                    if (first_mapping is None and example is not None):
                        first_mapping = {
                            **example, "local_record_key": row.local_record_key
                        }
                else:
                    match = {
                        "mapping_status": "INVALID_SOURCE_CHANNELS",
                        "imu_position_indices": "",
                        "maximum_difference": None,
                        "consecutive_index_steps": "",
                        "exact_matches": 0,
                    }
            rows.append({
                "local_record_key": row.local_record_key,
                "subject_id": row.subject_id,
                "source_file": filename,
                "task": row.task,
                "originally_empty_fe": bool(empty),
                "source_identity_matches": True,
                "imu_channel_status": imu_meta["imu_status"],
                "fe_channel_status": fe_status,
                **match,
                "imu_t0": imu_meta.get("imu_t0"),
                "imu_tlast": imu_meta.get("imu_tlast"),
                "fe_t0": (
                    float(fe_time[0]) if fe_time is not None else None
                ),
                "fe_tlast": (
                    float(fe_time[-1]) if fe_time is not None else None
                ),
            })
    return (
        pd.DataFrame(rows),
        pd.DataFrame(empty_rows),
        pd.DataFrame(file_errors, columns=[
            "source_file", "error_class", "error"
        ]),
        first_mapping,
    )



# =========== 04. Evidence Summaries and Independent Quality Gates ============
def summarise_temporal_evidence(records, empty_fields, reader_errors):
    """Separate reader success, source missingness and original mapping."""
    if len(records) != 1897:
        raise ValueError("Incomplete record-level temporal mapping audit.")
    original_empty = records["originally_empty_fe"]
    states = records["mapping_status"]
    matched = states.isin([
        "EXACT_TIME_COORDINATE_MAPPING",
        "TOLERANT_TIME_COORDINATE_MAPPING",
    ])
    empty_verified = empty_fields[
        "empty_representation_status"
    ].eq("READER_EMPTY_REPRESENTATION") if not empty_fields.empty else pd.Series(dtype=bool)
    all_source_valid = (
        records["source_identity_matches"].all()
        and len(reader_errors) == 0
        and int(original_empty.sum()) == 168
    )
    return {
        "source_records": len(records),
        "participants": int(records["subject_id"].nunique()),
        "source_identity_gate": bool(all_source_valid),
        "registry_declared_fe_empty": int(original_empty.sum()),
        "reader_verified_empty_representations": int(empty_verified.sum()),
        "reader_unresolved_empty_representations": int(
            len(empty_fields) - empty_verified.sum()
        ),
        "fe_with_original_target": int((~original_empty).sum()),
        "exact_or_tolerant_coordinate_mappings": int(matched.sum()),
        "unmatched_nonempty_fe_records": int(
            ((~original_empty) & (~matched)).sum()
        ),
        "consistent_unique_source_mapping": bool(
            all_source_valid and (states[~original_empty].isin([
                "EXACT_TIME_COORDINATE_MAPPING",
                "TOLERANT_TIME_COORDINATE_MAPPING",
            ])).all()
        ),
        "source_fe_empty_resolution_complete": bool(
            len(empty_fields) == 168 and empty_verified.all()
        ),
        "physical_units_verified": False,
        "regression_ready": False,
    }

