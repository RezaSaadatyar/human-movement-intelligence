# ==================== Presented by: Reza Saadatyar (2026) ====================

"""Audit original Essex MATLAB records without fabricating numeric signals."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============== 01. Workbook Participant Identifier Validation ===============

def normalise_subject_id(value):
    """Recognise only explicit SubjXX labels; reject spreadsheet summary rows."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = re.fullmatch(r"subj\s*0*(\d+)", str(value).strip(), flags=re.I)
    return f"subj{int(match.group(1)):02d}" if match else None


def audit_workbook_ids(path, mat_ids):
    """Verify exact participant coverage without exporting sensitive columns."""
    table = pd.read_excel(path)
    if "Participent ID" not in table.columns:
        raise ValueError("Missing original workbook column: Participent ID")
    identifiers = [
        normalise_subject_id(value) for value in table["Participent ID"]
    ]
    recognised = [value for value in identifiers if value is not None]
    unique = sorted(set(recognised))
    expected = set(mat_ids)
    report = {
        "workbook_row_count": len(table),
        "recognised_participant_rows": len(recognised),
        "unique_participant_ids": len(unique),
        "duplicate_ids": len(recognised) - len(unique),
        "mat_ids_not_in_workbook": sorted(expected - set(unique)),
        "workbook_ids_not_in_mat": sorted(set(unique) - expected),
        "id_alignment_passed": (
            set(unique) == expected
            and len(recognised) == len(unique) == len(expected)
        ),
    }
    return report, pd.DataFrame({"subject_id": unique})


# =============== 02. Conservative MATLAB Field Representation ================

def _field(record, field):
    """Get a named original struct field, without inventing defaults."""
    if hasattr(record, "_fieldnames"):
        return getattr(record, field, None)
    if isinstance(record, np.void) and record.dtype.names:
        return record[field] if field in record.dtype.names else None
    return None


def _text(value):
    """Extract only unambiguous strings or scalar integers from MATLAB fields."""
    for _ in range(5):
        if value is None:
            return None
        if isinstance(value, (str, np.str_)):
            return value.strip() or None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return str(int(value)) if np.isfinite(value) and float(value).is_integer() else None
        if not isinstance(value, np.ndarray) or value.dtype.names:
            return None
        if value.dtype.kind in "US" and value.size:
            return "".join(map(str, value.reshape(-1))).strip() or None
        if value.size != 1:
            return None
        value = value.item()
    return None


def _representation(value):
    """Describe encoding; object arrays are not accepted as FE waveforms."""
    if value is None:
        return "missing", ""
    array = np.asarray(value)
    shape = str(tuple(array.shape))
    if array.size == 0:
        status = "empty"
    elif array.dtype.names:
        status = "structured_or_opaque"
    elif array.dtype.kind in "iuf":
        status = "numeric_candidate" if np.isfinite(array).all() else "nonfinite_numeric"
    elif array.dtype.kind == "O":
        status = "matlab_object_unresolved"
    else:
        status = "non_numeric"
    return status, shape


# ================= 03. Original Record and Trial Provenance ==================

def inspect_mat_records(path):
    """Read source Data elements and preserve file-local provenance per record."""
    path = Path(path)
    subject = normalise_subject_id(path.stem)
    if subject is None:
        raise ValueError(f"Unrecognised participant filename: {path.name}")
    data = loadmat(
        path, variable_names=["Data"], struct_as_record=False,
        squeeze_me=False, verify_compressed_data_integrity=True,
    )
    if "Data" not in data:
        raise ValueError(f"No Data variable in {path.name}")
    records = np.asarray(data["Data"]).reshape(-1)
    if not records.size:
        raise ValueError(f"Empty Data variable in {path.name}")
    rows = []
    for index, record in enumerate(records):
        if not hasattr(record, "_fieldnames"):
            raise ValueError(f"Unsupported struct at {path.name}:{index}")
        imu_status, imu_shape = _representation(_field(record, "IMU"))
        fe_status, fe_shape = _representation(_field(record, "e1e3"))
        rows.append({
            "subject_id": subject,
            "source_file": path.name,
            "local_record_index": index,
            "local_record_key": f"{subject}:{index:04d}",
            "source_subject_field": _text(_field(record, "SubjID")),
            "task": _text(_field(record, "Task")),
            "subtask": _text(_field(record, "SubTask")),
            "trial_number": _text(_field(record, "TrialNum")),
            "updated_trial_id": _text(_field(record, "UpdatedTrialID")),
            "imu_representation": imu_status,
            "imu_shape": imu_shape,
            "fe_representation": fe_status,
            "fe_shape": fe_shape,
        })
    return rows


def audit_mat_records(paths):
    """Inspect all MAT files; errors cannot be silently excluded."""
    rows = []
    for path in sorted(map(Path, paths), key=lambda value: value.name):
        rows.extend(inspect_mat_records(path))
    result = pd.DataFrame(rows)
    if result.empty or not result["local_record_key"].is_unique:
        raise ValueError("No records or ambiguous local source keys")
    return result


def summarise_trial_evidence(records):
    """Count readable trial IDs and representation types descriptively."""
    readable = records.dropna(subset=["updated_trial_id"])
    keyed = (
        readable["subject_id"].str.cat(readable["updated_trial_id"], sep=":")
    )
    return {
        "source_records": int(len(records)),
        "source_participants": int(records["subject_id"].nunique()),
        "readable_updated_trial_ids": int(len(readable)),
        "duplicate_subject_trial_ids": int(keyed.duplicated().sum()),
        "readable_tasks": int(records["task"].notna().sum()),
        "imu_encoding_counts": dict(Counter(records["imu_representation"])),
        "fe_encoding_counts": dict(Counter(records["fe_representation"])),
    }

