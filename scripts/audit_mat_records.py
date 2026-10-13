"""Audit all records in subject MAT files without exporting movement signals.

This is a metadata and missingness audit, NOT a dataset loader. MATLAB opaque
object references are marked unresolved rather than interpreted as numeric data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

REQUIRED_FIELDS = (
    "SubjID", "Task", "SubTask", "TrialNum", "TrialNumUpdated",
    "UpdatedTrialID", "Cycle", "IMU", "ID", "FP", "e1e3", "JCF", "MuscleF",
)
OPAQUE_FIELDS = {"_TypeSystem", "_Class", "_ObjectMetadata"}


def classify_field(value: Any) -> str:
    """Classify source representation; never treat object references as signals."""
    if value is None:
        return "missing"
    if isinstance(value, mat_struct):
        return "struct"
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "empty"
        if value.dtype.names:
            if OPAQUE_FIELDS.issubset(value.dtype.names):
                return "matlab_object_reference"
            return "structured_array"
        if value.dtype.kind in "iufcb":
            return "numeric_array"
        if value.dtype.kind in "US":
            return "text_array"
        if value.dtype.kind == "O":
            if value.size == 1:
                return classify_field(value.flat[0])
            return "object_array"
        return "other_array"
    if isinstance(value, (str, bytes, np.str_)):
        return "text"
    if np.isscalar(value):
        return "scalar"
    return type(value).__name__


def text_scalar(value: Any) -> str | None:
    """Decode source labels only when genuinely scalar text, not object metadata."""
    if isinstance(value, (str, np.str_)):
        return str(value).strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        if not value.size or value.size != 1:
            return None
        if value.dtype.kind in "US":
            return str(value.flat[0]).strip()
        if value.dtype.kind == "O":
            return text_scalar(value.flat[0])
    return None


def audit_records(data: np.ndarray, filename: str) -> dict[str, Any]:
    """Summarise all entries in Data; do not export signal or trial values."""
    if not isinstance(data, np.ndarray) or data.dtype.kind != "O":
        raise ValueError(f"{filename}: expected an object array named Data.")
    profiles = {name: Counter() for name in REQUIRED_FIELDS}
    tasks, subtasks, subject_ids, updated_ids = Counter(), Counter(), set(), []
    missing_by_record = 0
    for index, record in enumerate(data.flat):
        if not isinstance(record, mat_struct):
            raise ValueError(f"{filename}: Data entry {index} is not a MATLAB struct.")
        names = set(record._fieldnames or [])
        if names != set(REQUIRED_FIELDS):
            missing_by_record += 1
        for field in REQUIRED_FIELDS:
            value = getattr(record, field, None)
            profiles[field][classify_field(value)] += 1
        for field, target in (("Task", tasks), ("SubTask", subtasks)):
            value = text_scalar(getattr(record, field, None))
            target[value if value else "[unreadable]"] += 1
        subj = text_scalar(getattr(record, "SubjID", None))
        if subj:
            subject_ids.add(subj.casefold())
        identifier = text_scalar(getattr(record, "UpdatedTrialID", None))
        if identifier:
            updated_ids.append(identifier)

    updated_counts = Counter(updated_ids)
    duplicate_ids = sum(n - 1 for n in updated_counts.values() if n > 1)
    return {
        "filename": filename,
        "data_shape": list(data.shape),
        "records": int(data.size),
        "subject_ids_inside": sorted(subject_ids),
        "filename_subject_matches": subject_ids == {Path(filename).stem.casefold()},
        "records_with_nonstandard_field_set": missing_by_record,
        "tasks": dict(sorted(tasks.items())),
        "subtasks": dict(sorted(subtasks.items())),
        "updated_trial_id_readable": len(updated_ids),
        "updated_trial_id_unique": len(updated_counts),
        "updated_trial_id_duplicates": duplicate_ids,
        "fields": {name: dict(sorted(profiles[name].items())) for name in REQUIRED_FIELDS},
        "warnings": [
            "MATLAB object references are not decoded numeric arrays.",
            "Presence of an object reference does not prove a valid numeric target.",
            "Trial IDs, task labels and source sample alignment need further review.",
        ],
    }


def inspect_subject(path: Path) -> dict[str, Any]:
    """Read Data only, excluding the MATLAB internal function workspace."""
    obj = loadmat(
        str(path), variable_names=["Data"],
        squeeze_me=False, struct_as_record=False,
        verify_compressed_data_integrity=True,
    )
    if "Data" not in obj:
        raise ValueError(f"{path.name}: Data variable absent.")
    return audit_records(obj["Data"], path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "data/raw/essex_femoral_neck_loading"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subjects", nargs="+", default=None,
                           help="Names without .mat, e.g., subj04 subj16")
    selection.add_argument("--all", action="store_true", help="Inspect all MAT files")
    parser.add_argument("--report", type=Path, default=Path(
        "outputs/essex_all_record_audit.json"))
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.mat")) if args.root.is_dir() else []
    if not paths:
        parser.error(f"No MAT files found under {args.root}")
    index = {p.stem.casefold(): p for p in paths}
    if len(index) != len(paths):
        parser.error("Duplicate MAT filenames found under the data root.")
    if args.all:
        chosen = paths
    else:
        wanted = args.subjects or ["subj04", "subj16"]
        missing = [name for name in wanted if Path(name).stem.casefold() not in index]
        if missing:
            parser.error(f"Unknown subjects: {missing}")
        chosen = [index[Path(name).stem.casefold()] for name in wanted]

    results = []
    try:
        for path in chosen:
            entry = inspect_subject(path)
            entry["relative_path"] = path.relative_to(args.root).as_posix()
            results.append(entry)
            e = entry["fields"]["e1e3"]
            print(
                f"{path.name}: {entry['records']} records | "
                f"e1e3 empty={e.get('empty', 0)}, "
                f"opaque={e.get('matlab_object_reference', 0)} | "
                f"ID duplicates={entry['updated_trial_id_duplicates']}"
            )
    except (OSError, ValueError, TypeError, NotImplementedError) as exc:
        parser.exit(1, f"Record audit failed: {exc}\n")

    report = {
        "scope": "Metadata-only; MATLAB object values are unresolved",
        "mat_files_found": len(paths),
        "files_inspected": len(results),
        "total_records_inspected": sum(item["records"] for item in results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
