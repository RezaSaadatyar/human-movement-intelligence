# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Validate MATLAB-native numeric Essex exports against the frozen registry.

No source-object reinterpretation, unit inference, FE interpolation, or
model-ready claim is made. Exported numeric MAT files remain git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat

MAX_ARCHIVE_BYTES = 2 * 1024 ** 3
MAX_UNPACKED_BYTES = 3 * 1024 ** 3
MAX_FILES = 10_000


# =================== 01. Optional Native Export Discovery ====================
def find_native_archive(raw_dir):
    """Resolve one export ZIP, or None when MATLAB conversion is unavailable."""
    locations = [
        Path("/content/native_essex_export.zip"),
        Path(raw_dir) / "native_essex_export.zip",
    ]
    existing = [p for p in locations if p.is_file()]
    if len(existing) > 1:
        raise ValueError("Ambiguous native exports in /content and raw-data folder.")
    return existing[0] if existing else None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_native_archive(archive_path, target):
    """Validate and extract only an approved MATLAB-native export layout.

    Never overwrite a preexisting different file or write outside target.
    """
    archive_path = Path(archive_path)
    target = Path(target)
    if not 0 < archive_path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("Native export ZIP size outside safety bounds.")
    with ZipFile(archive_path) as zipped:
        members = zipped.infolist()
        if not members or len(members) > MAX_FILES:
            raise ValueError("Unexpected native-export member count.")
        names = set()
        total = 0
        for member in members:
            name = member.filename
            entry = PurePosixPath(name)
            if (
                entry.is_absolute() or ".." in entry.parts
                or "\\" in name or ":" in name
                or any(part.startswith(".") for part in entry.parts)
            ):
                raise ValueError(f"Unsafe export archive path: {name}")
            if name in names:
                raise ValueError(f"Duplicate ZIP member: {name}")
            names.add(name)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("MATLAB export archive symlinks are not permitted.")
            if not (
                name in ("export_manifest.csv", "export_readme.txt", "numeric/")
                or (name.startswith("numeric/") and name.lower().endswith(".mat"))
            ):
                raise ValueError(f"Unexpected export archive file: {name}")
            total += member.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Native export expands beyond the safety bound.")
        if "export_manifest.csv" not in names:
            raise ValueError("Missing original MATLAB export manifest.")
        if zipped.testzip() is not None:
            raise ValueError("Native-export ZIP CRC integrity failed.")
        target.mkdir(parents=True, exist_ok=True)
        for member in members:
            if member.is_dir():
                continue
            dest = target / member.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Reuse only byte-for-byte identical validated outputs.
                digest = hashlib.sha256()
                with zipped.open(member) as stream:
                    for block in iter(lambda: stream.read(2 * 1024 ** 2), b""):
                        digest.update(block)
                if sha256_file(dest) != digest.hexdigest():
                    raise FileExistsError(f"Existing numeric export differs: {dest}")
                continue
            with zipped.open(member) as source, dest.open("xb") as out:
                shutil.copyfileobj(source, out, length=2 * 1024 ** 2)
    return target


# ================ 02. Cross-Source Numeric Manifest Contract =================
EXPORT_COLUMNS = [
    "subject_id", "source_file", "source_record_index", "task",
    "updated_trial_id", "modality", "source_class", "status",
    "n_time", "n_channels", "column_names", "export_relpath",
]


def validate_native_manifest(path, registry):
    """Require exactly one IMU and FE export status per original Data element."""
    table = pd.read_csv(
        path, keep_default_na=False,
        dtype={"subject_id": "string", "source_file": "string",
               "task": "string", "updated_trial_id": "string",
               "modality": "string", "status": "string"},
    )
    if not set(EXPORT_COLUMNS).issubset(table):
        raise ValueError("MATLAB native manifest lacks mandatory source fields.")
    if len(table) != 2 * len(registry):
        raise ValueError("Expected exactly two modality statuses per original record.")
    index = pd.to_numeric(table["source_record_index"], errors="raise")
    if not np.isfinite(index).all() or not (index >= 0).all():
        raise ValueError("Invalid original MATLAB record indexes.")
    if not (index == index.astype("int64")).all():
        raise ValueError("MATLAB record indexes must be whole numbers.")
    table["source_record_index"] = index.astype("int64")
    table["local_record_key"] = (
        table["subject_id"].astype(str)
        + ":" + table["source_record_index"].map(lambda i: f"{i:04d}")
    )
    counts = table.groupby(["local_record_key", "modality"]).size()
    if counts.ne(1).any() or len(counts) != 2 * len(registry):
        raise ValueError("Duplicate or missing record-modality combinations.")
    if set(table["modality"]) != {"IMU", "FE"}:
        raise ValueError("Only original IMU and FE source fields are permitted.")

    ref = registry[
        ["local_record_key", "subject_id", "source_file",
         "task", "updated_trial_id"]
    ].copy()
    merged = table.merge(
        ref, on="local_record_key", validate="many_to_one",
        suffixes=("", "_reference"), indicator=True,
    )
    if len(merged) != len(table) or not merged["_merge"].eq("both").all():
        raise ValueError("Export rows do not match approved record registry.")
    for field in ["subject_id", "source_file", "task", "updated_trial_id"]:
        if not merged[field].astype(str).eq(
            merged[f"{field}_reference"].astype(str)
        ).all():
            raise ValueError(f"Original identity mismatch: {field}")

    known_status = {
        "exported", "empty", "unsupported_class", "unsupported_column",
        "nonfinite", "duplicate_columns",
    }
    if not set(table["status"]).issubset(known_status):
        raise ValueError("Unrecognised original MATLAB conversion status.")
    fe_empty = set(
        registry.loc[
            registry["fe_representation"] == "empty",
            "local_record_key",
        ]
    )
    if set(
        table.loc[
            (table["modality"] == "FE") & (table["status"] == "empty"),
            "local_record_key",
        ]
    ) != fe_empty:
        raise ValueError("Empty FE targets disagree with source audit.")
    return table


# ================= 03. Numerical Payload and Axis Validation =================
def validate_numeric_payloads(export_dir, manifest):
    """Read genuine native MATLAB v7 numeric arrays and report QA per export."""
    export_dir = Path(export_dir)
    rows = []
    for rec in manifest.itertuples(index=False):
        result = {
            "local_record_key": rec.local_record_key,
            "subject_id": rec.subject_id,
            "task": rec.task,
            "updated_trial_id": rec.updated_trial_id,
            "modality": rec.modality,
            "conversion_status": rec.status,
            "n_time": 0,
            "n_channels": 0,
            "finite": False,
            "label_count_matches": False,
            "time_101_candidate": False,
            "fe_11_candidate": False,
            "expected_imu_42_channels": False,
            "named_fe_e1_e3_candidate": False,
            "sha256": "",
        }
        if rec.status != "exported":
            rows.append(result)
            continue
        name = str(rec.export_relpath)
        if not re.fullmatch(
            r"numeric/subj[0-9]+__r[0-9]{4}__(IMU|FE)\.mat", name
        ):
            raise ValueError(f"Invalid native-export file path: {name}")
        expected_name = (
            f"{rec.subject_id}__r{int(rec.source_record_index):04d}"
            f"__{rec.modality}.mat"
        )
        if name != f"numeric/{expected_name}":
            raise ValueError("Numeric file name does not match original record.")
        source = export_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing numeric export: {name}")
        saved = loadmat(
            source, variable_names=["values"],
            struct_as_record=False, squeeze_me=False,
        )
        if "values" not in saved:
            raise ValueError(f"Missing numeric 'values' array: {name}")
        arr = np.asarray(saved["values"])
        if arr.ndim != 2 or arr.dtype.kind not in "iuf":
            raise ValueError(f"Source numeric payload has invalid dtype: {name}")
        if not arr.size:
            raise ValueError(f"Exported payload is empty: {name}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Exported payload contains NaN/Inf: {name}")
        if arr.shape != (int(rec.n_time), int(rec.n_channels)):
            raise ValueError(f"Manifest shape differs from payload: {name}")
        try:
            names = json.loads(str(rec.column_names))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid column name JSON: {name}") from exc
        if not isinstance(names, list) or not all(isinstance(x,str) for x in names):
            raise ValueError(f"Non-string column labels: {name}")
        if len(names) != arr.shape[1] or len(names) != len(set(names)):
            raise ValueError(f"Missing/duplicate original source column names: {name}")
        norm_names = {x.strip().lower() for x in names}
        result.update({
            "n_time": int(arr.shape[0]),
            "n_channels": int(arr.shape[1]),
            "finite": True,
            "label_count_matches": True,
            "time_101_candidate": bool(arr.shape[0] == 101),
            "fe_11_candidate": bool(rec.modality == "FE" and arr.shape[0] == 11),
            "expected_imu_42_channels": bool(
                rec.modality == "IMU" and arr.shape[1] == 42
            ),
            "named_fe_e1_e3_candidate": bool(
                rec.modality == "FE" and arr.shape[1] == 2
                and norm_names == {"e1", "e3"}
            ),
            "sha256": sha256_file(source),
        })
        rows.append(result)
    return pd.DataFrame(rows)


def conversion_summary(manifest, verified):
    """Report numerical accessibility separately from biomedical validation."""
    status = manifest.groupby(["modality", "status"]).size()
    valid = verified.loc[verified["finite"]]
    return {
        "manifest_rows": int(len(manifest)),
        "exported_numeric_payloads": int((manifest["status"]=="exported").sum()),
        "validated_finite_payloads": int(len(valid)),
        "imu_time_101_candidate": int(valid.loc[
            valid["modality"]=="IMU", "time_101_candidate"
        ].sum()),
        "imu_42_channel_candidate": int(
            valid["expected_imu_42_channels"].sum()
        ),
        "fe_time_11_candidate": int(valid["fe_11_candidate"].sum()),
        "fe_named_pair_candidate": int(valid["named_fe_e1_e3_candidate"].sum()),
        "conversion_status_counts": {
            f"{m}/{s}": int(n) for (m,s),n in status.items()
        },
        "physical_units_verified": False,
        "imu_fe_temporal_alignment_verified": False,
        "model_ready": False,
    }
