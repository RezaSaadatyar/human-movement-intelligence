# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Inspect Camargo's subject/date/mode/sensor/trial source organisation.

Only original source metadata are assumed: sensor folders (imu/fp), subject IDs
and a 'Header' timestamp field. MATLAB table/MCOS decoding is not universal;
unsupported source layouts fail explicitly instead of inventing data columns.
"""
from __future__ import annotations

from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

SENSOR_NAMES = {"imu", "fp", "conditions", "gcRight", "gcLeft"}
SUPPORTED = {".csv", ".parquet", ".mat"}


# =========================== 01. Source Discovery ===========================
def inventory_files(source_root: Path) -> pd.DataFrame:
    """List original sensor files without reading entire trial recordings."""
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Camargo source directory is missing: {source_root}. "
            "Extract one ABxx participant archive from the official source."
        )
    rows = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(source_root)
        parts = rel.parts
        subject = next((x for x in parts if re.fullmatch(r"AB\d+", x, re.I)), "")
        found = [(i, x) for i, x in enumerate(parts[:-1]) if x in SENSOR_NAMES]
        if not subject or not found:
            continue
        sensor_i, sensor = found[-1]
        mode = parts[sensor_i - 1] if sensor_i > 0 else ""
        date = parts[sensor_i - 2] if sensor_i > 1 else ""
        rows.append({
            "subject_id": subject.upper(), "session_id": date,
            "condition": mode, "sensor": sensor, "trial_id": path.stem,
            "relative_path": rel.as_posix(), "format": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        })
    return pd.DataFrame(rows, columns=[
        "subject_id", "session_id", "condition", "sensor", "trial_id",
        "relative_path", "format", "size_bytes",
    ])


def list_archives(source_root: Path) -> pd.DataFrame:
    """List local subject ZIPs without downloading or extracting them."""
    rows = []
    for path in sorted(Path(source_root).glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
        rows.append({"archive_name": path.name, "member_count": len(members),
                     "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows, columns=["archive_name", "member_count", "size_bytes"])


def paired_trials(inventory: pd.DataFrame) -> pd.DataFrame:
    """Match same-subject/session/condition/trial IMU and FP file paths."""
    fields = ["subject_id", "session_id", "condition", "trial_id"]
    if inventory.empty:
        return pd.DataFrame(columns=fields + [
            "imu_path", "fp_path", "candidate_pair", "pair_issue"
        ])
    rows = []
    for keys, group in inventory.groupby(fields, dropna=False, sort=True):
        imu = group.loc[group.sensor == "imu", "relative_path"].tolist()
        fp = group.loc[group.sensor == "fp", "relative_path"].tolist()
        issue = "" if len(imu) == len(fp) == 1 else (
            f"expected exactly one IMU and one FP file; found {len(imu)}/{len(fp)}"
        )
        rows.append(dict(zip(fields, keys), imu_path=imu[0] if len(imu) == 1 else "",
                         fp_path=fp[0] if len(fp) == 1 else "",
                         candidate_pair=not bool(issue), pair_issue=issue))
    return pd.DataFrame(rows)


# ========================== 02. Signal Table Reader =========================
def _dict_to_dataframe(data: dict) -> pd.DataFrame:
    """Accept numeric vector columns only; reject ambiguous MATLAB objects."""
    if "Header" not in data and "time_s" not in data:
        raise ValueError("No numeric 'Header' or 'time_s' field was decoded.")
    values = {}
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            values[str(key)] = arr.astype(float)
    if not values or len(set(map(len, values.values()))) != 1:
        raise ValueError("MATLAB field vectors have inconsistent lengths.")
    return pd.DataFrame(values)


def read_sensor_table(path: Path) -> pd.DataFrame:
    """Read a source sensor table, without assuming its physical channel map."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
            content = loadmat(path, simplify_cells=True)
            if "data" in content and isinstance(content["data"], dict):
                content = content["data"]
            data = _dict_to_dataframe(content)
        except (NotImplementedError, ValueError, TypeError) as error:
            raise ValueError(
                f"Cannot decode this MATLAB table directly: {path}. "
                "Export the source table as CSV with its original Header "
                "timestamps using MATLAB/Octave, or supply a validated "
                "numeric table decoder; do not reconstruct timestamps. "
                f"Original error: {error}"
            ) from error
    else:
        raise ValueError(f"Unsupported sensor file: {path.name}")
    if data.empty:
        raise ValueError(f"Empty sensor table: {path}")
    return data


def validate_timebase(table: pd.DataFrame, source: str = "sensor"):
    """Return seconds if stored timestamps are finite and strictly increasing."""
    candidates = [x for x in ["Header", "time_s"] if x in table.columns]
    if not candidates:
        raise ValueError(f"{source}: explicit Header/time_s timestamp missing.")
    t = pd.to_numeric(table[candidates[0]], errors="coerce").to_numpy(float)
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{source}: invalid/non-monotonic source timestamps.")
    return t


def inspect_columns(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Report observed fields without inferring axis, side or vertical force."""
    rows = []
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        rows.append({
            "relative_file": str(path), "column": str(column),
            "dtype": str(table[column].dtype),
            "numeric_fraction": float(numeric.notna().mean()),
            "min_observed": float(numeric.min()) if numeric.notna().any() else np.nan,
            "max_observed": float(numeric.max()) if numeric.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)
