# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Validate trial-time overlap and explicitly mapped sensor channels."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from camargo_source import read_sensor_table, validate_timebase


# ============================ 01. Channel Mapping ===========================
def candidate_channels(table: pd.DataFrame, modality: str) -> list[str]:
    """Identify review candidates, not scientifically approved channel mappings."""
    term = (r"gyr|gyro|angular|foot|shank" if modality == "imu"
            else r"force|fz|vertical|fp[0-9]")
    return [str(col) for col in table.columns
            if re.search(term, str(col), flags=re.I)
            and col not in ("Header", "time_s")]


def mapped_series(table: pd.DataFrame, column: str, kind: str):
    """Read a verified channel selected after inspecting the original headings."""
    if not column or column not in table:
        raise ValueError(
            f"{kind} channel '{column}' absent. Inspect Notebook 01's "
            "observed sensor columns and set the explicit mapping in Notebook 02."
        )
    x = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"{kind} channel contains invalid samples.")
    return x


def validated_overlap(imu_time, force_time, min_seconds=4.0):
    """Report true timestamp overlap; never estimate an arbitrary phase shift."""
    imu_time = np.asarray(imu_time, dtype=float)
    force_time = np.asarray(force_time, dtype=float)
    if (len(imu_time) < 3 or len(force_time) < 3
            or not np.all(np.diff(imu_time) > 0)
            or not np.all(np.diff(force_time) > 0)):
        raise ValueError("Sensor timestamps must be strictly increasing.")
    begin = max(float(imu_time[0]), float(force_time[0]))
    end = min(float(imu_time[-1]), float(force_time[-1]))
    if end - begin < min_seconds:
        raise ValueError("No sufficiently long shared native timestamp interval.")
    return begin, end


def inspect_pair(source_root: Path, imu_relative: str, fp_relative: str):
    """Read paired original files and return timing, tables and inspected columns."""
    imu = read_sensor_table(Path(source_root) / imu_relative)
    force = read_sensor_table(Path(source_root) / fp_relative)
    ti = validate_timebase(imu, imu_relative)
    tf = validate_timebase(force, fp_relative)
    start, stop = validated_overlap(ti, tf)
    return {"imu_table": imu, "fp_table": force, "imu_time": ti,
            "fp_time": tf, "start_s": start, "end_s": stop,
            "imu_candidates": candidate_channels(imu, "imu"),
            "force_candidates": candidate_channels(force, "fp")}
