# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Descriptive, source-grounded evidence for Camargo Gait Notebook 02.

This module does not infer vertical-force direction, foot side, force units,
or a verified common clock from channel names or visual signal shape.
It does not create contact-event labels or select model settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ====================== 01. Timestamp and Channel Inspection =================

def inspect_timestamp(time_values, label: str) -> dict:
    """Describe sampling in original Header units, without asserting seconds."""
    values = np.asarray(time_values, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError(f"{label}: need >=3 finite timestamp values")
    steps = np.diff(values)
    if (steps <= 0).any():
        raise ValueError(f"{label}: non-increasing original timestamps")
    middle = float(np.median(steps))
    return {
        f"{label}_start_header": float(values[0]),
        f"{label}_end_header": float(values[-1]),
        f"{label}_median_header_step": middle,
        f"{label}_max_step_over_median": float(steps.max() / middle),
        f"{label}_samples_per_header_unit": float(1.0 / middle),
        f"{label}_n_samples": int(values.size),
    }


def summarise_force_component(values) -> dict:
    """Summarise a force component without assigning axis, sign or units."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        raise ValueError("Force component must contain >=3 finite samples")
    p01, p05, p50, p95, p99 = np.percentile(x, [1, 5, 50, 95, 99])
    scale = float(max(np.percentile(np.abs(x), 99), 1e-12))
    return {
        "minimum_source": float(x.min()),
        "p01_source": float(p01),
        "p05_source": float(p05),
        "median_source": float(p50),
        "p95_source": float(p95),
        "p99_source": float(p99),
        "maximum_source": float(x.max()),
        "range_p01_p99_source": float(p99 - p01),
        "near_zero_fraction_relative": float((np.abs(x) < 0.02 * scale).mean()),
    }


def inspect_source_pair(imu: pd.DataFrame, fp: pd.DataFrame,
                        imu_time, fp_time) -> tuple[dict, list[dict]]:
    """Inspect actual table fields without generating contacts or labels."""
    ti = np.asarray(imu_time, float)
    tf = np.asarray(fp_time, float)
    evidence = {
        **inspect_timestamp(ti, "imu"),
        **inspect_timestamp(tf, "fp"),
        "start_difference_header": float(ti[0] - tf[0]),
        "end_difference_header": float(ti[-1] - tf[-1]),
        "overlap_header": float(min(ti[-1], tf[-1]) - max(ti[0], tf[0])),
    }
    if evidence["overlap_header"] <= 0:
        raise ValueError("No common original Header interval")
    evidence["time_overlap_not_clock_proof"] = True
    channels: list[dict] = []
    for prefix in ("Treadmill_R", "Treadmill_L"):
        for component in ("vx", "vy", "vz"):
            name = f"{prefix}_{component}"
            if name not in fp.columns:
                continue
            channels.append({
                "channel": name,
                **summarise_force_component(
                    pd.to_numeric(fp[name], errors="raise").to_numpy(float)
                ),
            })
    if not channels:
        raise ValueError("No named treadmill R/L force components were readable")
    evidence["observed_force_component_count"] = len(channels)
    evidence["observed_foot_gyro_axes"] = ",".join(
        name for name in ("foot_Gyro_X", "foot_Gyro_Y", "foot_Gyro_Z")
        if name in imu.columns
    )
    return evidence, channels


# ========================= 02. Trial-Level Evidence ==========================

def audit_candidate_pairs(pairs: pd.DataFrame, source_root: Path,
                          read_sensor_table: Callable,
                          validate_timebase: Callable
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit current candidate pairs; record missing files, never download."""
    source_root = Path(source_root)
    records: list[dict] = []
    force_records: list[dict] = []
    required = {
        "subject_id", "session_id", "condition", "trial_id",
        "imu_path", "fp_path", "candidate_pair",
    }
    missing_columns = required - set(pairs.columns)
    if missing_columns:
        raise ValueError(f"Pair registry is missing: {sorted(missing_columns)}")
    for row in pairs.itertuples(index=False):
        if not (bool(row.candidate_pair) and row.condition == "treadmill"):
            continue
        key = {
            "subject_id": str(row.subject_id),
            "session_id": str(row.session_id),
            "condition": str(row.condition),
            "trial_id": str(row.trial_id),
        }
        imu_path = source_root / row.imu_path
        fp_path = source_root / row.fp_path
        try:
            absent = [str(x) for x in (imu_path, fp_path) if not x.is_file()]
            if absent:
                raise FileNotFoundError(
                    "Original MAT input not mounted: " + ", ".join(absent)
                )
            imu = read_sensor_table(imu_path)
            fp = read_sensor_table(fp_path)
            ti = validate_timebase(imu, "IMU")
            tf = validate_timebase(fp, "force")
            evidence, components = inspect_source_pair(imu, fp, ti, tf)
            records.append({
                **key, "inspection_status": "inspected",
                **evidence, "inspection_note": "",
            })
            force_records.extend({**key, **record} for record in components)
        except (ValueError, KeyError, FileNotFoundError, TypeError, OSError) as error:
            records.append({
                **key, "inspection_status": "not_inspected",
                "inspection_note": str(error),
            })
    return (pd.DataFrame.from_records(records),
            pd.DataFrame.from_records(force_records))


# ========================== 03. Descriptive Signal Plot ======================

def plot_source_contract_probe(pair: pd.Series, source_root: Path,
                               read_sensor_table: Callable,
                               validate_timebase: Callable,
                               out_path: Path, duration_header: float = 8.0):
    """Plot original force components and gyro axes without contact labels."""
    import matplotlib.pyplot as plt

    source_root = Path(source_root)
    imu = read_sensor_table(source_root / pair.imu_path)
    force = read_sensor_table(source_root / pair.fp_path)
    ti = np.asarray(validate_timebase(imu, "IMU"), float)
    tf = np.asarray(validate_timebase(force, "force"), float)
    start = max(ti[0], tf[0])
    stop = min(ti[-1], tf[-1], start + duration_header)
    if stop <= start:
        raise ValueError("No original Header interval for plotting")
    mi = (ti >= start) & (ti <= stop)
    mf = (tf >= start) & (tf <= stop)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    for name in ("foot_Gyro_X", "foot_Gyro_Y", "foot_Gyro_Z"):
        if name in imu:
            axes[0].plot(ti[mi], imu.loc[mi, name], label=name, linewidth=0.9)
    for prefix, ax in (("Treadmill_R", axes[1]),
                       ("Treadmill_L", axes[2])):
        for component in ("vx", "vy", "vz"):
            name = f"{prefix}_{component}"
            if name in force:
                ax.plot(tf[mf], force.loc[mf, name],
                        label=name, linewidth=0.9)
    for ax, title in zip(axes, (
        "Foot gyroscope: original channels",
        "Right treadmill: original force components",
        "Left treadmill: original force components",
    )):
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Original Header (time unit not independently validated)")
        ax.set_ylabel("Original recorded units")
        if ax.lines:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Descriptive source audit: no force-axis, side or IC/TC approval",
        fontsize=12,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ====================== 04. Source Contract Data Validation ==================

def validate_explicit_contract(*, gyro_column: str | None,
                               force_column: str | None,
                               foot_side: str | None,
                               force_upward_sign: int | None,
                               time_unit_in_seconds: float | None,
                               force_scale_to_newtons: float | None,
                               imu_force_timebase_confirmed: bool,
                               verified_original_source_note: str | None,
                               observed_imu_columns: set[str],
                               observed_force_columns: set[str]) -> dict:
    """Check explicit settings, not their independent scientific provenance."""
    missing = []
    if not gyro_column or gyro_column not in observed_imu_columns:
        missing.append("GYRO_COLUMN (observed, reviewed axis)")
    if not force_column or force_column not in observed_force_columns:
        missing.append("VERTICAL_FORCE_COLUMN (observed, reviewed axis)")
    if foot_side not in {"left", "right"}:
        missing.append("FOOT_SIDE")
    if force_upward_sign not in {-1, 1}:
        missing.append("FORCE_UPWARD_SIGN")
    if (time_unit_in_seconds is None or
            not np.isfinite(time_unit_in_seconds) or time_unit_in_seconds <= 0):
        missing.append("TIME_UNIT_IN_SECONDS")
    if (force_scale_to_newtons is None or
            not np.isfinite(force_scale_to_newtons) or
            force_scale_to_newtons <= 0):
        missing.append("FORCE_SCALE_TO_NEWTONS")
    if imu_force_timebase_confirmed is not True:
        missing.append("IMU_FORCE_TIMEBASE_CONFIRMED")
    if not (verified_original_source_note and
            verified_original_source_note.strip()):
        missing.append("ORIGINAL_SOURCE_NOTE")
    if missing:
        raise ValueError("Unresolved source contract: " + "; ".join(missing))
    return {
        "GYRO_COLUMN": gyro_column,
        "VERTICAL_FORCE_COLUMN": force_column,
        "FOOT_SIDE": foot_side,
        "FORCE_UPWARD_SIGN": int(force_upward_sign),
        "TIME_UNIT_IN_SECONDS": float(time_unit_in_seconds),
        "FORCE_SCALE_TO_NEWTONS": float(force_scale_to_newtons),
        "IMU_FORCE_TIMEBASE_CONFIRMED": True,
    }


# ================ 06. Candidate Contact Diagnostics (NOT Reference) ===========

def recorded_unit_contact_candidates(
    time_s, force_recorded, *, threshold_recorded=20.0,
    min_contact_s=0.08, min_swing_s=0.06,
) -> pd.DataFrame:
    """Extract exploratory source-amplitude crossings, never verified IC/TC.

    The threshold is in ORIGINAL RECORDED UNITS, not asserted to be newtons.
    A force source with an unknown physical axis/side may generate attractive
    but invalid candidates. This output must not enter final evaluation.
    """
    t = np.asarray(time_s, dtype=float)
    f = np.asarray(force_recorded, dtype=float)
    if (t.ndim != 1 or f.ndim != 1 or t.size != f.size or t.size < 4
            or not np.isfinite(t).all() or not np.isfinite(f).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Candidate force/time arrays must be finite, 1-D, "
                         "strictly increasing and matched in length.")
    if (not np.isfinite(threshold_recorded) or threshold_recorded <= 0
            or min_contact_s <= 0 or min_swing_s <= 0):
        raise ValueError("Invalid exploratory contact parameters.")
    active = f >= threshold_recorded
    changes = np.r_[0, np.flatnonzero(np.diff(active.astype(np.int8))) + 1,
                    len(t)]
    segments = [
        (int(a), int(b), bool(active[a]))
        for a, b in zip(changes[:-1], changes[1:])
    ]
    rows = []
    for i, (a, b, loaded) in enumerate(segments):
        if not loaded or i == 0 or i == len(segments) - 1:
            continue
        pre = segments[i - 1]
        post = segments[i + 1]
        if (t[b - 1] - t[a] < min_contact_s
                or t[a] - t[pre[0]] < min_swing_s
                or t[post[1] - 1] - t[post[0]] < min_swing_s):
            continue
        rows.append({"candidate_onset_header": float(t[a]),
                     "candidate_offset_header": float(t[b]),
                     "candidate_duration_header": float(t[b] - t[a]),
                     "threshold_recorded_units": float(threshold_recorded)})
    return pd.DataFrame(rows, columns=[
        "candidate_onset_header", "candidate_offset_header",
        "candidate_duration_header", "threshold_recorded_units",
    ])


def audit_recorded_force_candidates(pairs: pd.DataFrame, source_root: Path,
                                    read_sensor_table: Callable,
                                    validate_timebase: Callable,
                                    thresholds_recorded=(10.0, 20.0, 30.0)
                                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen both treadmill sides for all candidate trials in original units.

    No sensor unit, physical vertical axis, side or event clock is declared
    verified. A per-trial exclusion is kept visible, never silently dropped.
    This may be executed before the final independent reference contract.
    """
    source_root = Path(source_root)
    required = {"subject_id", "session_id", "condition", "trial_id",
                "fp_path", "candidate_pair"}
    absent = required - set(pairs.columns)
    if absent:
        raise ValueError(f"Candidate registry missing fields: {sorted(absent)}")
    if not thresholds_recorded or any(
        not np.isfinite(x) or x <= 0 for x in thresholds_recorded
    ):
        raise ValueError("Invalid candidate source-unit thresholds.")
    records, events = [], []
    identities = set()
    for row in pairs.itertuples(index=False):
        if not (bool(row.candidate_pair) and row.condition == "treadmill"):
            continue
        identity = tuple(str(getattr(row, field)) for field in
                         ("subject_id", "session_id", "condition", "trial_id"))
        if identity in identities:
            raise ValueError(f"Duplicate candidate trial identity: {identity}")
        identities.add(identity)
        base = dict(zip(("subject_id", "session_id", "condition", "trial_id"),
                        identity))
        try:
            path = source_root / row.fp_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Original force recording is unavailable: {path}"
                )
            force = read_sensor_table(path)
            t = np.asarray(validate_timebase(force, "force"), dtype=float)
            for name in ("Treadmill_R_vy", "Treadmill_L_vy"):
                if name not in force:
                    raise KeyError(f"Missing observed candidate: {name}")
                values = pd.to_numeric(force[name], errors="raise").to_numpy(
                    dtype=float
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite recorded force: {name}")
                for threshold in thresholds_recorded:
                    c = recorded_unit_contact_candidates(
                        t, values, threshold_recorded=threshold
                    )
                    onset = c.candidate_onset_header.to_numpy(dtype=float)
                    intervals = np.diff(onset)
                    plausible = ((intervals >= 0.35) &
                                 (intervals <= 3.0))
                    records.append({
                        **base, "candidate_channel": name,
                        "threshold_recorded_units": float(threshold),
                        "inspection_status": "inspected",
                        "n_contact_candidates": len(c),
                        "n_plausible_same_side_intervals": int(plausible.sum()),
                        "median_same_side_interval_header": (
                            float(np.median(intervals[plausible]))
                            if plausible.any() else np.nan
                        ),
                        "median_candidate_duration_header": (
                            float(c.candidate_duration_header.median())
                            if len(c) else np.nan
                        ),
                        "source_p99": float(np.percentile(values, 99)),
                        "source_p01": float(np.percentile(values, 1)),
                        "physical_reference_validated": False,
                        "inspection_note": "",
                    })
                    # Save primary, predeclared 20 original-unit candidate
                    # events only. The other thresholds are QC comparisons.
                    if threshold == 20.0:
                        events.extend({
                            **base, "candidate_channel": name,
                            "threshold_recorded_units": 20.0,
                            "candidate_onset_header":
                                float(r.candidate_onset_header),
                            "candidate_offset_header":
                                float(r.candidate_offset_header),
                            "physical_reference_validated": False,
                        } for r in c.itertuples(index=False))
        except (FileNotFoundError, KeyError, TypeError, ValueError,
                OSError) as error:
            records.append({
                **base, "candidate_channel": "",
                "threshold_recorded_units": np.nan,
                "inspection_status": "not_inspected",
                "n_contact_candidates": 0,
                "n_plausible_same_side_intervals": 0,
                "median_same_side_interval_header": np.nan,
                "median_candidate_duration_header": np.nan,
                "source_p99": np.nan, "source_p01": np.nan,
                "physical_reference_validated": False,
                "inspection_note": str(error),
            })
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(events)


def plot_recorded_force_candidate_probe(
    force_table: pd.DataFrame, original_header, candidates: pd.DataFrame,
    out_path: Path, *, window_s: float = 8.0
) -> Path:
    """Display candidate contact intervals, not ground-truth annotations."""
    import matplotlib.pyplot as plt
    times = np.asarray(original_header, dtype=float)
    if len(times) < 3 or not np.isfinite(times).all():
        raise ValueError("Missing original source timestamps for plot.")
    start = float(times[0])
    use = (times >= start) & (times <= start + window_s)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             layout="constrained")
    for ax, name in zip(axes, ("Treadmill_R_vy", "Treadmill_L_vy")):
        if name not in force_table:
            raise KeyError(f"Missing force channel {name}")
        ax.plot(times[use], force_table.loc[use, name],
                linewidth=0.8, label=f"{name} (recorded units)")
        ax.axhline(20.0, linewidth=1.0, linestyle="--",
                   label="20 recorded units (exploratory)")
        group = candidates.loc[
            candidates.candidate_channel.eq(name)
        ] if not candidates.empty else pd.DataFrame()
        for r in group.itertuples(index=False):
            if start <= r.candidate_onset_header <= start + window_s:
                ax.axvline(r.candidate_onset_header,
                           linewidth=0.7, alpha=0.45)
        ax.set_ylabel("Original recorded units")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Original Header (seconds per source metadata)")
    axes[0].set_title("Source-unit candidate transitions; NOT verified IC/TC",
                      loc="left")
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output
