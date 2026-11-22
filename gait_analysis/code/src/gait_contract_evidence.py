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
    # Preserve the event schema even when source files are unavailable.
    # An empty table must never be interpreted as verified zero contacts.
    event_fields = [
        "subject_id", "session_id", "condition", "trial_id",
        "candidate_channel", "threshold_recorded_units",
        "candidate_onset_header", "candidate_offset_header",
        "physical_reference_validated",
    ]
    return (pd.DataFrame.from_records(records),
            pd.DataFrame.from_records(events, columns=event_fields))


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


# ======================== 07. Reference Quality Review =======================

def audit_contact_candidate_quality(
    trial_audit: pd.DataFrame,
    candidate_intervals: pd.DataFrame,
    paired_timebases: pd.DataFrame,
) -> pd.DataFrame:
    """Review per-trial sensitivity without treating source units as newtons.

    Flags describe evidence for source inspection, not automatic exclusions,
    selected parameters or force-reference approval.
    """
    keys = ["subject_id", "session_id", "condition", "trial_id"]
    columns = keys + ["candidate_channel"]
    required = set(columns + [
        "threshold_recorded_units", "n_contact_candidates",
        "inspection_status",
    ])
    if trial_audit.empty or candidate_intervals.empty:
        raise ValueError(
            "No usable source-contact evidence: original Camargo MAT files "
            "may be missing from this Colab runtime. Do not overwrite "
            "previously published 02_* source outputs with empty tables. "
            "Use the current committed evidence-only review or restore "
            "the original selected IMU/force MAT pairs."
        )
    if required.difference(trial_audit.columns):
        raise ValueError("Candidate audit is missing required source columns")
    expected = set(columns + [
        "candidate_onset_header", "candidate_offset_header",
        "physical_reference_validated",
    ])
    if expected.difference(candidate_intervals.columns):
        raise ValueError("Candidate intervals are missing source columns")
    if set(keys + ["overlap_header"]).difference(paired_timebases.columns):
        raise ValueError("Trial timebase metadata missing")
    if trial_audit.duplicated(columns + [
        "threshold_recorded_units"
    ]).any():
        raise ValueError("Duplicate source trial/side/threshold entries")
    if paired_timebases.duplicated(keys).any():
        raise ValueError("Duplicate source trial timebase entries")
    if not trial_audit.inspection_status.eq("inspected").all():
        raise ValueError("Uninspected source trial needs separate review")
    if candidate_intervals.physical_reference_validated.astype(
        str
    ).str.lower().eq("true").any():
        raise ValueError("Source candidate unexpectedly claims verification")
    audit = trial_audit.copy()
    audit["threshold_recorded_units"] = pd.to_numeric(
        audit.threshold_recorded_units, errors="raise"
    )
    audit["n_contact_candidates"] = pd.to_numeric(
        audit.n_contact_candidates, errors="raise"
    )
    if (audit.n_contact_candidates.isna().any()
            or (audit.n_contact_candidates < 0).any()):
        raise ValueError("Invalid source contact counts")
    count_table = audit.pivot(
        index=columns,
        columns="threshold_recorded_units",
        values="n_contact_candidates",
    ).reset_index()
    for level in (10.0, 20.0, 30.0):
        if level not in count_table.columns:
            raise ValueError(f"Missing exploratory threshold {level}")
    intervals = candidate_intervals.copy()
    for field in ("candidate_onset_header", "candidate_offset_header"):
        intervals[field] = pd.to_numeric(intervals[field], errors="raise")
    measured_times = intervals[[
        "candidate_onset_header", "candidate_offset_header"
    ]].to_numpy(float)
    if (not np.isfinite(measured_times).all() or
            (measured_times[:, 1] <= measured_times[:, 0]).any()):
        raise ValueError("Invalid candidate onset/offset times")
    intervals["contact_duration_header"] = (
        intervals.candidate_offset_header -
        intervals.candidate_onset_header
    )
    measured = intervals.groupby(columns, dropna=False).agg(
        n_saved_candidates=("candidate_onset_header", "size"),
        median_contact_duration_header=("contact_duration_header", "median"),
    ).reset_index()
    quality = count_table.merge(
        measured, on=columns, how="left", validate="one_to_one"
    )
    quality["n_saved_candidates"] = (
        quality.n_saved_candidates.fillna(0).astype(int)
    )
    if not np.array_equal(
        quality[20.0].to_numpy(int),
        quality.n_saved_candidates.to_numpy(int),
    ):
        raise ValueError("Saved 20-unit candidate rows disagree with trial audit")
    durations = paired_timebases[keys + ["overlap_header"]].copy()
    durations["overlap_header"] = pd.to_numeric(
        durations.overlap_header, errors="raise"
    )
    if (not np.isfinite(durations.overlap_header).all()
            or (durations.overlap_header <= 0).any()):
        raise ValueError("Non-positive original trial overlap")
    quality = quality.merge(durations, on=keys, validate="many_to_one")
    quality["candidates_per_minute"] = (
        60 * quality[20.0] / quality.overlap_header
    )
    quality["count_ratio_30_over_10"] = (
        (quality[30.0] + 1) / (quality[10.0] + 1)
    )
    quality["threshold_sensitive_review"] = (
        quality.count_ratio_30_over_10 > 1.5
    )
    quality["low_yield_review"] = quality.candidates_per_minute < 10
    quality["physical_reference_validated"] = False
    quality["role"] = "descriptive QC only; not reference exclusion"
    return quality.sort_values(columns, kind="stable").reset_index(drop=True)


def audit_original_gc_right(
    pairs: pd.DataFrame,
    source_root: Path,
    read_sensor_table: Callable,
    validate_timebase: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect existing original gcRight streams and possible phase resets.

    No download, fitted time shift or automatic acceptance of original gait
    cycle phase as an independent force-derived ground-truth event.
    """
    keys = ["subject_id", "session_id", "condition", "trial_id"]
    if set(keys + ["fp_path", "candidate_pair"]) - set(pairs.columns):
        raise ValueError("Source pairs missing identifiers or force paths")
    rows, resets = [], []
    for row in pairs.itertuples(index=False):
        if not (bool(row.candidate_pair) and row.condition == "treadmill"):
            continue
        identity = {k: str(getattr(row, k)) for k in keys}
        fp_rel = Path(row.fp_path)
        gc_rel = fp_rel.parent.parent / "gcRight" / fp_rel.name
        gc_path = Path(source_root) / gc_rel
        record = {
            **identity, "gc_right_path": str(gc_rel),
            "source_status": "original_file_unavailable",
            "observed_columns": "",
            "n_phase_wrap_candidates": 0,
            "independent_reference_verified": False,
        }
        if gc_path.is_file():
            try:
                data = read_sensor_table(gc_path)
                t = np.asarray(validate_timebase(data, "gcRight"), float)
                record["observed_columns"] = ",".join(map(str, data.columns))
                if "HeelStrike" in data.columns:
                    p = pd.to_numeric(
                        data["HeelStrike"], errors="coerce"
                    ).to_numpy(float)
                    if (p.size == t.size and np.isfinite(p).all()
                            and p.min() >= -1 and p.max() <= 101):
                        hits = np.flatnonzero(np.diff(p) < -50) + 1
                        record["n_phase_wrap_candidates"] = len(hits)
                        record["source_status"] = "original_phase_inspected"
                        resets.extend({
                            **identity, "phase_wrap_header": float(t[i]),
                            "source_channel": "gcRight.HeelStrike",
                            "physical_reference_validated": False,
                        } for i in hits)
                    else:
                        record["source_status"] = "unknown_phase_scale"
                else:
                    record["source_status"] = "no_heelstrike_column"
            except (OSError, KeyError, ValueError, TypeError) as exc:
                record["source_status"] = (
                    f"read_error: {type(exc).__name__}: {exc}"
                )
        rows.append(record)
    return (pd.DataFrame(rows),
            pd.DataFrame(resets, columns=keys + [
                "phase_wrap_header", "source_channel",
                "physical_reference_validated",
            ]))


def restore_selected_original_gc_right(
    archive_path: Path,
    source_root: Path,
    selected_pairs: pd.DataFrame,
    *,
    max_member_bytes: int = 30_000_000,
) -> pd.DataFrame:
    """Restore only exact, same-trial original gcRight members from one ZIP.

    Existing recordings are never overwritten, and no unrelated source
    member, original archive or paired IMU/force file is removed.
    """
    import shutil
    import tempfile
    import zipfile

    keys = ["subject_id", "session_id", "condition", "trial_id"]
    if set(keys).difference(selected_pairs.columns):
        raise ValueError("Source pairing identifiers missing")
    archive_path, source_root = Path(archive_path), Path(source_root)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Original Camargo ZIP absent: {archive_path}")
    wanted = {
        tuple(str(getattr(row, k)) for k in keys)
        for row in selected_pairs.itertuples(index=False)
        if str(row.condition) == "treadmill"
    }
    if not wanted:
        raise ValueError("No same-trial original treadmill IDs supplied")
    results = []
    matched = set()
    with zipfile.ZipFile(archive_path) as package:
        for entry in package.infolist():
            if entry.is_dir():
                continue
            raw_name = entry.filename.replace("\\", "/")
            source_parts = Path(raw_name).parts
            if (".." in source_parts or
                    Path(raw_name).is_absolute()):
                raise ValueError(f"Unsafe original archive member {raw_name}")
            possible_subjects = [
                i for i, part in enumerate(source_parts)
                if part.startswith("AB") and part[2:].isdigit()
            ]
            if len(possible_subjects) != 1:
                continue
            rel_parts = source_parts[possible_subjects[0]:]
            if (len(rel_parts) != 5 or
                    rel_parts[2].lower() != "treadmill" or
                    rel_parts[3].lower() != "gcright" or
                    Path(rel_parts[4]).suffix.lower() != ".mat"):
                continue
            identity = (
                rel_parts[0], rel_parts[1], "treadmill",
                Path(rel_parts[4]).stem,
            )
            if identity not in wanted:
                continue
            if identity in matched:
                raise ValueError(f"Duplicate original gcRight member {identity}")
            matched.add(identity)
            if entry.file_size > max_member_bytes:
                raise ValueError(
                    f"Original gcRight exceeds bounded limit: {raw_name}"
                )
            # Preserve the original case and exact source relative path.
            destination = source_root / Path(*rel_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            status = "already_present"
            if not destination.is_file():
                with package.open(entry) as stream:
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, delete=False
                    ) as f:
                        temporary = Path(f.name)
                        try:
                            shutil.copyfileobj(stream, f)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                if temporary.stat().st_size != entry.file_size:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(
                        f"Incomplete gcRight member extraction: {raw_name}"
                    )
                temporary.replace(destination)
                status = "restored_from_original_archive"
            results.append({
                **dict(zip(keys, identity)),
                "source_relative_path": str(Path(*rel_parts)),
                "status": status,
            })
    return pd.DataFrame(results, columns=keys + [
        "source_relative_path", "status",
    ])


def compare_original_phase_with_force_candidates(
    phase_resets: pd.DataFrame,
    force_candidates: pd.DataFrame,
    *,
    tolerance_s: float = 0.150,
    candidate_channel: str = "Treadmill_R_vy",
) -> pd.DataFrame:
    """Descriptive, one-to-one clock comparison without fitting a time shift.

    Both signals may share upstream processing: matched phase resets cannot
    establish independent validation or justify force-unit assumptions.
    """
    keys = ["subject_id", "session_id", "condition", "trial_id"]
    columns = keys + [
        "n_phase_wrap_candidates", "n_recorded_force_candidates",
        "n_time_near_matches", "median_absolute_difference_ms",
        "unmatched_phase_wraps", "unmatched_force_candidates",
        "independent_reference_verified",
    ]
    if phase_resets.empty:
        return pd.DataFrame(columns=columns)
    if set(keys + ["phase_wrap_header"]) - set(phase_resets.columns):
        raise ValueError("Phase wraps missing source trial identifiers")
    if set(keys + ["candidate_channel", "candidate_onset_header"]) - set(
        force_candidates.columns
    ):
        raise ValueError("Force candidates missing required original fields")
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("Invalid fixed source-clock diagnostic tolerance")
    if candidate_channel not in ("Treadmill_R_vy", "Treadmill_L_vy"):
        raise ValueError("Unsupported source-labelled force channel.")
    forces = force_candidates.loc[
        force_candidates.candidate_channel.eq(candidate_channel)
    ].copy()
    results = []
    for identity, group in phase_resets.groupby(keys, sort=True):
        selected = forces.copy()
        for field, value in zip(keys, identity):
            selected = selected.loc[selected[field].astype(str).eq(str(value))]
        phase = np.sort(pd.to_numeric(
            group.phase_wrap_header, errors="raise"
        ).to_numpy(float))
        force = np.sort(pd.to_numeric(
            selected.candidate_onset_header, errors="raise"
        ).to_numpy(float))
        if not np.isfinite(phase).all() or not np.isfinite(force).all():
            raise ValueError("Non-finite recorded phase or force timestamps")
        i = j = 0
        diffs = []
        while i < len(phase) and j < len(force):
            delta = float(phase[i] - force[j])
            if abs(delta) <= tolerance_s:
                diffs.append(abs(delta) * 1000)
                i += 1
                j += 1
            elif delta < -tolerance_s:
                i += 1
            else:
                j += 1
        results.append({
            **dict(zip(keys, identity)),
            "n_phase_wrap_candidates": len(phase),
            "n_recorded_force_candidates": len(force),
            "n_time_near_matches": len(diffs),
            "median_absolute_difference_ms": (
                float(np.median(diffs)) if diffs else np.nan
            ),
            "unmatched_phase_wraps": len(phase) - len(diffs),
            "unmatched_force_candidates": len(force) - len(diffs),
            "independent_reference_verified": False,
        })
    return pd.DataFrame(results, columns=columns)


def compare_phase_by_recorded_force_side(
    phase_resets: pd.DataFrame,
    force_candidates: pd.DataFrame,
    *,
    tolerance_s: float = 0.150,
) -> pd.DataFrame:
    """Describe both treadmill sides against right gait-phase resets.

    The same fixed clock tolerance is used for both sides. No channel is
    selected or validated by this diagnostic; it is not a contact label.
    """
    sides = ("Treadmill_R_vy", "Treadmill_L_vy")
    tables = []
    for channel in sides:
        table = compare_original_phase_with_force_candidates(
            phase_resets, force_candidates, tolerance_s=tolerance_s,
            candidate_channel=channel,
        )
        table.insert(4, "candidate_channel", channel)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def audit_source_contact_filter_loss(
    pairs: pd.DataFrame, source_root: Path,
    read_sensor_table: Callable, validate_timebase: Callable,
    *, threshold_recorded: float = 20.0,
    min_contact_s: float = 0.08, min_swing_s: float = 0.06,
) -> pd.DataFrame:
    """Count complete source-threshold segments removed by duration checks.

    This analysis does not change the detector, threshold or reference
    eligibility. All forces remain in original units pending calibration.
    """
    if (threshold_recorded <= 0 or min_contact_s <= 0 or min_swing_s <= 0
            or not np.isfinite([threshold_recorded, min_contact_s,
                                min_swing_s]).all()):
        raise ValueError("Invalid descriptive contact-filter settings.")
    keys = ("subject_id", "session_id", "condition", "trial_id")
    if set(keys + ("fp_path", "candidate_pair")) - set(pairs.columns):
        raise ValueError("Original trial registry lacks required fields.")
    rows = []
    for record in pairs.itertuples(index=False):
        if record.condition != "treadmill" or not bool(record.candidate_pair):
            continue
        original = Path(source_root) / record.fp_path
        if not original.is_file():
            raise FileNotFoundError(f"Original source force missing: {original}")
        force = read_sensor_table(original)
        t = np.asarray(validate_timebase(force, "force"), dtype=float)
        for channel in ("Treadmill_R_vy", "Treadmill_L_vy"):
            if channel not in force.columns:
                raise ValueError(f"Missing source force component: {channel}")
            values = pd.to_numeric(force[channel], errors="raise").to_numpy(float)
            if (len(t) != len(values) or len(t) < 4
                    or not np.isfinite(t).all()
                    or not np.isfinite(values).all()
                    or (np.diff(t) <= 0).any()):
                raise ValueError(f"Invalid force data for {channel}: {original}")
            active = values >= threshold_recorded
            changes = np.r_[
                0, np.flatnonzero(np.diff(active.astype(np.int8))) + 1,
                len(t),
            ]
            segments = [
                (int(a), int(b), bool(active[a]))
                for a, b in zip(changes[:-1], changes[1:])
            ]
            raw_complete = short_contact = short_swing = kept = 0
            for i, (a, b, loaded) in enumerate(segments):
                if not loaded or i == 0 or i == len(segments) - 1:
                    continue
                raw_complete += 1
                previous, following = segments[i - 1], segments[i + 1]
                duration = t[b - 1] - t[a]
                pre = t[a] - t[previous[0]]
                post = t[following[1] - 1] - t[following[0]]
                if duration < min_contact_s:
                    short_contact += 1
                elif min(pre, post) < min_swing_s:
                    short_swing += 1
                else:
                    kept += 1
            rows.append({
                **{k: str(getattr(record, k)) for k in keys},
                "candidate_channel": channel,
                "threshold_recorded_units": float(threshold_recorded),
                "n_complete_loaded_segments": raw_complete,
                "n_rejected_short_contact": short_contact,
                "n_rejected_short_swing": short_swing,
                "n_retained_candidates": kept,
                "physical_reference_validated": False,
            })
    return pd.DataFrame(rows)


# ==================== 08. Current Committed Source Evidence ===================

def read_committed_source_evidence(
    repo_root: Path,
    filename: str,
) -> pd.DataFrame:
    """Read the current Git HEAD's published stage-02 CSV without modifying it.

    Use only for a clearly labelled evidence-only quality review when
    original MAT recordings are absent from an ephemeral Colab runtime.
    This is NOT a historical fallback, fresh source extraction, or a
    replacement for original data in force-reference construction.
    The current checked-out commit is the provenance of every read.
    """
    import subprocess
    from io import BytesIO

    allowed = {
        "02_contract_evidence_by_trial.csv",
        "02_force_component_evidence.csv",
        "02_force_channel_screening.csv",
        "02_candidate_contact_trial_audit.csv",
        "02_candidate_contact_intervals.csv",
    }
    if filename not in allowed:
        raise ValueError(f"Not a published source-evidence input: {filename}")
    repo_root = Path(repo_root)
    relative = f"gait_analysis/output/{filename}"
    try:
        fetched = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FileNotFoundError(
            f"Cannot read published {relative} at current local Git HEAD. "
            "Synchronise the repository without discarding local edits, "
            "or restore original MAT recordings before retrying."
        ) from error
    if not fetched.stdout or len(fetched.stdout) > 8_000_000:
        raise ValueError(f"Unexpected published CSV size for {filename}")
    try:
        evidence = pd.read_csv(BytesIO(fetched.stdout))
    except (pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError(
            f"Current committed {filename} is not a readable source CSV"
        ) from error
    if evidence.empty:
        raise ValueError(
            f"Current committed {filename} has no scientific evidence rows"
        )
    for field in ("subject_id", "session_id", "trial_id", "condition"):
        if field in evidence.columns:
            evidence[field] = evidence[field].astype(str)
    return evidence
