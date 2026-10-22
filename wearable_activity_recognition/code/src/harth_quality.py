# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Source-grounded, training-free quality audit for HARTH v2.0 recordings.

Inspect original source labels and timestamps before windowing, then audit
pure-label and time-continuous candidate windows. Unmapped labels are never
interpreted as undocumented activities or silently reassigned.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.signal_preprocessing import create_signal_windows
from harth_data import (
    ACTIVITY_NAMES,
    HARTH_SUBJECTS,
    SENSOR_COLUMNS,
    ensure_harth_sample,
    load_harth_subject,
)


# ======================= 1. Timestamp and Window Boundaries =====================
def window_rejection_reasons(labels, timestamps, window_size=100,
                             step_size=50, sampling_rate=50):
    """
    Enumerate mutually exclusive reasons for rejecting candidate windows.

    Args:
        labels (array-like): One original annotation code per source sample.
        timestamps (array-like): Acquisition times in original row order.
        window_size (int): Signal samples in each candidate window.
        step_size (int): Samples between candidate window starts.
        sampling_rate (float): Expected Hz for the same source recording.

    Returns:
        dict: Candidate, retained, annotation-transition-only,
            timestamp-gap-only and joint-rejection window counts, plus
            total non-positive timestamps and long intersample intervals.

    Notes:
        Matches the strict-purity and 1.5/sampling_rate gap logic of
        create_signal_windows(). A candidate touching BOTH an annotation
        transition and a timestamp anomaly belongs to the joint category;
        it must never be double-counted.
    """
    # 1. Check alignment and unmodified source annotation values.
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or pd.isna(y).any():
        raise ValueError("Source labels must be a non-empty 1D array.")
    if len(timestamps) != len(y):
        raise ValueError("Labels and timestamps must have identical row counts.")
    for name, value, minimum in [
        ("window_size", window_size, 2),
        ("step_size", step_size, 1),
    ]:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if (not np.isscalar(sampling_rate) or
            not np.isfinite(sampling_rate) or sampling_rate <= 0):
        raise ValueError("sampling_rate must be positive.")

    # 2. Define each timestamp/annotation boundary at its later row.
    times = pd.to_datetime(pd.Series(timestamps), errors="raise")
    if times.isna().any():
        raise ValueError("Missing source timestamps.")
    deltas = times.diff().dt.total_seconds().to_numpy()[1:]
    nonpositive = ~np.isfinite(deltas) | (deltas <= 0)
    long_gap = np.isfinite(deltas) & (deltas > 1.5 / sampling_rate)
    bad_intervals = np.zeros(len(y), dtype=np.int64)
    bad_intervals[1:] = nonpositive | long_gap
    transitions = np.zeros(len(y), dtype=np.int64)
    transitions[1:] = y[1:] != y[:-1]

    # 3. Test each candidate using prefix sums, not a slow inner Python loop.
    starts = np.arange(0, max(0, len(y)-window_size+1),
                       step_size, dtype=np.int64)
    end = starts + window_size
    gap_prefix = np.cumsum(bad_intervals)
    label_prefix = np.cumsum(transitions)
    crosses_gap = (gap_prefix[end-1] - gap_prefix[starts]) > 0
    crosses_label = (label_prefix[end-1] - label_prefix[starts]) > 0

    # 4. Keep rejection reasons mutually exclusive and additive.
    result = {
        "candidate_windows": int(len(starts)),
        "retained_pure_time_safe": int((~crosses_gap & ~crosses_label).sum()),
        "rejected_label_only": int((crosses_label & ~crosses_gap).sum()),
        "rejected_time_only": int((crosses_gap & ~crosses_label).sum()),
        "rejected_both": int((crosses_gap & crosses_label).sum()),
        "nonpositive_intervals": int(nonpositive.sum()),
        "long_intervals": int(long_gap.sum()),
        "median_positive_interval_seconds": (
            float(np.median(deltas[np.isfinite(deltas) & (deltas > 0)]))
            if (np.isfinite(deltas) & (deltas > 0)).any() else np.nan
        ),
    }
    count_sum = sum(result[k] for k in (
        "retained_pure_time_safe", "rejected_label_only",
        "rejected_time_only", "rejected_both"
    ))
    if count_sum != result["candidate_windows"]:
        raise RuntimeError("Candidate-window reasons are not exhaustive.")
    return result


# ======================= 2. Annotation Bout Characterisation ====================
def characterise_annotation_bouts(labels, timestamps, subject_id,
                                  sampling_rate=50, window_size=100):
    """
    Characterise uninterrupted activity-label bouts in one source recording.

    Args:
        labels (array-like): Unmapped original activity codes, one per sample.
        timestamps (array-like): Source timestamp per recorded sample.
        subject_id (str): Source participant identifier.
        sampling_rate (float): Nominal sensor acquisition frequency.
        window_size (int): Minimum samples for a complete candidate window.

    Returns:
        pd.DataFrame: Source label, start/end-exclusive sample indices,
            bout duration (nominal seconds), sample count, whether it is
            shorter than one analysis window, and boundary-gap flag.

    Notes:
        A bout is split at annotation changes OR discontinuous source
        timestamps. It is an *annotation segment*, not a verified real-world
        behavioural event. Duration uses sample count / nominal Hz, and
        is only approximate where timestamp jitter is present.
    """
    # 1. Validate source record and identify physically discontinuous intervals.
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or len(y) != len(timestamps):
        raise ValueError("Aligned non-empty source labels/timestamps required.")
    if pd.isna(y).any() or not subject_id:
        raise ValueError("Valid original annotations and subject ID required.")
    if (not np.isfinite(sampling_rate) or sampling_rate <= 0
            or not isinstance(window_size, (int, np.integer)) or window_size < 2):
        raise ValueError("Positive sampling rate and valid window size required.")
    times = pd.to_datetime(pd.Series(timestamps), errors="raise")
    if times.isna().any():
        raise ValueError("Missing source timestamps.")
    delta = times.diff().dt.total_seconds().to_numpy()[1:]
    discontinuity = (~np.isfinite(delta) | (delta <= 0) |
                     (delta > 1.5 / sampling_rate))
    new_bout = np.zeros(len(y), dtype=bool)
    new_bout[0] = True
    new_bout[1:] = (y[1:] != y[:-1]) | discontinuity

    # 2. Measure each label segment without bridging recording time gaps.
    starts = np.flatnonzero(new_bout)
    ends = np.r_[starts[1:], len(y)]
    lengths = ends - starts
    bouts = pd.DataFrame({
        "subject_id": subject_id,
        "source_code": y[starts].astype(np.int64),
        "start_index": starts,
        "end_index_exclusive": ends,
        "samples": lengths,
        "nominal_duration_seconds": lengths / sampling_rate,
        "shorter_than_window": lengths < window_size,
    })
    return bouts


# ========================== 3. Source-Channel Diagnostics ========================
def inspect_raw_sensor_channels(data, subject_id, channel_names=SENSOR_COLUMNS):
    """
    Report non-diagnostic signal distributions for each accelerometer axis.

    Args:
        data (pd.DataFrame): One decoded original participant CSV.
        subject_id (str): Identifies source recording.
        channel_names (sequence[str]): Original six accelerometer axis names.

    Returns:
        pd.DataFrame: Source min/max, mean, SD, absolute p99, percentage
            of identical adjacent values and non-finite sample count.

    Notes:
        Outliers, repeated values and wide signal ranges are diagnostic flags,
        not proof of faulty instrumentation. Original sensor physical units
        are not inferred from numeric magnitudes.
    """
    # 1. Keep all original sensor samples, without standardising or fitting.
    names = list(channel_names)
    if not names or len(set(names)) != len(names) or not set(names).issubset(data):
        raise ValueError("Expected distinct accelerometer source channels.")
    rows = []
    for name in names:
        values = pd.to_numeric(data[name], errors="raise").to_numpy(dtype=float)
        if not len(values):
            raise ValueError("Empty source accelerometer channel.")
        rows.append({
            "subject_id": subject_id, "channel": name,
            "raw_samples": len(values),
            "nonfinite_samples": int((~np.isfinite(values)).sum()),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "absolute_p99": float(np.quantile(np.abs(values), .99)),
            "repeated_adjacent_pct": (
                float(np.mean(values[1:] == values[:-1]) * 100)
                if len(values) > 1 else 0.
            ),
        })
    return pd.DataFrame(rows)


# ========================= 4. Full Development Source Audit ======================
def audit_harth_development_sources(
    subjects, data_dir, task_classes=tuple(range(1, 9)),
    window_size=100, step_size=50, sampling_rate=50,
    reserved_subjects=("S023", "S032"),
):
    """
    Audit original HARTH CSVs individually, without model fitting.

    Args:
        subjects (sequence[str]): Exact distinct authorised HARTH source IDs.
        data_dir (str | Path): Local cache for pinned official source files.
        task_classes (sequence[int]): Fixed documented supervised task codes.
        window_size (int): Window length in samples.
        step_size (int): Window step in samples.
        sampling_rate (float): Nominal acquisition rate (Hz).
        reserved_subjects (sequence[str]): IDs excluded from development.

    Returns:
        overview (pd.DataFrame): Per-person signal/label/window integrity.
        raw_codes (pd.DataFrame): Original sample counts by person/source code.
        window_codes (pd.DataFrame): All accepted pure/time-safe window codes.
        eligible_metadata (pd.DataFrame): Aligned accepted 8-class windows,
            preserving original participant and window start/end index.
        bouts (pd.DataFrame): Consecutive source annotation bouts.
        channels (pd.DataFrame): Source accelerometer descriptive statistics.
        examples (dict): Up to one genuine 2-second six-axis window per
            documented task code, with its source participant and start index.

    Notes:
        - Original source codes remain unchanged (unknowns are not relabelled).
        - Source windows are formed BEFORE excluding out-of-task annotations.
        - Recordings are segmented one person at a time; no cross-person
          windows or random splitting of overlapping windows can occur.
        - No Train/Validation/Test model data are fitted or scored.
        - Protected Test people are NEVER accessed.
    """
    # 1. Refuse unknown, repeated or originally protected source identities.
    ids = list(subjects)
    protected = {str(s).casefold() for s in reserved_subjects}
    codes = tuple(task_classes)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Development subjects must be distinct and non-empty.")
    if not set(ids).issubset(HARTH_SUBJECTS):
        raise ValueError("Unknown HARTH v2.0 source subject ID.")
    if protected & {str(s).casefold() for s in ids}:
        raise ValueError("Reserved Test subject cannot enter the source audit.")
    if not codes or len(codes) != len(set(codes)) or not set(codes).issubset(ACTIVITY_NAMES):
        raise ValueError("Task labels must be distinct officially documented codes.")

    # 2. Audit each public source CSV and form clean windows separately.
    overview_rows, raw_parts, window_parts = [], [], []
    metadata_parts, bouts_parts, channel_parts = [], [], []
    examples = {}
    for subject in ids:
        path = ensure_harth_sample(data_dir, subject=subject)
        source, names = load_harth_subject(path, allow_unmapped_labels=True)
        labels = source["label"].to_numpy()
        timestamps = source["timestamp"]
        outcomes = window_rejection_reasons(
            labels, timestamps, window_size, step_size, sampling_rate
        )
        X, _, info = create_signal_windows(
            signals=source[names].to_numpy(),
            labels=labels,
            timestamps=timestamps,
            window_size=window_size, step_size=step_size,
            sampling_rate=sampling_rate, min_label_purity=1.0,
            subject_id=subject,
        )
        if len(X) != outcomes["retained_pure_time_safe"]:
            raise RuntimeError(f"Window reason counts mismatch for {subject}.")
        eligible = info["label"].isin(codes).to_numpy()
        if not eligible.any():
            raise ValueError(f"No eligible documented source windows: {subject}")

        # 3. Retain source metadata and source-code audits, never guessed names.
        original_counts = pd.Series(labels).value_counts().sort_index()
        for code, count in original_counts.items():
            raw_parts.append({
                "subject_id": subject, "source_code": int(code),
                "source_activity": ACTIVITY_NAMES.get(
                    int(code), "UNDOCUMENTED source code"
                ),
                "source_samples": int(count),
                "documented_source_code": int(code) in ACTIVITY_NAMES,
                "in_eight_class_task": int(code) in codes,
            })
        for code, count in info["label"].value_counts().sort_index().items():
            window_parts.append({
                "subject_id": subject, "source_code": int(code),
                "source_activity": ACTIVITY_NAMES.get(
                    int(code), "UNDOCUMENTED source code"
                ),
                "accepted_pure_windows": int(count),
                "in_eight_class_task": int(code) in codes,
            })
        overview_rows.append({
            "subject_id": subject,
            "raw_samples": len(source),
            "candidate_windows": outcomes["candidate_windows"],
            "retained_pure_time_safe": len(info),
            "eligible_eight_class_windows": int(eligible.sum()),
            "excluded_after_windowing": int((~eligible).sum()),
            "undocumented_raw_samples": int(
                (~pd.Series(labels).isin(ACTIVITY_NAMES)).sum()
            ),
            **{key: outcomes[key] for key in [
                "rejected_label_only", "rejected_time_only",
                "rejected_both", "nonpositive_intervals",
                "long_intervals", "median_positive_interval_seconds"
            ]},
        })
        metadata_parts.append(info.loc[eligible].reset_index(drop=True))
        bouts_parts.append(characterise_annotation_bouts(
            labels, timestamps, subject_id=subject,
            sampling_rate=sampling_rate, window_size=window_size,
        ))
        channel_parts.append(inspect_raw_sensor_channels(source, subject, names))

        # 4. Keep just one representative actual source window per task class.
        for code in codes:
            if code in examples:
                continue
            locations = np.flatnonzero(info["label"].to_numpy() == code)
            if len(locations):
                row = int(locations[0])
                examples[code] = {
                    "subject_id": subject,
                    "start_index": int(info.iloc[row]["start_index"]),
                    "channel_names": list(names),
                    "signals": np.asarray(X[row], dtype=np.float32).copy(),
                }
        print(
            f"Audited {subject}: {len(source):,} source rows, "
            f"{int(eligible.sum()):,} task windows, "
            f"{int((~eligible).sum()):,} excluded accepted windows"
        )

    # 5. Cross-check joined metadata and independently counted source rows.
    overview = pd.DataFrame(overview_rows)
    raw_codes = pd.DataFrame(raw_parts)
    window_codes = pd.DataFrame(window_parts)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    bouts = pd.concat(bouts_parts, ignore_index=True)
    channels = pd.concat(channel_parts, ignore_index=True)
    if int(overview["raw_samples"].sum()) != int(raw_codes["source_samples"].sum()):
        raise RuntimeError("Per-subject raw label totals are inconsistent.")
    if len(metadata) != int(overview["eligible_eight_class_windows"].sum()):
        raise RuntimeError("Eligible window/source metadata misalignment.")
    if int(window_codes["accepted_pure_windows"].sum()) != int(
        overview["retained_pure_time_safe"].sum()
    ):
        raise RuntimeError("Accepted source-code window totals disagree.")
    if metadata["subject_id"].nunique() != len(ids):
        raise RuntimeError("Missing participant from development source audit.")
    return overview, raw_codes, window_codes, metadata, bouts, channels, examples
