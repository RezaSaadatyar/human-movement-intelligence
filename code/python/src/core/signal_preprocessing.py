# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Create reproducible signal windows without crossing annotation or time gaps."""

import numpy as np
import pandas as pd


# ============================= 1. Signal Windowing =============================
def create_signal_windows(
    signals,
    labels,
    window_size,
    step_size,
    timestamps=None,
    sampling_rate=None,
    min_label_purity=1.0,
    subject_id=None,
):
    """
    Convert a continuous multichannel signal into annotated fixed-size windows.

    Args:
        signals (np.ndarray): Continuous numerical array (samples, channels).
        labels (array-like): One categorical activity label per sample.
        window_size (int): Number of samples in each window.
        step_size (int): Number of samples between candidate window starts.
        timestamps (array-like | None): Optional acquisition timestamps.
        sampling_rate (float | None): Expected frequency in Hz, required
            when timestamps are supplied for detecting discontinuities.
        min_label_purity (float): Required fraction of the majority class,
            from 0.5 to 1.0. Use 1.0 to reject activity transitions.
        subject_id (str | None): Source participant identifier for metadata.

    Returns:
        X_windows (np.ndarray): (windows, channels, window_size).
        y_windows (np.ndarray): (windows,) majority-class labels.
        window_info (pd.DataFrame): Source start/end indices (end exclusive),
            subject_id, label and label_purity for retained windows.

    Notes:
        - No filtering, resampling, feature scaling or data splitting is done.
        - Windows crossing duplicate/backwards timestamps or sufficiently
          large timing gaps are discarded.
        - Source labels must already be correctly aligned with signals.
        - Overlapping windows from the same participant must never be
          randomly divided between training and test sets.
    """
    data = np.asarray(signals)
    target = np.asarray(labels)

    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("signals must have shape (samples, channels).")
    if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
        raise ValueError("signals must contain finite numeric samples.")
    if target.ndim != 1 or len(target) != len(data):
        raise ValueError("labels must contain one value per signal sample.")
    if pd.isna(target).any():
        raise ValueError("labels contain missing values.")
    if not isinstance(window_size, (int, np.integer)) or window_size < 2:
        raise ValueError("window_size must be an integer >= 2.")
    if not isinstance(step_size, (int, np.integer)) or step_size < 1:
        raise ValueError("step_size must be an integer >= 1.")
    if not 0.5 <= min_label_purity <= 1.0:
        raise ValueError("min_label_purity must be between 0.5 and 1.0.")

    invalid_interval = np.zeros(len(data), dtype=bool)
    if timestamps is not None:
        if sampling_rate is None or not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("A positive sampling_rate is required with timestamps.")
        if len(timestamps) != len(data):
            raise ValueError("timestamps must align with signal samples.")
        times = pd.to_datetime(pd.Series(timestamps), errors="raise")
        if times.isna().any():
            raise ValueError("timestamps contain missing values.")
        delta_seconds = times.diff().dt.total_seconds().to_numpy()[1:]
        # Allow small timing jitter but reject gaps > 1.5 expected intervals.
        invalid_interval[1:] = (
            ~np.isfinite(delta_seconds)
            | (delta_seconds <= 0)
            | (delta_seconds > (1.5 / sampling_rate))
        )

    # Prefix sum: test whether a window crosses a timing discontinuity in O(1).
    bad_interval_cumsum = np.cumsum(invalid_interval)
    selected_windows = []
    selected_labels = []
    records = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        if bad_interval_cumsum[end - 1] - bad_interval_cumsum[start] > 0:
            continue

        codes, counts = np.unique(target[start:end], return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_label = codes[majority_index]
        label_purity = float(counts[majority_index] / window_size)

        if label_purity < min_label_purity:
            continue

        selected_windows.append(data[start:end].T)
        selected_labels.append(majority_label)
        records.append({
            "subject_id": subject_id,
            "start_index": start,
            "end_index_exclusive": end,
            "label": majority_label.item()
            if isinstance(majority_label, np.generic) else majority_label,
            "label_purity": label_purity,
        })

    if selected_windows:
        X_windows = np.stack(selected_windows, axis=0)
    else:
        X_windows = np.empty((0, data.shape[1], window_size), dtype=data.dtype)
    y_windows = np.asarray(selected_labels, dtype=target.dtype)
    columns = [
        "subject_id", "start_index", "end_index_exclusive",
        "label", "label_purity",
    ]
    window_info = pd.DataFrame.from_records(records, columns=columns)

    print(f"Input signal: {data.shape[0]:,} samples, {data.shape[1]} channels")
    print(f"Candidate windows: {max(0, (len(data) - window_size) // step_size + 1):,}")
    print(f"Retained windows: {len(X_windows):,} | Shape: {X_windows.shape}")
    print(f"Rejected candidates: "
          f"{max(0, (len(data) - window_size) // step_size + 1) - len(X_windows):,}")
    return X_windows, y_windows, window_info
