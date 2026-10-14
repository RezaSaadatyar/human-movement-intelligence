# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Read HARTH accelerometer recordings using the common Python CSV reader."""

from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
import warnings

from core.data_loading import load_data


# ============================ 1. Dataset Information ============================
# HARTH v2.0, official NTNU AI Lab repository; pinned for reproducibility.
HARTH_REVISION = "dad2cfbe89a26f72f19770419469ac037de200df"
HARTH_SOURCE = (
    "https://raw.githubusercontent.com/ntnu-ai-lab/harth-ml-experiments/"
    f"{HARTH_REVISION}/harth/"
)
HARTH_SUBJECTS = frozenset("S006 S008 S009 S010 S012 S013 S014 S015 S016 S017 S018 S019 S020 S021 S022 S023 S024 S025 S026 S027 S028 S029 S030 S031 S032 S033 S034 S035 S036 S037 S038".split())
SENSOR_COLUMNS = [
    "back_x", "back_y", "back_z",
    "thigh_x", "thigh_y", "thigh_z",
]
REQUIRED_COLUMNS = ["timestamp", *SENSOR_COLUMNS, "label"]
ACTIVITY_NAMES = {
    1: "Walking", 2: "Running", 3: "Shuffling",
    4: "Stairs ascending", 5: "Stairs descending",
    6: "Standing", 7: "Sitting", 8: "Lying",
    13: "Cycling seated", 14: "Cycling standing",
    130: "Cycling seated inactive", 140: "Cycling standing inactive",
}


# ============================== 2. Data Acquisition ==============================
def ensure_harth_sample(data_dir, subject="S023"):
    """
    Obtain one public participant CSV from the official HARTH v2.0 repository.

    Args:
        data_dir (str | Path): Ignored local raw-data directory.
        subject (str): Participant filename stem; first demonstration uses S023.

    Returns:
        Path: The downloaded or cached CSV file.

    Notes:
        The first study intentionally downloads one file (~12 MB), not the
        entire HARTH archive. The source revision is pinned above.
    """
    if subject not in HARTH_SUBJECTS:
        raise ValueError(f"Unknown HARTH v2.0 subject: {subject}")

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{subject}.csv"
    if csv_path.exists():
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            raise ValueError(f"Invalid cached source: {csv_path}")
        print(f"Using cached HARTH data: {csv_path}")
        return csv_path

    partial_path = destination / f"{subject}.csv.part"
    if partial_path.exists():
        raise FileExistsError(f"Partial file already exists: {partial_path}")
    url = HARTH_SOURCE + f"{subject}.csv"
    print(f"Downloading HARTH v2.0 source file: {subject}.csv")
    total_bytes = 0
    try:
        request = Request(url, headers={"User-Agent": "human-movement-intelligence/0.1"})
        with urlopen(request, timeout=90) as response, partial_path.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > 70 * 1024 * 1024:
                    raise ValueError("HARTH file exceeds the 70 MiB download limit.")
                stream.write(chunk)
        if total_bytes == 0:
            raise ValueError("Downloaded file is empty.")
        partial_path.replace(csv_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    print(f"Downloaded {total_bytes:,} bytes to {csv_path}")
    return csv_path


# ================================ 3. Data Loading ================================
def load_harth_subject(file_path, allow_unmapped_labels=False):
    """
    Read and validate one subject's wearable accelerometer recording.

    Args:
        file_path (str | Path): HARTH subject CSV with eight required columns.
        allow_unmapped_labels (bool): Explicit inspection-only option; unknown
            labels are retained with a warning. Strict by default.

    Returns:
        data (pd.DataFrame): Timestamp, six acceleration channels and
            annotated activity labels. Sampling rate is reported as 50 Hz.
        channel_names (list[str]): The six acceleration channels.

    Notes:
        This is a continuous recording, not a set of independent windows.
        Subject identity is taken from the source filename (e.g. S023).
        Do not split individual timestamps at random when evaluating
        generalisation to unseen people.
    """
    path = Path(file_path)
    data = load_data(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"HARTH source is missing columns: {missing}")
    if data.empty:
        raise ValueError("HARTH source CSV is empty.")

    data = data[REQUIRED_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    for name in SENSOR_COLUMNS:
        data[name] = pd.to_numeric(data[name], errors="raise")
    data["label"] = pd.to_numeric(data["label"], errors="raise")
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing timestamps, sensor values or activity labels.")
    # Check published HARTH labels without altering undocumented source codes.
    unmapped_mask = ~data["label"].isin(ACTIVITY_NAMES)
    if unmapped_mask.any():
        unknown = sorted(data.loc[unmapped_mask, "label"].unique().tolist())
        count = int(unmapped_mask.sum())
        message = (
            f"Unexpected HARTH activity codes: {unknown} "
            f"({count:,} rows in {path.name}). "
            "The HARTH v2.0 activity table does not define these codes."
        )
        if not allow_unmapped_labels:
            raise ValueError(
                message + " Use allow_unmapped_labels=True for inspection only."
            )
        # Never infer a label's meaning or silently include it for training.
        warnings.warn(
            message + " Source codes retained without reinterpretation; "
            "resolve them before supervised training.",
            UserWarning,
            stacklevel=2,
        )
    print(f"HARTH subject {path.stem}: {len(data):,} rows, {len(SENSOR_COLUMNS)} sensors")
    return data, SENSOR_COLUMNS.copy()


# ========================= 4. Multi-Subject Cohort Preparation ======================
def prepare_harth_feature_cohort(
    subjects,
    data_dir,
    task_classes=tuple(range(1, 9)),
    window_size=100,
    step_size=50,
    sampling_rate=50,
):
    """
    Assemble a labelled HARTH feature cohort from independent subject CSVs.

    Args:
        subjects (sequence[str]): Distinct, official HARTH v2.0 subject IDs.
        data_dir (str | Path): Local source cache; files downloaded if absent.
        task_classes (sequence[int]): Predefined, documented activity codes.
            Labels outside the task are excluded ONLY AFTER windowing.
        window_size (int): Source samples per signal window.
        step_size (int): Source samples between candidate window starts.
        sampling_rate (float): Acquisition rate (Hz) for time-gap checks.

    Returns:
        features (pd.DataFrame): Aligned (eligible_windows, signal_features).
        metadata (pd.DataFrame): Original subject, window indices and labels.
        overview (pd.DataFrame): Per-person raw, retained and excluded counts.
        excluded (pd.DataFrame): Excluded-window counts by person/source code.

    Notes:
        Generic windowing and feature engineering are reused unchanged.
        Undocumented HARTH codes are kept for an explicit source audit;
        they are never silently mapped to a documented activity class.
        The returned arrays must still be partitioned by SUBJECT before
        fitting any scaler or machine-learning model.
    """
    # 1. Validate the scientific cohort and target labels before downloading.
    from core.feature_engineering import extract_signal_features
    from core.signal_preprocessing import create_signal_windows

    subject_list = list(subjects)
    if not subject_list or len(set(subject_list)) != len(subject_list):
        raise ValueError("subjects must be non-empty and have no duplicate IDs.")
    if any(subject not in HARTH_SUBJECTS for subject in subject_list):
        raise ValueError("The cohort includes unknown HARTH v2.0 subject IDs.")
    class_list = list(task_classes)
    if not class_list or len(set(class_list)) != len(class_list):
        raise ValueError("task_classes must contain distinct documented labels.")
    if not set(class_list).issubset(ACTIVITY_NAMES):
        raise ValueError("Every selected task class must be officially documented.")

    # 2. Build windows and deterministic features independently for each person.
    feature_parts, metadata_parts, overview_rows, excluded_rows = [], [], [], []
    for subject in subject_list:
        source_path = ensure_harth_sample(data_dir, subject=subject)
        data, channels = load_harth_subject(
            source_path, allow_unmapped_labels=True
        )
        windows, _, info = create_signal_windows(
            signals=data[channels].to_numpy(),
            labels=data["label"].to_numpy(),
            window_size=window_size,
            step_size=step_size,
            timestamps=data["timestamp"],
            sampling_rate=sampling_rate,
            min_label_purity=1.0,
            subject_id=subject,
        )
        if not len(windows):
            raise ValueError(f"No valid source windows for {subject}.")

        features_for_subject = extract_signal_features(
            windows, channel_names=channels
        )

        # 3. Audit and exclude out-of-task labels AFTER windowing.
        eligible = info["label"].isin(class_list).to_numpy()
        excluded_counts = info.loc[~eligible, "label"].value_counts().sort_index()
        for code, count in excluded_counts.items():
            excluded_rows.append({
                "subject_id": subject,
                "source_code": int(code),
                "source_activity": ACTIVITY_NAMES.get(
                    int(code), "UNDOCUMENTED source code"
                ),
                "excluded_windows": int(count),
            })
        unknown_rows = int((~data["label"].isin(ACTIVITY_NAMES)).sum())
        overview_rows.append({
            "subject_id": subject,
            "raw_samples": len(data),
            "retained_windows": len(info),
            "eligible_windows": int(eligible.sum()),
            "excluded_windows": int((~eligible).sum()),
            "undocumented_raw_rows": unknown_rows,
            "task_classes_present": int(info.loc[eligible, "label"].nunique()),
        })
        if not eligible.any():
            raise ValueError(f"No windows with the predefined task labels: {subject}.")
        feature_parts.append(
            features_for_subject.loc[eligible].reset_index(drop=True)
        )
        metadata_parts.append(info.loc[eligible].reset_index(drop=True))

    # 4. Preserve identical row order between the shared feature and ID tables.
    features = pd.concat(feature_parts, ignore_index=True)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    overview = pd.DataFrame(overview_rows)
    excluded = pd.DataFrame(
        excluded_rows,
        columns=["subject_id", "source_code", "source_activity", "excluded_windows"]
    )
    if len(features) != len(metadata):
        raise RuntimeError("Feature and metadata rows have become misaligned.")
    if metadata["subject_id"].nunique() != len(subject_list):
        raise RuntimeError("A participant is missing from the eligible cohort.")

    print(
        f"Eligible dataset: {len(subject_list)} subjects, "
        f"{len(features):,} windows, {features.shape[1]} features"
    )
    return features, metadata, overview, excluded


# ====================== 5. Raw Windows for Temporal Models ======================
def prepare_harth_window_cohort(
    subjects,
    data_dir,
    task_classes=tuple(range(1, 9)),
    max_windows_per_subject=500,
    window_size=100,
    step_size=50,
    sampling_rate=50,
):
    """
    Prepare genuine IMU sequences for small, CPU-feasible temporal experiments.

    Args:
        subjects (sequence[str]): Distinct HARTH v2.0 development IDs.
        data_dir (str | Path): Local directory for official source CSV files.
        task_classes (sequence[int]): Predefined documented target classes.
        max_windows_per_subject (int): Upper limit per participant for this
            bounded demonstration. Selection is uniformly spaced in source
            window order and does NOT use activity labels.
        window_size (int): Signal samples per window (default 2 s at 50 Hz).
        step_size (int): Signal samples per candidate-window stride.
        sampling_rate (float): Sampling rate for discontinuity checks.

    Returns:
        X (np.ndarray): float32, (windows, six_channels, time).
        y (np.ndarray): Source class labels, one per retained window.
        metadata (pd.DataFrame): Aligned subject/source-index/label details.
        overview (pd.DataFrame): Per-subject window retention and exclusions.

    Notes:
        The full source is windowed BEFORE out-of-task labels are removed.
        The bounded temporal subset is for model engineering, not an
        unbiased full-HARTH benchmark. No fitting or normalisation occurs.
    """
    from core.signal_preprocessing import create_signal_windows

    # 1. Validate the cohort and fixed classification problem.
    subject_list = list(subjects)
    classes = list(task_classes)
    if not subject_list or len(set(subject_list)) != len(subject_list):
        raise ValueError("subjects must be distinct, non-empty IDs.")
    if not set(subject_list).issubset(HARTH_SUBJECTS):
        raise ValueError("Unknown HARTH v2.0 participant ID.")
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(ACTIVITY_NAMES):
        raise ValueError("task_classes must be distinct documented activity codes.")
    if isinstance(max_windows_per_subject, bool) or not isinstance(
        max_windows_per_subject, (int, np.integer)
    ) or max_windows_per_subject < 1:
        raise ValueError("max_windows_per_subject must be a positive integer.")

    # 2. Apply the established time-gap and annotation-purity protocol.
    inputs, metadata_parts, overview_rows = [], [], []
    for subject in subject_list:
        source_path = ensure_harth_sample(data_dir, subject=subject)
        data, channels = load_harth_subject(
            source_path, allow_unmapped_labels=True
        )
        X, _, info = create_signal_windows(
            signals=data[channels].to_numpy(),
            labels=data["label"].to_numpy(),
            window_size=window_size, step_size=step_size,
            timestamps=data["timestamp"], sampling_rate=sampling_rate,
            min_label_purity=1.0, subject_id=subject,
        )

        # 3. Remove out-of-task windows AFTER temporal segmentation.
        eligible = info["label"].isin(classes).to_numpy()
        retained_indices = np.flatnonzero(eligible)
        if len(retained_indices) == 0:
            raise ValueError(f"No eligible activity windows for {subject}.")

        # 4. Select evenly along the recording, never by class labels.
        count = min(len(retained_indices), int(max_windows_per_subject))
        positions = np.linspace(
            0, len(retained_indices) - 1, num=count, dtype=np.int64
        )
        chosen = retained_indices[positions]
        selected = X[chosen].astype(np.float32, copy=False)
        chosen_info = info.iloc[chosen].reset_index(drop=True)
        inputs.append(selected)
        metadata_parts.append(chosen_info)
        overview_rows.append({
            "subject_id": subject,
            "raw_samples": len(data),
            "candidate_retained_windows": len(info),
            "eligible_windows": len(retained_indices),
            "sampled_windows": len(selected),
            "excluded_out_of_task_windows": int((~eligible).sum()),
            "observed_sampled_classes": int(chosen_info["label"].nunique()),
        })

    # 5. Preserve the exact row order for windows, labels, and subject IDs.
    windows = np.concatenate(inputs, axis=0)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    labels = metadata["label"].to_numpy()
    overview = pd.DataFrame(overview_rows)
    if len(windows) != len(labels) or metadata["subject_id"].nunique() != len(subject_list):
        raise RuntimeError("Source-window alignment or group coverage failed.")
    if not np.isfinite(windows).all():
        raise ValueError("The selected signal windows contain nonfinite values.")
    print(f"Temporal cohort: {len(subject_list)} subjects, X={windows.shape}")
    return windows, labels, metadata, overview
