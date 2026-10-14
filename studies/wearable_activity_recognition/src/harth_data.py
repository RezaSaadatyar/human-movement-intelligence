# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Read HARTH accelerometer recordings using the common Python CSV reader."""

from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

from core.data_loading import load_data


# ============================ 1. Dataset Information ============================
# HARTH v2.0, official NTNU AI Lab repository; pinned for reproducibility.
HARTH_REVISION = "dad2cfbe89a26f72f19770419469ac037de200df"
HARTH_SOURCE = (
    "https://raw.githubusercontent.com/ntnu-ai-lab/harth-ml-experiments/"
    f"{HARTH_REVISION}/harth/"
)
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
    if subject != "S023":
        raise ValueError("This first example is limited to verified source file S023.csv.")

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
                if total_bytes > 30 * 1024 * 1024:
                    raise ValueError("HARTH file exceeds the 30 MiB sample download limit.")
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
def load_harth_subject(file_path):
    """
    Read and validate one subject's wearable accelerometer recording.

    Args:
        file_path (str | Path): HARTH subject CSV with eight required columns.

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
    if not data["label"].isin(ACTIVITY_NAMES).all():
        unknown = data.loc[~data["label"].isin(ACTIVITY_NAMES), "label"].unique()
        raise ValueError(f"Unexpected HARTH activity codes: {unknown.tolist()}")
    print(f"HARTH subject {path.stem}: {len(data):,} rows, {len(SENSOR_COLUMNS)} sensors")
    return data, SENSOR_COLUMNS.copy()
