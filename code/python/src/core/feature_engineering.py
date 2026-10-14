# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Reusable statistical and temporal features for multichannel signal windows."""

import numpy as np
import pandas as pd


# ============================== 1. Feature Definitions ==============================
AVAILABLE_FEATURES = (
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "waveform_length",
)


# ============================ 2. Feature Extraction ===============================
def extract_signal_features(
    X_windows,
    channel_names=None,
    methods=AVAILABLE_FEATURES,
):
    """
    Extract interpretable features independently from each signal window.

    Args:
        X_windows (np.ndarray): Array of shape (windows, channels, time).
        channel_names (list[str] | None): Names in the original channel order.
            If None, labels Channel_1, Channel_2, ... are used.
        methods (list[str] | tuple[str, ...]): Selected feature names.
            Supported: mean, std, rms, peak_to_peak, waveform_length.

    Returns:
        pd.DataFrame: One row per window; columns named channel__feature.
            With six channels and five methods, the shape is (windows, 30).

    Notes:
        - std uses population standard deviation (ddof=0).
        - waveform_length is the mean absolute difference between adjacent
          samples (normalised by window_size - 1).
        - No normalisation is fitted and no subject information or labels
          are used by this function.
        - Subject-independent splitting is required before fitting any
          data-dependent transforms or training models.
    """
    # 1. Validate the input tensor and reject non-numeric/missing data.
    data = np.asarray(X_windows)
    if data.ndim != 3:
        raise ValueError("X_windows must have shape (windows, channels, time).")
    n_windows, n_channels, n_points = data.shape
    if n_windows == 0 or n_channels == 0 or n_points < 2:
        raise ValueError("Input needs at least one window, one channel and two time points.")
    if data.dtype.kind not in "biuf":
        raise ValueError("X_windows must contain real numeric values.")
    if not np.isfinite(data).all():
        raise ValueError("X_windows contains NaN or infinite values.")

    # 2. Validate the channel labels and requested feature methods.
    if channel_names is None:
        names = [f"Channel_{i + 1}" for i in range(n_channels)]
    else:
        names = list(channel_names)
        if (
            len(names) != n_channels
            or not all(isinstance(name, str) and name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("channel_names must be distinct, non-empty strings per channel.")

    if isinstance(methods, str):
        raise ValueError("methods must be a sequence, not one string.")
    selected = tuple(methods)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("methods must contain unique, non-empty feature names.")
    unsupported = set(selected) - set(AVAILABLE_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported feature methods: {sorted(unsupported)}")

    # 3. Calculate each feature along the time axis only.
    #    Arrays below have shape (windows, channels).
    numeric = data.astype(np.float64, copy=False)
    calculations = {
        "mean": lambda x: x.mean(axis=2),
        "std": lambda x: x.std(axis=2, ddof=0),
        "rms": lambda x: np.sqrt(np.mean(x ** 2, axis=2)),
        "peak_to_peak": lambda x: np.ptp(x, axis=2),
        "waveform_length": lambda x: np.mean(np.abs(np.diff(x, axis=2)), axis=2),
    }
    computed = {name: calculations[name](numeric) for name in selected}

    # 4. Assemble an interpretable feature matrix for pandas and classical ML.
    columns = {
        f"{channel}__{method}": computed[method][:, channel_index]
        for channel_index, channel in enumerate(names)
        for method in selected
    }
    features = pd.DataFrame(columns)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature computation resulted in non-finite values.")

    print(f"Input windows: {data.shape}")
    print(f"Feature matrix: {features.shape} | Methods: {', '.join(selected)}")
    return features
