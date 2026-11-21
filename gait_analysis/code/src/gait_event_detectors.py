# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Transparent IMU event detectors and a lightweight learned peak scorer."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


# ========================== 01. IMU Signal Filtering ========================
def smooth_gyro(time_s, gyro, cutoff_hz=10.0):
    """Offline zero-phase low-pass filtering (not valid for online inference)."""
    time_s = np.asarray(time_s, float)
    gyro = np.asarray(gyro, float)
    if len(time_s) != len(gyro) or len(time_s) < 24:
        raise ValueError("Gyroscope and timestamp lengths are invalid.")
    if not np.isfinite(gyro).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError("Non-finite gyroscope or non-monotonic timestamps.")
    dt = np.diff(time_s)
    fs = 1.0 / np.median(dt)
    if np.max(dt) > 2.0 * np.median(dt):
        raise ValueError("IMU has sample gaps; process continuous segments.")
    if not 0 < cutoff_hz < 0.45 * fs:
        raise ValueError("Low-pass cutoff exceeds valid Nyquist range.")
    sos = butter(4, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, gyro)


# ======================== 02. Transparent Peak Detector =====================
def peak_events(time_s, gyro, *, polarity=1, prominence=0.5,
                min_interval_s=0.4, cutoff_hz=10.0):
    """Return candidate IC timestamps from an explicitly selected gyro axis.

    Polarity, prominence, axis and interval need Development-only validation.
    This is an operational event proxy, not a native anatomical IC label.
    """
    if polarity not in (-1, 1) or prominence <= 0 or min_interval_s <= 0:
        raise ValueError("Invalid polarity/prominence/refractory settings.")
    t = np.asarray(time_s, float)
    signal = smooth_gyro(t, gyro, cutoff_hz)
    dt = np.median(np.diff(t))
    peaks, properties = find_peaks(
        polarity * signal, prominence=prominence,
        distance=max(1, int(np.ceil(min_interval_s / dt))),
    )
    return t[peaks]


# ========================== 03. Learned Candidate Peaks =====================
def candidate_features(time_s, gyro, polarity=1, cutoff_hz=10.0,
                       min_interval_s=0.25):
    """Return candidate times and local signal features for supervised ranking."""
    t = np.asarray(time_s, float)
    filt = smooth_gyro(t, gyro, cutoff_hz)
    dt = np.median(np.diff(t))
    p, prop = find_peaks(
        polarity * filt, prominence=max(np.std(filt) * 0.15, 1e-7),
        distance=max(1, int(min_interval_s / dt)),
    )
    if not len(p):
        return np.empty(0), np.empty((0, 5))
    radius = max(1, int(round(0.15 / dt)))
    feats = []
    for idx, prominence in zip(p, prop["prominences"]):
        segment = filt[max(0, idx - radius):min(len(filt), idx + radius + 1)]
        feats.append([filt[idx], float(prominence),
                      np.std(segment), np.ptp(segment), np.mean(np.abs(segment))])
    return t[p], np.asarray(feats, float)


def fit_peak_scorer(X, y, seed=42):
    """Fit a small Random Forest on Development-only labelled candidate peaks."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = np.asarray(X, float), np.asarray(y, int)
    if X.ndim != 2 or X.shape[0] != len(y) or np.unique(y).size != 2:
        raise ValueError("Need finite candidate features and both label classes.")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite candidate features.")
    clf = RandomForestClassifier(
        n_estimators=120, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def postprocess_probabilities(times, probabilities, threshold=0.5,
                              min_interval_s=0.4):
    """Apply a frozen Development threshold and refractory period."""
    times = np.asarray(times, float)
    probs = np.asarray(probabilities, float)
    if len(times) != len(probs) or not 0 <= threshold <= 1:
        raise ValueError("Candidate and probability vector mismatch.")
    selected = np.flatnonzero(probs >= threshold)
    # Higher-confidence candidate wins inside a refractory cluster.
    chosen = []
    for ix in selected[np.argsort(-probs[selected], kind="stable")]:
        if all(abs(times[ix] - times[j]) >= min_interval_s for j in chosen):
            chosen.append(int(ix))
    return np.sort(times[chosen])
