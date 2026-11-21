# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Portable prepared-trial loading and gait baseline comparison utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gait_event_detectors import peak_events, candidate_features, fit_peak_scorer
from gait_evaluation import event_metrics, match_events


# ========================= 01. Prepared Trial Inputs ========================
def read_prepared_trial(cache_dir: Path, trial_key: str):
    """Read aligned, verified original-time trial vectors saved by Notebook 02."""
    path = Path(cache_dir) / f"{trial_key}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run Notebook 02 for eligible paired trials first."
        )
    with np.load(path, allow_pickle=False) as arr:
        required = {"imu_time_s", "gyro", "reference_ic_s"}
        if not required.issubset(arr.files):
            raise ValueError(f"{path}: required numeric arrays absent.")
        t, g, ic = [np.asarray(arr[k], float) for k in
                    ("imu_time_s", "gyro", "reference_ic_s")]
    if len(t) != len(g) or not np.all(np.diff(t) > 0):
        raise ValueError(f"{trial_key}: invalid IMU signal/time mapping.")
    return t, g, ic


def save_prepared_trial(cache_dir: Path, trial_key: str, imu_time_s,
                        gyro, reference_ic_s):
    """Store only reusable numerical arrays; no unreviewed source header edits."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if "/" in trial_key or "\\" in trial_key or trial_key in ("", ".", ".."):
        raise ValueError("Use a portable, single-component trial key.")
    np.savez_compressed(root / f"{trial_key}.npz",
                        imu_time_s=np.asarray(imu_time_s, float),
                        gyro=np.asarray(gyro, float),
                        reference_ic_s=np.asarray(reference_ic_s, float))


# ============================ 02. Detector Fitting ===========================
def candidate_labels(candidate_times, reference_times, tolerance_s):
    """Assign candidate positives using strict one-to-one reference matching."""
    pairs, _, _ = match_events(reference_times, candidate_times, tolerance_s)
    cand = np.asarray(candidate_times, float)
    return np.isin(cand, pairs.estimated_s.to_numpy(float)).astype(int)


def evaluate_rule_trial(t, gyro, ref, *, prominence, polarity,
                        min_interval_s=0.4, cutoff_hz=10.0,
                        tolerance_s=0.15):
    estimates = peak_events(t, gyro, polarity=polarity,
                            prominence=prominence,
                            min_interval_s=min_interval_s,
                            cutoff_hz=cutoff_hz)
    return estimates, event_metrics(ref, estimates, tolerance_s)


def build_training_candidates(trials, polarity=1, cutoff_hz=10.0,
                              tolerance_s=0.15):
    """Collect candidate features and labels only from supplied Training trials."""
    features, labels = [], []
    for t, gyro, refs in trials:
        times, x = candidate_features(t, gyro, polarity, cutoff_hz)
        if len(x):
            features.append(x)
            labels.append(candidate_labels(times, refs, tolerance_s))
    if not features:
        raise ValueError("No candidate events in training trials.")
    return np.vstack(features), np.concatenate(labels)


def fit_classical_candidate_model(training_trials, seed=42):
    X, y = build_training_candidates(training_trials)
    return fit_peak_scorer(X, y, seed=seed)
