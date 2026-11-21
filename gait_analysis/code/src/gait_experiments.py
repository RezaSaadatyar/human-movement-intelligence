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


# ========================== 03. Classical Comparisons ======================
def parameter_grid():
    """Return the same predefined, finite candidate grid on every call."""
    return [dict(polarity=p, prominence=level,
                 min_interval_s=0.45, cutoff_hz=10.0)
            for p in (-1, 1) for level in (0.3, 0.6, 1.0, 1.5)]


def evaluate_trial_collection(trials, configuration, model_id,
                              tolerance_s=0.15):
    """Evaluate prepared IMU/force pairs without pooling dependent events."""
    rows = []
    for meta, (t, gyro, reference) in trials:
        _, result = evaluate_rule_trial(
            t, gyro, reference, tolerance_s=tolerance_s, **configuration
        )
        rows.append({**meta, "model_id": model_id, **result})
    return pd.DataFrame(rows)


def participant_selection_score(results):
    """Penalise missed and extra events as well as localisation error.

    A method predicting no events must not win by an artificially low MAE.
    The score is an a priori Development-only engineering selection objective;
    its constituent metrics remain separately reported.
    """
    if results.empty:
        return float("inf")
    aggregates = []
    for subject_id, group in results.groupby("subject_id", sort=True):
        valid = group.loc[group.n_matched > 0]
        matched = int(group.n_matched.sum())
        aggregates.append({
            "subject_id": subject_id,
            "n_ref": group.n_reference.sum(),
            "n_pred": group.n_predicted.sum(),
            "n_match": matched,
            "mae": (float(np.average(valid.timing_mae_ms,
                      weights=valid.n_matched)) if matched else 1000.0),
        })
    by_person = pd.DataFrame(aggregates)
    sens = by_person.n_match / by_person.n_ref.replace(0, np.nan)
    prec = by_person.n_match / by_person.n_pred.replace(0, np.nan)
    penalty = 150.0 * (2.0 - sens.fillna(0) - prec.fillna(0))
    return float((by_person.mae + penalty).mean())


def choose_rule_config(training_trials, tolerance_s=0.15):
    """Select the rule configuration from provided Development training data."""
    if not training_trials:
        raise ValueError("Cannot tune detector without Development trials.")
    candidates = []
    for config in parameter_grid():
        scored = evaluate_trial_collection(
            training_trials, config, "B1_tuned", tolerance_s
        )
        candidates.append((participant_selection_score(scored), config))
    return min(candidates, key=lambda item: item[0])[1]


FIXED_BASELINE = {
    "polarity": 1, "prominence": 0.5, "min_interval_s": 0.45,
    "cutoff_hz": 10.0,
}


def load_trial_collection(cache_root, eligible_rows):
    """Load precomputed signals with identifying columns from the trial table."""
    records = []
    for row in eligible_rows.itertuples(index=False):
        t, gyro, refs = read_prepared_trial(cache_root, row.trial_key)
        meta = {k: getattr(row, k) for k in
                ("subject_id", "session_id", "condition", "trial_id", "trial_key")}
        records.append((meta, (t, gyro, refs)))
    return records
