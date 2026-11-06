# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Trial-wise Essex IMU regression baselines and participant-aware evaluation.

Every input row is one original movement trial. This module never splits
individual timepoints, reads held-out Test targets or saves fitted estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================== 01. Original-Trial Feature Contract ====================
STATISTICS = ("mean", "std", "minimum", "maximum", "rms")
TARGETS = ("e1", "e3")
MODELS = ("Training Mean", "Ridge", "Random Forest")


def trial_features(x_full, channel_names):
    """Extract five fixed features per original channel, with no target data."""
    x = np.asarray(x_full, dtype=np.float64)
    channels = tuple(channel_names)
    if x.ndim != 3 or x.shape[1:] != (101, 42) or len(channels) != 42:
        raise ValueError("Expected original-trial IMU array (N, 101, 42).")
    if not len(x) or not np.isfinite(x).all() or len(set(channels)) != 42:
        raise ValueError("Invalid, nonfinite or duplicate original channel data.")
    blocks = (
        x.mean(axis=1), x.std(axis=1), x.min(axis=1),
        x.max(axis=1), np.sqrt(np.mean(np.square(x), axis=1)),
    )
    values = np.concatenate(blocks, axis=1)
    names = [f"{stat}_{channel}" for stat in STATISTICS for channel in channels]
    if values.shape != (len(x), 210) or not np.isfinite(values).all():
        raise ValueError("Feature extraction did not produce 210 finite columns.")
    return values, names


def flatten_targets(target):
    """Keep one learning row per trial; flatten eleven original FE coordinates."""
    array = np.asarray(target, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (11, 2):
        raise ValueError("Expected original FE data with shape (N, 11, 2).")
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Source FE data must be finite and non-empty.")
    return array.reshape(len(array), 22)


# ======================== 02. Fixed Regression Models ========================
def make_regressor(name, seed=2026):
    """Declare models before fitting; tuning uses Training folds only."""
    if name == "Training Mean":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, solver="svd")),
        ])
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=1.0,
            random_state=int(seed), n_jobs=-1,
        )
    raise ValueError(f"Unknown pre-specified regressor: {name}")


# ====================== 03. Participant-Disjoint Folds =======================
def training_group_folds(groups, n_splits=5):
    """Return fixed GroupKFold indices and a per-trial audit table."""
    ids = np.asarray(groups, dtype=str)
    if ids.ndim != 1 or not len(ids) or any(not x.strip() for x in ids):
        raise ValueError("Training subject IDs must be non-empty.")
    if len(np.unique(ids)) < n_splits or n_splits < 2:
        raise ValueError("Insufficient participants for grouped CV.")
    record = np.full(len(ids), -1, dtype=int)
    folds = list(GroupKFold(n_splits=n_splits).split(
        np.zeros((len(ids), 1)), groups=ids,
    ))
    for fold, (train, valid) in enumerate(folds, start=1):
        if set(ids[train]) & set(ids[valid]):
            raise ValueError("Participant overlap in a Training fold.")
        if not np.all(record[valid] == -1):
            raise ValueError("A Training trial belongs to multiple folds.")
        record[valid] = fold
    if np.any(record < 1):
        raise ValueError("Training folds do not cover every trial.")
    return folds, record


# ================== 04. Trial and Subject-Level Diagnostics ==================
def regression_metrics(y, prediction):
    """Return two-target errors, including source-grid peak diagnostics."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 3 or y.shape[1:] != (11, 2) or y.shape != p.shape:
        raise ValueError("Observed and predicted FE waveforms must align.")
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite observed and predicted curves.")
    rows = []
    for j, target in enumerate(TARGETS):
        actual, estimate = y[:, :, j], p[:, :, j]
        diff = estimate - actual
        actual_peak = actual.max(axis=1) if j == 0 else actual.min(axis=1)
        pred_peak = estimate.max(axis=1) if j == 0 else estimate.min(axis=1)
        actual_idx = actual.argmax(axis=1) if j == 0 else actual.argmin(axis=1)
        pred_idx = estimate.argmax(axis=1) if j == 0 else estimate.argmin(axis=1)
        rows.append({
            "target": target,
            "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt(np.square(diff).mean())),
            "peak_value_mae": float(np.abs(pred_peak - actual_peak).mean()),
            "peak_index_mae": float(np.abs(pred_idx - actual_idx).mean()),
            "trials": int(len(actual)),
        })
    return rows


def participant_metrics(y, prediction, subjects, tasks, *, model, phase, fold=0):
    """Subject-wise and activity-wise metrics, with trial counts explicit."""
    ids = np.asarray(subjects, dtype=str)
    activities = np.asarray(tasks, dtype=str)
    if len(ids) != len(y) or len(activities) != len(y):
        raise ValueError("Participant/activity metadata do not match trials.")
    results = []
    for person in sorted(set(ids)):
        for activity in sorted(set(activities[ids == person])):
            mask = (ids == person) & (activities == activity)
            for row in regression_metrics(y[mask], prediction[mask]):
                results.append({
                    "phase": phase, "fold": fold, "model": model,
                    "subject_id": person, "task": activity, **row,
                })
    return pd.DataFrame(results)


def subject_balanced_mae(y, prediction, subjects):
    """Equal participant weight and equal weight for the two FE targets."""
    ids = np.asarray(subjects, dtype=str)
    if len(ids) != len(y):
        raise ValueError("Subject identity length mismatch.")
    scores = []
    for person in sorted(set(ids)):
        mask = ids == person
        scores.append(np.abs(y[mask] - prediction[mask]).mean(axis=(0, 1)))
    return float(np.asarray(scores).mean(axis=0).mean())


def trial_predictions(y, prediction, record_keys, subjects, tasks, model):
    """Long-form validation predictions at eleven original FE positions."""
    y = np.asarray(y)
    p = np.asarray(prediction)
    keys = np.asarray(record_keys, dtype=str)
    ids = np.asarray(subjects, dtype=str)
    activities = np.asarray(tasks, dtype=str)
    if y.shape != p.shape or y.shape[1:] != (11, 2) or len(keys) != len(y):
        raise ValueError("Validation data and prediction identities do not align.")
    if len(ids) != len(y) or len(activities) != len(y):
        raise ValueError("Prediction identity metadata mismatch.")
    return pd.DataFrame({
        "model": np.repeat(model, len(y) * 22),
        "local_record_key": np.repeat(keys, 22),
        "subject_id": np.repeat(ids, 22),
        "task": np.repeat(activities, 22),
        "fe_position": np.tile(np.repeat(np.arange(11), 2), len(y)),
        "target": np.tile(np.asarray(TARGETS), len(y) * 11),
        "observed": y.reshape(-1),
        "predicted": p.reshape(-1),
    })
