# ==================== Presented by: Reza Saadatyar (2026) ====================
"""One-to-one gait event matching, participant metrics and uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================ 01. Event Matching ============================
def match_events(reference_s, estimated_s, tolerance_s=0.15):
    """Match one-to-one within predeclared tolerance; retain all misses/extras."""
    r = np.sort(np.asarray(reference_s, float))
    p = np.sort(np.asarray(estimated_s, float))
    if tolerance_s <= 0 or not np.isfinite(r).all() or not np.isfinite(p).all():
        raise ValueError("Invalid event tolerance/timestamps.")
    if len(r) == 0 or len(p) == 0:
        return pd.DataFrame(columns=["reference_s", "estimated_s", "error_ms"]), len(r), len(p)
    cost = np.abs(r[:, None] - p[None, :])
    # Avoid matching beyond tolerance at the expense of available valid matches.
    penalty = tolerance_s * 2 + 1
    padded = np.full((len(r) + len(p), len(r) + len(p)), penalty)
    padded[:len(r), :len(p)] = np.where(cost <= tolerance_s, cost, penalty * 4)
    padded[len(r):, len(p):] = 0
    ir, ip = linear_sum_assignment(padded)
    pairs = [(int(i), int(j)) for i, j in zip(ir, ip)
             if i < len(r) and j < len(p) and cost[i, j] <= tolerance_s]
    matches = pd.DataFrame([
        {"reference_s": r[i], "estimated_s": p[j],
         "error_ms": 1000.0 * (p[j] - r[i])} for i, j in pairs
    ], columns=["reference_s", "estimated_s", "error_ms"])
    return matches, len(r) - len(pairs), len(p) - len(pairs)


def event_metrics(reference_s, estimated_s, tolerance_s=0.15):
    """Return timing MAE plus event sensitivity, precision and support."""
    matches, missed, extra = match_events(reference_s, estimated_s, tolerance_s)
    n_match = len(matches)
    n_ref, n_pred = n_match + missed, n_match + extra
    return {
        "n_reference": n_ref, "n_predicted": n_pred, "n_matched": n_match,
        "n_missed": missed, "n_extra": extra,
        "timing_mae_ms": (float(matches.error_ms.abs().mean())
                          if n_match else np.nan),
        "timing_bias_ms": (float(matches.error_ms.mean())
                           if n_match else np.nan),
        "sensitivity": n_match / n_ref if n_ref else np.nan,
        "precision": n_match / n_pred if n_pred else np.nan,
    }


# ======================== 02. Participant-Level Summary =====================
def participant_metrics(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched-event MAE by participant; retain coverage per person."""
    if trial_metrics.empty:
        return pd.DataFrame(columns=["subject_id", "n_trials", "n_matched",
                                     "timing_mae_ms", "sensitivity", "precision"])
    rows = []
    for subject, group in trial_metrics.groupby("subject_id", sort=True):
        good = group.loc[group.n_matched > 0]
        n_match = int(group.n_matched.sum())
        n_ref = int(group.n_reference.sum())
        n_pred = int(group.n_predicted.sum())
        rows.append({
            "subject_id": subject, "n_trials": len(group), "n_matched": n_match,
            "timing_mae_ms": (float(np.average(good.timing_mae_ms,
                                weights=good.n_matched)) if n_match else np.nan),
            "sensitivity": n_match / n_ref if n_ref else np.nan,
            "precision": n_match / n_pred if n_pred else np.nan,
        })
    return pd.DataFrame(rows)


def paired_subject_bootstrap(table: pd.DataFrame, model_a: str, model_b: str,
                             draws=2000, seed=42):
    """Descriptive paired whole-person bootstrap of model B minus model A MAE."""
    wide = table.pivot(index="subject_id", columns="model_id",
                       values="timing_mae_ms")
    differences = (wide[model_b] - wide[model_a]).dropna().to_numpy(float)
    if len(differences) < 2:
        return {"n_participants": len(differences), "mean_difference_ms":
                float(np.mean(differences)) if len(differences) else np.nan,
                "lower_ms": np.nan, "upper_ms": np.nan}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, size=(draws, len(differences)), replace=True)
    lo, hi = np.percentile(sampled.mean(axis=1), [2.5, 97.5])
    return {"n_participants": len(differences),
            "mean_difference_ms": float(differences.mean()),
            "lower_ms": float(lo), "upper_ms": float(hi)}


# ============================= 03. Study Splits ==============================
def participant_roles(subject_ids, test_fraction=0.2, seed=42):
    """Reserve independent people before any model tuning."""
    people = np.asarray(sorted(set(map(str, subject_ids))))
    if len(people) < 6:
        raise ValueError("Need at least six people for Development/Test splitting.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(people)
    n_test = max(2, int(round(test_fraction * len(people))))
    n_test = min(n_test, len(people) - 4)
    test = set(shuffled[:n_test])
    return pd.DataFrame({"subject_id": people,
                         "role": ["Test" if p in test else "Development"
                                  for p in people]})


def development_folds(assignments: pd.DataFrame, n_splits=5):
    """Group Development participants into non-overlapping deterministic folds."""
    from sklearn.model_selection import KFold
    dev = sorted(assignments.loc[assignments.role == "Development",
                                 "subject_id"].astype(str).tolist())
    if len(dev) < n_splits:
        raise ValueError("Not enough Development participants for grouped folds.")
    fold = np.empty(len(dev), int)
    for i, (_, validation) in enumerate(
            KFold(n_splits=n_splits, shuffle=True, random_state=42).split(dev)):
        fold[validation] = i
    return pd.DataFrame({"subject_id": dev, "validation_fold": fold})
