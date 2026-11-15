# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Fixed-class, participant-balanced scoring and matched study comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ======================== 01. Fixed-Class Predictions ========================
def aligned_probabilities(model, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Put model probabilities in a fixed source-label class order."""
    observed = model.predict_proba(features)
    observed_classes = model.classes_
    result = np.zeros((len(features), len(classes)), np.float32)
    lookup = {int(value): i for i, value in enumerate(classes)}
    for j, label in enumerate(observed_classes):
        if int(label) not in lookup:
            raise ValueError(f"Model emitted an unknown class: {label}")
        result[:, lookup[int(label)]] = observed[:, j]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Invalid or unnormalised class probabilities.")
    return result


def predictions_from_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.asarray(classes)[np.asarray(probabilities).argmax(axis=1)]


def build_classical(name: str, seed: int = 42):
    """Define equally specified classical candidates, without Test tuning."""
    if name == "B0_prior":
        return DummyClassifier(strategy="prior")
    if name.endswith("_logreg"):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=500, class_weight="balanced", random_state=seed,
        ))
    if name.endswith("_rf"):
        return RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown classical model: {name}")


def classical_features(name: str, emg: np.ndarray, acc: np.ndarray) -> np.ndarray:
    if name == "B0_prior":
        return np.zeros((len(emg), 1), np.float32)
    if name.startswith("B1_"):
        return emg
    if name.startswith("B2_"):
        return acc
    if name.startswith("F1_"):
        return np.concatenate((emg, acc), axis=1)
    raise ValueError(f"Unknown classical feature mapping: {name}")


# ====================== 02. Independent Participant Metrics ==================
def fixed_class_metrics(y: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> dict:
    """Treat absent true-label classes as zero recall/F1 in the fixed task.

    Fixed-class balanced accuracy is the average of fixed-set recalls with
    zero support mapped to zero, reported with explicit supported-class count.
    This differs from sklearn's observed-class balanced_accuracy_score.
    """
    matrix = confusion_matrix(y, prediction, labels=classes)
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(len(classes), float),
                       where=support > 0)
    return {
        "macro_f1": float(f1_score(y, prediction, labels=classes,
                                    average="macro", zero_division=0)),
        "fixed_class_balanced_accuracy": float(np.mean(recall)),
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(prediction))),
        "supported_classes": int(np.sum(support > 0)),
        "total_classes": len(classes),
        "windows": len(y),
    }


def participant_metrics(
    metadata: pd.DataFrame, probabilities: np.ndarray, classes: np.ndarray,
    *, model_id: str, fold: int | None = None, seed: int = 42,
) -> pd.DataFrame:
    """Each participant gets an equal vote regardless of their window count."""
    if len(metadata) != len(probabilities):
        raise ValueError("Predictions and source window identities are misaligned.")
    prediction = predictions_from_probabilities(probabilities, classes)
    output = []
    for subject, part in metadata.groupby("subject_id", sort=True):
        loc = part.index.to_numpy(np.int64)
        score = fixed_class_metrics(part.label.to_numpy(int), prediction[loc], classes)
        output.append({"model_id": model_id, "subject_id": subject,
                       "validation_fold": fold, "seed": seed, **score})
    return pd.DataFrame(output)


def select_model(metrics: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> tuple[str, pd.DataFrame]:
    """Choose by mean person-F1; break exact ties by model ID, not Test."""
    summary = (metrics.loc[metrics.model_id.isin(candidates)]
               .groupby("model_id", as_index=False)
               .agg(mean_subject_macro_f1=("macro_f1", "mean"),
                    subjects=("subject_id", "nunique")))
    if summary.empty:
        raise ValueError(f"No completed Development results for {candidates}")
    summary = summary.sort_values(["mean_subject_macro_f1", "model_id"],
                                  ascending=[False, True]).reset_index(drop=True)
    return str(summary.iloc[0].model_id), summary


# ========================= 03. Paired Uncertainty ============================
def paired_subject_bootstrap(differences: np.ndarray, *, seed: int = 42,
                             draws: int = 2000) -> dict:
    """Descriptive interval conditional on fitted models and Test people."""
    diffs = np.asarray(differences, np.float64)
    if not len(diffs) or not np.isfinite(diffs).all():
        raise ValueError("Paired differences must be nonempty and finite.")
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(draws, len(diffs)), replace=True).mean(axis=1)
    return {"mean_difference": float(diffs.mean()),
            "bootstrap_2_5pct": float(np.quantile(samples, .025)),
            "bootstrap_97_5pct": float(np.quantile(samples, .975)),
            "independent_test_people": int(len(diffs)), "bootstrap_draws": draws}
