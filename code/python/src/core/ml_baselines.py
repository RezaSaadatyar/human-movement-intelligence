# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Reusable leakage-aware classical classification baselines."""

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ========================= 1. Data and Label Validation =========================
def _validate_classification_arrays(X_train, y_train, X_validation, y_validation, classes):
    """Validate finite feature matrices, label alignment and task coverage."""
    train = np.asarray(X_train)
    valid = np.asarray(X_validation)
    y_tr = np.asarray(y_train)
    y_va = np.asarray(y_validation)
    task_classes = np.asarray(list(classes))

    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("Train and validation must be 2D arrays with matching features.")
    if not len(train) or not len(valid):
        raise ValueError("Train and validation cannot be empty.")
    if train.dtype.kind not in "biuf" or valid.dtype.kind not in "biuf":
        raise ValueError("Features must be numeric.")
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise ValueError("Features must not contain NaN or infinite values.")
    if y_tr.ndim != 1 or y_va.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    if len(y_tr) != len(train) or len(y_va) != len(valid):
        raise ValueError("Feature and label counts must match.")
    if len(task_classes) < 2 or len(np.unique(task_classes)) != len(task_classes):
        raise ValueError("Specify at least two distinct task classes.")
    if not np.isin(y_tr, task_classes).all() or not np.isin(y_va, task_classes).all():
        raise ValueError("Unexpected label outside the predefined task classes.")
    if not np.isin(task_classes, np.unique(y_tr)).all():
        raise ValueError("All predefined task classes must be represented in Train.")
    return train, y_tr, valid, y_va, task_classes


# ============================== 2. Model Training ================================
def train_validation_classifiers(
    X_train,
    y_train,
    X_validation,
    y_validation,
    classes,
    random_state=42,
):
    """
    Train simple classifiers and compare them on Validation only.

    Args:
        X_train (array-like): (train_samples, features), without subject leakage.
        y_train (array-like): One documented class label per train sample.
        X_validation (array-like): (validation_samples, same_features).
        y_validation (array-like): One documented class per validation sample.
        classes (sequence): Pre-specified classification labels in report order.
        random_state (int): Random seed for reproducible model fitting.

    Returns:
        models (dict): Fitted scikit-learn estimators; no test data used.
        comparison (pd.DataFrame): Validation Accuracy, Balanced Accuracy,
            Macro Precision, Macro Recall and Macro F1, sorted by Macro F1.
        predictions (dict): Validation predictions per model.

    Notes:
        - DummyClassifier is a majority-class benchmark.
        - Logistic Regression fits StandardScaler inside its own pipeline
          using TRAIN ONLY; no validation/test statistics leak into scaling.
        - Random Forest uses raw features and class-balanced bootstraps.
        - No hyperparameter search, final test scoring or external validation.
        - Comparing models on only two validation subjects is exploratory.
    """
    # 1. Validate the task before fitting any model.
    train, y_tr, valid, y_va, task_classes = _validate_classification_arrays(
        X_train, y_train, X_validation, y_validation, classes
    )

    # 2. Build three transparent baselines; keep all fitted state in Train.
    candidates = {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=800, random_state=random_state
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # 3. Fit with Train only; compute consistent, label-aware validation metrics.
    models, predictions, results = {}, {}, []
    for model_name, model in candidates.items():
        model.fit(train, y_tr)
        predicted = model.predict(valid)
        models[model_name] = model
        predictions[model_name] = predicted
        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_va, predicted),
            "balanced_accuracy": balanced_accuracy_score(y_va, predicted),
            "macro_precision": precision_score(
                y_va, predicted, labels=task_classes, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_va, predicted, labels=task_classes, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_va, predicted, labels=task_classes, average="macro", zero_division=0
            ),
        })
        print(f"{model_name}: Validation Macro F1 = {results[-1]['macro_f1']:.4f}")

    comparison = (
        pd.DataFrame(results)
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    return models, comparison, predictions


# ============================== 3. Model Evaluation ==============================
def classification_class_report(y_true, y_pred, classes, class_names=None):
    """
    Calculate class-wise validation support, precision, recall and F1.

    Args:
        y_true (array-like): Ground-truth labels from validation only.
        y_pred (array-like): Validation predictions from a trained classifier.
        classes (sequence): Fixed documented class codes.
        class_names (sequence | None): Optional human-readable class names.

    Returns:
        pd.DataFrame: One row per class, including validation support.
    """
    from sklearn.metrics import precision_recall_fscore_support

    labels = list(classes)
    if class_names is None:
        names = [str(code) for code in labels]
    else:
        names = list(class_names)
    if len(labels) != len(names) or len(set(labels)) != len(labels):
        raise ValueError("classes and class_names must have matching unique labels.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame({
        "class": names,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def classification_confusion(y_true, y_pred, classes):
    """Return the validation confusion matrix with fixed class order."""
    return confusion_matrix(y_true, y_pred, labels=list(classes))
