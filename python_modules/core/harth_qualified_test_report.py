# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Read-only reporting utilities for the approved HARTH qualified held-out evaluation.

Accept frozen predictions; do not train, calibrate, choose models or access files.
The eight-class task denominator stays fixed, including unsupported source classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

SOURCE_CLASSES = tuple(range(1, 9))
RESERVED_PEOPLE = ("S023", "S032")
FINAL_MODELS = ("1D-ResNet Weighted CE", "Random Forest")


def validate_final_test_inputs(y_true, subject_ids, predictions, classes=SOURCE_CLASSES):
    """
    Validate source-aligned frozen predictions for both reserved individuals.

    Args:
        y_true (array-like): Original true HARTH activity labels.
        subject_ids (array-like): Aligned original participant identifiers.
        predictions (dict[str, array-like]): Model name to predicted source codes.
        classes (sequence[int]): Fixed eight-class source-code specification.

    Returns:
        tuple: Validated target, person, predictions and task class arrays.

    Notes:
        A failed validation must not lead to retraining or changing the
        predeclared evaluation task. Only two original Test subjects are valid.
    """
    codes = np.asarray(classes, dtype=int)
    if not np.array_equal(codes, np.arange(1, 9)):
        raise ValueError("Qualified scoring requires fixed source classes 1..8.")
    target = np.asarray(y_true)
    subjects = np.asarray(subject_ids, dtype=str)
    if target.ndim != 1 or subjects.ndim != 1 or not len(target):
        raise ValueError("Source labels and subjects must be one-dimensional and nonempty.")
    if len(subjects) != len(target) or set(subjects) != set(RESERVED_PEOPLE):
        raise ValueError("Qualified scoring requires aligned S023 and S032 samples.")
    if not np.isin(target, codes).all():
        raise ValueError("True labels outside the frozen eight-class task.")
    if set(predictions) != set(FINAL_MODELS):
        raise ValueError("Expected exactly the two frozen, named reference models.")
    checked = {}
    for name in FINAL_MODELS:
        values = np.asarray(predictions[name])
        if values.ndim != 1 or len(values) != len(target):
            raise ValueError(f"Unaligned predictions from {name}.")
        if not np.isin(values, codes).all():
            raise ValueError(f"Unknown predicted source code from {name}.")
        checked[name] = values.astype(int, copy=False)
    return target.astype(int, copy=False), subjects, checked, codes


def summarise_qualified_heldout(y_true, subject_ids, predictions,
                                activity_names, classes=SOURCE_CLASSES):
    """
    Summarise frozen Test predictions without any model fitting.

    Args:
        y_true (array-like): Original Test activity labels aligned to windows.
        subject_ids (array-like): Original Test participant per window.
        predictions (dict): Pre-computed source-code predictions of both models.
        activity_names (dict): Official activity code-to-name mapping.
        classes (sequence[int]): Fixed task label order.

    Returns:
        dict: DataFrame tables for model, participant and class metrics,
        along with eight-by-eight count confusion matrices keyed by model.

    Notes:
        Macro F1 explicitly counts all eight target classes (zero F1 for
        unsupported classes). Balanced accuracy is the mean recall of
        *observed* true classes, so it is not a fixed-eight-class statistic.
        Overlapping windows and only two, historically inspected participants
        preclude independent/external/clinical validation claims.
    """
    true, people, checked, codes = validate_final_test_inputs(
        y_true, subject_ids, predictions, classes=classes
    )
    model_rows, subject_rows, class_rows, matrices = [], [], [], {}
    for model_name in FINAL_MODELS:
        predicted = checked[model_name]
        cm = confusion_matrix(true, predicted, labels=codes)
        matrices[model_name] = cm
        report = classification_report(
            true, predicted, labels=codes, output_dict=True, zero_division=0
        )
        model_rows.append({
            "model": model_name,
            "windows": int(len(true)),
            "subjects": int(len(set(people))),
            "fixed_eight_class_macro_f1": float(f1_score(
                true, predicted, labels=codes, average="macro", zero_division=0
            )),
            "balanced_accuracy_observed_classes": float(balanced_accuracy_score(
                true, predicted
            )),
            "accuracy": float(accuracy_score(true, predicted)),
            "true_classes_present": int(len(set(true))),
        })
        for code in codes:
            cls = report[str(int(code))]
            class_rows.append({
                "model": model_name, "source_code": int(code),
                "activity": str(activity_names[int(code)]),
                "precision": float(cls["precision"]),
                "recall": float(cls["recall"]),
                "f1": float(cls["f1-score"]),
                "support": int(cls["support"]),
            })
        for person in RESERVED_PEOPLE:
            mask = people == person
            yt, yp = true[mask], predicted[mask]
            subject_rows.append({
                "model": model_name, "subject_id": person,
                "windows": int(mask.sum()),
                "true_classes_present": int(len(set(yt))),
                "fixed_eight_class_macro_f1": float(f1_score(
                    yt, yp, labels=codes, average="macro", zero_division=0
                )),
                "balanced_accuracy_observed_classes": float(
                    balanced_accuracy_score(yt, yp)
                ),
                "accuracy": float(accuracy_score(yt, yp)),
            })
    return {
        "summary": pd.DataFrame(model_rows),
        "per_subject": pd.DataFrame(subject_rows),
        "per_class": pd.DataFrame(class_rows),
        "confusion": matrices,
    }
