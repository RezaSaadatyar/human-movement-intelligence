# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Grouped cross-validation and per-participant classification diagnostics."""

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from core.ml_baselines import train_validation_classifiers


# ================= 01. Subject-Independent Cross-Validation ==================
def evaluate_group_cross_validation(
    features,
    labels,
    group_ids,
    classes,
    n_splits=3,
    random_state=42,
):
    """
    Compare classical ML baselines in subject-disjoint development folds.

    Args:
        features (array-like): Numerical matrix (development_windows, features).
            The reserved final Test rows must NOT be included.
        labels (array-like): One documented activity code per development window.
        group_ids (array-like): One participant identifier per window.
        classes (sequence): Fixed predefined activity codes for the task.
        n_splits (int): Number of disjoint participant folds.
        random_state (int): Fixed model-training seed; GroupKFold itself
            deterministically allocates groups by their sample counts.

    Returns:
        fold_results (pd.DataFrame): Validation metrics and group membership
            for each model and held-out fold.
        subject_results (pd.DataFrame): Classifier Macro F1 by validation
            participant, including class-coverage diagnostics.
        summary (pd.DataFrame): Mean and sample SD of fold-wise Macro F1,
            alongside subject-level means. All are exploratory.

    Notes:
        Train-only scaling is handled inside the scikit-learn Pipeline in
        train_validation_classifiers(). New estimators are fitted per fold.
        These diagnostics do NOT access the reserved final Test subjects.
        Folds are not statistically independent; the small subject cohort
        precludes strong uncertainty or generalisation claims.
    """
    # 1. Validate development data and participant identifiers before fitting.
    x = np.asarray(features)
    y = np.asarray(labels)
    groups = np.asarray(group_ids)
    task_classes = np.asarray(list(classes))

    if x.ndim != 2 or not len(x) or x.dtype.kind not in "biuf":
        raise ValueError("features must be a non-empty numeric 2D matrix.")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinite values.")
    if y.ndim != 1 or groups.ndim != 1 or len(x) != len(y) or len(x) != len(groups):
        raise ValueError("features, labels and group_ids must align by row.")
    if pd.isna(groups).any() or any(not str(s).strip() for s in groups):
        raise ValueError("Missing or empty participant IDs are not permitted.")
    # Keep original subject IDs for human-readable reporting.
    # Case-folded keys are used only for disjoint-group comparisons.
    display_groups = np.asarray([str(value).strip() for value in groups])
    groups = np.asarray([subject.casefold() for subject in display_groups])
    display_names = {}
    for canonical_id, source_id in zip(groups, display_groups):
        display_names.setdefault(canonical_id, source_id)
    if len(task_classes) < 2 or len(set(task_classes.tolist())) != len(task_classes):
        raise ValueError("classes must list at least two distinct labels.")
    if not np.isin(y, task_classes).all():
        raise ValueError("Found labels outside the predefined task.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer.")
    if n_splits < 2 or n_splits > len(np.unique(groups)):
        raise ValueError("n_splits must be between 2 and the number of subjects.")

    # 2. Split complete subjects, never randomly allocated windows.
    cv = GroupKFold(n_splits=int(n_splits))
    fold_records, subject_records, held_out_rows = [], [], []
    for fold, (training_rows, validation_rows) in enumerate(
        cv.split(x, y, groups=groups), start=1
    ):
        train_groups = set(groups[training_rows])
        validation_groups = set(groups[validation_rows])
        if train_groups & validation_groups:
            raise RuntimeError("Participant overlap within a CV fold.")
        held_out_rows.extend(validation_rows.tolist())
        if not np.isin(task_classes, np.unique(y[training_rows])).all():
            raise ValueError(
                f"Fold {fold}: not all predefined classes are represented in Train."
            )

        # 3. Fit fresh models on fold Train and predict fold Validation only.
        print(f"Fold {fold}: Train={len(train_groups)} subjects, "
              f"Validation={len(validation_groups)} subjects")
        _, comparison, predictions = train_validation_classifiers(
            X_train=x[training_rows],
            y_train=y[training_rows],
            X_validation=x[validation_rows],
            y_validation=y[validation_rows],
            classes=task_classes,
            random_state=random_state,
        )

        # 4. Store per-fold model scores and exact subject membership.
        for result in comparison.to_dict(orient="records"):
            fold_records.append({
                "fold": fold,
                "train_subjects": ", ".join(
                    sorted(display_names[name] for name in train_groups)
                ),
                "validation_subjects": ", ".join(
                    sorted(display_names[name] for name in validation_groups)
                ),
                "validation_windows": len(validation_rows),
                "validation_classes_present": len(np.unique(y[validation_rows])),
                **result,
            })

        # 5. Record performance separately for each held-out subject.
        #    The observed-class metric is distinguished from fixed 8-class F1.
        for model_name, predicted in predictions.items():
            for subject in sorted(validation_groups):
                mask = groups[validation_rows] == subject
                true_subject = y[validation_rows][mask]
                predicted_subject = predicted[mask]
                observed = np.unique(true_subject)
                subject_records.append({
                    "fold": fold,
                    "model": model_name,
                    "subject_id": display_names[subject],
                    "windows": int(mask.sum()),
                    "observed_classes": len(observed),
                    "macro_f1_fixed_task": f1_score(
                        true_subject, predicted_subject, labels=task_classes,
                        average="macro", zero_division=0,
                    ),
                    "macro_f1_observed_classes": f1_score(
                        true_subject, predicted_subject, labels=observed,
                        average="macro", zero_division=0,
                    ),
                })

    # 6. Each development window must be held out exactly once in GroupKFold.
    if sorted(held_out_rows) != list(range(len(x))):
        raise RuntimeError("CV folds do not form a complete, disjoint partition.")

    fold_results = pd.DataFrame(fold_records)
    subject_results = pd.DataFrame(subject_records)
    fold_summary = (
        fold_results.groupby("model", sort=False)["macro_f1"]
        .agg(fold_macro_f1_mean="mean", fold_macro_f1_sd="std",
             fold_macro_f1_min="min", fold_macro_f1_max="max")
    )
    subject_summary = (
        subject_results.groupby("model", sort=False)
        .agg(
            subject_macro_f1_fixed_mean=("macro_f1_fixed_task", "mean"),
            subject_macro_f1_observed_mean=("macro_f1_observed_classes", "mean"),
        )
    )
    summary = (
        fold_summary.join(subject_summary)
        .reset_index()
        .sort_values("fold_macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    return fold_results, subject_results, summary


# ===================== 02. Paired Participant Comparison =====================
def compare_models_by_subject(
    subject_results,
    model_a,
    model_b,
    score_column="macro_f1_fixed_task",
):
    """
    Compare two models on the SAME held-out independent participants.

    Args:
        subject_results (pd.DataFrame): Per-model, per-subject CV results.
            Must contain model, subject_id and the selected numeric score.
        model_a (str): First model's name in the results table.
        model_b (str): Second model's name in the results table.
        score_column (str): Metric with the same scientific definition for
            each participant, e.g. macro_f1_fixed_task.

    Returns:
        paired (pd.DataFrame): One row per common participant, with model_a,
            model_b and their score difference (model_a - model_b).
        summary (dict): Participant count, mean/median paired difference,
            and number of wins for each model and ties.

    Notes:
        This comparison weights every participant equally rather than
        treating overlapping windows as independent observations. It is
        descriptive, NOT a significance test or a confidence interval.
    """
    # 1. Require one comparable score from each model for every subject.
    required = {"model", "subject_id", score_column}
    if not isinstance(subject_results, pd.DataFrame) or not required.issubset(
        subject_results.columns
    ):
        raise ValueError("subject_results does not contain the required columns.")
    if model_a == model_b or not model_a or not model_b:
        raise ValueError("Select two distinct model names.")

    selected = subject_results.loc[
        subject_results["model"].isin([model_a, model_b]),
        ["model", "subject_id", score_column],
    ].copy()
    if selected["model"].nunique() != 2:
        raise ValueError("Both selected models must be present.")
    if selected.duplicated(["model", "subject_id"]).any():
        raise ValueError("Each model must have only one score per subject.")
    if selected["subject_id"].isna().any() or selected[score_column].isna().any():
        raise ValueError("Missing subjects or score values are not permitted.")
    if not pd.api.types.is_numeric_dtype(selected[score_column]):
        raise ValueError("Model comparison scores must be numeric.")
    if not np.isfinite(selected[score_column].to_numpy()).all():
        raise ValueError("Model comparison scores must be finite.")

    # 2. Check matching participant coverage before joining the scores.
    a = selected.loc[selected["model"] == model_a].set_index("subject_id")
    b = selected.loc[selected["model"] == model_b].set_index("subject_id")
    if set(a.index) != set(b.index):
        raise ValueError("The models must be evaluated on identical subjects.")

    # 3. Calculate signed paired differences, one per held-out subject.
    paired = pd.DataFrame({
        "subject_id": sorted(a.index),
    })
    paired["score_a"] = [
        float(a.at[subject, score_column]) for subject in paired["subject_id"]
    ]
    paired["score_b"] = [
        float(b.at[subject, score_column]) for subject in paired["subject_id"]
    ]
    paired["difference_a_minus_b"] = paired["score_a"] - paired["score_b"]
    differences = paired["difference_a_minus_b"].to_numpy()

    # 4. Report descriptive differences without unsupported p-values.
    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "metric": score_column,
        "participants": int(len(paired)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "model_a_wins": int((differences > 1e-12).sum()),
        "model_b_wins": int((differences < -1e-12).sum()),
        "ties": int((np.abs(differences) <= 1e-12).sum()),
    }
    return paired, summary
