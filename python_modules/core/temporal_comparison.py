# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Matched subject-independent evaluation of classic and temporal models."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from core.feature_engineering import extract_signal_features
from core.ml_baselines import (
    classification_class_report,
    evaluate_predictions_by_subject,
)
from core.temporal_cnn import train_temporal_cnn


# ======================== 1. Group-Disjoint Model Comparison ========================
def compare_models_group_cv(
    windows,
    labels,
    subject_ids,
    classes,
    channel_names,
    n_splits=3,
    epochs=5,
    batch_size=128,
    learning_rate=1e-3,
    random_state=42,
    device="cpu",
    forbidden_subject_ids=(),
):
    """
    Evaluate three activity classifiers on identical held-out subjects.

    Args:
        windows (np.ndarray): IMU sequences (N, channels, time).
        labels (array-like): Source activity code per sequence window.
        subject_ids (array-like): Source participant ID per window.
        classes (sequence): Fixed documented activity codes in report order.
        channel_names (sequence): Sensor names aligned with IMU channels.
        n_splits (int): Number of participant-disjoint GroupKFold folds.
        epochs (int): Equal training epochs for both CNN variants.
        batch_size (int): Mini-batch size for CNN training and inference.
        learning_rate (float): Adam learning rate for each CNN.
        random_state (int): Fixed forest and both CNN random seeds.
        device (str): PyTorch training device, normally CPU.
        forbidden_subject_ids (sequence): IDs reserved outside development.

    Returns:
        fold_results (pd.DataFrame): Macro F1, Balanced Accuracy and participant
            membership for each model/fold (one record per combination).
        subject_results (pd.DataFrame): One fixed-task and observed-class
            Macro F1 score per held-out person and model.
        class_results (pd.DataFrame): Per-class precision/recall/F1 pooled from
            all out-of-fold development predictions, by model.
        prediction_records (pd.DataFrame): Actual source labels and predictions
            for each model and held-out row; supports independent checking.

    Notes:
        - All 3 models see identical subject-held-out windows and labels.
        - Random Forest uses deterministic 30-feature representations; CNNs
          use raw six-channel sequences, with training-only normalisation.
        - CrossEntropyLoss weighting is based ONLY on each CNN fold's Train.
        - Every development row is validated once per model; no reserved
          Test participant is loaded or scored by this function.
        - These development folds are not statistically independent trials.
          Pooled per-class values are diagnostic, not final Test metrics.
    """
    # 1. Validate aligned source arrays and the protected-subject boundary.
    x = np.asarray(windows)
    y = np.asarray(labels)
    source_ids = np.asarray(subject_ids, dtype=object)
    codes = list(classes)
    names = list(channel_names)
    if x.ndim != 3 or not len(x) or x.dtype.kind not in "biuf":
        raise ValueError("windows must be a non-empty numeric (N, channels, time) array.")
    if not np.isfinite(x).all():
        raise ValueError("Windows contain NaN or infinite values.")
    if y.ndim != 1 or source_ids.ndim != 1 or len(y) != len(x) or len(source_ids) != len(x):
        raise ValueError("Windows, source labels and participant IDs must align.")
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("classes must contain at least two distinct documented codes.")
    if not np.isin(y, codes).all():
        raise ValueError("Observed labels outside the declared task.")
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("One distinct channel name must be supplied per IMU channel.")
    if any(pd.isna(value) or not str(value).strip() for value in source_ids):
        raise ValueError("Missing participant IDs are not permitted.")
    display_ids = np.asarray([str(value).strip() for value in source_ids])
    group_keys = np.asarray([value.casefold() for value in display_ids])
    protected = {str(value).strip().casefold() for value in forbidden_subject_ids}
    if protected & set(group_keys):
        raise ValueError("Reserved participant found in development input.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer.")
    if not 2 <= n_splits <= len(set(group_keys)):
        raise ValueError("n_splits must be between 2 and the number of subjects.")

    # 2. Extract deterministic features once from the EXACT CNN window rows.
    #    This stateless operation cannot fit statistics on held-out labels.
    features = extract_signal_features(x, channel_names=names).to_numpy()
    group_folds = GroupKFold(n_splits=n_splits)

    # 3. Refit models from scratch within each subject-independent fold.
    fold_rows, subject_frames, prediction_frames = [], [], []
    validation_row_ids = []
    for fold, (train_idx, val_idx) in enumerate(
        group_folds.split(x, y, groups=group_keys), start=1
    ):
        train_people = set(group_keys[train_idx])
        val_people = set(group_keys[val_idx])
        if train_people & val_people:
            raise RuntimeError("Participant leakage in cross-validation.")
        if not set(codes).issubset(set(y[train_idx].tolist())):
            raise ValueError(f"Fold {fold} lacks one or more training task classes.")
        validation_row_ids.extend(val_idx.tolist())

        # 4. Fit Random Forest using the original Stage 9/11 hyperparameters.
        forest = RandomForestClassifier(
            n_estimators=120, min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )
        forest.fit(features[train_idx], y[train_idx])
        predictions = {"Random Forest": forest.predict(features[val_idx])}

        # 5. Compare two CNN variants using identical data and random seeds.
        for weighting, model_name in [
            ("none", "Standard CNN"), ("inverse_sqrt", "Weighted CNN")
        ]:
            print(f"Fold {fold}/{n_splits}: fitting {model_name}")
            _, _, predicted, _ = train_temporal_cnn(
                X_train=x[train_idx], y_train=y[train_idx],
                X_validation=x[val_idx], y_validation=y[val_idx],
                classes=codes, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, seed=random_state,
                device=device, class_weighting=weighting,
            )
            predictions[model_name] = predicted

        # 6. Record fixed-task metrics and complete validation row provenance.
        for name, predicted in predictions.items():
            fold_rows.append({
                "fold": fold,
                "model": name,
                "train_subjects": ", ".join(sorted(set(display_ids[train_idx]))),
                "validation_subjects": ", ".join(sorted(set(display_ids[val_idx]))),
                "train_windows": len(train_idx),
                "validation_windows": len(val_idx),
                "validation_classes_present": int(len(np.unique(y[val_idx]))),
                "accuracy": accuracy_score(y[val_idx], predicted),
                "balanced_accuracy": balanced_accuracy_score(y[val_idx], predicted),
                "macro_f1": f1_score(
                    y[val_idx], predicted, labels=codes,
                    average="macro", zero_division=0,
                ),
            })
            people = evaluate_predictions_by_subject(
                y_true=y[val_idx], y_pred=predicted,
                subject_ids=display_ids[val_idx], classes=codes,
                model_name=name,
            )
            people.insert(0, "fold", fold)
            subject_frames.append(people)
            prediction_frames.append(pd.DataFrame({
                "fold": fold,
                "model": name,
                "source_row": val_idx,
                "subject_id": display_ids[val_idx],
                "true_label": y[val_idx],
                "predicted_label": predicted,
            }))
        print(f"Fold {fold}: {len(train_people)} Train subjects, "
              f"{len(val_people)} held-out subjects")

    # 7. Every source row must appear in precisely one held-out group fold.
    if sorted(validation_row_ids) != list(range(len(x))):
        raise RuntimeError("Cross-validation failed to cover each window exactly once.")
    fold_results = pd.DataFrame(fold_rows)
    subject_results = pd.concat(subject_frames, ignore_index=True)
    prediction_records = pd.concat(prediction_frames, ignore_index=True)
    for model in ("Random Forest", "Standard CNN", "Weighted CNN"):
        model_predictions = prediction_records.loc[
            prediction_records["model"] == model
        ].sort_values("source_row")
        if model_predictions["source_row"].tolist() != list(range(len(x))):
            raise RuntimeError(f"Missing or repeated held-out rows for {model}.")
        if not np.array_equal(model_predictions["true_label"].to_numpy(), y):
            raise RuntimeError("Out-of-fold source-label alignment failed.")

    # 8. Derive class diagnostics from genuinely out-of-fold predictions.
    class_tables = []
    for model in ("Random Forest", "Standard CNN", "Weighted CNN"):
        subset = prediction_records.loc[prediction_records["model"] == model]
        table = classification_class_report(
            subset["true_label"].to_numpy(),
            subset["predicted_label"].to_numpy(),
            classes=codes,
        )
        table.insert(0, "model", model)
        class_tables.append(table)
    class_results = pd.concat(class_tables, ignore_index=True)
    return fold_results, subject_results, class_results, prediction_records
