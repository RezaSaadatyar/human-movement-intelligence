# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Matched subject-group CV for nested training-data scaling experiments."""

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


# ===================== 01. Nested-Data Cross-Validation ======================
def evaluate_nested_training_group_cv(
    windows,
    metadata,
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
    Compare Random Forest and Weighted CNN at two nested Train sizes.

    Args:
        windows (np.ndarray): Expanded eligible raw windows (N, channels, time).
        metadata (pd.DataFrame): Row-aligned source window data containing
            subject_id, label and in_base_500. Each expanded participant
            must have at least one original base window.
        classes (sequence): Documented, fixed activity codes.
        channel_names (sequence): Names for channels in their source order.
        n_splits (int): Subject-disjoint development folds.
        epochs (int): Fixed CNN training epochs for both data conditions.
        batch_size (int): CNN batch size.
        learning_rate (float): Adam learning rate.
        random_state (int): Random Forest/CNN reproducibility seed.
        device (str): PyTorch device, normally CPU.
        forbidden_subject_ids (sequence): Excluded original Test identities.

    Returns:
        fold_results (pd.DataFrame): Scores, model and participant membership
            for each model and group-held-out fold.
        subject_results (pd.DataFrame): Fixed-task and observed-class Macro
            F1 per individual validation participant and model.
        class_results (pd.DataFrame): Out-of-fold class support/precision/
            recall/F1 using the SAME base-cap Validation source windows.
        predictions (pd.DataFrame): Out-of-fold predictions with original
            source row indices and group identities for independent checks.

    Notes:
        - GroupKFold partitions ONLY the original 500-cap base rows.
        - Each fold validates on exactly the same base source rows under
          each of the four training conditions.
        - Expanded Train strictly includes all of its base Train rows.
        - All additional source rows are used ONLY for Train participants.
          Expanded rows from held-out/unused people never enter fitting.
        - CNN standardisation/class weights are fitted on each fold's Train.
        - Comparisons are exploratory development results, not a final
          untouched-Test result or independent hyperparameter selection.
    """
    # 1. Validate source-window alignment and explicit Test protection.
    x = np.asarray(windows)
    codes = list(classes)
    names = list(channel_names)
    required = {"subject_id", "label", "in_base_500"}
    if x.ndim != 3 or not len(x) or x.dtype.kind not in "biuf":
        raise ValueError("windows must be a non-empty numeric 3D array.")
    if not np.isfinite(x).all():
        raise ValueError("windows must be finite.")
    if not isinstance(metadata, pd.DataFrame) or not required.issubset(metadata):
        raise ValueError("metadata must contain subject_id, label and in_base_500.")
    if len(metadata) != len(x):
        raise ValueError("windows and metadata rows must align.")
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("classes must list distinct documented codes.")
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("One distinct source name is required per channel.")
    y = metadata["label"].to_numpy()
    if pd.isna(y).any() or not np.isin(y, codes).all():
        raise ValueError("labels outside the declared task.")
    raw = metadata["subject_id"].to_numpy(dtype=object)
    if any(pd.isna(value) or not str(value).strip() for value in raw):
        raise ValueError("Missing or blank participant ID.")
    display_ids = np.asarray([str(value).strip() for value in raw])
    groups = np.asarray([value.casefold() for value in display_ids])
    protected = {str(value).strip().casefold() for value in forbidden_subject_ids}
    if protected & set(groups):
        raise ValueError("Reserved Test subject found in development data.")
    if metadata["in_base_500"].dtype != bool:
        raise ValueError("in_base_500 must contain Boolean source membership.")
    is_base = metadata["in_base_500"].to_numpy(dtype=bool)
    if not is_base.any():
        raise ValueError("No original base windows found.")
    base_rows = np.flatnonzero(is_base)
    base_groups = groups[base_rows]
    if set(base_groups) != set(groups):
        raise ValueError("Every expanded subject must contain base windows.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer.")
    if not 2 <= n_splits <= len(set(base_groups)):
        raise ValueError("Invalid number of subject folds.")

    # 2. Stateless features from identical raw source windows for both forests.
    #    No scaler, label statistics or fitted preprocessing is used here.
    features = extract_signal_features(x, channel_names=names).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    fold_records, subject_tables, prediction_tables = [], [], []
    held_out_source_rows = []

    for fold, (train_local, validation_local) in enumerate(
        splitter.split(
            np.zeros(len(base_rows)), y[base_rows], groups=base_groups
        ), start=1
    ):
        # 3. Compute base/expanded Train from the same participant identities.
        train_people = set(base_groups[train_local])
        validation_people = set(base_groups[validation_local])
        if train_people & validation_people:
            raise RuntimeError("Subject leakage in grouped validation.")
        base_train_rows = base_rows[train_local]
        validation_rows = base_rows[validation_local]
        expanded_train_rows = np.flatnonzero(
            np.isin(groups, list(train_people))
        )
        if not np.isin(base_train_rows, expanded_train_rows).all():
            raise RuntimeError("Expanded Train must contain all base Train rows.")
        if set(groups[expanded_train_rows]) & validation_people:
            raise RuntimeError("Expanded Train contains Validation participants.")
        if not set(codes).issubset(set(y[base_train_rows])):
            raise ValueError(f"Fold {fold} lacks a training activity class.")
        held_out_source_rows.extend(validation_rows.tolist())
        print(
            f"Fold {fold}/{n_splits}: {len(train_people)} Train people, "
            f"{len(validation_people)} Validation people; "
            f"{len(base_train_rows)} vs {len(expanded_train_rows)} Train windows"
        )

        # 4. Fit both RF configurations on matching classical features.
        variant_rows = {
            "500": base_train_rows,
            "1000": expanded_train_rows,
        }
        predicted = {}
        for variant, train_rows in variant_rows.items():
            forest = RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )
            forest.fit(features[train_rows], y[train_rows])
            predicted[f"RF {variant}"] = forest.predict(
                features[validation_rows]
            )

        # 5. Train weighted CNNs from scratch with Train-only statistics.
        for variant, train_rows in variant_rows.items():
            print(f"Fold {fold}: fitting Weighted CNN {variant}")
            _, _, prediction, _ = train_temporal_cnn(
                X_train=x[train_rows],
                y_train=y[train_rows],
                X_validation=x[validation_rows],
                y_validation=y[validation_rows],
                classes=codes,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=random_state,
                device=device,
                class_weighting="inverse_sqrt",
            )
            predicted[f"Weighted CNN {variant}"] = prediction

        # 6. Record metrics and exact validation source identities.
        for name, predictions in predicted.items():
            fold_records.append({
                "fold": fold,
                "model": name,
                "train_subjects": ", ".join(sorted(set(display_ids[base_train_rows]))),
                "validation_subjects": ", ".join(sorted(set(display_ids[validation_rows]))),
                "train_windows": (
                    len(base_train_rows) if name.endswith("500")
                    else len(expanded_train_rows)
                ),
                "validation_windows": len(validation_rows),
                "validation_classes_present": len(np.unique(y[validation_rows])),
                "accuracy": accuracy_score(y[validation_rows], predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    y[validation_rows], predictions
                ),
                "macro_f1": f1_score(
                    y[validation_rows], predictions, labels=codes,
                    average="macro", zero_division=0,
                ),
            })
            individual = evaluate_predictions_by_subject(
                y_true=y[validation_rows],
                y_pred=predictions,
                subject_ids=display_ids[validation_rows],
                classes=codes,
                model_name=name,
            )
            individual.insert(0, "fold", fold)
            subject_tables.append(individual)
            prediction_tables.append(pd.DataFrame({
                "fold": fold,
                "model": name,
                "source_row": validation_rows,
                "subject_id": display_ids[validation_rows],
                "true_label": y[validation_rows],
                "predicted_label": predictions,
            }))

    # 7. Every original base sample must be validated exactly once per model.
    if sorted(held_out_source_rows) != sorted(base_rows.tolist()):
        raise RuntimeError("Fold validation omitted or duplicated a base window.")
    fold_results = pd.DataFrame(fold_records)
    subject_results = pd.concat(subject_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    variants = ["RF 500", "RF 1000", "Weighted CNN 500", "Weighted CNN 1000"]
    for name in variants:
        subset = predictions.loc[predictions["model"] == name]
        if sorted(subset["source_row"]) != sorted(base_rows.tolist()):
            raise RuntimeError(f"Missing or duplicate validation rows: {name}")
        if not np.array_equal(
            subset.sort_values("source_row")["true_label"].to_numpy(),
            y[base_rows[np.argsort(base_rows)]],
        ):
            raise RuntimeError("Ground-truth source-row alignment failed.")

    # 8. Produce real out-of-fold class diagnostics for the fixed base cohort.
    results = []
    for name in variants:
        subset = predictions.loc[predictions["model"] == name]
        report = classification_class_report(
            subset["true_label"].to_numpy(),
            subset["predicted_label"].to_numpy(),
            classes=codes,
        )
        report.insert(0, "model", name)
        results.append(report)
    class_results = pd.concat(results, ignore_index=True)
    return fold_results, subject_results, class_results, predictions
