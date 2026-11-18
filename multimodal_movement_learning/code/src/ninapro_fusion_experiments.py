# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Participant-grouped fusion experiments and explicit Notebook 04 outputs."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ninapro_evaluation import (
    aligned_probabilities, build_classical, classical_features,
    participant_metrics, predictions_from_probabilities, select_model,
)
from ninapro_models import FUSION_MODELS, UNIMODAL_MODELS
from ninapro_training import fit_neural, predict_neural


EARLY_FUSION = ("F1_logreg", "F1_rf")
CLASSICAL_LATE = ("F2_logreg", "F2_rf")
DEEP_LATE = ("F4_deep_late",)
FUSION_ARMS = (*EARLY_FUSION, *CLASSICAL_LATE, *DEEP_LATE, *FUSION_MODELS)
ABLATION_MODELS = ("F3_dual_cnn", "F5_cross_attention")
PERTURBATIONS = (
    "acc_missing_mean", "acc_sensor_01_missing_mean",
    "acc_contiguous_gap_mean",
)


# ========================== 01. Prior OOF Inputs ============================
def load_unimodal_oof(index, dev, classes, output_dir):
    """Read real Notebook 03 soft probabilities with exact window-key checks.

    The notebook-specific ZIP is a portable source when .npz files have been
    excluded by the repository's Git ignore rules. Never approximate class
    probabilities from hard class predictions.
    """
    output_dir = Path(output_dir)
    archive_path = Path("/content/multimodal_movement_learning_Notebook_03_Outputs.zip")
    probability_file = output_dir / "03_oof_probabilities.npz"
    if not probability_file.is_file():
        if not archive_path.is_file():
            raise FileNotFoundError(
                "Notebook 04 requires 03_oof_probabilities.npz. "
                "Place the original Notebook 03 output ZIP at "
                "/content/multimodal_movement_learning_Notebook_03_Outputs.zip "
                "or restore its 03_oof_probabilities.npz into the study output/ "
                "directory. Hard-label CSV files cannot replace probabilities."
            )
        with ZipFile(archive_path) as archive:
            if "03_oof_probabilities.npz" not in archive.namelist():
                raise ValueError("Notebook 03 ZIP lacks its OOF probability file.")
            with archive.open("03_oof_probabilities.npz") as source:
                with probability_file.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        print("Recovered Notebook 03 OOF probabilities from its output ZIP.")

    classes = np.asarray(classes, dtype=int)
    dev = dev.reset_index(drop=True)
    dev_rows = dev.window_row.to_numpy(dtype=int)
    test_rows = index.loc[index.role == "Test", "window_row"].to_numpy(dtype=int)
    if not index.window_key.is_unique or not dev.window_key.is_unique:
        raise ValueError("Repeated source window identity.")
    if not np.array_equal(
        np.sort(index.window_row.to_numpy(dtype=int)),
        np.arange(len(index)),
    ):
        raise ValueError("Source window row mapping is not contiguous.")
    expected_models = {
        "B0_prior", "B1_logreg", "B1_rf", "B2_logreg", "B2_rf",
        *UNIMODAL_MODELS,
    }
    with np.load(probability_file, allow_pickle=False) as archive:
        if set(archive.files) != expected_models:
            raise ValueError("Notebook 03 probability model set has changed.")
        previous = {name: np.asarray(archive[name], dtype=np.float32)
                    for name in sorted(expected_models)}
    for name, values in previous.items():
        if values.shape != (len(index), len(classes)):
            raise ValueError(f"{name}: OOF probability shape mismatch.")
        part = values[dev_rows]
        if (not np.isfinite(part).all()
            or not np.allclose(part.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError(f"{name}: incomplete Development probabilities.")
        if test_rows.size and not np.isnan(values[test_rows]).all():
            raise ValueError(f"{name}: reserved Test rows must remain empty.")
        predictions_file = output_dir / f"03_{name}_oof_predictions.csv"
        if not predictions_file.is_file():
            raise FileNotFoundError(
                f"Missing Notebook 03 keyed predictions: {predictions_file.name}"
            )
        frame = pd.read_csv(predictions_file)
        if len(frame) != len(dev) or not frame.window_key.is_unique:
            raise ValueError(f"{name}: missing or repeated OOF window keys.")
        if set(frame.window_key) != set(dev.window_key):
            raise ValueError(f"{name}: OOF window identities do not match.")
        aligned = frame.set_index("window_key").loc[dev.window_key]
        if (not np.array_equal(
                aligned.subject_id.to_numpy(str),
                dev.subject_id.to_numpy(str))
            or not np.array_equal(
                aligned.validation_fold.to_numpy(int),
                dev.validation_fold.to_numpy(int))
            or not np.array_equal(
                aligned.true_label.to_numpy(int),
                dev.label.to_numpy(int))
            or not np.array_equal(
                aligned.predicted_label.to_numpy(int),
                predictions_from_probabilities(part, classes))):
            raise ValueError(f"{name}: OOF predictions disagree with probabilities.")
    print(
        f"Loaded matched Notebook 03 probabilities for {len(previous)} models "
        f"and {dev.subject_id.nunique()} Development participants."
    )
    return previous


# ========================= 02. Fold Output Contract =========================
def _prediction_frame(name, validation, proba, classes, fold):
    frame = validation[["window_key", "subject_id", "label"]].copy()
    frame["validation_fold"] = int(fold)
    frame["model_id"] = name
    frame["true_label"] = frame.pop("label").to_numpy(int)
    frame["predicted_label"] = predictions_from_probabilities(proba, classes)
    return frame.reset_index(drop=True)


def _write_arm(
    name, dev, classes, output_dir, *, metrics, predictions,
    probabilities, seed, history=None, robustness=None, settings=None,
):
    """Replace just this model's named current files after every fold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"04_{name}"
    pd.concat(metrics, ignore_index=True).to_csv(
        f"{prefix}_metrics.csv", index=False,
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        f"{prefix}_oof_predictions.csv", index=False,
    )
    record = {
        "window_key": dev.window_key.to_numpy(dtype=str),
        "validation_fold": dev.validation_fold.to_numpy(dtype=int),
        "class_labels": np.asarray(classes, dtype=int),
        "seed": np.array(int(seed)),
        "scores": np.asarray(probabilities, dtype=np.float32),
    }
    if settings:
        record.update({key: np.array(int(value))
                       for key, value in settings.items()})
    np.savez_compressed(f"{prefix}_oof_probabilities.npz", **record)
    if history:
        pd.concat(history, ignore_index=True).to_csv(
            f"{prefix}_learning_history.csv", index=False,
        )
    if robustness:
        pd.concat(robustness, ignore_index=True).to_csv(
            f"{prefix}_robustness.csv", index=False,
        )


def _check_fold(fitting, validation):
    if fitting.empty or validation.empty:
        raise ValueError("An empty Training or Validation fold was requested.")
    if set(fitting.subject_id) & set(validation.subject_id):
        raise ValueError("Training and Validation participants overlap.")


def _summarise(name, scores):
    data = pd.concat(scores, ignore_index=True)
    print(
        f"{name} | mean participant Development macro F1: "
        f"{data.macro_f1.mean():.4f}",
        flush=True,
    )
    return data.groupby("validation_fold", as_index=False).agg(
        participants=("subject_id", "nunique"),
        macro_f1=("macro_f1", "mean"),
    )


# ========================== 03. Classical Fusion ============================
def run_early_fusion_arm(
    name, dev, emg_features, acc_features, classes, *,
    output_dir, folds, seed=42,
):
    if name not in EARLY_FUSION:
        raise ValueError(f"Unknown early-fusion arm: {name}")
    dev = dev.reset_index(drop=True)
    classes = np.asarray(classes, dtype=int)
    loc = pd.Series(np.arange(len(dev)), index=dev.window_row.to_numpy(int))
    probs = np.full((len(dev), len(classes)), np.nan, dtype=np.float32)
    metrics, predictions = [], []
    for fold in folds:
        fitting = dev.loc[dev.validation_fold != fold]
        validation = dev.loc[dev.validation_fold == fold]
        _check_fold(fitting, validation)
        started = perf_counter()
        model = build_classical(name, seed=seed)
        x_train = classical_features(
            name, emg_features[fitting.window_row.to_numpy(int)],
            acc_features[fitting.window_row.to_numpy(int)],
        )
        model.fit(x_train, fitting.label.to_numpy(int))
        x_validation = classical_features(
            name, emg_features[validation.window_row.to_numpy(int)],
            acc_features[validation.window_row.to_numpy(int)],
        )
        p = aligned_probabilities(model, x_validation, classes)
        probs[loc.loc[validation.window_row.to_numpy(int)].to_numpy(int)] = p
        score = participant_metrics(
            validation.reset_index(drop=True), p, classes,
            model_id=name, fold=int(fold), seed=seed,
        )
        score["fit_and_prediction_seconds"] = perf_counter() - started
        metrics.append(score)
        predictions.append(_prediction_frame(name, validation, p, classes, fold))
        _write_arm(name, dev, classes, output_dir, metrics=metrics,
                   predictions=predictions, probabilities=probs, seed=seed)
        print(
            f"  {name} | fold {int(fold) + 1}/{len(folds)} | "
            f"macro F1 {score.macro_f1.mean():.4f}",
            flush=True,
        )
    return _summarise(name, metrics)


def run_fixed_late_arm(
    name, index, dev, previous, classes, *, output_dir, folds,
    emg_name, acc_name, seed=42,
):
    if name not in (*CLASSICAL_LATE, *DEEP_LATE):
        raise ValueError(f"Unknown fixed-weight late-fusion arm: {name}")
    classes = np.asarray(classes, dtype=int)
    dev = dev.reset_index(drop=True)
    source = 0.5 * (
        previous[emg_name] + previous[acc_name]
    )
    if source.shape != (len(index), len(classes)):
        raise ValueError("Late-fusion class probability shapes disagree.")
    probs = source[dev.window_row.to_numpy(dtype=int)]
    metrics, predictions = [], []
    for fold in folds:
        validation = dev.loc[dev.validation_fold == fold]
        if validation.empty:
            raise ValueError(f"Empty late-fusion validation fold: {fold}")
        p = source[validation.window_row.to_numpy(dtype=int)]
        score = participant_metrics(
            validation.reset_index(drop=True), p, classes,
            model_id=name, fold=int(fold), seed=seed,
        )
        metrics.append(score)
        predictions.append(_prediction_frame(name, validation, p, classes, fold))
        _write_arm(name, dev, classes, output_dir, metrics=metrics,
                   predictions=predictions, probabilities=probs, seed=seed)
    print(f"{name}: equal weights for {emg_name} and {acc_name}")
    return _summarise(name, metrics)


# =========================== 04. Neural Fusion ==============================
def run_neural_fusion_arm(
    name, dev, processed_dir, classes, *, output_dir, folds, device,
    seed=42, epochs=10, patience=3, batch_size=96, figure_save=None,
):
    if name not in FUSION_MODELS:
        raise ValueError(f"Unknown neural-fusion model: {name}")
    classes = np.asarray(classes, dtype=int)
    dev = dev.reset_index(drop=True)
    loc = pd.Series(np.arange(len(dev)), index=dev.window_row.to_numpy(int))
    probs = np.full((len(dev), len(classes)), np.nan, dtype=np.float32)
    metrics, predictions, histories, ablations = [], [], [], []
    print(
        f"\nModel: {name} | paired sEMG + acceleration "
        f"| device: {device} | {len(folds)} Development folds",
        flush=True,
    )
    for fold in folds:
        fitting = dev.loc[dev.validation_fold != fold]
        validation = dev.loc[dev.validation_fold == fold]
        _check_fold(fitting, validation)
        print(
            f"\n  Fold {int(fold) + 1}/{len(folds)} | "
            f"{fitting.subject_id.nunique()} Training people, "
            f"{validation.subject_id.nunique()} Validation people",
            flush=True,
        )
        model, stats, history, info = fit_neural(
            name, fitting, processed_dir, classes,
            device=device, seed=seed, epochs=epochs,
            patience=patience, batch_size=batch_size, show_progress=True,
        )
        p = predict_neural(
            name, model, validation, processed_dir, classes,
            stats, device=device,
        )
        probs[loc.loc[validation.window_row.to_numpy(int)].to_numpy(int)] = p
        score = participant_metrics(
            validation.reset_index(drop=True), p, classes,
            model_id=name, fold=int(fold), seed=seed,
        )
        for key, value in info.items():
            if key not in ("seed", "model_id"):
                score[key] = value
        history["validation_fold"] = int(fold)
        metrics.append(score)
        histories.append(history)
        predictions.append(_prediction_frame(name, validation, p, classes, fold))
        if name in ABLATION_MODELS:
            for condition in PERTURBATIONS:
                altered = predict_neural(
                    name, model, validation, processed_dir, classes, stats,
                    device=device, perturbation=condition,
                )
                measured = participant_metrics(
                    validation.reset_index(drop=True), altered, classes,
                    model_id=name, fold=int(fold), seed=seed,
                )
                measured["perturbation"] = condition
                ablations.append(measured)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _write_arm(
            name, dev, classes, output_dir, metrics=metrics,
            predictions=predictions, probabilities=probs, seed=seed,
            history=histories, robustness=ablations,
            settings={"epochs": epochs, "patience": patience,
                      "batch_size": batch_size},
        )
        print(
            f"  Fold {int(fold) + 1} completed | "
            f"macro F1 {score.macro_f1.mean():.4f} | "
            f"best epoch {info['best_epochs']} | "
            f"model fit {info['fit_seconds']:.1f} s",
            flush=True,
        )
    combined = pd.concat(histories, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for fold, part in combined.groupby("validation_fold", sort=True):
        ax.plot(part.epoch, part.inner_validation_macro_f1, marker="o",
                label=f"Fold {int(fold) + 1}")
    ax.set(
        xlabel="Training epoch", ylabel="Inner Training-only macro F1",
        title=f"{name}: inner validation by Development fold",
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=min(len(folds), 5), fontsize=9)
    figure_path = Path(output_dir) / f"04_{name}_learning_curve.png"
    if figure_save is None:
        fig.savefig(figure_path, dpi=140)
    else:
        figure_save(fig, figure_path)
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()
    return _summarise(name, metrics)


# ====================== 05. Consolidate Paired Evidence ======================
def load_completed_fusion_arm(
    name, dev, classes, output_dir, folds, *, seed=42,
    neural_settings=None,
):
    """Reject partial, stale or incorrectly indexed model results."""
    if name not in FUSION_ARMS:
        raise ValueError(f"Unknown fusion arm: {name}")
    output_dir = Path(output_dir)
    dev = dev.reset_index(drop=True)
    prefix = output_dir / f"04_{name}"
    required = [
        Path(f"{prefix}_metrics.csv"),
        Path(f"{prefix}_oof_predictions.csv"),
        Path(f"{prefix}_oof_probabilities.npz"),
    ]
    if name in FUSION_MODELS:
        required.append(Path(f"{prefix}_learning_history.csv"))
    for item in required:
        if not item.is_file():
            raise FileNotFoundError(
                f"Complete the {name} experiment before comparison: {item.name}"
            )
    with np.load(required[2], allow_pickle=False) as record:
        if not np.array_equal(
            record["window_key"], dev.window_key.to_numpy(str)
        ):
            raise ValueError(f"{name}: source window-key mismatch.")
        if not np.array_equal(
            record["validation_fold"], dev.validation_fold.to_numpy(int)
        ):
            raise ValueError(f"{name}: changed participant-fold assignments.")
        if not np.array_equal(record["class_labels"], np.asarray(classes)):
            raise ValueError(f"{name}: class label mismatch.")
        if int(record["seed"]) != int(seed):
            raise ValueError(f"{name}: training seed mismatch.")
        if neural_settings and name in FUSION_MODELS:
            if any(int(record[key]) != int(value)
                   for key, value in neural_settings.items()):
                raise ValueError(f"{name}: changed neural training settings.")
        proba = np.asarray(record["scores"], dtype=np.float32)
    if proba.shape != (len(dev), len(classes)):
        raise ValueError(f"{name}: OOF matrix shape mismatch.")
    if not np.isfinite(proba).all() or not np.allclose(
        proba.sum(axis=1), 1.0, atol=1e-4,
    ):
        raise ValueError(f"{name}: OOF probabilities are incomplete.")
    metric = pd.read_csv(required[0])
    predictions = pd.read_csv(required[1])
    if (set(metric.validation_fold.astype(int)) != set(folds)
        or set(metric.subject_id.astype(str)) != set(dev.subject_id.astype(str))
        or len(metric) != dev.subject_id.nunique()):
        raise ValueError(f"{name}: incomplete participant metrics.")
    if len(predictions) != len(dev) or not predictions.window_key.is_unique:
        raise ValueError(f"{name}: missing or repeated OOF prediction windows.")
    if set(predictions.window_key) != set(dev.window_key):
        raise ValueError(f"{name}: OOF prediction keys do not match.")
    ordered = predictions.set_index("window_key").loc[dev.window_key]
    if (not np.array_equal(
            ordered.predicted_label.to_numpy(int),
            predictions_from_probabilities(proba, classes))
        or not np.array_equal(
            ordered.true_label.to_numpy(int), dev.label.to_numpy(int))):
        raise ValueError(f"{name}: OOF predictions do not match the source.")
    history = (pd.read_csv(required[3]) if name in FUSION_MODELS else None)
    robust_path = Path(f"{prefix}_robustness.csv")
    robustness = (
        pd.read_csv(robust_path)
        if name in ABLATION_MODELS and robust_path.is_file() else None
    )
    if name in ABLATION_MODELS and (
        robustness is None
        or len(robustness) != len(PERTURBATIONS) * len(dev.subject_id.unique())
        or set(robustness.perturbation) != set(PERTURBATIONS)
    ):
        raise ValueError(f"{name}: incomplete synthetic perturbation scores.")
    return proba, metric, history, robustness


def consolidate_fusion_outputs(
    index, dev, classes, unimodal_metrics, previous, *,
    output_dir, folds, seed=42, neural_settings=None,
    chosen_emg, chosen_acc,
):
    """Create the unchanged Notebook 05 selection and OOF file contracts."""
    output_dir = Path(output_dir)
    dev = dev.reset_index(drop=True)
    output_probabilities = {}
    all_metrics, all_histories, ablations = [], [], []
    for name in FUSION_ARMS:
        part, metric, history, robustness = load_completed_fusion_arm(
            name, dev, classes, output_dir, folds, seed=seed,
            neural_settings=neural_settings,
        )
        full = np.full((len(index), len(classes)), np.nan, np.float32)
        full[dev.window_row.to_numpy(int)] = part
        output_probabilities[name] = full
        all_metrics.append(metric)
        if history is not None:
            all_histories.append(history)
        if robustness is not None:
            ablations.append(robustness)
        print(f"Loaded {name}: {metric.subject_id.nunique()} people", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    classical_metrics = metrics.loc[
        metrics.model_id.isin((*EARLY_FUSION, *CLASSICAL_LATE))
    ]
    deep_metrics = metrics.loc[
        metrics.model_id.isin((*DEEP_LATE, *FUSION_MODELS))
    ]
    classical_metrics.to_csv(output_dir / "04_dev_fusion_metrics.csv", index=False)
    deep_metrics.to_csv(output_dir / "04_dev_deep_fusion_metrics.csv", index=False)
    metrics.to_csv(output_dir / "04_all_fusion_metrics.csv", index=False)
    pd.concat(all_histories, ignore_index=True).to_csv(
        output_dir / "04_neural_learning_history.csv", index=False,
    )
    robustness = pd.concat(ablations, ignore_index=True)
    robustness.to_csv(output_dir / "04_fusion_robustness.csv", index=False)
    best_unimodal, u_summary = select_model(
        unimodal_metrics, list(previous),
    )
    best_fusion, f_summary = select_model(
        metrics, list(output_probabilities),
    )
    selection = pd.DataFrame([
        {"selection_role": "unimodal_comparator", "model_id": best_unimodal,
         "component_emg": "", "component_acc": ""},
        {"selection_role": "fusion_comparator", "model_id": best_fusion,
         "component_emg": chosen_emg if best_fusion in DEEP_LATE else "",
         "component_acc": chosen_acc if best_fusion in DEEP_LATE else ""},
    ])
    selection.to_csv(output_dir / "04_model_selection.csv", index=False)
    reference = participant_metrics(
        dev, previous[best_unimodal][dev.window_row.to_numpy(int)],
        classes, model_id=best_unimodal,
    ).set_index("subject_id").macro_f1
    pairs = []
    for name, full in output_probabilities.items():
        values = participant_metrics(
            dev, full[dev.window_row.to_numpy(int)],
            classes, model_id=name,
        )
        values["unimodal_reference"] = best_unimodal
        values["unimodal_macro_f1"] = values.subject_id.map(reference)
        values["fusion_minus_unimodal_macro_f1"] = (
            values.macro_f1 - values.unimodal_macro_f1
        )
        pairs.append(values)
    paired_effects = pd.concat(pairs, ignore_index=True)
    paired_effects.to_csv(
        output_dir / "04_oof_paired_differences.csv", index=False,
    )
    np.savez_compressed(
        output_dir / "04_oof_probabilities.npz", **output_probabilities,
    )
    return (best_unimodal, best_fusion, selection, u_summary, f_summary,
            paired_effects, robustness, metrics)


# ======================= 06. Additional Neural Seeds ========================
def evaluate_fusion_finalist_seeds(
    name, dev, processed_dir, classes, *, output_dir, folds, device,
    seeds=(43, 44), epochs=10, patience=3, batch_size=96,
):
    """Keep seed replicates separate from the primary model-selection run."""
    if name not in FUSION_MODELS:
        raise ValueError(f"Unknown neural finalist: {name}")
    metrics, histories = [], []
    output_dir = Path(output_dir)
    for seed in seeds:
        for fold in folds:
            fitting = dev.loc[dev.validation_fold != fold]
            validation = dev.loc[dev.validation_fold == fold]
            _check_fold(fitting, validation)
            print(
                f"\n{name} | additional seed {seed} | fold "
                f"{int(fold)+1}/{len(folds)}",
                flush=True,
            )
            model, stats, history, info = fit_neural(
                name, fitting, processed_dir, classes, device=device,
                seed=seed, epochs=epochs, patience=patience,
                batch_size=batch_size, show_progress=True,
            )
            p = predict_neural(
                name, model, validation, processed_dir, classes,
                stats, device=device,
            )
            score = participant_metrics(
                validation.reset_index(drop=True), p, classes,
                model_id=name, fold=int(fold), seed=seed,
            )
            score["fit_seconds"] = info["fit_seconds"]
            metrics.append(score)
            history["validation_fold"] = int(fold)
            histories.append(history)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.concat(metrics, ignore_index=True).to_csv(
                output_dir / f"04_{name}_extra_seed_metrics.csv", index=False,
            )
            pd.concat(histories, ignore_index=True).to_csv(
                output_dir / f"04_{name}_extra_seed_history.csv", index=False,
            )
            print(
                f"  Additional seed {seed}, fold {int(fold) + 1}: "
                f"macro F1 {score.macro_f1.mean():.4f}",
                flush=True,
            )
    summary = pd.concat(metrics).groupby("seed", as_index=False).agg(
        participant_mean_macro_f1=("macro_f1", "mean"),
        participants=("subject_id", "nunique"),
    )
    return summary
