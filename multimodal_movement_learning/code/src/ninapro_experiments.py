# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Participant-grouped NinaPro experiments and canonical stage-03 outputs.

Study-specific orchestration belongs here; architecture definitions and fitting
primitives remain in ninapro_models.py, ninapro_evaluation.py and
ninapro_training.py. Each study output is replaced by the stage that owns it.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning

from ninapro_evaluation import (
    aligned_probabilities, build_classical, classical_features,
    participant_metrics, predictions_from_probabilities, select_model,
)
from ninapro_models import UNIMODAL_MODELS
from ninapro_training import fit_neural, predict_neural


CLASSICAL_MODELS = (
    "B0_prior", "B1_logreg", "B1_rf", "B2_logreg", "B2_rf",
)


def _check_fold_participants(fitting: pd.DataFrame, validation: pd.DataFrame) -> None:
    if fitting.empty or validation.empty:
        raise ValueError("An outer Training/Validation fold is empty.")
    if set(fitting.subject_id) & set(validation.subject_id):
        raise ValueError("The outer Training/Validation groups share participants.")


def _save_model_progress(
    name: str, dev: pd.DataFrame, classes: np.ndarray, output_dir: Path,
    scores: list[pd.DataFrame], predictions: list[pd.DataFrame],
    proba: np.ndarray, *, seed: int, histories: list[pd.DataFrame] | None = None,
    settings: dict | None = None,
) -> None:
    """Replace one model's current results after each completed outer fold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(scores, ignore_index=True).to_csv(
        output_dir / f"03_{name}_metrics.csv", index=False,
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / f"03_{name}_oof_predictions.csv", index=False,
    )
    if histories is not None:
        pd.concat(histories, ignore_index=True).to_csv(
            output_dir / f"03_{name}_learning_history.csv", index=False,
        )
    payload = {
        "window_key": dev.window_key.to_numpy(dtype=str),
        "validation_fold": dev.validation_fold.to_numpy(dtype=int),
        "class_labels": np.asarray(classes, dtype=int),
        "scores": proba,
        "seed": np.asarray(seed, dtype=int),
    }
    for key, value in (settings or {}).items():
        payload[key] = np.asarray(value, dtype=int)
    np.savez_compressed(
        output_dir / f"03_{name}_oof_probabilities.npz", **payload,
    )


def _predictions_frame(
    name: str, validation: pd.DataFrame, proba: np.ndarray,
    classes: np.ndarray, fold: int,
) -> pd.DataFrame:
    return pd.DataFrame({
        "window_key": validation.window_key.to_numpy(),
        "subject_id": validation.subject_id.to_numpy(),
        "validation_fold": fold,
        "model_id": name,
        "true_label": validation.label.to_numpy(int),
        "predicted_label": predictions_from_probabilities(proba, classes),
    })


def _fit_classical_with_convergence_check(
    name: str, features: np.ndarray, labels: np.ndarray, *, seed: int,
):
    """Allow an lbfgs solve to finish instead of concealing an iteration warning.

    Numerical retries keep the same data, objective, solver, class weights and
    tolerance. A still-unconverged run fails visibly and is not scored as final.
    """
    model = build_classical(name, seed=seed)
    if not name.endswith("_logreg"):
        model.fit(features, labels)
        return model, None

    logreg = model.named_steps["logisticregression"]
    for limit in (2000, 4000, 6000):
        logreg.set_params(max_iter=limit)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, labels)
        for item in captured:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn(str(item.message), item.category, stacklevel=2)
        nonconverged = any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        )
        iterations = int(np.max(logreg.n_iter_))
        if not nonconverged and iterations < limit:
            return model, iterations
        if limit < 6000:
            print(
                f"  {name}: solver iteration limit {limit} reached "
                f"(observed iterations={iterations}); retrying with more "
                "iterations on the same Training data.",
                flush=True,
            )
    raise RuntimeError(
        f"{name} did not converge after 6000 lbfgs iterations. "
        "Inspect feature conditioning or review the numerical solver "
        "before interpreting this model's probabilities."
    )


def run_classical_arm(
    name: str, dev: pd.DataFrame, emg_features: np.ndarray,
    acc_features: np.ndarray, classes: np.ndarray, *, output_dir: Path,
    folds: list[int], seed: int = 42,
) -> pd.DataFrame:
    """Fit one classical model across the same person-disjoint outer folds."""
    if name not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown classical model: {name}")
    dev = dev.reset_index(drop=True)
    scores_list, predictions_list = [], []
    probabilities = np.full((len(dev), len(classes)), np.nan, np.float32)
    location = pd.Series(np.arange(len(dev)), index=dev.window_row.to_numpy(int))
    print(f"\nModel: {name} | modality: "
          f"{'sEMG' if name.startswith('B1_') else 'Acceleration' if name.startswith('B2_') else 'Training prior'} "
          f"| {len(folds)} independent Development folds", flush=True)
    for fold in folds:
        fitting = dev.loc[dev.validation_fold == fold].copy()
        training = dev.loc[dev.validation_fold != fold].copy()
        _check_fold_participants(training, fitting)
        start = perf_counter()
        train_x = classical_features(
            name, emg_features[training.window_row.to_numpy(int)],
            acc_features[training.window_row.to_numpy(int)],
        )
        model, iterations = _fit_classical_with_convergence_check(
            name, train_x, training.label.to_numpy(int), seed=seed,
        )
        val_x = classical_features(
            name, emg_features[fitting.window_row.to_numpy(int)],
            acc_features[fitting.window_row.to_numpy(int)],
        )
        proba = aligned_probabilities(model, val_x, classes)
        seconds = perf_counter() - start
        probabilities[location.loc[fitting.window_row.to_numpy(int)].to_numpy(int)] = proba
        metric = participant_metrics(
            fitting.reset_index(drop=True), proba, classes,
            model_id=name, fold=int(fold), seed=seed,
        )
        metric["fit_and_prediction_seconds"] = seconds
        if iterations is not None:
            metric["solver_iterations"] = iterations
        scores_list.append(metric)
        predictions_list.append(_predictions_frame(name, fitting, proba, classes, int(fold)))
        _save_model_progress(
            name, dev, classes, output_dir, scores_list, predictions_list,
            probabilities, seed=seed,
        )
        convergence = f" | converged in {iterations} iterations" if iterations is not None else ""
        print(
            f"  Fold {int(fold) + 1}/{len(folds)} | "
            f"{fitting.subject_id.nunique()} validation people | "
            f"mean participant macro F1 {metric.macro_f1.mean():.4f} | "
            f"{seconds:.1f} s{convergence}",
            flush=True,
        )
    completed = pd.concat(scores_list, ignore_index=True)
    summary = completed.groupby("validation_fold", as_index=False).agg(
        participants=("subject_id", "nunique"),
        macro_f1=("macro_f1", "mean"),
        fit_seconds=("fit_and_prediction_seconds", "max"),
    )
    if name.endswith("_logreg"):
        iteration_counts = completed.groupby("validation_fold").solver_iterations.max()
        summary["solver_iterations"] = summary.validation_fold.map(iteration_counts).astype(int)
    print(
        f"{name} | equal-participant Development macro F1: "
        f"{completed.macro_f1.mean():.4f}", flush=True,
    )
    return summary


def run_unimodal_arm(
    name: str, dev: pd.DataFrame, processed_dir: Path,
    classes: np.ndarray, *, output_dir: Path, folds: list[int], device,
    seed: int = 42, epochs: int = 10, patience: int = 3,
    batch_size: int = 96, figure_save=None,
) -> pd.DataFrame:
    """Train and report one temporal network with independent outer folds."""
    if name not in UNIMODAL_MODELS:
        raise ValueError(f"Unknown neural model: {name}")
    dev = dev.reset_index(drop=True)
    scores_list, predictions_list, histories = [], [], []
    probabilities = np.full((len(dev), len(classes)), np.nan, np.float32)
    location = pd.Series(np.arange(len(dev)), index=dev.window_row.to_numpy(int))
    modality = "sEMG" if "_emg_" in name else "Acceleration"
    print(
        f"\nModel: {name} | modality: {modality} | device: {device} "
        f"| {len(folds)} independent Development folds", flush=True,
    )
    for fold in folds:
        training = dev.loc[dev.validation_fold != fold].copy()
        validation = dev.loc[dev.validation_fold == fold].copy()
        _check_fold_participants(training, validation)
        print(
            f"\n  Fold {int(fold) + 1}/{len(folds)} | "
            f"{training.subject_id.nunique()} Training, "
            f"{validation.subject_id.nunique()} Validation people", flush=True,
        )
        model, stats, history, info = fit_neural(
            name, training, processed_dir, classes,
            device=device, seed=seed, epochs=epochs, patience=patience,
            batch_size=batch_size, show_progress=True,
        )
        proba = predict_neural(
            name, model, validation, processed_dir, classes, stats,
            device=device,
        )
        probabilities[location.loc[validation.window_row.to_numpy(int)].to_numpy(int)] = proba
        metric = participant_metrics(
            validation.reset_index(drop=True), proba, classes,
            model_id=name, fold=int(fold), seed=seed,
        )
        for key, value in info.items():
            if key not in ("seed", "model_id"):
                metric[key] = value
        history["validation_fold"] = int(fold)
        scores_list.append(metric)
        histories.append(history)
        predictions_list.append(_predictions_frame(name, validation, proba, classes, int(fold)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save_model_progress(
            name, dev, classes, output_dir, scores_list, predictions_list,
            probabilities, seed=seed, histories=histories,
            settings={"epochs": epochs, "patience": patience,
                      "batch_size": batch_size},
        )
        print(
            f"  Fold {int(fold) + 1} completed | "
            f"participant-mean macro F1 {metric.macro_f1.mean():.4f} | "
            f"best epoch {info['best_epochs']} | "
            f"fit {info['fit_seconds']:.1f} s", flush=True,
        )

    completed = pd.concat(scores_list, ignore_index=True)
    all_history = pd.concat(histories, ignore_index=True)
    summary = completed.groupby("validation_fold", as_index=False).agg(
        participants=("subject_id", "nunique"),
        macro_f1=("macro_f1", "mean"),
        best_epoch=("best_epochs", "max"),
        fit_seconds=("fit_seconds", "max"),
    )
    print(
        f"\n{name} | equal-participant Development macro F1: "
        f"{completed.macro_f1.mean():.4f}", flush=True,
    )
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for fold, part in all_history.groupby("validation_fold", sort=True):
        ax.plot(part.epoch, part.inner_validation_macro_f1, marker="o",
                label=f"Fold {int(fold) + 1}")
    ax.set(
        xlabel="Training epoch",
        ylabel="Inner validation macro F1",
        title=f"{name}: inner Training-only validation",
    )
    ax.legend(ncol=min(len(folds), 5), fontsize=9)
    ax.grid(alpha=0.2)
    figure_path = Path(output_dir) / f"03_{name}_learning_curve.png"
    if figure_save is None:
        fig.savefig(figure_path, dpi=140)
    else:
        figure_save(fig, figure_path)
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()
    return summary


def load_completed_model(
    name: str, dev: pd.DataFrame, classes: np.ndarray, output_dir: Path,
    folds: list[int], *, seed: int = 42,
    neural_settings: dict | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Accept only complete current OOF outputs with matching identities."""
    output_dir = Path(output_dir)
    metrics_file = output_dir / f"03_{name}_metrics.csv"
    preds_file = output_dir / f"03_{name}_oof_predictions.csv"
    proba_file = output_dir / f"03_{name}_oof_probabilities.npz"
    history_file = output_dir / f"03_{name}_learning_history.csv"
    required = [metrics_file, preds_file, proba_file]
    if name in UNIMODAL_MODELS:
        required.append(history_file)
    missing = [f.name for f in required if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Run the {name} experiment cell first: {missing}")
    with np.load(proba_file, allow_pickle=False) as saved:
        if not np.array_equal(saved["window_key"], dev.window_key.to_numpy(dtype=str)):
            raise ValueError(f"{name}: Development window identities changed.")
        if not np.array_equal(saved["validation_fold"],
                              dev.validation_fold.to_numpy(dtype=int)):
            raise ValueError(f"{name}: Development fold identities changed.")
        if not np.array_equal(saved["class_labels"], np.asarray(classes, dtype=int)):
            raise ValueError(f"{name}: class definitions changed.")
        if int(saved["seed"]) != seed:
            raise ValueError(f"{name}: mismatched primary training seed.")
        if neural_settings and name in UNIMODAL_MODELS:
            if any(int(saved[k]) != int(v) for k, v in neural_settings.items()):
                raise ValueError(f"{name}: neural training settings changed.")
        probs = np.asarray(saved["scores"], dtype=np.float32)
    if probs.shape != (len(dev), len(classes)):
        raise ValueError(f"{name}: incorrect probability matrix dimensions.")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{name}: OOF probabilities are missing or invalid; complete all folds.")
    metrics = pd.read_csv(metrics_file)
    predictions = pd.read_csv(preds_file)
    if set(metrics.validation_fold.astype(int)) != set(folds):
        raise ValueError(f"{name}: not all Development folds have metrics.")
    if set(metrics.subject_id.astype(str)) != set(dev.subject_id.astype(str)):
        raise ValueError(f"{name}: incomplete participant-level metrics.")
    if len(predictions) != len(dev) or not predictions.window_key.is_unique:
        raise ValueError(f"{name}: incomplete or duplicated window predictions.")
    if not np.array_equal(predictions.window_key.to_numpy(dtype=str),
                          dev.window_key.to_numpy(dtype=str)):
        # Fold-wise output files are saved in validation-fold order.
        actual = predictions.set_index("window_key")
        if set(actual.index) != set(dev.window_key):
            raise ValueError(f"{name}: OOF prediction window identities do not match.")
        predictions = actual.loc[dev.window_key].reset_index()
    if not np.array_equal(
        predictions.predicted_label.to_numpy(dtype=int),
        predictions_from_probabilities(probs, classes),
    ):
        raise ValueError(f"{name}: prediction labels disagree with saved probabilities.")
    history = pd.read_csv(history_file) if name in UNIMODAL_MODELS else None
    return probs, metrics, predictions, history


def consolidate_unimodal_outputs(
    index: pd.DataFrame, dev: pd.DataFrame, classes: np.ndarray,
    *, output_dir: Path, folds: list[int], seed: int = 42,
    neural_settings: dict | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    """Build existing Notebook-04 input contracts from fully finished models."""
    output_dir = Path(output_dir)
    dev = dev.reset_index(drop=True)
    all_probabilities = {}
    classical_metrics, classical_predictions = [], []
    neural_metrics, neural_predictions, neural_histories = [], [], []
    for name in (*CLASSICAL_MODELS, *UNIMODAL_MODELS):
        p_dev, metric, predictions, history = load_completed_model(
            name, dev, classes, output_dir, folds, seed=seed,
            neural_settings=neural_settings,
        )
        full = np.full((len(index), len(classes)), np.nan, dtype=np.float32)
        full[dev.window_row.to_numpy(dtype=int)] = p_dev
        all_probabilities[name] = full
        if name in CLASSICAL_MODELS:
            classical_metrics.append(metric)
            classical_predictions.append(predictions)
        else:
            neural_metrics.append(metric)
            neural_predictions.append(predictions)
            neural_histories.append(history)
        print(f"Loaded {name}: {metric.subject_id.nunique()} Development people", flush=True)
    cm = pd.concat(classical_metrics, ignore_index=True)
    cp = pd.concat(classical_predictions, ignore_index=True)
    nm = pd.concat(neural_metrics, ignore_index=True)
    nh = pd.concat(neural_histories, ignore_index=True)
    npreds = pd.concat(neural_predictions, ignore_index=True)
    all_metrics = pd.concat([cm, nm], ignore_index=True)
    selected, summary = select_model(all_metrics, list(all_probabilities))
    cm.to_csv(output_dir / "03_dev_unimodal_metrics.csv", index=False)
    cp.to_csv(output_dir / "03_oof_predictions.csv", index=False)
    nm.to_csv(output_dir / "03_dev_neural_metrics.csv", index=False)
    nh.to_csv(output_dir / "03_neural_learning_history.csv", index=False)
    npreds.to_csv(output_dir / "03_neural_oof_predictions.csv", index=False)
    all_metrics.to_csv(output_dir / "03_all_unimodal_metrics.csv", index=False)
    summary.to_csv(output_dir / "03_unimodal_model_summary.csv", index=False)
    np.savez_compressed(output_dir / "03_oof_probabilities.npz", **all_probabilities)
    return all_probabilities, cm, nm, nh, selected, summary


def evaluate_finalist_seeds(
    name: str, dev: pd.DataFrame, processed_dir: Path, classes: np.ndarray,
    *, output_dir: Path, folds: list[int], device, seeds: tuple[int, ...] = (43, 44),
    epochs: int = 10, patience: int = 3, batch_size: int = 96,
) -> pd.DataFrame:
    """Repeat one Development-selected neural model at fixed extra seeds."""
    if name not in UNIMODAL_MODELS:
        raise ValueError(f"Unexpected robustness finalist: {name}")
    all_metrics, all_histories = [], []
    for seed in seeds:
        for fold in folds:
            fitting = dev.loc[dev.validation_fold != fold].copy()
            validation = dev.loc[dev.validation_fold == fold].copy()
            _check_fold_participants(fitting, validation)
            print(f"\n{name} | extra seed {seed} | fold {int(fold)+1}/{len(folds)}", flush=True)
            model, stats, history, info = fit_neural(
                name, fitting, processed_dir, classes,
                device=device, seed=seed, epochs=epochs,
                patience=patience, batch_size=batch_size, show_progress=True,
            )
            proba = predict_neural(
                name, model, validation, processed_dir, classes, stats,
                device=device,
            )
            metric = participant_metrics(
                validation.reset_index(drop=True), proba, classes,
                model_id=name, fold=int(fold), seed=seed,
            )
            metric["fit_seconds"] = info["fit_seconds"]
            history["validation_fold"] = int(fold)
            all_metrics.append(metric)
            all_histories.append(history)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.concat(all_metrics, ignore_index=True).to_csv(
                Path(output_dir) / f"03_{name}_extra_seed_metrics.csv", index=False,
            )
            pd.concat(all_histories, ignore_index=True).to_csv(
                Path(output_dir) / f"03_{name}_extra_seed_history.csv", index=False,
            )
            print(
                f"  participant-mean macro F1 {metric.macro_f1.mean():.4f} | "
                f"fit {info['fit_seconds']:.1f} s", flush=True,
            )
    metrics = pd.concat(all_metrics, ignore_index=True)
    return metrics.groupby("seed", as_index=False).agg(
        participant_mean_macro_f1=("macro_f1", "mean"),
        fits=("validation_fold", "nunique"),
    )


def consolidate_seed_robustness(
    names: tuple[str, str], output_dir: Path,
    *, folds: list[int], seeds: tuple[int, ...] = (43, 44),
) -> pd.DataFrame:
    """Aggregate two completed finalist robustness records by model and seed."""
    rows, histories = [], []
    for name in names:
        metrics = pd.read_csv(Path(output_dir) / f"03_{name}_extra_seed_metrics.csv")
        history = pd.read_csv(Path(output_dir) / f"03_{name}_extra_seed_history.csv")
        for seed in seeds:
            subset = metrics.loc[metrics.seed == seed]
            if set(subset.validation_fold.astype(int)) != set(folds):
                raise ValueError(f"{name} seed {seed} has incomplete robustness folds.")
        rows.append(metrics)
        histories.append(history)
    combined = pd.concat(rows, ignore_index=True)
    pd.concat(histories, ignore_index=True).to_csv(
        Path(output_dir) / "03_neural_multiseed_history.csv", index=False,
    )
    combined.to_csv(Path(output_dir) / "03_neural_multiseed_metrics.csv", index=False)
    return combined.groupby(["model_id", "seed"], as_index=False).macro_f1.mean()
