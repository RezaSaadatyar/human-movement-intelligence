# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Fold-safe neural fitting, fixed-epoch final refits and inference costs."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ninapro_dataset import PairedWindowDataset, training_channel_stats
from ninapro_evaluation import participant_metrics
from ninapro_models import build_neural_model, count_parameters


# ========================== 01. Reproducible Fitting ==========================
def set_training_seed(seed: int):
    import random
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_logits(name: str, model, emg, acc):
    if "_emg_" in name:
        return model(emg)
    if "_acc_" in name:
        return model(acc)
    return model(emg, acc)


def _loader(table, processed_dir, classes, stats, *, batch_size, shuffle):
    dataset = PairedWindowDataset(table, processed_dir, classes=classes, **stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def fit_neural(
    model_name: str, train_table: pd.DataFrame, processed_dir,
    classes: np.ndarray, *, device, seed: int = 42, epochs: int = 10,
    patience: int = 3, batch_size: int = 96, learning_rate: float = 1e-3,
    fixed_epochs: int | None = None, show_progress: bool = False,
):
    """Train on permitted subjects only; use a Training-only inner holdout.

    With fixed_epochs, train all supplied participants without early stopping.
    With no fixed_epochs, hold out 15% of supplied *training people* for
    checkpoint selection; neither outer Validation nor Test is consulted.
    """
    seed = int(seed)
    set_training_seed(seed)
    train_table = train_table.reset_index(drop=True)
    if train_table.empty:
        raise ValueError("Neural training needs at least one training window.")
    if fixed_epochs is None:
        people = np.random.default_rng(seed).permutation(sorted(train_table.subject_id.unique()))
        if len(people) < 3:
            raise ValueError("Inner stopping requires at least three Training participants.")
        holdout_count = max(1, int(round(len(people) * .15)))
        inner_ids = set(people[:holdout_count])
        inner = train_table.loc[train_table.subject_id.isin(inner_ids)].reset_index(drop=True)
        fitting = train_table.loc[~train_table.subject_id.isin(inner_ids)].reset_index(drop=True)
    else:
        inner = None
        fitting = train_table
    stats = training_channel_stats(fitting, processed_dir)
    train_loader = _loader(fitting, processed_dir, classes, stats,
                           batch_size=batch_size, shuffle=True)
    val_loader = None if inner is None else _loader(
        inner, processed_dir, classes, stats, batch_size=batch_size, shuffle=False)
    model = build_neural_model(model_name, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-3)
    codes = {int(v): i for i, v in enumerate(classes)}
    counts = np.bincount([codes[int(v)] for v in fitting.label], minlength=len(classes))
    weight = np.zeros(len(classes), np.float32)
    present = counts > 0
    weight[present] = len(fitting) / (counts[present] * max(1, present.sum()))
    criterion_weight = torch.from_numpy(weight).to(device)
    best_score, best_state, best_epoch, misses = -np.inf, None, 0, 0
    history = []
    start = perf_counter()
    n_epochs = int(fixed_epochs if fixed_epochs is not None else epochs)
    for epoch in range(1, n_epochs + 1):
        epoch_started = perf_counter()
        model.train()
        losses = []
        for emg, acc, label in train_loader:
            emg, acc, label = emg.to(device), acc.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model_name, model, emg, acc)
            loss = F.cross_entropy(logits, label, weight=criterion_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_f1 = np.nan
        if val_loader is not None:
            probs = _predict_loader(model_name, model, val_loader, device)
            val_f1 = float(participant_metrics(
                inner.reset_index(drop=True), probs, classes,
                model_id=model_name, seed=seed,
            ).macro_f1.mean())
            if val_f1 > best_score + 1e-6:
                best_score, best_epoch, misses = val_f1, epoch, 0
                best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                misses += 1
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "inner_validation_macro_f1": val_f1, "model_id": model_name,
                        "seed": seed})
        if show_progress:
            validation_text = (
                f"{val_f1:.4f}" if np.isfinite(val_f1) else "not evaluated"
            )
            print(
                f"  {model_name} | seed={seed} | epoch {epoch}/{n_epochs} "
                f"| loss={epoch_loss:.4f} | inner macro F1={validation_text} "
                f"| {perf_counter() - epoch_started:.1f} s",
                flush=True,
            )
        if val_loader is not None and misses >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    fit_seconds = perf_counter() - start
    return model, stats, pd.DataFrame(history), {
        "model_id": model_name, "seed": seed, "best_epochs": int(best_epoch),
        "fit_seconds": float(fit_seconds),
        "parameter_count": count_parameters(model),
        "inner_validation_people": 0 if inner is None else inner.subject_id.nunique(),
        "device": str(device),
    }


# ========================== 02. Paired Probability Inference =================
@torch.inference_mode()
def _predict_loader(name, model, loader, device, *, perturbation: str | None = None):
    model.eval()
    probabilities = []
    for emg, acc, _ in loader:
        emg, acc = emg.to(device), acc.to(device)
        if perturbation == "acc_missing_mean":
            acc = torch.zeros_like(acc)
        elif perturbation == "acc_sensor_01_missing_mean":
            acc = acc.clone()
            acc[:, :3, :] = 0.0
        elif perturbation == "acc_contiguous_gap_mean":
            acc = acc.clone()
            gap = max(1, acc.shape[-1] // 4)
            acc[:, :, :gap] = 0.0
        logits = _model_logits(name, model, emg, acc)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 0), float)


def predict_neural(name, model, table, processed_dir, classes, stats, *,
                   device, batch_size=128, perturbation=None):
    loader = _loader(table.reset_index(drop=True), processed_dir, classes, stats,
                     batch_size=batch_size, shuffle=False)
    return _predict_loader(name, model, loader, device, perturbation=perturbation)


def benchmark_cpu(name, model, table, processed_dir, classes, stats, *, windows=32):
    """Measure CPU inference after model transfer, excluding data initialisation."""
    if table.empty:
        return float("nan")
    cpu_model = deepcopy(model).to("cpu").eval()
    subset = table.iloc[:min(windows, len(table))].reset_index(drop=True)
    loader = _loader(subset, processed_dir, classes, stats,
                     batch_size=min(windows, len(table)), shuffle=False)
    with torch.inference_mode():
        emg, acc, _ = next(iter(loader))
        for _ in range(2):
            _model_logits(name, cpu_model, emg, acc)
        times = []
        for _ in range(5):
            started = perf_counter()
            _model_logits(name, cpu_model, emg, acc)
            times.append(perf_counter() - started)
    return 1000 * float(np.median(times)) / len(subset)
