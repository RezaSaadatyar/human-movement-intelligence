# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 07: controlled Essex temporal-model robustness and native-FE diagnostics.

Designed for TRAIN-only original-trial GroupKFold. No Test/Validation labels,
model fitting, target interpolation or unseen-participant scoring in this module.
Job caches live under ignored data/processed and are bound to source fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from essex_temporal_regression import (
    TemporalCNN, DilatedTCN, check_temporal_arrays,
    gather_original_positions, peak_aware_loss, training_fold_statistics,
)


ARCHITECTURES = ("Temporal CNN", "Dilated TCN", "GRU", "BiGRU")
NO_PEAK = "Dilated TCN (no peak)"
CONFIGS = (*ARCHITECTURES, NO_PEAK)
SENSOR_SUFFIXES = ("PEL", "RTH", "LTH", "RSK", "LSK", "RFT", "LFT")
SEEDS = (2026, 2027, 2028)
FOLDS = (1, 2, 3, 4, 5)
EPOCHS = 35
BATCH_SIZE = 64
PEAK_WEIGHT = 0.25
LEARNING_RATE = 0.001
NOISE_STD_NORMALISED = 0.10


# =================== 01. Recurrence at the Original FE Grid ==================
class NativeGRU(nn.Module):
    """Two-layer GRU with readout at verified source FE sample positions."""

    def __init__(self, *, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        width = 32 if self.bidirectional else 48
        self.encoder = nn.GRU(
            input_size=42, hidden_size=width, num_layers=2,
            bidirectional=self.bidirectional, batch_first=True,
        )
        self.readout = nn.Linear(
            width * (2 if self.bidirectional else 1), 2,
        )

    def forward(self, x, indices):
        if x.ndim != 3 or x.shape[1:] != (101, 42):
            raise ValueError("Input must be (N, 101, 42).")
        sequences, _ = self.encoder(x)
        dense = self.readout(sequences).transpose(1, 2)
        return gather_original_positions(dense, indices)


def make_stage07_model(name):
    if name == "Temporal CNN":
        return TemporalCNN()
    if name in {"Dilated TCN", NO_PEAK}:
        return DilatedTCN()
    if name == "GRU":
        return NativeGRU()
    if name == "BiGRU":
        return NativeGRU(bidirectional=True)
    raise ValueError(f"Unapproved model configuration: {name}")


def config_peak_weight(name):
    if name not in CONFIGS:
        raise ValueError("Configuration is not part of the frozen Stage 07 study.")
    return 0.0 if name == NO_PEAK else PEAK_WEIGHT


def sensor_channel_groups(channel_names):
    """Six raw channels per physical sensor; no assumption of contiguous groups."""
    names = tuple(str(c) for c in channel_names)
    if len(names) != 42 or len(set(names)) != 42:
        raise ValueError("Source-ordered 42-channel schema required.")
    groups = {}
    for suffix in SENSOR_SUFFIXES:
        group = tuple(i for i, c in enumerate(names) if c.endswith("_" + suffix))
        if len(group) != 6:
            raise ValueError(f"Expected six channels for sensor {suffix}.")
        groups[suffix] = group
    if set(i for values in groups.values() for i in values) != set(range(42)):
        raise ValueError("Sensor groups do not cover original channels exactly.")
    return groups


# ====================== 02. Deterministic Fit Contract =======================
def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def _predict(model, normalised_x, positions, statistics, *, device, batch_size):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(normalised_x), batch_size):
            stop = start + batch_size
            features = torch.from_numpy(np.ascontiguousarray(
                normalised_x[start:stop], dtype=np.float32,
            )).to(device)
            indices = torch.from_numpy(np.ascontiguousarray(
                positions[start:stop], dtype=np.int64,
            )).to(device)
            result.append(model(features, indices).cpu().numpy())
    prediction = np.concatenate(result, axis=0).astype(np.float64)
    prediction = prediction * statistics.target_std + statistics.target_mean
    if prediction.shape != (len(normalised_x), 11, 2):
        raise ValueError("Source FE prediction grid was not preserved.")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite FE predictions.")
    return prediction


def train_stage07_fold(
    config, x_fit, y_fit, pos_fit, x_held, pos_held,
    *, seed, device, sensor_groups=None, sensitivity=False,
):
    """Same 35 epochs, optimiser, scaler and budget for all configurations."""
    if config not in CONFIGS or seed not in SEEDS:
        raise ValueError("Unregistered model configuration or seed.")
    fit_x, fit_idx = check_temporal_arrays(x_fit, pos_fit, y_fit)
    held_x, held_idx = check_temporal_arrays(x_held, pos_held)
    truth = np.asarray(y_fit, dtype=np.float64)
    _seed_everything(seed)
    stats = training_fold_statistics(fit_x, truth)
    train_x = ((fit_x - stats.input_mean) / stats.input_std).astype("float32")
    train_y = ((truth - stats.target_mean) / stats.target_std).astype("float32")
    eval_x = ((held_x - stats.input_mean) / stats.input_std).astype("float32")
    batch_dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(train_x)),
        torch.from_numpy(np.ascontiguousarray(train_y)),
        torch.from_numpy(fit_idx.copy()),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        batch_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0,
    )
    dev = torch.device(device)
    model = make_stage07_model(config).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4,
    )
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x_batch, y_batch, i_batch in loader:
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            i_batch = i_batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            p = model(x_batch, i_batch)
            loss = peak_aware_loss(
                p, y_batch, config_peak_weight(config),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite temporal objective.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x_batch)
        history.append(total / len(train_x))
    prediction = _predict(
        model, eval_x, held_idx, stats, device=dev, batch_size=BATCH_SIZE,
    )
    perturbed = {}
    if sensitivity:
        if config != "Dilated TCN" or seed != SEEDS[0]:
            raise ValueError("Sensitivity restricted to predeclared TCN/seed.")
        if sensor_groups is None or set(sensor_groups) != set(SENSOR_SUFFIXES):
            raise ValueError("Complete original physical sensor schema required.")
        for sensor, channels in sensor_groups.items():
            variant = eval_x.copy()
            variant[:, :, list(channels)] = 0.0
            # Zero in fold-normalised space = that fold's Training mean.
            perturbed[f"sensor_mean_replace_{sensor}"] = _predict(
                model, variant, held_idx, stats, device=dev,
                batch_size=BATCH_SIZE,
            )
        rng = np.random.default_rng(seed + 910)
        variant = eval_x + rng.normal(
            0, NOISE_STD_NORMALISED, size=eval_x.shape,
        ).astype(np.float32)
        perturbed["gaussian_noise_normalised_0p10"] = _predict(
            model, variant, held_idx, stats, device=dev,
            batch_size=BATCH_SIZE,
        )
    return {
        "prediction": prediction,
        "history": np.asarray(history, dtype=np.float64),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "perturbed": perturbed,
        "fit_trials": len(train_x),
    }


# ======================== 03. Fingerprinted Job Caches ======================
def stable_fingerprint(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _job_path(folder, config, seed, fold, fingerprint):
    slug = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
    return Path(folder) / (
        f"{slug}__s{seed}__f{fold}__{fingerprint[:16]}.npz"
    )


def run_cached_fold(
    cache_dir, *, config, seed, fold, x, y, positions, fit_rows, held_rows,
    record_keys, participant_ids, provenance, device, sensor_groups=None,
    sensitivity=False,
):
    """Return verified, resumable OOF prediction; never cache fit model weights.

    Fingerprint binds the job, source SHA256, fixed folds, code hash and
    hyperparameters. Mismatched files are never reused or overwritten.
    """
    if config not in CONFIGS or seed not in SEEDS or fold not in FOLDS:
        raise ValueError("Job differs from the predeclared research matrix.")
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    held_rows = np.asarray(held_rows, dtype=np.int64)
    ids = np.asarray(participant_ids, dtype=str)
    keys = np.asarray(record_keys, dtype=str)
    if not len(fit_rows) or not len(held_rows) or len(ids) != len(x):
        raise ValueError("Invalid trial or participant registry.")
    if np.intersect1d(ids[fit_rows], ids[held_rows]).size:
        raise ValueError("Participant leakage within a Training fold.")
    if np.intersect1d(keys[fit_rows], keys[held_rows]).size:
        raise ValueError("Trial identity leakage across Training fold.")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicated source trial keys.")
    payload = {
        **dict(provenance),
        "config": config, "seed": int(seed), "fold": int(fold),
        "peak_weight": config_peak_weight(config),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "run_device": str(device),
        "fit_keys_hash": stable_fingerprint(keys[fit_rows].tolist()),
        "held_keys_hash": stable_fingerprint(keys[held_rows].tolist()),
        "sensitivity": bool(sensitivity),
        "noise_std_normalised": NOISE_STD_NORMALISED,
    }
    fingerprint = stable_fingerprint(payload)
    cache_file = _job_path(cache_dir, config, seed, fold, fingerprint)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    expected_conditions = [
        *(f"sensor_mean_replace_{s}" for s in SENSOR_SUFFIXES),
        "gaussian_noise_normalised_0p10",
    ] if sensitivity else []
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                raise RuntimeError("Saved experiment fingerprint mismatch.")
            if data["held_keys"].astype(str).tolist() != keys[held_rows].tolist():
                raise RuntimeError("Cached prediction/record alignment changed.")
            pred = data["prediction"].copy()
            hist = data["history"].copy()
            params = int(data["parameter_count"])
            nfit = int(data["fit_trials"])
            cond = data["conditions"].astype(str).tolist()
            perturb = data["perturbations"].copy()
        if cond != expected_conditions:
            raise RuntimeError("Cached perturbation protocol differs.")
        status = "REUSED"
    else:
        fitted = train_stage07_fold(
            config, x[fit_rows], y[fit_rows], positions[fit_rows],
            x[held_rows], positions[held_rows],
            seed=seed, device=device,
            sensor_groups=sensor_groups, sensitivity=sensitivity,
        )
        pred = fitted["prediction"]
        hist = fitted["history"]
        params = int(fitted["parameter_count"])
        nfit = int(fitted["fit_trials"])
        cond = expected_conditions
        perturb = np.stack(
            [fitted["perturbed"][c] for c in cond], axis=0,
        ) if cond else np.empty((0, len(held_rows), 11, 2), dtype=np.float64)
        temp = cache_file.with_suffix(".partial")
        try:
            with temp.open("wb") as handle:
                np.savez_compressed(
                    handle, fingerprint=np.asarray(fingerprint),
                    held_keys=keys[held_rows],
                    prediction=pred, history=hist,
                    parameter_count=np.asarray(params),
                    fit_trials=np.asarray(nfit),
                    conditions=np.asarray(cond, dtype=str),
                    perturbations=perturb,
                )
            os.replace(temp, cache_file)
        finally:
            if temp.exists():
                temp.unlink()
        status = "TRAINED"
    if pred.shape != (len(held_rows), 11, 2):
        raise RuntimeError("OOF prediction shape changed.")
    if hist.shape != (EPOCHS,) or not np.isfinite(hist).all():
        raise RuntimeError("Incomplete or nonfinite 35-epoch loss history.")
    if perturb.shape != (len(cond), len(held_rows), 11, 2):
        raise RuntimeError("Sensitivity prediction shape changed.")
    if not np.isfinite(pred).all() or not np.isfinite(perturb).all():
        raise RuntimeError("Nonfinite saved predictions.")
    return {
        "prediction": pred, "history": hist, "perturbed": dict(zip(cond, perturb)),
        "status": status, "fingerprint": fingerprint, "cache_file": cache_file,
        "parameters": params, "fit_trials": nfit,
    }


# ================== 04. Paired Participant-Level Inference ==================
def paired_subject_bootstrap(a, b, *, draws=2000, seed=2026):
    """Exploratory CI of (model A - model B), resampling whole participants."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        raise ValueError("Paired per-participant score vectors are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite.")
    diff = a - b
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(diff), size=(int(draws), len(diff)))
    means = diff[sample].mean(axis=1)
    return {
        "difference_mean": float(diff.mean()),
        "bootstrap_low_95": float(np.quantile(means, 0.025)),
        "bootstrap_high_95": float(np.quantile(means, 0.975)),
        "bootstrap_participants": len(diff),
        "bootstrap_draws": int(draws),
    }
