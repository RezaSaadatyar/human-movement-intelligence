# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 08: final evaluation using the existing RF and TCN model recipes.

The final model and recipe are selected by Stage 07 Training-only evidence.
Fit uses only the predetermined TRAIN+VALIDATION development pool.
Executing the final notebook replaces its current outputs without release tokens.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor

try:
    from .essex_regression_baselines import (
        trial_features,
        flatten_targets,
    )
    from .essex_temporal_regression import (
        check_temporal_arrays,
        make_temporal_model,
        training_fold_statistics,
        peak_aware_loss,
    )
except ImportError:  # Colab source-path installation
    from essex_regression_baselines import (
        trial_features,
        flatten_targets,
    )
    from essex_temporal_regression import (
        check_temporal_arrays,
        make_temporal_model,
        training_fold_statistics,
        peak_aware_loss,
    )


EXPECTED_PRIMARY = "Random Forest (Stage 05 locked)"
SECONDARY = "Dilated TCN"


def fit_frozen_primary(x_development, y_development, channels, protocol):
    """Deterministic full-development fit, never receives Test data."""
    rf = protocol["primary_model"]
    features, names = trial_features(x_development, channels)
    targets = flatten_targets(y_development)
    if features.shape != (len(x_development), 210) or targets.shape != (
        len(x_development),
        22,
    ):
        raise ValueError("Frozen RF development shape mismatch.")
    model = RandomForestRegressor(
        n_estimators=rf["n_estimators"],
        min_samples_leaf=rf["min_samples_leaf"],
        max_features=rf["max_features"],
        random_state=rf["random_state"],
        n_jobs=-1,
    )
    model.fit(features, targets)
    if model.n_outputs_ != 22:
        raise RuntimeError("Fitted Random Forest does not match protocol.")
    return model, names


def predict_frozen_primary(model, x, channels, expected_names):
    features, names = trial_features(x, channels)
    if names != list(expected_names):
        raise ValueError("Frozen source channel/feature ordering changed.")
    result = np.asarray(model.predict(features), dtype=np.float64)
    if result.shape != (len(x), 22) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen primary prediction.")
    return result.reshape(-1, 11, 2)


def fit_frozen_secondary(x, y, pos, protocol, *, device):
    """One full-development TCN fit, no Test inputs or hyperparameter search."""
    source_x, original_pos = check_temporal_arrays(x, pos, y)
    cfg = protocol["secondary_model"]
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    scaler = training_fold_statistics(source_x, y)
    sx = ((source_x - scaler.input_mean) / scaler.input_std).astype(np.float32)
    sy = ((np.asarray(y) - scaler.target_mean) / scaler.target_std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(sx)),
        torch.from_numpy(np.ascontiguousarray(sy)),
        torch.from_numpy(original_pos.copy()),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    dev = torch.device(device)
    model = make_temporal_model(SECONDARY).to(dev)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    losses = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total = 0.0
        for signals, labels, positions in loader:
            signals, labels, positions = (
                signals.to(dev),
                labels.to(dev),
                positions.to(dev),
            )
            optimiser.zero_grad(set_to_none=True)
            prediction = model(signals, positions)
            loss = peak_aware_loss(prediction, labels, float(cfg["peak_weight"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frozen TCN training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach().cpu()) * len(signals)
        losses.append(
            {
                "epoch": epoch,
                "training_loss": total / len(dataset),
                "model": SECONDARY,
                "split": "FROZEN_TRAIN_PLUS_VALIDATION",
            }
        )
    model.eval()
    if not np.isfinite([x["training_loss"] for x in losses]).all():
        raise RuntimeError("Incomplete frozen TCN training.")
    return model, scaler, pd.DataFrame(losses)


def predict_frozen_secondary(model, scaler, x, pos, *, device, batch=64):
    """Read native FE positions using the already frozen development scaler."""
    original_x, original_pos = check_temporal_arrays(x, pos)
    features = ((original_x - scaler.input_mean) / scaler.input_std).astype(np.float32)
    dev = torch.device(device)
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch)):
            last = start + int(batch)
            bx = torch.from_numpy(np.ascontiguousarray(features[start:last])).to(dev)
            bi = torch.from_numpy(np.ascontiguousarray(original_pos[start:last])).to(
                dev
            )
            out.append(model(bx, bi).cpu().numpy())
    pred = np.concatenate(out, axis=0).astype(np.float64)
    pred = pred * scaler.target_std + scaler.target_mean
    if pred.shape != (len(x), 11, 2) or not np.isfinite(pred).all():
        raise ValueError("Frozen TCN output invalid.")
    return pred


def participant_bootstrap(y, prediction, subjects, *, draws=2000, seed=2026):
    """Exploratory confidence interval resampling whole participants."""
    ids = np.asarray(subjects, dtype=str)
    if len(ids) != len(y):
        raise ValueError("Source trial identities do not match observations.")
    people = sorted(set(ids))
    if len(people) < 2:
        raise ValueError("Participant bootstrap requires at least two participants.")
    values = np.array(
        [
            np.abs(y[ids == person] - prediction[ids == person]).mean()
            for person in people
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(people), size=(int(draws), len(people)))
    boot = values[sampled].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "bootstrap_low_95": float(np.quantile(boot, 0.025)),
        "bootstrap_high_95": float(np.quantile(boot, 0.975)),
        "participants": len(people),
        "resamples": int(draws),
        "scope": "exploratory_same_source_test",
    }


def paired_test_bootstrap(y, primary, secondary, subjects, *, draws=2000, seed=2026):
    ids = np.asarray(subjects, dtype=str)
    people = sorted(set(ids))
    if len(ids) != len(y) or len(people) < 2:
        raise ValueError("Aligned trials from at least two participants are required.")
    delta = np.array(
        [
            np.abs(y[ids == p] - secondary[ids == p]).mean()
            - np.abs(y[ids == p] - primary[ids == p]).mean()
            for p in people
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    resampled = delta[rng.integers(0, len(people), (int(draws), len(people)))].mean(
        axis=1
    )
    return {
        "comparison": "secondary_minus_primary",
        "difference": float(delta.mean()),
        "bootstrap_low_95": float(np.quantile(resampled, 0.025)),
        "bootstrap_high_95": float(np.quantile(resampled, 0.975)),
        "participants": len(people),
        "resamples": int(draws),
        "scope": "exploratory_not_confirmatory",
    }


def final_model_settings():
    """Return a fresh copy of the original final-model hyperparameters."""
    return {
        "primary_model": {
            "name": "Random Forest",
            "n_estimators": 200,
            "min_samples_leaf": 3,
            "max_features": 1,
            "random_state": 2026,
            "feature_statistics": ["mean", "std", "minimum", "maximum", "rms"],
            "channels": 42,
            "feature_count": 210,
            "output_shape": [11, 2],
        },
        "secondary_model": {
            "name": "Dilated TCN",
            "role": "predeclared descriptive peak-tradeoff comparator only; never used to replace primary after test",
            "seed": 2128,
            "epochs": 35,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "peak_weight": 0.25,
            "fold_fit_only_scale": True,
            "source_native_fe_indices": True,
            "output_shape": [11, 2],
        },
    }
