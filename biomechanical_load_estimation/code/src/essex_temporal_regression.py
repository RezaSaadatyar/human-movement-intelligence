# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Source-time-aligned temporal regression for the Essex IMU/FE study.

Only TRAIN or fold-training observations may enter fit_temporal_model().
Original FE positions are indexed, never interpolated. No Test scoring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


MODEL_NAMES = ("Temporal CNN", "Dilated TCN")
FE_POINTS = 11
IMU_SAMPLES = 101
CHANNELS = 42
TARGETS = 2


# ===================== 01. Source-Time Output Alignment ======================
def check_temporal_arrays(x, positions, targets=None):
    """Reject invalid source grids or interpolated/substituted FE geometry."""
    array = np.asarray(x, dtype=np.float64)
    idx = np.asarray(positions)
    if array.ndim != 3 or array.shape[1:] != (IMU_SAMPLES, CHANNELS):
        raise ValueError("IMU features must have shape (N, 101, 42).")
    if idx.shape != (len(array), FE_POINTS):
        raise ValueError("FE index array must have shape (N, 11).")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("IMU data must be finite and non-empty.")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("FE coordinate indices must be integers.")
    if not np.all((idx >= 0) & (idx < IMU_SAMPLES)):
        raise ValueError("FE indices exceed the native IMU grid.")
    if not np.all(np.diff(idx, axis=1) > 0):
        raise ValueError("Native FE indices must increase strictly.")
    if targets is not None:
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (len(array), FE_POINTS, TARGETS):
            raise ValueError("Original FE targets must have shape (N, 11, 2).")
        if not np.isfinite(y).all():
            raise ValueError("Original FE target values must be finite.")
    return array, idx.astype(np.int64, copy=False)


def gather_original_positions(features, native_indices):
    """Gather model features at original, verified FE sample indices."""
    if features.ndim != 3 or native_indices.ndim != 2:
        raise ValueError("Expected (N, features, time) and (N, FE samples).")
    if features.size(0) != native_indices.size(0):
        raise ValueError("Trial dimensions do not agree.")
    gather_idx = native_indices.long().unsqueeze(1).expand(
        -1, features.size(1), -1
    )
    return torch.gather(features, dim=2, index=gather_idx).transpose(1, 2)


# =================== 02. Comparable Temporal Architectures ===================
class TemporalCNN(nn.Module):
    """Two local convolutions followed by native FE-position readout."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(CHANNELS, 48, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(48, TARGETS, kernel_size=1),
        )

    def forward(self, signal, indices):
        x = self.encoder(signal.transpose(1, 2))
        return gather_original_positions(x, indices)


class ResidualTemporalBlock(nn.Module):
    """Two residual dilated temporal convolutions with fixed sequence length."""

    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                width, width, kernel_size=3, padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class DilatedTCN(nn.Module):
    """Residual temporal network; no FE target interpolation or upsampling."""

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(CHANNELS, 48, kernel_size=5, padding=2), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(48, 1),
            ResidualTemporalBlock(48, 2),
            ResidualTemporalBlock(48, 4),
        )
        self.readout = nn.Conv1d(48, TARGETS, kernel_size=1)

    def forward(self, signal, indices):
        x = self.input_layer(signal.transpose(1, 2))
        x = self.readout(self.blocks(x))
        return gather_original_positions(x, indices)


def make_temporal_model(name):
    if name == "Temporal CNN":
        return TemporalCNN()
    if name == "Dilated TCN":
        return DilatedTCN()
    raise ValueError(f"Unexpected temporal architecture: {name}")


# ========================= 03. Training-Only Scaling =========================
@dataclass(frozen=True)
class FoldStatistics:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def training_fold_statistics(x_fit, y_fit):
    """Estimate independent channel and target moments using fit rows only."""
    mean_x = x_fit.mean(axis=(0, 1), keepdims=True)
    std_x = x_fit.std(axis=(0, 1), keepdims=True)
    mean_y = y_fit.mean(axis=(0, 1), keepdims=True)
    std_y = y_fit.std(axis=(0, 1), keepdims=True)
    if not all(np.isfinite(v).all() for v in (
        mean_x, std_x, mean_y, std_y
    )):
        raise ValueError("Nonfinite training-only scaler statistics.")
    return FoldStatistics(
        input_mean=mean_x,
        input_std=np.where(std_x > 1e-10, std_x, 1.0),
        target_mean=mean_y,
        target_std=np.where(std_y > 1e-10, std_y, 1.0),
    )


def peak_aware_loss(pred, target, weight=0.25):
    """Same predeclared normalised loss for both temporal architectures."""
    waveform = F.smooth_l1_loss(pred, target, beta=0.5)
    peak_e1 = torch.abs(
        pred[:, :, 0].max(dim=1).values -
        target[:, :, 0].max(dim=1).values
    ).mean()
    peak_e3 = torch.abs(
        pred[:, :, 1].min(dim=1).values -
        target[:, :, 1].min(dim=1).values
    ).mean()
    return waveform + float(weight) * 0.5 * (peak_e1 + peak_e3)


# ===================== 04. Bounded Fitting and Inference =====================
def fit_temporal_model(
    name, x_fit, y_fit, positions_fit, x_eval, positions_eval,
    *, seed=2026, epochs=35, batch_size=64, learning_rate=1e-3,
    peak_weight=0.25, device="cpu",
):
    """Fit on explicit Training rows and predict separate OOF/Validation rows.

    No early stopping, hyperparameter search or access to held-out Test rows.
    Returns original-unit predictions, actual epoch loss, and training facts.
    """
    x_fit, positions_fit = check_temporal_arrays(
        x_fit, positions_fit, y_fit
    )
    x_eval, positions_eval = check_temporal_arrays(x_eval, positions_eval)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    if int(epochs) != epochs or epochs < 1 or batch_size < 1:
        raise ValueError("Training budget must be positive.")
    if not 0 <= peak_weight <= 1:
        raise ValueError("Peak loss weight must lie between zero and one.")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    fold = training_fold_statistics(x_fit, y_fit)
    train_x = ((x_fit - fold.input_mean) / fold.input_std).astype("float32")
    train_y = ((y_fit - fold.target_mean) / fold.target_std).astype("float32")
    eval_x = ((x_eval - fold.input_mean) / fold.input_std).astype("float32")
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(train_x)),
        torch.from_numpy(np.ascontiguousarray(train_y)),
        torch.from_numpy(positions_fit.copy()),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True,
        generator=generator, num_workers=0,
    )
    device = torch.device(device)
    model = make_temporal_model(name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    rows = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y, batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_idx = batch_idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimated = model(batch_x, batch_idx)
            loss = peak_aware_loss(estimated, batch_y, peak_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite temporal training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_x)
            seen += len(batch_x)
        rows.append({"epoch": epoch, "training_loss": total / seen})

    model.eval()
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(eval_x), int(batch_size)):
            end = start + int(batch_size)
            batch_x = torch.from_numpy(
                np.ascontiguousarray(eval_x[start:end])
            ).to(device)
            batch_idx = torch.from_numpy(
                np.ascontiguousarray(positions_eval[start:end])
            ).to(device)
            predicted.append(model(batch_x, batch_idx).cpu().numpy())
    normalised = np.concatenate(predicted, axis=0)
    estimate = normalised * fold.target_std + fold.target_mean
    if estimate.shape != (len(x_eval), FE_POINTS, TARGETS):
        raise ValueError("Native-time FE prediction shape mismatch.")
    if not np.isfinite(estimate).all():
        raise FloatingPointError("Nonfinite native FE predictions.")
    details = {
        "model": name, "seed": int(seed),
        "epochs": int(epochs), "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "peak_weight": float(peak_weight),
        "fit_trials": int(len(x_fit)), "predict_trials": int(len(x_eval)),
        "trainable_parameters": int(parameter_count),
        "scaling": "fit_rows_only",
        "device": device.type,
    }
    return estimate.astype(np.float64), pd.DataFrame(rows), details
