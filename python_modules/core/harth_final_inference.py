# ==================== Presented by: Reza Saadatyar (2026) ====================
"""HARTH Development model fitting, serialisation and source-window inference.

This module fits the selected HARTH 1D-ResNet using Development data only,
stores inference artefacts and applies models without refitting preprocessing.
Qualified same-study evaluation metrics are handled separately by
core.harth_qualified_test_report.
"""

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.advanced_temporal_models import build_temporal_model
from core.temporal_cnn import compute_training_class_weights


# ===================== 01. Development Normalisation ========================
def fit_development_scaler(windows):
    """
    Fit a per-axis scaler on full Development windows only.

    Args:
        windows (ndarray): Float32 (N, 6, 100) development training data.

    Returns:
        dict: Float32 six-element mean/std arrays from Development only.
    """
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (6, 100) or not len(x):
        raise ValueError("Expected (nonzero Development, 6, 100) raw windows.")
    if not np.isfinite(x).all():
        raise ValueError("Development source windows must be finite.")
    mean = x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    return {"mean": mean, "std": np.maximum(std, np.float32(1e-6))}


def apply_frozen_scaler(windows, scaler):
    """
    Transform unseen windows without learning anything from them.

    Args:
        windows (ndarray): Original source (N, 6, 100) float32 IMU windows.
        scaler (dict): Development-only mean/std returned from fit.

    Returns:
        ndarray: Finite float32 standardised windows with same shape.
    """
    x = np.asarray(windows, dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (6, 100) or mean.shape != (6,) or std.shape != (6,):
        raise ValueError("Incompatible source IMU channels/window dimensions.")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Missing or nonfinite source/scaler statistics.")
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


# =================== 02. Final Development ResNet Fit ======================
def fit_final_resnet(windows, labels, protocol, *, device="cuda"):
    """
    Fit ONE predeclared final ResNet on Development, without validation/Test.

    Args:
        windows (ndarray): Training-only original six-axis signal windows.
        labels (array-like): Original development activity codes 1..8.
        protocol (dict): Selected Development fitting specification.
        device (str): CUDA recommended for final experiment, or CPU.

    Returns:
        model (nn.Module): Model after its fixed 400th optimizer step.
        scaler (dict): Development-only six-axis scaler.
        history (pd.DataFrame): Actual training losses sampled at steps.
        weights (ndarray): Development-derived inverse-sqrt class weights.
        elapsed_seconds (float): Measured training wall time.

    Notes:
        A fitted final model is not an independent result. Never select
        its checkpoint using protected Test data or tune on Test labels.
    """
    # 1. Strictly require the designated frozen Development training task.
    if torch is None:
        raise ImportError("Optional PyTorch is required for final model fitting.")
    if device not in ("cuda", "cpu") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested neural-training device unavailable.")
    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels)
    model_spec = protocol["primary_model"]
    codes = model_spec["classes"]
    if x.ndim != 3 or x.shape[1:] != (6, 100) or len(x) == 0 or len(y) != len(x):
        raise ValueError("Expected aligned, non-empty (N, 6, 100) Development windows.")
    if not np.isin(y, codes).all():
        raise ValueError("Unexpected source class in Development.")
    scaler = fit_development_scaler(x)
    scaled = apply_frozen_scaler(x, scaler)
    weights = compute_training_class_weights(y, codes, method="inverse_sqrt")

    # 2. Reinitialise every RNG and create the exact architecture and DataLoader.
    seed = model_spec["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mapping = {code: i for i, code in enumerate(codes)}
    targets = np.asarray([mapping[int(item)] for item in y], dtype=np.int64)
    training_data = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(targets))
    loader = DataLoader(
        training_data, batch_size=model_spec["batch_size"],
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )
    model = build_temporal_model(
        "1D-ResNet", in_channels=6, n_classes=8, hidden_channels=32
    ).to(device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_spec["learning_rate"])
    iterator = iter(loader)

    # 3. Perform all 400 updates; never inspect any held-out validation curve.
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logs = []
    for step in range(1, model_spec["optimizer_steps"] + 1):
        try:
            xb, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, target = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb.to(device))
        loss = nn.functional.cross_entropy(
            logits, target.to(device), weight=weight_tensor
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Nonfinite final training loss at step {step}")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == model_spec["optimizer_steps"]:
            logs.append({"step": step, "development_train_loss": float(loss.item())})
            print(f"  Final 1D-ResNet: step {step}/400 | Train loss={loss.item():.4f}")
    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - t0
    model.eval()
    return model, scaler, pd.DataFrame(logs), weights, duration


# ===================== 03. Save and Reload Model Artefacts =====================
def save_final_models(
    model, scaler, reference, columns, protocol, artifact_dir, *,
    source_revision,
):
    """Save one Development-fitted neural/classical model pair and its settings.

    The inference configuration contains feature order, channel statistics,
    source classes and architecture parameters; it is not a release gate.
    Only load locally generated and trusted joblib artefacts.
    """
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    net_path = directory / "resnet_state_dict.pt"
    rf_path = directory / "random_forest.joblib"
    config_path = directory / "inference_config.json"

    torch.save(
        {name: tensor.detach().cpu()
         for name, tensor in model.state_dict().items()},
        net_path,
    )
    joblib.dump(reference, rf_path)
    model_spec = protocol["primary_model"]
    config = {
        "source_dataset": "HARTH v2.0",
        "source_revision": str(source_revision),
        "sensor_columns": [
            "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"
        ],
        "source_codes": list(model_spec["classes"]),
        "channel_mean": np.asarray(scaler["mean"]).tolist(),
        "channel_std": np.asarray(scaler["std"]).tolist(),
        "feature_columns": list(columns),
        "resnet_architecture": model_spec["architecture"],
        "resnet_hidden_channels": 32,
        "window_size": 100,
        "window_stride": 50,
        "nominal_hz": 50,
        "preprocessing_fit_scope": "Development only",
        "test_fitted_values": False,
        "optimiser_updates": int(model_spec["optimizer_steps"]),
        "training_seed": int(model_spec["random_seed"]),
    }
    config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "resnet": net_path,
        "reference": rf_path,
        "config": config_path,
    }


def load_final_models(directory, *, device="cpu"):
    """
    Load locally trusted Development-fitted models and inference settings.

    Args:
        directory (str | Path): Directory created by save_final_models().
        device (str): CPU or available CUDA for ResNet inference.

    Returns:
        tuple: Evaluating ResNet, trusted Random Forest and exact JSON
        preprocessing/class-mapping configuration.

    Notes:
        Only load locally generated trusted joblib data. Torch uses
        weights_only=True, and both models are never refitted on input data.
    """
    if torch is None:
        raise ImportError("Optional PyTorch is required.")
    root = Path(directory)
    config = json.loads((root / "inference_config.json").read_text())
    model = build_temporal_model(
        "1D-ResNet", in_channels=6, n_classes=8,
        hidden_channels=config["resnet_hidden_channels"]
    )
    state = torch.load(
        root / "resnet_state_dict.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    reference = joblib.load(root / "random_forest.joblib")
    return model, reference, config


def infer_source_windows(windows, model, reference, config, *, device="cpu"):
    """
    Produce original eight-class code predictions for validated IMU windows.

    Args:
        windows (ndarray): New complete six-axis (N, 6, 100) source windows.
        model (nn.Module): Loaded final 1D-ResNet.
        reference (RandomForestClassifier): Loaded trained RF reference.
        config (dict): Saved Train-only scaling and exact feature-order map.
        device (str): Torch inference device.

    Returns:
        tuple[np.ndarray, np.ndarray]: ResNet and RF original class codes.

    Notes:
        No label, Test metadata, scaling fitting or Test-specific preprocessing
        adjustment occurs here. Window segmentation has already happened.
    """
    from core.feature_engineering import extract_signal_features
    if torch is None:
        raise ImportError("Optional PyTorch is required.")
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (6, 100) or len(x) == 0:
        raise ValueError("Expected non-empty (windows, six channels, 100 steps).")
    stats = {"mean": np.asarray(config["channel_mean"]),
             "std": np.asarray(config["channel_std"])}
    scaled = apply_frozen_scaler(x, stats)
    net_indices = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(scaled), 256):
            segment = torch.from_numpy(scaled[start:start+256]).to(device)
            net_indices.extend(model(segment).argmax(1).cpu().tolist())
    codes = np.asarray(config["source_codes"], dtype=int)
    predicted_resnet = codes[np.asarray(net_indices, dtype=int)]
    features = extract_signal_features(
        x, channel_names=config["sensor_columns"]
    )
    if list(features.columns) != config["feature_columns"]:
        raise ValueError("Saved feature-order mismatch.")
    predicted_rf = reference.predict(features.to_numpy())
    return predicted_resnet, np.asarray(predicted_rf, dtype=int)
