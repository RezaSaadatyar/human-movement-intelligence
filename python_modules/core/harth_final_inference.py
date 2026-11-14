# ==================== Presented by: Reza Saadatyar (2026) ====================
"""HARTH Development model fitting, serialisation and source-window inference.

This module fits the selected HARTH 1D-ResNet using Development data only,
stores inference artefacts and applies models without refitting preprocessing.
Qualified same-study evaluation metrics are handled separately by
core.harth_qualified_test_report.
"""

import csv
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
# -------------------- 1. Record the Inference CSV Schema --------------------
_SCALAR_STRING_FIELDS = (
    "source_dataset", "source_revision", "resnet_architecture",
    "preprocessing_fit_scope",
)
_SCALAR_INTEGER_FIELDS = (
    "resnet_hidden_channels", "window_size", "window_stride",
    "nominal_hz", "optimiser_updates", "training_seed",
)
_VECTOR_STRING_FIELDS = ("sensor_columns", "feature_columns")
_VECTOR_INTEGER_FIELDS = ("source_codes",)
_VECTOR_FLOAT_FIELDS = ("channel_mean", "channel_std")


def _write_inference_csv(config, path):
    """Write typed scalar and ordered-array settings in a simple CSV table."""
    rows = []
    for key in (*_SCALAR_STRING_FIELDS, *_SCALAR_INTEGER_FIELDS,
                "test_fitted_values"):
        rows.append((key, "", str(config[key])))
    for key in (*_VECTOR_STRING_FIELDS, *_VECTOR_INTEGER_FIELDS,
                *_VECTOR_FLOAT_FIELDS):
        for index, value in enumerate(config[key]):
            if key in _VECTOR_FLOAT_FIELDS:
                value = repr(float(value))
            rows.append((key, index, str(value)))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "position", "value"])
        writer.writerows(rows)


def _read_inference_csv(path):
    """Restore the exact configured scalar types and ordered sensor vectors."""
    scalars, vectors = {}, {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["parameter", "position", "value"]:
            raise ValueError("Inference CSV must have parameter,position,value.")
        for row in reader:
            key, index, value = (
                row["parameter"], row["position"], row["value"]
            )
            if not key or value is None:
                raise ValueError("Incomplete inference configuration row.")
            if index == "":
                if key in scalars or key in vectors:
                    raise ValueError("Duplicate inference configuration key.")
                scalars[key] = value
            else:
                try:
                    position = int(index)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid configuration array position.") from exc
                if position < 0 or key in scalars:
                    raise ValueError("Invalid configuration array position.")
                entries = vectors.setdefault(key, {})
                if position in entries:
                    raise ValueError("Duplicate configuration array position.")
                entries[position] = value

    scalar_keys = set((*_SCALAR_STRING_FIELDS, *_SCALAR_INTEGER_FIELDS,
                       "test_fitted_values"))
    vector_keys = set((*_VECTOR_STRING_FIELDS, *_VECTOR_INTEGER_FIELDS,
                       *_VECTOR_FLOAT_FIELDS))
    if set(scalars) != scalar_keys or set(vectors) != vector_keys:
        raise ValueError("Incomplete or unexpected inference configuration fields.")
    config = {key: scalars[key] for key in _SCALAR_STRING_FIELDS}
    for key in _SCALAR_INTEGER_FIELDS:
        config[key] = int(scalars[key])
    if scalars["test_fitted_values"] not in ("True", "False"):
        raise ValueError("Invalid boolean inference configuration value.")
    config["test_fitted_values"] = (
        scalars["test_fitted_values"] == "True"
    )
    for key in vector_keys:
        values = vectors[key]
        if sorted(values) != list(range(len(values))):
            raise ValueError("Configuration array positions must be contiguous.")
        ordered = [values[i] for i in range(len(values))]
        if key in _VECTOR_INTEGER_FIELDS:
            ordered = [int(value) for value in ordered]
        elif key in _VECTOR_FLOAT_FIELDS:
            ordered = [float(value) for value in ordered]
        config[key] = ordered

    expected_channels = [
        "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"
    ]
    if (config["sensor_columns"] != expected_channels
            or config["source_codes"] != list(range(1, 9))
            or len(config["channel_mean"]) != 6
            or len(config["channel_std"]) != 6
            or len(config["feature_columns"]) != 30
            or len(set(config["feature_columns"])) != 30):
        raise ValueError("Incompatible HARTH source or feature configuration.")
    if (not np.isfinite(config["channel_mean"]).all()
            or not np.isfinite(config["channel_std"]).all()
            or any(std <= 0 for std in config["channel_std"])):
        raise ValueError("Invalid Development-fitted channel statistics.")
    if (config["resnet_architecture"] != "1D-ResNet"
            or config["preprocessing_fit_scope"] != "Development only"
            or config["test_fitted_values"]
            or config["window_size"] != 100
            or config["window_stride"] != 50):
        raise ValueError("Unsupported model or preprocessing configuration.")
    return config


# ------------------------- 2. Save Trained Model Pair ------------------------
def save_final_models(
    model, scaler, reference, columns, protocol, artifact_dir, *,
    source_revision,
):
    """Save fitted ResNet, Random Forest and ordered inference settings as CSV.

    The CSV contains all values needed to apply the Development-only scaler
    and feature ordering. These are model inference settings, not an audit
    manifest or historical approval record. Load only trusted joblib files.
    """
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    net_path = directory / "resnet_state_dict.pt"
    rf_path = directory / "random_forest.joblib"
    config_path = directory / "inference_config.csv"

    torch.save(
        {name: tensor.detach().cpu()
         for name, tensor in model.state_dict().items()},
        net_path,
    )
    joblib.dump(reference, rf_path)
    spec = protocol["primary_model"]
    config = {
        "source_dataset": "HARTH v2.0",
        "source_revision": str(source_revision),
        "sensor_columns": [
            "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"
        ],
        "source_codes": list(spec["classes"]),
        "channel_mean": np.asarray(scaler["mean"]).tolist(),
        "channel_std": np.asarray(scaler["std"]).tolist(),
        "feature_columns": list(columns),
        "resnet_architecture": spec["architecture"],
        "resnet_hidden_channels": 32,
        "window_size": 100,
        "window_stride": 50,
        "nominal_hz": 50,
        "preprocessing_fit_scope": "Development only",
        "test_fitted_values": False,
        "optimiser_updates": int(spec["optimizer_steps"]),
        "training_seed": int(spec["random_seed"]),
    }
    _write_inference_csv(config, config_path)
    (directory / "inference_config.json").unlink(missing_ok=True)
    return {
        "resnet": net_path,
        "reference": rf_path,
        "config": config_path,
    }


# ---------------------- 3. Load Trusted Model Artefacts ----------------------
def load_final_models(directory, *, device="cpu"):
    """Load trusted Development-fitted models and their typed CSV settings.

    The torch weights are loaded with weights_only=True. The joblib file
    must come from a trusted source, because it deserialises Python objects.
    """
    root = Path(directory)
    config = _read_inference_csv(root / "inference_config.csv")
    model = build_temporal_model(
        config["resnet_architecture"], in_channels=6, n_classes=8,
        hidden_channels=config["resnet_hidden_channels"],
    )
    state = torch.load(
        root / "resnet_state_dict.pt",
        map_location="cpu", weights_only=True,
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
