# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Approved-protocol-gated HARTH final fitting and reproducible inference.

Training sees Development ONLY. Qualified Test loading is a separate operation
that requires an explicitly approved frozen source-fingerprinted protocol.
Never selects architecture, random seed, stopping point or thresholds on Test.
"""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from core.advanced_temporal_models import build_temporal_model
    from core.temporal_cnn import compute_training_class_weights
except ImportError:
    torch = None


# ==================== 1. Locked Reviewable Protocol ====================
REQUIRED_APPROVAL = "APPROVED"
TEST_CONFIRMATION = "I_APPROVE_QUALIFIED_TEST_EVALUATION"


def sha256_file(path):
    """
    Hash an exact on-disk source/artifact file.

    Args:
        path (str | pathlib.Path): Existing file to fingerprint.

    Returns:
        str: SHA256 hex digest of its original bytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_final_protocol(path, repository_root, *, require_approval=False):
    """
    Validate original Stage 25 protocol and its recorded code fingerprints.

    Args:
        path (str | Path): Existing stage_25_protocol_proposal.json.
        repository_root (str | Path): Exact checked-out repository root.
        require_approval (bool): Block all Test operations unless the user
            explicitly reviewed and set status to APPROVED.

    Returns:
        dict: Validated protocol settings with no data loaded.

    Notes:
        Source-module edits after approval invalidate this frozen proposal.
        A status flag alone is not a claim of independent evaluation.
    """
    # 1. Refuse missing/malformed or unsupported protocol candidates.
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "harth-final-protocol-v1":
        raise ValueError("Unknown HARTH final protocol schema.")
    if manifest.get("status") not in ("PENDING_USER_REVIEW", "APPROVED"):
        raise ValueError("Unrecognised protocol approval state.")
    if require_approval and manifest["status"] != REQUIRED_APPROVAL:
        raise PermissionError(
            "The Stage 25 protocol is not APPROVED. Explicit user review "
            "is required BEFORE accessing original reserved Test recordings."
        )

    # 2. Validate every historical frozen code fingerprint; no silent changes.
    root = Path(repository_root).resolve()
    hashes = manifest.get("module_sha256", {})
    if len(hashes) != 6:
        raise ValueError("Expected six exact frozen source module fingerprints.")
    for relative, expected in hashes.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError("Missing or unsafe frozen code path.")
        if sha256_file(target) != expected:
            raise ValueError(f"Frozen code was modified: {relative}")

    # 3. Enforce approved HARTH task and development settings.
    source = manifest["source"]
    model = manifest["primary_model"]
    reference = manifest["reference_model"]
    if (
        source["task_codes"] != list(range(1, 9))
        or source["reserved_test_people"] != ["S023", "S032"]
        or source["cap_per_development_person"] != 500
        or source["expected_development_windows"] != 14201
        or source["development_participants"] != 29
        or source["test_cap_per_person"] != 500
        or source["sampling_rate_hz"] != 50
        or source["window_size"] != 100 or source["step_size"] != 50
        or source["annotation_purity"] != 1.0
        or source["timestamp_continuity"] is not True
    ):
        raise ValueError("Unsupported or changed frozen HARTH source protocol.")
    if (
        model["architecture"] != "1D-ResNet"
        or model["training_loss"] != "inverse_sqrt_weighted_ce"
        or model["hidden_channels"] != 32
        or model["in_channels"] != 6
        or model["classes"] != list(range(1, 9))
        or model["random_seed"] != 42
        or model["optimizer"] != "Adam"
        or model["learning_rate"] != 0.001
        or model["batch_size"] != 128
        or model["optimizer_steps"] != 400
        or model["checkpoint"] != "final_step"
        or reference["architecture"] != "Random Forest"
        or reference["n_estimators"] != 120
        or reference["min_samples_leaf"] != 3
        or reference["class_weight"] != "balanced_subsample"
        or reference["seed"] != 42
        or manifest["evaluation"]["use_test_for_tuning"] is not False
        or manifest["evaluation"]["temperature_scaling"] is not False
    ):
        raise ValueError("Proposed training controls were changed without review.")
    return manifest


def enforce_qualified_test_approval(protocol, confirmation):
    """
    Check explicit, two-part user authorisation before Test source access.

    Args:
        protocol (dict): load_final_protocol(...require_approval=True) result.
        confirmation (str): Human-supplied full approval confirmation phrase.

    Returns:
        bool: True only for an already APPROVED protocol and exact phrase.
    """
    if protocol.get("status") != REQUIRED_APPROVAL:
        raise PermissionError("Final protocol is not yet approved.")
    if confirmation != TEST_CONFIRMATION:
        raise PermissionError("Explicit qualified-Test confirmation is missing.")
    return True


# ==================== 2. Frozen Development Normalisation ====================
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


# ==================== 3. Final ResNet Model Fit and Artifact ====================
def fit_final_resnet(windows, labels, protocol, *, device="cuda"):
    """
    Fit ONE predeclared final ResNet on Development, without validation/Test.

    Args:
        windows (ndarray): Training-only original six-axis signal windows.
        labels (array-like): Original development activity codes 1..8.
        protocol (dict): Validated Stage 25 frozen training proposal.
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
    if x.shape != (14201, 6, 100) or len(y) != len(x):
        raise ValueError("Final fit requires exact frozen 14,201 development windows.")
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


# ==================== 4. Save and Reload Trusted Local Model Artefacts ====================
def save_final_models(model, scaler, reference, columns, protocol,
                      development_fingerprint, artifact_dir):
    """
    Save executable model and preprocessing metadata for later inference.

    Args:
        model (nn.Module): Trained final ResNet with saved state_dict.
        scaler (dict): Development-fitted original sensor means and SDs.
        reference (RandomForestClassifier): Fitted classical reference model.
        columns (sequence[str]): Ordered 30 deterministic feature names.
        protocol (dict): Stage 25 proposal/approved exact training settings.
        development_fingerprint (str): SHA256 of original Development cohort.
        artifact_dir (str | Path): Ignored local output directory.

    Returns:
        dict: Artifact paths and source hash dictionary.

    Notes:
        PyTorch state_dict only. joblib serialisation must be loaded only
        from a locally trusted source, never from arbitrary untrusted files.
        No identifiable raw source windows are persisted in these artefacts.
    """
    if torch is None:
        raise ImportError("Optional PyTorch is required.")
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    net_path = directory / "resnet_state_dict.pt"
    rf_path = directory / "random_forest.joblib"
    config_path = directory / "inference_config.json"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, net_path)
    joblib.dump(reference, rf_path)
    config = {
        "source_protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "development_cohort_sha256": development_fingerprint,
        "sensor_columns": [
            "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"
        ],
        "source_codes": list(range(1, 9)),
        "channel_mean": scaler["mean"].tolist(),
        "channel_std": scaler["std"].tolist(),
        "feature_columns": list(columns),
        "resnet_architecture": "1D-ResNet",
        "resnet_hidden_channels": 32,
        "window_size": 100, "window_stride": 50, "nominal_hz": 50,
        "preprocessing_fit_scope": "Development only",
        "test_fitted_values": False,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "resnet": str(net_path), "reference": str(rf_path),
        "config": str(config_path),
        "sha256": {name: sha256_file(path) for name, path in [
            ("resnet", net_path), ("reference", rf_path), ("config", config_path)
        ]},
    }


def load_final_models(directory, *, device="cpu"):
    """
    Load locally trusted saved final models with all frozen inference settings.

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
        raise ValueError("Frozen feature-order mismatch.")
    predicted_rf = reference.predict(features.to_numpy())
    return predicted_resnet, np.asarray(predicted_rf, dtype=int)


# ==================== 5. Explicitly Authorised Qualified Test Evaluation ====================
def score_qualified_heldout(y_true, predicted, subject_ids, classes):
    """
    Generate qualified single-evaluation metrics for fixed original class codes.

    Args:
        y_true (array-like): Accepted held-out source labels 1..8.
        predicted (array-like): Frozen model predictions as original source codes.
        subject_ids (array-like): Original reserved Test identity per window.
        classes (sequence[int]): Frozen all-eight-class metric denominator.

    Returns:
        dict: Fixed-eight-class Macro F1, balanced accuracy, raw accuracy,
        per-subject metrics, class precision/recall/F1/support and confusion.

    Notes:
        Only call AFTER enforce_qualified_test_approval(). Returned scores
        are descriptive qualified same-study held-out estimates, not an
        externally validated independent or prospective result.
    """
    y = np.asarray(y_true)
    p = np.asarray(predicted)
    ids = np.asarray(subject_ids)
    if not len(y) or len(y) != len(p) or len(y) != len(ids):
        raise ValueError("Source labels, predictions and persons must align.")
    if not np.isin(y, classes).all() or not np.isin(p, classes).all():
        raise ValueError("Unexpected Test class code.")
    if set(ids) != {"S023", "S032"}:
        raise ValueError("Qualified evaluation requires both reserved people.")
    each = []
    for person in sorted(set(ids)):
        mask = ids == person
        each.append({
            "subject_id": person, "support_windows": int(mask.sum()),
            "fixed_eight_class_macro_f1": float(f1_score(
                y[mask], p[mask], labels=classes, average="macro",
                zero_division=0
            )),
            "accuracy": float(accuracy_score(y[mask], p[mask])),
        })
    return {
        "macro_f1": float(f1_score(
            y, p, labels=classes, average="macro", zero_division=0
        )),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "per_subject": each,
        "classification_report": classification_report(
            y, p, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion": confusion_matrix(y, p, labels=classes).tolist(),
    }
