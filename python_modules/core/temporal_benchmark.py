# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Matched, resumable development benchmark of seven HARTH classifiers.

All models use exactly the same genuine eligible 500-cap source windows,
participant-held-out GroupKFold splits and fixed eight-class outcome.
RF learns five deterministic per-sensor statistical features. The six
neural conditions use raw multichannel windows and TRAIN-ONLY scaling.
No Test subjects or Calibration/Validation labels are used to fit.
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

from core.feature_engineering import extract_signal_features
from core.ml_baselines import evaluate_predictions_by_subject
from core.advanced_temporal_models import (
    ARCHITECTURES, build_temporal_model, count_trainable_parameters,
)
from core.temporal_cnn import (
    TemporalCNN, standardise_imu_train_only, compute_training_class_weights,
)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ==================== 1. Explicit Experimental Contract ====================
MODEL_NAMES = ("Random Forest", "Standard CNN", "Weighted CNN", *ARCHITECTURES)
PROTOCOL_VERSION = "harth-stage23-matched-500cap-group3-fixed400-v1"


def make_benchmark_plan(
    windows, labels, subject_ids, classes=tuple(range(1, 9)),
    n_splits=3, forbidden_subjects=("S023", "S032"),
):
    """
    Freeze the source-tensor identity and participant-disjoint held-out folds.

    Args:
        windows (np.ndarray): Original 500-cap (N, channels, time) windows.
        labels (array-like): Original eight-class source labels, row aligned.
        subject_ids (array-like): Source participant ID for every window.
        classes (sequence[int]): Declared source class codes in report order.
        n_splits (int): Number of participant-held-out development folds.
        forbidden_subjects (sequence[str]): Original protected Test identities.

    Returns:
        dict: Float32 source tensors, labels, cleaned IDs, SHA256 fingerprint,
            and (fold, train row indices, validation row indices) tuples.

    Notes:
        SHA256 covers every actual signal value, original label, identity,
        declared task and protocol version. No Test subject is loaded.
    """
    # 1. Validate the complete source contract before creating any split.
    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels)
    groups_raw = np.asarray(subject_ids, dtype=object)
    codes = list(classes)
    if x.ndim != 3 or len(x) == 0 or x.shape[1] < 1 or x.shape[2] < 8:
        raise ValueError("Expected nonempty raw multichannel signal windows.")
    if not np.isfinite(x).all():
        raise ValueError("Signal windows must all be finite.")
    if y.ndim != 1 or groups_raw.ndim != 1 or len(y) != len(x) or len(groups_raw) != len(x):
        raise ValueError("Windows, original labels and group IDs must align.")
    if len(codes) < 2 or len(set(codes)) != len(codes) or not np.isin(y, codes).all():
        raise ValueError("Source labels must belong to distinct task classes.")
    if any(pd.isna(item) or not str(item).strip() for item in groups_raw):
        raise ValueError("Each source window requires a valid participant ID.")
    groups = np.asarray([str(item).strip() for item in groups_raw])
    canonical = np.asarray([item.casefold() for item in groups])
    protected = {str(item).strip().casefold() for item in forbidden_subjects}
    if protected & set(canonical):
        raise ValueError("Reserved Test subject present in development benchmark.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    if len(set(canonical)) < n_splits:
        raise ValueError("Insufficient independent subjects.")

    # 2. Include source bytes and class/group ordering in the cache fingerprint.
    digest = hashlib.sha256()
    digest.update(PROTOCOL_VERSION.encode())
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(json.dumps(y.astype(int).tolist()).encode())
    digest.update(json.dumps(groups.tolist()).encode())
    digest.update(json.dumps(codes).encode())
    digest.update(str(n_splits).encode())
    folds = []
    for fold, (train, validation) in enumerate(
        GroupKFold(n_splits=n_splits).split(x, y, groups=canonical), 1
    ):
        if set(canonical[train]) & set(canonical[validation]):
            raise RuntimeError("Participant leakage detected in held-out fold.")
        if not set(codes).issubset(set(y[train].tolist())):
            raise ValueError(f"Training class absent in fold {fold}.")
        folds.append((fold, train, validation))
    if sorted(np.concatenate([v for _, _, v in folds]).tolist()) != list(range(len(x))):
        raise RuntimeError("Every source row must be evaluated once per model.")
    return {
        "windows": x, "labels": y, "groups": groups,
        "classes": codes, "fingerprint": digest.hexdigest(), "folds": folds,
    }


# ==================== 2. Deterministic Fold-Specific Model Fitting ====================
def _fit_and_predict(plan, model_name, train_rows, validation_rows, *,
                     feature_matrix, max_steps, batch_size,
                     learning_rate, seed, device):
    """Fit a fresh estimator using Train only and time its Validation predictions."""
    x, y, codes = plan["windows"], plan["labels"], plan["classes"]
    if model_name == "Random Forest":
        # 1. RF uses deterministic statistical features from identical windows.
        t0 = time.perf_counter()
        forest = RandomForestClassifier(
            n_estimators=120, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
        forest.fit(feature_matrix[train_rows], y[train_rows])
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        predicted = forest.predict(feature_matrix[validation_rows])
        inference_seconds = time.perf_counter() - t1
        return predicted, fit_seconds, inference_seconds, 0

    # 2. Reinitialise seeds, channel scaler, loss weights and model each job.
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested PyTorch computation device is unavailable.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tr, va, _ = standardise_imu_train_only(
        x[train_rows], x[validation_rows]
    )
    mapping = {code: i for i, code in enumerate(codes)}
    class_indices = np.asarray([mapping[item] for item in y[train_rows]], dtype=np.int64)
    training_dataset = TensorDataset(
        torch.from_numpy(tr), torch.from_numpy(class_indices)
    )
    loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0
    )
    if model_name in ("Standard CNN", "Weighted CNN"):
        model = TemporalCNN(in_channels=x.shape[1], n_classes=len(codes)).to(device)
    else:
        model = build_temporal_model(
            model_name, in_channels=x.shape[1], n_classes=len(codes)
        ).to(device)
    scheme = "none" if model_name == "Standard CNN" else "inverse_sqrt"
    weights = torch.as_tensor(
        compute_training_class_weights(y[train_rows], codes, method=scheme),
        dtype=torch.float32, device=device
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    parameters = count_trainable_parameters(model)
    iterator = iter(loader)

    # 3. Fix exactly the same number of optimizer updates for all six networks.
    #    Note: equal steps do NOT imply equal FLOPs, wall time or convergence.
    if device == "cuda":
        torch.cuda.synchronize()
    fit_start = time.perf_counter()
    for step in range(1, max_steps + 1):
        try:
            batch, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, target = next(iterator)
        model.train()
        optimiser.zero_grad(set_to_none=True)
        logits = model(batch.to(device))
        loss = nn.functional.cross_entropy(logits, target.to(device), weight=weights)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite Train loss: {model_name}, step {step}")
        loss.backward()
        optimiser.step()
        if step % 100 == 0 or step == max_steps:
            print(f"  {model_name}: {step}/{max_steps} updates | Train loss {loss.item():.4f}")
    if device == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_start

    # 4. Use last-update weights; no validation-selected checkpoint.
    model.eval()
    indices = []
    validation_tensor = torch.from_numpy(va)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(va), 256):
            batch = validation_tensor[offset:offset + 256].to(device)
            indices.extend(model(batch).argmax(dim=1).cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    return np.asarray([codes[i] for i in indices]), fit_seconds, inference_seconds, parameters


# ==================== 3. Resumable Source- and Code-Fingerprinted Jobs ====================
def run_benchmark_job(
    plan, fold, model_name, cache_dir, feature_matrix=None,
    max_steps=400, batch_size=128, learning_rate=1e-3,
    seed=42, device="cpu",
):
    """
    Run or load exactly one independently cached fold/model experiment.

    Args:
        plan (dict): Validated make_benchmark_plan() output.
        fold (int): Existing development GroupKFold fold number.
        model_name (str): One of seven MODEL_NAMES, including Random Forest.
        cache_dir (str | Path): Ignored local data/processed experiment folder.
        feature_matrix (np.ndarray | None): Stateless 30-feature matrix for RF.
        max_steps (int): Fixed neural optimizer updates, default 400.
        batch_size (int): Common CNN/Transformer batch size.
        learning_rate (float): Shared Adam learning rate.
        seed (int): Fixed model initialization and shuffling seed.
        device (str): cpu/cuda; validated before neural training.

    Returns:
        dict: Fold/model identity, source-row indices, original labels and
            source-code predictions, metrics, timing, parameters and cache key.

    Notes:
        - Each job writes one atomic JSON checkpoint AFTER full completion.
        - SHA256 links original source signals/labels/groups, relevant source
          code, hyperparameters and method; changing any invalidates cache.
        - Cached predictions are checked against current held-out row IDs and
          source truths before use. No pickle is used to load cached models.
        - Ignored cache is local: user pushes executed notebook outputs,
          not huge model files. Resume by rerunning this one notebook cell.
        - Never claim a job succeeded if its JSON is missing or corrupt.
    """
    # 1. Validate fold/model and registered experimental settings.
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown benchmark model name.")
    chosen = [(tr, va) for k, tr, va in plan["folds"] if k == fold]
    if len(chosen) != 1:
        raise ValueError("Fold number not present in predeclared plan.")
    train_rows, val_rows = chosen[0]
    if (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Positive training budget, batch size and learning rate required.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if model_name == "Random Forest":
        if feature_matrix is None or np.shape(feature_matrix)[0] != len(plan["labels"]):
            raise ValueError("RF requires source-row aligned statistical features.")
        if not np.isfinite(feature_matrix).all():
            raise ValueError("Nonfinite RF features.")

    # 2. Fingerprint all estimator and preprocessing source implementations.
    code_paths = [
        Path(__file__), Path(__file__).with_name("advanced_temporal_models.py"),
        Path(__file__).with_name("temporal_cnn.py"),
        Path(__file__).with_name("feature_engineering.py"),
    ]
    code_digest = hashlib.sha256()
    for path in code_paths:
        code_digest.update(path.read_bytes())
    config = {
        "protocol": PROTOCOL_VERSION, "data_hash": plan["fingerprint"],
        "code_hash": code_digest.hexdigest(), "fold": int(fold),
        "model": model_name, "steps": int(max_steps), "batch": int(batch_size),
        "lr": float(learning_rate), "seed": int(seed),
        "device": device if model_name != "Random Forest" else "sklearn",
        "train_rows_hash": hashlib.sha256(train_rows.tobytes()).hexdigest(),
        "validation_rows_hash": hashlib.sha256(val_rows.tobytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    path = folder / f"fold_{fold:02d}_{safe_name}.json"

    # 3. Reuse results only after verifying the complete source/control hash.
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key:
            if (cached.get("source_rows") != val_rows.tolist()
                or cached.get("true_labels") != plan["labels"][val_rows].astype(int).tolist()
                or len(cached.get("predicted_labels", [])) != len(val_rows)
                or not np.isin(cached["predicted_labels"], plan["classes"]).all()):
                raise RuntimeError(f"Invalid cached source alignment: {path}")
            print(f"RESTORED validated cache: Fold {fold} | {model_name}")
            return cached
        print(f"Cache configuration changed: refitting Fold {fold}, {model_name}")

    # 4. Fit once, validate all predictions, calculate source-grounded metrics.
    print(f"RUNNING Fold {fold} | {model_name} | "
          f"Train={len(train_rows):,} Validation={len(val_rows):,}")
    predicted, fit_seconds, prediction_seconds, parameters = _fit_and_predict(
        plan, model_name, train_rows, val_rows,
        feature_matrix=feature_matrix, max_steps=max_steps,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed, device=device,
    )
    pred = np.asarray(predicted)
    original = plan["labels"][val_rows]
    if len(pred) != len(original) or not np.isin(pred, plan["classes"]).all():
        raise RuntimeError("Prediction/source label alignment failed.")
    result = {
        "cache_key": key, "config": config, "fold": fold,
        "model": model_name, "source_rows": val_rows.tolist(),
        "true_labels": original.astype(int).tolist(),
        "predicted_labels": pred.astype(int).tolist(),
        "validation_subjects": sorted(set(plan["groups"][val_rows])),
        "train_subjects": sorted(set(plan["groups"][train_rows])),
        "train_windows": len(train_rows), "validation_windows": len(val_rows),
        "macro_f1": float(f1_score(
            original, pred, labels=plan["classes"], average="macro", zero_division=0
        )),
        "balanced_accuracy": float(balanced_accuracy_score(original, pred)),
        "accuracy": float(accuracy_score(original, pred)),
        "train_seconds": float(fit_seconds),
        "validation_inference_seconds": float(prediction_seconds),
        "trainable_parameters": int(parameters),
    }
    if not (0 <= result["macro_f1"] <= 1):
        raise RuntimeError("Invalid held-out model metric.")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    print(f"FINISHED Fold {fold} | {model_name} | "
          f"Macro F1={result['macro_f1']:.4f} | "
          f"Train={fit_seconds:.1f} s")
    return result


# ==================== 4. Consolidated Out-of-Fold Scientific Reporting ====================
def summarise_benchmark_results(plan, jobs):
    """
    Recalculate matched fold/participant/class metrics from cached predictions.

    Args:
        plan (dict): Original frozen dataset/GroupKFold plan.
        jobs (sequence[dict]): Exactly one successful result per model/fold.

    Returns:
        fold_results (pd.DataFrame): Matched validation metrics and cost.
        subject_results (pd.DataFrame): 29 subject-wise fixed-task Macro F1.
        class_results (pd.DataFrame): Pooled out-of-fold class diagnostics.
        prediction_results (pd.DataFrame): Per-model original source row IDs,
            subject IDs, true labels and predicted activity labels.
        confusion (dict): 8x8 source-code confusion arrays per model.

    Notes:
        Completeness and exact out-of-fold row/source matching are asserted.
        Pooled window-level F1 and equal-person statistics are distinct.
        Correlated overlapping windows must not be treated as independent
        experimental participants.
    """
    # 1. Reject incomplete seven-by-three result grids or duplicate jobs.
    n_folds = len(plan["folds"])
    expected = {(name, fold) for name in MODEL_NAMES
                for fold, _, _ in plan["folds"]}
    actual = [(item["model"], item["fold"]) for item in jobs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated model-by-fold job matrix.")
    y, groups = plan["labels"], plan["groups"]
    folds, all_predictions = [], []
    for item in jobs:
        name, fold = item["model"], item["fold"]
        train, validation = next((tr, va) for k, tr, va in plan["folds"] if k == fold)
        ids = np.asarray(item["source_rows"], dtype=int)
        truth = np.asarray(item["true_labels"])
        pred = np.asarray(item["predicted_labels"])
        if not np.array_equal(ids, validation) or not np.array_equal(truth, y[validation]):
            raise RuntimeError("Mismatched source-row identity in a fold.")
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("Participant overlap in recorded evaluation.")
        folds.append({
            "model": name, "fold": fold, "macro_f1": f1_score(
                truth, pred, labels=plan["classes"], average="macro", zero_division=0
            ),
            "accuracy": accuracy_score(truth, pred),
            "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "train_subjects": len(set(groups[train])),
            "validation_subjects": len(set(groups[validation])),
            "train_windows": len(train), "validation_windows": len(validation),
            "train_seconds": item["train_seconds"],
            "validation_inference_seconds": item["validation_inference_seconds"],
            "trainable_parameters": item["trainable_parameters"],
        })
        all_predictions.append(pd.DataFrame({
            "model": name, "fold": fold, "source_row": ids,
            "subject_id": groups[ids], "true_label": truth, "predicted_label": pred,
        }))
    fold_results = pd.DataFrame(folds).sort_values(["model", "fold"]).reset_index(drop=True)
    prediction_results = pd.concat(all_predictions, ignore_index=True)
    individual, per_class, confusion = [], [], {}
    for name in MODEL_NAMES:
        subset = prediction_results.loc[prediction_results["model"] == name]
        if not subset["source_row"].is_unique or sorted(subset["source_row"]) != list(range(len(y))):
            raise RuntimeError(f"Missing/duplicate out-of-fold source rows: {name}")
        sorted_rows = subset.sort_values("source_row")
        if not np.array_equal(sorted_rows["true_label"].to_numpy(), y):
            raise RuntimeError("Wrong original ground truth in held-out predictions.")
        individual.append(evaluate_predictions_by_subject(
            subset["true_label"].to_numpy(), subset["predicted_label"].to_numpy(),
            subset["subject_id"].to_numpy(), plan["classes"], name
        ))
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["true_label"], subset["predicted_label"],
            labels=plan["classes"], zero_division=0
        )
        per_class.append(pd.DataFrame({
            "model": name, "class_code": plan["classes"],
            "support": support, "precision": precision, "recall": recall, "f1": f1
        }))
        confusion[name] = confusion_matrix(
            subset["true_label"], subset["predicted_label"],
            labels=plan["classes"]
        )
    return (
        fold_results,
        pd.concat(individual, ignore_index=True),
        pd.concat(per_class, ignore_index=True),
        prediction_results,
        confusion,
    )
