# =================== Presented by: Reza Saadatyar (2026) =====================
"""Subject-matched multi-seed and controlled loss ablations for HARTH.

Scientific scope: 29 development subjects, 500-cap original windows,
subject-held-out GroupKFold. No original Test participants are loaded.
Use a fixed 400-update budget, Train-only normalisation and Train-only
class weights. Store actual successful held-out predictions per job.
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)

from core.advanced_temporal_models import build_temporal_model, count_trainable_parameters
from core.temporal_cnn import standardise_imu_train_only, compute_training_class_weights


ROBUST_MODELS = ("1D-ResNet", "TCN")
LOSS_METHODS = ("weighted_ce", "focal", "class_balanced")
ROBUST_VERSION = "harth-stage24-fixed400-three-seeds-three-folds-v1"


# ===================== 1. Class Imbalance Loss Functions =====================
def calculate_loss_weights(y_train, classes, method="weighted_ce",
                           beta=0.9999):
    """
    Calculate only training-derived class weights for a specified loss.

    Args:
        y_train (array-like): Source-code annotations of Fold Train only.
        classes (sequence[int]): Fixed task class-code ordering.
        method (str): 'weighted_ce', 'focal' or 'class_balanced'.
        beta (float): Effective-number weighting hyperparameter fixed
            before evaluation, used only for 'class_balanced'.

    Returns:
        np.ndarray: Positive float32 class weights with arithmetic mean one.

    Notes:
        Focal Loss uses the SAME inverse-sqrt weights as weighted CE,
        with a fixed focusing exponent gamma=2. Class-balanced CE uses
        (1-beta)/(1-beta**count) as originally motivated by effective
        sample number, but overlapping windows are correlated, so these
        effective counts should not be treated as independent information.
    """
    # 1. Verify observed Train classes and exclude unseen or invalid source codes.
    codes = list(classes)
    y = np.asarray(y_train)
    if y.ndim != 1 or not len(y) or len(set(codes)) < 2:
        raise ValueError("Aligned non-empty training class codes are required.")
    if not np.isin(y, codes).all():
        raise ValueError("Unknown source code in Train labels.")
    counts = np.asarray([(y == c).sum() for c in codes], dtype=np.float64)
    if (counts == 0).any():
        raise ValueError("Every task class must be present in Fold Train.")

    # 2. Keep the original weighted-ce baseline identical to Stage 23.
    if method in ("weighted_ce", "focal"):
        return compute_training_class_weights(y, codes, method="inverse_sqrt")

    # 3. Change only the weighting formula for the class-balanced ablation.
    if method != "class_balanced":
        raise ValueError("Unknown loss mode.")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("class-balanced beta must lie strictly between zero and one.")
    effective = -np.expm1(counts * np.log(beta))
    weights = (1.0 - beta) / effective
    weights /= weights.mean()
    return weights.astype(np.float32)


def calculate_minibatch_loss(logits, encoded_targets, weights, method="weighted_ce",
                             focal_gamma=2.0):
    """
    Compute train-only multiclass classification loss on raw model logits.

    Args:
        logits (torch.Tensor): Unnormalised (batch, classes) outputs.
        encoded_targets (torch.Tensor): Zero-based integer class positions.
        weights (torch.Tensor): Positive per-class Train-only loss weights.
        method (str): One of LOSS_METHODS.
        focal_gamma (float): Fixed focus exponent for focal mode.

    Returns:
        torch.Tensor: Differentiable, finite scalar class-aware loss.

    Notes:
        Weighted and class-balanced CE use standard PyTorch normalisation.
        Focal mode computes a per-example focal term based on UNWEIGHTED
        negative log likelihood before applying identical Train-only alpha.
        Its weighted reduction divides by the sum of batch alpha weights.
        Other parameters, windows and update counts remain unchanged.
    """
    # 1. Validate the shared architecture/class-code interface.
    if logits.ndim != 2 or encoded_targets.ndim != 1:
        raise ValueError("Expected raw logits and 1D target indices.")
    if len(logits) != len(encoded_targets) or logits.shape[1] != len(weights):
        raise ValueError("Batch targets, logits and class weights must align.")
    if method not in LOSS_METHODS:
        raise ValueError("Unknown loss mode.")
    if not np.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError("focal_gamma must be nonnegative.")

    # 2. Apply standard weighted/class-balanced Cross-Entropy where requested.
    if method != "focal":
        return nn.functional.cross_entropy(logits, encoded_targets, weight=weights)

    # 3. Apply focal emphasis only to difficult examples, without labels leakage.
    per_sample_ce = nn.functional.cross_entropy(
        logits, encoded_targets, reduction="none"
    )
    prob_true = torch.exp(-per_sample_ce)
    focus = (1.0 - prob_true).pow(focal_gamma)
    alpha = weights[encoded_targets]
    return (alpha * focus * per_sample_ce).sum() / alpha.sum()


# ==================== 2. Validation and Fold-specific Fit ====================
def _fit_predict_one(
    plan, fold, architecture, seed, loss_method, *,
    device="cuda", max_steps=400, batch_size=128, learning_rate=1e-3,
    focal_gamma=2.0, beta=0.9999,
):
    """Fit one freshly seeded network and score exactly its held-out source rows."""
    # 1. Establish unmodified Train and Validation indices from the fixed plan.
    if architecture not in ROBUST_MODELS or loss_method not in LOSS_METHODS:
        raise ValueError("Unsupported robustness architecture/loss.")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested PyTorch device is unavailable.")
    _, train_rows, validation_rows = next(
        ((k, tr, va) for k, tr, va in plan["folds"] if k == fold),
        (None, None, None)
    )
    if train_rows is None:
        raise ValueError("Unknown participant-held-out fold.")
    x, y, codes = plan["windows"], plan["labels"], plan["classes"]

    # 2. Fit scaler/weights using Train ONLY and seed every random generator.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tr, va, _ = standardise_imu_train_only(x[train_rows], x[validation_rows])
    lookup = {c: j for j, c in enumerate(codes)}
    indexed = np.asarray([lookup[v] for v in y[train_rows]], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(tr), torch.from_numpy(indexed))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_temporal_model(
        architecture, in_channels=x.shape[1], n_classes=len(codes)
    ).to(device)
    class_weights = torch.as_tensor(
        calculate_loss_weights(y[train_rows], codes, loss_method, beta),
        dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterator = iter(loader)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    # 3. Train for the same predeclared 400 updates in every condition.
    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb.to(device))
        loss = calculate_minibatch_loss(
            logits, yb.to(device), class_weights,
            method=loss_method, focal_gamma=focal_gamma
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                f"Nonfinite loss: {architecture}, {loss_method}, seed {seed}, fold {fold}"
            )
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == max_steps:
            print(
                f"  {architecture} | {loss_method} | seed {seed} | "
                f"Fold {fold}: {step}/{max_steps}, Train loss={loss.item():.4f}"
            )
    if device == "cuda":
        torch.cuda.synchronize()
    fit_s = time.perf_counter() - started

    # 4. Score only after the last update; no Validation checkpoint search.
    model.eval()
    predicted_indices = []
    held_out = torch.from_numpy(va)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for k in range(0, len(va), 256):
            batch = held_out[k:k + 256].to(device)
            predicted_indices.extend(model(batch).argmax(1).cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    inference_s = time.perf_counter() - inference_start
    predictions = np.asarray([codes[p] for p in predicted_indices])
    return predictions, fit_s, inference_s, count_trainable_parameters(model)


# ============ 3. Source-Fingerprinted, Resumable Experiment Jobs =============
def run_robustness_job(
    plan, architecture, fold, seed, loss_method,
    cache_dir, *, device="cuda", max_steps=400, batch_size=128,
    learning_rate=1e-3, focal_gamma=2.0, beta=0.9999,
):
    """
    Run or safely restore one model/seed/loss/subject-fold experiment.

    Args:
        plan (dict): Source-fingerprinted output of make_benchmark_plan().
        architecture (str): '1D-ResNet' or 'TCN'.
        fold (int): Existing development fold.
        seed (int): Explicit model and DataLoader seed.
        loss_method (str): Weighted CE, Focal or class-balanced CE.
        cache_dir (str | Path): Local ignored JSON cache path.
        device (str): Prefer CUDA for substantive Colab training.
        max_steps (int): Fixed optimizer-update count.
        batch_size (int): Same size for all comparisons.
        learning_rate (float): Shared Adam step size.
        focal_gamma (float): Fixed focal focusing exponent.
        beta (float): Class-balanced effective-number parameter.

    Returns:
        dict: SHA256-verified source/prediction alignment, original
            labels, class predictions, fold/seed/method metadata, metrics,
            actual time cost and model parameter count.

    Notes:
        Every successful job is written atomically and can be restored only
        when data, code fingerprint, hyperparameters, folds and device match.
        The local cache is ignored by Git; real notebook outputs are pushed.
        Google Colab runtime resets may delete the cache.
    """
    # 1. Validate predeclared scientific configurations before any training.
    if architecture not in ROBUST_MODELS or loss_method not in LOSS_METHODS:
        raise ValueError("Unknown architecture/loss configuration.")
    if type(seed) is not int or seed < 0 or type(fold) is not int:
        raise ValueError("seed must be nonnegative and fold must be an integer.")
    if type(max_steps) is not int or max_steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive optimizer budget and batch size required.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive learning rate required.")
    if not np.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError("Nonnegative focal exponent required.")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("Invalid effective-number beta.")
    pair = [(tr, va) for k, tr, va in plan["folds"] if k == fold]
    if len(pair) != 1:
        raise ValueError("Unrecognised source fold.")
    train_rows, validation_rows = pair[0]
    group_ids = plan["groups"]
    if set(group_ids[train_rows]) & set(group_ids[validation_rows]):
        raise RuntimeError("Subject leakage in development fold.")

    # 2. Key the cache to actual raw signals, full model code and the scientific protocol.
    code_hash = hashlib.sha256()
    for name in (
        "robustness_experiments.py", "advanced_temporal_models.py",
        "temporal_cnn.py", "temporal_benchmark.py",
    ):
        code_hash.update((Path(__file__).parent / name).read_bytes())
    config = {
        "version": ROBUST_VERSION, "data_sha256": plan["fingerprint"],
        "code_sha256": code_hash.hexdigest(),
        "architecture": architecture, "fold": fold, "seed": seed,
        "loss": loss_method, "max_steps": max_steps, "batch_size": batch_size,
        "lr": float(learning_rate), "focal_gamma": float(focal_gamma),
        "class_balanced_beta": float(beta), "device": device,
        "torch_version": torch.__version__,
        "train_rows_sha256": hashlib.sha256(train_rows.tobytes()).hexdigest(),
        "validation_rows_sha256": hashlib.sha256(validation_rows.tobytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    slug = architecture.lower().replace(" ", "_").replace("-", "_")
    path = root / f"f{fold:02d}_{slug}_{loss_method}_seed{seed}.json"

    # 3. Validate cache provenance; never silently reuse old predictions.
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key:
            if (
                cached.get("source_rows") != validation_rows.tolist()
                or cached.get("true_labels") != plan["labels"][validation_rows].astype(int).tolist()
                or len(cached.get("predicted_labels", [])) != len(validation_rows)
                or not np.isin(cached["predicted_labels"], plan["classes"]).all()
            ):
                raise RuntimeError(f"Cached source label alignment failed: {path}")
            print(f"RESTORED: {architecture} | {loss_method} | seed={seed} | fold={fold}")
            return cached
        print(f"Invalidated old cache: {path.name}")

    # 4. Fit from scratch once and store only after successful held-out scoring.
    print(f"RUNNING: {architecture} | {loss_method} | seed={seed} | fold={fold}")
    predictions, fit_s, inference_s, parameters = _fit_predict_one(
        plan, fold, architecture, seed, loss_method, device=device,
        max_steps=max_steps, batch_size=batch_size,
        learning_rate=learning_rate, focal_gamma=focal_gamma, beta=beta,
    )
    truth = plan["labels"][validation_rows]
    if len(predictions) != len(validation_rows) or not np.isin(
        predictions, plan["classes"]
    ).all():
        raise RuntimeError("Held-out predictions are incomplete or misaligned.")
    result = {
        "cache_key": key, "config": config,
        "architecture": architecture, "fold": fold, "seed": seed,
        "loss_method": loss_method, "source_rows": validation_rows.tolist(),
        "true_labels": truth.astype(int).tolist(),
        "predicted_labels": predictions.astype(int).tolist(),
        "train_subjects": sorted(set(group_ids[train_rows])),
        "validation_subjects": sorted(set(group_ids[validation_rows])),
        "train_windows": len(train_rows), "validation_windows": len(validation_rows),
        "macro_f1": float(f1_score(
            truth, predictions, labels=plan["classes"],
            average="macro", zero_division=0
        )),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "train_seconds": float(fit_s),
        "inference_seconds": float(inference_s),
        "trainable_parameters": int(parameters),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    os.replace(temporary, path)
    print(
        f"COMPLETED: {architecture} | {loss_method} | seed {seed} | "
        f"fold {fold} | Macro F1={result['macro_f1']:.4f}"
    )
    return result


# ============== 4. Paired Out-of-Fold Evidence Aggregation ===================
def summarise_robustness_results(plan, results, required_experiments):
    """
    Reconstruct complete fold, source-window, class and participant evidence.

    Args:
        plan (dict): Frozen source and group folds.
        results (list[dict]): Successful cache-checked experimental jobs.
        required_experiments (sequence[tuple]): Unique architecture, loss,
            seed combinations, each requiring ALL subject-held-out folds.

    Returns:
        folds (pd.DataFrame): Per-job source-aligned Macro F1, Balanced
            Accuracy, train/validation counts and actual timing.
        people (pd.DataFrame): One fixed-eight-class Macro F1 per person
            for each architecture/loss/seed.
        classes (pd.DataFrame): Pooled OOF per-class precision/recall/F1
            and actual support for each full model condition.
        predictions (pd.DataFrame): Genuine per-source-row held-out labels
            and predictions for reproducibility and paired diagnostics.

    Notes:
        A complete expected model/loss/seed/fold matrix is REQUIRED.
        Every condition must predict every base source row exactly once.
        Models evaluated on overlapping windows are not independent trials.
    """
    # 1. Require all uniquely specified conditions, folds and model identities.
    expected = {
        (architecture, loss, seed, fold)
        for architecture, loss, seed in required_experiments
        for fold, _, _ in plan["folds"]
    }
    observed = [
        (j["architecture"], j["loss_method"], j["seed"], j["fold"])
        for j in results
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated robustness experiment matrix.")
    rows, predictions = [], []
    for job in results:
        k, train, validation = next(
            (a, tr, va) for a, tr, va in plan["folds"] if a == job["fold"]
        )
        source = np.asarray(job["source_rows"], dtype=int)
        true = np.asarray(job["true_labels"])
        pred = np.asarray(job["predicted_labels"])
        if (not np.array_equal(source, validation)
                or not np.array_equal(true, plan["labels"][validation])
                or not np.isin(pred, plan["classes"]).all()):
            raise RuntimeError("Wrong real source-row provenance in robustness result.")
        if set(plan["groups"][train]) & set(plan["groups"][validation]):
            raise RuntimeError("Train/Validation participant leakage.")
        rows.append({
            "architecture": job["architecture"], "loss_method": job["loss_method"],
            "seed": job["seed"], "fold": job["fold"],
            "macro_f1": f1_score(
                true, pred, labels=plan["classes"], average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(true, pred),
            "accuracy": accuracy_score(true, pred),
            "train_windows": len(train), "validation_windows": len(validation),
            "train_seconds": job["train_seconds"],
            "inference_seconds": job["inference_seconds"],
            "trainable_parameters": job["trainable_parameters"],
        })
        predictions.append(pd.DataFrame({
            "architecture": job["architecture"], "loss_method": job["loss_method"],
            "seed": job["seed"], "fold": job["fold"],
            "source_row": source,
            "subject_id": plan["groups"][source],
            "true_label": true, "predicted_label": pred,
        }))
    folds = pd.DataFrame(rows)
    prediction_table = pd.concat(predictions, ignore_index=True)
    people, classes = [], []
    for architecture, method, seed in required_experiments:
        mask = (
            prediction_table["architecture"].eq(architecture)
            & prediction_table["loss_method"].eq(method)
            & prediction_table["seed"].eq(seed)
        )
        rows = prediction_table.loc[mask]
        if len(rows) != len(plan["labels"]) or not rows["source_row"].is_unique:
            raise RuntimeError("Missing/duplicate held-out rows in one experiment.")
        ordered = rows.sort_values("source_row")
        if not np.array_equal(
            ordered["true_label"].to_numpy(), plan["labels"]
        ):
            raise RuntimeError("Out-of-fold labels mismatch original source ordering.")
        for subject, part in rows.groupby("subject_id"):
            people.append({
                "architecture": architecture, "loss_method": method, "seed": seed,
                "subject_id": subject, "windows": len(part),
                "macro_f1_fixed_task": float(f1_score(
                    part["true_label"], part["predicted_label"],
                    labels=plan["classes"], average="macro", zero_division=0
                )),
            })
        pre, rec, f1, support = precision_recall_fscore_support(
            rows["true_label"], rows["predicted_label"],
            labels=plan["classes"], zero_division=0,
        )
        for idx, activity in enumerate(plan["classes"]):
            classes.append({
                "architecture": architecture, "loss_method": method,
                "seed": seed, "class_code": activity,
                "support": int(support[idx]), "precision": float(pre[idx]),
                "recall": float(rec[idx]), "f1": float(f1[idx]),
            })
    return folds, pd.DataFrame(people), pd.DataFrame(classes), prediction_table
