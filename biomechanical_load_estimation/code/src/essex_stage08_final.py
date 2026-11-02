# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stage 08: pre-frozen one-time source-cohort evaluation, no Test tuning.

The final model and recipe are selected by Stage 07 Training-only evidence.
Fit uses only the predetermined TRAIN+VALIDATION development pool.
TEST may be released once for two predeclared models after manual authorisation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor

try:
    from .essex_regression_baselines import (
        trial_features, flatten_targets, regression_metrics,
        subject_balanced_mae,
    )
    from .essex_temporal_regression import (
        check_temporal_arrays, make_temporal_model,
        training_fold_statistics, peak_aware_loss,
    )
except ImportError:  # Colab source-path installation
    from essex_regression_baselines import (
        trial_features, flatten_targets, regression_metrics,
        subject_balanced_mae,
    )
    from essex_temporal_regression import (
        check_temporal_arrays, make_temporal_model,
        training_fold_statistics, peak_aware_loss,
    )


PROTOCOL_FILE = "08_frozen_final_protocol.json"
FINAL_GATE_FILE = "08_final_test_quality_gate.json"
RELEASE_PHRASE = "I_APPROVE_ONE_TIME_TEST_EVALUATION"
EXPECTED_PRIMARY = "Random Forest (Stage 05 locked)"
SECONDARY = "Dilated TCN"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_protocol(path, gate04, gate05, gate07, candidate_row):
    """Fail closed if model selection, source, splits or freeze changed."""
    file = Path(path)
    protocol = json.loads(file.read_text(encoding="utf-8"))
    assert protocol["protocol_status"] == "FROZEN_BEFORE_TEST"
    assert protocol["stage"] == "08"
    source = protocol["canonical_pair_sha256"]
    if not (
        source == gate04["canonical_pair_sha256"]
        == gate05["stage04_canonical_pair_sha256"]
        == gate07["canonical_pair_sha256"]
    ):
        raise ValueError("Frozen source fingerprint differs from prior gates.")
    if gate04["stage04_quality_gate"] != "PASSED" or (
        gate05["quality_gate"] != "PASSED" or
        gate07["quality_gate"] != "PASSED"
    ):
        raise ValueError("A required earlier quality gate failed.")
    if protocol["original_archive_sha256"] != gate04["original_sha256"]:
        raise ValueError("Original source archive differs from the frozen source.")
    if gate07["total_completed_jobs"] != 75:
        raise ValueError("Stage 07 experimental evidence is incomplete.")
    if gate07["provisional_final_candidate_from_training_only"] != EXPECTED_PRIMARY:
        raise ValueError("Stage 07 candidate no longer matches freeze.")
    if candidate_row["candidate"] != EXPECTED_PRIMARY or (
        str(candidate_row["training_cv_selected"]).lower() != "true"
    ):
        raise ValueError("Training-only decision does not match frozen primary.")
    if abs(float(candidate_row["selection_mae"]) -
           protocol["locked_primary_cv_subject_balanced_mae"]) > 1e-12:
        raise ValueError("Primary Training-CV decision was changed.")
    if protocol["locked_primary_candidate"] != EXPECTED_PRIMARY:
        raise ValueError("Unexpected frozen primary model.")
    rf = protocol["primary_model"]
    if (rf["name"] != "Random Forest" or rf["n_estimators"] != 200 or
            rf["min_samples_leaf"] != 3 or rf["max_features"] != 1.0 or
            rf["random_state"] != 2026 or rf["feature_count"] != 210):
        raise ValueError("Frozen Random Forest protocol was modified.")
    tcn = protocol["secondary_model"]
    if (tcn["name"] != SECONDARY or tcn["seed"] != 2128 or
            tcn["epochs"] != 35 or tcn["batch_size"] != 64 or
            tcn["peak_weight"] != 0.25):
        raise ValueError("Secondary comparison protocol was modified.")
    policy = protocol["final_fitting_policy"]
    if policy["development_splits"] != ["TRAIN", "VALIDATION"] or (
        policy["development_trials"] != 1546 or
        policy["development_participants"] != 33
    ):
        raise ValueError("Final development pool does not match source registry.")
    if protocol["single_final_evaluation"]["trials"] != 183 or (
        protocol["single_final_evaluation"]["participants"] != 4
    ):
        raise ValueError("Held-out Test design changed.")
    if protocol["manual_release"]["token"] != RELEASE_PHRASE:
        raise ValueError("One-time Test release procedure was changed.")
    return protocol, file_sha256(file)


def fit_frozen_primary(x_development, y_development, channels, protocol):
    """Deterministic full-development fit, never receives Test data."""
    rf = protocol["primary_model"]
    features, names = trial_features(x_development, channels)
    targets = flatten_targets(y_development)
    if features.shape != (1546, 210) or targets.shape != (1546, 22):
        raise ValueError("Frozen RF development shape mismatch.")
    model = RandomForestRegressor(
        n_estimators=rf["n_estimators"],
        min_samples_leaf=rf["min_samples_leaf"],
        max_features=rf["max_features"],
        random_state=rf["random_state"],
        n_jobs=-1,
    )
    model.fit(features, targets)
    if model.n_outputs_ != 22 or len(model.estimators_) != 200:
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
    if len(source_x) != 1546:
        raise ValueError("Expected 1,546 frozen development trials.")
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
    sx = ((source_x - scaler.input_mean) /
          scaler.input_std).astype(np.float32)
    sy = ((np.asarray(y) - scaler.target_mean) /
          scaler.target_std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(sx)),
        torch.from_numpy(np.ascontiguousarray(sy)),
        torch.from_numpy(original_pos.copy()),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    dev = torch.device(device)
    model = make_temporal_model(SECONDARY).to(dev)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    losses = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total = 0.0
        for signals, labels, positions in loader:
            signals, labels, positions = (
                signals.to(dev), labels.to(dev), positions.to(dev)
            )
            optimiser.zero_grad(set_to_none=True)
            prediction = model(signals, positions)
            loss = peak_aware_loss(
                prediction, labels, float(cfg["peak_weight"])
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frozen TCN training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach().cpu()) * len(signals)
        losses.append({
            "epoch": epoch,
            "training_loss": total / len(dataset),
            "model": SECONDARY,
            "split": "FROZEN_TRAIN_PLUS_VALIDATION",
        })
    model.eval()
    if len(losses) != 35 or not np.isfinite(
        [x["training_loss"] for x in losses]
    ).all():
        raise RuntimeError("Incomplete frozen TCN training.")
    return model, scaler, pd.DataFrame(losses)


def predict_frozen_secondary(model, scaler, x, pos, *, device, batch=64):
    """Read native FE positions using the already frozen development scaler."""
    original_x, original_pos = check_temporal_arrays(x, pos)
    features = ((original_x - scaler.input_mean) /
                scaler.input_std).astype(np.float32)
    dev = torch.device(device)
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch)):
            last = start + int(batch)
            bx = torch.from_numpy(
                np.ascontiguousarray(features[start:last])
            ).to(dev)
            bi = torch.from_numpy(
                np.ascontiguousarray(original_pos[start:last])
            ).to(dev)
            out.append(model(bx, bi).cpu().numpy())
    pred = np.concatenate(out, axis=0).astype(np.float64)
    pred = pred * scaler.target_std + scaler.target_mean
    if pred.shape != (len(x), 11, 2) or not np.isfinite(pred).all():
        raise ValueError("Frozen TCN output invalid.")
    return pred


def release_one_time_test(
    *, approval, protocol, protocol_sha256, output_dir, ledger_dir,
):
    """Atomic local release log; block repeated evaluation within the runtime.

    This is an integrity guard, not a global lock across unrelated machines.
    Final results already saved in output/ also prohibit a new Test run.
    """
    if approval != protocol["manual_release"]["token"]:
        raise PermissionError(
            "Final Test is sealed. Review frozen protocol, then enter its "
            "explicit approval phrase in Notebook 08, Section 06."
        )
    dest = Path(output_dir)
    ledger_dir = Path(ledger_dir)
    previous = [dest / FINAL_GATE_FILE,
                dest / "08_test_trial_predictions.csv",
                dest / "08_test_primary_secondary_metrics.csv"]
    if any(p.exists() for p in previous):
        raise FileExistsError(
            "Final Test results already exist. Do not rescore or select "
            "models using the same reserved participants."
        )
    ledger_dir.mkdir(parents=True, exist_ok=True)
    mark = ledger_dir / "08_one_time_final_test_release.json"
    info = {
        "status": "TEST_RELEASE_CONSUMED",
        "stage": "08",
        "protocol_sha256": protocol_sha256,
        "authorisation": "explicit_notebook_release",
        "test_evaluation_rounds_permitted": 1,
    }
    try:
        descriptor = os.open(
            str(mark), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "This runtime already released the frozen Test set. No retesting."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2)
        stream.write("\n")
    return info


def participant_bootstrap(y, prediction, subjects, *, draws=2000, seed=2026):
    """Exploratory participant-cluster CI (four clusters only)."""
    ids = np.asarray(subjects, dtype=str)
    if len(ids) != len(y):
        raise ValueError("Source trial identities do not match observations.")
    people = sorted(set(ids))
    if len(people) != 4:
        raise ValueError("Frozen Test must have exactly four participants.")
    values = np.array([
        np.abs(y[ids == person] - prediction[ids == person]).mean()
        for person in people
    ], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, 4, size=(int(draws), 4))
    boot = values[sampled].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "bootstrap_low_95": float(np.quantile(boot, 0.025)),
        "bootstrap_high_95": float(np.quantile(boot, 0.975)),
        "participants": 4,
        "resamples": int(draws),
        "scope": "exploratory_same_source_test",
    }


def paired_test_bootstrap(y, primary, secondary, subjects, *, draws=2000, seed=2026):
    ids = np.asarray(subjects, dtype=str)
    people = sorted(set(ids))
    if len(people) != 4:
        raise ValueError("Expected four paired Test subjects.")
    delta = np.array([
        np.abs(y[ids == p] - secondary[ids == p]).mean() -
        np.abs(y[ids == p] - primary[ids == p]).mean()
        for p in people
    ], dtype=float)
    rng = np.random.default_rng(seed)
    resampled = delta[rng.integers(0, 4, (int(draws), 4))].mean(axis=1)
    return {
        "comparison": "secondary_minus_primary",
        "difference": float(delta.mean()),
        "bootstrap_low_95": float(np.quantile(resampled, 0.025)),
        "bootstrap_high_95": float(np.quantile(resampled, 0.975)),
        "participants": 4, "resamples": int(draws),
        "scope": "exploratory_not_confirmatory",
    }
