# =================== Presented by: Reza Saadatyar (2026) =====================
"""Small, reproducible PyTorch baseline for multichannel movement sequences."""

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ========================== 1. Temporal CNN Model ============================
class TemporalCNN(nn.Module):
    """
    Predict activity from time-ordered multichannel IMU windows.

    Args:
        in_channels (int): Number of input sensors, e.g. six axes.
        n_classes (int): Number of target activity codes.
        hidden_channels (int): Filters in the first convolutional stage.

    Input:
        Float tensor (batch, channels, time).

    Output:
        Class logits (batch, classes); no softmax before CrossEntropyLoss.
    """
    def __init__(self, in_channels=6, n_classes=8, hidden_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels * 2, n_classes)

    def forward(self, x):
        features = self.features(x).squeeze(-1)
        return self.classifier(features)


# ====================== 2. Train-Only Normalisation ==========================
def standardise_imu_train_only(train, validation):
    """
    Scale each channel using training windows only.

    Args:
        train (np.ndarray): (N_train, channels, time).
        validation (np.ndarray): (N_val, same_channels, time).

    Returns:
        train_scaled, val_scaled (np.ndarray): float32 arrays.
        stats (dict): Training channel-wise mean and standard deviation.

    Notes:
        All time points in the training set contribute to channel statistics.
        Validation windows cannot influence the fitted transform.
    """
    train = np.asarray(train, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    if train.ndim != 3 or validation.ndim != 3 or train.shape[1:] != validation.shape[1:]:
        raise ValueError("Expected matching (windows, channels, time) arrays.")
    if not len(train) or not len(validation) or not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Non-empty finite train and validation arrays required.")
    mean = train.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = train.std(axis=(0, 2), keepdims=True, dtype=np.float64)
    safe_std = np.maximum(std, 1e-6)
    return (
        ((train - mean) / safe_std).astype(np.float32),
        ((validation - mean) / safe_std).astype(np.float32),
        {"channel_mean": mean.reshape(-1), "channel_std": safe_std.reshape(-1)},
    )


# ======================= 3. Training-Only Class Weights ======================
def compute_training_class_weights(
    y_train, classes, method="inverse_sqrt",
):
    """
    Calculate class-loss weights using TRAIN labels only.

    Args:
        y_train (array-like): Training-only documented activity labels.
        classes (sequence): Ordered, unique activity codes, all in Train.
        method (str): 'inverse_sqrt' or 'none'. The square-root approach
            moderates the influence of classes with very few observations.

    Returns:
        np.ndarray: float32 weights in the exact class order, normalised
            to have unweighted arithmetic mean one.

    Notes:
        No validation/test labels are used to compute these weights.
        These are window-frequency weights, not patient/subject-level
        prevalence estimates, and do not guarantee improved minority F1.
    """
    # 1. Require a complete documented training label set.
    labels = np.asarray(y_train)
    codes = list(classes)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("y_train must be a non-empty 1D label vector.")
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("classes must contain at least two distinct codes.")
    if not np.isin(labels, codes).all():
        raise ValueError("Training labels outside the declared classes.")
    counts = np.asarray([(labels == code).sum() for code in codes], dtype=np.float64)
    if (counts == 0).any():
        raise ValueError("Every declared class must be present in Train.")

    # 2. Compute a moderate inverse-square-root weighting from Train only.
    if method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
        weights /= weights.mean()
    elif method == "none":
        weights = np.ones(len(codes), dtype=np.float64)
    else:
        raise ValueError("method must be 'inverse_sqrt' or 'none'.")
    return weights.astype(np.float32)


# ======================== 4. Training and Evaluation =========================
def train_temporal_cnn(
    X_train, y_train, X_validation, y_validation,
    classes=tuple(range(1, 9)), epochs=5, batch_size=128,
    learning_rate=1e-3, seed=42, device="cpu",
    class_weighting="none",
):
    """
    Train one 1D-CNN using a subject-disjoint Train/Validation assignment.

    Args:
        X_train, X_validation (np.ndarray): (windows, channels, time).
        y_train, y_validation (array-like): Original documented activity codes.
        classes (sequence): Fixed ordering of original class codes.
        epochs (int): Training epochs; report the chosen setting.
        batch_size (int): Mini-batch size.
        learning_rate (float): Adam learning rate.
        seed (int): Python, NumPy and PyTorch seed.
        device (str): CPU default; CUDA may be specified if available.
        class_weighting (str): 'none' for unweighted Cross-Entropy or
            'inverse_sqrt' for weights derived from Train class counts only.

    Returns:
        model (nn.Module): Last-epoch trained model.
        history (pd.DataFrame): Epoch loss, validation Macro F1 and accuracy.
        prediction (np.ndarray): Last-epoch validation class-code predictions.
        scaler (dict): Training-only channel statistics.

    Notes:
        The function never sees reserved Test data. It reports the last epoch,
        rather than selecting a checkpoint with Validation and then claiming
        a separate independent validation score.
        Weighting is the ONLY training intervention in Stage 12; models
        otherwise retain the same seed, optimiser, data and architecture.
    """
    # 1. Verify class definitions and aligned input arrays.
    codes = list(classes)
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("At least two unique class codes are needed.")
    y_train, y_validation = np.asarray(y_train), np.asarray(y_validation)
    if y_train.ndim != 1 or y_validation.ndim != 1 or len(y_train) != len(X_train) or len(y_validation) != len(X_validation):
        raise ValueError("Window and label lengths must match.")
    if not np.isin(y_train, codes).all() or not np.isin(y_validation, codes).all():
        raise ValueError("Labels outside declared classes.")
    if not set(codes).issubset(set(y_train.tolist())):
        raise ValueError("Every declared class must be present in Train.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("The requested accelerator is unavailable.")

    # 2. Make initialisation and DataLoader shuffling reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Standardise channels from Train only; map codes to indices.
    train_scaled, validation_scaled, statistics = standardise_imu_train_only(
        X_train, X_validation
    )
    mapping = {code: idx for idx, code in enumerate(codes)}
    train_labels = np.array([mapping[code] for code in y_train], dtype=np.int64)
    training = TensorDataset(
        torch.from_numpy(train_scaled), torch.from_numpy(train_labels)
    )
    # 4. Make a loss-weight tensor using training labels only.
    class_weights = compute_training_class_weights(
        y_train, codes, method=class_weighting
    )
    loss_weights = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    model = TemporalCNN(train_scaled.shape[1], len(codes)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Learn the network on training participants with the chosen loss.
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(signals), labels, weight=loss_weights
            )
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach().cpu()) * len(labels)

        # 6. Evaluate on disjoint Validation participants only.
        model.eval()
        predicted_indices = []
        with torch.no_grad():
            val_tensor = torch.from_numpy(validation_scaled)
            for (start) in range(0, len(val_tensor), batch_size):
                logits = model(val_tensor[start:start + batch_size].to(device))
                predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
        prediction = np.asarray([codes[i] for i in predicted_indices])
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_scaled),
            "validation_macro_f1": f1_score(
                y_validation, prediction, labels=codes, average="macro", zero_division=0
            ),
            "validation_accuracy": accuracy_score(y_validation, prediction),
            "validation_balanced_accuracy": balanced_accuracy_score(y_validation, prediction),
        }
        log.append(metrics)
        print(
            f"Epoch {epoch}/{epochs} | loss={metrics['train_loss']:.4f} | "
            f"validation Macro F1={metrics['validation_macro_f1']:.4f}"
        )

    return model, pd.DataFrame(log), prediction, statistics


# ====================== 5. Step-Matched CNN Training =========================
def train_temporal_cnn_fixed_steps(
    X_train,
    y_train,
    X_validation,
    y_validation,
    classes=tuple(range(1, 9)),
    max_steps=600,
    eval_every=200,
    batch_size=128,
    learning_rate=1e-3,
    seed=42,
    device="cpu",
    class_weighting="inverse_sqrt",
):
    """
    Fit the existing TemporalCNN for an EXACT optimizer-update budget.

    Args:
        X_train (array-like): Training windows (N, channels, time).
        y_train (array-like): Documented original Train class codes.
        X_validation (array-like): Subject-disjoint validation windows.
        y_validation (array-like): Validation labels; never used to fit.
        classes (sequence): Fixed, distinct documented class codes.
        max_steps (int): Total number of optimizer updates for this run.
        eval_every (int): Evaluate after each number of updates and at end.
        batch_size (int): Training mini-batch size (fixed across conditions).
        learning_rate (float): Adam learning rate.
        seed (int): Fixed initialisation/DataLoader seed for paired runs.
        device (str): 'cpu' or available 'cuda'.
        class_weighting (str): Train-only 'inverse_sqrt' or 'none'.

    Returns:
        model (nn.Module): Model after the final optimizer update.
        history (pd.DataFrame): Optimizer step, recent Train loss, Validation
            Macro F1/accuracy/balanced accuracy at each evaluation.
        predictions (np.ndarray): Last-step validation source class codes.
        statistics (dict): Channel mean/SD estimated using Train only.

    Notes:
        - No validation/test signals or labels fit model/scaler/weights.
        - Mini-batch order reshuffles at the end of each DataLoader pass.
        - Equal steps AND batch size approximately equate presentations,
          not independent source windows nor unique observations.
        - Models with larger Train have fewer revisits per observation.
        - No checkpoint selection based on validation metrics.
    """
    # 1. Validate scientific task, aligned data and training controls.
    codes = list(classes)
    train_labels = np.asarray(y_train)
    validation_labels = np.asarray(y_validation)
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("At least two unique task classes are required.")
    if (
        train_labels.ndim != 1 or validation_labels.ndim != 1
        or len(train_labels) != len(X_train)
        or len(validation_labels) != len(X_validation)
    ):
        raise ValueError("Train and Validation window/label counts must align.")
    if not np.isin(train_labels, codes).all() or not np.isin(
        validation_labels, codes
    ).all():
        raise ValueError("Labels outside the declared task.")
    if not set(codes).issubset(set(train_labels.tolist())):
        raise ValueError("Every task class must be present in Train.")
    for name, count in [("max_steps", max_steps), ("eval_every", eval_every),
                        ("batch_size", batch_size)]:
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ) or count < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested PyTorch device is not available.")

    # 2. Seed the same model architecture and train-only preprocessing.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training, validation, statistics = standardise_imu_train_only(
        X_train, X_validation
    )
    code_to_index = {code: index for index, code in enumerate(codes)}
    encoded = np.asarray(
        [code_to_index[code] for code in train_labels], dtype=np.int64
    )
    dataset = TensorDataset(
        torch.from_numpy(training), torch.from_numpy(encoded)
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)

    model = TemporalCNN(training.shape[1], len(codes)).to(device)
    loss_weights = torch.as_tensor(
        compute_training_class_weights(train_labels, codes, class_weighting),
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Run the requested number of parameter updates, reshuffling each pass.
    log = []
    recent_losses = []
    predictions = None
    for step in range(1, int(max_steps) + 1):
        try:
            batch_signals, batch_labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_signals, batch_labels = next(iterator)
        model.train()
        batch_signals = batch_signals.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            model(batch_signals), batch_labels, weight=loss_weights
        )
        loss.backward()
        optimizer.step()
        recent_losses.append(float(loss.detach().cpu()))

        # 4. Evaluate only after predeclared training-update checkpoints.
        if step % eval_every != 0 and step != max_steps:
            continue
        model.eval()
        predicted_indices = []
        with torch.no_grad():
            val_tensor = torch.from_numpy(validation)
            for start in range(0, len(val_tensor), batch_size):
                logits = model(
                    val_tensor[start:start + batch_size].to(device)
                )
                predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
        predictions = np.asarray(
            [codes[index] for index in predicted_indices]
        )
        record = {
            "step": step,
            "train_loss_recent": float(np.mean(recent_losses)),
            "validation_macro_f1": f1_score(
                validation_labels, predictions, labels=codes,
                average="macro", zero_division=0
            ),
            "validation_accuracy": accuracy_score(
                validation_labels, predictions
            ),
            "validation_balanced_accuracy": balanced_accuracy_score(
                validation_labels, predictions
            ),
        }
        log.append(record)
        print(
            f"Step {step}/{max_steps} | loss={record['train_loss_recent']:.4f}"
            f" | Validation Macro F1={record['validation_macro_f1']:.4f}"
        )
        recent_losses.clear()

    return model, pd.DataFrame(log), predictions, statistics


# ======================== 6. Softmax Inference ===============================
def predict_temporal_cnn_probabilities(
    model, windows, train_statistics, classes,
    batch_size=256, device="cpu",
):
    """
    Obtain class-aligned softmax scores without fitting on inference data.

    Args:
        model (nn.Module): Already trained temporal classifier.
        windows (array-like): Held-out signal tensor (N, channels, time).
        train_statistics (dict): Train-only channel_mean/channel_std from
            train_temporal_cnn_fixed_steps() or train_temporal_cnn().
        classes (sequence): Ordered source class codes used during training.
        batch_size (int): Inference mini-batch size.
        device (str): Existing CPU or available CUDA device.

    Returns:
        np.ndarray: Float32 probabilities (N, classes), with columns in
            the supplied class-code order.

    Notes:
        Softmax values are NOT automatically calibrated probabilities,
        clinical certainty or measures of model epistemic uncertainty.
        No Validation/Test data are used to estimate the scaler or fit
        model parameters during inference.
    """
    # 1. Verify source tensor, class order and fitted Train statistics.
    X = np.asarray(windows, dtype=np.float32)
    codes = list(classes)
    if X.ndim != 3 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("windows must be a non-empty finite 3D array.")
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("classes must contain at least two unique codes.")
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a fitted PyTorch nn.Module.")
    if not isinstance(train_statistics, dict) or not {
        "channel_mean", "channel_std"
    }.issubset(train_statistics):
        raise ValueError("Train-only channel statistics are required.")
    mean = np.asarray(train_statistics["channel_mean"], dtype=np.float64)
    sd = np.asarray(train_statistics["channel_std"], dtype=np.float64)
    if mean.shape != (X.shape[1],) or sd.shape != (X.shape[1],):
        raise ValueError("Channel statistics do not match the input channels.")
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("Channel means/standard deviations must be valid.")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested PyTorch device is unavailable.")

    # 2. Transform with Train-only channel statistics and forward in eval mode.
    scaled = ((X.astype(np.float64) - mean[None, :, None]) /
              sd[None, :, None]).astype(np.float32)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(scaled[start:start + batch_size]).to(device)
            logits = model(batch)
            if logits.ndim != 2 or logits.shape[1] != len(codes):
                raise ValueError("Model logits do not match class ordering.")
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(scores, axis=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite softmax values.")
    return probabilities
