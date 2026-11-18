# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Fold-safe neural fitting, fixed-epoch final refits and inference costs."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ninapro_dataset import PairedWindowDataset, training_channel_stats
from ninapro_evaluation import participant_metrics
from ninapro_models import build_neural_model, count_parameters


# ========================== 01. Reproducible Fitting ==========================
def set_training_seed(seed: int):
    import random
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_logits(name: str, model, emg, acc):
    if "_emg_" in name:
        return model(emg)
    if "_acc_" in name:
        return model(acc)
    return model(emg, acc)


def _input_modality(model_name: str) -> str:
    """Select the used modality without changing the registered model."""
    if "_emg_" in model_name:
        return "emg"
    if "_acc_" in model_name:
        return "acc"
    return "both"


def _loader(table, processed_dir, classes, stats, *, batch_size, shuffle,
            model_name: str, device):
    """Prefetch only required source channels for the selected compute device."""
    use_cuda = torch.device(device).type == "cuda"
    workers = 2 if use_cuda else 0
    dataset = PairedWindowDataset(
        table, processed_dir, classes=classes,
        modality=_input_modality(model_name), **stats,
    )
    options = dict(
        batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=use_cuda,
    )
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


class _GPUAccelerationBatches:
    """Index contiguous GPU-resident acceleration windows in whole batches.

    Training-only centring has already been applied by PairedWindowDataset.
    This avoids per-window Python indexing and repeated source-file reads in
    every epoch. One batch is assembled with a single GPU index operation.
    """

    def __init__(self, acceleration, targets, *, batch_size: int, shuffle: bool):
        if len(acceleration) != len(targets):
            raise ValueError("Cached acceleration windows and labels differ.")
        self.acceleration = acceleration
        self.targets = targets
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = 0
        self.pin_memory = False

    def __len__(self):
        return (len(self.targets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.targets)
        device = self.acceleration.device
        order = (
            torch.randperm(n, device=device)
            if self.shuffle else torch.arange(n, device=device)
        )
        for indices in order.split(self.batch_size):
            batch_acc = self.acceleration.index_select(0, indices)
            batch_labels = self.targets.index_select(0, indices)
            unused_emg = torch.empty(
                (len(indices), 0, 0), dtype=torch.float32, device=device,
            )
            yield unused_emg, batch_acc, batch_labels


class _GPUPairedBatches:
    """Return paired float32 windows directly from fold-local GPU storage."""

    def __init__(self, emg, acceleration, targets, *, batch_size, shuffle):
        if not (len(emg) == len(acceleration) == len(targets)):
            raise ValueError("GPU paired cache has misaligned sample counts.")
        self.emg = emg
        self.acceleration = acceleration
        self.targets = targets
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = 0
        self.pin_memory = False

    def __len__(self):
        return (len(self.targets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.targets)
        device = self.targets.device
        indices = (
            torch.randperm(n, device=device)
            if self.shuffle else torch.arange(n, device=device)
        )
        for selected in indices.split(self.batch_size):
            yield (
                self.emg.index_select(0, selected),
                self.acceleration.index_select(0, selected),
                self.targets.index_select(0, selected),
            )


def _stage_paired_windows(
    table, processed_dir, classes, stats, *, batch_size, device, shuffle,
    model_name,
):
    """Stage identical source-aligned sEMG and acceleration windows once."""
    if table.empty:
        raise ValueError("Cannot stage empty paired windows.")
    lengths = (table.end_sample - table.start_sample).unique()
    if len(lengths) != 1 or int(lengths[0]) < 1:
        raise ValueError("Paired GPU staging requires a fixed window length.")
    n, length = len(table), int(lengths[0])
    emg = torch.empty((n, 12, length), dtype=torch.float32, device=device)
    acc = torch.empty((n, 36, length), dtype=torch.float32, device=device)
    targets = torch.empty(n, dtype=torch.long, device=device)
    source = _loader(
        table, processed_dir, classes, stats,
        batch_size=batch_size, shuffle=False, model_name=model_name,
        device=device,
    )
    offset = 0
    for source_emg, source_acc, labels in source:
        count = len(labels)
        emg[offset:offset + count].copy_(
            source_emg.to(device, non_blocking=True), non_blocking=True,
        )
        acc[offset:offset + count].copy_(
            source_acc.to(device, non_blocking=True), non_blocking=True,
        )
        targets[offset:offset + count].copy_(
            labels.to(device, non_blocking=True), non_blocking=True,
        )
        offset += count
    if offset != n:
        raise ValueError("Not all requested paired windows were staged.")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    del source
    return _GPUPairedBatches(
        emg, acc, targets, batch_size=batch_size, shuffle=shuffle,
    )


def _stage_acceleration_windows(
    table, processed_dir, classes, stats, *, batch_size, device,
    shuffle: bool,
):
    """Read each requested acceleration window once into contiguous GPU memory.

    The index order is preserved, and normalisation is identical to the
    existing per-window NumPy implementation. This helper can also be tested
    on a CPU with a small synthetic cohort; production staging is CUDA-only.
    """
    if table.empty:
        raise ValueError("Cannot stage empty acceleration windows.")
    lengths = (table.end_sample - table.start_sample).unique()
    if len(lengths) != 1 or int(lengths[0]) < 1:
        raise ValueError("Acceleration staging requires a fixed window length.")
    length = int(lengths[0])
    n = len(table)
    acceleration = torch.empty(
        (n, 36, length), dtype=torch.float32, device=device,
    )
    targets = torch.empty(n, dtype=torch.long, device=device)
    source = _loader(
        table, processed_dir, classes, stats, batch_size=batch_size,
        shuffle=False, model_name="D1_acc_cnn", device=device,
    )
    offset = 0
    for _, acc, label in source:
        batch = len(label)
        acceleration[offset:offset + batch].copy_(
            acc.to(device, non_blocking=True), non_blocking=True,
        )
        targets[offset:offset + batch].copy_(
            label.to(device, non_blocking=True), non_blocking=True,
        )
        offset += batch
    if offset != n:
        raise ValueError("Not all expected acceleration windows were staged.")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    del source
    return _GPUAccelerationBatches(
        acceleration, targets, batch_size=batch_size, shuffle=shuffle,
    )


def _can_stage_channels(fitting, inner, device, *, channels):
    """Reserve ample VRAM for convolution activations before caching windows."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return False, 0.0
    lengths = (fitting.end_sample - fitting.start_sample).unique()
    if len(lengths) != 1 or int(lengths[0]) < 1:
        return False, 0.0
    windows = len(fitting) + (0 if inner is None else len(inner))
    bytes_needed = windows * channels * int(lengths[0]) * 4
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = max(4 * (1024 ** 3), int(total_bytes * 0.25))
    safe_space = max(0, free_bytes - reserve_bytes)
    return bytes_needed <= safe_space * 0.75, bytes_needed / (1024 ** 3)


def _can_stage_acceleration(fitting, inner, device) -> tuple[bool, float]:
    """Preserve the tested acceleration-only capacity policy."""
    return _can_stage_channels(fitting, inner, device, channels=36)


def _can_stage_paired(fitting, inner, device) -> tuple[bool, float]:
    """Budget twelve sEMG and thirty-six acceleration channels jointly."""
    return _can_stage_channels(fitting, inner, device, channels=48)


def fit_neural(
    model_name: str, train_table: pd.DataFrame, processed_dir,
    classes: np.ndarray, *, device, seed: int = 42, epochs: int = 10,
    patience: int = 3, batch_size: int = 96, learning_rate: float = 1e-3,
    fixed_epochs: int | None = None, show_progress: bool = False,
):
    """Train on permitted subjects only; use a Training-only inner holdout.

    With fixed_epochs, train all supplied participants without early stopping.
    With no fixed_epochs, hold out 15% of supplied *training people* for
    checkpoint selection; neither outer Validation nor Test is consulted.
    """
    seed = int(seed)
    set_training_seed(seed)
    train_table = train_table.reset_index(drop=True)
    if train_table.empty:
        raise ValueError("Neural training needs at least one training window.")
    if fixed_epochs is None:
        people = np.random.default_rng(seed).permutation(sorted(train_table.subject_id.unique()))
        if len(people) < 3:
            raise ValueError("Inner stopping requires at least three Training participants.")
        holdout_count = max(1, int(round(len(people) * .15)))
        inner_ids = set(people[:holdout_count])
        inner = train_table.loc[train_table.subject_id.isin(inner_ids)].reset_index(drop=True)
        fitting = train_table.loc[~train_table.subject_id.isin(inner_ids)].reset_index(drop=True)
    else:
        inner = None
        fitting = train_table
    preparation_started = perf_counter()
    stats = training_channel_stats(
        fitting, processed_dir, modality=_input_modality(model_name),
    )
    train_loader, val_loader = None, None
    staged = False
    modality = _input_modality(model_name)
    if modality in ("acc", "both"):
        can_stage, required_gib = (
            _can_stage_acceleration(fitting, inner, device)
            if modality == "acc" else _can_stage_paired(fitting, inner, device)
        )
        cache_name = "Acceleration" if modality == "acc" else "Paired"
        if can_stage:
            if show_progress:
                print(
                    f"  {cache_name} cache: preparing {required_gib:.2f} GiB "
                    "of Training-normalised windows on CUDA...",
                    flush=True,
                )
            stage_started = perf_counter()
            try:
                def prepare(table, shuffle):
                    if modality == "acc":
                        return _stage_acceleration_windows(
                            table, processed_dir, classes, stats,
                            batch_size=batch_size, device=device, shuffle=shuffle,
                        )
                    return _stage_paired_windows(
                        table, processed_dir, classes, stats,
                        batch_size=batch_size, device=device, shuffle=shuffle,
                        model_name=model_name,
                    )
                train_loader = prepare(fitting, True)
                if inner is not None:
                    val_loader = prepare(inner, False)
                staged = True
            except torch.cuda.OutOfMemoryError:
                train_loader, val_loader = None, None
                torch.cuda.empty_cache()
                if show_progress:
                    print(
                        f"  {cache_name} cache exceeded available VRAM; "
                        "using streamed input windows.",
                        flush=True,
                    )
            if staged and show_progress:
                print(
                    f"  {cache_name} cache ready in "
                    f"{perf_counter() - stage_started:.1f} s; "
                    "reused across the fold's training epochs.",
                    flush=True,
                )
        elif show_progress and torch.device(device).type == "cuda":
            print(
                f"  {cache_name} cache: {required_gib:.2f} GiB required; "
                "using streamed input windows to preserve GPU memory.",
                flush=True,
            )
    if train_loader is None:
        train_loader = _loader(
            fitting, processed_dir, classes, stats, batch_size=batch_size,
            shuffle=True, model_name=model_name, device=device,
        )
        val_loader = None if inner is None else _loader(
            inner, processed_dir, classes, stats, batch_size=batch_size,
            shuffle=False, model_name=model_name, device=device,
        )
    if show_progress:
        print(
            f"  Data loader: {modality} "
            f"| {'GPU-cached' if staged else 'streamed'} "
            f"| workers={train_loader.num_workers} "
            f"| pinned memory={train_loader.pin_memory} "
            f"| batch size={batch_size}",
            flush=True,
        )
    preparation_seconds = perf_counter() - preparation_started
    model = build_neural_model(model_name, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-3)
    codes = {int(v): i for i, v in enumerate(classes)}
    counts = np.bincount([codes[int(v)] for v in fitting.label], minlength=len(classes))
    weight = np.zeros(len(classes), np.float32)
    present = counts > 0
    weight[present] = len(fitting) / (counts[present] * max(1, present.sum()))
    criterion_weight = torch.from_numpy(weight).to(device)
    best_score, best_state, best_epoch, misses = -np.inf, None, 0, 0
    history = []
    start = perf_counter()
    n_epochs = int(fixed_epochs if fixed_epochs is not None else epochs)
    for epoch in range(1, n_epochs + 1):
        epoch_started = perf_counter()
        model.train()
        # Accumulate training loss on-device; synchronise once per epoch,
        # rather than transferring each minibatch's loss to the CPU.
        accumulated_loss = torch.zeros((), device=device)
        number_of_batches = 0
        for emg, acc, label in train_loader:
            if "_emg_" in model_name:
                emg = emg.to(device, non_blocking=True)
            elif "_acc_" in model_name:
                acc = acc.to(device, non_blocking=True)
            else:
                emg = emg.to(device, non_blocking=True)
                acc = acc.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model_name, model, emg, acc)
            loss = F.cross_entropy(logits, label, weight=criterion_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            accumulated_loss.add_(loss.detach())
            number_of_batches += 1
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = perf_counter() - epoch_started
        validation_started = perf_counter()
        val_f1 = np.nan
        if val_loader is not None:
            probs = _predict_loader(model_name, model, val_loader, device)
            val_f1 = float(participant_metrics(
                inner.reset_index(drop=True), probs, classes,
                model_id=model_name, seed=seed,
            ).macro_f1.mean())
            if val_f1 > best_score + 1e-6:
                best_score, best_epoch, misses = val_f1, epoch, 0
                best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                misses += 1
        validation_seconds = perf_counter() - validation_started
        epoch_loss = float((accumulated_loss / number_of_batches).item())
        epoch_seconds = perf_counter() - epoch_started
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "inner_validation_macro_f1": val_f1, "model_id": model_name,
                        "seed": seed, "training_seconds": train_seconds,
                        "inner_validation_seconds": validation_seconds,
                        "epoch_seconds": epoch_seconds})
        if show_progress:
            validation_text = (
                f"{val_f1:.4f}" if np.isfinite(val_f1) else "not evaluated"
            )
            print(
                f"  {model_name} | seed={seed} | epoch {epoch}/{n_epochs} "
                f"| loss={epoch_loss:.4f} | inner macro F1={validation_text} "
                f"| train={train_seconds:.1f} s | "
                f"inner validation={validation_seconds:.1f} s | "
                f"total={epoch_seconds:.1f} s",
                flush=True,
            )
        if val_loader is not None and misses >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    fit_seconds = perf_counter() - start
    return model, stats, pd.DataFrame(history), {
        "model_id": model_name, "seed": seed, "best_epochs": int(best_epoch),
        "fit_seconds": float(fit_seconds),
        "parameter_count": count_parameters(model),
        "data_preparation_seconds": float(preparation_seconds),
        "acceleration_gpu_cached": bool(staged and modality == "acc"),
        "paired_gpu_cached": bool(staged and modality == "both"),
        "inner_validation_people": 0 if inner is None else inner.subject_id.nunique(),
        "device": str(device),
    }


# ========================== 02. Paired Probability Inference =================
@torch.inference_mode()
def _predict_loader(name, model, loader, device, *, perturbation: str | None = None):
    model.eval()
    probabilities = []
    for emg, acc, _ in loader:
        if "_emg_" in name:
            emg = emg.to(device, non_blocking=True)
        elif "_acc_" in name:
            acc = acc.to(device, non_blocking=True)
        else:
            emg = emg.to(device, non_blocking=True)
            acc = acc.to(device, non_blocking=True)
        if perturbation == "acc_missing_mean":
            acc = torch.zeros_like(acc)
        elif perturbation == "acc_sensor_01_missing_mean":
            acc = acc.clone()
            acc[:, :3, :] = 0.0
        elif perturbation == "acc_contiguous_gap_mean":
            acc = acc.clone()
            gap = max(1, acc.shape[-1] // 4)
            acc[:, :, :gap] = 0.0
        logits = _model_logits(name, model, emg, acc)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 0), float)


def predict_neural(name, model, table, processed_dir, classes, stats, *,
                   device, batch_size=128, perturbation=None):
    loader = _loader(
        table.reset_index(drop=True), processed_dir, classes, stats,
        batch_size=batch_size, shuffle=False, model_name=name, device=device,
    )
    return _predict_loader(name, model, loader, device, perturbation=perturbation)


def benchmark_cpu(name, model, table, processed_dir, classes, stats, *, windows=32):
    """Measure CPU inference after model transfer, excluding data initialisation."""
    if table.empty:
        return float("nan")
    cpu_model = deepcopy(model).to("cpu").eval()
    subset = table.iloc[:min(windows, len(table))].reset_index(drop=True)
    loader = _loader(
        subset, processed_dir, classes, stats,
        batch_size=min(windows, len(table)), shuffle=False,
        model_name=name, device=torch.device("cpu"),
    )
    with torch.inference_mode():
        emg, acc, _ = next(iter(loader))
        for _ in range(2):
            _model_logits(name, cpu_model, emg, acc)
        times = []
        for _ in range(5):
            started = perf_counter()
            _model_logits(name, cpu_model, emg, acc)
            times.append(perf_counter() - started)
    return 1000 * float(np.median(times)) / len(subset)
