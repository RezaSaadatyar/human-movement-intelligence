# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Construct subject-disjoint, label-pure paired NinaPro DB2 windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ninapro_source import RELEASED_RATE_HZ, ExerciseRecord, contiguous_segments


# ========================= 01. Pairing and Window Index =======================
WINDOW_MS = 300
STEP_MS = 100


def build_window_registry(
    record: ExerciseRecord, allowed_labels: set[int], *,
    window_ms: int = WINDOW_MS, step_ms: int = STEP_MS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index label-pure time intervals without crossing corrected segments.

    A window covers [start_sample, end_sample) in the released 2000-Hz grid.
    All modality arrays and targets share the same original sample indices.
    """
    window = int(round(window_ms * RELEASED_RATE_HZ / 1000))
    step = int(round(step_ms * RELEASED_RATE_HZ / 1000))
    if window < 1 or step < 1:
        raise ValueError("Window and step must be positive source-sample spans.")
    valid_samples = np.isfinite(record.emg).all(axis=1) & np.isfinite(record.acc).all(axis=1)
    invalid_cumsum = np.r_[0, np.cumsum(~valid_samples, dtype=np.int64)]
    entries, segments = [], []
    for segment_number, (start, end, label, repetition) in enumerate(
        contiguous_segments(record.restimulus, record.rerepetition), start=1
    ):
        segment_key = f"{record.subject_id}_E{record.exercise_id}_G{segment_number:05d}"
        possible = max(0, 1 + (end - start - window) // step)
        retained = 0
        if label in allowed_labels and possible:
            for a in range(start, end - window + 1, step):
                b = a + window
                if invalid_cumsum[b] > invalid_cumsum[a]:
                    continue
                entries.append({
                    "window_key": f"{segment_key}_{a:09d}",
                    "subject_id": record.subject_id,
                    "exercise_id": record.exercise_id,
                    "segment_id": segment_key,
                    "repetition_id": repetition,
                    "label": label,
                    "start_sample": a,
                    "end_sample": b,
                    "start_s": a / RELEASED_RATE_HZ,
                    "end_s": b / RELEASED_RATE_HZ,
                    "source_file": record.source_name,
                })
                retained += 1
        segments.append({
            "segment_id": segment_key, "subject_id": record.subject_id,
            "exercise_id": record.exercise_id, "label": label,
            "repetition_id": repetition, "start_sample": start,
            "end_sample": end, "duration_s": (end - start) / RELEASED_RATE_HZ,
            "potential_windows": possible if label in allowed_labels else 0,
            "valid_windows": retained,
        })
    return pd.DataFrame(entries), pd.DataFrame(segments)


def select_windows_uniformly(index: pd.DataFrame, cap_per_subject: int | None) -> pd.DataFrame:
    """Apply a predeclared time-order subsample, independent of model scores.

    This leaves per-subject window counts comparable; the entire E1 exercise
    is still inspected, with time-uniform retained windows across movements.
    """
    if cap_per_subject is None:
        selected = index.copy()
    else:
        if cap_per_subject < 1:
            raise ValueError("cap_per_subject must be positive or None.")
        pieces = []
        for _, part in index.groupby("subject_id", sort=True):
            if len(part) > cap_per_subject:
                keep = np.linspace(0, len(part) - 1, cap_per_subject, dtype=int)
                part = part.iloc[np.unique(keep)]
            pieces.append(part)
        selected = pd.concat(pieces, ignore_index=True)
    selected = selected.reset_index(drop=True)
    selected.insert(0, "window_row", np.arange(len(selected), dtype=np.int64))
    if not selected.window_key.is_unique:
        raise ValueError("Duplicate source window keys are not permitted.")
    return selected


# ======================== 02. Fixed Feature Extraction =======================
def window_features(
    emg: np.ndarray, acc: np.ndarray, starts: np.ndarray, *, length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fold-independent features on exactly the same paired windows."""
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        return np.empty((0, 48), np.float32), np.empty((0, 96), np.float32)
    idx = starts[:, None] + np.arange(length, dtype=np.int64)[None, :]
    x = np.asarray(emg[idx], dtype=np.float32)
    a = np.asarray(acc[idx], dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("The feature extractor received an invalid paired window.")
    rms = np.sqrt(np.mean(x * x, axis=1))
    mav = np.mean(np.abs(x), axis=1)
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1) / max(1, length - 1)
    logvar = np.log(np.var(x, axis=1) + 1e-12)
    # Channel-major order: RMS, MAV, waveform length, log variance per channel.
    emg_f = np.stack([rms, mav, wl, logvar], axis=-1).reshape(len(x), -1)
    means = np.mean(a, axis=1)
    stds = np.std(a, axis=1)
    vectors = a.reshape(len(a), length, 12, 3)
    magnitude = np.linalg.norm(vectors, axis=3)
    mag_mean = magnitude.mean(axis=1)
    mag_std = magnitude.std(axis=1)
    acc_f = np.concatenate([means, stds, mag_mean, mag_std], axis=1)
    return emg_f.astype(np.float32), acc_f.astype(np.float32)


def save_source_arrays(record: ExerciseRecord, processed_dir: Path) -> None:
    """Save original paired channel values; preserve raw files separately."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for field in ("emg", "acc"):
        destination = processed_dir / f"02_{record.subject_id}_E{record.exercise_id}_{field}.npy"
        tmp = destination.with_name(destination.name + ".part")
        with tmp.open("wb") as handle:
            np.save(handle, np.asarray(getattr(record, field), dtype=np.float32))
        tmp.replace(destination)


def construct_feature_arrays(
    index: pd.DataFrame, processed_dir: Path, *, batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned features from source-length arrays in bounded batches."""
    if index.empty:
        raise ValueError("No eligible paired windows were found.")
    emg_features = np.empty((len(index), 48), np.float32)
    acc_features = np.empty((len(index), 96), np.float32)
    for subject, part in index.groupby("subject_id", sort=True):
        emg = np.load(Path(processed_dir) / f"02_{subject}_E1_emg.npy", mmap_mode="r")
        acc = np.load(Path(processed_dir) / f"02_{subject}_E1_acc.npy", mmap_mode="r")
        for k in range(0, len(part), batch_size):
            batch = part.iloc[k:k + batch_size]
            starts = batch.start_sample.to_numpy(np.int64)
            sizes = (batch.end_sample - batch.start_sample).unique()
            if len(sizes) != 1:
                raise ValueError("Mixed window lengths are not supported in this study run.")
            fe, fa = window_features(emg, acc, starts, length=int(sizes[0]))
            rows = batch.window_row.to_numpy(np.int64)
            emg_features[rows], acc_features[rows] = fe, fa
    return emg_features, acc_features


# ====================== 03. Participant-Disjoint Assignment ==================
def assign_participants(
    index: pd.DataFrame, *, test_people: int = 5, folds: int = 5, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze person roles and Development validation folds before fitting."""
    participants = sorted(index.subject_id.unique().tolist())
    if len(participants) < test_people + folds:
        raise ValueError(
            f"Need at least {test_people + folds} eligible people for "
            f"{test_people} Test people and {folds} Development folds; "
            f"found {len(participants)}. Revise the design explicitly."
        )
    shuffled = np.random.default_rng(seed).permutation(participants).tolist()
    test_ids = set(shuffled[:test_people])
    development = sorted(set(participants) - test_ids)
    split = pd.DataFrame({
        "subject_id": participants,
        "role": ["Test" if s in test_ids else "Development" for s in participants],
    })
    fold_groups = np.array_split(np.random.default_rng(seed + 1).permutation(development), folds)
    fold_assignment = pd.DataFrame([
        {"subject_id": str(person), "validation_fold": int(fold)}
        for fold, people in enumerate(fold_groups) for person in people
    ]).sort_values("subject_id").reset_index(drop=True)
    if set(split.loc[split.role == "Test", "subject_id"]) & set(fold_assignment.subject_id):
        raise AssertionError("Test people must not appear in Development fold assignments.")
    return split, fold_assignment


def check_artifact_alignment(index: pd.DataFrame, features: tuple[np.ndarray, np.ndarray]) -> None:
    """Guard row identity, label purity and pairing before any model fit."""
    emg, acc = features
    if len(index) != len(emg) or len(index) != len(acc):
        raise ValueError("Window rows and unimodal feature arrays are misaligned.")
    if not index.window_key.is_unique:
        raise ValueError("Duplicate window keys indicate possible cross-record pairing.")
    if not np.array_equal(index.window_row.to_numpy(), np.arange(len(index))):
        raise ValueError("Window rows are not in their canonical source order.")
    if not np.isfinite(emg).all() or not np.isfinite(acc).all():
        raise ValueError("Feature arrays contain non-finite values.")
    if not (index.end_sample > index.start_sample).all():
        raise ValueError("Found an invalid half-open window interval.")


# ========================= 04. Lazy Neural Windows ===========================
class PairedWindowDataset:
    """Read original source-grid windows lazily from memory-mapped subject files.

    A model receives source rows [time, channel] for the same physical window.
    No signal from other subjects, labels or cue timing enters the tensors.
    """

    def __init__(self, window_index: pd.DataFrame, processed_dir: Path, *,
                 emg_mean=None, emg_std=None, acc_mean=None, acc_std=None,
                 classes=None):
        self.table = window_index.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.emg_mean = np.zeros(12, np.float32) if emg_mean is None else np.asarray(emg_mean, np.float32)
        self.emg_std = np.ones(12, np.float32) if emg_std is None else np.asarray(emg_std, np.float32)
        self.acc_mean = np.zeros(36, np.float32) if acc_mean is None else np.asarray(acc_mean, np.float32)
        self.acc_std = np.ones(36, np.float32) if acc_std is None else np.asarray(acc_std, np.float32)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.class_to_index = {int(label): i for i, label in enumerate(classes)} if classes is not None else None

    def __len__(self) -> int:
        return len(self.table)

    def _arrays(self, subject: str) -> tuple[np.ndarray, np.ndarray]:
        if subject not in self._cache:
            self._cache[subject] = (
                np.load(self.processed_dir / f"02_{subject}_E1_emg.npy", mmap_mode="r"),
                np.load(self.processed_dir / f"02_{subject}_E1_acc.npy", mmap_mode="r"),
            )
        return self._cache[subject]

    def __getitem__(self, idx: int):
        import torch
        row = self.table.iloc[int(idx)]
        emg, acc = self._arrays(str(row.subject_id))
        a, b = int(row.start_sample), int(row.end_sample)
        x = (np.asarray(emg[a:b], np.float32) - self.emg_mean) / np.maximum(self.emg_std, 1e-5)
        z = (np.asarray(acc[a:b], np.float32) - self.acc_mean) / np.maximum(self.acc_std, 1e-5)
        return (torch.from_numpy(np.ascontiguousarray(x.T)),
                torch.from_numpy(np.ascontiguousarray(z.T)),
                torch.tensor(
                    int(row.label) if self.class_to_index is None
                    else self.class_to_index[int(row.label)], dtype=torch.long))


def training_channel_stats(index: pd.DataFrame, processed_dir: Path, *,
                           max_windows: int = 4000, seed: int = 42) -> dict[str, np.ndarray]:
    """Fit per-channel centring/scaling from Training people only.

    Deterministic time-uniform subset bounds the scan; no Test rows can be
    supplied by the caller for a legitimate inductive evaluation.
    """
    if index.empty:
        raise ValueError("Cannot estimate channel statistics from an empty training set.")
    sample = np.linspace(0, len(index) - 1, min(len(index), max_windows), dtype=int)
    sample_rows = index.iloc[np.unique(sample)]
    totals = {key: np.zeros(n, np.float64) for key, n in (("emg", 12), ("acc", 36))}
    squares = {key: np.zeros(n, np.float64) for key, n in (("emg", 12), ("acc", 36))}
    count = 0
    caches: dict[tuple[str, str], np.ndarray] = {}
    for row in sample_rows.itertuples(index=False):
        length = int(row.end_sample - row.start_sample)
        for name in ("emg", "acc"):
            key = (row.subject_id, name)
            if key not in caches:
                caches[key] = np.load(
                    Path(processed_dir) / f"02_{row.subject_id}_E1_{name}.npy", mmap_mode="r"
                )
            block = np.asarray(caches[key][row.start_sample:row.end_sample], np.float64)
            totals[name] += block.sum(axis=0)
            squares[name] += np.square(block).sum(axis=0)
        count += length
    result = {}
    for name in ("emg", "acc"):
        mean = totals[name] / count
        var = np.maximum(0, squares[name] / count - mean ** 2)
        result[f"{name}_mean"] = mean.astype(np.float32)
        result[f"{name}_std"] = np.maximum(np.sqrt(var), 1e-5).astype(np.float32)
    return result

# ==================== 05. Reconstruct Local Numerical Cache ===================
def ensure_numerical_cache(
    root: Path, index: pd.DataFrame, *,
    allow_source_download: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse aligned feature/source arrays; rebuild only missing numerical inputs.

    The canonical Notebook 02 window index and its participant assignments are
    not modified. Logging reports whether the executing runtime actually has
    the saved source and feature arrays. It does not require the original ZIP
    archives when valid processed arrays are already available.
    """
    from time import perf_counter
    from ninapro_source import ensure_subject_archive, load_exercise

    root = Path(root)
    raw = root / "data" / "raw" / "ninapro_db2"
    processed = root / "data" / "processed" / "ninapro_db2"
    processed.mkdir(parents=True, exist_ok=True)

    # Derive required source spans once, instead of repeatedly filtering all
    # 40,000 window rows for each participant.
    maximum_by_subject = index.groupby("subject_id", sort=True)["end_sample"].max()
    required_pairs = [
        (
            str(subject_id),
            int(str(subject_id).removeprefix("S")),
            int(maximum),
            processed / f"02_{subject_id}_E1_emg.npy",
            processed / f"02_{subject_id}_E1_acc.npy",
        )
        for subject_id, maximum in maximum_by_subject.items()
    ]
    ready_count = sum(
        emg_path.is_file() and acc_path.is_file()
        for _, _, _, emg_path, acc_path in required_pairs
    )
    emg_file = processed / "02_emg_features.npy"
    acc_file = processed / "02_acc_features.npy"

    if verbose:
        print(f"Numerical cache directory: {processed}", flush=True)
        print(
            f"Paired E1 source arrays: {ready_count}/{len(required_pairs)} "
            "participants present.",
            flush=True,
        )
        print(
            "Statistical feature files: "
            f"sEMG={'available' if emg_file.is_file() else 'missing'}, "
            f"Acc={'available' if acc_file.is_file() else 'missing'}.",
            flush=True,
        )

    # Deep temporal models require the saved source-grid arrays, even when
    # the compact classical feature matrices are already available.
    for subject_id, subject, maximum, emg_path, acc_path in required_pairs:
        if not emg_path.is_file() or not acc_path.is_file():
            started = perf_counter()
            if verbose:
                print(
                    f"{subject_id}: missing processed source arrays; "
                    "recovering E1 from the local archive or official source...",
                    flush=True,
                )
            source = ensure_subject_archive(
                raw, subject, download=allow_source_download
            )
            record = load_exercise(source, subject, exercise=1)
            save_source_arrays(record, processed)
            if verbose:
                print(
                    f"{subject_id}: paired arrays ready "
                    f"({perf_counter() - started:.1f} s).",
                    flush=True,
                )

        # Opening .npy files with mmap_mode reads array metadata rather than
        # materialising all source samples in RAM.
        emg = np.load(emg_path, mmap_mode="r", allow_pickle=False)
        acc = np.load(acc_path, mmap_mode="r", allow_pickle=False)
        if (
            emg.ndim != 2 or acc.ndim != 2
            or emg.shape[1] != 12 or acc.shape[1] != 36
            or maximum > len(emg) or maximum > len(acc)
        ):
            raise ValueError(
                f"{subject_id}: cached source arrays are incompatible with "
                "Notebook 02's window boundaries or sensor dimensions."
            )
        del emg, acc

    if verbose:
        print(
            f"Validated paired source arrays for {len(required_pairs)} "
            "participants.",
            flush=True,
        )

    # Existing valid features are loaded without recomputation; both arrays
    # must retain the exact number/order of Notebook 02's window rows.
    if emg_file.is_file() and acc_file.is_file():
        try:
            emg_features = np.load(
                emg_file, mmap_mode="r", allow_pickle=False
            )
            acc_features = np.load(
                acc_file, mmap_mode="r", allow_pickle=False
            )
            check_artifact_alignment(index, (emg_features, acc_features))
            if emg_features.shape[1] != 48 or acc_features.shape[1] != 96:
                raise ValueError("Unexpected saved feature dimensions.")
        except (ValueError, OSError, EOFError) as exc:
            if verbose:
                print(
                    f"Existing feature cache could not be reused ({exc}). "
                    "Recomputing features from the fixed window index...",
                    flush=True,
                )
        else:
            if verbose:
                print(
                    "Reused both saved statistical feature arrays; "
                    "no feature extraction required.",
                    flush=True,
                )
            return emg_features, acc_features

    if verbose:
        print(
            "Extracting statistical features for the fixed paired windows...",
            flush=True,
        )
    started = perf_counter()
    emg_features, acc_features = construct_feature_arrays(index, processed)
    check_artifact_alignment(index, (emg_features, acc_features))
    np.save(emg_file, emg_features)
    np.save(acc_file, acc_features)
    if verbose:
        print(
            f"Saved statistical features ({perf_counter() - started:.1f} s).",
            flush=True,
        )
    return emg_features, acc_features
