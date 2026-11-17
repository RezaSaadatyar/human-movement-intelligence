"""Synthetic integration tests for NinaPro source pairing and model contracts.

These tests do not contain, redistribute or pretend to reproduce NinaPro data.
"""

from pathlib import Path
import io
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

STUDY_SRC = Path(__file__).resolve().parents[1] / "multimodal_movement_learning" / "code" / "src"
sys.path.insert(0, str(STUDY_SRC))

from ninapro_source import load_exercise, contiguous_segments
from ninapro_dataset import (
    build_window_registry, select_windows_uniformly, window_features,
    save_source_arrays, construct_feature_arrays, assign_participants,
    check_artifact_alignment, PairedWindowDataset, ensure_numerical_cache,
    training_channel_stats,
)
from ninapro_evaluation import (
    build_classical, aligned_probabilities, participant_metrics,
    paired_subject_bootstrap, fixed_class_metrics,
)
from ninapro_models import UNIMODAL_MODELS, FUSION_MODELS, build_neural_model
from ninapro_training import fit_neural, predict_neural


def _fake_archive(location: Path, person: int, *, seed: int):
    rng = np.random.default_rng(seed)
    n = 4500
    labels = np.repeat([0, 1, 0, 2, 3], 900)
    repetitions = np.repeat([0, 1, 0, 2, 3], 900)
    emg = rng.normal(0, .3, size=(n, 12)).astype(np.float32)
    acc = rng.normal(0, 1, size=(n, 36)).astype(np.float32)
    # The two sensor arrays share sample rows and corrected labels.
    payload = {
        "emg": emg, "acc": acc,
        "stimulus": labels.reshape(-1, 1),
        "restimulus": labels.reshape(-1, 1),
        "repetition": repetitions.reshape(-1, 1),
        "rerepetition": repetitions.reshape(-1, 1),
    }
    buffer = io.BytesIO()
    savemat(buffer, payload)
    path = location / f"DB2_s{person}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"DB2/S{person}_E1_A1.mat", buffer.getvalue())
    return path


def _cohort(tmp_path, people=10):
    processed = tmp_path / "data" / "processed" / "ninapro_db2"
    raw = tmp_path / "data" / "raw" / "ninapro_db2"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    pieces = []
    for person in range(1, people + 1):
        src = _fake_archive(raw, person, seed=person)
        record = load_exercise(src, person, exercise=1)
        assert record.emg.shape == (4500, 12)
        assert record.acc.shape == (4500, 36)
        index, segments = build_window_registry(record, {0, 1, 2, 3})
        assert not index.empty
        assert (index.end_sample - index.start_sample == 600).all()
        assert len(segments) == 5
        pieces.append(index)
        save_source_arrays(record, processed)
    canonical = select_windows_uniformly(pd.concat(pieces, ignore_index=True), cap_per_subject=10)
    fe, fa = construct_feature_arrays(canonical, processed)
    return canonical, processed, fe, fa


def test_source_zip_pairing_and_segment_identity(tmp_path):
    src = _fake_archive(tmp_path, 1, seed=1)
    record = load_exercise(src, 1)
    assert record.restimulus.shape == (4500,)
    assert contiguous_segments(record.restimulus, record.rerepetition) == [
        (0, 900, 0, 0), (900, 1800, 1, 1), (1800, 2700, 0, 0),
        (2700, 3600, 2, 2), (3600, 4500, 3, 3),
    ]
    index, _ = build_window_registry(record, {0, 1, 2, 3})
    assert index.window_key.is_unique
    # Every window is fully contained in a corrected-label/repetition segment.
    for row in index.itertuples(index=False):
        assert np.unique(record.restimulus[row.start_sample:row.end_sample]).tolist() == [row.label]


def test_feature_alignment_and_frozen_person_groups(tmp_path):
    index, processed, fe, fa = _cohort(tmp_path)
    assert fe.shape == (100, 48)
    assert fa.shape == (100, 96)
    check_artifact_alignment(index, (fe, fa))
    split, folds = assign_participants(index, test_people=2, folds=4, seed=42)
    test_people = set(split.loc[split.role == "Test", "subject_id"])
    assert test_people.isdisjoint(set(folds.subject_id))
    assert folds.subject_id.nunique() == 8
    dataset = PairedWindowDataset(index.iloc[:5], processed, classes=np.array([0, 1, 2, 3]))
    x, a, y = dataset[0]
    assert tuple(x.shape) == (12, 600)
    assert tuple(a.shape) == (36, 600)
    assert y.item() in {0, 1, 2, 3}


def test_fixed_class_probabilities_and_paired_metrics(tmp_path):
    index, processed, fe, fa = _cohort(tmp_path)
    classes = np.array([0, 1, 2, 3])
    model = build_classical("B1_logreg")
    model.fit(fe[:60], index.label.to_numpy()[:60])
    probs = aligned_probabilities(model, fe[60:], classes)
    assert np.allclose(probs.sum(axis=1), 1)
    rows = index.iloc[60:].reset_index(drop=True)
    scores = participant_metrics(rows, probs, classes, model_id="B1_logreg")
    assert len(scores) == rows.subject_id.nunique()
    assert np.isfinite(scores.macro_f1).all()
    ci = paired_subject_bootstrap(np.array([-.05, .04, .0]))
    assert ci["independent_test_people"] == 3


def test_all_neural_forward_interfaces():
    import torch
    emg = torch.randn(2, 12, 600)
    acc = torch.randn(2, 36, 600)
    for name in (*UNIMODAL_MODELS, *FUSION_MODELS):
        model = build_neural_model(name, 4).eval()
        with torch.no_grad():
            if "_emg_" in name:
                output = model(emg)
            elif "_acc_" in name:
                output = model(acc)
            else:
                output = model(emg, acc)
        assert output.shape == (2, 4), name
        assert np.isfinite(output.numpy()).all(), name


def test_neural_training_and_inference_with_synthetic_people(tmp_path):
    import torch
    index, processed, fe, fa = _cohort(tmp_path, people=7)
    training = index.loc[index.subject_id.isin(["S01", "S02", "S03", "S04", "S05"])].copy()
    validation = index.loc[index.subject_id == "S06"].copy()
    classes = np.array([0, 1, 2, 3])
    model, stats, history, info = fit_neural(
        "D1_emg_cnn", training, processed, classes,
        device=torch.device("cpu"), seed=4, fixed_epochs=1, batch_size=16,
    )
    probs = predict_neural("D1_emg_cnn", model, validation, processed,
                           classes, stats, device=torch.device("cpu"))
    assert probs.shape == (len(validation), len(classes))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert info["best_epochs"] == 1
    assert not history.empty


def test_numerical_cache_reuses_existing_features_without_recomputation(
    tmp_path, monkeypatch, capsys,
):
    """Complete local caches should not read ZIPs or extract features again."""
    import ninapro_dataset
    import ninapro_source

    index, processed, expected_emg, expected_acc = _cohort(tmp_path, people=3)
    np.save(processed / "02_emg_features.npy", expected_emg)
    np.save(processed / "02_acc_features.npy", expected_acc)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("A complete cache must not require reconstruction.")

    monkeypatch.setattr(ninapro_dataset, "construct_feature_arrays", unexpected_work)
    monkeypatch.setattr(ninapro_source, "ensure_subject_archive", unexpected_work)

    emg, acc = ensure_numerical_cache(
        tmp_path, index, allow_source_download=False, verbose=True,
    )
    np.testing.assert_allclose(emg, expected_emg)
    np.testing.assert_allclose(acc, expected_acc)
    output = capsys.readouterr().out
    assert "Paired E1 source arrays: 3/3 participants present." in output
    assert "Reused both saved statistical feature arrays" in output


def test_numerical_cache_recovers_only_missing_subject_arrays(
    tmp_path, capsys,
):
    """A missing local array is regenerated from its existing source ZIP."""
    index, processed, expected_emg, expected_acc = _cohort(tmp_path, people=3)
    np.save(processed / "02_emg_features.npy", expected_emg)
    np.save(processed / "02_acc_features.npy", expected_acc)
    (processed / "02_S02_E1_acc.npy").unlink()

    emg, acc = ensure_numerical_cache(
        tmp_path, index, allow_source_download=False, verbose=True,
    )
    assert (processed / "02_S02_E1_emg.npy").is_file()
    assert (processed / "02_S02_E1_acc.npy").is_file()
    np.testing.assert_allclose(emg, expected_emg)
    np.testing.assert_allclose(acc, expected_acc)
    output = capsys.readouterr().out
    assert "Paired E1 source arrays: 2/3 participants present." in output
    assert "S02: missing processed source arrays" in output
    assert "Reused both saved statistical feature arrays" in output


def test_numerical_cache_recreates_missing_feature_matrices(tmp_path, capsys):
    """Feature extraction happens only if required saved matrices are absent."""
    index, processed, expected_emg, expected_acc = _cohort(tmp_path, people=3)
    emg, acc = ensure_numerical_cache(
        tmp_path, index, allow_source_download=False, verbose=True,
    )
    np.testing.assert_allclose(emg, expected_emg)
    np.testing.assert_allclose(acc, expected_acc)
    assert (processed / "02_emg_features.npy").is_file()
    assert (processed / "02_acc_features.npy").is_file()
    assert "Saved statistical features" in capsys.readouterr().out


def test_modality_specific_windows_match_original_paired_tensor(tmp_path):
    """Reading one modality leaves its original samples and targets unchanged."""
    import torch

    index, processed, _, _ = _cohort(tmp_path, people=3)
    subset = index.iloc[:6]
    labels = np.array([0, 1, 2, 3])
    paired = PairedWindowDataset(subset, processed, classes=labels)
    emg_only = PairedWindowDataset(subset, processed, classes=labels, modality="emg")
    acc_only = PairedWindowDataset(subset, processed, classes=labels, modality="acc")
    for i in range(len(subset)):
        full_emg, full_acc, target = paired[i]
        emg, empty_acc, emg_target = emg_only[i]
        empty_emg, acc, acc_target = acc_only[i]
        torch.testing.assert_close(emg, full_emg, rtol=0, atol=0)
        torch.testing.assert_close(acc, full_acc, rtol=0, atol=0)
        assert empty_acc.shape == (0, 0) and empty_emg.shape == (0, 0)
        assert emg_target.item() == acc_target.item() == target.item()
    assert all(acc is None for _, acc in emg_only._cache.values())
    assert all(emg is None for emg, _ in acc_only._cache.values())
    with pytest.raises(ValueError, match="Unsupported input modality"):
        PairedWindowDataset(subset, processed, modality="unknown")


def test_unimodal_training_statistics_match_used_paired_channels(tmp_path):
    """Removing unused I/O must not alter fold-specific normalisation."""
    index, processed, _, _ = _cohort(tmp_path, people=3)
    paired = training_channel_stats(index, processed, max_windows=12)
    for name in ("emg", "acc"):
        subset = training_channel_stats(
            index, processed, max_windows=12, modality=name,
        )
        np.testing.assert_allclose(
            subset[f"{name}_mean"], paired[f"{name}_mean"], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            subset[f"{name}_std"], paired[f"{name}_std"], rtol=0, atol=0
        )
        other = "acc" if name == "emg" else "emg"
        assert np.all(subset[f"{other}_mean"] == 0)
        assert np.all(subset[f"{other}_std"] == 1)


def test_loader_prefetch_policy_and_modality_only_batches(tmp_path):
    """CPU tests use local workers; CUDA configuration enables prefetching."""
    import torch
    from ninapro_training import _loader

    index, processed, _, _ = _cohort(tmp_path, people=3)
    subset = index.iloc[:8]
    classes = np.array([0, 1, 2, 3])
    stats = training_channel_stats(subset, processed, max_windows=8, modality="emg")
    cpu_loader = _loader(
        subset, processed, classes, stats, batch_size=4,
        shuffle=False, model_name="D1_emg_cnn", device=torch.device("cpu"),
    )
    emg, acc, labels = next(iter(cpu_loader))
    assert emg.shape == (4, 12, 600)
    assert acc.shape == (4, 0, 0)
    assert labels.shape == (4,)
    assert cpu_loader.num_workers == 0
    assert cpu_loader.pin_memory is False

    cuda_loader = _loader(
        subset, processed, classes, stats, batch_size=4,
        shuffle=False, model_name="D1_emg_cnn", device=torch.device("cuda"),
    )
    assert cuda_loader.num_workers == 2
    assert cuda_loader.pin_memory is True
    assert cuda_loader.persistent_workers is True


def test_staged_acceleration_matches_streamed_windows_and_targets(tmp_path):
    """GPU-cache preparation preserves every original float32 window and label."""
    import torch
    from ninapro_training import _loader, _stage_acceleration_windows

    index, processed, _, _ = _cohort(tmp_path, people=3)
    rows = index.iloc[:23].reset_index(drop=True)
    classes = np.array([0, 1, 2, 3])
    stats = training_channel_stats(
        rows, processed, modality="acc", max_windows=12,
    )
    original = _loader(
        rows, processed, classes, stats, batch_size=5, shuffle=False,
        model_name="D1_acc_cnn", device=torch.device("cpu"),
    )
    cached = _stage_acceleration_windows(
        rows, processed, classes, stats, batch_size=5,
        device=torch.device("cpu"), shuffle=False,
    )
    assert len(cached) == len(original)
    for (emg_old, acc_old, labels_old), (emg_new, acc_new, labels_new) in zip(
        original, cached,
    ):
        assert emg_old.shape == emg_new.shape
        assert emg_new.numel() == 0
        torch.testing.assert_close(acc_new, acc_old, rtol=0, atol=0)
        torch.testing.assert_close(labels_new, labels_old, rtol=0, atol=0)

    # A second epoch reuses exactly the same prepared windows.
    repeated = list(cached)
    assert len(repeated) == len(cached)
    for (_, acc_first, labels_first), (_, acc_again, labels_again) in zip(
        cached, repeated,
    ):
        torch.testing.assert_close(acc_first, acc_again, rtol=0, atol=0)
        torch.testing.assert_close(labels_first, labels_again, rtol=0, atol=0)


def test_acceleration_cuda_cache_does_not_run_on_cpu(tmp_path):
    """Memory-dependent caching cannot accidentally affect a CPU-only run."""
    import torch
    from ninapro_training import _can_stage_acceleration

    index, _, _, _ = _cohort(tmp_path, people=3)
    use_cache, requested_gib = _can_stage_acceleration(
        index, None, torch.device("cpu"),
    )
    assert use_cache is False
    assert requested_gib == 0.0


def test_gpu_acceleration_batches_retain_window_count_and_epoch_order():
    """Shuffled GPU-resident batches cover every sample exactly once."""
    import torch
    from ninapro_training import _GPUAccelerationBatches

    windows = torch.arange(8 * 36 * 4, dtype=torch.float32).reshape(8, 36, 4)
    labels = torch.arange(8, dtype=torch.long)
    cached = _GPUAccelerationBatches(
        windows, labels, batch_size=3, shuffle=True,
    )
    assert len(cached) == 3
    for _ in range(2):
        emitted = []
        for unused, acc, target in cached:
            assert unused.numel() == 0
            assert acc.shape[1:] == (36, 4)
            torch.testing.assert_close(acc, windows.index_select(0, target))
            emitted.extend(target.tolist())
        assert sorted(emitted) == list(range(8))
