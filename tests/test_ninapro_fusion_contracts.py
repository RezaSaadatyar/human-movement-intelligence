"""Offline contracts for source-aligned Notebook 04 fusion comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "multimodal_movement_learning" / "code" / "src"
sys.path.insert(0, str(SRC))

from ninapro_dataset import training_channel_stats
from ninapro_fusion_experiments import (
    load_unimodal_oof, load_completed_fusion_arm,
    run_early_fusion_arm, run_fixed_late_arm,
)
from ninapro_models import UNIMODAL_MODELS
from ninapro_training import (
    _loader, _stage_paired_windows, _can_stage_paired,
)


def _tiny_cohort(tmp_path):
    entries = []
    rng = np.random.default_rng(11)
    processed = tmp_path / "data" / "processed" / "ninapro_db2"
    processed.mkdir(parents=True, exist_ok=True)
    for person in range(8):
        subject_id = f"S{person + 1:02d}"
        role = "Development" if person < 6 else "Test"
        for kind, channels in [("emg", 12), ("acc", 36)]:
            values = rng.normal(size=(64, channels)).astype(np.float32)
            np.save(processed / f"02_{subject_id}_E1_{kind}.npy", values)
        for j in range(4):
            entries.append({
                "window_row": len(entries),
                "window_key": f"{subject_id}_W{j}",
                "subject_id": subject_id,
                "role": role,
                "label": j % 2,
                "validation_fold": person % 2 if person < 6 else np.nan,
                "start_sample": j * 8,
                "end_sample": j * 8 + 32,
            })
    index = pd.DataFrame(entries)
    dev = index.loc[index.role == "Development"].copy()
    return index, dev, processed


def _write_notebook03_evidence(index, dev, output_dir, classes):
    expected_models = [
        "B0_prior", "B1_logreg", "B1_rf", "B2_logreg", "B2_rf",
        *UNIMODAL_MODELS,
    ]
    full = {}
    for model in expected_models:
        p = np.full((len(index), len(classes)), np.nan, dtype=np.float32)
        targets = dev.label.to_numpy(int)
        p[dev.window_row.to_numpy(int)] = np.where(
            targets[:, None] == classes, .8, .2,
        ).astype(np.float32)
        full[model] = p
        source = dev[["window_key", "subject_id", "validation_fold", "label"]].copy()
        source["model_id"] = model
        source["true_label"] = source.pop("label").to_numpy(int)
        source["predicted_label"] = targets
        source.to_csv(output_dir / f"03_{model}_oof_predictions.csv",
                      index=False)
    np.savez_compressed(output_dir / "03_oof_probabilities.npz", **full)
    return full


def test_notebook03_soft_probabilities_are_matched_to_original_keys(tmp_path):
    index, dev, _ = _tiny_cohort(tmp_path)
    labels = np.array([0, 1])
    expected = _write_notebook03_evidence(index, dev, tmp_path, labels)
    loaded = load_unimodal_oof(index, dev, labels, tmp_path)
    assert set(loaded) == set(expected)
    for name in expected:
        np.testing.assert_allclose(loaded[name], expected[name], equal_nan=True)
    path = tmp_path / "03_B1_rf_oof_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "predicted_label"] = 1 - int(frame.loc[0, "predicted_label"])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="predictions disagree"):
        load_unimodal_oof(index, dev, labels, tmp_path)


def test_classical_early_and_fixed_late_fusion_share_folds(tmp_path):
    index, dev, _ = _tiny_cohort(tmp_path)
    labels = np.array([0, 1])
    previous = _write_notebook03_evidence(index, dev, tmp_path, labels)
    rng = np.random.default_rng(42)
    emg = rng.normal(size=(len(index), 48)).astype(np.float32)
    acc = rng.normal(size=(len(index), 96)).astype(np.float32)
    emg[:, 0] += index.label.to_numpy() * .6
    acc[:, 0] += index.label.to_numpy() * .6
    folds = [0, 1]
    early = run_early_fusion_arm(
        "F1_logreg", dev, emg, acc, labels, output_dir=tmp_path,
        folds=folds, seed=42,
    )
    late = run_fixed_late_arm(
        "F2_logreg", index, dev, previous, labels,
        output_dir=tmp_path, folds=folds,
        emg_name="B1_logreg", acc_name="B2_logreg", seed=42,
    )
    assert early.participants.tolist() == [3, 3]
    assert late.participants.tolist() == [3, 3]
    for name in ("F1_logreg", "F2_logreg"):
        p, metrics, hist, robust = load_completed_fusion_arm(
            name, dev, labels, tmp_path, folds,
        )
        assert p.shape == (len(dev), 2)
        assert len(metrics) == 6
        assert hist is None and robust is None
    with np.load(
        tmp_path / "04_F2_logreg_oof_probabilities.npz",
        allow_pickle=False,
    ) as saved:
        probabilities = saved["scores"].copy()
        probabilities[0] = np.nan
        record = {key: saved[key] for key in saved.files if key != "scores"}
    record["scores"] = probabilities
    np.savez_compressed(
        tmp_path / "04_F2_logreg_oof_probabilities.npz", **record,
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_completed_fusion_arm(
            "F2_logreg", dev, labels, tmp_path, folds,
        )


def test_paired_window_cache_preserves_float32_samples_and_labels(tmp_path):
    _, dev, processed = _tiny_cohort(tmp_path)
    classes = np.array([0, 1])
    subset = dev.iloc[:13].copy()
    stats = training_channel_stats(
        subset, processed, modality="both", max_windows=12,
    )
    cpu = torch.device("cpu")
    streamed = _loader(
        subset, processed, classes, stats,
        batch_size=4, shuffle=False, model_name="F3_dual_cnn",
        device=cpu,
    )
    prepared = _stage_paired_windows(
        subset, processed, classes, stats,
        batch_size=4, device=cpu, shuffle=False,
        model_name="F3_dual_cnn",
    )
    assert len(prepared) == len(streamed)
    for old, new in zip(streamed, prepared):
        for original, cached in zip(old, new):
            torch.testing.assert_close(original, cached, atol=0, rtol=0)
    for old, new in zip(prepared, prepared):
        for original, cached in zip(old, new):
            torch.testing.assert_close(original, cached, atol=0, rtol=0)
    shuffled = prepared.__class__(
        prepared.emg, prepared.acceleration, prepared.targets,
        batch_size=4, shuffle=True,
    )
    for _ in range(2):
        emitted = [int(y) for _, _, targets in shuffled
                   for y in targets.tolist()]
        assert len(emitted) == len(subset)
        assert sorted(emitted) == sorted(prepared.targets.tolist())
    allowed, size_gib = _can_stage_paired(subset, None, cpu)
    assert not allowed and size_gib == 0


def test_notebook04_cells_are_structured_and_unexecuted():
    notebook = (
        REPO / "multimodal_movement_learning" / "code" / "notebooks"
        / "04_fusion_and_modality_ablations.ipynb"
    )
    content = json.loads(notebook.read_text(encoding="utf-8"))
    cells = content["cells"]
    assert len(cells) == 37
    assert len({cell["id"] for cell in cells}) == len(cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and not cell["outputs"]
               for cell in code_cells)
    assert sum("run_neural_fusion_arm(" in "".join(cell["source"])
               for cell in code_cells) == 4
    assert sum("run_early_fusion_arm(" in "".join(cell["source"])
               for cell in code_cells) == 2
    assert sum("run_fixed_late_arm(" in "".join(cell["source"])
               for cell in code_cells) == 3
    for cell in code_cells:
        source = "".join(cell["source"])
        if not source.lstrip().startswith(
            "# ======================== 00. Colab Repository Setup"
        ):
            compile(source, "notebook04-cell", "exec")
