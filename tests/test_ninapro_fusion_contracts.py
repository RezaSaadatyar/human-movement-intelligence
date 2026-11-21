"""Offline contracts for source-aligned Notebook 04 fusion comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
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
from ninapro_models import UNIMODAL_MODELS, FUSION_MODELS
from ninapro_training import (
    _loader, _stage_paired_windows, _can_stage_paired,
    fit_neural, predict_neural,
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


def test_notebook04_cells_are_structured_and_preserve_execution():
    notebook = (
        REPO / "multimodal_movement_learning" / "code" / "notebooks"
        / "04_fusion_and_modality_ablations.ipynb"
    )
    content = json.loads(notebook.read_text(encoding="utf-8"))
    cells = content["cells"]
    assert cells and cells[0]["cell_type"] == "markdown"
    assert len({cell["id"] for cell in cells}) == len(cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    # Notebooks may be committed with original execution evidence. Validate
    # the notebook structure without requiring the recorded outputs to vanish.
    assert all(isinstance(cell["outputs"], list) for cell in code_cells)
    assert all(
        cell["execution_count"] is None
        or isinstance(cell["execution_count"], int)
        for cell in code_cells
    )
    assert sum("run_neural_fusion_arm(" in "".join(cell["source"])
               for cell in code_cells) == 4
    assert sum("run_early_fusion_arm(" in "".join(cell["source"])
               for cell in code_cells) == 2
    assert sum("run_fixed_late_arm(" in "".join(cell["source"])
               for cell in code_cells) == 3
    all_code = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "04_all_fusion_per_class_metrics.csv" in all_code
    assert "04_fusion_class_errors_normalised.png" in all_code
    assert "04_fusion_per_class_f1_heatmap.png" in all_code
    for cell in code_cells:
        source = "".join(cell["source"])
        if not source.lstrip().startswith(
            "# ======================== 00. Colab Repository Setup"
        ):
            compile(source, "notebook04-cell", "exec")


def test_notebook04_scientific_svg_matches_five_band_reference():
    """Fusion figure keeps editable scientific content in the common geometry."""
    svg_file = (
        REPO / "multimodal_movement_learning" / "images"
        / "04_fusion_and_modality_ablations.svg"
    )
    root = ET.parse(svg_file).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.get("width") == "1700"
    assert root.get("height") == "816"
    assert root.get("viewBox") == "0 0 1700 816"
    labelled = [
        "".join(element.itertext())
        for element in root.iter(f"{namespace}text")
    ]
    assert not any(text.startswith("BAND ") for text in labelled)
    assert any("Development" in text for text in labelled)
    assert any("Test" in text for text in labelled)
    assert any("F2 / F4" in text for text in labelled)
    assert any("F3" in text for text in labelled)
    assert any("F5" in text for text in labelled)
    panels = [
        element for element in root.iter(f"{namespace}g")
        if element.get("id", "").startswith("evidence-panel-")
    ]
    assert len(panels) == 4
    assert len({
        element.get("id") for element in root.iter()
        if element.get("id") is not None
    }) == sum(element.get("id") is not None for element in root.iter())


@pytest.mark.parametrize("model_name", FUSION_MODELS)
def test_neural_fusion_cpu_training_and_predictive_contract(
    tmp_path, model_name,
):
    """One synthetic epoch tests paired forward, backward and held-out OOF."""
    _, development, processed = _tiny_cohort(tmp_path)
    classes = np.array([0, 1])
    fitting = development.loc[
        development.validation_fold != 0
    ].reset_index(drop=True)
    validation = development.loc[
        development.validation_fold == 0
    ].reset_index(drop=True)
    device = torch.device("cpu")
    trained, stats, history, info = fit_neural(
        model_name, fitting, processed, classes, device=device,
        seed=42, epochs=1, patience=1, batch_size=8,
    )
    assert info["model_id"] == model_name
    assert len(history) == 1
    assert not info["paired_gpu_cached"]
    predicted = predict_neural(
        model_name, trained, validation, processed,
        classes, stats, device=device, batch_size=8,
    )
    assert predicted.shape == (len(validation), len(classes))
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(
        predicted.sum(axis=1), np.ones(len(validation)),
        atol=1e-5,
    )
    if model_name in ("F3_dual_cnn", "F5_cross_attention"):
        ablated = predict_neural(
            model_name, trained, validation, processed,
            classes, stats, device=device, batch_size=8,
            perturbation="acc_missing_mean",
        )
        assert ablated.shape == predicted.shape
        assert np.isfinite(ablated).all()


def test_notebook05_frozen_features_and_five_band_figure():
    """Verify Test-stage inputs, recorded execution and editable diagram."""
    study = REPO / "multimodal_movement_learning"
    notebook = json.loads((
        study / "code" / "notebooks"
        / "05_frozen_person_independent_test.ipynb"
    ).read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells
    # Notebook 05 now retains its recorded Test execution and plots.
    # A notebook may be published either before or after execution.
    assert all(isinstance(c.get("outputs"), list) for c in code_cells)
    assert all(c.get("execution_count") is None
               or isinstance(c["execution_count"], int) for c in code_cells)
    assert not any(o.get("output_type") == "error"
                   for c in code_cells for o in c["outputs"])
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for required in (
        "02_emg_features.npy",
        "02_acc_features.npy",
        "check_artifact_alignment",
        "05_per_class_support.csv",
        "05_confusion_matrix_normalised.png",
        "precision_recall_fscore_support",
    ):
        assert required in source

    picture = study / "images" / "05_frozen_person_independent_test.svg"
    svg = ET.parse(picture).getroot()
    assert svg.attrib["width"] == "1700"
    assert svg.attrib["height"] == "816"
    assert svg.attrib["viewBox"] == "0 0 1700 816"
    ids = [node.attrib["id"] for node in svg.iter()
           if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    for band in range(5):
        assert any(item.startswith(f"band-{band}-") for item in ids)
    labels = " ".join(
        (node.text or "") for node in svg.iter()
        if node.tag.endswith("text")
    )
    assert "B1_rf" in labels and "F1_rf" in labels
    assert "Test" in labels and "Development" in labels

def test_frozen_test_predictions_reconstruct_saved_metrics():
    """Verify fixed selected recipes, matched source keys and saved Test F1."""
    from sklearn.metrics import f1_score

    study = REPO / "multimodal_movement_learning"
    saved = study / "output"
    original = pd.read_csv(
        saved / "02_window_index.csv",
        usecols=["window_key", "subject_id", "label"],
    )
    roles = pd.read_csv(saved / "02_participant_split_assignments.csv")
    test_people = set(roles.loc[roles.role == "Test", "subject_id"])
    test_index = original.loc[
        original.subject_id.isin(test_people)
    ].reset_index(drop=True)
    classes = np.sort(
        pd.read_csv(saved / "01_label_dictionary.csv").label.to_numpy(int)
    )
    selection = pd.read_csv(saved / "04_model_selection.csv")
    chosen = dict(zip(selection.selection_role, selection.model_id))
    predictions = pd.read_csv(saved / "05_test_predictions.csv")
    metrics = pd.read_csv(saved / "05_test_metrics.csv")
    summary = pd.read_csv(saved / "05_test_summary.csv").iloc[0]

    assert test_people
    assert test_index.window_key.is_unique
    assert set(predictions.comparison_role) == set(chosen)
    assert not predictions.duplicated(["comparison_role", "window_key"]).any()
    assert len(predictions) == len(test_index) * len(chosen)

    computed = {}
    for role, model_id in chosen.items():
        frame = predictions.loc[predictions.comparison_role == role]
        assert set(frame.model_id) == {model_id}
        assert set(frame.window_key) == set(test_index.window_key)
        aligned = frame.set_index("window_key").loc[test_index.window_key]
        assert np.array_equal(
            aligned.subject_id.to_numpy(), test_index.subject_id.to_numpy()
        )
        assert np.array_equal(
            aligned.label.to_numpy(int), test_index.label.to_numpy(int)
        )
        scores = aligned[[f"p_class_{int(x)}" for x in classes]].to_numpy(float)
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-5)
        predicted = classes[scores.argmax(axis=1)]
        assert np.array_equal(
            aligned.predicted_label.to_numpy(int), predicted
        )
        actual_rows = metrics.loc[metrics.comparison_role == role]
        assert set(actual_rows.subject_id) == test_people
        scores_by_person = []
        for person in sorted(test_people):
            mask = test_index.subject_id.to_numpy() == person
            score = f1_score(
                test_index.label.to_numpy(int)[mask], predicted[mask],
                labels=classes, average="macro", zero_division=0,
            )
            stored = actual_rows.loc[
                actual_rows.subject_id == person, "macro_f1"
            ]
            assert len(stored) == 1
            np.testing.assert_allclose(stored.iloc[0], score, atol=1e-12)
            scores_by_person.append(score)
        computed[role] = float(np.mean(scores_by_person))

    np.testing.assert_allclose(
        summary["unimodal_mean_subject_macro_f1"],
        computed["unimodal_comparator"], atol=1e-12,
    )
    np.testing.assert_allclose(
        summary["fusion_mean_subject_macro_f1"],
        computed["fusion_comparator"], atol=1e-12,
    )
    np.testing.assert_allclose(
        summary["mean_difference"],
        computed["fusion_comparator"] - computed["unimodal_comparator"],
        atol=1e-12,
    )
