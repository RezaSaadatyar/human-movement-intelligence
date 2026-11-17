"""Focused synthetic contracts for NinaPro stage-03 orchestration."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / 'multimodal_movement_learning' / 'code' / 'src'
sys.path.insert(0, str(SRC))

from ninapro_experiments import (
    CLASSICAL_MODELS, _save_model_progress, consolidate_unimodal_outputs,
    load_completed_model, run_classical_arm, run_unimodal_arm,
)
from ninapro_models import UNIMODAL_MODELS
from ninapro_evaluation import build_classical, per_class_oof_report, participant_metrics


def synthetic_metadata():
    entries = []
    for person in range(6):
        for k in range(4):
            entries.append({
                'window_row': len(entries), 'window_key': f'S{person:02d}_{k}',
                'subject_id': f'S{person:02d}', 'label': k % 2,
                'validation_fold': person % 2, 'role': 'Development',
                'start_sample': 8 * k, 'end_sample': 8 * k + 32,
            })
    return pd.DataFrame(entries)


def arrays_and_table():
    rng = np.random.default_rng(5)
    table = synthetic_metadata()
    emg = rng.normal(size=(len(table), 48)).astype(np.float32)
    acc = rng.normal(size=(len(table), 96)).astype(np.float32)
    emg[:, 0] += table.label.to_numpy() * .3
    acc[:, 0] += table.label.to_numpy() * .3
    return table, emg, acc


def test_folded_classical_models_converge_and_preserve_window_order(tmp_path):
    index, emg, acc = arrays_and_table()
    for name in ('B0_prior', 'B1_logreg', 'B2_logreg', 'B1_rf'):
        scores = run_classical_arm(
            name, index, emg, acc, np.array([0, 1]),
            output_dir=tmp_path, folds=[0, 1], seed=42,
        )
        assert len(scores) == 2
        assert scores.participants.tolist() == [3, 3]
        prob, metrics, preds, history = load_completed_model(
            name, index, np.array([0, 1]), tmp_path, [0, 1],
        )
        assert prob.shape == (24, 2)
        assert not np.isnan(prob).any()
        assert len(metrics) == 6
        assert list(preds.window_key) == list(index.window_key)
        assert history is None
        if name.endswith('_logreg'):
            assert (scores.solver_iterations > 0).all()
            assert (scores.solver_iterations < 2000).all()
    assert build_classical('F1_logreg').named_steps['logisticregression'].max_iter == 2000


def test_partial_model_probabilities_cannot_enter_consolidation(tmp_path):
    index, emg, acc = arrays_and_table()
    run_classical_arm('B0_prior', index, emg, acc, np.array([0, 1]),
                      output_dir=tmp_path, folds=[0, 1])
    f = tmp_path / '03_B0_prior_oof_probabilities.npz'
    with np.load(f, allow_pickle=False) as saved:
        record = dict(saved)
    record['scores'][0, :] = np.nan
    np.savez_compressed(f, **record)
    with pytest.raises(ValueError, match='OOF probabilities are missing'):
        load_completed_model('B0_prior', index, np.array([0, 1]),
                             tmp_path, [0, 1])


def test_aggregate_preserves_notebook04_contract(tmp_path):
    index, emg, acc = arrays_and_table()
    labels = np.array([0, 1])
    for name in (*CLASSICAL_MODELS, *UNIMODAL_MODELS):
        proba = np.where(index.label.to_numpy()[:, None] == labels,
                         .8, .2).astype(np.float32)
        scores, predictions, histories = [], [], []
        for fold in [0, 1]:
            rows = index.loc[index.validation_fold == fold].reset_index(drop=True)
            from ninapro_evaluation import participant_metrics
            part = proba[rows.window_row.to_numpy(int)]
            metric = participant_metrics(rows, part, labels, model_id=name,
                                         fold=fold, seed=42)
            scores.append(metric)
            predictions.append(pd.DataFrame({
                'window_key': rows.window_key, 'subject_id': rows.subject_id,
                'validation_fold': fold, 'model_id': name,
                'true_label': rows.label,
                'predicted_label': rows.label,
            }))
            histories.append(pd.DataFrame({
                'epoch': [1], 'train_loss': [.5],
                'inner_validation_macro_f1': [.5],
                'model_id': [name], 'seed': [42],
                'validation_fold': [fold],
            }))
        _save_model_progress(
            name, index, labels, tmp_path, scores, predictions, proba,
            seed=42,
            histories=histories if name in UNIMODAL_MODELS else None,
            settings={'epochs':10,'patience':3,'batch_size':96}
              if name in UNIMODAL_MODELS else None,
        )
    probs, cm, nm, nh, chosen, summary = consolidate_unimodal_outputs(
        index, index, labels, output_dir=tmp_path, folds=[0, 1],
        seed=42, neural_settings={'epochs':10,'patience':3,'batch_size':96},
    )
    assert set(probs) == set((*CLASSICAL_MODELS, *UNIMODAL_MODELS))
    assert len(cm) == 5 * 6 and len(nm) == 6 * 6
    assert len(summary) == 11 and chosen in probs
    assert len(nh) == 6 * 2
    assert {'03_oof_probabilities.npz', '03_all_unimodal_metrics.csv',
            '03_neural_oof_predictions.csv', '03_dev_neural_metrics.csv'} <= {
                x.name for x in tmp_path.iterdir()
            }
    with np.load(tmp_path / '03_oof_probabilities.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(probs)
        assert saved['B1_logreg'].shape == (24, 2)


def test_neural_fits_with_synthetic_source_windows(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    index, _, _ = arrays_and_table()
    rng = np.random.default_rng(22)
    for person in index.subject_id.unique():
        for channel_type, cols in [('emg',12),('acc',36)]:
            values = rng.normal(size=(64, cols)).astype(np.float32)
            np.save(data / f'02_{person}_E1_{channel_type}.npy', values)
    rows = index[['window_row','window_key','subject_id','label','validation_fold',
                  'start_sample','end_sample']].copy()
    import torch
    summary = run_unimodal_arm(
        'D1_emg_cnn', rows, data, np.array([0,1]), output_dir=tmp_path,
        folds=[0,1], device=torch.device('cpu'), seed=42,
        epochs=1, patience=1, batch_size=8,
    )
    assert len(summary) == 2
    assert (summary.participants == 3).all()
    assert (tmp_path / '03_D1_emg_cnn_learning_curve.png').is_file()
    _, metrics, predictions, history = load_completed_model(
        'D1_emg_cnn', rows, np.array([0,1]), tmp_path, [0,1], seed=42,
        neural_settings={'epochs':1,'patience':1,'batch_size':8},
    )
    assert len(metrics) == 6 and len(predictions) == len(rows)
    assert set(history.validation_fold) == {0, 1}


def test_per_class_oof_report_preserves_participant_macro_f1_definition():
    """Absent classes count as zero for each person's fixed-class macro F1."""
    labels = np.array([0, 1, 2, 3])
    table = pd.DataFrame({
        "subject_id": ["S01", "S02", "S03", "S04"],
        "window_key": ["one", "two", "three", "four"],
        "label": [0, 1, 2, 0],
    })
    probabilities = np.eye(4, dtype=np.float32)[[0, 1, 2, 0]]
    report = per_class_oof_report(
        table, probabilities, labels, model_id="B1_rf",
    )
    metrics = participant_metrics(
        table, probabilities, labels, model_id="B1_rf",
    )
    assert len(report) == len(labels)
    assert report.class_label.tolist() == labels.tolist()
    assert report.support_windows.tolist() == [2, 1, 1, 0]
    assert report.participants_with_support.tolist() == [2, 1, 1, 0]
    assert (report.loc[report.class_label == 3, [
        "pooled_precision", "pooled_recall", "pooled_f1",
        "mean_participant_precision", "mean_participant_recall",
        "mean_participant_f1",
    ]] == 0).all().all()
    assert np.isclose(
        report.mean_participant_f1.mean(), metrics.macro_f1.mean(),
    )
    assert np.isclose(metrics.macro_f1.mean(), 0.25)
    assert np.isclose(report.pooled_f1.mean(), 0.75)


def test_per_class_oof_report_catches_alignment_and_invalid_inputs():
    labels = np.array([0, 1])
    table = pd.DataFrame({
        "subject_id": ["S01", "S02"],
        "window_key": ["first", "second"],
        "label": [0, 1],
    })
    probabilities = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="probability rows differ"):
        per_class_oof_report(
            table, probabilities[:1], labels, model_id="B1_rf",
        )
    with pytest.raises(ValueError, match="missing values"):
        per_class_oof_report(
            table, np.full((2, 2), np.nan), labels,
            model_id="B1_rf",
        )
    duplicated = table.copy()
    duplicated.loc[1, "window_key"] = duplicated.loc[0, "window_key"]
    with pytest.raises(ValueError, match="duplicate window keys"):
        per_class_oof_report(
            duplicated, probabilities, labels, model_id="B1_rf",
        )
