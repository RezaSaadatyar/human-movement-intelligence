# Stage 14: chronological sampling sensitivity audit

## Motivation
Stage 13 evaluated 29 development participants on a bounded 14,201-window cohort. Its held-out fold means for Macro F1 were Random Forest 0.8076, Weighted CNN 0.7834 and Standard CNN 0.6301. Despite a strong improvement of Weighted over Standard CNN, Shuffling remained weak (pooled out-of-fold F1 0.1345). The per-participant cap of 500 windows may discard scarce minority-class windows, but its effects have not yet been quantified.

## Exact data and sampling protocol
- Official pinned HARTH v2.0 source CSVs, 29 development subjects, with original Test IDs S023/S032 strictly excluded.
- Preprocess **per recording** using the prior 100-point, 50-point step, timestamp-gap and label-purity policies.
- Exclude undocumented codes and all labels outside the documented eight-class task **after** window formation.
- Preserve eligible source-window order inside each individual recording.
- Evaluate each deterministic cap of **250, 500, 1000** windows/person against the **full 132,397-window eligible development cohort** (full cohort count verified previously, but new notebook must reproduce it).
- Evenly spaced selection matches Stages 10–13: `np.linspace(0, n_eligible-1, num=min(n_eligible, cap), dtype=np.int64)`. No label-balanced sampling.
- Produce per-class window counts, retained percentage, number of contributing independent people, and per-person support for Shuffling and both stair classes.
- The reusable function `audit_temporal_sampling` in `code/python/src/core/sampling_audit.py` is dataset-independent and does not fit any model.
- Offline regression tests in `tests/test_sampling_audit.py` verify cap positions, original row identity, no duplicate source rows and strict reserved-Test protection.

## Scientific limitations
- Chronologically even selection is deterministic but **not activity-stratified**, and it may disproportionately discard very short activities.
- Retention counts provide no causal estimate of downstream model improvement.
- This is only an inspection of development-data design; no training, evaluation or hidden Test access is performed.
- Overlapping windows remain highly correlated. The number of independent people is more important than the number of windows when discussing generalisation.
- The original Test participants were inspected in early stages, and should be described as *reserved from subsequent model development*, not completely unseen in a strict preregistered sense.

## Run in Cursor
```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_sampling_audit.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/14_sampling_sensitivity_audit.ipynb`, run every cell, inspect source counts and charts, and push the executed Notebook. The code reuses locally cached HARTH data. Only after viewing real outputs should a subsequent Stage 15 compare model training on expanded per-person window caps with strict identical group splits.
