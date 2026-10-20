# Stage 11: Matched Classical ML vs PyTorch Temporal CNN

## Research objective

Compare the established Random Forest baseline and raw-IMU 1D-CNN **using identical source signal windows, original activity codes, independent participants and deterministic split indices**. The models intentionally use different representations: Random Forest receives 30 deterministic, handcrafted features from the same windows, whereas the CNN receives six raw acceleration channels over 100 time points. This is a *matched data and validation* comparison, not proof that differences are caused solely by model architecture.

## Method and reproducible inputs

- Source: HARTH v2.0 official authors' CSVs, locally cached under `data/raw/harth_v2/`.
- Cohort: same 29 development participants as Notebook 10. Original Test IDs S023/S032 excluded.
- Input: strict single-label 100-sample windows, 50-sample step, source-timestamp continuity. A maximum of 500 eligible windows per subject are selected deterministically in chronological order, without label-based sampling.
- Task: documented HARTH classes 1–8, with excluded cycling/unknown source codes removed only after windowing.
- Group split: seed 42, 70% Train / 20% Validation / 10% unused-development by subject; the same split as Notebook 10.
- Forest: five original features from each of six channels; 120 trees, minimum leaf size 3, balanced-subset class weights, seed 42. No fitted scaler is needed.
- CNN: two temporal convolutions, train-only channel scaling, Adam (1e-3), five epochs, batch size 128; last-epoch Validation predictions. Same configuration as Notebook 10.
- Main outcome: Validation Macro F1 using the eight-class denominator. Also report Balanced Accuracy, per-class F1, support, two confusion matrices, and paired per-subject Macro F1.
- No reserved Test loading, hyperparameter search, final Test inference, p-values or confidence intervals.

## Shared research code

- `code/python/src/core/ml_baselines.py`: new reusable `evaluate_predictions_by_subject()` calculates per-person fixed-task and observed-class F1.
- `code/python/src/core/group_cross_validation.py`: existing `compare_models_by_subject()` computes paired participant-level differences.
- `code/python/src/core/temporal_cnn.py`: existing CNN, channel scaling and PyTorch trainer.
- `code/python/src/core/feature_engineering.py`: existing five signal features.
- `tests/test_matched_evaluation.py`: offline reporting/alignment tests.
- `notebooks/11_matched_cnn_random_forest.ipynb`: executable, commented, research-facing analysis.

## Interpretation restrictions

The bounded 14,201-window cohort intentionally differs from Stage 9's 132,397-window full-development experiment. Therefore do not compare the Stage 9 Random Forest value to Stage 10's CNN as if their validation protocols were identical. Notebook 11 supplies a common basis, but still only one development split; neither model is established as statistically superior. The original Test individuals were inspected in early dataset exploration, so the Test should be described as *reserved from model development* rather than strictly never observed.

## Run in Cursor

```powershell
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_matched_evaluation.py" -v
```

Open `notebooks/11_matched_cnn_random_forest.ipynb`, execute every cell, inspect the class-wise F1 and participant charts, then commit the executed notebook. Real model results must be checked before further conclusions.
