# Stage 9: full-development GroupKFold comparison

## Purpose

Assess the established Dummy, Logistic Regression and Random Forest baselines on **all 29 HARTH v2.0 development participants**, preserving the previously designated final Test IDs S023 and S032.

## Reproducible protocol

- Original source: pinned official NTNU HARTH v2.0 participant CSVs cached locally.
- Individual recordings: 50 Hz, 100-sample windows and 50-sample step, only label-pure segments without timestamp discontinuities.
- Task: 8 documented activity codes 1–8. Codes outside the predefined task are explicitly counted and excluded **after** windowing, without relabelling.
- Five existing deterministic features for each of six channels: 30 features per window.
- GroupKFold with five folds over participant IDs. Every participant contributes to validation in exactly one fold. The split is *not* class-stratified.
- Confirm training and validation contain all eight classes before training.
- Refit three model families from scratch per fold. Only Logistic Regression fits a StandardScaler, always on fold Train.
- Primary descriptive model comparison: held-out-fold Macro F1, Balanced Accuracy and per-participant fixed-eight-class Macro F1.
- Paired subject comparison: mean/median and signed per-person difference of Random Forest vs Logistic Regression, no p-value claims or treating overlapping windows as independent.
- No training, loading or evaluation on the reserved Test IDs in Notebook 09.

## Interpretation

This is a **development-only exploratory model-family comparison**: the models were already inspected on smaller overlapping development subsets, and the final Test individuals were inspected in early exploratory analyses. Thus the original Test is reserved from Stage 9 training and selection, but it is not a preregistered, completely unseen benchmark. Fold scores are not independent observations for confidence interval claims. Any future final test evaluation must be explicitly approved and transparently described.

## Run in Cursor

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

Open `notebooks/09_full_development_cross_validation.ipynb` and execute in order. All source CSVs should already be cached from Notebook 08. Rebuilding the feature cohort and fitting 15 classifiers will take longer than Notebook 08. Commit the executed Notebook and review its results before further modelling.

Shared code: `code/python/src/core/group_cross_validation.py` (cross-validation and paired comparison); `code/python/src/core/data_splitting.py` (fold class coverage).
