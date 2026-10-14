# First executed BoneNet regression baseline — exploratory only

**Execution date:** 14 October 2026  
**Actual public source:** authors' numeric `Data.mat` and `labels.csv` from the BoneNet repository.  
**GitHub Actions evidence:** https://github.com/RezaSaadatyar/human-movement-intelligence/actions/runs/38558753849  
**Status:** Completed on a temporary GitHub Actions Python runner. This is **not** the complete six-exercise Figshare dataset.

## Executed source and task
- Input: `X=(444,42,101)`, numeric time-normalised IMU signals.
- Target: `Y=(444,101,2)`, authors' named `e1` and `e3` model-derived strain arrays.
- Missingness: no numeric NaN or Inf in these tensors.
- Split: `GroupShuffleSplit`, random seed 42, 25% of subjects held out.
- Train: **27 distinct subjects, 324 records**.
- Test: **10 distinct subjects, 120 records**.
- Descriptors: 42 per-channel temporal means and 42 standard deviations (84 features).
- Ridge model: training-only `StandardScaler`, `Ridge(alpha=10.0)`; one multi-output fit for the flattened time trajectories.
- Training-mean model: training-only per-timepoint mean target.

## Actual test metrics

Both targets and all timepoints are flattened for pooled error evaluation. Values reflect dataset source scale; the dataset-specific strain unit/conversion has **not yet been independently verified**.

| Baseline | Pooled MAE | Pooled RMSE |
| --- | ---: | ---: |
| Training mean | **1325.5737** | **1851.0981** |
| Ridge descriptors | 1446.1677 | 2191.9425 |

Ridge performed worse than the training-mean comparator in this single unoptimised experimental split. These results must not be interpreted as evidence of robust generalisation or inferiority of linear methods in general.

## Major scientific limitations

1. The association between `Data.mat` rows and `labels.csv` subjects relies on the source authors' row order and remains independently unverified. The CSV has **222** repeated `SubjID+Task+TrialNum` combinations and lacks an intensity identifier.
2. This analysis uses the **walking/running derivative**, not the full 40-subject source archive.
3. Metrics pool `e1` and `e3` across all timepoints; next results should separate the two outputs, report units, waveform/peak errors and per-participant statistics.
4. No uncertainty intervals, repeated group folds, hyperparameter tuning or external validation have been conducted.
5. FE-generated strains are model-based estimates, not measured in vivo and not clinical diagnostic outcomes.

## Next development work

- Independently verify the source-to-label ordering through the authors' export logic or reconstruct subject-specific records from a validated source reader.
- Add per-target, per-participant evaluation and physical unit checks.
- Establish subject-group cross-validation within the training cohort and preserve an untouched final test group.
- Compare simple linear, tree-based and temporal deep learning models under the same verified grouping and target specification.
- Continue Python-first development. The original Figshare 40-file MCOS decoding issue remains a separate data-ingestion workstream.

**Code:** `scripts/bonenet_python_pilot.py`. No source raw data or model weights were committed.
