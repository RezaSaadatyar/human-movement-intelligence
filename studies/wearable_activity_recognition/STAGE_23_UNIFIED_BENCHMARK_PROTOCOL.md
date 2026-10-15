# Stage 23 | Seven-Model Matched Grouped HARTH Benchmark

## Aim and source boundaries

Scientifically compare **Random Forest, Standard CNN, Weighted CNN, TCN, Multi-Scale CNN, 1D-ResNet and Time-Series Transformer** using an identical original-source-row and participant-held-out protocol. The real HARTH v2.0 source files are pinned by `HARTH_REVISION`, and only 29 Development participants are loaded. The original S023/S032 Test people remain excluded; they were inspected in early historical experiments, so any future final-Test claim must explicitly disclose that context.

## Exact preprocessing and evaluation

- Per-person time-safe, label-pure windows of 100 samples at 50 Hz, 50-sample step; original numeric activity codes 1–8 only. Source annotations outside the task are excluded **after** windowing.
- Chronological, non-label-stratified selection of up to 500 eligible windows per person, expected total **14,201** Development windows. This is a bounded experiment, not a full-dataset benchmark.
- **Three GroupKFold splits** by person. All seven classifiers evaluate exactly the same held-out source indices, one prediction per Development source row. Original Test IDs never enter this process.
- RF uses the same 30 handcrafted features computed from identical windows, 120 trees, minimum leaf 3, `balanced_subsample` weights.
- Six neural models use original (N, 6, 100) windows, per-channel standardisation fitted on **Fold Train only**, freshly initialised PyTorch networks for each fold, Adam 0.001, batch size 128, **400 optimiser updates**, seed 42. Standard CNN is unweighted; Weighted CNN and four advanced models use inverse-square-root class weights fitted from **Fold Train only**.
- The final-step network is scored without Validation-selected checkpoints. Equal updates do not mean equal compute, source-row exposures, fitted-parameter counts, optimisation convergence or architectural effects.
- Primary metric: fixed-eight-class Macro F1; also Balanced Accuracy, raw Accuracy, class support/precision/recall/F1, participant-level fixed-task and observed-class Macro F1, confusion matrices, fold means/SD, train-time and inference-time measures. These are exploratory *development* estimates, not significance tests.

## Resume-safe execution

`code/python/src/core/temporal_benchmark.py` implements explicit model and group-fold controls. It writes each completed model/fold as a compact JSON record under **ignored** `data/processed/harth_stage23_matched_benchmark/`, including true/predicted source labels, Validation source row IDs, group IDs, model hyperparameters, timing and SHA256 of dataset and relevant code. An atomic rename occurs only upon success. Re-running the Notebook resumes completed jobs after verifying cache provenance; configuration or code changes invalidate the saved key and force retraining. The cache contains predictions, not model weights. **Do not commit ignored raw datasets, cache files or model weights in Stage 23.**

This is an intentionally substantial workload: seven model families times three folds, comprising **18 neural fits (400 updates each)** and **three RF fits**. The Transformer may be markedly slower on CPU. Prefer the functioning CUDA environment previously observed in Stage 22; exact runtime depends on hardware. Do not present partial model jobs as a finished seven-model benchmark.

## Files and quality controls

- `notebooks/23_unified_matched_model_benchmark.ipynb`: 12 substantial code sections plus full explanatory Bold Markdown, numbered code comments, output tables, cross-fold and class figures, original source confusion plots, person-level heatmap, resource reporting and strict research Quality Gate.
- `code/python/src/core/temporal_benchmark.py`: reusable model/fold runner, verified resumable job cache and source-grounded result aggregation.
- `tests/test_temporal_benchmark.py`: offline synthetic group, source fingerprint, cache and incomplete-job checks. The standard lightweight CI environment does not install optional PyTorch, so neural tests are skipped there and **must actually run** in the user's local PyTorch environment.

## Execution in Cursor

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_temporal_benchmark.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/23_unified_matched_model_benchmark.ipynb`, restart the kernel and execute cells in order. The long **Section 05** will train and cache the 21 runs; if execution is interrupted, rerun that cell to restore prior matching finished jobs. Then complete Sections 06–12, inspect all saved tables and charts and ensure the final Quality Gate says `PASSED`. Save and commit/push the executed **Notebook**. If a model fails, retain the actual error message for diagnosis; never replace failures with invented outcomes.

## Scientific interpretation and next step

Model-family differences combine architecture, representation, loss and optimiser/compute properties. Shuffling and stair support from Stage 21 should shape the interpretation. Stage 24 will investigate cross-seed stability and targeted imbalance interventions on a justified shortlist. Selection of a final model and any qualified original Test evaluation are deferred until Stages 25–26.
