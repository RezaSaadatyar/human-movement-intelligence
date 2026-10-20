# Stage 13: matched subject-level GroupKFold comparison

## Research question

Stage 12 indicated a rise in development Validation Macro F1 from 0.6659 (unweighted CNN) to 0.7968 (inverse-square-root class-weighted CNN). This notebook checks whether that descriptive improvement persists across different held-out participants. It compares three established models: Random Forest, Standard CNN and Weighted CNN.

## Methods and scientific controls

- Dataset: pinned official HARTH v2.0 CSVs for all 29 development subjects; original reserved Test IDs `S023` and `S032` are excluded before loading.
- Windowing: 50 Hz, 100-point windows, 50-point step, complete label purity and timestamp-gap checks, using the existing adapter.
- Eight predefined documented classes (1–8). Cycling and undocumented source codes excluded only **after** windowing.
- Bounded pilot sample: at most 500 windows per participant selected evenly from eligible windows without using the class labels to choose samples. Resulting activity proportions are not guaranteed representative of the original recording.
- GroupKFold: 3 participant-disjoint folds. Each development participant and every eligible sampled window appear in validation exactly once per model.
- Three estimators refitted from scratch per fold on identical Train/Validation participant IDs and source-window rows:
  - Random Forest: 120 trees, min leaf 3, balanced-subset class weights; original five features per each of six sensors from precisely those source windows.
  - Standard CNN: original two-convolution TemporalCNN with standard Cross-Entropy.
  - Weighted CNN: identical network, random seed and optimiser, with `1/sqrt(class_count)` loss weights computed using **fold Train only**.
- Both CNNs: Adam at 1e-3, five epochs, batch size 128, training-only channel normalisation and fixed random seed 42.

## Evaluation and interpretation

Report validation-fold Macro F1 and Balanced Accuracy, per-person fixed-eight-class Macro F1, and per-class precision/recall/F1 pooled from out-of-fold development predictions. Paired differences weight participants equally instead of interpreting overlapping signal windows as independent research subjects.

This is exploratory **development-only** model-family and loss-weighting evaluation. The models and class-weighting intervention were already examined on a portion of these participants in earlier notebooks. GroupKFold folds are not independent statistical trials, and no confidence intervals, p-values, unbiased nested selection or final Test metrics are claimed. Some participants do not have all eight classes, so fixed-task and observed-class F1 must not be conflated.

## Running in Cursor

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_temporal_comparison.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/13_matched_grouped_model_comparison.ipynb` and execute in order. The bounded data should already exist locally from the previous stages. The notebook runs **six CNN training jobs** (two architectures/loss settings over three folds), so CPU execution may be significantly slower than Stage 12. New real-data results are not claimed until the user runs and pushes the executed notebook.
