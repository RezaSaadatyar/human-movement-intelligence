# HARTH Stage 6: development-only subject-level cross-validation

## Scientific purpose

Notebook 05 compared three baseline classifiers using only two Validation participants and selected Logistic Regression on that Validation subset. Such a result depends strongly on which subjects were held out. Notebook 06 performs exploratory 3-fold **GroupKFold** assessment across the six *development* participants, leaving the two original Test participants untouched.

## Fixed protocol

- Source: official HARTH v2.0 (pinned repository revision); eight selected participants.
- Keep the original eight-class task (codes 1–8); exclude cycling and undocumented codes **after** label-pure time-series windowing.
- Keep deterministic 100-sample windows, stride 50, five features for each of six channels.
- Reproduce the same original participant-group Train / Validation / Test assignment as Notebook 05 (seed 42).
- Concatenate the four original Train participants and two original Validation participants as **development data only**. The two original Test subjects are *never passed to GroupKFold or the ML training function*.
- GroupKFold with three held-out folds of two subjects each. Refit each model and any fitted scaler from scratch in each fold's Train.
- Compare Dummy, Logistic Regression and Random Forest; report fold-level Macro F1 and Balanced Accuracy, plus per-participant fixed-eight-class and observed-class Macro F1.
- Report exact participant membership and missing class coverage. Require all task classes in each training fold.

## Interpretation restrictions

- GroupKFold results support an **exploratory development audit**, not unbiased prospective or external validation.
- The model family was already compared on a portion of these development subjects in Notebook 05; this is not nested CV and results are not an independent generalisation estimate.
- Individual participants have different observed classes, so eight-class Macro F1 and observed-class Macro F1 answer different questions.
- No confidence intervals or statistically independent repeated trials are claimed.
- The final Test participants remain reserved. Do not tune using their data.
- Moving to full HARTH should use a declared multi-participant benchmarking and class-handling protocol.

## Run in Cursor

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

Then open `studies/wearable_activity_recognition/notebooks/06_subject_level_cross_validation.ipynb` and execute cells in order. Results are generated from the actual participant CSV files, not hard-coded.
