# Stage 21 | HARTH Minority-Class and Source Data Quality Analysis

## Research objective

Inspect the actual official, pinned HARTH v2.0 development recordings for the sources of minority-class evidence and data-quality limitations before building four new temporal architectures. The aim is **audit and hypothesis formation**, not model fitting, causal attribution, clinician-facing validation or prediction improvement claims.

## Scope, implementation and provenance

- Read the **29 established development subjects** separately using the pinned official HARTH data loader. The original S023/S032 model-development Test subjects are excluded; they were inspected in earlier exploratory stages, so are not described as historically strictly unseen.
- Preserve all original source activity codes and timestamps, including unknown codes **10, 11 and 17** without assigning unverified activity names.
- Use six original acceleration channels with **100 samples / 2-second windows, 50-sample step, nominal 50 Hz, perfect annotation purity and timestamp continuity**. Construct windows *before* excluding labels outside the documented 1–8 task.
- Attribute candidate-window rejection to four mutually exclusive categories: retained; annotation-boundary only; timestamp problem only; both. A timing discontinuity is a diagnostic condition, not proven sensor corruption.
- Calculate per-subject and per-class eligible windows, class proportions, independent-person support and participant x class heatmaps.
- Describe consecutive source annotation **bouts** and the fraction shorter than two seconds. Treat bouts as annotation segments, not verified behavioural events; nominal duration uses sample count / 50 Hz.
- Repeat **label-blind chronological caps of 250, 500 and 1000** against full development data using exactly the previous sampling policy. The independently spaced cap 1000 here differs from later *nested* cap 1000.
- Report original six-channel raw signal ranges, SD, p99 of absolute values and adjacent repeated samples. Do not invent sensor physical units or infer clipping and malfunction from unverified thresholds.
- Plot genuine (unmodified) source windows for Walking, Shuffling and both stair classes with source participant/start index provenance.
- Contextualise previously saved Shuffling errors carefully, without claiming that class imbalance, activity transitions or sensor properties **caused** a model outcome.

## Source files

- `notebooks/21_minority_class_and_data_quality.ipynb`: **11 complete analysis sections** with numbered, explanatory code comments and Bold-only Markdown section titles. Tables, two-panel visualisations, actual examples and final Quality Gate retained.
- `src/harth_quality.py`: reusable source-specific audit, timestamp/window diagnostics, annotation-bout descriptions and channel summaries, with complete Args/Returns documentation.
- `../../tests/test_harth_quality.py`: offline tests covering mutually exclusive window reasons, timestamp-induced bout boundaries, raw channel descriptive reports and invalid inputs.
- Existing `code/python/src/core/signal_preprocessing.py` and `code/python/src/core/sampling_audit.py` are reused, not copied or modified.

## Notable limitations

- The audit examines source annotation and measured signals; it does not validate the truth of those labels against human experts.
- Consecutive overlapping windows are dependent observations, whereas the participant counts represent independent people.
- Data inspected here are development source recordings, not an external study or independent generalisation evaluation.
- Missing or rarely occurring classes may undermine grouped-fold class support without implying a definite modelling mechanism.
- The lack of verified definitions for unknown label codes must remain explicit.
- A passed quality gate proves reconciliation of the selected source data and method checks, **not** source measurement validity or model performance.
- The original S023/S032 Test participants were inspected earlier; their absence in this notebook does not erase that history.

## Run in Cursor

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_harth_quality.py" -v
```

Then open `studies/wearable_activity_recognition/notebooks/21_minority_class_and_data_quality.ipynb`, Restart Kernel and Run All Cells. Data will be loaded from the local ignored cache where available; the pinned official source will be fetched only if any cached files are missing. This stage reprocesses 29 original CSVs and can take several minutes but **does not train any neural or classical classifier**. Inspect and Push the **real saved notebook outputs** before beginning Stage 22.

## Next approved stage

Stage 22 will implement and smoke-test four distinct, documented temporal models: **TCN, Multi-Scale CNN, 1D-ResNet and Time-Series Transformer**. Stage 23 will carry out the matched held-out-subject model comparison; smoke-test outputs alone must not be represented as benchmark evidence.
