# Current research development roadmap

## Phase 1: BoneNet data loading with visible outputs

- [x] Use simple general Python readers and inspectors; no custom HMI package required.
- [x] Load actual BoneNet IMU and model-derived FE strain arrays.
- [x] Review **maintainer-executed** notebook in GitHub: 6 code cells, real labels, signal plots and quality gate; no model training.
- [x] Preserve the version the maintainer committed, including cell outputs.
- [ ] Independently verify BoneNet source-to-label row alignment before treating previous regression results as valid evidence.

## Phase 2: demonstrate reuse across a second public dataset

- [x] Review authoritative HARTH sources and versions: UCI v1.2 (22 participants), NTNU official v2.0 (31 participants).
- [x] Keep existing reusable functions unchanged.
- [x] Add a small HARTH adapter, one-participant data-inspection notebook and synthetic source tests.
- [ ] Run the HARTH notebook on an actual authors' `S023.csv` and inspect results/plots.
- [ ] Review notebook execution in Cursor with the maintainer.
- [x] Implement a reusable, annotation-safe signal windowing function and add an interactive HARTH Notebook 02.
- [x] Execute and review Notebook 02 on actual HARTH: 2,751 candidate windows, 2,567 retained, 184 rejected, six channels × 100 points.
- [x] Implement shared, stateless feature extraction and prepare Notebook 03 with commented Python cells.
- [x] Execute and review Notebook 03 on real HARTH: 2,567 windows and 30 finite features.
- [x] Add Notebook 04 and reusable subject-disjoint split function for eight verified HARTH participants.
- [x] Run and review Notebook 04 on eight participants; no subject overlap; 25,963 windows and missing-class analysis.
- [ ] Determine full-dataset subject-independent validation before model training.

## Phase 3: progressively build ML / DL tools

- [ ] Develop justified signal preprocessing and windowing, with tests for sample/subject leakage.
- [ ] Add reusable feature engineering, grouped train/validation/test splits and evaluation.
- [x] Implement three exploratory classification baselines and commented Notebook 05 for Validation-only comparison.
- [x] Execute and inspect Notebook 05: Logistic Regression Validation Macro F1 = 0.8343; 1,400 excluded out-of-task windows; final Test untouched.
- [x] Implement GroupKFold diagnostics with commented Notebook 06 and unit tests.
- [ ] Execute Notebook 06 on real six-person development set and review per-fold/per-subject metrics.
- [ ] Expand full-cohort subject-independent assessment before PyTorch CNN/temporal experiments.
- [ ] Document real per-subject outcomes and limitations.

The repository remains modular: each study has a shareable study-specific notebook and README.

## Stage 7: expanded HARTH development evaluation (14 October 2026)
- [x] Verify maintainer-executed Notebook 06: quality gate passed, 6 development subjects, 2 protected Test subjects.
- [x] Extend the HARTH study adapter to prepare ten development subjects without loading original Test subjects.
- [x] Add offline cohort tests, annotated Notebook 07, and a documented label/coverage protocol.
- [ ] Execute Notebook 07 and review actual subject coverage and 5-fold CV outcomes in Cursor before interpreting model performance.
- [ ] Design full-cohort evaluation and a final locked-test policy before expanding to deep learning.

## Stage 8: HARTH full-development cohort audit
- [x] Review executed Notebook 07: ten development subjects, 31,903 windows; Random Forest 5-fold Macro F1 0.7235, Logistic Regression 0.7188 (exploratory).
- [x] Prepare a reusable subject/class coverage and GroupKFold feasibility audit with offline tests.
- [x] Add Notebook 08 with 29 development participants, separate pre-download inventory, class coverage and fold audit.
- [ ] Execute Notebook 08 on real data and review coverage, errors and missing classes before any further model training.
- [ ] Freeze full-cohort evaluation strategy and report reproducibility limitations.

## Stage 9: full-development classical ML Cross-Validation
- [x] Verify executed Notebook 08: 29 development subjects, 132,397 eligible windows, 30 features, no missing training or validation classes in five audited folds.
- [x] Add reusable paired-subject performance comparison and offline tests.
- [x] Add Notebook 09 for five-fold grouped CV, subject-level model comparison and visual audit, with original Test subjects excluded.
- [ ] Run Notebook 09 on the full development cohort in Cursor and review real results before advancing.
- [ ] Decide model-selection/final test-reporting protocol without presenting the reserved Test as previously unseen.

## Stage 10: raw IMU temporal CNN with PyTorch
- [x] Verify executed Notebook 09: 29 development subjects; 5-fold Random Forest Macro F1 0.8339 and Logistic Regression 0.7710, with protected Test excluded.
- [x] Add source-specific bounded raw-window cohort loader and general 1D-CNN/training-only scaling utilities.
- [x] Add optional PyTorch dependencies, offline unit tests, documented training protocol and commented Notebook 10.
- [ ] Execute Notebook 10 on genuine HARTH source windows, review model outputs and validation plots, and then plan a matched CNN vs classical ML comparison.
