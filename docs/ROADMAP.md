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
- [ ] Execute and inspect Notebook 05; audit class exclusions, Macro F1 and confusion matrix.
- [ ] Expand full-cohort subject-independent assessment before PyTorch CNN/temporal experiments.
- [ ] Document real per-subject outcomes and limitations.

The repository remains modular: each study has a shareable study-specific notebook and README.
