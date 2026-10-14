# Current research development roadmap

## Phase 1: simple data loading with visible outputs

- [x] Remove unnecessary custom HMI package from the active code.
- [x] Create general CSV/Excel/TXT/DAT/standard MAT reader.
- [x] Create data inspector and time-series plotter.
- [x] Keep BoneNet mapping in one dataset-specific adapter.
- [x] Prepare first notebook and tests on synthetic source files.
- [ ] Run and review the first notebook in Cursor on real BoneNet data.
- [ ] Confirm the new GitHub Actions notebook execution and tests.

## Phase 2: verify generalisation of tools
- [ ] Inspect actual HARTH files and develop only a needed adapter.
- [ ] Run and review a second notebook using common functions.

## Phase 3: scientific machine learning
- [ ] Verify labels, subject/trial keys and leakage-safe splitting.
- [ ] Add modular feature engineering and preprocessing with visible outputs.
- [ ] Add classical ML baselines, then justified PyTorch models and evaluation.

Historical dataset audits and preliminary BoneNet regression evidence remain available in study documentation, but are not a verified full research workflow.
