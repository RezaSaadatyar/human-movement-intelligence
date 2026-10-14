# Implementation roadmap

**Current status (13 October 2026):** Figshare v13 inventory and 40 participant records inspected. The downloaded MATLAB archive has been safely extracted on the maintainer's Windows machine (40 MAT files); top-level headers in five files contain `Data` and `__function_workspace__`. First-element nested fields were inspected for `subj04` and `subj16` (`Data` shape `[48,1]` in both). Full-record shapes and trial mappings are not yet validated. No model has been trained.

## Phase 0 — Repository foundation
- [x] Confirm broad independent scope and repository name.
- [x] Establish proposed architecture and initial study catalogue.
- [x] Record data access and citation rules.
- [x] Publish research and code-writing standards.
- [x] Select biomechanical load estimation and verify the six-file Figshare v13 remote inventory.

## Phase 1 — First dataset inspection
- [x] Review the publication, participant exclusions and authors' example.
- [x] Add Figshare inventory and MATLAB header-inspection utilities.
- [x] Download README, subject workbook and MATLAB.zip; verify the downloaded byte counts and optional checksums.
- [x] Inspect source README and subject metadata; document discrepancies.
- [x] Audit 42 ZIP entries (40 MAT files, 224.79 MiB uncompressed, no path-safety warnings).
- [x] Extract 40 MAT files into an ignored local directory.
- [x] Inspect top-level MAT variable headers in five subjects.
- [x] Add a bounded nested-MAT metadata inspector and synthetic-data tests.
- [x] Run first-element nested field/shape inspection on actual subj04 and subj16 files (both have `Data` shape `[48,1]` and the same 13 first-element field names).
- [ ] Review generated nested schema JSON for actual IMU, ID, JCF and e1e3 shapes/dtypes.
- [x] Audit all existing Data elements across 40 files: 1,897 records, 168 explicitly empty e1e3 fields and 1,729 nonempty opaque objects.
- [ ] Recover MATLAB-native classes, numeric-variable shapes and usable target completeness.
- [ ] Validate all 40 subject IDs against the workbook, target coverage and exclusions.
- [ ] Produce dataset schema, missingness audit and quality report.
- [ ] Finalise one well-defined input/target contract and source-record keys.

## Phase 2 — Reproducible baseline
- [ ] Implement and test dataset adapters after validating nested fields.
- [ ] Create immutable subject-independent splits and train-only preprocessing.
- [ ] Compare transparent statistical and classical regression baselines.
- [ ] Store predictions, metrics, configurations and actual run metadata.
- [ ] Publish runnable notebook and study README.

## Phase 3 — Temporal deep learning
- [ ] Compare 1D-CNN and BiLSTM/TCN on the same group splits.
- [ ] Report subject-wise errors and target-specific trajectory/peak metrics.
- [ ] Justify any additional Transformer experiments.

## Phase 4 — Data efficiency and transfer
- [ ] Evaluate learning curves, missing sensors, domain shift and optional self-supervised learning.
- [ ] Record ablations and negative results.

## Phase 5 — Additional studies
- [ ] Activity recognition, gait kinematics, video-based pose analysis.
- [ ] Neurorehabilitation and motor signals.
- [ ] Co-recorded multimodal sensing when suitable datasets are available.

A model or scientific result is only called implemented/validated after actual execution. **Next action:** use native MATLAB (if available) to inspect IMU/ID/e1e3 class and variable names for two subjects; do not train a model with opaque object references.


## Python-first implementation decision (13 October 2026)
- [x] Set Python as the default computational stack; MATLAB-native inspection is optional and not required for the next step.
- [x] Implement a Python-only download, structural audit and opt-in subject-held-out Ridge baseline for the authors' numeric BoneNet derivative.
- [x] Pass five unit tests with synthetic source-shaped MAT data.
- [ ] Verify actual BoneNet Data.mat/labels.csv tensors and pairing through one reproducible Python audit.
- [ ] Evaluate a separate Python MCOS decoding adapter for the *original* Figshare files without assuming support for this particular object type.

See [Python-only pipeline](PYTHON_FIRST_PIPELINE.md). No authentic-data model metrics have been reported.


## Confirmed Python-only BoneNet pilot (14 October 2026)
- [x] Real authors' public Data.mat and labels.csv downloaded and inspected in GitHub Actions.
- [x] Numeric tensors verified: X=(444,42,101), Y=(444,101,2), zero non-finite values.
- [x] Fix synthetic MAT audit test fixture; CI passes 14/14 tests.
- [ ] Review exploratory grouped Mean/Ridge results and verify source sample-to-label pairing before making any scientific generalisation claim.

See [actual-data validation record](../studies/biomechanical_load_estimation/BONENET_REAL_DATA_AUDIT.md).


## Phase 1 of reusable HMI Framework (14 October 2026)
- [x] Create installable Python package `hmi`, first structural `DatasetBundle` contract and seven offline tests.
- [ ] Execute the interactive contract notebook on actual BoneNet data and review outputs.
- [ ] Implement generic readers, dataset adapters and modality-aware profiling, then validate against a second dataset.

See [data contract](DATA_CONTRACT.md). The earlier BoneNet baseline remains exploratory and is not a completed general ML framework.
