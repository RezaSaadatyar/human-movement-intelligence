# Implementation roadmap

**Current status:** Phase 0 architecture and data-source documentation. No modelling results are claimed.

## Phase 0 — Repository foundation
- [x] Confirm broad independent scope and repository name.
- [x] Establish proposed architecture and initial study catalogue.
- [x] Record data access and citation rules.
- [x] Publish research and code-writing standards.
- [ ] Select the first study and verify the actual data archive inventory.

## Phase 1 — First dataset inspection
- [ ] Open the source landing page, check record version/licence and list files.
- [ ] Decide between local archive acquisition and a reproducible API-based fetch.
- [ ] Confirm file types, subject IDs, sampling/normalisation conventions and feature/target availability.
- [ ] Produce a data dictionary, missingness report and data-quality summary.
- [ ] Decide one feasible initial research question and a genuinely available target.
- [ ] Document any exclusions and assess effective subject-level sample size.

**Initial candidate:** the 2026 Essex-linked femoral-neck loading dataset (see DATASETS.md). This is a candidate, **not yet inspected at file level**. If its archive is difficult to access or an appropriate target cannot be mapped, a UCI activity-recognition dataset is the practical fallback.

## Phase 2 — Reproducible baseline
- [ ] Add a validated dataset loader and tests.
- [ ] Implement group-aware splitting and explicit train-only preprocessing.
- [ ] Establish simple statistical or naive baseline, then classical ML.
- [ ] Record config, predictions and evaluation artefacts.
- [ ] Publish the first study README and one runnable notebook.

## Phase 3 — Deep learning experiments
- [ ] Implement a task-appropriate 1D-CNN and a temporal sequence model.
- [ ] Compare models under identical group splits.
- [ ] Evaluate participant-wise errors and meaningful biomechanics-specific outcomes.
- [ ] Add optional TCN/Transformer only if its added value is testable.

## Phase 4 — Data efficiency and transfer
- [ ] Evaluate learning curves and reduced training-label fractions.
- [ ] Investigate subject-independent transfer, domain shift or self-supervised learning where the dataset supports it.
- [ ] Run ablations and uncertainty analysis; report negative as well as positive findings.

## Phase 5 — Additional movement studies
- [ ] Wearable activity recognition across public benchmarks.
- [ ] Gait and joint kinematics.
- [ ] Video-based pose estimation and markerless analysis.
- [ ] Rehabilitation and movement-related physiological signals.
- [ ] Multimodal fusion only when aligned modalities are available.

## Delivery standards

A study is marked **Implemented** only when it has actual source code, a documented dataset interface and a tested execution path. Mark **Results available** only after the stated experiment has run and the metrics are recorded. No timelines or accuracy guarantees are implied by this plan.

**Next action:** verify the archive contents and licensing of the first selected dataset before writing specialised loaders or model code.
