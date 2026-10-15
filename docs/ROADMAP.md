# Current Research Status (15 October 2026)

**The original planning/checkpoint log below is preserved as a historical record. Its earlier unchecked items and "pending" descriptions reflect their status when written, not the latest GitHub execution state.** For current results, see [the executed HARTH study](../studies/wearable_activity_recognition/README.md).

| Area | Verified present status | Evidence |
| --- | --- | --- |
| HARTH Stages 01–19 | Historical executed exploratory analyses; different cohort sizes and model protocols | [Evidence register](../studies/wearable_activity_recognition/experiments/stage_01_19_evidence_register.csv) |
| Stage 20–22 | Executed evidence consolidation, quality audit and temporal architecture checks | [Notebooks](../studies/wearable_activity_recognition/notebooks/) |
| Stage 23 | Executed seven-model, 3-fold matched 29-person Development benchmark; 21 jobs | [Notebook 23](../studies/wearable_activity_recognition/notebooks/23_unified_matched_model_benchmark.ipynb) |
| Stage 24 | Executed multi-seed/loss robustness, 36 Development fits | [Notebook 24](../studies/wearable_activity_recognition/notebooks/24_multiseed_and_imbalance_robustness.ipynb) |
| Stage 25 | Executed Development-only model selection and source-fingerprinted protocol | [Notebook 25](../studies/wearable_activity_recognition/notebooks/25_model_selection_and_protocol_freeze.ipynb) |
| Stage 26 | Executed Development final fitting, approved one-time qualified evaluation on S023/S032; final gate PASSED | [Notebook 26](../studies/wearable_activity_recognition/notebooks/26_final_development_training_and_inference.ipynb) |
| Stage 27 | Executed saved-evidence synthesis, ten code cells, final gate PASSED; no raw/Test reloading | [Notebook 27](../studies/wearable_activity_recognition/notebooks/27_final_research_evidence_and_report.ipynb) |
| Biomechanical Load Estimation | Original data inspection and separately documented preliminary exploratory results, not completed robust load-estimation benchmark | [Study notes](../studies/biomechanical_load_estimation/) |
| Other movement science studies | Planned, not yet implemented | [Study catalogue](../studies/README.md) |

**Current scientific result:** 1D-ResNet Weighted CE / 120-tree Random Forest qualified pooled fixed-eight-class Macro F1 **0.7182 / 0.6817**, from only two historically inspected same-study held-out people and 1,000 overlapping windows. This is not independent external validation or clinical deployment evidence.

## Remaining repository hygiene and release decisions

- [x] Stages 23–27 executed evidence exists in GitHub and final notebook Quality Gates were reviewed.
- [x] Align root and HARTH study README with the actual results and historical Test limitations.
- [x] Correct obsolete Stage 27 offline test assumption that the notebook should be unexecuted. **A successful latest CI run must still be independently verified.**
- [x] Read-only reproducibility/provenance audit implemented and checked by offline CI; reviewed dataset/source and model-artifact limits. The three original local model files and byte-for-byte GPU reproduction remain **unverified**, not silently declared resolved. See [the audit guide](HARTH_REPRODUCIBILITY.md).
- [x] Close superseded draft PR #1 without merging or deleting history.\n- [ ] Decide whether to create a versioned GitHub Release (a named snapshot/tag plus concise release notes). **No release is authorised or created merely by these documentation changes.**
- [ ] Decide the next independent study (e.g. biomechanical load estimation), with separate datasets, protocol, baselines, models and evaluation. Do not blend its outcomes into HARTH.

---

## Historical plan and stage-by-stage log (retained unchanged)

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

## Stage 11: matched Random Forest vs temporal CNN validation
- [x] Inspect completed Notebook 10: 14,201 sampled windows, last-epoch CNN Validation Macro F1 0.6659, Shuffling F1 0, subject-disjoint data.
- [x] Add reusable per-subject prediction evaluation and alignment tests.
- [x] Add Notebook 11 with identical development windows and splits for Random Forest and CNN, plus paired subject diagnostics.
- [ ] Execute Notebook 11 on real source data, review performance and rare-class failures before pursuing larger CNNs.

## Stage 12: training-only class-weighted temporal CNN
- [x] Inspect executed Notebook 11: matched Validation Macro F1 Random Forest 0.8266, CNN 0.6659; Shuffling CNN F1 0.
- [x] Add inverse-square-root training-only class weighting without changing the default unweighted CNN.
- [x] Add offline weight/loss tests, Notebook 12, and scientific protocol.
- [ ] Execute Notebook 12 on real development subjects and inspect all class-level and subject-level results.

## Stage 13: matched development GroupKFold for model and loss robustness
- [x] Inspect completed Notebook 12: Standard CNN Macro F1 0.6659, Weighted CNN 0.7968 on the same six Validation subjects; improvement is exploratory.
- [x] Implement reusable subject-disjoint grouped comparison of RF, Standard CNN and Weighted CNN with out-of-fold prediction provenance.
- [x] Add offline group-alignment tests, commented Notebook 13, and protocol documentation.
- [ ] Execute Notebook 13 on actual bounded HARTH development data and analyse fold-, subject- and class-level outcomes.
- [ ] Decide whether the weighted CNN warrants a fuller dataset/compute budget before any reserved Test scoring.

## Stage 14: temporal sampling and minority-class coverage
- [x] Inspect executed Notebook 13: 29-person 3-fold Macro F1 mean Random Forest 0.8076, Weighted CNN 0.7834 and Standard CNN 0.6301; Shuffling weighted CNN pooled F1 0.1345.
- [x] Add reusable source-window sampling audit and offline source-index integrity tests.
- [x] Add Notebook 14 to compare per-subject caps 250, 500 and 1000 against full eligible development data, without fitting models.
- [ ] Run Notebook 14 and review minority-activity retention and independent-person support before increasing deep-learning compute.

## Stage 15: nested training-data scaling on fixed participant validation
- [x] Verify executed Notebook 14: 29 development people, full 132,397 valid windows; cap 500 retains 219 Shuffling windows and cap 1000 retains 386.
- [x] Add nested chronological 500/1000 sample utility and HARTH source adapter with offline tests.
- [x] Add Notebook 15 comparing Random Forest and Weighted CNN at two nested Train sizes on the original fixed six-person Validation set.
- [ ] Execute Notebook 15 and inspect actual class/person effects before selecting the next data-volume experiment.

## Stage 16: grouped evaluation of nested training scale
- [x] Inspect executed Notebook 15: RF Macro F1 0.8266→0.8644 (Train 500→1000), Weighted CNN 0.7968→0.7805 at last epoch on one fixed 6-person Validation group.
- [x] Add reusable nested training-scale grouped CV module, synthetic group-alignment tests, Notebook 16 and protocol.
- [ ] Run Notebook 16 on true data, review paired participant results and low-frequency class effects before changing architectures.

## Stage 17: fixed-update temporal CNN training-size sensitivity
- [x] Inspect executed Notebook 16: GroupKFold RF 500/1000 Macro F1 0.8076/0.8097; Weighted CNN 500/1000 0.7834/0.7962; Shuffling CNN pooled F1 0.1345/0.2513.
- [x] Implement opt-in fixed-step PyTorch trainer preserving existing epoch-based function, with regression tests.
- [x] Add annotated Notebook 17 and predeclared four-checkpoint, 400-update training protocol.
- [ ] Execute Notebook 17 and verify full outputs before designing the next experiment.

## Stage 18: development-only out-of-fold CNN confidence audit
- [x] Inspect executed Notebook 17: Weighted CNN 500/1000 fixed-400-step Macro F1 0.7746/0.7927 across three grouped folds, Shuffling still weak.
- [x] Add train-statistics-only softmax inference and reusable prediction-confidence audit with synthetic tests.
- [x] Add annotated Notebook 18 and confidence interpretation protocol.
- [ ] Execute Notebook 18 and review reliability bins, per-class errors and participant-level variation before making any calibration or selective prediction claims.

## Stage 19: subject-disjoint scalar temperature calibration
- [x] Inspect executed Notebook 18: 14,201 OOF development windows, ECE 0.0453, Brier 0.0905, 40 incorrect predictions with maximum softmax >=0.90; Shuffling recall 0.1233.
- [x] Add reusable scalar-temperature fitting and multiclass held-out NLL with offline tests.
- [x] Add documented Notebook 19 with independent Train, Calibration and outer Validation participant groups and top-1 invariance quality gate.
- [ ] Execute Notebook 19, review real held-out fold-level calibration metrics, and avoid claims based on calibration-fit performance.


## Agreed HARTH completion plan: Stages 20-27

**Status:** Approved future study plan, not yet implemented. Stage 19 was executed and reviewed. The goal is to COMPLETE the HARTH study with purposeful experiments and reproducible reporting, rather than create one new notebook for every minor concern.

### Notebook consolidation rules

- Combine closely related investigations in one complete, readable numbered notebook; preserve full methodology, tables, visualisations, results and limitations. Do not reduce evidence just to shorten a notebook.
- Keep reusable dataset adapters, neural architectures, training loops, evaluation functions and tests in documented Python modules. Notebooks call those modules, narrate the method and retain real executed outputs.
- Implement and actually run **all four** additional temporal architectures: **Temporal Convolutional Network (TCN), Multi-Scale CNN / InceptionTime-style CNN, 1D-ResNet and Time-Series Transformer**. None of these is merely a suggested optional architecture.
- Prefer **one combined architecture-study notebook** and **one matched multi-model evaluation notebook**, rather than a separate notebook per architecture. If a complete CPU experiment proves too costly, split model training into two coherent families with saved intermediate results, then retain one unified final comparison; do not silently omit a model.
- Clearly distinguish implemented/smoke-tested models from those trained and empirically validated. No improvement should be claimed before actual outputs are reviewed.

### Planned stages and deliverables

- [ ] **Stage 20 – Evidence consolidation.** Summarise completed Stages 01–19 in a traceable table of source cohort, sampling cap, subject split, metric denominator, training budget, calibration scope, outcomes and limitations. Harmonise README/roadmap status. Prefer one analysis notebook or concise reproducible report, not rerun all models.
- [ ] **Stage 21 – Minority-class and data-quality assessment.** Audit Shuffling, ascending/descending stairs, between-subject class availability, chronological sampling effects, and officially undocumented codes without guessing meanings. Combine coverage and error diagnostics into one coherent notebook where practical.
- [ ] **Stage 22 – Four temporal-model implementations and smoke checks.** Create reusable PyTorch definitions/tests for TCN, Multi-Scale CNN, 1D-ResNet and Time-Series Transformer. Use **one** notebook for architecture diagrams/tensor flow, shape tests, parameter and resource counts, controlled tiny training runs and rationale, with full English code comments/documentation.
- [ ] **Stage 23 – Unified matched subject-held-out architecture benchmark.** Actually train/evaluate **all four new models** alongside existing Random Forest, Standard CNN and Weighted CNN reference baselines using identical source participant identities, windowing, sampled rows, class specification, held-out folds and transparent compute budget. Save fold/person/class metrics, confusion matrices, inference latency and resource usage. Use **one** final comparison notebook; modular/cached training may be split only if computing constraints require it.
- [ ] **Stage 24 – Robustness and imbalance follow-up.** Run multi-seed analysis on a justified shortlist and a targeted training-only Focal Loss / class-balanced loss or physically valid augmentation ablation as indicated by Stage 21/23 results. Keep losses and architecture/data changes isolated. Combine connected tests into one notebook where practical; do not claim a method improves results without evidence.
- [ ] **Stage 25 – Freeze final model and evaluation protocol.** Select model(s) on Development only using subject-held-out Macro F1, minority-class outcomes, variability and compute/reproducibility. Lock code, hyperparameters, classes and preprocessing before any reserved-Test scoring. Document that reserved Test subjects S023/S032 were inspected in earlier exploratory stages and are NOT fully preregistered-unseen.
- [ ] **Stage 26 – Final evaluation and reproducible inference.** After explicit protocol approval, fit under the frozen procedure and report transparently qualified held-out results (if using the reserved Test). Save compatible model weights, sensor order, label map and Train-fitted preprocessing; test loading and inference end to end.
- [ ] **Stage 27 – Study report and GitHub release.** Reconcile all source-grounded figures and tables; write final methods, results, class-level errors, comparisons, limitations, lightweight tests/CI and README instructions. Archive or clearly label exploratory notebooks; deliver a coherent reproducible version-one HARTH study.

### Fair comparison boundaries

- Development: 29 HARTH v2.0 participants; protect original `S023` and `S032` from further model development and selection.
- Predeclare the 8-class task and scientifically justified handling of undocumented source codes; keep patient/participant identity separated between Train and Validation and avoid raw-window leakage.
- Compare architectures on **identical subject groups and window-row selection**; use explicitly documented Train-only scaling/weighting, fixed update budgets (with realistic model-specific compute reporting) and equal evaluation definitions. Parameter counts, compute and training stability must accompany scores; equal steps are NOT equal compute or proof of architectural causality.
- Avoid comparing experiments with different cohorts/window caps as though they were directly matched. Do not cherry-pick the best epoch on Validation and then call it independent evaluation.
- Report all models even if one is weak or too costly; if a run cannot complete, document the failure and reason rather than fabricate outcomes.

### Stage 20 implementation checkpoint (awaiting local execution)

- [x] Review and inventory all 19 historical HARTH notebooks and their saved outputs.
- [x] Create comprehensive **bold-heading** Notebook 20 and source-linked 19-stage evidence register.
- [x] Add reusable saved-notebook provenance utilities and offline regression tests.
- [x] Document matched/unmatched experiments, rare-class evidence, and the original Test provenance limitation.
- [ ] Run Notebook 20 locally, inspect inventory findings and execute its quality gate; commit and push real results.
- [ ] Review that execution before opening Stage 21.

**Formatting convention for all upcoming notebooks:** Use `**01. Section Title**` or equivalent **bold Markdown**, not hash-style headings. Preserve complete explanations, analysis and outputs despite notebook consolidation.


### Notebook presentation and review standard (Stages 21–27)

See [HARTH Notebook Authoring Standards](../studies/wearable_activity_recognition/NOTEBOOK_AUTHORING_STANDARDS.md): comprehensive `**Bold Markdown**` section names, clearly labelled and numbered English code comments, readable tables/figures, separated printed summaries and source-grounded quality gates. Do not compress methodological detail merely to reduce the number of notebooks. Every stage must have its locally executed results pushed and reviewed before beginning the next stage.


### Stage 21 implementation checkpoint (awaiting local source-data execution)

- [x] Confirm saved Stage 20 Quality Gate displays PASSED; note its final cell's GitHub `execution_count` is still null despite the saved successful output.
- [x] Implement official HARTH source-only data quality, activity-label, annotation-bout and timestamp-boundary utilities.
- [x] Add complete 11-section Notebook 21 with bold Markdown titles, numbered scientific code comments, detailed tables, coherent figures and descriptive Quality Gate.
- [x] Add synthetic offline audit tests and the Stage 21 scientific protocol.
- [ ] Execute Notebook 21 in Cursor on real official development CSVs, review all errors and outputs, then Save/Commit/Push.
- [ ] Inspect its saved real-data results and Quality Gate before starting Stage 22 (TCN, Multi-Scale CNN, 1D-ResNet and Transformer).


### Stage 22 engineering implementation checkpoint

- [x] Verify saved Stage 21 results: 11/11 executed cells, no stored errors, original 29-person development source audit 132,397 eight-class windows, Quality Gate PASSED.
- [x] Implement all four mandatory PyTorch architectures (TCN, Multi-Scale CNN, 1D-ResNet, Time-Series Transformer) with one shared source-tensor interface.
- [x] Add parameter counter, controlled tensor/graduation tests, one-step optimiser smoke checks and Notebook 22 using professional bold Markdown and numbered Python comments.
- [x] Record detailed Stage 22 protocol, engineering scope, limitations and Stage 23 controls.
- [ ] Run Stage 22 locally on real development source windows, inspect saved outputs and final Quality Gate, then commit/push executed notebook.
- [ ] Check GitHub Actions for the full latest head and resolve errors if any before Stage 23.


### Stage 23 implementation checkpoint (pending benchmark execution)

- [x] Inspect Stage 22 pushed notebook: ten executed source and engineering cells, four architectures, final Quality Gate PASSED.
- [x] Add source-and-code-fingerprinted seven-model/three-fold matched benchmarking functions with safe local JSON resume.
- [x] Create comprehensive Notebook 23 with bold Markdown, numbered code comments, actual training loop, source-aligned fold/class/person diagnostics and strict complete-result Quality Gate.
- [x] Add offline synthetic integrity tests and formal Stage 23 scientific protocol.
- [ ] Execute all 21 model-fold jobs on actual bounded Development HARTH recordings; verify local PyTorch tests and all final notebook outputs.
- [ ] Review GitHub CI for latest head and resolve issues before treating Stage 23 as completed.
- [ ] Commit/push the executed Notebook 23. Inspect genuine class and participant outcomes before designing Stage 24.


### Stage 24 implementation checkpoint (awaiting genuine Colab GPU results)

- [x] Inspect executed Stage 23: all 21 matched jobs complete on Tesla T4; all 12 code cells executed and final Quality Gate PASSED.
- [x] Predeclare ResNet/TCN multi-seed comparison (42, 123, 2026), paired Fold membership and original Weighted Cross-Entropy.
- [x] Add controlled ResNet Weighted CE vs Focal Loss vs effective-number Class-Balanced CE comparison; share original Weighted CE runs between analyses.
- [x] Implement 36 source/code-fingerprinted resumable GPU fits and 12-section comprehensive Notebook 24 with numbered English comments, bold Markdown, real outcome plots and strict scientific Quality Gate.
- [x] Add synthetic offline loss/gradient and reproducibility-integrity tests, scientific protocol and interpretation boundaries.
- [ ] Run Notebook 24 in Google Colab GPU; inspect and push all genuine output tables/figures and final Quality Gate before Stage 25.


### Stage 25 implementation checkpoint (proposed protocol, no Test access)

- [x] Inspect executed Stage 24: 36/36 GPU model/fold jobs, 12 experiment conditions, no errors, final Quality Gate PASSED.
- [x] Recover Stage 23/24 result evidence directly from complete saved notebook tables and matching Development fingerprint.
- [x] Add a comprehensive 11-section selection Notebook 25 with bold Markdown titles, numbered explanatory code comments, comparison charts, code SHA256 and a review-required protocol JSON proposal.
- [x] Document final ResNet Weighted CE development candidate, fixed RF comparison, sample cap, source class policy and reserved-Test interpretation limitations.
- [x] Add read-only evidence parsing tests, model-selection protocol and explicit user approval boundary.
- [ ] Run Notebook 25 and offline tests, review resulting `stage_25_protocol_proposal.json`, save and push real outputs.
- [ ] User explicitly approves exact protocol before Stage 26 final model fitting / qualified held-out scoring.


### Stage 26 safe Development fit and inference implementation (approval pending)

- [x] Inspect executed Stage 25: 11/11 code cells with stored successful outputs; proposed final manifest still `PENDING_USER_REVIEW`.
- [x] Create final Development-only ResNet training and fixed RF reference, Train-only scaling, complete trusted inference artefacts, SHA256 checks and tests.
- [x] Add complete Notebook 26 with ten substantial bold Markdown sections, numbered code comments and a readable Development-only quality gate; **no Test loading cell**.
- [x] Document two explicit Test authorisation barriers and original reserved-subject interpretive limits.
- [ ] Run Notebook 26 on Colab GPU, verify saved model reload, retain local artefacts and push executed notebook.
- [ ] Obtain specific user review/approval before any original reserved Test load or score.
- [ ] Implement and run one qualified held-out evaluation only after that approval; review real results before Stage 27.
