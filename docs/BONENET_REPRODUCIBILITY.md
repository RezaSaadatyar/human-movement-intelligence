# BoneNet / Essex | Research Reproducibility and Evidence Boundaries

This guide accompanies the completed [Biomechanical Load Estimation research project](../biomechanical_load_estimation/README.md) in Human Movement Intelligence, in the same documentation role as [the HARTH reproducibility guide](HARTH_REPRODUCIBILITY.md). The project has **eight numbered Jupyter notebooks**, a source-verified Essex dataset reconstruction and a **one-time, pre-frozen participant-held-out Test** in Stage 08.

For original scientific results, model specifications and limitations, see [Research Methods and Experimental Evidence](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md). The present guide describes what a third party can **verify from committed evidence**, what requires reconstructing the original data, and which operations must **not** be repeated merely to improve the recorded final score.

## 1. Source version and reproducibility boundary

- **Original source:** [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), originally 40 participants and 1,897 trials; source record recovery retained 37 participants and 1,729 trials with eligible original IMU–FE pairs.
- **Pinned original archive SHA256:** \`c50f10d1364686412b7ab64cac435dc85864dc8dbb34d2804b9cf37f2e29132d\`.
- **Original paired numeric source SHA256:** \`57a0e97fd68c7b0c813bc02c892d879926464c85e3cac08325aaabc2e2762e95\`.
- **Input:** Seven six-channel lower-body IMUs, 101 original temporal samples, 42 total channels per trial.
- **Targets:** Original FE \`e1\` and \`e3\` strains at eleven source-matched IMU positions (11×2). Published target values are **unitless**. Neither FE target interpolation nor conversion to microstrain was performed.
- **Participants:** Training 29 / 1,354 trials; Validation 4 / 192; final reserved Test 4 / 183. The final Development fit used Training+Validation (33 people / 1,546 trials) after model choice was frozen.
- **Task:** Regression of original model-derived femoral-neck FE trajectories, not activity classification, medical diagnosis or direct measurement of physical bone loading.

Source archives, private reconstructed full numeric arrays, per-fold training caches and final estimator weights are **not committed** to the repository. The study provides algorithmic code, notebook logs, original-source checksums and saved scientific output files; that does **not** imply bitwise independent reproducibility in a different runtime or GPU.

## 2. Recorded scientific evidence available in GitHub

The [stage-wise evidence register](../biomechanical_load_estimation/evidence/stage_01_08_evidence_register.csv) points to the current committed artifacts. It is a **read-only index**, not an independent scientific audit or a new model experiment.

| Evidence | Committed research record | What can be independently checked |
| --- | --- | --- |
| Original archive / source record coverage | [Stage 01 source audit](../biomechanical_load_estimation/output/01_consolidated_source_and_trial_quality_gate.json), [trial registry](../biomechanical_load_estimation/output/01_original_record_registry.csv) | Dataset identity, source-trial inventory and coverage assertions |
| MATLAB object decoding | [Stage 02 decoding gate](../biomechanical_load_estimation/output/02_consolidated_decoding_quality_gate.json), [reader diagnostics](../biomechanical_load_estimation/output/02_python_decode_audit.csv) | Source representation and recovery decisions recorded at that stage |
| Numeric and original temporal coordinate checks | [Stage 03 numeric/time gate](../biomechanical_load_estimation/output/03_consolidated_numeric_temporal_quality_gate.json), [original time mapping](../biomechanical_load_estimation/output/03_original_time_coordinate_mapping.csv) | Original FE-index alignment and source numeric traceability |
| Final source-governed split | [Stage 04 split gate](../biomechanical_load_estimation/output/04_dataset_and_split_quality_gate.json), [assignments](../biomechanical_load_estimation/output/04_trial_split_assignments.csv) | Source fingerprint, subject disjointness and eligible/excluded trial counts |
| Classical model selection | [Stage 05 CV summary](../biomechanical_load_estimation/output/05_training_cv_model_summary.csv), [quality gate](../biomechanical_load_estimation/output/05_regression_protocol_and_quality_gate.json) | Training-only classical ranking and frozen GroupKFold alignment |
| Temporal development | [Stage 06 CV and Validation results](../biomechanical_load_estimation/output/06_training_cv_model_summary.csv), [temporal gate](../biomechanical_load_estimation/output/06_temporal_protocol_and_quality_gate.json) | Temporal CNN/TCN development scores and native FE-grid design |
| Recurrent and seed robustness | [Stage 07 candidate decision](../biomechanical_load_estimation/output/07_training_candidate_decision.csv), [75-job manifest](../biomechanical_load_estimation/output/07_job_manifest.csv), [Stage 07 gate](../biomechanical_load_estimation/output/07_robustness_protocol_and_quality_gate.json) | Training-only multiseed design, peak-loss ablation and pre-Test RF choice |
| Final pre-Test freeze | [Stage 08 frozen protocol](../biomechanical_load_estimation/output/08_frozen_final_protocol.json) | Model identities, exact fit budgets, outcome definitions and one-time Test release rule recorded independently from final results |
| One-time authorised Test evidence | [Release receipt](../biomechanical_load_estimation/output/08_final_test_release_receipt.json), [Final gate](../biomechanical_load_estimation/output/08_final_test_quality_gate.json) | Recorded release, 4-person/183-trial scope, absence of Test-based model reselection |
| Saved Test predictions and comparison | [Original-grid predictions](../biomechanical_load_estimation/output/08_test_trial_predictions.csv), [model metrics](../biomechanical_load_estimation/output/08_test_primary_secondary_metrics.csv), [bootstrap](../biomechanical_load_estimation/output/08_test_subject_bootstrap.csv) | Recompute participant-weighted errors and confirm descriptive uncertainty from published trial-level outputs |

**Important execution qualification:** All saved code cells are executed in Stages **03–08**, with no saved error outputs; Stages **01–02** retain some unexecuted source-inspection cells. Do not claim that every cell in all eight notebooks executed. The downstream Stage 04–08 saved gates support a consistent *recorded* source/model lineage, not independent replication.

## 3. Inspecting saved results without fitting or opening Test again

The safest research review is **read-only**. Begin with the [Stage 08 final notebook](../biomechanical_load_estimation/code/notebooks/08_frozen_final_model_evaluation_and_research_synthesis.ipynb) and the [final gate](../biomechanical_load_estimation/output/08_final_test_quality_gate.json), then examine the [source/split gate](../biomechanical_load_estimation/output/04_dataset_and_split_quality_gate.json) and prior Training-only [candidate selection](../biomechanical_load_estimation/output/07_training_candidate_decision.csv).

There are **16 committed \`08_*\` artifacts**, including the original before-Test freeze, final fit manifest, 35-epoch TCN history, one-time release receipt, 8,052-row long-form predictions, Test model/subject/task diagnostics, three bootstrap rows, five genuine scientific figures and the final PASSED quality gate.

A reviewer can recalculate **subject-balanced MAE** by grouping \`08_test_trial_predictions.csv\` by model and **participant**, averaging absolute FE error across each participant's original trial/time/target observations, then averaging the four participant means equally. The two reported values are:

| Pre-frozen role | Final same-source Test subject-balanced MAE |
| --- | ---: |
| **Random Forest, primary** | 0.0009384 |
| Dilated TCN, fixed descriptive secondary | 0.0008759 |

The secondary model has lower observed waveform and peak-magnitude error, but the primary Random Forest was not and must not be changed after Test. A four-participant paired cluster bootstrap (2,000 draws) for TCN minus RF has difference **−0.0000625** and exploratory 95% interval **[−0.0001859, +0.0000431]**, overlapping zero. This is a limited same-cohort comparison, not independent external validation.

**Do not rerun the Stage 08 Test release and evaluation** to generate an improved final score, even in another checkout where the local release guard cannot enforce exclusivity. Results must remain a faithful record of the predefined original assessment.

## 4. Full original-source reconstruction and computational requirements

For technical replication of early stages, start from the original dataset rights and exact published source version, then follow the code and installation cells under:

- [\`biomechanical_load_estimation/code/notebooks/01–04\`](../biomechanical_load_estimation/code/notebooks/04_governed_imu_fe_dataset_and_participant_splits.ipynb) for raw source, nested MATLAB decoding, original FE indexing and participant splits.
- [Stage 05](../biomechanical_load_estimation/code/notebooks/05_classical_regression_baselines_and_participant_independent_validation.ipynb) for classical grouped regression and its training-fold registry.
- [Stage 06](../biomechanical_load_estimation/code/notebooks/06_deep_temporal_regression_and_peak_prediction.ipynb) and [Stage 07](../biomechanical_load_estimation/code/notebooks/07_multi_seed_robustness_gru_bigru_and_peak_ablation.ipynb) for original-grid temporal methods and Training-only robustness.

From the repository root, baseline dependencies are described by \`requirements.txt\`, with optional deep-learning packages in \`requirements-dl.txt\`. The notebook cells include additional original-source reader setup, supported Colab repository synchronisation and shared \`core.computer_device()\` reporting. Original reader availability, Figshare downloads, device memory and runtime differences can affect full recomputation. Stage 07 has **75 Training-only neural fits**, so a compatible GPU and persistent local, Git-ignored per-fold cache are advisable.

**The Stage 08 one-time Test is not a routine reproducibility cell.** Reproducing the source and Development computations can be a legitimate technical exercise; reproducing the final protected Test assessment as if it were a new confirmatory result after seeing its scores would violate the originally frozen evaluation boundary. For this completed study, use the recorded public Test predictions and gate for scientific review.

## 5. Cross-stage comparability and model integrity

- Stage 05 Random Forest CV (**0.0007950**) comes from its earlier five-fold classical benchmark. Stage 07 temporal-model metrics are means of **three** seed-specific five-fold Training OOF evaluations; they do not represent identical model capacity or computation.
- Stage 05 and 06 Validation results came from four people who had been inspected diagnostically. They were not used for Stage 07 model re-selection, and cannot be described as independent external Validation.
- Final Stage 08 models were **refitted once on Training+Validation** after the primary Random Forest choice had been frozen. Their Test results are not predictions from the earlier CV models.
- Stage 07's peak penalty (0.25) did not exhibit a dependable improvement in the controlled no-peak TCN ablation; no retrospective loss retuning was performed using Test.
- The source FE targets remain **unitless** and model-derived. Peak-index MAE is measured on the **11 original FE indices**, not seconds, microstrain or degrees of injury risk.
- The local Test release marker is a protocol safety aid, not a globally enforced lock across machines.

## 6. Scientific boundaries and final status

The final evaluation has only **four independent Test participants** from the **same Essex collection**. Performance under distribution shift, independent institutions, differently mounted wearables, real FE-model calibration, prospective use or clinical effectiveness was not established.

The project records actual results and a frozen-before-Test single-use evaluation under a **PASSED** final recorded gate. Its output is a **source-verified exploratory computational benchmark**, not a medical device, validated clinical model or promise of same accuracy in unseen populations.

**Navigation:** [Study README](../biomechanical_load_estimation/README.md) · [Methods report](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) · [Stage evidence register](../biomechanical_load_estimation/evidence/stage_01_08_evidence_register.csv) · [Study overview](../biomechanical_load_estimation/images/study_overview.svg) · [Final workflow](../biomechanical_load_estimation/images/final_workflow.svg).
