# HARTH study documentation and historical protocols

**Current study status:** All 27 stages have saved outputs, with the final Stage 27 research synthesis recorded as **PASSED**. For the verified findings, actual subject-level results and limitations, start with the [study README](../README.md). For raw notebook execution evidence, open the [primary notebooks](../notebooks/).

The 25 protocols indexed here are **distinct, stage-specific historical records**, not competing versions of the final README. Some state that an experiment, approval or later stage is *pending*, *planned* or *not yet performed*: those statements describe the situation **when the relevant protocol was written**. They do not override the currently saved notebooks or separately approved Stage 25/26 evidence. The historical methods and scientific caveats remain available for audit.

All relative links below point to the current repository locations. Plain repository paths mentioned *inside* historical documents are usually written relative to the repository root.

## Stages 01–08: acquisition, preprocessing and exploratory baselines

| Stage | Historical stage protocol | Executed notebook |
| --- | --- | --- |
| 01 | Not issued as a separate protocol; source inspection documented in notebook | [Notebook](../notebooks/01_data_loading_and_inspection.ipynb) |
| 02 | [Protocol](protocols/WINDOWING_PROTOCOL.md) | [Notebook](../notebooks/02_signal_preprocessing_and_windowing.ipynb) |
| 03 | [Protocol](protocols/FEATURE_ENGINEERING_PROTOCOL.md) | [Notebook](../notebooks/03_statistical_feature_engineering.ipynb) |
| 04 | [Protocol](protocols/SUBJECT_SPLITTING_PROTOCOL.md) | [Notebook](../notebooks/04_subject_independent_splitting.ipynb) |
| 05 | Not issued as a separate protocol; original ML baselines documented in notebook | [Notebook](../notebooks/05_classical_ml_baselines.ipynb) |
| 06 | [Protocol](protocols/CROSS_VALIDATION_PROTOCOL.md) | [Notebook](../notebooks/06_subject_level_cross_validation.ipynb) |
| 07 | [Protocol](protocols/EXPANDED_DEVELOPMENT_PROTOCOL.md) | [Notebook](../notebooks/07_expanded_subject_evaluation.ipynb) |
| 08 | [Protocol](protocols/FULL_DEVELOPMENT_AUDIT_PROTOCOL.md) | [Notebook](../notebooks/08_full_development_audit.ipynb) |

## Stages 09–16: grouped evaluation and learning-data analysis

| Stage | Historical stage protocol | Executed notebook |
| --- | --- | --- |
| 09 | [Protocol](protocols/FULL_DEVELOPMENT_CV_PROTOCOL.md) | [Notebook](../notebooks/09_full_development_cross_validation.ipynb) |
| 10 | [Protocol](protocols/TEMPORAL_CNN_PROTOCOL.md) | [Notebook](../notebooks/10_pytorch_temporal_cnn.ipynb) |
| 11 | [Protocol](protocols/MATCHED_MODEL_COMPARISON_PROTOCOL.md) | [Notebook](../notebooks/11_matched_cnn_random_forest.ipynb) |
| 12 | [Protocol](protocols/CLASS_WEIGHTED_CNN_PROTOCOL.md) | [Notebook](../notebooks/12_class_weighted_temporal_cnn.ipynb) |
| 13 | [Protocol](protocols/MATCHED_GROUP_CV_PROTOCOL.md) | [Notebook](../notebooks/13_matched_grouped_model_comparison.ipynb) |
| 14 | [Protocol](protocols/SAMPLING_SENSITIVITY_PROTOCOL.md) | [Notebook](../notebooks/14_sampling_sensitivity_audit.ipynb) |
| 15 | [Protocol](protocols/NESTED_TRAINING_SCALE_PROTOCOL.md) | [Notebook](../notebooks/15_nested_training_data_scaling.ipynb) |
| 16 | [Protocol](protocols/NESTED_GROUP_CV_PROTOCOL.md) | [Notebook](../notebooks/16_nested_training_group_cv.ipynb) |

## Stages 17–24: training budgets, confidence, engineering and robustness

| Stage | Historical stage protocol | Executed notebook |
| --- | --- | --- |
| 17 | [Protocol](protocols/FIXED_STEP_TRAINING_PROTOCOL.md) | [Notebook](../notebooks/17_fixed_step_training_budget.ipynb) |
| 18 | [Protocol](protocols/CONFIDENCE_AUDIT_PROTOCOL.md) | [Notebook](../notebooks/18_oof_confidence_audit.ipynb) |
| 19 | [Protocol](protocols/TEMPERATURE_CALIBRATION_PROTOCOL.md) | [Notebook](../notebooks/19_subject_disjoint_temperature_calibration.ipynb) |
| 20 | [Protocol](protocols/STAGE_20_EVIDENCE_CONSOLIDATION_PROTOCOL.md) | [Notebook](../notebooks/20_research_evidence_consolidation.ipynb) |
| 21 | [Protocol](protocols/STAGE_21_DATA_QUALITY_PROTOCOL.md) | [Notebook](../notebooks/21_minority_class_and_data_quality.ipynb) |
| 22 | [Protocol](protocols/STAGE_22_ADVANCED_MODELS_PROTOCOL.md) | [Notebook](../notebooks/22_advanced_temporal_architectures.ipynb) |
| 23 | [Protocol](protocols/STAGE_23_UNIFIED_BENCHMARK_PROTOCOL.md) | [Notebook](../notebooks/23_unified_matched_model_benchmark.ipynb) |
| 24 | [Protocol](protocols/STAGE_24_ROBUSTNESS_PROTOCOL.md) | [Notebook](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) |

## Stages 25–27: protocol freeze, approved evaluation and evidence synthesis

| Stage | Historical stage protocol | Executed notebook |
| --- | --- | --- |
| 25 | [Protocol](protocols/STAGE_25_MODEL_SELECTION_PROTOCOL.md) | [Notebook](../notebooks/25_model_selection_and_protocol_freeze.ipynb) |
| 26 | [Protocol](protocols/STAGE_26_FINAL_INFERENCE_PROTOCOL.md) | [Notebook](../notebooks/26_final_development_training_and_inference.ipynb) |
| 27 | [Protocol](protocols/STAGE_27_FINAL_REPORT_PROTOCOL.md) | [Notebook](../notebooks/27_final_research_evidence_and_report.ipynb) |

## Current approval and execution record

| Record or stage | Current verified interpretation |
| --- | --- |
| [Original Stage 25 proposal](../evidence/stage_25_protocol_proposal.json) | Historical `PENDING_USER_REVIEW`, preserved unchanged |
| [Reconciled Stage 25 proposal](../evidence/stage_25_protocol_proposal_reconciled.json) | Historical `PENDING_USER_REVIEW`, with separately reconciled source-code fingerprints |
| [Qualified-Test approval](../evidence/stage_25_protocol_approved_qualified_test.json) | Separate `APPROVED` record for the original one-time same-study held-out evaluation |
| [Stage 26 notebook](../notebooks/26_final_development_training_and_inference.ipynb) | 13 executed code cells; qualified held-out gate `PASSED` |
| [Stage 27 notebook](../notebooks/27_final_research_evidence_and_report.ipynb) | 11 executed code cells, including one Colab setup cell; read-only synthesis gate `PASSED` |

**Do not** overwrite original or reconciled manifests, infer a new Test approval from a passed replay, or re-score protected Test data merely to review documentation. Neither the Development evaluation nor the qualified same-study held-out check establishes blind external generalisation, prospective validation or clinical readiness.

## Supporting technical records

- [Notebook Authoring Standards](NOTEBOOK_AUTHORING_STANDARDS.md): notebook presentation, scientific caveats and preservation of saved outputs.
- [Stage 26 Source-Code Reconciliation](STAGE_26_CODE_RECONCILIATION.md): why an initial source SHA256 guard blocked progress and how the later reconciled candidate was recorded. Earlier *pending* notices in this record are historical, not current blockers.
- [Version-controlled evidence](../evidence/): original evidence register, historical candidates and distinct approval record.
- [HARTH reproducibility and provenance guide](../../../docs/HARTH_REPRODUCIBILITY.md): pinned historical execution, checkout requirements, integrity checks and release readiness.
- [Primary project overview](../README.md): interpretation, actual metric tables, limitations and local execution instructions.

Historical records are preserved; this navigation update does not create new model outcomes, rerun a notebook or renew permission to access the qualified Test.
