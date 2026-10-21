# HARTH documentation and protocols

[Study overview and current results](../README.md) · [Executed notebooks](../notebooks/) · [Historical research records](../evidence/) · [Research reproducibility](../../../docs/HARTH_REPRODUCIBILITY.md)

This directory provides navigation for the original experimental protocols. **Historical stage-specific status statements are not the current study completion status.** The study README summarises the executed 27-stage result; the original protocol text is retained unmodified for provenance.

## Stage-specific protocols

| Stage | Protocol | Executed notebook |
| --- | --- | --- |
| 02 | [Protocol](protocols/WINDOWING_PROTOCOL.md) | [Notebook](../notebooks/02_signal_preprocessing_and_windowing.ipynb) |
| 03 | [Protocol](protocols/FEATURE_ENGINEERING_PROTOCOL.md) | [Notebook](../notebooks/03_statistical_feature_engineering.ipynb) |
| 04 | [Protocol](protocols/SUBJECT_SPLITTING_PROTOCOL.md) | [Notebook](../notebooks/04_subject_independent_splitting.ipynb) |
| 06 | [Protocol](protocols/CROSS_VALIDATION_PROTOCOL.md) | [Notebook](../notebooks/06_subject_level_cross_validation.ipynb) |
| 07 | [Protocol](protocols/EXPANDED_DEVELOPMENT_PROTOCOL.md) | [Notebook](../notebooks/07_expanded_subject_evaluation.ipynb) |
| 08 | [Protocol](protocols/FULL_DEVELOPMENT_AUDIT_PROTOCOL.md) | [Notebook](../notebooks/08_full_development_audit.ipynb) |
| 09 | [Protocol](protocols/FULL_DEVELOPMENT_CV_PROTOCOL.md) | [Notebook](../notebooks/09_full_development_cross_validation.ipynb) |
| 10 | [Protocol](protocols/TEMPORAL_CNN_PROTOCOL.md) | [Notebook](../notebooks/10_pytorch_temporal_cnn.ipynb) |
| 11 | [Protocol](protocols/MATCHED_MODEL_COMPARISON_PROTOCOL.md) | [Notebook](../notebooks/11_matched_cnn_random_forest.ipynb) |
| 12 | [Protocol](protocols/CLASS_WEIGHTED_CNN_PROTOCOL.md) | [Notebook](../notebooks/12_class_weighted_temporal_cnn.ipynb) |
| 13 | [Protocol](protocols/MATCHED_GROUP_CV_PROTOCOL.md) | [Notebook](../notebooks/13_matched_grouped_model_comparison.ipynb) |
| 14 | [Protocol](protocols/SAMPLING_SENSITIVITY_PROTOCOL.md) | [Notebook](../notebooks/14_sampling_sensitivity_audit.ipynb) |
| 15 | [Protocol](protocols/NESTED_TRAINING_SCALE_PROTOCOL.md) | [Notebook](../notebooks/15_nested_training_data_scaling.ipynb) |
| 16 | [Protocol](protocols/NESTED_GROUP_CV_PROTOCOL.md) | [Notebook](../notebooks/16_nested_training_group_cv.ipynb) |
| 17 | [Protocol](protocols/FIXED_STEP_TRAINING_PROTOCOL.md) | [Notebook](../notebooks/17_fixed_step_training_budget.ipynb) |
| 18 | [Protocol](protocols/CONFIDENCE_AUDIT_PROTOCOL.md) | [Notebook](../notebooks/18_oof_confidence_audit.ipynb) |
| 19 | [Protocol](protocols/TEMPERATURE_CALIBRATION_PROTOCOL.md) | [Notebook](../notebooks/19_subject_disjoint_temperature_calibration.ipynb) |
| 20 | [Protocol](protocols/STAGE_20_EVIDENCE_CONSOLIDATION_PROTOCOL.md) | [Notebook](../notebooks/20_research_evidence_consolidation.ipynb) |
| 21 | [Protocol](protocols/STAGE_21_DATA_QUALITY_PROTOCOL.md) | [Notebook](../notebooks/21_minority_class_and_data_quality.ipynb) |
| 22 | [Protocol](protocols/STAGE_22_ADVANCED_MODELS_PROTOCOL.md) | [Notebook](../notebooks/22_advanced_temporal_architectures.ipynb) |
| 23 | [Protocol](protocols/STAGE_23_UNIFIED_BENCHMARK_PROTOCOL.md) | [Notebook](../notebooks/23_unified_matched_model_benchmark.ipynb) |
| 24 | [Protocol](protocols/STAGE_24_ROBUSTNESS_PROTOCOL.md) | [Notebook](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) |
| 25 | [Protocol](protocols/STAGE_25_MODEL_SELECTION_PROTOCOL.md) | [Notebook](../notebooks/25_model_selection_and_protocol_freeze.ipynb) |
| 26 | [Protocol](protocols/STAGE_26_FINAL_INFERENCE_PROTOCOL.md) | [Notebook](../notebooks/26_final_development_training_and_inference.ipynb) |
| 27 | [Protocol](protocols/STAGE_27_FINAL_REPORT_PROTOCOL.md) | [Notebook](../notebooks/27_final_research_evidence_and_report.ipynb) |

## Additional historical and development documentation

- [Notebook Authoring Standards](NOTEBOOK_AUTHORING_STANDARDS.md): formatting, figures, execution provenance and source integrity.
- [Stage 26 Code Reconciliation](STAGE_26_CODE_RECONCILIATION.md): historical SHA256 mismatch, reconciliation and safeguards; do not interpret its initial pending status as the current state.
- [Versioned evidence records](../evidence/): preserved source manifests and stage evidence register.

Moving these historical documents does not constitute a rerun, new model result, new validation or update to approved research evidence.

**Stage 24 historical evidence:** The original executed Git revision includes the Colab setup and all 36 scored Development jobs. The current 12-cell scientific notebook has saved results; 18 jobs were restored from the matching local cache. Approval hashes, scientific source code and reported scores are checked against the immutable historical records. See the [reproducibility guide](../../../docs/HARTH_REPRODUCIBILITY.md).
