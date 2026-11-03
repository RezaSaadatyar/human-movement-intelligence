# Biomechanical Load Estimation | Research Workflow Figures

This directory mirrors the **study-level and notebook-level figure organisation** of the completed HARTH Wearable Activity Recognition project. The current original-source Essex research pathway uses **seven lower-body IMUs**, original 101×42 IMU inputs and eleven source-verified FE positions for two unitless FE-model targets.

## Study-wide scientific figures

| File | Purpose |
| --- | --- |
| [\`study_overview.svg\`](study_overview.svg) | Complete eight-stage original Essex source–model–Test workflow, with original participant splits and actual final results |
| [\`final_workflow.svg\`](final_workflow.svg) | Read-only evidence closure: source lineage, frozen Random Forest primary, TCN secondary, one authorised Test, uncertainty and final PASSED gate |
| [\`bonenet_derivative_pilot.svg\`](bonenet_derivative_pilot.svg) | **Historical** 444-row BoneNet public derivative pilot; row-to-subject linkage remained unverified and is **not** the original Essex model study |

## Individual stage workflows

| Source notebook | Detailed research figure | Other stage view |
| --- | --- | --- |
| [01 — Original source](../code/notebooks/01_original_dataset_audit_and_trial_registry.ipynb) | [\`01_consolidated_workflow.svg\`](01_consolidated_workflow.svg) | [\`01_workflow.svg\`](01_workflow.svg) |
| [02 — MATLAB decoding](../code/notebooks/02_matlab_object_decoding_and_numeric_recovery.ipynb) | [\`02_consolidated_workflow.svg\`](02_consolidated_workflow.svg) | [\`02_workflow.svg\`](02_workflow.svg) |
| [03 — Native temporal recovery](../code/notebooks/03_full_cohort_numeric_and_temporal_validation.ipynb) | [\`03_consolidated_workflow.svg\`](03_consolidated_workflow.svg) | [\`03_workflow.svg\`](03_workflow.svg) |
| [04 — Governed splits](../code/notebooks/04_governed_imu_fe_dataset_and_participant_splits.ipynb) | [\`04_consolidated_workflow.svg\`](04_consolidated_workflow.svg) | [\`04_workflow.svg\`](04_workflow.svg) |
| [05 — Classical regression](../code/notebooks/05_classical_regression_baselines_and_participant_independent_validation.ipynb) | [\`05_consolidated_workflow.svg\`](05_consolidated_workflow.svg) | [\`05_workflow.svg\`](05_workflow.svg) |
| [06 — Temporal regression](../code/notebooks/06_deep_temporal_regression_and_peak_prediction.ipynb) | [\`06_consolidated_workflow.svg\`](06_consolidated_workflow.svg) | [\`06_workflow.svg\`](06_workflow.svg) |
| [07 — GRU/BiGRU robustness](../code/notebooks/07_multi_seed_robustness_gru_bigru_and_peak_ablation.ipynb) | [\`07_consolidated_workflow.svg\`](07_consolidated_workflow.svg) | [\`07_workflow.svg\`](07_workflow.svg) |
| [08 — Frozen final Test](../code/notebooks/08_frozen_final_model_evaluation_and_research_synthesis.ipynb) | [\`08_consolidated_workflow.svg\`](08_consolidated_workflow.svg) | Not a second independent Test figure; the consolidated workflow is the source-grounded stage view |

**Original and derived data must not be conflated.** The historical pilot diagram represents a different derivative data linkage that has not been verified against the original source records. Current study-level diagrams reflect the original-source Essex final results. The published FE targets are **unitless model-derived estimates**, and the Test comprises only four unseen participants from the same Essex study.

See the [project README](../README.md), [full methods report](../BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) and [repository visual standards](../../../docs/RESEARCH_VISUAL_STANDARDS.md).
