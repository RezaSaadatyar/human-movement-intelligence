# Stage 26B | Qualified Held-Out Evaluation Preflight (Approval Pending)

## Current verified boundary (15 October 2026)
- The executed Notebook 26 reports **PASSED** for Development training/inference: 29 people, 14,201 6-axis windows, one weighted-CE 1D-ResNet (400 optimiser updates) and 120-tree Random Forest reference.
- The notebook states the original reserved participants `S023` and `S032` were **not accessed in Stage 26**.
- The reconciled Stage 25 manifest still states `PENDING_USER_REVIEW`. **No Test loading, predictions or scores are authorised.**
- The saved model files (`resnet_state_dict.pt`, `random_forest.joblib`, `inference_config.json`) are ignored by Git and were produced under `/content/human-movement-intelligence/outputs/harth_stage26_final/`. The executed notebook shows their creation and a successful local reload, **not their availability in any future Colab session**.

## Exact unchanged scientific protocol proposed for user review

| Item | Frozen setting |
| --- | --- |
| Dataset | Pinned HARTH v2.0 revision `dad2cfbe89a26f72f19770419469ac037de200df` |
| Sensor order | `back_x, back_y, back_z, thigh_x, thigh_y, thigh_z` |
| Temporal processing | 50 Hz, 100 samples per window, stride 50, time-continuous, label-purity 1.0 |
| Source labels | Codes 1 through 8; undocumented labels excluded only after source-safe windowing |
| Development | 29 distinct persons, deterministic cap 500 per person, 14,201 windows |
| Test | `S023` and `S032`, maximum 500 eligible windows each, **one-time qualified evaluation only** |
| Primary | 1D-ResNet, 6 channels, hidden channels 32, inverse-sqrt weighted cross-entropy, Adam, 0.001, 400 updates, batch 128, seed 42, final-step checkpoint |
| Reference | Random Forest, 120 trees, minimum leaf 3, balanced_subsample, seed 42, 30 fixed features |
| Scaling | Development-only six-axis channel statistics; **never** fit on Test |
| Primary metric | Fixed-eight-class Macro F1 |
| Additional | Accuracy, balanced accuracy, per-class support/precision/recall/F1, per-person scores, confusion matrices |
| No Test adaptation | No refit, tuning, threshold selection, calibration, additional seed search, or post-hoc model replacement |

**Interpretation constraint:** These two people were descriptively inspected in early exploratory stages, so this is a **qualified same-study held-out evaluation**, **not** a fully blind, pre-registered or external validation.

## Non-destructive runtime preflight before any later approval

1. Confirm the existing Colab `outputs/harth_stage26_final/` directory exists. Do **not** assume Git pull restores it.
2. Verify that `resnet_state_dict.pt`, `random_forest.joblib` and `inference_config.json` exist. Compare each exact file SHA256 to the three values recorded in the executed Notebook 26 (the notebook displays SHA256 values).
3. Verify the checked-out Git commit, the six reconciled source-module fingerprints, and unchanged manifest scientific settings. Keep the original protocol and reconciled proposal unchanged.
4. Reload **only locally trusted** artefacts via the existing `load_final_models` function. Ensure no Test CSV is read, downloaded or inspected during this preflight.
5. If Colab artefacts have been lost, recover the verified files from trusted storage or **rerun the same frozen Development-only Stage 26 fit**; never retrain using Test results.
6. Stop at the approval gate. A passing CI job is not permission to unlock Test.

## User decision and implementation boundary

**Status: WAITING FOR EXPLICIT USER APPROVAL.** The existing code requires both (a) a reviewed manifest status `APPROVED` and (b) the exact explicit confirmation `I_APPROVE_QUALIFIED_TEST_EVALUATION` immediately before source access. Neither may be self-granted by a code change or CI run.

Only after the user specifically approves the above unchanged protocol, prepare a separate numbered held-out evaluation notebook with clear source identifiers, actual supports, reproducible scores for both preselected models, true class-level and per-person tables, figures, non-selected diagnostic findings, and a strict read-only Quality Gate. Do not revise frozen model/hyperparameters in response to Test outputs. Preserve Stage 25 original source and reconciled manifests for provenance.

## Reporting boundaries

The training loss in Notebook 26 is an in-sample Development optimisation trace, not a Test metric. The successful 24-window round-trip confirms inference compatibility on Development data only. GitHub Actions checks synthetic/offline tests and earlier small notebooks; its success does not demonstrate that Notebook 26 has run end-to-end in CI on GPU.
