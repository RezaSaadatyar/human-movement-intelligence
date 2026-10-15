# Stage 26 | Final Development Training and Qualified-Test Approval Gate

## Purpose and verified upstream evidence

The executed Stage 25 notebook has all eleven code cells executed and its final research gate PASSED. The protocol JSON is still **PENDING_USER_REVIEW**, and that status must not be misrepresented as an authorised Test evaluation.

**This initial Stage 26 implementation trains the frozen final models on Development only and verifies reusable inference. It does not run reserved-Test evaluation without a separate explicit approval.** This preserves the scientific boundary agreed in Stage 25.

## Frozen research configuration

- Original pinned HARTH v2.0 recordings and the six established sensor axes (`back_x, back_y, back_z, thigh_x, thigh_y, thigh_z`); 50 Hz nominal, 100 samples (2 seconds) per window, step 50, perfect label purity and no abnormal timestamp gaps.
- Valid target original source label codes 1–8 only. Unmapped source codes remain untouched in raw data and are excluded only after source-safe temporal segmentation.
- Exactly **29 Development participants**, chronological label-independent cap 500 per person, yielding 14,201 source windows, verified against the Stage 25 source fingerprint.
- **Main model:** 1D-ResNet (6 input channels, 8 classes, hidden width 32); Development-only per-axis standardisation and Development-only inverse-square-root class weights; seed 42; 400 Adam updates, batch size 128, learning rate 0.001, last update retained.
- **Classical reference:** 120-tree Random Forest, `balanced_subsample` weighting, min leaf size 3, seed 42 and five deterministic statistical features per sensor (30 features).
- No calibration, checkpoint choice, threshold tuning, class relabelling or architecture revision based on Test data.

## Source modules, notebook and tests

- `code/python/src/core/harth_final_inference.py`: robust code-hash and protocol validation, Train-only six-axis scaler, one frozen Development ResNet fit, trusted local `state_dict` and `joblib` saving/loading, source-window inference, Test-access gate and qualified metric utilities.
- `tests/test_harth_final_inference.py`: synthetic tests for Development-only statistics, required explicit Test authorisation, rejection of pending protocol and reserved-participant checks.
- `notebooks/26_final_development_training_and_inference.ipynb`: ten comprehensive, technically explained sections with **bold-only Markdown section headings**, numbered code comments, actual GPU Development fitting, saved artefact checks and an evidence-bounded final quality gate.

## Current execution path: no Test access

On Google Colab, enable Runtime GPU and use the latest Repository code:

```bash
cd /content/human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_harth_final_inference.py" -v
```

Open `notebooks/26_final_development_training_and_inference.ipynb` and execute all cells; **it does not contain a cell that loads or scores S023/S032**. It trains two Development models, saves the weights/forest plus original activity and sensor metadata in ignored `outputs/harth_stage26_final/`, reloads them and tests inference on 24 Development windows. The output files in Colab may disappear with Runtime reset and are not currently committed to Git. Retain these artefacts securely for the later approved report or rerun the frozen fit with the same seeds.

A returned gate `PASSED` establishes successful Development fitting/reload and source alignment, **not** independent generalisation or an approved final Test report.

## Separate qualified Test evaluation: not yet authorised

Stage 25 explicitly requested user review of final model, RF reference, 500-cap sampling policy on **both S023 and S032**, and the candid limitation that these two participants were descriptively inspected during early exploratory work; thus they are not strictly unseen/preregistered or external data. Before adding/executing a Test-loading cell, obtain the user's specific confirmation of **this exact protocol**. Then explicitly set the reviewed manifest to `APPROVED` and have a separate exact confirmation at the actual Test gate. The code contains `load_final_protocol(..., require_approval=True)` and `enforce_qualified_test_approval(...)` safeguards, but the Test-load/evaluation workflow remains unimplemented until approval. No Test scores, predictions or claims have been created.

After approval, report original source Test support, full eight-class Macro F1, per-class precision/recall/F1 and true support, balanced accuracy, overall accuracy, separate subject-level scores and confusion matrices for the frozen ResNet and RF, without using Test for training or choices. Report this as **qualified same-study held-out evaluation**, not blind external validation.

## Remaining Stage 26 completion boundary

- [x] Source-grounded Stage 25 evidence and proposed protocol read.
- [x] Frozen Development model training, trusted artefact, inference and approval-gate code created.
- [x] Full 10-section notebook, tests and reproducibility documentation created.
- [ ] User runs notebook on Colab GPU, confirms real output and saved files, and pushes executed notebook.
- [ ] User explicitly approves exact qualified-Test protocol, or requests changes before Test access.
- [ ] With approval, implement and execute the separate, one-time source-safe qualified Test report and inspect its actual results before Stage 27.
