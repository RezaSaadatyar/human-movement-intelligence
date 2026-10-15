# Stage 18: Out-of-fold prediction confidence audit

## Research question
The Stage 17 fixed-update study gave Weighted CNN 500/1000 mean held-out-fold Macro F1 0.7746/0.7927 on the 29-person HARTH development cohort; larger Train improved all three Fold scores and 18/29 participant-level scores. However, Shuffling recall remained weak. Stage 18 asks whether the **existing Weighted CNN 1000**, trained for 400 updates per fold, emits high softmax scores on erroneous predictions, especially for difficult activities.

## Fixed methodology
- HARTH v2.0, 29 development subjects only. Original Test IDs S023 and S032 excluded *before loading*. They were inspected in earlier exploration, so cannot be described as strictly blind Test.
- Same nested 500/1000 source-window preparation and GroupKFold three-fold assignment as Stage 17.
- Validate on original 500-cap windows; expanded Train uses 1000-cap windows belonging only to each fold's Train participants.
- Fixed existing two-layer TemporalCNN, Train-only per-channel scaling and inverse-square-root class-loss weights, Adam 1e-3, batch 128, seed 42 and **400 optimizer updates**.
- For each fold, the final trained network is evaluated on held-out participants. Softmax is applied to logits using the **same Train-fitted means and SD**; probabilities correspond to original activity codes 1–8.
- Check each source window has precisely one out-of-fold prediction and its original class label.
- Report sample-level Brier score, 10-bin ECE, reliability diagram, prediction confidence/entropy/margin, error confidence, activity-class recall and high-confidence (>0.90) errors, and participant-level diagnostics.
- **No parameter fitting based on Validation/Test probabilities**, threshold selection, temperature scaling or refusal/deployment logic.

## Limitations
Softmax scores should not be called reliable uncertainty estimates or probabilities of being clinically correct without further calibration and testing. Correlated overlapping time-series windows mean window-level ECE and Brier are descriptive, with no independent confidence interval attached. Bin choice affects ECE. Small-class counts and individual variations must be shown. This does not prove faithful reasoning, reliability in deployment, or improved predictions.

## Running
```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_prediction_reliability.py" -v
```
Open `studies/wearable_activity_recognition/notebooks/18_oof_confidence_audit.ipynb` and execute all cells. This includes three independent 400-step CNN training runs, so local CPU time is material. Commit and push the executed notebook. Results must be reviewed before any model confidence claims.
