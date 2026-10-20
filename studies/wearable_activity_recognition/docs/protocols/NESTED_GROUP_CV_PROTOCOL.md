# Stage 16: grouped validation of nested training-set expansion

## Motivation
Stage 15 increased Train from 10,000 to 20,000 windows for a fixed six-person development Validation cohort. RF Macro F1 increased from 0.8266 to 0.8644. Weighted CNN last-epoch Macro F1 changed from 0.7968 to 0.7805; its expanded-data fourth epoch scored 0.8428, highlighting epoch sensitivity. Neither observation establishes generalisation.

## Method
- Official pinned HARTH v2.0, 29 development participants, original Test IDs S023 and S032 excluded before source loading.
- Same annotation-pure 100-point windows, stride 50 and timestamp-gap rejection, with eight documented activity labels. All other labels excluded after segmentation.
- For each person, original 500-cap windows are retained exactly. Extra label-blind chronological positions extend Train to 1,000 per person, preserving inclusion of all original windows.
- GroupKFold partitions **only original cap-500 windows and their subject IDs** into three held-out folds. Each participant's original base windows are predicted exactly once by each model.
- Each held-out fold has four conditions: RF 500, RF 1000, Weighted CNN 500, Weighted CNN 1000.
- Within a fold the **same base Validation source-window indices** and people are used in all four conditions. The expanded Train contains only additional windows belonging to Train participants, never Validation participants.
- RF uses 120 trees, minimum leaf size 3 and balanced-subset class weights, with original five features per six IMU channels.
- CNN uses the same small 1D architecture, inverse-square-root loss weights fitted only on the current Train, channel normalisation using Train only, five epochs, batch size 128 and Adam at 1e-3. Fresh models and seeds for each condition/fold.
- Evaluate fixed-eight-class Macro F1 and Balanced Accuracy per fold, pooled out-of-fold class F1, and equal-person paired Macro F1 at different training sizes.

## Interpretation
This is a 29-person **development-only** sensitivity analysis. The original final Test people were examined in early exploratory stages; they remain reserved from subsequent fitting and selection, rather than representing strict preregistered unseen Test participants. Cross-Validation folds are not independent trials. Fixed five-epoch training means the expanded datasets also receive more optimiser steps, so score changes cannot be attributed to dataset size alone. Do not select the most favourable epoch and present that score as independent validation. No p-values, Test scores or external validity claims.

## Run in Cursor
```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_training_scale_cv.py" -v
```
Open `studies/wearable_activity_recognition/notebooks/16_nested_training_group_cv.ipynb`, execute the cells sequentially and save/push the real outputs. The three folds fit **six Weighted CNN jobs**, plus six Random Forest estimators, so CPU execution can be lengthy. The large downloaded CSVs remain only in the local ignored cache.
