# Stage 19: subject-disjoint temperature scaling

## Background
The executed Stage 18 notebook reports out-of-fold Weighted CNN 1000 accuracy 0.9458, ten-bin ECE 0.0453, multiclass Brier 0.0905 and 40 erroneous predictions with softmax >=0.90 on 14,201 development validation windows. Shuffling recall is 0.1233. These observations alone do **not** establish good calibration or a validated confidence measure.

## Scientific question
Can calibration-only scalar temperature scaling improve class probability quality on **separate held-out participant groups**, as measured by NLL, Brier score and ten-bin ECE, without changing the predicted class?

## Protocol
- Pinned HARTH v2.0 source; eight documented class labels; 29 development people. Protected original Test subjects S023 and S032 excluded.
- Nested base-500 / expanded-1000 chronological, label-blind windows, strict label purity and timestamp-gap checks. GroupKFold with 3 outer folds based on base-window IDs.
- Inside each outer fold, deterministic GroupShuffleSplit among **outer Train subjects**, allocating 20% of those people to Calibration. All their base-500 windows are used for calibration; their expanded rows are *never* used for CNN Train.
- CNN Train consists only of expanded eligible windows from inner Train subjects. Fold Calibration and outer Validation participants are entirely distinct from Train and one another.
- Fit one new Weighted CNN per outer fold, 400 Adam steps, batch size 128, learning rate 0.001, seed 42, inverse-square-root class-loss weights and channel scaling from Train only.
- Predict Calibration softmax scores from the fitted network. Fit a single positive scalar temperature in [0.25,4.0] by minimising calibration-only multiclass NLL.
- Apply the fitted scalar **without further fitting** to the outer fold's original base-500 Validation windows. Report uncalibrated/calibrated outer-validation NLL, Brier, ten-bin ECE, accuracy and reliability plots.
- Verify one Validation prediction per base window per method, all subject roles disjoint, and identical top-1 predictions after temperature scaling.

## Interpretive restrictions
This stage is exploratory development within previously inspected subjects, not independent final Test or verified clinical uncertainty. Softmax is not automatically calibrated. Temperature scaling should preserve predictions, so it does not fix low Shuffling recall directly. ECE depends on bins and windows overlap within subjects; no independence-based confidence intervals or clinical thresholds are claimed. Loss weighting and changed inner Train memberships mean Stage 19 raw accuracy need not reproduce Stage 18.

## Run in Cursor
```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_temperature_calibration.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/19_subject_disjoint_temperature_calibration.ipynb` and execute all cells. Three CNN training runs of 400 steps each are required on the local CPU. Commit the notebook with its actual outputs and review fold results before making any calibration claim.
