# Stage 15: nested development training-data scaling

## Research question
Does adding more genuine, documented HARTH IMU windows to training improve Random Forest and Weighted CNN performance on a **fixed development Validation set**?

Stage 14 demonstrated that a 500-window cap preserved just 219 of 1,946 eligible Shuffling windows (11.25%) across 29 development subjects. A separate 1,000-window chronological sample preserved 386 (19.84%). Those source retention figures do not establish predictive benefit.

## Strict experiment design
- Source: official pinned HARTH v2.0, 29 development subjects. Original Test IDs S023 and S032 are never loaded or used.
- Each recording is segmented independently using 100-point windows, 50-point stride, exact label purity and timestamp continuity; labels outside the eight-class documented task are excluded only after segmentation.
- The original **500-cap source-window positions are reproduced exactly**.
- An expanded 1,000-cap sample **contains all 500 original positions**, plus new source windows selected without using labels. The expanded set therefore differs from an independently spaced cap-1,000 sample in Stage 14. Counts should not be conflated.
- Recreate the original fixed subject-disjoint split from the 500-cap sample (seed 42): 20 training people, 6 validation people, 3 unused development people.
- Train is expanded from the 500-cap to the nested 1,000-cap sample **only for the 20 training people**. The original 2,701 Validation windows from the same 6 people remain completely identical.
- Refit Random Forest (120 trees, min_samples_leaf=3, balanced-subset weights) on the handcrafted features derived from the same windows.
- Refit Weighted CNN with the established architecture, five epochs, train-only channel statistics and train-only inverse-square-root class weights on each condition. Identical seed, optimisation settings and validation windows.
- Report Macro F1, Balanced Accuracy, per-class F1/support, and equal-person paired differences between caps. Particularly inspect Shuffling and stair classes.

## Interpretation limits
This is a **development-only single-split learning-data ablation** on people whose Validation labels have been inspected repeatedly in earlier stages. A change in score is not independent evidence of generalisation or proof that added data alone caused improvement. Training steps increase as training data increase under a fixed epoch budget. Grouped validation and additional run seeds would be appropriate subsequent checks. The original Test subjects were explored earlier, so they should be described as reserved from development, not strictly preregistered-unseen.

## Run locally in Cursor
```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_sampling_audit.py" -v
python -m unittest discover -s tests -p "test_harth_cohort.py" -v
```
Open `studies/wearable_activity_recognition/notebooks/15_nested_training_data_scaling.ipynb`, run all cells, verify the quality gate, and commit with the real saved outputs. CPU time may be significant because two CNN fits and two forests are trained. No scores are claimed before maintainer execution.
