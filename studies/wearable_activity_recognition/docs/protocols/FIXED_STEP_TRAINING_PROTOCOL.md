# Stage 17: fixed-optimiser-step weighted CNN evaluation

## Motivation
Notebook 16 compared nested HARTH Train caps of 500 and 1,000 windows/person across the same three group-held-out development folds, with original Test subjects S023 and S032 excluded. Mean validation Macro F1 was Random Forest 0.8076/0.8097 and Weighted CNN 0.7834/0.7962. Shuffling out-of-fold F1 improved for Weighted CNN from 0.1345 to 0.2513 when increasing the Train cap.

**Important confounder:** Stage 16 used five epochs in both cases. The larger Train had more mini-batches and therefore more optimiser updates. Stage 17 controls the **number of updates**, not the number of distinct subject-window presentations.

## Fixed protocol

- Same official HARTH v2.0 29-person development cohort, eight documented activity codes 1–8, 100-sample/50-stride purity-and-timestamp-safe windows, label-independent nested Train caps.
- Protected original Test identities S023 and S032 are not loaded. They were inspected in earlier exploratory stages and should not be described as completely unseen in a preregistration sense.
- GroupKFold with three folds on original 500-cap subject-window identities, identical to Stage 16.
- For each fold: train the existing Weighted TemporalCNN from scratch using base 500-cap or nested 1,000-cap Train windows. **Identical original base Validation people and rows** are used for both runs. Expanded Train contains no Validation people.
- Same architecture, Adam with learning rate 0.001, batch size 128, train-only channel scaling and train-only inverse-square-root loss weighting. Both configurations start with the same seed.
- Exactly **400 optimiser updates per condition**; descriptive validation checks at predeclared steps 100, 200, 300, 400. The main comparison uses **step 400**, not the best-looking intermediate validation result.
- Six CNN fits across three folds, no new Random Forest fitting. RF results from earlier stages remain context and are not presented as matched fixed-update results.
- Report fold-wise and participant-wise eight-class Macro F1, Balanced Accuracy, learning trajectories and separate Shuffling/stair diagnostics.

## Limitations

Equal optimiser updates and nominal batch size do not equalise the number of passes over each individual window. Different training sample sets alter class counts, fitted normalisation statistics and training class weights. There is no independent causal attribution or prospective/clinical/generalisation claim. Validation subjects were inspected in prior development; results are exploratory.

## Running in Cursor

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_fixed_steps_temporal_cnn.py" -v
```

Run `studies/wearable_activity_recognition/notebooks/17_fixed_step_training_budget.ipynb` sequentially and commit/push the executed notebook. CPU execution is substantial: six 400-step CNN jobs. Real results are not known until the notebook is executed and reviewed.
