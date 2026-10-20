# Stage 12: training-only class-weighted Cross-Entropy ablation

## Research question

Does inverse-square-root weighting of the **Train** class counts improve rare activity recognition in the established HARTH 1D-CNN, compared with standard unweighted Cross-Entropy? Notebook 11 found CNN macro F1 0.6659 versus Random Forest 0.8266 on the same Validation subjects/windows, with CNN F1=0 for Shuffling. This motivates a controlled loss-weighting pilot but does not establish that imbalance is the only explanation.

## Strict controls

- HARTH v2.0, 29 development subjects, original protected Test S023/S032 excluded.
- Same capped raw IMU windows as Notebooks 10/11: maximum 500 per subject, 100 samples x 6 channels, strict single-label and timestamp-safe windows.
- Same deterministic subject-disjoint 70/20/10 allocation (seed 42), with the 10% remainder unused, and the same Validation rows in each run.
- Same TemporalCNN architecture, 5 epochs, Adam, learning rate 0.001, batch size 128, random seed 42 and Train-only channel normalisation.
- Only change: CrossEntropyLoss weights. Standard CNN uses ones; Weighted CNN uses `1/sqrt(training_class_count)` for each documented code, normalised to mean 1. No Validation or Test counts determine weights.
- Metrics: Validation Macro F1 and Balanced Accuracy, per-class precision/recall/F1, both confusion matrices, loss curves and equal-weight per-subject paired Macro F1.

## Reusable functions

- `code/python/src/core/temporal_cnn.py`: `compute_training_class_weights()` and optional `class_weighting` parameter of `train_temporal_cnn()`.
- `tests/test_weighted_temporal_cnn.py`: checks weight ordering, missing classes and tiny deterministic training.
- `notebooks/12_class_weighted_temporal_cnn.ipynb`: documented real-data comparison.

## Scientific restrictions

- Weighting is a heuristic derived from *window counts*, not clinical/patient prevalence.
- The two training objectives produce different loss scales; compare Validation metrics rather than interpreting the two raw training losses as directly equivalent.
- The six Validation subjects were already inspected during prior model development. Results are exploratory, not independent confirmation.
- The class-imbalanced bounded set is not interchangeable with Stage 9 full-data grouped CV. No p-values, Test scores, clinical conclusions or claims of improvement before executing the notebook.
- The Test participants were inspected in earlier exploratory stages and should be described as reserved from later model development, not strictly unseen.

## Run locally

From the project root:

```powershell
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_weighted_temporal_cnn.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/12_class_weighted_temporal_cnn.ipynb` and execute all cells. It fits both variants on CPU when no CUDA accelerator exists, so allow more time than Notebook 10. Commit and push executed outputs only after reviewing the quality gate.
