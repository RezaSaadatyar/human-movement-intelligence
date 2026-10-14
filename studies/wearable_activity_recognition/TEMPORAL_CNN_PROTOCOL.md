# Stage 10: raw-signal temporal CNN (PyTorch)

## Goal

Establish a readable and reproducible **raw IMU -> PyTorch 1D-CNN -> validation metrics** pipeline that can later be extended to more sophisticated temporal architectures. The Stage 9 baseline uses 30 hand-engineered statistical features; the first CNN accepts **six original accelerometer channels by 100 time points** instead.

This is the first engineering baseline, **not** a matched comparison with Stage 9 Random Forest or a full-HARTH benchmark.

## Reusable code

- `code/python/src/core/temporal_cnn.py`
  - `TemporalCNN`: two temporal convolution layers, BatchNorm/ReLU, global average pooling, linear classifier.
  - `standardise_imu_train_only`: compute each channel mean/SD solely from Train windows, then apply unchanged to Validation.
  - `train_temporal_cnn`: deterministic seed, fresh model/optimiser, DataLoader, per-epoch training loss and validation Macro F1, final-epoch prediction.
- `studies/wearable_activity_recognition/src/harth_data.py`
  - `prepare_harth_window_cohort`: source-specific full-recording windowing, excluded-label audit and bounded, deterministic window selection.
- `tests/test_temporal_cnn.py`: tests for tensor shape, leakage-safe scaling, tiny training and invalid labels.
- `studies/wearable_activity_recognition/notebooks/10_pytorch_temporal_cnn.ipynb`: documented, commented and interactive real-data training experiment.

## Fixed experiment scope

- All **29 development subjects**, with original reserved Test IDs S023 and S032 excluded entirely.
- Source HARTH v2.0 50 Hz, six accelerometer axes, 100-point windows and 50-point stride, strict label purity and timestamp continuity.
- **At most 500 eligible windows per participant**, selected evenly across each recording **without using labels to determine which windows to sample**. This deliberately changes the original sample distribution.
- Eight documented activity codes 1–8; all other source labels excluded **after** segmentation, with no reinterpretation.
- Group-disjoint development split: 70% Train, 20% Validation, 10% separately unused development groups, seed 42. The function allocates integer subject counts and does not guarantee class stratification.
- 1D-CNN, 5 epochs, batch size 128, Adam learning rate 0.001. These are **pilot settings**, not tuned results.
- Only Train windows fit channel normalisation and CNN parameters. Validation is descriptive; no Test inference, no model selection using the reserved Test.
- CPU runs are supported. PyTorch is an **optional** installation because not every classical-ML user needs it.

## Running

From the project root in Cursor (PowerShell):

```powershell
git pull origin main
python -m pip install -r requirements-dl.txt
```

Then open `notebooks/10_pytorch_temporal_cnn.ipynb` and execute cells in order. If the notebook kernel uses a different environment than the terminal, select the same Python interpreter in Cursor first.

The first model run may take several minutes or longer on CPU. All 29 development CSVs should already be locally available from Notebook 08. Data files stay in ignored `data/raw/harth_v2/`.

## Interpretation and next work

The retained raw windows overlap in time, and individuals may have different numbers of activity classes. The bounded window cap and single development split **do not match** Stage 9's full-cohort 5-fold evaluation protocol. Therefore differences in Macro F1 between Notebook 10 and Notebook 09 must not be described as an improvement or decline caused by deep learning.

After inspecting actual Notebook 10 results, the next justified investigation is an equivalently sampled, subject-disjoint comparison of CNN and classical baselines, or full-development grouped CV for the CNN if computationally feasible. The reserved Test remains excluded during all model-family development.
