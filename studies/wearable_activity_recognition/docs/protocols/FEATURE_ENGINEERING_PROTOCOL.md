# HARTH: interpretable feature engineering

**Stage:** Extract descriptive features from already validated activity windows.

## Shared code

`code/python/src/core/feature_engineering.py` exposes `extract_signal_features(X_windows, channel_names, methods)`.

- Input: `(windows, channels, time)`, finite numeric arrays.
- Output: `pandas.DataFrame` with one row per window and names `channel__feature`.
- Five initially implemented methods: mean, population STD, RMS, peak-to-peak, normalised waveform length (average absolute adjacent difference).
- For the current six accelerometer channels, the initial **30-feature** representation is expected.
- The function has no fitted state, subject labels or model parameters.

**Reuse:** the same function supports other three-dimensional channel-first windows, including different channel counts such as BoneNet's 42-channel tensor, where axis/time interpretation has been verified.

## Interactive notebook

`notebooks/03_statistical_feature_engineering.ipynb` repeats Notebook 02's protocol and displays the actual feature table, basic statistical summaries, RMS scatter and boxplot across labelled activities. Python source code stays in the shared module, while the notebook provides markdown descriptions and brief English comments within code cells.

## Quality and scientific limitations

This first example uses **S023 only**. It is not valid unseen-participant training or testing. 50%-overlapping windows are not independent, and class proportions may be altered by the strict no-transition window policy. Scaling, learned feature selection and classifier fitting must occur **after subject-disjoint partitioning**. Correlated features may need investigation; do not infer superiority without evaluation.

No baseline model or feature selection has been trained at this stage.
