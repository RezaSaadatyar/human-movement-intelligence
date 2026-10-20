# HARTH | Signal windowing

**Stage:** Reusable preprocessing demonstration with the actual HARTH v2.0 `S023.csv` recording.

## Why do we need windows?

Raw HARTH is one continuous time series with six acceleration channels. Most supervised activity-recognition models expect shorter fixed-size sequences with one label. This step constructs them explicitly, without confusing individual timestamps with independent participants.

## Implemented generic Python function

`code/python/src/core/signal_preprocessing.py`

```python
from core.signal_preprocessing import create_signal_windows

X_windows, y_windows, info = create_signal_windows(
    signals=signals, labels=labels,
    window_size=100, step_size=50,
    timestamps=timestamps, sampling_rate=50,
    min_label_purity=1.0, subject_id="S023",
)
```

- Input: `(samples, channels)`.
- Output: `(windows, channels, 100)` plus one activity label per window and a metadata table.
- 100 points at 50 Hz is 2 seconds; stride 50 gives overlapping candidate windows.
- Each retained window must contain **100% one label** and cannot span significant acquisition timing gaps or nonmonotonic timestamps.
- No feature normalisation is fitted and no random split is performed. Later splits must hold out full participants **before learning preprocessing statistics**.

## Execute Notebook 02 in Cursor

`studies/wearable_activity_recognition/notebooks/02_signal_preprocessing_and_windowing.ipynb`

The notebook reuses the cached source and the functions from the first notebook. It displays actual tensor dimensions, retained windows, class frequencies and example sensor traces. Unit tests: `tests/test_signal_preprocessing.py`.

## Boundaries

Results in a single participant are a **data engineering demonstration**, not classification performance. Cross-subject machine learning requires multiple independent participants and a scientifically justified validation design. Strict transition exclusion changes the class distribution; report both pre- and post-windowing distributions.

The original shared BoneNet notebook remains unchanged.
