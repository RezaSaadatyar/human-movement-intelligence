# Wearable Activity Recognition

## First implemented stage: HARTH data inspection

**Scientific direction:** classify activities from wearable motion signals with participant-independent evaluation once full multi-subject data and a documented splitting policy are available.

### Dataset for the first reusable-code demonstration

- **HARTH v2.0:** official [NTNU AI Lab repository](https://github.com/ntnu-ai-lab/harth-ml-experiments), 31 participants in its dataset release.
- **This first notebook:** only `S023.csv` (one participant, about 12 MB), to confirm file loading, label inspection and waveform plotting without downloading the entire dataset.
- **Do not confuse with:** [UCI HARTH](https://archive.ics.uci.edu/dataset/779/harth), an older v1.2 22-participant release.
- **Sensor data:** 3-axis lower-back and right-thigh acceleration, 50 Hz; six acceleration columns, timestamp and activity code.
- **Source paper:** Logacjov et al., *HARTH: A Human Activity Recognition Dataset for Machine Learning* (2021). Dataset DOI: https://doi.org/10.24432/C5NC90.

## Run

From the project root:

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

Open `notebooks/01_data_loading_and_inspection.ipynb` in Cursor and execute cells in order. The notebook downloads the single public CSV only if not present in `data/raw/harth_v2/`.

## Implemented code

- Common functions: `code/python/src/core/data_loading.py`, `data_inspection.py`
- HARTH-specific adapter: `src/harth_data.py`
- Notebook: `notebooks/01_data_loading_and_inspection.ipynb`
- Tests: `tests/test_harth_data.py`

**Research limits:** No supervised model has been trained; a one-person example cannot be used for participant-independent evaluation. Activity class counts will be computed by the notebook, not assumed. Future steps must build sample windows carefully and hold out entire participants.

## Step 2: annotation-safe signal windowing

The [second notebook](notebooks/02_signal_preprocessing_and_windowing.ipynb) calls the reusable [signal preprocessing function](../../code/python/src/core/signal_preprocessing.py) to make 2-second, 50%-overlap candidate windows at 50 Hz. It discards windows crossing label transitions or timestamp gaps, prints retained-window metadata and plots real signals. It does **not** train a classifier or perform a train/test split. See [protocol](WINDOWING_PROTOCOL.md).

## Step 3: Statistical feature engineering

[Notebook 03](notebooks/03_statistical_feature_engineering.ipynb) demonstrates reusable mean, standard deviation, RMS, peak-to-peak and waveform-length features on real 2-second HARTH windows. It shows the resulting 30-feature matrix and activity-dependent plots, without fitting or evaluating a model. See the [feature protocol](FEATURE_ENGINEERING_PROTOCOL.md). All new notebook code cells include concise English purpose comments.
