# Essex IMU–FE study: running the notebooks

The study has eight notebooks under `biomechanical_load_estimation/code/notebooks/`. Run them in numerical order for a full calculation. In Colab, start with **Colab Repository Setup**. Locally, install the shared `requirements.txt`, which includes PyTorch for neural experiments. The notebooks install `mat-io==1.0.1` for the original MATLAB tables and use the shared device and plotting helpers.

## Current input and output files

| Notebook | Reads | Writes for later notebooks |
| --- | --- | --- |
| 01 | Essex Figshare v13 source files | `01_original_record_registry.csv` and source summaries |
| 02 | Current 01 registry and MATLAB files | `02_python_column_schema.csv`, decoding tables |
| 03 | Current 01 registry and 02 schema | `03_original_time_coordinate_mapping.csv`, numeric summaries |
| 04 | Current registry, mapping and MATLAB files | `04_trial_split_assignments.csv`, schemas and local paired NPZ |
| 05 | Current trial assignments and paired arrays | Classical metrics and `05_training_fold_registry.csv` |
| 06 | Current arrays, assignments and 05 folds/results | Temporal metrics and diagnostic figures |
| 07 | Current arrays, 05 folds/results and 06 results | Multi-seed metrics, ablation and candidate comparison |
| 08 | Current arrays, assignments and model settings in Python | Final model predictions, metrics and figures |

Public tables and figures are written directly to `biomechanical_load_estimation/output/`, with the notebook number as prefix. Rerunning a notebook replaces its current files. If a notebook has not been rerun, its current saved output remains available to downstream notebooks.

Raw files live under ignored `data/raw/essex_femoral_neck_loading/`. Existing downloads are reused. New downloads are checked for incomplete transfer; ZIP extraction retains path safety checks and replaces the current extracted files. Remove or replace a raw file deliberately if a new download is needed.

Notebook 04 writes `data/processed/essex_femoral_neck_loading/essex_imu_fe_pairs_v13.npz`. Notebooks 05–08 reuse that file. When it is missing or older than the current registry, time mapping, assignments or raw archive, it is reconstructed from those inputs. Arrays are matched to metadata by trial key.

## Scientific settings

The original participant allocation remains unchanged: 29 Training, four Validation and four Test participants in the recorded dataset. Original IMU and FE dimensions are 101 × 42 and 11 × 2. FE values remain unitless and use original time positions, without interpolated targets. Required shape, missing-data, trial-key and participant-overlap checks remain where necessary for valid calculations.

Notebook 07 executes **75 neural fits** each time its training cell runs; previous per-fold files do not skip training. A GPU is advisable. Notebook 08 refits the Random Forest primary and descriptive TCN secondary on Development data and overwrites its current Test outputs. Notebook 05's Random Forest used float `max_features=1.0` (all 210 features per split), while Notebook 08 uses integer `max_features=1` (one candidate feature per split), as recorded in the final model settings. The existing Test results are not used to select model settings.

## What has been verified

Offline tests cover current-file replacement, array-to-trial alignment, original participant allocation, model tensor shapes, participant isolation and small synthetic training examples. Notebook syntax and current prediction-table consistency are checked without fitting on the real Test set.

All eight notebooks include execution results, CSV tables and figures. The recorded Colab runs include CUDA execution for Notebooks 06–08, with Tesla T4 hardware reported for Notebooks 07–08. Neural results may vary across computing environments.

See the [research report](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) for methods and limitations.
