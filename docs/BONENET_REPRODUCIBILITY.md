# Essex IMU–FE study: running the notebooks

The study has eight notebooks under `biomechanical_load_estimation/code/notebooks/`. Run them in numerical order for a full calculation. In Colab, start with **Colab Repository Setup**. Locally, install `requirements.txt`; neural experiments also need `requirements-dl.txt`. The notebooks install `mat-io==1.0.1` for the original MATLAB tables and use the shared device and plotting helpers.

## Current input and output files

| Stage | Reads | Writes for later stages |
| --- | --- | --- |
| 01 | Essex Figshare v13 source files | `01_original_record_registry.csv` and source summaries |
| 02 | Current 01 registry and MATLAB files | `02_python_column_schema.csv`, decoding tables |
| 03 | Current 01 registry and 02 schema | `03_original_time_coordinate_mapping.csv`, numeric summaries |
| 04 | Current registry, mapping and MATLAB files | `04_trial_split_assignments.csv`, schemas and local paired NPZ |
| 05 | Current trial assignments and paired arrays | Classical metrics and `05_training_fold_registry.csv` |
| 06 | Current arrays, assignments and 05 folds/results | Temporal metrics and diagnostic figures |
| 07 | Current arrays, 05 folds/results and 06 results | Multi-seed metrics, ablation and candidate comparison |
| 08 | Current arrays, assignments and model settings in Python | Final model predictions, metrics and figures |

Public tables and figures are written directly to `biomechanical_load_estimation/output/`, with the notebook number as prefix. Rerunning a notebook replaces its current files. If a notebook has not been rerun, its current saved output remains available to downstream notebooks. No historical folder migration, hash matching or JSON approval report is needed.

Raw files live under ignored `data/raw/essex_femoral_neck_loading/`. Existing downloads are reused. New downloads are checked for incomplete transfer; ZIP extraction retains path safety checks and replaces the current extracted files. Remove or replace a raw file deliberately if a new download is needed.

Notebook 04 writes `data/processed/essex_femoral_neck_loading/essex_imu_fe_pairs_v13.npz`. Notebooks 05–08 reuse that file. When it is missing or older than the current registry, time mapping, assignments or raw archive, it is reconstructed from those inputs. Arrays are matched to metadata by trial key. No earlier output directories are searched.

## Scientific settings

The original participant allocation remains unchanged: 29 Training, four Validation and four Test participants in the recorded dataset. Original IMU and FE dimensions are 101 × 42 and 11 × 2. FE values remain unitless and use original time positions, without interpolated targets. Required shape, missing-data, trial-key and participant-overlap checks remain where necessary for valid calculations.

Stage 07 executes **75 neural fits** each time its training cell runs; previous per-fold files do not skip training. A GPU is advisable. Stage 08 refits the existing Random Forest primary and descriptive TCN secondary on Development data and overwrites current Test outputs. Test results must not be used to retune models; rerunning the calculation does not create an independent validation sample.

## What has been verified

Offline tests cover current-file replacement, array-to-trial alignment, original participant allocation, model tensor shapes, participant isolation and small synthetic training examples. Notebook syntax and current prediction-table consistency are checked without fitting on the real Test set.

The simplified notebooks have **not** been run end to end on the full Essex dataset or in hosted Colab. Cleared cell outputs reflect that status. Existing numerical CSVs and plots describe the previous completed run and were not regenerated during the simplification. GPU results can vary across environments.

See the [research report](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) for methods and limitations, and the [change record](BONENET_WORKFLOW_SIMPLIFICATION.md) for the removed workflow components.
