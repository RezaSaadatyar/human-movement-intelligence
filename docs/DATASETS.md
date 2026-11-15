# Research Datasets and Access

The active studies use distinct original datasets, model inputs, targets and participant-level evaluation protocols. Follow each study's scientific documentation before downloading data or reproducing an analysis.

## 1. Datasets Used in the Research

| Study | Original source | Inputs and targets | Protocol and access |
| --- | --- | --- | --- |
| [Wearable Activity Recognition](../wearable_activity_recognition/README.md) | [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments) | Six lower-back/thigh acceleration channels; eight original activity codes | Original source revision and windowing rules are given in the [HARTH methods](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) |
| [Biomechanical Load Estimation](../biomechanical_load_estimation/README.md) | [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3) | Seven lower-body IMUs and model-derived femoral-neck principal-strain targets | Source files and participant/trial exclusions are documented in the [Essex methods](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |
| [Multimodal Movement Learning](../multimodal_movement_learning/README.md) **(source/cohort executed)** | [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html) / [Dryad](https://doi.org/10.5061/dryad.1k84r) | Paired 12-channel sEMG and 36-column triaxial acceleration; prompted hand-movement labels | Original E1 participant archives and common stored source-grid pairing inspected; 40 people, 337,853 eligible paired windows, 40,000 selected windows, and person-disjoint 35/5 roles. No independent native sensor clock validation or final model Test results. |

The active HARTH study uses the **NTNU AI Lab v2.0 source**, not the [UCI HARTH release](https://archive.ics.uci.edu/dataset/779/harth). The Essex study uses **model-derived FE targets**, not direct in-vivo strain measurements or clinical outcomes.

## 2. Source Access and Local Storage

### 2.1 HARTH v2.0

Use the official [NTNU AI Lab HARTH repository](https://github.com/ntnu-ai-lab/harth-ml-experiments) at the study's fixed revision `dad2cfbe89a26f72f19770419469ac037de200df`; it is **not** the UCI HARTH v1.2 release. The recorded task uses six lower-back/thigh accelerometer channels at nominal 50 Hz and source activity codes 1–8, with 100-sample windows and stride 50. Do not infer labels for other source codes.

Use the acquisition and window-building routines in [HARTH Notebook 01](../wearable_activity_recognition/code/notebooks/01_harth_data_foundation.ipynb) and the maintained source adapter in [`code/src/`](../wearable_activity_recognition/code/src/). Source recordings are stored locally under `data/raw/harth_v2/`. Model artefacts generated later are stored under `data/processed/harth_final_models/`. See the [HARTH README](../wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) for execution order and [scientific methods](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) for eligibility and split rules.

### 2.2 Essex Figshare v13

**Original dataset:** [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3). Figshare article ID **30225739**, version **13**. This is the original MATLAB collection, not the separate 444-record BoneNet derivative.

The following file inventory was recorded from the official v13 source list. These figures describe source files; they are not hard-coded runtime assertions.

| File | Size (MiB) | Figshare file ID | Recommended action |
| --- | ---: | ---: | --- |
| README.md | 0.01 | 63343417 | Download and inspect first |
| SubCharacteristics.xlsx | 0.03 | 63342400 | Download and inspect first |
| MATLAB.zip | 224.68 | 63190690 | Core data, obtain after README |
| angles.xlsx | 29.73 | 63342406 | Joint-angle analysis, later |
| OpenSim.zip | 10.40 | 63190519 | Optional MSK modelling |
| Visual3D.zip | 18414.04 | 63341380 | Optional, not used in this eight-notebook analysis (approximately 18 GiB) |

Begin with the source `README.md` and `SubCharacteristics.xlsx`, then obtain `MATLAB.zip` for the executed study. The joint-angle, OpenSim and Visual3D files are optional for these notebooks; in particular the approximately 18 GiB Visual3D archive was not needed. The initial source inventory recorded **42 ZIP entries**, including **40 participant MAT files** under `MATLAB/Processed Data/`, and approximately **224.79 MiB** of uncompressed content.

The current original-source workflow is managed by [Essex Notebook 01](../biomechanical_load_estimation/code/notebooks/01_original_dataset_audit_and_trial_registry.ipynb):

| Local path | Contents |
| --- | --- |
| `data/raw/essex_femoral_neck_loading/` | Downloaded original source files |
| `data/raw/essex_femoral_neck_loading/matlab_records/` | Extracted participant MAT files |
| `data/processed/essex_femoral_neck_loading/essex_imu_fe_pairs_v13.npz` | Locally reconstructed IMU–FE trial pairs |
| [`biomechanical_load_estimation/output/`](../biomechanical_load_estimation/output/) | Public source audits, trial metadata, metrics and figures |

Existing source downloads are reused. When `MATLAB.zip` is unavailable, the source loader tries the project data release and official Figshare endpoints for the same v13 archive. A new transfer is checked for completion before replacement; extraction checks ZIP paths and CRC integrity. Original participant/trial identifiers are preserved. Notebooks 02–03 inspect nested tables, numeric columns and **source FE time coordinates**. Notebook 04 creates paired arrays and participant-disjoint splits; FE observations are **not** assumed to occur at every tenth IMU sample. Transformations used for learning are fitted within the appropriate Training/Development partition.

The current [Figshare file manifest](../biomechanical_load_estimation/output/01_dataset_figshare_file_manifest.csv), [Essex README](../biomechanical_load_estimation/README.md) and [scientific methods](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) provide the source and study-specific details.

### 2.3 NinaPro DB2 — Multimodal Movement Learning (planned)

The official [NinaPro DB2 source](https://ninapro.hevs.ch/instructions/DB2.html) lists 40 intact participants and separate source ZIP archives. The [original data descriptor](https://doi.org/10.1038/sdata.2014.53) identifies sEMG sampled at 2,000 Hz and triaxial accelerometers originally sampled at approximately 148 Hz; published streams were synchronised to a common higher-rate representation. The released dictionary lists 12 `emg` and 36 `acc` columns, as well as labels and repetition identifiers. The data descriptor records correction of movement boundaries after acquisition.

The [Dryad original deposition](https://doi.org/10.5061/dryad.1k84r) lists individual DB2 subject ZIPs; its full 23.27 GB listing includes DB1–DB3, not only DB2. Retrieve a single participant's archive first, verify file integrity, observe actual MATLAB fields, and then acquire eligible people incrementally. Do not commit original ZIP/MAT archives, processed sample arrays or participant-specific source data. Standard local working paths:

| Planned local path | Purpose |
| --- | --- |
| `data/raw/ninapro_db2/` | Source subject archives and MATLAB exercise files |
| `data/processed/ninapro_db2/` | Compact aligned feature matrices and source row/window identifiers |
| [`multimodal_movement_learning/output/`](../multimodal_movement_learning/output/) | Future generated, study-owned CSV summaries and figures only after execution |

The intended task is a selected non-force exercise plus rest, using sEMG and acceleration only; `glove`, `force`, subject/repetition IDs and displayed cue timing are not predictors. The [study README](../multimodal_movement_learning/README.md) and [scientific plan](../multimodal_movement_learning/MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md) describe the five planned notebooks and person-disjoint evaluation. The pilot file has not been downloaded or inspected in this project.

## 3. Additional Public Collections

These collections may be relevant to future independent comparisons, but **they are not interchangeable with the current HARTH or Essex datasets**:

| Dataset | Source | General content |
| --- | --- | --- |
| PAMAP2 | [UCI dataset 231](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | Activity monitoring using wearable signals |
| MHEALTH | [UCI dataset 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | Wearable movement and physiological signals |
| UCI HAR Smartphones | [UCI dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Smartphone inertial activity recognition |
| Auckland walking/running biomechanics | [Auckland Figshare project](https://auckland.figshare.com/projects/Dataset_of_walking_and_running_biomechanics_with_different_step_widths_across_different_speeds/220732) | Gait and running biomechanics at varied speeds and step widths |

Any future comparison would require its own source/label mapping, participant-disjoint design and evaluation plan. Separate sources must not be described as subject-linked multimodal observations without actual record linkage.

## 4. Data Integrity and Reuse

For a scientific reproduction, verify the dataset citation and version, licence, source units and channel meanings, subject/trial identifiers, sampling frequency, timestamp continuity, exclusion rules and target provenance. Establish participant separation before fitting transformations or selecting models, and document discrepancies rather than silently recoding labels.

Public availability of a dataset does not automatically grant unrestricted redistribution or establish clinical representativeness.
