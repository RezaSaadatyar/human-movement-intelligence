# Research Datasets and Access

The active studies use distinct original datasets, model inputs, targets and participant-level evaluation protocols. Follow each study's scientific documentation before downloading data or reproducing an analysis.

## 1. Datasets Used in the Research

| Study | Original source | Inputs and targets | Protocol and access |
| --- | --- | --- | --- |
| [Wearable Activity Recognition](../wearable_activity_recognition/README.md) | [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments) | Six lower-back/thigh acceleration channels; eight original activity codes | Original source revision and windowing rules are given in the [HARTH methods](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) |
| [Biomechanical Load Estimation](../biomechanical_load_estimation/README.md) | [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3) | Seven lower-body IMUs and model-derived femoral-neck principal-strain targets | Source files and participant/trial exclusions are documented in the [Essex methods](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |
| [Multimodal Movement Learning](../multimodal_movement_learning/README.md) **(five notebooks executed)** | [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html) / [Dryad](https://doi.org/10.5061/dryad.1k84r) | Paired 12-channel sEMG and 36-column triaxial acceleration; corrected E1 hand-movement labels | Original E1 source and 40-person paired cohort inspected; 40,000 selected windows (35 Development and five reserved Test people). Frozen Test macro F1: `F1_rf` **0.3126** versus `B1_rf` **0.2512**; offline same-source evaluation, not native sensor-clock verification or external clinical validation. |
| [Gait Analysis](../gait_analysis/README.md) **(nine-person source pilot complete; reference validation pending)** | [Camargo et al. (2021)](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) / [Mendeley Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2) | Original foot/shank gyroscope IMU candidate and independent vertical-force contact reference | Initially treadmill; source-side, channel and native-time pairing must be confirmed in Notebook 01/02 before any IC timing MAE is reported. Original-data source auditing is complete; no independently verified IC/TC timing result is available. |

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

### 2.3 NinaPro DB2 — Multimodal Movement Learning (executed)

The official [NinaPro DB2 source](https://ninapro.hevs.ch/instructions/DB2.html) lists 40 intact participants and separate source ZIP archives. The [original data descriptor](https://doi.org/10.1038/sdata.2014.53) identifies sEMG sampled at 2,000 Hz and triaxial accelerometers originally sampled at approximately 148 Hz; published streams were synchronised to a common higher-rate representation. The released dictionary lists 12 `emg` and 36 `acc` columns, as well as labels and repetition identifiers. The data descriptor records correction of movement boundaries after acquisition.

The [Dryad original deposition](https://doi.org/10.5061/dryad.1k84r) lists individual DB2 subject ZIPs; its full 23.27 GB listing includes DB1–DB3, not only DB2. The executed workflow first inspected S01 E1, then processed all 40 original participant archives incrementally. Original ZIP/MAT archives and source-length per-subject arrays remain untracked; the **two compact statistical feature matrices** are a deliberate exception tracked in Git for reproducing the frozen classical Test. Preserve original-source attribution and apply applicable redistribution terms. Current storage paths:

| Path | Purpose and current tracking |
| --- | --- |
| `data/raw/ninapro_db2/` | Source subject ZIP/MAT files, downloaded into the runtime and not tracked in Git |
| `data/processed/ninapro_db2/` | Git-tracked `02_emg_features.npy` (48 columns) and `02_acc_features.npy` (96 columns); large source-length signals remain local/untracked |
| [`multimodal_movement_learning/output/`](../multimodal_movement_learning/output/) | Current `01_*`–`05_*` source, Development and frozen Test CSV/PNG evidence, plus original Notebook 03 OOF probability NPZ |

The implemented task is corrected Exercise E1 (Rest plus 17 non-force finger movements), using sEMG and acceleration only; `glove`, `force`, subject/repetition IDs and displayed cue timing are not predictors. All five notebooks have recorded source, Development or Test execution. The [study README](../multimodal_movement_learning/README.md) and [scientific methods and evidence](../multimodal_movement_learning/MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md) document the fixed selection, the five-person participant-disjoint Test findings and the important limits on within-dataset inference.

### 2.4 Camargo — Gait Analysis (selected nine-person source pilot complete)

The original [Camargo et al. (2021) dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) provides 22 healthy participants' locomotion across treadmill, level-ground, ramp and stair conditions. Available measurements include IMU (nominally 200 Hz), force plates (nominally 1,000 Hz), EMG, motion capture and biomechanics. Source-specific timestamps, axes, force signs and foot/plate correspondence are **not inferred** solely from published rates or the dataset directory name.

Notebook 01 has completed the selected original **AB06–AB14 nine-person source audit**, with **27 candidate treadmill IMU/force pairs**, 27 readable native-time overlaps and **seven published source-audit files**. The updated executed source inventory contains **135 original MAT files**, including 81 matching `gcRight`, `gcLeft` and `conditions` files. These are source-identity and timebase observations, not validated IC/TC labels. For Colab Free, [Mendeley Part 1 Version 1](https://doi.org/10.17632/fcgm3chfff.1) provides **nine independent subject ZIPs, AB06–AB14**, rather than the much larger combined Part 1 Version 2 ZIP. Notebook 01 originally completed the resource-bounded nine-person IMU/force acquisition. Its **revised** acquisition stage now prepares all missing AB06–AB14 subject archives sequentially in the same Colab execution, retains up to three deterministically selected treadmill IMU/force pairs per participant and extracts same-trial original `gcRight`, `gcLeft` and `conditions` streams when those members exist. Supplemental availability is recorded in `01_auxiliary_source_inventory.csv`; it is not treated as reference-label validation. Only temporary ZIPs downloaded by the current run are removed. The total initial network transfer can still be substantial (roughly 9 GB); interruption may require rerunning Notebook 01 while reusing locally extracted source files. Raw source is held in `data/raw/camargo/` in the active Colab runtime, not Git-tracked. **Notebook 01 and Notebook 02 must run in the same Colab runtime**. Notebook 02 does not download the source, and it stops before writing scientific results if the selected original IMU or force files are missing.

The implemented [Notebook 01](../gait_analysis/code/notebooks/01_source_and_reference_eligibility.ipynb) records original field and trial pairing candidates. [Notebook 02](../gait_analysis/code/notebooks/02_alignment_and_event_reference.ipynb) constructs independent contact references only after channel/side/timebase review, then writes numeric per-trial caches to `data/processed/camargo/paired_trials/`. Gait score tables and figures will be produced under [`gait_analysis/output/`](../gait_analysis/README.md#5-files-and-saved-outputs) when the original data are actually executed.

No IMU model result, physical timing error or eligible source-cohort count is reported before that verification. See the [study README](../gait_analysis/README.md) and [scientific methods](../gait_analysis/GAIT_RESEARCH_METHODS_AND_EVIDENCE.md).

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
