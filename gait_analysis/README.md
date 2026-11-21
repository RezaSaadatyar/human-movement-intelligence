# Gait Analysis | Camargo Lower-Limb Biomechanics

![Complete source-to-Test gait measurement workflow](images/study_overview.svg)

*Figure 1. Proposed five-notebook analytical pipeline: source inspection, independently force-derived contact events, reference stride measures, participant-grouped IMU detectors and a reserved final Test. This is an implementation schematic, not recorded performance.*

This study implements a **reference-anchored, participant-independent gait-event measurement workflow** using [Camargo et al.'s wearable and laboratory biomechanics dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/). Its objective is to estimate initial-contact timing from foot/shank inertial signals, compare it with independent ground-force contact evidence, and assess variation across observed walking conditions.

**Study status (November 2026):** Five notebooks and reusable evaluation modules are implemented. **Notebook 01 has been executed with original AB06 treadmill IMU/force files**, producing a source-matched trial inventory, channel audit and recorded signal plot. Physical force-axis interpretation, independent contact references, the eligible cohort and final Test metrics still require subsequent investigation. No event-detection performance is claimed.

## 1. Dataset and Experimental Protocol

| Feature | Study definition |
| --- | --- |
| Original dataset | [Camargo et al. (2021)](https://doi.org/10.1016/j.jbiomech.2021.110320), 22 healthy participants in the released research dataset |
| Data access | [Georgia Tech/EPIC](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) · [Mendeley Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2) (CC BY 4.0) |
| IMU inputs | Candidate foot/shank gyroscope channels, to be confirmed against original source headers and units |
| Independent reference | Ground-force-derived contact events with verified vertical-force sign, foot side, native time correspondence and contact threshold |
| Candidate conditions | Initially treadmill, with separate treatment for ground/ramp/stair experiments if source pairing supports them |
| Research unit | Participant → condition → original trial → independent force contact → detected event |
| Primary endpoint | Mean per-participant **Initial Contact absolute timing error (ms)**, for correctly matched events, alongside coverage |
| Essential secondary metrics | Event sensitivity, precision, missed/extra events, Terminal Contact when supported, stride-time error and cadence/variability |
| Independence | Participant-disjoint Development folds and held-out Test participants, assigned only after original eligibility is known |

The original release reports roughly 200-Hz inertial and 1,000-Hz force sensing; recorded timestamps and verified units take precedence over those descriptive rates. The right/left foot mapping of each vertical-force channel must be checked. A phase-normalised gait-cycle curve alone is **not** a native-time reference for millisecond-level event accuracy.

## 2. Experimental Programme

| Stage | Notebook | Actual programmed purpose |
| --- | --- | --- |
| 01 | [Source and Reference Eligibility](code/notebooks/01_source_and_reference_eligibility.ipynb) | Inventory extracted ABxx archives and original files; inspect paired IMU/force candidate paths, field names and timestamps |
| 02 | [Alignment and Event Reference](code/notebooks/02_alignment_and_event_reference.ipynb) | Use a **reviewed** sensor/side/time contract; construct force-contact events, reusable trial arrays and participant roles |
| 03 | [Stride Measurement and Statistics](code/notebooks/03_stride_measurement_and_statistics.ipynb) | Extract force-reference stride intervals, cadence and variability using Development people only |
| 04 | [IMU Baselines and Grouped Validation](code/notebooks/04_imu_event_baselines_and_grouped_validation.ipynb) | Compare fixed and Training-tuned IMU peak detectors in participant-disjoint Development folds; freeze configuration |
| 05 | [Frozen Test and Research Synthesis](code/notebooks/05_frozen_test_and_research_synthesis.ipynb) | Apply the chosen Development configuration to reserved people; report event coverage, timing error and descriptive paired uncertainty |

Individual workflow diagrams are in [`images/`](images/). They show method stages only; **no diagram contains synthetic performance figures**.

## 3. Main Results

**Notebook 01 source-audit results (executed, not a model evaluation):** The tracked [trial inventory](output/01_trial_inventory.csv), [candidate pairs](output/01_candidate_pairs.csv), [observed columns](output/01_observed_columns.csv), [recorded signal preview](output/01_sensor_source_preview.png) and [eligibility summary](output/01_source_eligibility_summary.csv) establish:

| Source observation | Recorded pilot |
| --- | --- |
| Participant and trial | `AB06`, session `10_09_18`, `treadmill_01_01` |
| Candidate same-trial IMU/force pairs | 1 |
| Measurement fields, excluding `Header` | IMU: 24; treadmill force-platform: 18 |
| Median `Header` increments | IMU: 0.005; force platform: 0.001 (source-time units) |
| Independent force-contact labels and final Test metrics | Not yet generated |

The source pairing is **not yet a validated force-contact reference**. No eligible full-cohort count, force-contact performance or participant-level timing error has been established.

Stage 01 results are now saved and tracked; later stages will produce their own files when the required sources and scientific contracts have been verified:

| Evidence | Intended current output |
| --- | --- |
| Executed original source inspection | [`01_trial_inventory.csv`](output/01_trial_inventory.csv), [`01_observed_columns.csv`](output/01_observed_columns.csv), [`01_source_eligibility_summary.csv`](output/01_source_eligibility_summary.csv) and [recorded signal figure](output/01_sensor_source_preview.png) |
| Force-reference eligibility and original contact events | `02_eligible_trials.csv`, `02_reference_events.csv` |
| Reference-only Development stride measures | `03_reference_trial_metrics.csv`, `03_subject_condition_metrics.csv` |
| Independent Development comparator | `04_development_participant_metrics.csv`, `04_selected_configuration.csv` |
| Frozen Test outcome | `05_test_participant_metrics.csv`, `05_test_model_summary.csv` |

Any result tables will be interpreted alongside event matching support and source exclusions; an artificially low timing MAE obtained by predicting only a few events is not a successful detector.

## 4. Running and Reviewing the Study

**Use Google Colab:** Open [Notebook 01](code/notebooks/01_source_and_reference_eligibility.ipynb), run **00. Colab Repository Setup** and **01. Imports and Research Environment**, then execute in order. The repository uses the established `setup_colab("gait_analysis")` helper.

The official source archives are **not distributed with this repository**. For a first pilot, download only **AB06.zip** (approximately **1.12 GB**) from [Mendeley Part 1, Version 1](https://data.mendeley.com/datasets/fcgm3chfff/1), also linked by the [Georgia Tech dataset portal](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/). **Do not use Part 1 Version 2's combined nine-participant 9.38-GB ZIP** for this one-person pilot.

**Notebook 01, section 02, prepares the source:** its default `AUTO_DOWNLOAD_PILOT = True` requests only the original **AB06.zip** from Mendeley Version 1 if no local data exist. This downloads approximately **1.12 GB** into the Colab runtime, then extracts **one paired treadmill IMU/force trial** while preserving original relative paths. The official public-file API must be reachable; if it restricts the transfer, download AB06.zip from the linked Mendeley page and place it in `/content/`, or set `MOUNT_GOOGLE_DRIVE = True` and `PILOT_ZIP_PATH` to the mounted file. Existing extracted source files and local ZIP copies take precedence, and `AUTO_DOWNLOAD_PILOT = False` disables network transfer. Alternatively, set `CAMARGO_DATA_ROOT` to a reviewed original source folder containing `ABxx/session/mode/sensor/trial` files.

Notebook 01 then inventories the actual source columns and displays a genuine paired trial. This pilot **does not establish a valid shared sensor clock, foot side, vertical-force sign or final cohort**. Notebook 01 installs the pinned `mat-io==1.0.2` dependency when needed, allowing its source adapter to read MATLAB table/MCOS objects as named tabular data rather than treating the object as a numeric matrix. SciPy remains the reader for simple numeric MATLAB structures. The original `Header` time column and channel names are preserved; actual sensor side, physical units and inter-device timing still need source-based review.

**Notebook 02 has an intentional source-contract boundary:** the source-dependent gyroscope axis, independently measured vertical-force column, sign, foot side, timestamp scale and shared timebase must be specified after examining Notebook 01. It will stop with an actionable error rather than guessing. Similarly, Notebook 04 cannot run until Notebook 02 has established enough independent eligible people for grouped Development folds. These are scientific validity requirements, not approval/status-file gates.

**Research output download:** Every Notebook now ends with a **Research Summary and Output Download** cell. It packages only that notebook's saved `01_`–`05_` outputs as `/content/gait_analysis_Notebook_XX_Outputs.zip` (where `XX` is the notebook number). Retrieve the ZIP from **Colab Contents / Files**; packaging does not trigger GitHub commit/push or include raw Camargo archives. The Notebook 01 source plot displays observed `foot_Gyro_Y` / `shank_Gyro_Y` and `Treadmill_R_vy` / `Treadmill_L_vy` channels when present. The waveform figure is source evidence, **not** an independently validated vertical-force contact reference.

Notebooks 03–05 consume explicit current results from their producers. Original raw recordings remain local; the study's useful CSV tables/PNG charts are written under `gait_analysis/output/` and numeric per-trial caches under `data/processed/camargo/paired_trials/`. Execution overwrites each stage's named current outputs; a material source or protocol change requires affected downstream outputs to be recomputed.

Use the repository's existing [dependencies](../requirements.txt) and the [data access guide](../docs/DATASETS.md) for original-source instructions. The original-data **Notebook 01 source inspection has been executed**; the complete source-to-Test experiment has **not** been run or validated.

## 5. Files and Saved Outputs

| Location | Responsibility |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five numbered Colab/Jupyter scientific stages |
| [`code/src/`](code/src/) | Source inventory, timebase/force pairing, reference events, detectors and metrics |
| [`images/study_overview.svg`](images/study_overview.svg) | Study-wide editable five-band method diagram |
| [`images/`](images/) | Five additional stage-specific editable SVGs |
| [`output/`](output/) | Five tracked Notebook 01 CSV/PNG outputs; subsequent-stage outputs when generated |
| `../data/raw/camargo/` | Original participant files (local or mounted; not tracked) |
| `../data/processed/camargo/` | Reusable numeric paired-trial arrays (local or mounted) |
| [Scientific methods document](GAIT_RESEARCH_METHODS_AND_EVIDENCE.md) | Full scientific design, model interfaces, evaluation and limits |

No Excel stage tracker, administrative approval document or mandatory history/manifest file is part of this study.

## 6. Interpretation and Scope

The programmed research question is **how accurately a simple, source-aligned IMU detector can estimate force-referenced initial contacts for unseen participants**. Measurement must use the same trial, genuine physical time and a validated foot/force assignment. The principal result will be a participant-balanced timing error **together with missed/extra event counts and event sensitivity**.

The analysed population comes from healthy volunteers. This workflow does **not** establish clinical gait validity, fall-risk prediction, pathological gait detection, online inference or deployment. The filtering in the initial detectors is zero-phase and therefore **offline**. Generalisation across terrains is not claimed unless actual eligible source trials support comparisons.

If independent original-time references cannot be constructed, the appropriate next step is to revisit the research question **before** interpreting IMU predictions as contact-timing accuracy. Read [Research Methods and Evidence](GAIT_RESEARCH_METHODS_AND_EVIDENCE.md) for the complete scientific contract and pending original-data validation.

**Source attribution:** Camargo, J., Ramanathan, A., Flanagan, W. and Young, A. (2021), *Journal of Biomechanics*, 119, 110320, [DOI](https://doi.org/10.1016/j.jbiomech.2021.110320); [dataset Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2).
