# Gait Analysis | Camargo Lower-Limb Biomechanics

![Complete source-to-Test gait measurement workflow](images/study_overview.svg)

*Figure 1. Proposed five-notebook analytical pipeline: source inspection, independently force-derived contact events, reference stride measures, participant-grouped IMU detectors and a reserved final Test. This is an implementation schematic, not recorded performance.*

This study implements a **reference-anchored, participant-independent gait-event measurement workflow** using [Camargo et al.'s wearable and laboratory biomechanics dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/). Its objective is to estimate initial-contact timing from foot/shank inertial signals, compare it with independent ground-force contact evidence, and assess variation across observed walking conditions.

**Study status (November 2026):** Notebook 01's latest committed execution and all six source-audit files now cover **AB06–AB14: nine participants and 27 paired treadmill IMU/force trials**. All 27 original sensor pairs were readable with overlapping native timestamps. Notebook 01 source preparation is complete for this selected pilot. Notebook 02 has executed its diagnostic source-contract review; force-contact references remain independently unverified. Physical force-axis interpretation, independent contact references, the eligible cohort and final Test metrics still require subsequent investigation. No event-detection performance is claimed.

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
| Recorded participants and source pairs | AB06–AB14, 27 candidate treadmill IMU/force pairs (three per participant) |
| Candidate same-trial IMU/force pairs | 1 |
| Measurement fields, excluding `Header` | IMU: 24; treadmill force-platform: 18 |
| Median `Header` increments | IMU: 0.005; force platform: 0.001 (source-time units) |
| Source timebase audit | 27/27 pairs readable with positive native-time overlap; not an event-level synchronization validation |
| Independent force-contact labels and final Test metrics | Not yet generated |

The source pairing is **not yet a validated force-contact reference**. No eligible full-cohort count, force-contact performance or participant-level timing error has been established.

Stage 01 results are now saved and tracked; later stages will produce their own files when the required sources and scientific contracts have been verified:

| Evidence | Intended current output |
| --- | --- |
| Executed source inspection (nine participants, 27 pairs) | [`01_trial_inventory.csv`](output/01_trial_inventory.csv), [`01_candidate_pairs.csv`](output/01_candidate_pairs.csv), [`01_observed_columns.csv`](output/01_observed_columns.csv), [`01_trial_timebase_audit.csv`](output/01_trial_timebase_audit.csv), [`01_source_eligibility_summary.csv`](output/01_source_eligibility_summary.csv) and [recorded signal figure](output/01_sensor_source_preview.png) |
| Executed source-contract inspection (reference pending) | [`02_source_contract_review.csv`](output/02_source_contract_review.csv), [`02_source_contract_diagnostic.png`](output/02_source_contract_diagnostic.png) |
| Force-reference eligibility and original contact events | `02_eligible_trials.csv`, `02_reference_events.csv` |
| Reference-only Development stride measures | `03_reference_trial_metrics.csv`, `03_subject_condition_metrics.csv` |
| Independent Development comparator | `04_development_participant_metrics.csv`, `04_selected_configuration.csv` |
| Frozen Test outcome | `05_test_participant_metrics.csv`, `05_test_model_summary.csv` |

Any result tables will be interpreted alongside event matching support and source exclusions; an artificially low timing MAE obtained by predicting only a few events is not a successful detector.

## 4. Running and Reviewing the Study

**Import policy:** All Python imports in the five Gait notebooks are grouped exclusively in each Notebook's **01. Imports and Research Environment** code cell. The preliminary **00. Colab Repository Setup** cell uses the IPython shell for checkout and contains no Python imports. A dedicated offline test enforces this rule for future edits.

**Use Google Colab:** Open [Notebook 01](code/notebooks/01_source_and_reference_eligibility.ipynb), run **00. Colab Repository Setup** and **01. Imports and Research Environment**, then execute in order. The repository uses the established `setup_colab("gait_analysis")` helper.

The original recordings are **not distributed with this repository**. The original AB06 source pilot was expanded to the complete selected **AB06–AB14** subset. All refreshed `01_*.csv` files are now committed and match the latest executed source-audit notebook; original MAT files remain local and are not included in GitHub. The revised Colab Free acquisition workflow targets the **nine separately available AB06–AB14 participant archives** in [Mendeley Part 1, Version 1](https://data.mendeley.com/datasets/fcgm3chfff/1), not all 22 participants and not Part 1 Version 2's 9.38-GB combined archive. The selected nine-person acquisition is **completed: nine of nine source-paired people**, with three paired treadmill trials per person. This does not establish independent reference eligibility.

**Notebook 01, Section 02, prepares the source:** the default bounded workflow downloads **one subject ZIP at a time** and up to **three new participants per cell execution**, extracts at most **three original paired treadmill trials per subject**, and removes only archives newly downloaded and successfully extracted in that execution. Existing source files are reused; rerun Section 02 until the desired nine-person source sample is prepared, then rerun Sections 03–07. The full nine-person acquisition still transfers roughly 9 GB over time, but avoids simultaneous storage of all full ZIPs. Its progress report is descriptive, not a fixed-count scientific eligibility check. Use the same Colab runtime for downstream notebooks, as temporary original data are lost when the runtime is reset. The official API must be accessible; no source data are copied into GitHub.

Notebook 01 then inventories the actual source columns and displays a genuine paired trial. This pilot **does not establish a valid shared sensor clock, foot side, vertical-force sign or final cohort**. Notebook 01 installs the pinned `mat-io==1.0.2` dependency when needed, allowing its source adapter to read MATLAB table/MCOS objects as named tabular data rather than treating the object as a numeric matrix. SciPy remains the reader for simple numeric MATLAB structures. The original `Header` time column and channel names are preserved; actual sensor side, physical units and inter-device timing still need source-based review.

**Notebook 02 has an intentional source-contract boundary:** Section 02 has executed, saving `02_source_contract_review.csv` and a recorded two-panel `02_source_contract_diagnostic.png`. The Force signal is present, but its definitive physical mapping and event-reference contract are unresolved. With unresolved gyroscope/vertical-force mapping, sign, foot side, physical timestamp units or event-level timing, Sections 03–05 are **deferred without generating contact-event or participant-role results**. They run only after independent source-contract verification. Similarly, Notebook 04 cannot run until Notebook 02 has established enough independent eligible people for grouped Development folds. These are scientific validity requirements, not approval/status-file gates.

**Research output download:** Every Notebook now ends with a **Research Summary and Output Download** cell. It packages only that notebook's saved `01_`–`05_` outputs as `/content/gait_analysis_Notebook_XX_Outputs.zip` (where `XX` is the notebook number). Retrieve the ZIP from **Colab Contents / Files**; packaging does not trigger GitHub commit/push or include raw Camargo archives. The Notebook 01 source plot displays observed `foot_Gyro_Y` / `shank_Gyro_Y` and `Treadmill_R_vy` / `Treadmill_L_vy` channels when present. The waveform figure is source evidence, **not** an independently validated vertical-force contact reference.

Notebooks 03–05 consume explicit current results from their producers. Original raw recordings remain local; the study's useful CSV tables/PNG charts are written under `gait_analysis/output/` and numeric per-trial caches under `data/processed/camargo/paired_trials/`. Execution overwrites each stage's named current outputs; a material source or protocol change requires affected downstream outputs to be recomputed.

Use the repository's existing [dependencies](../requirements.txt) and the [data access guide](../docs/DATASETS.md) for original-source instructions. The original-data **Notebook 01 AB06–AB14 inventory and 27-pair timebase audit have executed**, and all six `01_` outputs are published. Notebook 02's recorded diagnostic and source-contract table have also executed, but no force-contact labels or participant roles are yet validated; the full source-to-Test experiment has **not** been validated. Under the existing 20% person-disjoint split, nine eligible subjects would give approximately seven Development and two Test people; this is an **exploratory within-dataset pilot**, with substantial Test uncertainty. Do not claim terrain transfer from the treadmill-only sample.

## 5. Files and Saved Outputs

| Location | Responsibility |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five numbered Colab/Jupyter scientific stages |
| [`code/src/`](code/src/) | Source inventory, timebase/force pairing, reference events, detectors and metrics |
| [`images/study_overview.svg`](images/study_overview.svg) | Study-wide editable five-band method diagram |
| [`images/`](images/) | Five additional stage-specific editable SVGs |
| [`output/`](output/) | Six Notebook 01 source-audit files and two Notebook 02 source-contract diagnostics; downstream stages when validated |
| `../data/raw/camargo/` | Original participant files (local or mounted; not tracked) |
| `../data/processed/camargo/` | Reusable numeric paired-trial arrays (local or mounted) |
| [Scientific methods document](GAIT_RESEARCH_METHODS_AND_EVIDENCE.md) | Full scientific design, model interfaces, evaluation and limits |

No Excel stage tracker, administrative approval document or mandatory history/manifest file is part of this study.

## 6. Interpretation and Scope

The programmed research question is **how accurately a simple, source-aligned IMU detector can estimate force-referenced initial contacts for unseen participants**. Measurement must use the same trial, genuine physical time and a validated foot/force assignment. The principal result will be a participant-balanced timing error **together with missed/extra event counts and event sensitivity**.

The analysed population comes from healthy volunteers. This workflow does **not** establish clinical gait validity, fall-risk prediction, pathological gait detection, online inference or deployment. The filtering in the initial detectors is zero-phase and therefore **offline**. Generalisation across terrains is not claimed unless actual eligible source trials support comparisons.

If independent original-time references cannot be constructed, the appropriate next step is to revisit the research question **before** interpreting IMU predictions as contact-timing accuracy. Read [Research Methods and Evidence](GAIT_RESEARCH_METHODS_AND_EVIDENCE.md) for the complete scientific contract and pending original-data validation.

**Source attribution:** Camargo, J., Ramanathan, A., Flanagan, W. and Young, A. (2021), *Journal of Biomechanics*, 119, 110320, [DOI](https://doi.org/10.1016/j.jbiomech.2021.110320); [dataset Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2).
