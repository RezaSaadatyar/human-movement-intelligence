# Gait Analysis | Research Methods and Evidence

This document defines the original-source, analytical and evaluation contract for the Camargo gait-event study. It is maintained alongside the [study README](README.md), [five notebooks](code/notebooks/) and [editable study workflow](images/study_overview.svg). **Implementation is present; original-data event-timing results are not yet available.** Text below describes the approved design and current code without asserting that any source-dependent notebook has been executed.

## 1. Research Objective and Scope

**Question:** To what degree can a parsimonious inertial detector estimate foot Initial Contact (IC) timing and stride timing relative to an independent, same-trial force reference in walking conditions?

**Primary endpoint:** participant-mean matched-event absolute IC timing error in **milliseconds**, reported with event sensitivity, precision and the number of unmatched reference/detected events. Low MAE alone does not establish event-detection quality. **Secondary:** reference-derived stride time, cadence and intra-person variability; timing-bias and condition-dependent performance where eligible.

The method contrasts a fixed IMU peak detector, a Development-tuned rule-based detector and their paired performance on independent people. The study does not inherit HARTH labels, Essex FE model targets or NinaPro hand-movement results.

Hypotheses remain prospective: event error may vary with speed and locomotor context; a tuned detector may or may not outperform a fixed rule; variability may prove more sensitive than average stride timing. Negative or null findings are reportable. Clinical interpretation, inverse dynamics estimated from IMUs and fall-risk prediction are out of scope.

## 2. Data Sources, Cohort and Target Definitions

The primary source is the [Camargo et al. (2021) dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) ([published article](https://doi.org/10.1016/j.jbiomech.2021.110320); [Mendeley Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2), CC BY 4.0). The reported source cohort consists of **22 healthy participants**, with treadmill, level-ground, ramp and stair conditions. Original recordings include IMU, force plates, EMG and biomechanics measurements; this specific estimator uses only **IMU gyroscope signals**, while the independently measured force serves as the **evaluation reference**, not a predictor.

The published source describes approximately **200-Hz IMU** and **1,000-Hz force-plate** data. The actual timestamp column, time units, source clock relationship, exact vertical-force axis, side and sign must be established from **the original reviewed files**. Different sampling rates alone do not preclude a valid comparison, but arbitrary resampling or phase alignment is not evidence of synchronisation.

The data hierarchy is **participant → session → locomotor condition → trial → individual contact → predicted event**. Person is the independent evaluation unit. The full original count of 22 is **not** an assertion that every person has eligible paired reference records. The eligibility and actual retained count will be reported from source-derived `01_*` and `02_*` files once executed.

## 3. Data Preparation and Representations

### Original source and reviewed channels

[Source inspection](code/notebooks/01_source_and_reference_eligibility.ipynb) enumerates participant directories and sensors, preserves original relative file paths and audits numeric channel columns and timestamp monotonicity. The [source adapter](code/src/camargo_source.py) reads original CSV/Parquet and simple numeric MATLAB dictionaries through SciPy. For MATLAB table/MCOS objects, Notebook 01 and Notebook 02 install the pinned `mat-io==1.0.2` reader when needed; the adapter accepts only an unambiguous decoded table containing the original numeric `Header` or `time_s` column and retains its actual channel names. Source-format conversion has been tested with synthetic MATLAB structures and simulated decoded tables, but decoding the original Camargo MATLAB tables remains subject to the first source-data execution.

[Alignment](code/src/gait_alignment.py) compares actual IMU/force time domains. This step requires explicit assignment of the gyroscope channel, vertical-force column, foot side, upward force sign and timestamp unit. It does **not** select a time offset by maximising IMU/Test agreement. Source pairing and eligible trials are recorded in `02_eligible_trials.csv`; failed/ambiguous trials in `02_exclusions.csv`.

### Independent force-contact reference

[Force reference extraction](code/src/gait_reference.py) applies a documented vertical-force threshold. The initial proposed value is **20 N**, with minimum continuous-contact **0.08 s** and swing-gap **0.06 s**; these are **study planning settings** and must be reviewed against observed raw force signs and sample rates before scientific use. Changes belong to Development/source analysis, not final Test outcomes.

The reference is defined from the original force signal, not from an IMU peak. Partial edge contacts and ambiguous belt/foot correspondence are excluded from timing-error scoring rather than silently assigned. IC and TC correspond to threshold entry and exit respectively; they are **derived contact estimates**, not directly observed neurological or anatomical events. When available, a valid IC-to-IC interval provides a same-foot stride; stride cadence uses two steps per stride, `120 / stride_time_seconds`.

### IMU representation

A single reviewed foot or shank gyroscope axis is used in the first interpretable detector. Offline fourth-order Butterworth low-pass filtering with **10 Hz initial cutoff** and `sosfiltfilt` produces a zero-phase version for peak analysis. This is **non-causal**, so inference speed and online use are not claimed. The retained original-time windows must be finite, monotonic and free from unmarked sample gaps. Source arrays are cached as `imu_time_s`, `gyro` and `reference_ic_s` keyed by original participant/trial.

## 4. Experimental Design and Evaluation

### Person-disjoint partitions

After the **source-eligible cohort** is determined, [Notebook 02](code/notebooks/02_alignment_and_event_reference.ipynb) uses deterministic seed **42** to reserve approximately 20% of eligible participants for a future final Test, requiring enough Development people to form five person-disjoint folds. This is not a guaranteed count or a retrospectively chosen five-person holdout.

[Notebook 03](code/notebooks/03_stride_measurement_and_statistics.ipynb) describes reference-stride properties only for Development people. [Notebook 04](code/notebooks/04_imu_event_baselines_and_grouped_validation.ipynb) evaluates B0 and B1 in the same **five grouped outer Development folds**; B1's parameter choices within each fold use only the other Development people. The final B1 recipe is then selected from all Development data, and `04_selected_configuration.csv` freezes the winner before Test. [Notebook 05](code/notebooks/05_frozen_test_and_research_synthesis.ipynb) reads that exact configuration and scores reserved Test participants; no Test-based retuning is permitted.

### Event matching and score definitions

The [evaluation module](code/src/gait_evaluation.py) uses **one-to-one matching** of force-reference and IMU-predicted IC timestamps, restricted to a predeclared **150-ms tolerance**. The tolerance is a protocol choice, not an experimentally demonstrated optimum. Matched-pair timing error is `1000 × (estimated_s − reference_s)`; its absolute mean is in milliseconds. Unmatched reference contacts are **misses**, unmatched estimates are **extras**, and no matched events produce undefined MAE rather than zero.

For each trial, report `n_reference`, `n_predicted`, `n_matched`, `n_missed`, `n_extra`, `timing_mae_ms`, `sensitivity` and `precision`. Participant summary combines matched timing errors weighted by each person's trial-level matched counts; the headline result averages participants **equally**. The Development selection score adds an explicit penalty for low sensitivity/precision to avoid a detector obtaining a favourable MAE by predicting very few events.

The Test paired bootstrap samples **whole participants**, not events or overlapping signal rows, using **2,000 draws** with seed 42. Any resulting interval is descriptive, conditional on the fitted models and the selected people.

## 5. Methods, Models and Configurations

| Identifier | Implemented method | Parameters and role |
| --- | --- | --- |
| `B0_fixed` | Transparent gyroscope peak detector | Positive polarity, 0.5 source-unit prominence, 0.45-s refractory interval, 10-Hz offline low-pass; no Test-tuned parameters |
| `B1_tuned` | Same detector family, tuned within Development | Polarity ±1 and prominence grid `[0.3, 0.6, 1.0, 1.5]`; fixed 0.45-s interval and 10-Hz cutoff; grouped Training-only selection |
| Optional learned candidate | Random Forest candidate-peak scorer | A 120-tree implementation exists in [detectors](code/src/gait_event_detectors.py); this arm is **not** an approved mandatory benchmark or executed result and is not added to final selection automatically |

The compact model comparison is deliberately limited. More complex TCN models or other sensor placements may be justified later if actual Development errors show a systematic question that simpler detectors cannot answer. No pretrained model or external-study accuracy is transferred.

The study-specific source, alignment, reference, detector, experiment and evaluation functions reside in [`code/src/`](code/src/), while the repository's approved `setup_colab` and plotting helpers remain shared. No extra stage tracker, approval flags or performance manifest is required.

## 6. Results and Comparative Analysis

**No original-data model result is currently reported.** The five notebooks and their output interfaces have been authored, but the actual eligible source cohort, independent reference event counts and participant-level timing metrics are not yet established.

After original-data execution, the mandatory evidence will include: the original [Notebook 01](code/notebooks/01_source_and_reference_eligibility.ipynb) file/channel inventory; [Notebook 02](code/notebooks/02_alignment_and_event_reference.ipynb) verified contact events and exclusions; [Notebook 03](code/notebooks/03_stride_measurement_and_statistics.ipynb) reference gait profiles; [Notebook 04](code/notebooks/04_imu_event_baselines_and_grouped_validation.ipynb) held-out Development participant timing and event coverage; and [Notebook 05](code/notebooks/05_frozen_test_and_research_synthesis.ipynb) final frozen participant results.

No result chart or numerical mean is inserted here until backed by saved source-derived CSVs. The [study diagram](images/study_overview.svg) is a **methodological schematic**, not a measured performance chart.

## 7. Robustness, Uncertainty and Additional Analyses

Planned analyses dependent on reference eligibility include condition-specific speed/terrain breakdowns, signed timing bias, the number of missed/extra events, source-gap exclusions and participant variability. The initial full comparison keeps the **treadmill context separate** from overground and terrain transitions; conditions cannot be pooled without a documented matched source and reference definition.

The present code provides participant-level uncertainty rather than incorrectly treating the much larger number of steps as statistically independent people. Timing tolerance and force threshold sensitivity analyses must be decided within Development; retrospective adjustment to improve Test outcomes would invalidate claims of independent Test model selection.

No robustness result, superiority interval, clinical generalisability or actual computational cost has been measured in the new study.

## 8. Current Outputs and Reproducibility

| Stage | Current-file output interface (produced only after execution) |
| --- | --- |
| 01 | `01_trial_inventory.csv`, `01_candidate_pairs.csv`, `01_observed_columns.csv`, `01_sensor_source_preview.png` |
| 02 | `02_eligible_trials.csv`, `02_exclusions.csv`, `02_reference_events.csv`, `02_participant_roles.csv`, `02_development_folds.csv` |
| 03 | `03_reference_strides.csv`, `03_reference_trial_metrics.csv`, `03_subject_condition_metrics.csv` |
| 04 | `04_development_trial_metrics.csv`, `04_development_participant_metrics.csv`, `04_selected_configuration.csv` |
| 05 | `05_test_trial_metrics.csv`, `05_test_participant_metrics.csv`, `05_test_model_summary.csv`, `05_paired_participant_uncertainty.csv` |

The study's reproducible method scripts and notebooks are tracked; original source data are obtained from [Georgia Tech](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/). The repository-root `data/raw/camargo/` contains local original files when provided, while `data/processed/camargo/paired_trials/` holds numeric trial caches that the source-aware Notebook 02 produces. Small numerical tables and figures belong in the study's `output/`.

The first run should inspect **one original ABxx participant**, not automatically acquire all Mendeley repository parts. Follow the [README execution guidance](README.md#4-running-and-reviewing-the-study). Stage-to-stage files are explicit and overwritten by their owning notebook; a missing file produces a clear instruction to execute its producer. Complete original-data reproduction and any source-specific MATLAB conversion remain to be validated against observed field layouts.

## 9. Discussion, Limitations and Future Work

The current contribution is a defined, staged **implementation and evaluation design**, not yet a measured gait-event accuracy result. Its central scientific dependency is a **genuinely independent, native-time aligned force contact reference** for the exact IMU trial/side. Camargo's published sensor inventory alone does not guarantee every trial satisfies this requirement.

The source population consists of **healthy volunteers**, the first proposed setting is offline treadmill walking, and the current rule filtering is non-causal. Therefore this study cannot assert diagnosis, neurological rehabilitation benefit, fall-risk forecasting, real-time operation or external clinical validity. Event timings derived from force thresholds are reference estimates with operational definitions, not universally unambiguous anatomical heel strikes.

If appropriate force-reference pairing is absent from the accessible original records, the design must be revised before proceeding with millisecond IMU-versus-force evaluation. A possible separate study of **force-reference-derived stride variability across conditions** would be a changed primary question requiring explicit approval, not an automatic substitution for failed timing validation.

**Source references:** Camargo, J. et al. (2021), *A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions*, *Journal of Biomechanics*, 119, 110320, [DOI](https://doi.org/10.1016/j.jbiomech.2021.110320); [Mendeley dataset Part 1 v2](https://doi.org/10.17632/fcgm3chfff.2).
