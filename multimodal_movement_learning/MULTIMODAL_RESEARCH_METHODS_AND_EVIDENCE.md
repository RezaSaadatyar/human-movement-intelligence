# Multimodal Movement Learning: Research Methods and Evidence

**Current status:** A source-reviewed research protocol for NinaPro DB2, not yet an implemented or executed study. The [README](README.md) gives navigation and entry points; this document defines the proposed scientific analysis, data contracts, model comparisons and reporting boundaries. It does not contain project-generated results.

## 1. Research Objective and Scope

**Working title:** *Do Paired Accelerometer Signals Add Value to sEMG for Cross-Participant Hand-Movement Recognition?*

**Primary question:** For the same source-defined hand movements, participant groups, aligned recording intervals and evaluation windows, does adding the Delsys accelerometer measurements improve macro F1 compared with the better of sEMG-only and accelerometer-only predictors when evaluated on unseen people?

- **H1:** Feature-level fusion may improve average participant-level macro F1 over the stronger unimodal predictor.
- **H2:** Any observed benefit may vary by person/class and may not persist with missing or corrupted acceleration data.
- **H3 (secondary):** Reducing the number of accelerometer sensors may retain useful complementary information.
- **Null:** Fusion offers no reliable gain under the common participant-disjoint protocol.

The intended contribution is a reproducible, controlled *comparison* of two modalities and their fusion under a person-independent split, rather than a novel network architecture or a claimed prosthetic controller. A negative result is valid. No other study's trained model, subject counts, label mapping or evaluation numbers are transferred into this one.

**Within scope:** Source audit, synchronised paired-window extraction, non-force prompted hand gestures plus rest, compact classical classifiers, early/late fusion, modality ablation, per-participant evaluation and measured computational cost.

**Outside scope:** Glove/force prediction, video prompts as model inputs, real-time prosthetic control, amputee/stroke clinical outcomes, cross-device external validation, large attention architectures and experiments using DB5/DB3 as interchangeable evaluation data.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Original sources and availability

- [NinaPro DB2 source and subject ZIP directory](https://ninapro.hevs.ch/instructions/DB2.html): 40 intact participants, 49 prompted movements plus rest across basic finger, grasp/functional and force exercises. It distributes one MATLAB file per exercise per participant in the subject archive.
- Atzori, M. et al. (2014), ‘[Electromyography data for non-invasive naturally-controlled robotic hand prostheses](https://doi.org/10.1038/sdata.2014.53)’, *Scientific Data*, 1, 140053. The paper describes acquisition, synchronisation, filtering and relabelling.
- [Original Dryad data deposition](https://doi.org/10.5061/dryad.1k84r) provides separate DB2 subject ZIPs (subject 1 approximately 470 MB). The deposited 23.27 GB total is for DB1–DB3 jointly, **not** DB2 alone. Dryad normally distributes its published datasets under CC0, but the usage/distribution conditions applying to the selected official NinaPro files should be checked before redistributing any data.

The official web page has inconsistent text (‘10 repetitions’ in one overview passage, ‘6 repetitions’ in the detailed protocol). Use six as a source-supported **planning value**, then inventory the exercise files before any fixed class/repetition assertions. The initial official data sample has **not** been downloaded or opened.

### 2.2 Sensing and original signal processing

| Variable (official dictionary) | Documented role | Proposed use |
| --- | --- | --- |
| `emg` — 12 channels | Delsys Trigno sEMG, acquired at 2,000 Hz | Primary physiological input |
| `acc` — 36 channels | Three acceleration axes per each of 12 Trigno sensors; native accelerometer acquisition at approximately 148 Hz | Second input; not 36 independent devices and not a gyroscope |
| `stimulus` / `restimulus` | Displayed-cue label / retrospectively refined movement-boundary label | Annotation source; provisional target is validated `restimulus` |
| `repetition` / `rerepetition` | Original / refined repetition identifiers | Segment identity and audit only |
| `glove`, `force` | Kinematic and force measurements | Excluded from predictive features |

The original data descriptor reports synchronisation of released streams through interpolation to the highest rate and interference cleaning of Delsys sEMG before release. Consequently an accelerometer column stored on a 2,000-Hz row grid is **not** evidence of 2,000 independent accelerometer readings per second. Record whether released `emg`, `acc` and labels have exactly matched row counts, and do not apply a default second 50-Hz filter without evidence it is needed.

Corrected movement boundaries were generated using a procedure involving the EMG data. The final protocol must cite that annotation provenance, since it can be relevant to comparative performance of the two sensing modalities. Neither `restimulus` nor any cue/repetition field may enter predictors.

### 2.3 Proposed inclusion cohort and class definition

**Primary task proposal for a compact first release:** the complete set of source-labelled gestures in the **basic finger-movement exercise (the first non-force exercise)** plus Rest, rather than selecting favourable gestures after training. The exact `exercise` identifier, per-exercise label codes, `restimulus` transitions and valid repetitions are to be frozen following Notebook 01's genuine source audit. The second non-force exercise (grasping/functional movement) is a separately reported extension, not silently pooled into the primary task.

Subjects will be considered eligible according to source availability, valid paired `emg`/`acc` streams and agreed class support, without reference to any model scores. Report the number of source people, eligible people, repetitions and windows after inspection; the 40-person number is an **original release count**, not a measured final modelling cohort.

**Unit hierarchy:** `subject_id` → `exercise_id` → `repetition_id` / `trial_key` → source sample interval → aligned window. Rest intervals require a stable source-based segment identity and cannot be pooled across different people's recordings. Independently identifying actual motion onset using future Test labels is not permitted.

## 3. Data Preparation and Representations

### 3.1 Source archive, MATLAB and pairing contract

Notebook 01 will download or locate a single official pilot subject archive, then check archive integrity and safe extraction, MATLAB keys, shapes, finite values, source pre-processing and the supplied label dictionary. Use the real released `emg` and `acc` arrays, not independently recorded or class-name-matched streams.

The intended trial/window registry contains: `subject_id`, `exercise_id`, `source_file`, `repetition_id`, `original_label`, `corrected_label`, `segment_id`, source sample `start/end`, physical-time `start_s/end_s`, `emg/acc` coverage, source rate assumptions and exclusion reason where relevant. Confirm which variables have a common stored row/time grid and whether native timestamps are actually retained. Never invent timestamps absent from source files.

### 3.2 Windowing and sensor features

**Initial authoring default (to be verified on source data):** 300-ms windows stepped every 100 ms, with full within-segment label consistency, no crossing between exercises, subjects, repetitions, corrected class transitions or source gaps. Alternative 200-ms/400-ms durations are Development-only sensitivity analyses. Avoid treating overlapping windows as independent evaluation subjects.

Signals that the distributed archive already synchronised may be windowed on the **released common time grid**, with time in seconds derived from verified stored sampling metadata. Inspect raw/native rate assumptions rather than independently reinterpolating one stream. For accelerometer statistical features, avoid frequency claims exceeding the native 148-Hz acquisition bandwidth.

Proposed lightweight features, computed separately per eligible paired window:

- **sEMG:** per-channel RMS, mean absolute value, waveform length and a log-variance or stable amplitude-dispersion feature (approximately 48 summary features from 12 channels).
- **Accelerometry:** per-axis mean and standard deviation (72 features for 36 axes), plus two summary features of each three-axis sensor magnitude when justified (approximately 24 additional features).
- **Early fusion:** concatenate the two feature vectors from the **same window key**. Count exact columns after real extraction; reject missing/misordered source channels rather than silently filling them.

Train-only per-feature scaling, class weights and any imputation belong inside an estimator pipeline fitted independently within each Development fold. No global per-subject feature normalisation using held-out Test signals unless explicitly defined as a separate unlabelled adaptation condition. Preserve raw source arrays locally, and store smaller row-keyed feature matrices under `data/processed/ninapro_db2/` rather than adding large derived datasets to Git history.

## 4. Experimental Design and Evaluation

### 4.1 Participant independence and frozen evaluation

After the complete pre-modelling eligibility audit, select the reserved Test participants deterministically (proposed `random_state=42`, five people when the expected 40 are usable), record the exact IDs, and never use their windows for feature/scaler fitting, hyperparameter selection or fusion weighting. Use all remaining eligible participants for Development-only five-fold grouped evaluation when feasible.

The plan of 35 Development and five Test participants is **conditional on source eligibility**, not an observed split. The Test participant IDs and the exact eligible task classes must be declared and recorded before fitting and before viewing Test predictions. If subject/class coverage makes the proposed design unsuitable, revise the plan using source feasibility information before any Test evaluation.

**Primary selection:** compare all candidate arms on the same eligible Development windows and the same participant-grouped folds. Choose models and fusion combination using Development evidence alone. A separate independent Validation split is not required if fold-based model selection and inner-Development calibration are used honestly. For tuned models, keep tuning and probability calibration inside the corresponding Training partition; never fit a scaler or calibrator on an evaluation fold's labels.

### 4.2 Endpoint definitions

- **Primary:** calculate fixed-class macro F1 **separately for each independent Test participant**, then average these per-person values equally.
- **Main comparison:** for each Test participant, compare the frozen fused method with the **Development-selected** stronger unimodal comparator, using the exact same paired windows. The choice of comparator may not be switched using Test results.
- **Secondary:** fixed-class balanced accuracy, per-class precision/recall/support, per-participant confusion matrices, coverage/exclusions, and absolute/relative fusion differences.
- **Uncertainty:** descriptive paired participant bootstrap over reserved independent Test people, with interval interpretation limited by their small number. Do not treat thousands of overlapping windows as independent bootstrap units.
- **Computation:** measured fitting time, per-window CPU inference time, model size, selected window length and modality availability.

No reported metric is available yet. Balanced accuracy over missing classes, zero-support class F1 and rest prevalence must follow explicit, consistent definitions and include support tables.

## 5. Methods, Models and Configurations

### 5.1 Matched baseline and fusion arms

| Arm | Predictor and scientific purpose | Configuration principle |
| --- | --- | --- |
| B0 | Training-prior / majority-class classifier | Sanity lower bound, predicts on same eligible windows |
| B1 | sEMG-only feature classifier | Regularised multinomial logistic regression; simple Random Forest as optional matched alternate |
| B2 | Accelerometer-only feature classifier | Same estimator family and Development tuning budget as B1 |
| F1 | Early fusion | Concatenate B1/B2 features and fit a matched classifier, no cross-person input leakage |
| F2 | Late probability fusion | Blend two independently trained, properly aligned class-probability vectors using fixed or Development-selected weights |

A small regularisation grid for logistic regression (e.g. `C=0.1,1,10`) and a bounded Random Forest option can be used **only in Development**. These are planning defaults rather than fitted values. Begin with the lightest interpretable baseline and do not impose new neural models on the five-notebook MVP. If probability calibration is needed, fit it using Training-only inner folds; class probability order must match the fixed class list.

### 5.2 Modality ablation and failure handling

Compare removal of either modality, contiguous missing-acceleration intervals, one-axis corruption and a reduced sensor subset. Distinguish *simulated* missingness from missing source data. Do not silently use zero-valued vectors as valid missing-modality inference unless the model has an explicitly trained and documented missing-input policy.

All arms must share the same subject and window keys for an interpretable fusion comparison. Source label correction, filtered sEMG, window purity and grouping controls apply equally across arms.

### 5.3 Implementation ownership

Reuse existing `python_modules/core/` functions only after inspecting their interface and testing compatibility with DB2: subject-disjoint splitting, grouped metrics, `plot_style`, `save_figure`, Colab setup, device selection and output saving. The existing HARTH windowing helpers were developed for a different rate and label taxonomy; they are **candidates**, not authorised cross-study conversions.

Add a small `code/src/` reader, synchronised-window/feature module and compact model-evaluation helpers only when Notebook 01–04 actually require them. Implement limited synthetic checks for row/key alignment, no subject overlap, train-only feature fitting, class/probability order and reproducible output replacement. Preserve existing shared tests and studies.

## 6. Results and Comparative Analysis

**No project results have been produced.** The intended comparisons are Development-only per-person macro F1 of B0/B1/B2/F1/F2, followed by one held-out person-independent Test comparison of preselected arms. The final figure should show per-person fused-minus-unimodal differences rather than a single selectively favourable pooled score.

Only results from executed compatible code and the associated current `output/` tables will be reported. For context only: [Hu et al. (2024)](https://doi.org/10.3390/s24092702) reported a DB2 fusion accuracy for a **within-person repetition-split design**, which is not directly comparable to this planned person-independent macro-F1 endpoint. [Zhang et al. (2023)](https://doi.org/10.1109/JSEN.2023.3266872) published an intersubject sEMG/accelerometer fusion approach; its existence precludes a novelty claim that intersubject fusion itself is unexplored.

## 7. Robustness, Uncertainty and Additional Analyses

**Planned, not executed:** 200-/400-ms window sensitivity; missing-acceleration segments, one-axis/sensor removal and compact acceleration sensor selection; per-class/person disagreement; training and inference cost. Fused and unimodal performance will be reported under the same Development folds and paired Test windows.

Five held-out Test people would provide only a small cluster-bootstrap sample. Any reported interval will be descriptive, not a claim that an effect generalises across all people, hand tasks or devices. If fusion fails or performs worse for some people, preserve those results.

An optional second non-force exercise may be studied as an explicitly separate task after the primary configuration and results have been recorded; it will not be selected because its Test result appears more favourable.

## 8. Current Outputs and Reproducibility

### 8.1 Planned notebook sequence and file contracts

The following paths are **owned output targets**, not files that currently exist.

| Notebook | Required input and operations | Named output targets |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Official pilot ZIP / MATLAB fields, sensor and label schema, source alignment and missingness | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Validated source inventory; label, time and pairing contracts; trial-safe windows and features | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local aligned arrays/features |
| `03_unimodal_grouped_baselines.ipynb` | Frozen eligible class list, feature matrices and Test isolation; B0/B1/B2 Training CV | `03_dev_unimodal_metrics.csv`, `03_oof_predictions.csv`, `03_class_errors.png` |
| `04_fusion_and_modality_ablations.ipynb` | Same windows/folds as 03; F1/F2 and perturbation evaluation on Development | `04_dev_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_comparison.png`, `04_modality_sensitivity.png` |
| `05_frozen_person_independent_test.ipynb` | Fixed settings; all Development people for final fit; reserved Test people for a single comparison | `05_test_predictions.csv`, `05_test_metrics.csv`, `05_subject_differences.png`, `05_cost_accuracy.png` |

Notebook-specific scientific sections anticipated:

- **01:** source download/inventory; MATLAB field audit; EMG–Acc pairing; original/corrected annotations; participant and repetition coverage; source evidence and interpretation.
- **02:** physical-time mapping; signal quality; label transitions and purity; paired windows; feature extraction; eligible counts and frozen split IDs; saved row keys.
- **03:** Development split integrity; train-only scalers and class weighting; majority/logistic regression and optionally RF; per-fold/participant metrics and OOF predictions; error interpretation.
- **04:** early fusion; probability order/alignment; late fusion weights and any inner-fold calibration; matched comparisons; sensor and dropout ablations; Development-only selection.
- **05:** read frozen Development decision; fit specified final models; apply to reserved people once; paired model/subject metrics; cluster-bootstrap uncertainty and compute metrics; concise scope conclusion.

Each implemented notebook will contain the established **00. Colab Repository Setup** and **01. Imports and Research Environment** sections, numbered scientific Markdown and code headings, and an independently curated SVG whose filename matches the notebook stem. SVGs explain actual implemented operations; no Test scores or empirical waveforms will be invented in a planning diagram.

### 8.2 Local and tracked artefact boundaries

- Local-only `data/raw/ninapro_db2/`: original ZIP archives and subject MATLAB files.
- Local-only `data/processed/ninapro_db2/`: row-keyed aligned arrays, feature matrices and compact pairing intermediates.
- Study-owned `output/`: current public CSV evidence and numerical plots **only after execution**.
- Curated `images/`: explanatory SVGs created when the corresponding notebook is implemented.
- Repository root `requirements.txt`: existing general dependencies; compatibility with source loading and runtime to be checked during implementation.

Pilot one person, inspect feasibility, then stream all eligible people; do not load all source ZIPs into memory at once. The Dryad listing shows single-person ZIPs of several hundred megabytes. Existing local data and outputs should be reused when scientifically compatible, with each notebook replacing its own current named outputs upon rerun.

## 9. Discussion, Limitations and Future Work

The study is planned, so no conclusions about EMG-only, acceleration-only or fusion performance can yet be drawn. The original healthy-participant, prompted-movement protocol is a controlled sensor classification task, not a demonstration of prosthetic control or rehabilitation impact. Any subsequent estimate will concern the source-defined labels, signal representations, participants and recording conditions actually evaluated.

Source priorities for Notebook 01 are accessible official ZIPs, confirmed MATLAB variable names and shapes, alignment after released preprocessing, the class/repetition dictionary, the provenance of corrected labels and the terms of using/distributing the original data. A pilot may reveal practical changes needed before fixing the full participant split; such changes should follow actual evidence, not a Test score.

**Key references:** [NinaPro DB2 guide](https://ninapro.hevs.ch/instructions/DB2.html); [Atzori et al. (2014)](https://doi.org/10.1038/sdata.2014.53); [Dryad source record](https://doi.org/10.5061/dryad.1k84r); [Hu et al. (2024)](https://doi.org/10.3390/s24092702); [Zhang et al. (2023)](https://doi.org/10.1109/JSEN.2023.3266872).
