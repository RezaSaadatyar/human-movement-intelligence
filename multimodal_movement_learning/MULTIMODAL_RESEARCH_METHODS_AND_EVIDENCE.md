# Multimodal Movement Learning: Research Methods and Evidence

**Research evidence:** Original NinaPro DB2 E1 source and cohort outputs are recorded in Notebooks 01–02. Executed Notebook 03 reports eleven unimodal Development methods; executed Notebook 04 reports nine fusion methods, paired Development differences, synthetic acceleration perturbations, neural seed repetitions and the frozen `B1_rf`/`F1_rf` comparison. **Notebook 05 has not yet produced independent Test results.**

## 1. Research Objective and Scope

**Working title:** *Do Paired Accelerometer Signals Add Value to sEMG for Cross-Participant Hand-Movement Recognition?*

**Primary question:** For the same source-defined hand movements, participant groups, aligned recording intervals and evaluation windows, does adding the Delsys accelerometer measurements improve macro F1 compared with the better of sEMG-only and accelerometer-only predictors when evaluated on unseen people?

- **H1:** Feature-level fusion may improve average participant-level macro F1 over the stronger unimodal predictor.
- **H2:** Any observed benefit may vary by person/class and may not persist with missing or corrupted acceleration data.
- **H3 (secondary):** Reducing the number of accelerometer sensors may retain useful complementary information.
- **Null:** Fusion offers no reliable gain under the common participant-disjoint protocol.

The intended contribution is a reproducible, controlled *comparison* of two modalities and their fusion under a person-independent split, rather than a novel network architecture or a claimed prosthetic controller. A negative result is valid. No other study's trained model, subject counts, label mapping or evaluation numbers are transferred into this one.

**Within scope:** Source audit, synchronised paired-window extraction, non-force prompted hand gestures plus rest, classical classifiers, **unimodal 1D-CNN/TCN/BiGRU**, **two-branch deep fusion, gated/cross-attention fusion and deep late fusion**, controlled modality ablations, per-participant evaluation, multi-seed robustness of selected finalists and measured computational cost.

**Outside scope:** Glove/force prediction, video prompts as model inputs, real-time prosthetic control, amputee/stroke clinical outcomes, cross-device external validation, training foundation models or unbounded architecture sweeps, and experiments using DB5/DB3 as interchangeable evaluation data. Compact attention-based fusion is explicitly within scope; pretrained large-scale language/vision Transformers are not.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Original sources and availability

- [NinaPro DB2 source and subject ZIP directory](https://ninapro.hevs.ch/instructions/DB2.html): 40 intact participants, 49 prompted movements plus rest across basic finger, grasp/functional and force exercises. It distributes one MATLAB file per exercise per participant in the subject archive.
- Atzori, M. et al. (2014), ‘[Electromyography data for non-invasive naturally-controlled robotic hand prostheses](https://doi.org/10.1038/sdata.2014.53)’, *Scientific Data*, 1, 140053. The paper describes acquisition, synchronisation, filtering and relabelling.
- [Original Dryad data deposition](https://doi.org/10.5061/dryad.1k84r) provides separate DB2 subject ZIPs (subject 1 approximately 470 MB). The deposited 23.27 GB total is for DB1–DB3 jointly, **not** DB2 alone. Dryad normally distributes its published datasets under CC0, but the usage/distribution conditions applying to the selected official NinaPro files should be checked before redistributing any data.

The official web page has inconsistent text (‘10 repetitions’ in one overview passage, ‘6 repetitions’ in the detailed protocol). The executed S01 E1 source inspection observed repetition codes **0–6**, with six corrected non-rest segments per E1 movement label. The participant archive contained three exercise MATLAB files and occupied **448.47 MiB** in the recorded environment. This observation supports the analysed E1 protocol but should not be extrapolated blindly to every exercise.

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

### 2.3 Recorded cohort and class definition

**Current primary task:** the basic finger-movement exercise **E1**, using the verified source corrected labels **0–17** (Rest plus seventeen non-rest movements). This is the complete observed E1 class set, rather than a post-hoc favourable class selection. The second non-force exercise (grasping/functional movement) is a separately reported extension, not silently pooled into the primary task.

Eligibility was determined from original source availability, paired `emg`/`acc` streams and corrected label support, without reference to model performance. The executed E1 inventory includes **40 participants** and **337,853 eligible paired source windows**. A predeclared, time-uniform cap of **1,000 windows per person** retained **40,000** for modelling. These are observed current-cohort counts, not fixed requirements for all future data releases.

**Unit hierarchy:** `subject_id` → `exercise_id` → `repetition_id` / `trial_key` → source sample interval → aligned window. Rest intervals require a stable source-based segment identity and cannot be pooled across different people's recordings. Independently identifying actual motion onset using future Test labels is not permitted.

## 3. Data Preparation and Representations

### 3.1 Source archive, MATLAB and pairing contract

Notebook 01 inspected the original S01 archive and E1 MATLAB fields. It recorded **1,808,331 common released sample rows** with `emg` shape **(1,808,331, 12)**, `acc` shape **(1,808,331, 36)**, **18 corrected label codes** and **205 contiguous corrected-label/repetition segments**. Both modality arrays were fully finite in this participant. Source fields were paired by the same published row grid, not by post-hoc class matching. The S01 nominal duration on that grid is **904.1655 seconds**; native accelerometer timestamps were not independently recovered.

The intended trial/window registry contains: `subject_id`, `exercise_id`, `source_file`, `repetition_id`, `original_label`, `corrected_label`, `segment_id`, source sample `start/end`, physical-time `start_s/end_s`, `emg/acc` coverage, source rate assumptions and exclusion reason where relevant. Confirm which variables have a common stored row/time grid and whether native timestamps are actually retained. Never invent timestamps absent from source files.

### 3.2 Windowing and sensor features

**Executed E1 windowing:** 300-ms windows stepped every 100 ms, with full within-segment label consistency and no crossing between exercises, subjects, repetitions, corrected class transitions or source gaps. Alternative 200-ms/400-ms durations are Development-only sensitivity analyses. Avoid treating overlapping windows as independent evaluation subjects.

Signals that the distributed archive already synchronised may be windowed on the **released common time grid**, with time in seconds derived from verified stored sampling metadata. Inspect raw/native rate assumptions rather than independently reinterpolating one stream. For accelerometer statistical features, avoid frequency claims exceeding the native 148-Hz acquisition bandwidth.

Implemented lightweight features, computed separately per selected paired window:

- **sEMG:** per-channel RMS, mean absolute value, normalised waveform length and log variance; **48 summary features** from the 12 channels, forming a current matrix **(40,000, 48)**.
- **Accelerometry:** per-axis mean and standard deviation (72 features for 36 axes), plus mean and standard deviation of the 12 sensor magnitudes (24 more); **96 total features**, forming a current matrix **(40,000, 96)**.
- **Early fusion:** concatenate the two feature vectors from the **same window key**. Count exact columns after real extraction; reject missing/misordered source channels rather than silently filling them.
- **Deep temporal inputs:** retain aligned signal tensors per window as `X_emg [W, T, 12]` and `X_acc [W, T, 36]` when supported by the released schema; adapt to `channels-first` within the PyTorch data interface and record actual tensor layout. The common stored row grid does not create new independent 148-Hz acceleration observations. Validate physical time correspondence and document any modality-specific downsampling or anti-alias filtering. Prevent normalisation leakage by fitting source-derived parameters inside Development folds.

Train-only per-feature scaling, class weights and any imputation belong inside an estimator pipeline fitted independently within each Development fold. No global per-subject feature normalisation using held-out Test signals unless explicitly defined as a separate unlabelled adaptation condition. Preserve raw source arrays locally, and store smaller row-keyed feature matrices under `data/processed/ninapro_db2/` rather than adding large derived datasets to Git history.

## 4. Experimental Design and Evaluation

### 4.1 Participant independence and frozen evaluation

After the completed source-cohort audit, Notebook 02 assigned participants deterministically using **`random_state=42`**. Five participants are reserved for Test and 35 for Development, with all windows of a person retaining one role. All five Development validation folds contain seven separate people. Test windows must not be used for feature/scaler fitting, hyperparameter selection or fusion weighting.

The **35 Development / five Test** split and eligible **0–17** class set are recorded in `output/02_participant_split_assignments.csv`, `02_development_fold_assignments.csv` and `01_label_dictionary.csv`. These assignments preceded the subsequently run Development comparisons. If subject/class coverage makes the proposed design unsuitable, revise the plan using source feasibility information before any Test evaluation.

**Primary selection:** compare all candidate arms on the same eligible Development windows and the same participant-grouped folds. Choose models and fusion combination using Development evidence alone. A separate independent Validation split is not required if fold-based model selection and inner-Development calibration are used honestly. For tuned models, keep tuning and probability calibration inside the corresponding Training partition; never fit a scaler or calibrator on an evaluation fold's labels.

### 4.2 Endpoint definitions

- **Primary:** calculate fixed-class macro F1 **separately for each independent Test participant**, then average these per-person values equally.
- **Main comparison:** for each Test participant, compare the frozen fused method with the **Development-selected** stronger unimodal comparator, using the exact same paired windows. The choice of comparator may not be switched using Test results.
- **Secondary:** fixed-class balanced accuracy, per-class precision/recall/support, per-participant confusion matrices, coverage/exclusions, and absolute/relative fusion differences.
- **Uncertainty:** descriptive paired participant bootstrap over reserved independent Test people, with interval interpretation limited by their small number. Do not treat thousands of overlapping windows as independent bootstrap units.
- **Computation:** measured fitting time, per-window CPU inference time, model size, selected window length and modality availability.

Notebooks 03 and 04 report participant-balanced Development macro F1 for the eleven unimodal and nine fusion arms; independent Test metrics remain unavailable. The updated notebook additionally computes per-class OOF support, precision, recall and F1 with explicit pooled-window versus participant-mean aggregation. Unsupported classes receive zero under the fixed-class participant-macro convention, and class-support counts are preserved.

## 5. Methods, Models and Configurations

### 5.1 Matched baseline and fusion arms

| Arm | Predictor and scientific purpose | Configuration principle |
| --- | --- | --- |
| B0 | Training-prior / majority-class classifier | Sanity lower bound, predicts on same eligible windows |
| B1 | sEMG-only feature classifier | Implemented Logistic Regression and 120-tree Random Forest on 48 statistical features |
| B2 | Acceleration-only feature classifier | Implemented Logistic Regression and 120-tree Random Forest on 96 statistical features |
| D1–D3 | Unimodal temporal networks | 1D-CNN, TCN and BiGRU on single-modality source windows using common subject folds |
| F1 | Early fusion | Concatenate B1/B2 features and fit a matched classifier, no cross-person input leakage |
| F2 | Late probability fusion | Fixed 0.5/0.5 average of matched class-aligned OOF model probabilities |
| F3–F5 | Neural multimodal fusion | Executed dual-branch CNN/TCN, deep late fusion and gated/cross-attention variants with Development-only selection |

The executed classical Logistic Regression uses `C=1.0`, `max_iter=2000` and class-balanced fitting with Training-fold scaling. Random Forest uses 120 trees, minimum leaf size two and class-balanced subsampling. No unexecuted hyperparameter grid or probability calibration is reported as performed. Both early and late fusion use the fixed source-class order; late-fusion weights are 0.5/0.5 and not fitted from outer Validation labels.

### 5.2 Deep-learning architectures and bounded training budget

Deep models were executed in the core Development comparison in Notebooks 03 and 04. The five-notebook study sequence and separate reserved Test boundary remain unchanged.

| Implemented deep arm | Signal input and research purpose | Architecture and training constraints |
| --- | --- | --- |
| D1 — sEMG 1D-CNN | Original aligned `X_emg` windows | Compact convolutional blocks, normalisation, pooling and classifier; standard neural baseline |
| D2 — accelerometer 1D-CNN / TCN | Original aligned `X_acc` windows | Compare a matched CNN or dilated temporal convolution under the same Development folds |
| D3 — temporal alternative (TCN or BiGRU) | Both modalities separately where feasible | A bounded TCN and BiGRU temporal comparator; report architecture-specific capacity rather than asserting equal parameter counts |
| F3 — dual-branch deep fusion | Paired sEMG and acceleration windows | Separate CNN/TCN encoders; concatenate learned embeddings; train a common classification head |
| F4 — deep late fusion | Outputs of independently fitted neural unimodal branches | Align class-probability order; determine any mixing weights inside Development only |
| F5 — gated / cross-attention fusion | Paired learned features from both branches | Small gated/attention fusion block with controlled heads/layers; assess whether it improves on simple embedding concatenation |

The original-data Development runs used five subject-disjoint folds, seed 42, a maximum of ten epochs, patience three, batch size 96 and Training-only inner validation for neural checkpoint selection. Architecture definitions, learning rate and optimiser implementation are preserved in `code/src/ninapro_models.py` and `code/src/ninapro_training.py`; these statements describe the executed settings rather than optional untested grids.

The executed neural fusion arms completed all five grouped Development folds at seed 42. The highest-scoring neural fusion architecture, `F3_dual_tcn`, was additionally fitted using seeds 43 and 44. These extra runs do not modify the original seed-42 selection, provide independent subjects or substitute for held-out Test evidence. Notebook 05 records fitted-model size, fit duration and CPU inference latency after the frozen classical refits; peak device-memory measurements are not asserted.

A larger deep model is not automatically stronger. Interpret any fusion gain against the **best Development-selected unimodal deep or classical model**, not just a weak baseline, with identical eligible windows and participant groups. Compare parameter counts and training budgets; if modality-specific sampling rates are used, align by physical time and ensure the two branches represent the same acquisition window.


**Notebook 04 engineering implementation:** The fusion models retain separate 12-channel sEMG and 36-column accelerometer encoders, fixed original 600-row windows, the five participant-disjoint folds and the same ten-epoch maximum, patience-three stopping and batch size 96. When safe GPU capacity permits, both modalities are Training-normalised and staged once per fold; a streamed fallback preserves the identical scientific inputs. Each of the four deep fusion architectures is executed and saved independently by fold. Classical early fusion uses the 48+96 source-window statistical concatenation; F2 and F4 late fusion use fixed 0.5/0.5 class-probability averaging. No Test-derived weights or outcomes enter these fits. Per-class OOF precision, recall, F1, support and confusion plots are derived from the resulting probabilities using the same participant-mean metric contract as Notebook 03. The recorded Notebook 04 source execution, CSV metrics and visualisations now constitute observed **Development** fusion evidence, not independent Test performance.

### 5.3 Modality ablation and failure handling

Notebook 04 executed three **synthetic acceleration mean-replacement conditions** for `F3_dual_cnn` and `F5_cross_attention`: all acceleration channels, one accelerometer group and a contiguous acceleration interval. The models were fitted without these interventions. These are inference sensitivity analyses, not observed physical sensor failures, missing-modality training or a general assessment of sEMG sensor loss. No unexecuted corruption or sensor-selection experiment is claimed as completed.

All arms must share the same subject and window keys for an interpretable fusion comparison. Source label correction, filtered sEMG, window purity and grouping controls apply equally across arms.

### 5.4 Implementation ownership

Reuse existing `python_modules/core/` functions only after inspecting their interface and testing compatibility with DB2: subject-disjoint splitting, grouped metrics, `plot_style`, `save_figure`, Colab setup, device selection and output saving. The existing HARTH windowing helpers were developed for a different rate and label taxonomy; they are **candidates**, not authorised cross-study conversions.

The implemented `code/src/` reader, paired-window/feature modules, PyTorch architectures, training routines and evaluation helpers support the five notebook stages. Offline tests cover relevant row/key alignment, subject separation, class/probability order, model interfaces and output contracts. These synthetic checks are not substitutes for reproducing the original-source results.

## 6. Results and Comparative Analysis

**Recorded data preparation:** S01 E1 contains 1,808,331 released sample rows (12 sEMG and 36 accelerometer columns), corrected labels 0–17 and 205 source segments. Across 40 participants, Notebook 02 found 337,853 source-eligible paired windows and retained 40,000 using 1,000 per person, including 35 Development people across five seven-person Validation folds and five reserved Test people.

**Recorded Notebook 03 Development evidence:** The following figures are mean fixed-18-class macro F1, calculated separately for each held-out Development participant and then averaged equally over 35 people. The original five grouped folds and source-window identities were shared across model arms.

| Model | sEMG | Acceleration |
| --- | ---: | ---: |
| Logistic Regression | 0.208411 | 0.103442 |
| Random Forest | **0.234836** | **0.128978** |
| 1D-CNN | 0.218176 | 0.067476 |
| TCN | 0.212824 | 0.070928 |
| BiGRU | 0.187360 | 0.065538 |

The Training-prior baseline yielded **0.034971**. The Development-selected unimodal comparator was **`B1_rf`**, the sEMG Random Forest. The saved evaluation indicates a marked gap between sEMG and acceleration, with acceleration Random Forest exceeding the acceleration neural architectures. These are descriptive model comparisons on Development people, not independent Test estimates or claims of significant superiority.

**Executed class-specific evidence:** Notebook 03 reports precision, recall, F1, predicted-window counts and true-class support for every corrected class using the original OOF probabilities. Both pooled-window and participant-mean metrics are retained; the equal-person mean of the class F1 values reconstructs the primary participant-balanced macro F1. The published table for the selected `B1_rf` reports 16,329 Rest windows (class 0) among 35,000 Development windows, with pooled recall 0.902321. In contrast, class 8 has pooled recall 0.022079. The count confusion matrix, true-class-row-normalised confusion matrix and eleven-model class-F1 heatmap reveal this marked per-class variation without suggesting independent Test performance.

**Executed Notebook 04 fusion comparison:** The nine fusion candidates were compared on the same 35 Development participants using original OOF window identities and the 18-class equal-participant macro F1 endpoint.

| Fusion model | Mean Development participant macro F1 |
| --- | ---: |
| `F1_rf` (early Random Forest) | **0.263012** |
| `F2_rf` (fixed late Random Forest) | 0.246213 |
| `F2_logreg` (fixed late Logistic Regression) | 0.164609 |
| `F1_logreg` (early Logistic Regression) | 0.163162 |
| `F3_dual_tcn` | 0.154582 |
| `F5_cross_attention` | 0.153039 |
| `F3_dual_cnn` | 0.149861 |
| `F4_deep_late` | 0.140728 |
| `F5_gated` | 0.140232 |

The fixed Development-selected unimodal comparator `B1_rf` scored **0.234836**. Relative to this comparator, selected `F1_rf` improved mean participant macro F1 by **0.028176** units, with positive paired differences for **25/35** Development participants and negative differences for **10/35**. No independent Test outcome entered model or fusion-weight selection. The selected fusion model's pooled Rest recall (class 0) was **0.942189**, whereas its pooled class-8 recall was **0.045078**. Count and normalised confusion plots and per-class F1 therefore provide important qualifications to the aggregate result.

**Next independent evaluation:** Notebook 05 will refit the frozen `B1_rf` and `F1_rf` recipes using only Development people, then score the five reserved Test participants once. This stage is not yet executed and no Test improvement is claimed.

Published third-party results are background literature with distinct protocols. In particular, Hu et al. (2024) reported a within-person repetition-split DB2 fusion accuracy that is not directly comparable to this person-independent macro-F1 endpoint. Zhang et al. (2023) established related intersubject sEMG/accelerometer fusion work, so intersubject fusion itself is not a novel concept.
## 7. Robustness, Uncertainty and Additional Analyses

**Recorded multi-seed stability:** Development-selected neural finalists were repeated with seeds 43 and 44 on the same five grouped folds. `D1_emg_cnn` yielded 0.213478 and 0.216074, compared with 0.218176 at seed 42; `D2_acc_tcn` yielded 0.070353 and 0.074801, compared with 0.070928 at seed 42. These repetitions probe fitting variability within the same cohort, not additional independent people. The executed fusion perturbations use three synthetic acceleration-replacement conditions for two neural arms. For neural fusion finalist `F3_dual_tcn`, additional seed scores were **0.119353** (43) and **0.124863** (44) versus **0.154582** (42). The original Development selection is retained. Participant-level bootstrap uncertainty and independent Test analysis remain **pending Notebook 05**.

Five held-out Test people would provide only a small cluster-bootstrap sample. Any reported interval will be descriptive, not a claim that an effect generalises across all people, hand tasks or devices. If fusion fails or performs worse for some people, preserve those results.

An optional second non-force exercise may be studied as an explicitly separate task after the primary configuration and results have been recorded; it will not be selected because its Test result appears more favourable.

## 8. Current Outputs and Reproducibility

### 8.1 Notebook sequence and file contracts

Notebooks 01–04 have tracked execution evidence: original-source audit, source-paired windows, eleven unimodal Development models and nine fusion Development models. Notebook 03's original aggregate `03_oof_probabilities.npz` is tracked in `output/` despite the general NPZ ignore rule, and Notebook 04's keyed CSV/PNG results and `04_model_selection.csv` are committed. Ten Notebook 04 model-specific/consolidated probability NPZ files remain in its downloadable Colab output archive, not Git. Notebook 05 is **prepared but not executed**; the Test comparison remains reserved.

| Notebook | Required input and operations | Named output targets |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Official pilot ZIP / MATLAB fields, sensor and label schema, source alignment and missingness | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Validated source inventory; label, time and pairing contracts; trial-safe windows and features | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local aligned arrays/features |
| `03_unimodal_grouped_baselines.ipynb` | Fixed class set and reserved Test isolation; grouped Development B0/B1/B2 plus temporal CNN/TCN/BiGRU; seed robustness and class-specific OOF summaries | `03_all_unimodal_metrics.csv`, `03_oof_probabilities.npz`, `03_selected_model_per_class_metrics.csv`, `03_all_models_per_class_metrics.csv`, `03_class_errors.png`, `03_class_errors_normalised.png`, `03_per_class_f1_heatmap.png` |
| `04_fusion_and_modality_ablations.ipynb` | Matched source keys, validated Notebook 03 soft OOF probabilities, independent classical early/late and neural paired-fusion fits, seed robustness and synthetic acceleration replacement | `04_dev_fusion_metrics.csv`, `04_dev_deep_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_model_selection.csv`, `04_fusion_robustness.csv`, `04_fusion_comparison.png`, `04_subject_paired_differences.png`, `04_modality_sensitivity.png`, `04_all_fusion_per_class_metrics.csv`, `04_selected_fusion_per_class_metrics.csv`, `04_fusion_class_errors_normalised.png`, `04_fusion_per_class_f1_heatmap.png` |
| `05_frozen_person_independent_test.ipynb` | Frozen `B1_rf` and `F1_rf` classical refits from Notebook 02 feature matrices; reserved Test comparison only after Development-only fitting | Planned: `05_test_predictions.csv`, `05_test_metrics.csv`, `05_per_class_support.csv` (precision/recall/F1), `05_confusion_matrix_normalised.png`, `05_subject_differences.png`, `05_cost_accuracy.png` |

Notebook-specific scientific sections anticipated:

- **01:** source download/inventory; MATLAB field audit; EMG–Acc pairing; original/corrected annotations; participant and repetition coverage; source evidence and interpretation.
- **02:** physical-time mapping; signal quality; label transitions and purity; paired windows; feature extraction; eligible counts and frozen split IDs; saved row keys.
- **03:** Development split integrity; train-only scalers/class weighting; majority/logistic regression/Random Forest; 1D-CNN, TCN and BiGRU unimodal models; matched person folds, fit/seed budgets, OOF predictions, learning curves and error interpretation.
- **04:** classical early/late fusion; neural two-branch embedding fusion and neural late fusion; bounded gated/cross-attention fusion; probability/label alignment; seed and sensor/dropout ablations; Development-only selection with measured computation.
- **05:** read frozen `B1_rf`/`F1_rf` Development decisions, reuse only the 48/96-column Notebook 02 feature matrices for these classical models, refit using all 35 Development people and compare once on five Test people; report paired subject metrics, descriptive bootstrap uncertainty, per-class F1, confusion matrices, fit time, model size and CPU inference cost.

Each implemented notebook will contain the established **00. Colab Repository Setup** and **01. Imports and Research Environment** sections, numbered scientific Markdown and code headings, and an independently curated SVG whose filename matches the notebook stem. SVGs explain actual implemented operations; no Test scores or empirical waveforms will be invented in a planning diagram.

### 8.2 Local and tracked artefact boundaries

- Local `data/raw/ninapro_db2/`: original subject ZIP archives, downloaded once per available runtime and reused when found.
- Local `data/processed/ninapro_db2/`: 40-person per-exercise source arrays and feature matrices, reused when available; missing arrays can be reconstructed from the original archives.
- Study-owned `output/`: tracked `01_*`–`04_*` CSV/PNG products and the original `03_oof_probabilities.npz` exception to Git's default NPZ ignore pattern. Notebook 04's 10 other probability NPZ outputs are held in the separate Colab archive. Notebook 05's current selected classical models require the two Notebook 02 statistical matrices in `data/processed/ninapro_db2/`, not all 40 source-length signal arrays; these small matrices must be restored if the local runtime was reset.
- Curated `images/`: five editable notebook-specific workflow SVGs; these are scientific process diagrams, not computed numerical plots.
- Repository root `requirements.txt`: the scientific environment dependencies used by the implementation; full runtime consistency remains dependent on the executing Colab/Python environment.

The executed sequence first inspected S01, then processed all 40 participants one at a time. Per-person original ZIPs are typically hundreds of megabytes, whereas the `output/` tables are small source/selection contracts. A fresh ephemeral Colab session may lack both `data/raw/` and `data/processed/`, in which case numeric input reconstruction can require fresh source downloads. Moving multi-GiB dense signal tensors to `output/` is not a practical replacement for persistent large-data storage. Each notebook replaces only its own named current outputs on rerun.

## 9. Discussion, Limitations and Future Work

Original source/cohort processing and the complete Notebook 03 unimodal and Notebook 04 fusion Development comparisons have recorded execution evidence. The dominant Rest class, participant variability, sEMG-informed corrected boundaries and interpolated acceleration stream complicate interpretation of class-level and modality-specific performance. Successful notebook execution does not establish that a model generalises to independent Test people, different acquisition devices or rehabilitation settings. Fusion Development results have been recorded, while final frozen-model Test results remain unmeasured.

The actual S01 and 40-participant E1 inspection supports the current source dimensions, corrected label set and frozen person assignments. Important interpretation limits remain: sEMG-informed corrected boundaries, the interpolated accelerometer stream, the high Rest proportion, and the reduced 1,000-window-per-person modelling cohort. None of these measurements establishes native sensor clock accuracy or performance in real-time prosthetic use.

**Key references:** [NinaPro DB2 guide](https://ninapro.hevs.ch/instructions/DB2.html); [Atzori et al. (2014)](https://doi.org/10.1038/sdata.2014.53); [Dryad source record](https://doi.org/10.5061/dryad.1k84r); [Hu et al. (2024)](https://doi.org/10.3390/s24092702); [Zhang et al. (2023)](https://doi.org/10.1109/JSEN.2023.3266872).
