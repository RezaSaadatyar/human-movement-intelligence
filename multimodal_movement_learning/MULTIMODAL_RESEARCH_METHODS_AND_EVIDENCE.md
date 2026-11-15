# Multimodal Movement Learning: Research Methods and Evidence

**Current status:** Five notebooks and the study-specific Python modules are implemented. Notebooks 01–02 have been executed on original NinaPro DB2 E1 data and their source, window-cohort, feature and participant-split evidence is recorded below. Notebook 03 has partial Development execution; neural/fusion comparisons and final Test evaluation remain incomplete. The [README](README.md) provides entry points, while this document distinguishes observed dataset results from planned predictive evaluation.

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

### 2.3 Proposed inclusion cohort and class definition

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

No reported metric is available yet. Balanced accuracy over missing classes, zero-support class F1 and rest prevalence must follow explicit, consistent definitions and include support tables.

## 5. Methods, Models and Configurations

### 5.1 Matched baseline and fusion arms

| Arm | Predictor and scientific purpose | Configuration principle |
| --- | --- | --- |
| B0 | Training-prior / majority-class classifier | Sanity lower bound, predicts on same eligible windows |
| B1 | sEMG-only feature classifier | Regularised multinomial logistic regression; simple Random Forest as optional matched alternate |
| B2 | Accelerometer-only feature classifier | Same estimator family and Development tuning budget as B1 |
| D1–D3 | Unimodal temporal networks | 1D-CNN, TCN and BiGRU on single-modality source windows using common subject folds |
| F1 | Early fusion | Concatenate B1/B2 features and fit a matched classifier, no cross-person input leakage |
| F2 | Late probability fusion | Blend two independently trained, properly aligned class-probability vectors using fixed or Development-selected weights |
| F3–F5 | Neural multimodal fusion | Two-branch learned-embedding fusion, deep late fusion and compact gated/cross-attention fusion, all tuned inside Development |

A small regularisation grid for logistic regression (e.g. `C=0.1,1,10`) and a bounded Random Forest option can be used **only in Development**. These are planning defaults rather than fitted values. Build interpretable baselines first, then evaluate the prespecified bounded neural models in the same five-notebook study; do not drop deep architectures from the authorised scope. If probability calibration is needed, fit it using Training-only inner folds; class probability order must match the fixed class list.

### 5.2 Deep-learning architectures and bounded training budget

Deep models are **included in the core planned comparison**, not an indefinite future add-on. The five-notebook sequence remains unchanged, but its third and fourth notebooks now include neural training.

| Planned deep arm | Signal input and research purpose | Architecture and training constraints |
| --- | --- | --- |
| D1 — sEMG 1D-CNN | Original aligned `X_emg` windows | Compact convolutional blocks, normalisation, pooling and classifier; standard neural baseline |
| D2 — accelerometer 1D-CNN / TCN | Original aligned `X_acc` windows | Compare a matched CNN or dilated temporal convolution under the same Development folds |
| D3 — temporal alternative (TCN or BiGRU) | Both modalities separately where feasible | A bounded TCN and BiGRU temporal comparator; report architecture-specific capacity rather than asserting equal parameter counts |
| F3 — dual-branch deep fusion | Paired sEMG and acceleration windows | Separate CNN/TCN encoders; concatenate learned embeddings; train a common classification head |
| F4 — deep late fusion | Outputs of independently fitted neural unimodal branches | Align class-probability order; determine any mixing weights inside Development only |
| F5 — gated / cross-attention fusion | Paired learned features from both branches | Small gated/attention fusion block with controlled heads/layers; assess whether it improves on simple embedding concatenation |

The exact depth, kernel lengths, widths, dropout, optimiser, learning rate and stopping policy will be **recorded before comparisons** and justified for the confirmed source lengths. Initial bounded options include 2–3 temporal blocks, hidden widths around 32–128, early stopping with a Training-only inner holdout, AdamW and cross-entropy; these are **proposals, not executed settings**.

To keep the work tractable while including deep learning: use one common five-fold grouped Development comparison for the principal arms where computationally feasible; start with one prespecified random seed for initial neural screening and use three seeds for the Development finalists. If compute forces a reduced fold/seed budget, label the exploratory comparison accordingly rather than combining it into a falsely matched leaderboard. Log accelerator, fit duration, peak device memory where available, parameter count and CPU inference latency. Do not let unapproved Test feedback select architecture or modify preprocessing.

A larger deep model is not automatically stronger. Interpret any fusion gain against the **best Development-selected unimodal deep or classical model**, not just a weak baseline, with identical eligible windows and participant groups. Compare parameter counts and training budgets; if modality-specific sampling rates are used, align by physical time and ensure the two branches represent the same acquisition window.

### 5.3 Modality ablation and failure handling

Compare removal of either modality, contiguous missing-acceleration intervals, one-axis corruption and a reduced sensor subset. Distinguish *simulated* missingness from missing source data. Do not silently use zero-valued vectors as valid missing-modality inference unless the model has an explicitly trained and documented missing-input policy.

All arms must share the same subject and window keys for an interpretable fusion comparison. Source label correction, filtered sEMG, window purity and grouping controls apply equally across arms.

### 5.4 Implementation ownership

Reuse existing `python_modules/core/` functions only after inspecting their interface and testing compatibility with DB2: subject-disjoint splitting, grouped metrics, `plot_style`, `save_figure`, Colab setup, device selection and output saving. The existing HARTH windowing helpers were developed for a different rate and label taxonomy; they are **candidates**, not authorised cross-study conversions.

Add a small `code/src/` reader, synchronised-window/feature module, PyTorch neural architectures and compact evaluation helpers only when Notebook 01–04 actually require them. Implement limited synthetic checks for row/key alignment, no subject overlap, train-only feature fitting, class/probability order and reproducible output replacement. Preserve existing shared tests and studies.

## 6. Results and Comparative Analysis

**Recorded data-preparation results:** S01 E1 has **1,808,331 stored rows** (12 sEMG and 36 acceleration columns), corrected labels **0–17**, 205 source segments and a **100% finite-value fraction** in the two inspected source arrays. Across 40 participants, the implemented E1 process found **337,853 source-eligible paired windows** and retained **40,000** under the predetermined 1,000-per-person selection. The retained cohort has **35 Development and five held-out Test people** with five Development folds. **These results describe data construction, not classifier performance.**

The intended predictive comparisons are Development-only per-person macro F1 of classical B0/B1/B2/F1/F2 plus unimodal neural D1–D3 and deep fusion F3–F5; only Development-selected arms progress to a single held-out person-independent Test comparison. The complete neural/fusion programme and final Test evaluation are not reported as executed. The final figure should show per-person fused-minus-unimodal differences rather than a single selectively favourable pooled score.

Only results from executed compatible code and the associated current `output/` tables will be reported. For context only: [Hu et al. (2024)](https://doi.org/10.3390/s24092702) reported a DB2 fusion accuracy for a **within-person repetition-split design**, which is not directly comparable to this planned person-independent macro-F1 endpoint. [Zhang et al. (2023)](https://doi.org/10.1109/JSEN.2023.3266872) published an intersubject sEMG/accelerometer fusion approach; its existence precludes a novelty claim that intersubject fusion itself is unexplored.

## 7. Robustness, Uncertainty and Additional Analyses

**Planned, not executed:** 200-/400-ms window sensitivity; missing-acceleration segments, one-axis/sensor removal and compact acceleration sensor selection; per-class/person disagreement; selected deep-model seed stability (initial seed plus up to three finalist seeds), cross-modal gate/attention ablation; measured training/inference cost, model size and GPU memory when applicable. Fused and unimodal performance will be reported under the same Development folds and paired Test windows.

Five held-out Test people would provide only a small cluster-bootstrap sample. Any reported interval will be descriptive, not a claim that an effect generalises across all people, hand tasks or devices. If fusion fails or performs worse for some people, preserve those results.

An optional second non-force exercise may be studied as an explicitly separate task after the primary configuration and results have been recorded; it will not be selected because its Test result appears more favourable.

## 8. Current Outputs and Reproducibility

### 8.1 Planned notebook sequence and file contracts

The Notebook 01 and 02 paths below are **existing current outputs** generated from original source data. The Notebook 03–05 entries describe their corresponding model-evaluation output contracts and should be interpreted alongside the actual completed runs.

| Notebook | Required input and operations | Named output targets |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Official pilot ZIP / MATLAB fields, sensor and label schema, source alignment and missingness | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Validated source inventory; label, time and pairing contracts; trial-safe windows and features | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local aligned arrays/features |
| `03_unimodal_grouped_baselines.ipynb` | Frozen class list and Test isolation; B0/B1/B2 classical grouped CV plus unimodal 1D-CNN, TCN and BiGRU Development comparisons | `03_dev_unimodal_metrics.csv`, `03_dev_neural_metrics.csv`, `03_oof_predictions.csv`, `03_neural_oof_predictions.csv`, `03_learning_curves.png`, `03_class_errors.png` |
| `04_fusion_and_modality_ablations.ipynb` | Same windows/folds as 03; classical F1/F2, two-branch F3, deep late F4 and gated/attention F5 plus Development-only ablations | `04_dev_fusion_metrics.csv`, `04_dev_deep_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_robustness.csv`, `04_fusion_comparison.png`, `04_modality_sensitivity.png` |
| `05_frozen_person_independent_test.ipynb` | Fixed settings; all Development people for final fit; reserved Test people for a single comparison | `05_test_predictions.csv`, `05_test_metrics.csv`, `05_subject_differences.png`, `05_cost_accuracy.png` |

Notebook-specific scientific sections anticipated:

- **01:** source download/inventory; MATLAB field audit; EMG–Acc pairing; original/corrected annotations; participant and repetition coverage; source evidence and interpretation.
- **02:** physical-time mapping; signal quality; label transitions and purity; paired windows; feature extraction; eligible counts and frozen split IDs; saved row keys.
- **03:** Development split integrity; train-only scalers/class weighting; majority/logistic regression/Random Forest; 1D-CNN, TCN and BiGRU unimodal models; matched person folds, fit/seed budgets, OOF predictions, learning curves and error interpretation.
- **04:** classical early/late fusion; neural two-branch embedding fusion and neural late fusion; bounded gated/cross-attention fusion; probability/label alignment; seed and sensor/dropout ablations; Development-only selection with measured computation.
- **05:** read frozen Development decisions for classical and deep arms; refit selected final configurations on all permitted Development people; apply to reserved people without Test-based selection; paired subject metrics, cluster-bootstrap uncertainty, CPU inference and measured GPU training costs; concise conclusion.

Each implemented notebook will contain the established **00. Colab Repository Setup** and **01. Imports and Research Environment** sections, numbered scientific Markdown and code headings, and an independently curated SVG whose filename matches the notebook stem. SVGs explain actual implemented operations; no Test scores or empirical waveforms will be invented in a planning diagram.

### 8.2 Local and tracked artefact boundaries

- Local `data/raw/ninapro_db2/`: original subject ZIP archives, downloaded once per available runtime and reused when found.
- Local `data/processed/ninapro_db2/`: 40-person per-exercise source arrays and feature matrices, reused when available; missing arrays can be reconstructed from the original archives.
- Study-owned `output/`: recorded `01_*` and `02_*` source/paired-window CSV and PNG outputs, including the frozen participant assignments; downstream `03_*`–`05_*` results belong to their actual executions.
- Curated `images/`: five editable notebook-specific workflow SVGs; these are scientific process diagrams, not computed numerical plots.
- Repository root `requirements.txt`: the scientific environment dependencies used by the implementation; full runtime consistency remains dependent on the executing Colab/Python environment.

The executed sequence first inspected S01, then processed all 40 participants one at a time. Per-person original ZIPs are typically hundreds of megabytes, whereas the `output/` tables are small source/selection contracts. A fresh ephemeral Colab session may lack both `data/raw/` and `data/processed/`, in which case numeric input reconstruction can require fresh source downloads. Moving multi-GiB dense signal tensors to `output/` is not a practical replacement for persistent large-data storage. Each notebook replaces only its own named current outputs on rerun.

## 9. Discussion, Limitations and Future Work

The source-audit and participant-cohort stages have been executed, but their successful execution alone does not validate the classification protocol. Complete, scientifically interpreted evidence for neural models, fusion effects and final Test performance is not available. The original healthy-participant, prompted-movement protocol is a controlled sensor classification task, not a demonstration of prosthetic control or rehabilitation impact. Any subsequent estimate will concern the source-defined labels, signal representations, participants and recording conditions actually evaluated.

The actual S01 and 40-participant E1 inspection supports the current source dimensions, corrected label set and frozen person assignments. Important interpretation limits remain: sEMG-informed corrected boundaries, the interpolated accelerometer stream, the high Rest proportion, and the reduced 1,000-window-per-person modelling cohort. None of these measurements establishes native sensor clock accuracy or performance in real-time prosthetic use.

**Key references:** [NinaPro DB2 guide](https://ninapro.hevs.ch/instructions/DB2.html); [Atzori et al. (2014)](https://doi.org/10.1038/sdata.2014.53); [Dryad source record](https://doi.org/10.5061/dryad.1k84r); [Hu et al. (2024)](https://doi.org/10.3390/s24092702); [Zhang et al. (2023)](https://doi.org/10.1109/JSEN.2023.3266872).
