# HARTH | Research Methods and Experimental Evidence

**Study:** Wearable Activity Recognition in Human Movement Intelligence  
**Research focus:** Recognising eight activities from wearable accelerometers, using participant-disjoint Development experiments and a qualified held-out evaluation within the same study.  
**Evidence:** [Notebooks 01–27](../notebooks/), [study overview](../README.md) and [reproducibility guide](../../../docs/HARTH_REPRODUCIBILITY.md).

This document summarises the main methods, results and interpretation limits of the completed HARTH study. It links the key findings to the saved notebooks and model code. Earlier experiments used different groups of participants and training settings, so their results cannot always be compared directly.

## 01. Research Objective and Dataset

**Research question.** How do machine-learning models using statistical features compare with deep-learning models using raw accelerometer signals when training and validation participants are kept separate? The study covers eight everyday activities. It also examines class imbalance, data availability, training duration, prediction confidence and variation across random seeds.

**Data source.** The study uses the official [NTNU AI Lab HARTH v2.0 source repository](https://github.com/ntnu-ai-lab/harth-ml-experiments), rather than the older UCI HARTH v1.2 dataset. The recordings contain six acceleration channels from sensors worn on the lower back (`back_x`, `back_y`, `back_z`) and thigh (`thigh_x`, `thigh_y`, `thigh_z`), sampled at approximately **50 Hz**. Participant IDs and timestamps support time-based checks and participant-level evaluation.

**Task labels.** The eight documented activity codes are **1 Walking, 2 Running, 3 Shuffling, 4 Stairs ascending, 5 Stairs descending, 6 Standing, 7 Sitting and 8 Lying**. The data checks also found undocumented codes **10, 11 and 17**. Their meaning is not assumed or taken from another dataset. The original codes are kept during inspection, and windows outside the eight-class task are removed only after window construction.

**Participants and scope.** The initial analyses used smaller groups before expanding to ten and then **29 Development participants** out of 31 available participants. **S023 and S032** were not used for the later Development model fitting or selection. However, both had been examined descriptively in earlier work. The final assessment is therefore a **qualified same-study held-out evaluation**, not a fully blind test or independent external validation.

The main Development benchmark contains **14,201** eligible windows, sampled chronologically without using activity labels, with at most **500 windows per participant**. An earlier full-data classical-model experiment used **132,397** eligible Development windows. Results from these two sample sizes should not be compared as though the evaluation conditions were identical. The final held-out evaluation uses **500 windows from each** of S023 and S032.

## 02. Preprocessing and Data Splitting

**Window construction (Stages 01–02).** Each participant's recording is split separately into windows of **100 samples (2 seconds)**, with a **50-sample step (50% overlap)**. Each retained window must contain one activity label and continuous, correctly ordered timestamps. Windows that cross label changes or time gaps are excluded. Labels outside the study task are filtered *after* windowing to avoid joining signals from separate periods. Each input window contains six channels and 100 time points. Overlapping windows are correlated and must not be treated as independent observations.

**Statistical features (Stage 03).** Classical models use five features from each acceleration channel: mean, population standard deviation, root mean square (RMS), peak-to-peak range and normalised waveform length (mean absolute difference between consecutive samples). With six channels, this produces **30 features per window**. Feature extraction requires no fitted scaler; where scaling is used, its parameters are learned from training data only.

**Participant-level splitting (Stages 04–09).** The study began with an eight-person train/validation/test example, then used three-fold and five-fold GroupKFold evaluation as more Development data became available. Splits are made by participant, keeping all windows from one person within the same group for each comparison. GroupKFold does not ensure equal activity coverage, so some validation participants or folds may have no examples of certain activities. Results are therefore reported alongside class counts for each fold and participant. These exploratory Development scores are not external-test results.

**Data-quality checks (Stages 08, 14 and 21).** The notebooks examine activity counts by participant, excluded windows, annotation changes, timestamp gaps, activity-label segments, signal distributions and examples of uncommon activities. A continuous stretch of one label is an annotation segment, not necessarily a verified real-world event. Time gaps or unusual signal values may indicate data issues but do not prove sensor failure or incorrect labels. Physical units are not assigned where the source documentation does not confirm them.

**Sampling design.** Windows are sampled at fixed chronological positions without using their activity labels. Stage 14 compares limits of **250, 500 and 1,000 windows per person** against all eligible Development windows. Stages 15–17 use a different, *nested* design: the 1,000-window training set retains every window from the original 500-window set and adds more training windows. Validation participants and windows stay unchanged. An independently sampled 1,000-window dataset is therefore not the same as the nested 1,000-window dataset.

**Preventing data leakage.** Participant IDs and source-window positions are tracked throughout training and evaluation. Channel scaling, Logistic Regression scaling and class-loss weights are calculated from the relevant training split only. In Stage 19, the people used for Training, Calibration and Validation are distinct. The final models are trained on the fixed 29-person Development cohort; S023 and S032 do not influence model selection, scaling or training settings. Their earlier descriptive inspection remains an important limitation.

## 03. Machine Learning and Deep Learning Models

**Classical baselines (Stages 05–09).** A Dummy classifier provides a simple reference. Logistic Regression uses scaling fitted to each fold's training data, while Random Forest uses the 30 statistical features. Performance is examined using participant-held-out Macro F1, balanced accuracy, class-level scores and participant-level results.

**Initial Temporal CNN (Stage 10).** The two-layer TemporalCNN takes the original six-channel, 100-point signals instead of statistical features. Training uses channel scaling fitted to the training set, convolutional layers and a classification head. The pilot ran for five epochs with batch size 128, Adam at 0.001 and a fixed random seed. Because it used a smaller single split, its scores cannot be directly compared with the full-data Stage 09 results.

**Matched model and loss comparisons (Stages 11–13).** Random Forest and CNN are tested on identical windows and participant splits. Random Forest uses statistical features, whereas CNN uses the raw time series. A weighted CNN adds inverse-square-root class weights calculated from training labels to address class imbalance. Three-fold evaluation then compares Random Forest, Standard CNN and Weighted CNN under a common Development protocol. Observed differences reflect more than architecture alone, because model inputs and loss functions also differ.

**Additional temporal architectures (Stage 22).**

| Model | Architecture | Scope and limitation |
| --- | --- | --- |
| TCN | Four residual dilated convolution blocks, with dilations 1, 2, 4 and 8 | Processes complete windows; not a causal streaming model |
| Multi-Scale CNN | Parallel 3/7/15 kernels with a bottleneck and residual connections, inspired by InceptionTime | Not a reproduction of the full InceptionTime ensemble |
| 1D-ResNet | Temporal stem, three residual blocks, stride-2 middle block and global pooling | Later selected using Development results |
| Time-Series Transformer | 48-dimensional embeddings, sinusoidal positions, two encoder layers and four attention heads | Uses bidirectional attention; attention alone is not an explanation |

All four models take `float32` inputs shaped `(batch, 6, 100)` and return eight class logits. Stage 22 checked that the architectures could process data and update their weights on a small Development sample. Those engineering checks do **not** establish model performance. The matched comparison of seven models is reported in Stage 23.

**Class-imbalance loss functions (Stage 24).** The baseline uses weighted cross-entropy, with inverse-square-root class weights calculated from training windows. ResNet was also tested with **Focal Loss** (gamma = 2.0, using the same training-derived alpha weights) and **Class-Balanced Cross-Entropy** (effective-number beta = 0.9999). Because windows overlap, their counts do not represent fully independent samples. Data augmentation was not part of this experiment.

## 04. Evaluation and Robustness Experiments

**Evaluation metrics.** The primary measure is **Macro F1 across the same eight task classes**, including classes missing from some validation sets. Balanced accuracy instead averages recall across classes present in the true labels of each validation set. This difference also explains some scikit-learn warnings when a model predicts a class absent from a fold's true labels. The study reports accuracy, per-class precision/recall/F1 and support, confusion matrices, fold variation and participant-level comparisons. Overlapping windows and validation folds are not independent participants.

**Development experiment sequence.**

| Stages | Experiment | Main limitation |
| --- | --- | --- |
| 01–04 | Inspect HARTH source recordings, windowing, statistical features and participant split behaviour | Early examples do not establish generalisation |
| 05–09 | Dummy/Logistic Regression/Random Forest; progressive GroupKFold on six, ten and 29 Development people | The full-data and later capped-window experiments are not directly comparable |
| 10–13 | Temporal CNN pilot, paired Random Forest/CNN, weighted loss and three-fold matched comparison | Model differences cannot be attributed only to architecture |
| 14 | Label-blind chronological sampling at caps 250/500/1,000 versus all eligible source windows | More retained windows do not by themselves show better prediction |
| 15–16 | Nested training increase from cap 500 to cap 1,000, first fixed-split and then matched three-fold comparison | Larger training sets also receive more optimiser updates |
| 17 | Compare nested training sets using **400 optimiser updates** for both conditions, with predeclared inspections at steps 100/200/300/400 | Equal updates do not equalise per-window exposure; step 400 is the main result |
| 18 | Out-of-fold confidence, ten-bin ECE, multiclass Brier, entropy, margin and high-confidence errors | Softmax confidence is not clinically validated uncertainty |
| 19 | Three outer subject-held-out folds with separate inner Calibration people; fit one scalar temperature on Calibration-only NLL | Different calibration metrics may change in opposite directions |
| 20 | Consolidate Stages 01–19 into a source-grounded evidence snapshot | The saved historical snapshot can differ from later runs |
| 21 | Audit unknown labels, sampling retention, annotation bouts and rare-class availability | Data-quality patterns do not prove the causes of errors |
| 22 | Engineering checks for four advanced temporal architectures | Engineering checks are not performance comparisons |
| 23 | Seven-model matched Development comparison, **21 model-fold fits** on common source windows | One seed; equal updates do not mean equal model size or compute |
| 24 | Three seeds (42, 123, 2026), paired participant folds, ResNet/TCN stability and ResNet loss ablations: **36 neural fits** | Only three seeds; no statistical proof or external replication |

**Prediction confidence and calibration (Stages 18–19).** Stage 18 found errors made with high softmax confidence and differences in recall across activities. Its out-of-fold Development results included ten-bin ECE **0.0453**, multiclass Brier score **0.0905** and **40 incorrect predictions with softmax confidence of at least 0.90**. Stage 19 uses different training and validation groups, so the Stage 18 values are not a direct baseline for temperature scaling.

Stage 19 keeps Training, Calibration and outer Validation participants separate in each of three Development folds. One positive temperature parameter is fitted using Calibration-only negative log-likelihood (NLL), then assessed on the outer Validation group. The **current saved fold means** are:

| Outer Validation metric | Uncalibrated | Temperature-scaled |
| --- | ---: | ---: |
| Ten-bin ECE | 0.0491 | 0.0274 |
| Multiclass Brier | 0.1088 | 0.1044 |
| NLL | 0.2501 | 0.2580 |
| Accuracy | 0.9277 | 0.9277 |

The three fitted temperatures are **0.6322, 0.3943 and 0.5096**. Average ECE and Brier scores decrease after scaling, while NLL increases and accuracy stays the same. One fold worsens on all three probability measures, and ECE depends on the number of bins. These are Development findings, not evidence of clinical calibration. [Notebook 19](../notebooks/19_subject_disjoint_temperature_calibration.ipynb) contains the current results; [Notebook 20](../notebooks/20_research_evidence_consolidation.ipynb) retains an older snapshot.

**Seven-model Development benchmark (Stage 23).** All seven models use the same **29 participants, 14,201 selected windows and three participant-disjoint GroupKFold splits**. Random Forest uses statistical features from those windows; the neural models use raw sequences. The six neural models use training-only channel scaling, Adam at 0.001, batch size 128, **400 updates**, final-step evaluation and seed 42. The executed [Notebook 23](../notebooks/23_unified_matched_model_benchmark.ipynb) reports the following mean fixed-eight-class Macro F1 values and fold standard deviations:

| Model | Development Macro F1 (mean) | Fold SD |
| --- | ---: | ---: |
| 1D-ResNet | **0.8328** | 0.0435 |
| TCN | 0.8138 | 0.0658 |
| Random Forest | 0.8094 | 0.0500 |
| Time-Series Transformer | 0.8006 | 0.0602 |
| Multi-Scale CNN | 0.7946 | 0.0459 |
| Weighted CNN | 0.7725 | 0.0274 |
| Standard CNN | 0.6664 | 0.0633 |

Random Forest achieved the highest pooled Shuffling F1 (**0.5032**), compared with **0.3962** for 1D-ResNet and **0.3392** for TCN. Although ResNet has the highest overall Development Macro F1, it does not perform best on every activity.

**Multi-seed and loss comparisons (Stage 24).** Stage 24 uses three seeds and the same three participant folds to examine ResNet and TCN stability and compare ResNet loss functions. The **36 distinct neural training runs** reuse existing weighted-cross-entropy results rather than repeating the same jobs. Executed [Notebook 24](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) reports:

| Development condition | Mean Macro F1 across seed-level fold means | Across-seed SD |
| --- | ---: | ---: |
| 1D-ResNet, weighted CE | **0.8286** | 0.0052 |
| TCN, weighted CE | 0.8105 | 0.0079 |
| 1D-ResNet, focal loss | 0.8058 | 0.0053 |
| 1D-ResNet, class-balanced CE | 0.7915 | 0.0191 |

The repeated Development results helped select the final model, but they do not establish statistical superiority or performance on a new dataset.

## 05. Final Model Selection and Results

**Model selection using Development data (Stage 25).** **1D-ResNet with inverse-square-root weighted cross-entropy** was selected as the primary model based on its Development Macro F1 and results across three random seeds. **Random Forest** remained the planned classical comparator, including because its performance differed across activity classes. Neither choice used the final held-out results. See [Notebook 25](../notebooks/25_model_selection_and_protocol_freeze.ipynb).

**Final training settings.** The ResNet was trained on the fixed 29-person Development cohort, with six input channels, eight output classes, hidden width 32, training-only channel scaling and class weights, Adam at 0.001, batch size 128, **400 updates**, seed 42 and final-step weights. The Random Forest uses **120 trees**, minimum leaf size 3, `balanced_subsample` weighting, 30 statistical features and seed 42. Stage 19 temperature scaling was not included. The held-out people were not used to tune models, thresholds, sampling limits or loss functions.

**Qualified held-out evaluation (Stage 26).** A separate approval permitted one evaluation on **S023 and S032**, with **500 windows from each participant**. Earlier Stage 25 proposal records still show their historical pending status; the later approval is stored separately. [Notebook 26](../notebooks/26_final_development_training_and_inference.ipynb) records the evaluation. [Notebook 27](../notebooks/27_final_research_evidence_and_report.ipynb) summarises those saved results without running the models again.

| Qualified held-out outcome | 1D-ResNet weighted CE | Random Forest |
| --- | ---: | ---: |
| Fixed-eight-class Macro F1, pooled | **0.7182** | 0.6817 |
| Accuracy, pooled | 0.9020 | **0.9270** |
| Balanced accuracy over observed reference classes | 0.9234 | 0.7599 |
| S023 Macro F1 (500 windows) | 0.7384 | 0.7391 |
| S032 Macro F1 (500 windows) | 0.5662 | 0.4597 |

**True activity support in the combined 1,000 windows:**

| Activity | True windows |
| --- | ---: |
| Walking | 538 |
| Running | 86 |
| Shuffling | **3** |
| Stairs ascending | **10** |
| Stairs descending | **6** |
| Standing | 127 |
| Sitting | 115 |
| Lying | 115 |

S023 contains all eight true activity classes, while S032 contains six. Pooled and participant-level scores therefore describe different aspects of performance. The very small numbers of Shuffling and stair windows make per-class recall especially uncertain. For full precision, recall, F1 and confusion matrices, see the actual outputs in [Notebook 26](../notebooks/26_final_development_training_and_inference.ipynb). Pooled accuracy alone is not sufficient to assess the rare classes.

**Final evidence report (Stage 27).** Notebook 27 reads and summarises saved results, approval records, tables, plots and limitations. It does **not** train models, reload S023 or S032, run new held-out predictions or repeat the GPU experiments. Reported values are rounded from saved Notebook 26 outputs; the full original predictions are not included in the public repository.

## 06. Scientific Limitations and Evidence References

**Study limitations.**

1. **Participants and validation:** all participants come from the same HARTH study. S023 and S032 were inspected descriptively in early work, so the final assessment is not fully blind, prospective or external validation.
2. **Sample size:** the final evaluation includes only two people. Adjacent windows overlap by 50%, so 1,000 windows are not 1,000 independent observations, and folds are not independent clinical trials.
3. **Rare and missing classes:** Shuffling and stair activities have very few final-report examples. Their performance scores need to be read alongside class counts and participant-level results. Fixed-eight-class Macro F1 differs from metrics calculated using only observed classes.
4. **Comparing experiments:** early studies used different participants, sample limits, splits, training budgets and model inputs. Only directly matched experiments support a fair comparison, and even these cannot isolate every effect of architecture, loss or computation.
5. **Source labels and signals:** codes 10, 11 and 17 have no verified activity definitions in this study. Labels and sensor measurements have not been independently checked, and excluding some windows changes the activity distribution being evaluated.
6. **Confidence estimates:** softmax scores do not establish clinical certainty or explain how a model reaches a decision. Temperature scaling had mixed effects and was not used in the final model.
7. **What the study does not show:** the results do not establish prospective clinical validity, readiness for deployment, patient safety, statistically significant model superiority or generalisation to new populations.

**Research evidence and reproducibility.**

- [Notebooks 01–27](../notebooks/) contain the recorded experiments. The [Stages 01–19 evidence register](../evidence/stage_01_19_evidence_register.csv) summarises the earlier saved results; subsequent reruns may produce different values.
- [Notebook 21](../notebooks/21_minority_class_and_data_quality.ipynb) examines rare activities and data quality. [Notebook 23](../notebooks/23_unified_matched_model_benchmark.ipynb) reports the seven-model benchmark, and [Notebook 24](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) reports the multi-seed and loss experiments.
- [Notebook 25](../notebooks/25_model_selection_and_protocol_freeze.ipynb) records model selection, [Notebook 26](../notebooks/26_final_development_training_and_inference.ipynb) the approved held-out evaluation, and [Notebook 27](../notebooks/27_final_research_evidence_and_report.ipynb) the final report based on saved evidence.
- The [original proposal](../evidence/stage_25_protocol_proposal.json), [reconciled proposal](../evidence/stage_25_protocol_proposal_reconciled.json) and separate [approval record](../evidence/stage_25_protocol_approved_qualified_test.json) document the model settings and the permitted scope of the final evaluation. They are preserved as historical records; reading the report does not permit new held-out testing.
- [Model code](../../../code/python/src/core/), [tests](../../../tests/), [HARTH data-processing code](../src/), [figures](../images/) and the [reproducibility guide](../../../docs/HARTH_REPRODUCIBILITY.md) explain how the results were produced and checked. The original Stage 23/24 approved versions remain available in Git history.
- Raw CSV files, cached training runs and final model files are not included in the public repository. Full GPU reproduction and verification of the original local model files require separate checks. The [release-readiness audit](../../../scripts/check_harth_release_readiness.py) examines saved evidence without repeating the held-out evaluation. The exact HARTH source revision is documented in the [study overview](../README.md).

**Research status.** The HARTH experiments and Stage 27 report are complete as a research portfolio study. Other movement-science studies in this repository have separate progress and results. Updating this document does not add a new held-out experiment or create a public release.