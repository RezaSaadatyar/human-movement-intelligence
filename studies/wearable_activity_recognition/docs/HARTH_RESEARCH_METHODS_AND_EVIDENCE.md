# HARTH v2.0 | Research Methods and Experimental Evidence

**Study:** Wearable Activity Recognition in Human Movement Intelligence  
**Research scope:** Eight-class activity recognition from two wearable accelerometers, with participant-disjoint Development experiments and a qualified same-study held-out evaluation.  
**Evidence:** [Executed notebooks 01–27](../notebooks/), [study overview](../README.md) and [reproducibility and provenance](../../../docs/HARTH_REPRODUCIBILITY.md).

This document consolidates the substantive methods, observed results and interpretation boundaries of the completed HARTH study. It is a guide to the actual investigations, not a substitute for the saved outputs, model code or source-data provenance. Earlier notebooks used progressively enlarged Development cohorts and different training budgets; their metrics are not all directly comparable.

## 01. Research Objective and Dataset

**Research question.** How do handcrafted-feature machine-learning baselines and temporal deep-learning classifiers perform on eight recognised everyday activities when the assessment separates people, rather than randomly distributing overlapping activity windows across training and validation? Subsequent investigations examine class imbalance, sample availability, training budget, predictive confidence and stability across training seeds.

**Data source.** The study uses the official [NTNU AI Lab HARTH v2.0 source](https://github.com/ntnu-ai-lab/harth-ml-experiments), fixed at revision **dad2cfbe89a26f72f19770419469ac037de200df**. This is not the older UCI HARTH v1.2 distribution. Recordings comprise six triaxial acceleration channels from two body sites: lower back (back_x, back_y, back_z) and thigh (thigh_x, thigh_y, thigh_z), at nominal **50 Hz**. Participant and timestamp information is retained for temporal checks and group-based evaluation.

**Task labels.** The eight documented source codes are **1 Walking, 2 Running, 3 Shuffling, 4 Stairs ascending, 5 Stairs descending, 6 Standing, 7 Sitting and 8 Lying**. Undocumented source codes **10, 11 and 17** have been observed during data audits. The project does not invent activities for them or import label definitions from another dataset. Source inspection retains their original labels; out-of-task windows are removed only *after* safe window formation.

**Participants and scope.** Exploratory notebooks initially used smaller cohorts, then ten Development participants, then **29 Development participants** from the available 31-person source. The two designated final-report people, **S023 and S032**, were excluded from the later Development fitting and selection procedures. However, they had been descriptively inspected during early exploratory work. Consequently, the final assessment is **qualified same-study held-out evaluation**, not historically blind testing or independent external validation.

The main bounded Development benchmark uses **14,201** eligible windows after a label-independent chronological cap of at most **500 per person**. The uncapped eligible Development cohort used in earlier classical-model analysis contained **132,397** windows; its metrics are not interchangeable with results on the capped cohort. The final qualified evaluation uses **500 selected windows for each** of S023 and S032.

## 02. Preprocessing and Data Splitting

**Window construction (Stages 01–02).** Each participant recording is processed independently into **100-sample (2-second)** windows, advancing **50 samples (50% overlap)**. Retained windows require a single activity label throughout, monotonically ordered timestamps and acceptable time continuity; windows are not allowed to bridge annotation transitions or acquisition gaps. Excluded source labels are handled after candidate windows are formed, rather than dropping source rows first and accidentally joining unrelated periods. Input windows are represented as (windows, six channels, 100 samples). Consecutive overlapping windows remain correlated observations.

**Statistical features (Stage 03).** The classical-model representation contains five deterministic features per acceleration channel: arithmetic mean, population standard deviation, root mean square, peak-to-peak range and normalised waveform length (mean absolute consecutive difference). Six channels yield **30 features per window**. This extraction has no fitted scaling state; any learned scaling is estimated from the training partition alone.

**Participant-level partitioning (Stages 04–09).** Exploratory analyses progressed from an eight-person train/validation/test demonstration through three-fold and five-fold GroupKFold studies to the 29-person Development cohort. Group membership is assigned at the *participant level*, preventing one person's overlapping windows from appearing in different training and validation groups within the same comparison. GroupKFold is not intrinsically activity-stratified: a held-out person or fold may lack particular activities. Fold-level and participant-level class support therefore accompany the metrics. Initial model selection used Development data; neither its earlier single-split nor later exploratory GroupKFold scores constitute a new external test.

**Data-quality audit (Stages 08, 14 and 21).** The project reports participant-by-class support; source and retained label counts; annotation-transition and timestamp-gap exclusions; annotation-bout durations; signal distributions; and genuine low-frequency activity examples. Annotation bouts are segments of labels, not independently verified real-world behavioural events. Gaps and unusual signal values are diagnostics, not proof of sensor failure or incorrect ground truth. The original six channels are not assigned unverified physical measurement units.

**Sampling design.** The capped cohort selects eligible chronological positions deterministically, without using the activity label to choose retained windows. The Stage 14 sensitivity audit compares independent cap sizes **250, 500 and 1,000 per person** with the full eligible Development data. The *nested* cap-1,000 protocol of Stages 15–17 is different: it includes **every source position from the original cap-500 set** and adds further training examples. Under nested comparisons, validation people and original source-window positions remain fixed, and the extra examples come only from training participants. Hence results from independently spaced cap-1,000 and nested cap-1,000 are not directly equivalent.

**Leakage controls across the study.** Source rows and subject IDs are tracked through selection and evaluation. Channel means and standard deviations, class-loss weights, and Logistic Regression scalers are fitted on the appropriate fold's Train partition only. In Stage 19, Calibration participants are distinct from Train and outer Validation participants. The final Development model is fitted with the locked 29-person Development cohort; originally reserved S023/S032 data do not determine preprocessing statistics, hyperparameters, class weights or model choice. Full independence is not claimed for earlier descriptive inspection of these people.

## 03. Machine Learning and Deep Learning Models

**Classical baselines (Stages 05–09).** A Dummy classifier establishes a reference; Logistic Regression uses fold-Train standardisation; and Random Forest learns from the 30 extracted statistical features. The classical evaluations compare participant-held-out Macro F1, observed-class balanced accuracy, activity-class metrics and person-level outcomes.

**Initial temporal baseline (Stage 10).** The two-convolution TemporalCNN consumes the original six-channel, 100-time-point window directly, using train-fitted channel scaling, convolution/activation/pooling and a classification head. Its first pilot used five epochs, batch size 128, Adam at 0.001 and a fixed seed. This smaller single-split experiment is not a matched comparison to the earlier full-window Stage 09 classical baseline.

**Matched input and loss comparisons (Stages 11–13).** Random Forest and CNN are compared on the *same source windows and participant assignments*, while still using different representations: handcrafted statistics for the forest and raw series for CNN. A class-weighted CNN variant changes only the training objective, introducing inverse-square-root class weights estimated from Train label counts. Subsequent three-fold comparisons jointly assess Random Forest, Standard CNN and Weighted CNN on a common bounded Development cohort. Differences reflect representation, loss and architecture; they cannot be assigned solely to the network family.

**Additional temporal architectures (Stage 22).**

| Model | Studied design | Important qualification |
| --- | --- | --- |
| TCN | Four residual dilated-convolution blocks, with dilations 1, 2, 4 and 8 | Noncausal full-window classifier |
| Multi-Scale CNN | InceptionTime-inspired parallel 3/7/15 kernels with bottleneck and residual path | Not the full published InceptionTime ensemble |
| 1D-ResNet | Temporal stem, three residual convolution blocks, a stride-2 middle block and global pooling | Selected architecture in the later Development analysis |
| Time-Series Transformer | 48-dimensional embeddings, sinusoidal position encodings, two encoder layers and four attention heads | Bidirectional sequence classifier, not interpretable by virtue of attention alone |

All models accept (batch, 6, 100) float32 sequences and predict logits for the eight original source classes. The Stage 22 one-update smoke checks verified engineering behaviour on a small Development subset; they are **not performance benchmarks**. The scientifically matched seven-model comparison occurs in Stage 23.

**Robustness loss functions (Stage 24).** The base weighted cross-entropy uses inverse-square-root *Train-window* class counts. ResNet variants also examine **Focal Loss** (gamma = 2.0, with the same training-derived alpha weights) and **Class-Balanced Cross-Entropy** (effective-number beta = 0.9999). Overlapping windows make sample-count-based effective-number weighting approximate. No augmentation intervention was included.

## 04. Evaluation and Robustness Experiments

**Metrics and denominators.** The main classifier metric is **Macro F1 across the eight predefined classes**, with absent true classes handled using the fixed task definition. Observed-class balanced accuracy averages recall for labels represented in a given reference set; a model may predict a class not present in the current validation truth, which explains some scikit-learn warnings. Additional reporting includes overall accuracy, per-class precision/recall/F1 and true support, confusion matrices, fold-level variability and equal-person comparisons. Scores on overlapping windows are descriptive; folds and windows are not treated as independent clinical subjects.

**Development experiment sequence.**

| Stages | Question and design | Main interpretation boundary |
| --- | --- | --- |
| 01–04 | Inspect HARTH source recordings, windowing, statistical features and participant split behaviour | Single-person source inspection and early example splits are not model generalisation estimates |
| 05–09 | Dummy/Logistic Regression/Random Forest; progressive GroupKFold on six, ten and 29 Development people | Different cohort sizes and five-fold full-data evaluation are not matched to later bounded experiments |
| 10–13 | Temporal CNN pilot, paired Random Forest/CNN, weighted loss and three-fold matched comparison | Controlled within a given cohort; not an architecture-only causal contrast |
| 14 | Label-blind chronological sampling at caps 250/500/1,000 versus all eligible source windows | Data retention is not itself prediction improvement |
| 15–16 | Nested training increase from cap 500 to cap 1,000, first fixed-split and then matched three-fold comparison | Five epochs imply more optimiser updates for the larger training set |
| 17 | Compare nested training sets using **400 optimiser updates** for both conditions, with predeclared inspections at steps 100/200/300/400 | Equal updates do not equalise repeated exposure to each window; the step-400 result is primary |
| 18 | Out-of-fold confidence, ten-bin ECE, multiclass Brier, entropy, margin and high-confidence errors | Softmax confidence is not verified clinical uncertainty |
| 19 | Three outer subject-held-out folds with separate inner Calibration people; fit one scalar temperature on Calibration-only NLL | Probability calibration may improve some scoring rules but worsen others |
| 20 | Consolidate Stages 01–19 into a source-grounded evidence snapshot | Earlier archived figures are historical and can differ from later reruns |
| 21 | Audit unknown labels, sampling retention, annotation bouts and rare-class availability | Observational data-quality evidence does not establish a cause of model error |
| 22 | Engineering checks for four advanced temporal architectures | Smoke tests are not model-comparison evidence |
| 23 | Seven-model matched Development comparison, **21 model-fold fits** on common source windows | One seed; 400 updates do not equalise parameter counts or computational budget |
| 24 | Three seeds (42, 123, 2026), paired participant folds, ResNet/TCN stability and ResNet loss ablations: **36 neural fits** | Limited seed-level robustness, not statistical proof or out-of-source replication |

**Probability-quality findings (Stages 18–19).** Stage 18 found high-confidence mistakes and unequal activity-class recall in the original weighted-CNN out-of-fold Development predictions; the reported ten-bin ECE was **0.0453**, multiclass Brier **0.0905**, with **40 incorrect predictions at softmax confidence of at least 0.90**. These refer to a different training/validation arrangement than Stage 19 and should not be substituted for its baseline.

Stage 19 uses distinct Train, inner Calibration and outer Validation subjects in each of three Development folds, fitting a positive scalar temperature by minimising Calibration-only NLL. Its **current saved fold means** are:

| Outer Validation metric | Uncalibrated | Temperature-scaled |
| --- | ---: | ---: |
| Ten-bin ECE | 0.0491 | 0.0274 |
| Multiclass Brier | 0.1088 | 0.1044 |
| NLL | 0.2501 | 0.2580 |
| Accuracy | 0.9277 | 0.9277 |

The fold-specific fitted temperatures are **0.6322, 0.3943 and 0.5096**. Mean ECE and Brier improve, NLL worsens, and the predicted top-1 class is unchanged. One fold worsened across the three probability metrics; ECE also depends on binning. This is Development probability-quality analysis, not a clinical calibration claim. [Notebook 19](../notebooks/19_subject_disjoint_temperature_calibration.ipynb) provides the current numbers; [Notebook 20](../notebooks/20_research_evidence_consolidation.ipynb) separately preserves its historical Stage 19 snapshot.

**Matched seven-model Development benchmark (Stage 23).** Each model uses the same **29 people, 14,201 bounded windows and three GroupKFold participant partitions**. Random Forest receives matched-window engineered features; neural models receive the raw sequences. The six neural approaches use fold-Train channel scaling, Adam learning rate 0.001, batch size 128, last-step evaluation after **400 updates**, and seed 42. Mean fixed-eight-class Macro F1 and fold SD from the executed [Notebook 23](../notebooks/23_unified_matched_model_benchmark.ipynb):

| Model | Development Macro F1 (mean) | Fold SD |
| --- | ---: | ---: |
| 1D-ResNet | **0.8328** | 0.0435 |
| TCN | 0.8138 | 0.0658 |
| Random Forest | 0.8094 | 0.0500 |
| Time-Series Transformer | 0.8006 | 0.0602 |
| Multi-Scale CNN | 0.7946 | 0.0459 |
| Weighted CNN | 0.7725 | 0.0274 |
| Standard CNN | 0.6664 | 0.0633 |

In this matched Development study, pooled Shuffling F1 was **0.5032** for Random Forest, **0.3962** for 1D-ResNet and **0.3392** for TCN. The ordering varies by class: higher overall Macro F1 does not imply that ResNet is strongest for every activity.

**Repeated-seed and imbalance study (Stage 24).** Three fixed seeds and three folds were combined with paired model/loss comparisons. The 36 distinct neural fits include ResNet and TCN under weighted CE plus the additional ResNet loss variants, with repeated weighted-CE jobs reused rather than trained twice. Executed [Notebook 24](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) reports:

| Development condition | Mean Macro F1 across seed-level fold means | Across-seed SD |
| --- | ---: | ---: |
| 1D-ResNet, weighted CE | **0.8286** | 0.0052 |
| TCN, weighted CE | 0.8105 | 0.0079 |
| 1D-ResNet, focal loss | 0.8058 | 0.0053 |
| 1D-ResNet, class-balanced CE | 0.7915 | 0.0191 |

These repeated Development outcomes informed the primary/reference selection; they do not establish independent statistical superiority.

## 05. Final Model Selection and Results

**Development-only selection (Stage 25).** The primary model was fixed as **1D-ResNet with inverse-square-root weighted cross-entropy** because of its Development Macro F1 and limited repeated-seed performance. **Random Forest** was retained as an explicitly declared classical comparator rather than dropped because the neural model ranked first overall. Selection was completed without using the qualified held-out results. [Notebook 25](../notebooks/25_model_selection_and_protocol_freeze.ipynb) records the frozen choices.

**Frozen final training.** On the 29-person bounded Development cohort, the selected ResNet uses six input channels, eight outputs, hidden width 32, training-only per-channel standardisation and training-only class weighting, Adam at 0.001, batch size 128, **400 updates**, final-step weights and seed 42. The Random Forest uses **120 trees**, minimum leaf size 3, balanced-subset weighting, the 30 fixed engineered features and seed 42. Neither candidate includes Stage 19 temperature scaling; no models, thresholds, sample caps or losses were tuned on the held-out people.

**Approved qualified same-study held-out evaluation (Stage 26).** A separate approved record authorised one assessment of the previously designated **S023 and S032**, with 500 retained windows per person. The original and reconciled Stage 25 proposals remain historically pending; their status does not erase the later distinct approval. The evaluation is recorded in [Notebook 26](../notebooks/26_final_development_training_and_inference.ipynb) and synthesised without rescoring in [Notebook 27](../notebooks/27_final_research_evidence_and_report.ipynb).

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

S023 has all eight reference classes; S032 has six. Thus individual fixed-class Macro F1, pooled scores and observed-class balanced accuracy answer distinct questions. Very small Shuffling/stair support makes some apparently strong class recall values unstable and unsuitable for broad conclusions. The full confusion matrices, per-class precision/recall/F1 and outputs should be consulted in the executed Stage 26 notebook, rather than inferred from pooled accuracy.

**Stage 27 evidence synthesis.** The final report reads saved, versioned research results and approval metadata, presents source-grounded tables/figures and limitations, and does **not** retrain models, reload S023/S032, perform new Test inference or independently reproduce a GPU run. Its displayed values are rounded extracts of saved Stage 26 results, not a public archive of all original full-precision predictions.

## 06. Scientific Limitations and Evidence References

**Interpretation boundaries.**

1. **Population and provenance:** all participants come from the same HARTH source collection. S023/S032 were examined descriptively in early exploration; the qualified assessment is not a blind preregistered study, a new-cohort evaluation or external validation.
2. **Independent experimental units:** the final report covers only two people. Adjacent time windows overlap by 50%, and neither 1,000 windows nor GroupKFold folds represent 1,000 independent subjects or repeated clinical trials.
3. **Class imbalance and incomplete coverage:** Shuffling and stair activities have exceptionally few true final-report windows. F1, recall, balanced accuracy and overall accuracy should be interpreted alongside supports and participant-level coverage; the fixed-eight-class Macro F1 denominator is not the same as averaging only observed classes.
4. **Study selection and comparability:** earlier stages used different people, sample caps, optimisers, steps/epochs, validation splits, feature representations and model families. Only within-protocol comparisons are controlled. Architecture effects cannot be isolated from all representation, loss and compute differences.
5. **Source uncertainty:** undocumented activity codes 10/11/17 are not assigned invented meanings; source annotations and sensor quality have not been independently verified against external clinical or measurement references. Label and window exclusions affect the evaluated distribution.
6. **Confidence and calibration:** model softmax is not calibrated clinical certainty, nor an explanation of model reasoning. Temperature scaling produced mixed scoring-rule results and was not added to the frozen final model.
7. **Scope of conclusions:** the evidence does not establish prospective clinical validation, deployment readiness, statistical significance of model superiority, safety on patients, or population-wide external generalisation.

**Evidence map and reproducibility.**

- [Notebooks 01–27](../notebooks/) are the primary stage-by-stage records. The [historic Stages 01–19 evidence register](../evidence/stage_01_19_evidence_register.csv) contains summaries based on their respective saved stages; later reruns may differ.
- [Notebook 21](../notebooks/21_minority_class_and_data_quality.ipynb) documents rare classes and data-quality observations; [Notebook 23](../notebooks/23_unified_matched_model_benchmark.ipynb) and [Notebook 24](../notebooks/24_multiseed_and_imbalance_robustness.ipynb) document the matched benchmark and robustness experiments.
- [Notebook 25](../notebooks/25_model_selection_and_protocol_freeze.ipynb), [Notebook 26](../notebooks/26_final_development_training_and_inference.ipynb) and [Notebook 27](../notebooks/27_final_research_evidence_and_report.ipynb) document selection, qualified evaluation and the final read-only synthesis.
- The immutable [original proposal](../evidence/stage_25_protocol_proposal.json), [reconciled proposal](../evidence/stage_25_protocol_proposal_reconciled.json) and distinct [approval record](../evidence/stage_25_protocol_approved_qualified_test.json) establish the historical review and permitted Test scope. Preserve them as versioned evidence; the approved model/test protocol is not reauthorised by running a report.
- [Reusable model code](../../../code/python/src/core/), [tests](../../../tests/), [dataset-specific adapters](../src/), [study figures](../images/) and the [reproducibility guide](../../../docs/HARTH_REPRODUCIBILITY.md) document the implementation and checks. Original Stage 23/24 approved notebook bytes remain recoverable from pinned Git revisions.
- Raw source CSVs, cached jobs and final model artefacts are not distributed in the public repository. Repeating the complete GPU fit and authenticating original local model artefacts require additional resources and checks. The read-only [release-readiness audit](../../../scripts/check_harth_release_readiness.py) is not a rerun of the held-out evaluation.

**Research status.** The reported HARTH experiments and the Stage 27 evidence synthesis are completed portfolio research. The repository's other movement-science studies have separate progress and must not be presented as results of this HARTH benchmark. No new Test experiment or public release is implied by consolidating documentation.
