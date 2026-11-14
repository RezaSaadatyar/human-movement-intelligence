# HARTH | Research Methods and Experimental Evidence

**Study:** Wearable Activity Recognition in Human Movement Intelligence

**Scope:** Eight-class HARTH accelerometer recognition, participant-disjoint
Development evaluation, source and minority-activity quality, robustness,
final Development fitting and the separately reported qualified assessment.
The maintained notebook sequence is **01–13**.

The earlier 24–26 and FINAL notebooks are historical executed sources,
recoverable from Git version history. Their original Test results are not
performance results of the new Notebook 13 model artefacts. The original
source-aligned Test prediction CSV is not in the current public output folder.

See the [study README](README.md) and
[reproducibility guide](../docs/HARTH_REPRODUCIBILITY.md).

## 01. Research objective, source and activity definitions

The central research question is how classical classifiers using statistical
accelerometer features compare with temporal deep-learning architectures using
raw time-series windows when participants in Training and Validation are
disjoint. Further analyses investigate annotation integrity, activity
imbalance, bounded sampling, training-budget sensitivity, model stability,
probability confidence and calibration.

The source is the official [NTNU AI Lab HARTH v2.0 repository](https://github.com/ntnu-ai-lab/harth-ml-experiments),
pinned to revision `dad2cfbe89a26f72f19770419469ac037de200df`.
This source is distinct from the UCI HARTH v1.2 release.

The six acceleration signals are `back_x`, `back_y`, `back_z`, `thigh_x`,
`thigh_y`, and `thigh_z`, with a nominal sampling rate of **50 Hz**.
The predefined eight-class task uses original source labels:

| Code | Activity | Code | Activity |
| ---: | --- | ---: | --- |
| 1 | Walking | 5 | Stairs descending |
| 2 | Running | 6 | Standing |
| 3 | Shuffling | 7 | Sitting |
| 4 | Stairs ascending | 8 | Lying |

Source codes **10, 11 and 17** occur in the inspected recordings but have no
verified mapping in this research. They are retained for source auditing and
are never silently assigned to another activity. The eight-class task excludes
out-of-task windows **after** source-safe window construction.

**Participants:** The source inventory contains 31 participants. The
Development cohort has **29 participants**; S023 and S032 are reserved from
the main Development training and model selection. Because S023 and S032 were
descriptively examined in initial exploratory work, the reported assessment is
a **qualified same-study held-out evaluation**, not a blinded independent
external test.

## 02. Signal processing, windows and evaluation units

Each participant recording is segmented separately. Windows have **100
samples (2 seconds)** and a stride of **50 samples (50% overlap)**.
Retained windows require one consistent source activity code and valid
timestamp continuity. Label boundaries and timing gaps are never bridged to
construct an artificial contiguous sequence.

Classical models use **30 numerical features**, five per channel: mean,
population standard deviation, root mean square, peak-to-peak range and
normalised waveform length. Temporal models use six-channel raw windows
with input shape `(N, 6, 100)` and eight-class targets.

**Development sample sets:**

| Dataset definition | Participants | Eligible/sampled eight-class windows |
| --- | ---: | ---: |
| Full Development source | 29 | 132,397 |
| Label-blind, at most 500 windows per participant | 29 | 14,201 |
| Nested, at most 1,000 windows per participant | 29 | 28,201 |
| Qualified same-study held-out assessment (S023/S032) | 2 | 1,000 total; 500 each |

The 500-window bounded cohort is selected by fixed chronological positions
after window eligibility checks, without consulting task labels.
Integrated Notebook 06 also analyses a 250-window limit. The
strictly nested scaling design used in Notebooks 06 and 07
retain the original 500-window indices in the larger training set; this is
distinct from independently resampling the 1,000-window population.

**Group separation:** Subject identifiers define GroupKFold or fixed
participant-level partitions. All windows belonging to one participant
stay in the same fold for each experimental comparison. Learned
transformations—including channel scaling, Logistic Regression scaling and
class weights—are estimated only from the corresponding Training subjects.
Stage 19 further separates Training, Calibration and Validation subjects.

Overlapping windows are correlated observations. Reporting thousands of
window-level predictions must not be interpreted as having thousands of
independent human participants or experiments.

## 03. Models, training and matched comparisons

**Classical reference models (integrated Notebooks 02–03).** Dummy,
scaled Logistic Regression and Random Forest use the same eight-class
activity task and 30-feature representation. Notebook 03 evaluates the full
29-person Development cohort in five participant-disjoint folds over
132,397 eligible windows. Mean fold Macro F1 is **0.8353** for Random Forest,
**0.7707** for Logistic Regression and **0.0815** for Dummy. These are not
the same sampling or evaluation conditions as the bounded neural studies.
Source: [Notebook 03](code/notebooks/03_harth_full_development_audit_and_classical_cv.ipynb).

**Bounded temporal comparison (Notebook 04).** A compact two-convolution
temporal CNN consumes six-channel, 100-sample windows with training-only
channel scaling and a predefined five-epoch endpoint. Random Forest uses
30 deterministic statistics from the identical source windows. The single
participant-disjoint Development split provides 10,000 Train and 2,701
Validation windows from 20 and six participants, respectively; another
three Development participants are unused. On the matched Validation data,
Random Forest achieves **Macro F1 0.8140** and **balanced accuracy 0.8259**,
versus **0.6659** and **0.6459** for the temporal CNN. Random Forest scores
higher for all six Validation participants, with mean signed paired
difference (CNN minus Random Forest) **−0.1505**. The CNN has **Shuffling F1
0.000** on 15 true Shuffling windows. This evidence is descriptive and does
not isolate a model-architecture effect, establish statistical significance
or provide independent external validation.
Source: [Notebook 04](code/notebooks/04_harth_temporal_cnn_and_matched_random_forest.ipynb).

**Class-weighting and grouped comparisons (integrated Notebook 05).**
The same bounded 14,201-window Development cohort supports two distinct
experiments. On the fixed six-participant Validation split, standard CNN
Macro F1 is **0.6659** and Train-weighted CNN Macro F1 is **0.7973**;
balanced accuracy is **0.6459** and **0.8126**, respectively. Five of the
six Validation participants have higher weighted-CNN fixed-task Macro F1.

In a separate three-fold participant-disjoint Development comparison,
mean fold Macro F1 is **0.8094** for Random Forest, **0.6301** for standard
CNN and **0.7832** for weighted CNN. Weighted CNN is higher than standard
CNN for **27 of 29** participants; against Random Forest, it is higher
for **14** and lower for **15**. These experiments differ from Notebook
04's matched Random Forest versus standard CNN single-split comparison,
and the grouped forest/CNN performance contrast includes differing
feature representations. They do not establish independent external
validation or a general architecture ranking.
Sources: [Notebook 05](code/notebooks/05_harth_class_weighting_and_grouped_model_comparison.ipynb),
[fixed-split metrics](output/05_loss_validation_metrics.csv) and
[grouped model summary](output/05_cv_model_summary.csv).

**Sampling sensitivity and nested training data (integrated Notebook 06).**
The full eligible Development source contains **132,397** windows from
29 participants. Independent cap-250, cap-500 and cap-1,000 sampling policies
retain **7,201**, **14,201** and **28,201** windows. Under cap 500,
Shuffling retains **219/1,946 (11.25%)** of its original windows and
**26/28** of its contributing participants. These are sampling-coverage
counts, not model recall.

A distinct, strictly nested training-size experiment retains every
500-cap Train source row inside the expanded 1,000-cap Train set
(**10,000/20,000** Train windows). On the identical **2,701**
Validation windows, Random Forest final Macro F1 changes from
**0.8140 to 0.8644**, while weighted CNN fifth-epoch Macro F1
changes from **0.7973 to 0.7807**. The 1,000-cap CNN reaches
**0.8454** at epoch four, but that is not the predefined endpoint.
The forest improves for five of six Validation participants; the
weighted CNN improves for three and declines for three. Equal CNN
epochs do not imply equal optimiser-update budgets. The separate grouped
and fixed-update protocols of integrated Notebook 07 are not pooled
with these single-split findings.
Sources: [Notebook 06](code/notebooks/06_harth_sampling_sensitivity_and_nested_training_scaling.ipynb),
[sampling counts](output/06_sampling_policy_class_counts.csv),
[fixed-split metrics](output/06_nested_validation_model_metrics.csv) and
[paired participant evidence](output/06_nested_paired_summary.csv).

**Participant-grouped nested scaling and fixed-update budgets
(integrated Notebook 07).** A shared 29-participant Development cohort
contains 14,201 base and 28,201 expanded windows. In three
participant-disjoint folds, five-epoch mean Macro F1 changes from
**0.8094 to 0.8080** for RF and **0.7832 to 0.7964** for weighted CNN
(Train caps 500 to 1,000). Weighted CNN improves for **20/29**
participants on paired fixed-task Macro F1; RF improves for **11/29**,
declines for 16 and ties for two. A separate matched 400-update
weighted-CNN protocol records fold-mean Macro F1 **0.7721 to 0.7921**
but balanced accuracy **0.8021 to 0.7968**, with 17/29 participants
showing higher Macro F1. No independent Test is performed in either.
The executed [Notebook 07](code/notebooks/07_harth_grouped_nested_training_and_fixed_update_comparison.ipynb)
contains the saved result tables and figures; its separate `07_`
CSV/PNG export files have not yet been committed.

**Advanced architectures (Stage 21).** Engineering validation covered:

| Architecture | Implemented design | Scientific boundary |
| --- | --- | --- |
| TCN | Dilated temporal residual convolutions (dilations 1, 2, 4, 8) | Window-level, non-causal; no real-time streaming demonstration |
| Multi-Scale CNN | Parallel 3/7/15 convolutions and residual connections | Inception-inspired; not the complete InceptionTime ensemble |
| 1D-ResNet | Temporal convolution stem and three residual stages | Used subsequently in Development model selection |
| Time-Series Transformer | Embedding size 48, two encoder layers, four attention heads | Bidirectional within-window attention; not an explanation |

Stage 21 included shape/gradient and small-data optimiser checks. Model
performance claims come from the executed matched Stage 22 evaluation,
not these engineering smoke tests.

## 04. Development benchmark and robustness evidence

The current matched Notebook 10 benchmark uses all **29 Development participants**,
the same **14,201 bounded windows** and **three participant-disjoint folds**.
It evaluates seven model configurations over **21 model-fold jobs**. Each
neural configuration uses raw sequences, training-only normalisation, Adam
at 0.001, batch size 128, **400 optimiser updates** and seed 42; Random
Forest instead uses deterministic statistical features and a separate
training procedure. Loss functions and model representations are not
identical across the seven configurations.

**Primary metric:** fixed-eight-class Macro F1. Accuracy, balanced
accuracy, per-class precision/recall/F1 and support, fold dispersion,
participant-level outcomes and compute usage supplement it.

| Notebook 10 model | Development Macro F1 mean | Fold SD |
| --- | ---: | ---: |
| **1D-ResNet** | **0.8336** | 0.0427 |
| TCN | 0.8138 | 0.0658 |
| Random Forest | 0.8094 | 0.0500 |
| Time-Series Transformer | 0.8006 | 0.0602 |
| Multi-Scale CNN | 0.7953 | 0.0483 |
| Weighted CNN | 0.7721 | 0.0269 |
| Standard CNN | 0.6660 | 0.0628 |

The highest overall mean does not imply the best performance for every
activity or demonstrate statistical superiority. Recorded results:
[Notebook 10 model summary](output/10_matched_cv_model_summary.csv),
[class metrics](output/10_matched_cv_class_metrics.csv), and
[Notebook 10](code/notebooks/10_harth_advanced_temporal_architectures_and_matched_benchmark.ipynb).

Notebook 11 repeats **four model/loss conditions** across seeds
**42, 123 and 2026** and three held-out participant folds, comprising
**36 neural model-fold-seed jobs**. Weighted CE uses training-only
inverse-square-root class weighting; ResNet also compares Focal Loss
(`gamma=2`) and effective-number class-balanced CE (`beta=0.9999`).

| Notebook 11 condition | Macro F1 mean across seed-level fold means | Across-seed SD |
| --- | ---: | ---: |
| **1D-ResNet + weighted CE** | **0.8289** | 0.0053 |
| TCN + weighted CE | 0.8106 | 0.0078 |
| 1D-ResNet + focal loss | 0.8057 | 0.0051 |
| 1D-ResNet + class-balanced CE | 0.7915 | 0.0190 |

These results support a documented Development choice but do not establish
generalisation to new collections or the causal effect of architecture alone.
Recorded results: [Notebook 11 summary](output/11_robustness_model_summary.csv),
[fold metrics](output/11_robustness_fold_metrics.csv), and
[Notebook 11](code/notebooks/11_harth_multiseed_stability_and_imbalance_robustness.ipynb).

## 05. Confidence, calibration and minority activities

[Integrated Notebook 08](code/notebooks/08_harth_oof_confidence_and_temperature_calibration.ipynb)
has a completed saved execution and **17 committed output CSV/PNG files**.
It uses the same 29-participant HARTH Development population, with
**14,201 base** and **28,201 expanded** windows, for two distinct
experiments, each with three separately fitted 400-update weighted CNNs.
The reserved S023/S032 assessment participants are excluded.

**Uncalibrated out-of-fold confidence (Experiment A).** Models fit
on all outer Training participants and produce one uncalibrated
softmax vector per held-out base window. On **14,201** windows,
accuracy is **0.9457**, mean confidence **0.9003**, ten-bin ECE
**0.0454** and multiclass Brier **0.0907**. There are **39**
incorrect predictions at confidence at least 0.90. Shuffling
recall is **0.1187** on 219 held-out windows, with 193 errors.
Twenty of the 39 confident mistakes are Standing misclassified
as Sitting. These metrics are descriptive; neither high accuracy
nor softmax confidence establishes calibrated uncertainty.
Source: [current confidence summary](output/08_confidence_summary.csv)
and [class error metrics](output/08_confidence_class_error_summary.csv).

**Participant-disjoint scalar temperature scaling (Experiment B).**
These are **newly fitted models**, not the CNNs of Experiment A.
Inside each outer fold, four Calibration participants are separated
from model Training, and one positive scalar temperature is fitted
to Calibration-only labels. The temperatures are **0.6274**,
**0.3973** and **0.5050**. On outer Validation, the
three-fold mean metrics are:

| Outer Validation metric | Uncalibrated | Temperature scaled |
| --- | ---: | ---: |
| NLL (lower is better) | **0.2494** | 0.2580 |
| Ten-bin ECE (lower is better) | 0.0494 | **0.0286** |
| Multiclass Brier (lower is better) | 0.1085 | **0.1044** |
| Accuracy | 0.9275 | 0.9275 |

ECE and Brier improve *on average*, but NLL deteriorates, with
Fold 1 worsening on all three probability metrics. A positive
scalar temperature preserves top-1 predictions and cannot be
credited with increased classification accuracy. The fold-mean
metrics must not be conflated with window-pooled reliability.
Calibration is a Development-only descriptive experiment and
is not included in the frozen final model. Sources:
[current calibration mean metrics](output/08_calibration_mean_metrics.csv)
and [fold metrics](output/08_calibration_fold_metrics.csv).

**Historical evidence:** The earlier Stage 18/19 figures recorded
in the [Stage 01–19 evidence register](evidence/stage_01_19_evidence_register.csv)
reflect their original executions. They are not substituted for
Notebook 08 results, and the previous Stage 18/19 notebook files
have been retired after integration.

**Development source-quality and minority-activity audit (Notebook 09).**
The [consolidated notebook](code/notebooks/09_harth_source_quality_and_minority_activity_audit.ipynb)
has recorded Development-only results and 27 published `09_` CSV/PNG
outputs. The [source accounting](output/09_development_source_accounting.csv)
covers 29 participants and 8,006,741 raw samples. Of 160,090 candidate
100-sample windows, 14,459 fail label purity only, 397 fail timestamp
continuity only and 1,129 fail both checks; 144,105 pass. A further
11,708 pure/time-safe windows contain out-of-task labels, leaving
**132,397 eight-class eligible windows**. Codes 13, 14, 130 and 140
denote documented cycling variants outside this task; codes 10, 11
and 17 are unmapped rather than assigned activity labels.

The [activity-support table](output/09_activity_support_summary.csv)
records Shuffling **1,946 windows from 28 people** (1.47%),
Stairs ascending **1,332 from 26** (1.01%) and Stairs descending
**1,066 from 25** (0.81%). The [bout summary](output/09_annotation_bout_summary.csv)
records **2,342 of 3,359 Shuffling annotation bouts (69.72%)**
shorter than the two-second analysis window, with median nominal
duration **1.28 s**. Bouts describe recorded annotation runs, not
independently validated movement events. Overlapping windows are
not independent participants.

Independent label-blind per-participant caps of 250, 500 and 1,000
retain **7,201, 14,201 and 28,201** eligible Development windows.
At cap 500, Shuffling retention is **219/1,946 (11.25%)**, with
**26 of 28** original contributing participants. These separately
sampled caps are not the nested training expansion used by
Notebooks 06–07. The [channel summary](output/09_sensor_channel_summary.csv)
contains no non-finite samples but records notable channel extrema
and repeated neighbouring measurements; these observations do not
establish hardware failure or annotation errors.

The [source-to-OOF context](output/09_source_support_and_oof_error_context.csv)
joins source coverage with read-only
[Notebook 08 class errors](output/08_confidence_class_error_summary.csv).
The latter has **193 Shuffling errors among 219 capped windows**
(recall **0.1187**). The source and prediction populations have
different denominators and evaluation purposes; their descriptive
association cannot establish a cause of classifier errors or
independent Test performance.

## 06. Current selection and historically qualified results

**Development selection and final fitting:** Current
[Notebook 12](code/notebooks/12_harth_development_model_selection_and_protocol_specification.ipynb)
selects weighted 1D-ResNet as primary (three-fold Macro F1
**0.8336**; three-seed average **0.8289**) and retains Random
Forest as the classical reference. The reference has higher
pooled Shuffling F1 (**0.5032** versus **0.3969** for ResNet)
in the matched Notebook 10 comparison.

Current [Notebook 13](code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb)
fits the ResNet on all **14,201 Development windows** from 29
people, using six input channels, 32 hidden channels, inverse-square-root
class weighting, Adam (learning rate 0.001), batch size 128, seed 42
and 400 updates. The fixed Random Forest uses 30 features, 120
trees, minimum leaf size 3 and balanced-subsample weighting.
The original Notebook 13 CPU run took **67.649 seconds** for ResNet.
Its 24-window Development inference check tests artefact loading,
not participant-held-out accuracy.

The updated model configuration is a typed `inference_config.csv` alongside
`resnet_state_dict.pt` and `random_forest.joblib`. The changed CSV export
has not yet been rerun. Its absence does not invalidate the original
Development fitting observations but does mean the new archive is not
yet verified by execution.

**Earlier qualified assessment, recorded by the previously executed
Notebook 25:** S023 and S032 each contributed 500 overlapping windows.
Both had been inspected descriptively before that evaluation, so it
is neither blind nor external validation. The original saved aligned
predictions are unavailable in the current output directory; these
historically reported numbers **cannot be recomputed here** and must
not be assigned to the newly fitted Notebook 13 artefacts.

| Historical qualified metric | Weighted 1D-ResNet | Random Forest |
| --- | ---: | ---: |
| Fixed-eight-class Macro F1 | **0.7182** | 0.6817 |
| Accuracy | 0.9020 | **0.9270** |
| Balanced accuracy (observed classes) | **0.9234** | 0.7599 |
| S023 Macro F1 | 0.7384 | 0.7391 |
| S032 Macro F1 | 0.5662 | 0.4597 |

| Activity | True windows | Activity | True windows |
| --- | ---: | --- | ---: |
| Walking | 538 | Stairs descending | **6** |
| Running | 86 | Standing | 127 |
| Shuffling | **3** | Sitting | 115 |
| Stairs ascending | **10** | Lying | 115 |

S023 contained eight true task classes and S032 contained six.
These window counts provide very limited support for class-specific
performance claims. The [original executed Notebook 25 at
bcffd956](https://github.com/RezaSaadatyar/human-movement-intelligence/blob/bcffd956dd3a89edd8eb23ad1f63e360a15c72b1/wearable_activity_recognition/code/notebooks/25_final_development_training_and_inference.ipynb)
remains recoverable through Git history.

## 07. Cross-stage evidence and interpretation

Historical Notebooks 26 and FINAL combined earlier saved results
and provenance checks. Their scientific comparison boundaries are
preserved here; their administrative hashes and approval manifests
are not requirements of the maintained 01–13 workflow.

| Development cohort | Independent participants | Windows |
| --- | ---: | ---: |
| Full eligible eight-class source | 29 | **132,397** |
| 500-cap bounded study | 29 | **14,201** |
| Nested 1,000-cap study | 29 | **28,201** |

The original Development source includes **1,946 Shuffling**,
**1,332 Stairs ascending** and **1,066 Stairs descending** windows;
the 500-cap selection retains **219, 192 and 182**, respectively.
These are sampling quantities, not model sensitivity.

The current separately fitted Notebook 08 calibration study found
mean ECE **0.0494 to 0.0286**, Brier **0.1085 to 0.1044**,
NLL **0.2494 to 0.2580**, and unchanged accuracy **0.9275**.
Improvements in ECE and Brier do not override worsened NLL.
Temperature scaling was not applied to the final fit.

The full-cohort classical comparisons, capped Development folds,
nested Train sizes, independent calibration experiments and historical
qualified assessment differ in participant selection, training budget
or metric interpretation. They do not form a single matched leaderboard.

Earlier executed
[Notebook 26](https://github.com/RezaSaadatyar/human-movement-intelligence/blob/bcffd956dd3a89edd8eb23ad1f63e360a15c72b1/wearable_activity_recognition/code/notebooks/26_final_research_evidence_and_report.ipynb)
and
[FINAL research audit](https://github.com/RezaSaadatyar/human-movement-intelligence/blob/bcffd956dd3a89edd8eb23ad1f63e360a15c72b1/wearable_activity_recognition/code/notebooks/final_harth_research_audit_and_closure.ipynb)
remain accessible in version history. Their archived outputs do
not constitute a new independent model reproduction.

## 08. Research limitations and reproducibility

1. **Not externally validated.** All participants belong to one source
   collection. S023/S032 were descriptively inspected during exploration.
   The final assessment is not prospective, independent or fully blind.
2. **Limited independent units.** The final assessment includes two people.
   The 1,000 selected windows overlap and cannot be treated as independent
   participants or trials.
3. **Minority-class scarcity.** The held-out assessment contains only
   three Shuffling, ten ascending-stairs and six descending-stairs examples.
   Per-class recall, precision and F1 have severe support limitations.
4. **Unmatched exploratory experiments.** Different group assignments,
   sampling caps, data representations and training budgets prevent a
   scientifically valid cross-stage leaderboard or causal architectural
   ranking.
5. **Source annotation and signal integrity.** Undocumented codes remain
   uninterpreted; recording annotations and unusual channel values have
   not been independently verified against real-world ground truth.
6. **Model probabilities.** Softmax confidence and attention weights
   are not proof of faithful explanation or clinical uncertainty.
   Stage 19 calibration had mixed effects and is not part of the final
   trained procedure.
7. **Reproducibility boundary.** Public CSVs, executed notebooks and
   provenance records allow auditing many reported outcomes. Raw HARTH
   recordings, model weights and some local training caches are not
   included. A successful quality gate is not an independent rerun of
   all experiments.

### Source of record

- [Notebooks 01–13](code/notebooks/) are the maintained current programme.
  The [Stage 01–19 register](evidence/stage_01_19_evidence_register.csv)
  provides early-stage historical context.
- [Notebook 10](code/notebooks/10_harth_advanced_temporal_architectures_and_matched_benchmark.ipynb),
  [Notebook 11](code/notebooks/11_harth_multiseed_stability_and_imbalance_robustness.ipynb),
  [Notebook 12](code/notebooks/12_harth_development_model_selection_and_protocol_specification.ipynb)
  and [Notebook 13](code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb)
  document the current benchmark, robustness, choice and fitting.
- The historical qualified assessment is sourced from the
  [executed former Notebook 25](https://github.com/RezaSaadatyar/human-movement-intelligence/blob/bcffd956dd3a89edd8eb23ad1f63e360a15c72b1/wearable_activity_recognition/code/notebooks/25_final_development_training_and_inference.ipynb),
  not from Notebook 13 inference artefacts.
- [HARTH source adapter](code/src/harth_data.py),
  [model inference module](../python_modules/core/harth_final_inference.py)
  and [reproducibility guide](../docs/HARTH_REPRODUCIBILITY.md)
  define the current source and model-file interfaces.

**Current status:** The numbered study notebooks are consolidated into
01–13. The initial final Development fit was executed, but the changed
CSV configuration and single-ZIP packaging require re-execution before
their new outputs can be claimed as generated. No blind external
validation or independent reproduction of historical Test predictions
is claimed.
