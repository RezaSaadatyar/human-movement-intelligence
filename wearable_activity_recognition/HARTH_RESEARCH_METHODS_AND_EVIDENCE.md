# HARTH | Research Methods and Experimental Evidence

**Study:** Wearable Activity Recognition in Human Movement Intelligence

**Scope:** Eight-class accelerometer-based recognition; participant-disjoint
Development experiments; qualified same-study held-out model assessment; saved
scientific results and reproducibility constraints. This document reflects the
**26 numbered research notebooks** and the distinct
[FINAL research audit](code/notebooks/final_harth_research_audit_and_closure.ipynb).
The FINAL notebook has **11 executed code cells, a saved PASSED Quality Gate
and ten committed `final_*` outputs**. It audits recorded scientific evidence
without new training or qualified Test inference.

See the [study README](README.md) for navigation, the
[Stage 26 evidence report](code/notebooks/26_final_research_evidence_and_report.ipynb)
for the executed final synthesis, and the
[reproducibility guide](../docs/HARTH_REPRODUCIBILITY.md) for availability and
limitations of publicly committed artefacts.

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
after window eligibility checks, without consulting task labels. Stage 14
also analyses a 250-window limit. Nested scaling experiments in Stages 15–17
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

**Classical reference models (Stages 05–09).** Dummy, scaled Logistic
Regression and Random Forest use the same defined activity task and the
statistical feature pipeline. Full-cohort Stage 09 uses five subject-grouped
folds over 132,397 windows. Its recorded mean fold Macro F1 is **0.8353**
for Random Forest, **0.7715** for Logistic Regression and **0.0815** for
Dummy. This is **not the same protocol** as the capped three-fold neural
benchmark. Source: [`09_development_cv_model_summary.csv`](output/09_development_cv_model_summary.csv).

**Initial temporal comparisons (Stages 10–13).** The small two-layer CNN
uses training-only channel scaling and raw windows. Stages 11–12 compare
Random Forest, standard CNN and inverse-square-root weighted CNN on
matched windows and a fixed Development split. Stage 13 introduces a
three-fold matched Development comparison. Recorded Stage 13 Macro F1
means are **0.8094** (Random Forest), **0.6300** (standard CNN) and
**0.7821** (weighted CNN). Differences between classical and neural
models include both their representations and training objectives.
Source: [`13_matched_cv_model_summary.csv`](output/13_matched_cv_model_summary.csv).

**Sampling and optimiser budgets (Stages 14–17).** Stage 14 audits
sampling-induced class retention. Fixed-validation nested sampling
(Stage 15) reports RF 500/1000 Macro F1 of **0.8140/0.8644** and
weighted CNN 500/1000 of **0.8014/0.7799**. Stage 16 extends this
to grouped three-fold comparisons: RF **0.8094/0.8080** and weighted
CNN **0.7821/0.7964**, respectively. Stage 17 fixes neural optimisation
to **400 updates** and records weighted CNN Macro F1 **0.7712/0.7931**
for the nested 500/1000 conditions. These experiments have distinct
evaluation designs and must not be pooled into a single ranking.
Sources: [Stage 15](output/15_nested_validation_model_metrics.csv),
[Stage 16](output/16_nested_cv_model_summary.csv) and
[Stage 17](output/17_fixed_step_model_summary.csv).

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

The matched Stage 22 benchmark uses all **29 Development participants**,
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

| Stage 22 model | Development Macro F1 mean | Fold SD |
| --- | ---: | ---: |
| **1D-ResNet** | **0.8328** | 0.0435 |
| TCN | 0.8138 | 0.0658 |
| Random Forest | 0.8094 | 0.0500 |
| Time-Series Transformer | 0.8006 | 0.0602 |
| Multi-Scale CNN | 0.7946 | 0.0459 |
| Weighted CNN | 0.7725 | 0.0274 |
| Standard CNN | 0.6664 | 0.0633 |

The highest overall mean does not imply the best performance for every
activity or demonstrate statistical superiority. Recorded results:
[Stage 22 model summary](output/22_matched_cv_model_summary.csv),
[class metrics](output/22_matched_cv_class_metrics.csv), and
[Notebook 22](code/notebooks/22_unified_matched_model_benchmark.ipynb).

Stage 23 repeats **four model/loss conditions** across seeds
**42, 123 and 2026** and three held-out participant folds, comprising
**36 neural model-fold-seed jobs**. Weighted CE uses training-only
inverse-square-root class weighting; ResNet also compares Focal Loss
(`gamma=2`) and effective-number class-balanced CE (`beta=0.9999`).

| Stage 23 condition | Macro F1 mean across seed-level fold means | Across-seed SD |
| --- | ---: | ---: |
| **1D-ResNet + weighted CE** | **0.8286** | 0.0052 |
| TCN + weighted CE | 0.8105 | 0.0079 |
| 1D-ResNet + focal loss | 0.8058 | 0.0053 |
| 1D-ResNet + class-balanced CE | 0.7915 | 0.0191 |

These results support a documented Development choice but do not establish
generalisation to new collections or the causal effect of architecture alone.
Recorded results: [Stage 23 summary](output/23_robustness_model_summary.csv),
[fold metrics](output/23_robustness_fold_metrics.csv), and
[Notebook 23](code/notebooks/23_multiseed_and_imbalance_robustness.ipynb).

## 05. Confidence, calibration and minority activities

**Development-only confidence (Stage 18).** A 14,201-window out-of-fold
audit recorded accuracy **0.9458**, ten-bin expected calibration error
(ECE) **0.0457**, multiclass Brier **0.0906** and **39 incorrect windows**
with maximum predicted class probability at least 0.90. These are
descriptive confidence statistics, not independent uncertainty
validation. Source: [Stage 18 confidence summary](output/18_oof_confidence_summary.csv).

**Subject-disjoint temperature scaling (Stage 19).** A scalar temperature
was fitted to Calibration-only negative log-likelihood (NLL), with
Training, Calibration and outer Validation participants kept separate.
The recorded mean metrics are:

| Outer Validation metric | Uncalibrated | Temperature scaled |
| --- | ---: | ---: |
| ECE (10 bins) | 0.0488 | 0.0286 |
| Multiclass Brier | 0.1082 | 0.1043 |
| NLL | **0.2487** | 0.2573 |
| Accuracy | 0.9275 | 0.9275 |

ECE and Brier decreased on average, **NLL increased** and accuracy did
not change. These effects are not universally favourable across folds.
Temperature scaling is **not included** in the frozen final model.
Source: [Stage 19 calibration means](output/19_temperature_scaling_mean_metrics.csv)
and [fold metrics](output/19_temperature_scaling_fold_metrics.csv).

**Source and minority-class audit (Stages 14 and 20).** Timestamp gaps,
annotation transitions, exclusion codes, channel diagnostics and
the distribution of rare activities are assessed without inventing
undocumented source labels. Deterministic per-participant sampling limits
can change minority-class availability before any classifier is fitted;
sampling retention is not predictive recall. Qualified held-out support
is especially sparse for Shuffling and stair activities.

## 06. Frozen selection and qualified held-out results

**Selection (Stage 24).** The primary model is **1D-ResNet with
inverse-square-root weighted cross-entropy**. The fixed comparator is
**Random Forest**. Both choices use Development results. The model
specifications and permission to perform the qualified assessment are
recorded in the [approved frozen protocol](evidence/stage_25_protocol_approved_qualified_test.json).
That historical filename is retained for provenance.

**Final Development fitting (Stage 25).** The ResNet has six inputs,
eight outputs, hidden channels 32, train-only normalisation and class
weights; it uses Adam at 0.001, batch 128, seed 42 and 400 optimiser
updates. Random Forest uses 30 deterministic features, 120 trees,
minimum leaf size 3, `balanced_subsample` and seed 42. Neither is
calibrated with the exploratory temperature-scaling experiment.

The separately approved qualified assessment used **S023 and S032**,
with **500 label-blind sampled windows per participant**. The same-study,
previously inspected status of these subjects is an explicit
interpretation limitation.

| Qualified held-out metric | 1D-ResNet Weighted CE | Random Forest |
| --- | ---: | ---: |
| Fixed-eight-class pooled Macro F1 | **0.7182** | 0.6817 |
| Pooled accuracy | 0.9020 | **0.9270** |
| Balanced accuracy over observed true classes | **0.9234** | 0.7599 |
| S023 Macro F1 (500 windows) | 0.7384 | 0.7391 |
| S032 Macro F1 (500 windows) | 0.5662 | 0.4597 |

**True activity support across the 1,000 selected windows:**

| Activity | Support | Activity | Support |
| --- | ---: | --- | ---: |
| Walking | 538 | Stairs descending | **6** |
| Running | 86 | Standing | 127 |
| Shuffling | **3** | Sitting | 115 |
| Stairs ascending | **10** | Lying | 115 |

S023 contains eight true task classes; S032 contains six. Because
rare-class support is extremely limited, per-class metrics should not be
interpreted as stable population estimates. Accuracy alone obscures
the minority-class limitations. See the [Stage 25 model summary](output/25_qualified_model_summary.csv),
[participant metrics](output/25_qualified_per_subject.csv),
[class metrics](output/25_qualified_per_class.csv) and
[qualified test provenance](output/25_qualified_test_provenance.json).

**Saved scientific synthesis (Stage 26).** Notebook 26 reads full-precision
committed Stage 22, 23 and 25 result CSVs and the frozen approval records.
It writes comparative figures, small report tables and
[`26_report_provenance.json`](output/26_report_provenance.json), including
the SHA256 of all seven input CSVs. It performs **no retraining or new Test
scoring**. It is not an independent reproduction of the fitted models.

## 07. FINAL cross-stage evidence audit and research closure

The independent [FINAL research notebook](code/notebooks/final_harth_research_audit_and_closure.ipynb)
was executed and committed after the Stage 26 saved-results report. All
**11 code cells were executed**, with **no saved runtime errors**, and its
last cell records:

`FINAL HARTH | CROSS-STAGE EVIDENCE AUDIT: PASSED`

**Audit design.** FINAL reads **20 committed CSV inputs** drawn from
Stages 08, 09, 10, 13–17, 19–20, 22–23, 25 and 26. The evidence
registry records their source filenames, column counts and row counts.
The notebook verifies participant and source-window accounting,
protocol comparability, rare-class sampling, temperature-scaling
results and the selected-model/qualified-evaluation summaries.

**Reconciled Development cohorts:**

| Cohort | Independent participants | Recorded task-eligible or selected windows |
| --- | ---: | ---: |
| Full Development | 29 | **132,397** |
| Up to 500 windows per participant | 29 | **14,201** |
| Nested up to 1,000 windows per participant | 29 | **28,201** |

The cohort audit confirms participant alignment and exclusion of S023/S032
from Development evidence. This does not make the two held-out participants
blind external cases.

**Minority sampling support.** The complete Development source includes
**1,946 Shuffling**, **1,332 ascending-stairs** and **1,066
descending-stairs** task windows. At the 500-window sampling cap, those
counts become **219, 192 and 182**, respectively; at the 1,000-window
cap they become **386, 375 and 348**. These percentages describe how
many windows the selection retains, **not classifier sensitivity**.
See the [sampling CSV](output/final_minority_sampling_audit.csv) and
[retention figure](output/final_minority_class_retention.png).

**Protocol comparability.** The audit preserves model results under
their original cohort, group-fold and training-budget definitions.
It explicitly distinguishes full-cohort classical CV, capped neural
comparisons, nested training-scale experiments, calibration folds and
qualified held-out evaluation. It does not calculate a new overall
leaderboard from these non-equivalent protocols. See
[earlier Development comparisons](output/final_earlier_development_comparisons.csv)
and the [protocol comparison register](output/final_protocol_comparability.csv).

**Probability calibration.** FINAL verifies the Stage 19 source metrics:
ten-bin ECE **0.0488 → 0.0286**, multiclass Brier
**0.1082 → 0.1043**, NLL **0.2487 → 0.2573**, with
accuracy **0.9275** in both conditions. Calibration is a
Development-only descriptive experiment and was not incorporated
in the frozen selected model.
See [the calibration audit](output/final_temperature_scaling_audit.csv).

**Scientific provenance.** FINAL recomputes the byte-level SHA256 for
all **20 named input CSVs** and compares the seven sources shared with
Stage 26 against the recorded Stage 26 input digests. It also checks
the frozen proposal/approval records and the Stage 26 report file.
The committed [FINAL provenance JSON](output/final_research_audit_provenance.json)
records these fingerprints and the qualified evaluation boundary.
The final result snapshot is sourced from the recorded
[Stage 25 evaluation](output/25_qualified_model_summary.csv);
it does not reproduce or rescore any Test predictions.

**Committed FINAL outputs:** 8 CSV audit tables, 1 PNG figure and
1 JSON provenance record are available in
[`output/`](output/). The
[research closure register](output/final_research_closure_and_limitations.csv)
separates completed work from scientific limitations, future
validation needs and claims the study does **not** establish.

The Quality Gate demonstrates consistency of the available recorded
results, source-file fingerprints and protocol constraints.
It is **not** independent training replication, a new external
evaluation or evidence of clinical readiness.

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

- [Notebooks 01–26](code/notebooks/) contain experimental methods and
  saved execution evidence. The [Stage 01–19 evidence register](evidence/stage_01_19_evidence_register.csv)
  provides cross-stage context.
- [Stage 20](code/notebooks/20_minority_class_and_data_quality.ipynb) contains
  the full source-data audit; [Stage 22](code/notebooks/22_unified_matched_model_benchmark.ipynb)
  and [Stage 23](code/notebooks/23_multiseed_and_imbalance_robustness.ipynb)
  provide the main matched model comparison and repeated-seed results.
- [Stage 24](code/notebooks/24_model_selection_and_protocol_freeze.ipynb)
  documents the frozen selection;
  [Stage 25](code/notebooks/25_final_development_training_and_inference.ipynb)
  contains qualified held-out results; and
  [Stage 26](code/notebooks/26_final_research_evidence_and_report.ipynb)
  consolidates saved evidence.
- [FINAL cross-stage audit](code/notebooks/final_harth_research_audit_and_closure.ipynb)
  contains executed cross-stage checks and the PASSED Quality Gate.
  Its [source inventory](output/final_source_inventory.csv),
  [research closure register](output/final_research_closure_and_limitations.csv)
  and [provenance JSON](output/final_research_audit_provenance.json)
  are committed alongside the other `final_*` outputs.
- [Protocol proposal](evidence/stage_25_protocol_proposal.json),
  [reconciled proposal](evidence/stage_25_protocol_proposal_reconciled.json)
  and [approval](evidence/stage_25_protocol_approved_qualified_test.json)
  retain the recorded research decisions.
- [Shared Python research modules](../python_modules/core/),
  [HARTH-specific code](code/src/), [offline tests](../tests/),
  [workflow figures](images/) and the
  [reproducibility guide](../docs/HARTH_REPRODUCIBILITY.md) document
  implementation and verification requirements.

**Research status:** All 26 numbered notebooks and the separate FINAL
cross-stage evidence audit have recorded executed results in GitHub.
The FINAL audit passes its source-file and scientific-consistency
Quality Gate, with ten committed output artefacts. This completes the
HARTH research portfolio documentation and evidence audit within
the available datasets and evaluation boundaries. It does **not**
establish independent replication, clinical validation or
generalisation to new participant populations.
