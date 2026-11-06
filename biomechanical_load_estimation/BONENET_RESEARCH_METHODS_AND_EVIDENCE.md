# BoneNet | Research Methods and Experimental Evidence

**Study:** Biomechanical Load Estimation, Human Movement Intelligence.

**Scope:** Verified original Essex IMU–finite-element (FE) record recovery; participant-disjoint development and one-time same-source Test evaluation; classical and deep temporal regression; peak-strain and source-grid error analysis; executed source and model evidence. This report covers [Stages 01–08](README.md#2-experimental-programme) and is the companion to the [project README](README.md) and [reproducibility guide](../docs/BONENET_REPRODUCIBILITY.md).

![Study-wide Essex IMU–FE research workflow](images/study_overview.svg)

**Execution status:** The numerical findings in this report describe the previous completed study run. The simplified notebook cells have not been rerun on the full dataset. Current CSV results and figures are retained; each notebook replaces its own outputs on execution. The separate 444-row derivative pilot is not merged into the original-source results.

## 01. Research objective, source and FE target definitions

The central research question is whether lower-body wearable inertial signals can predict **model-derived femoral-neck principal-strain trajectories**, and whether temporal neural models improve average waveform, peak-value or peak-position error compared with classical regression when participants are separated.

The study uses original MATLAB records from the published [Essex Figshare dataset v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3).

The source inventory contained **40 original participants** and **1,897 trials**. Source checks identified **168 trials with empty original FE arrays**, leaving **1,729 eligible paired trials from 37 participants**. This is a governed source-defined inclusion criterion, not an additional machine-learning target imputation.

| Component | Governed original-source contract |
| --- | --- |
| IMU placement | Seven lower-body inertial sensors, six source channels each |
| IMU tensor | `(N, 101, 42)`, original source time positions |
| Predicted FE tensor | `(N, 11, 2)`, original source positions and targets `e1` and `e3` |
| FE semantics | First and third principal finite-element strain model outputs |
| Published FE target unit | **Unitless**; no conversion to microstrain |
| Source alignment | Original FE-to-IMU indices, verified without target interpolation |
| Prediction unit | A source trial; correlated trials from the same person remain grouped |
| Activities in final Test | BilateralHopping, CounterJumps, Running, SquatJumps, UnilateralHopping and Walking |

These source FE predictions are **simulation-derived numerical targets**, not direct in-vivo bone-strain measurements, diagnoses, clinical decisions or claims of injury prevention.

## 02. Native-time reconstruction, cohort and evaluation units

The original nested MATLAB objects were decoded and numerically reconstructed without treating interpolated derivatives as original evidence. Original trial keys and provenance were retained; the time-coordinate audit established which eleven positions in the 101-sample IMU grid correspond to the original FE measurements. The temporal models read network features **at those exact positions**. No synthetic 101-point FE ground truth is used.

**Immutable split following source recovery (Stage 04):**

| Partition | Independent participants | Original eligible trials | Authorised role |
| --- | ---: | ---: | --- |
| Training | 29 | 1,354 | Grouped model selection and all Stage 07 robustness jobs |
| Diagnostic Validation | 4 | 192 | Predefined diagnostic comparison in Stages 05–06, not new Stage 07 selection |
| Reserved Test | 4 | 183 | One independently authorised final evaluation after model freezing |
| **Total** | **37** | **1,729** | All source-eligible observations |
| Final Development (Training + Validation) | 33 | 1,546 | One-time final refitting, after selecting the primary model |

The initial Training model comparisons reuse **five participant-disjoint GroupKFold assignments**. A held-out participant contributes *all* of their trials to the same fold. There is no between-participant leakage across Train/Validation/Test, and scaling is fitted only on the corresponding training-fold inputs.

The main scoring unit is the **participant**, not the individual FE position. In the primary **subject-balanced waveform MAE**, model error is averaged over both FE targets, eleven FE positions and trials *within each participant*, then the participants receive equal weight. Target-specific pointwise MAE/RMSE, peak **value**, peak **index**, activity-conditioned error and between-participant dispersion are separate endpoints.

The 183 final Test trials therefore represent **four independent human clusters**, not 183 independent people or 4,026 independent physiological measurements.

## 03. Original data audit and classical regression (Stages 01–05)

**Stages 01–03 — source understanding and numerical recovery.** The project inspected the official metadata, original archive inventory, 40-person MATLAB object layout, stable record identities, numerical nested fields and temporal correspondence. The presence and eligibility of each FE series were checked rather than silently filling missing labels. The resulting current trial registry and time-coordinate tables provide the inputs for pairing and modelling.

**Stage 04 — governed paired cohort.** All 1,729 eligible trials were assigned to the fixed **29/4/4 participant split**; channel schema, native FE indices and participant assignments were saved. The local reconstruction cache is intentionally Git-ignored rather than committed.

**Stage 05 — classical regression.** The training-time-mean control, Ridge regression and Random Forest benchmark were compared using five per-channel original-IMU summary features: **mean, standard deviation, minimum, maximum and root mean square**. With 42 input channels this yields **210 predictors**. Each model returns 22 values (eleven FE positions × two original FE targets) per trial. Five Training-only participant-disjoint folds determined the model-selection ranking.

| Stage 05 classical model | Training OOF subject-balanced MAE | Diagnostic Validation subject-balanced MAE |
| --- | ---: | ---: |
| Training mean | 0.0012562 | 0.0012673 |
| Ridge | 0.0012156 | 0.0008758 |
| **Random Forest** | **0.0007950** | **0.0006771** |

The earlier Validation set was inspected during model development. It is therefore an informative **diagnostic** set, but not a prospectively blind, external evaluation. The Stage 05 Random Forest retained the lowest predeclared **Training** metric and ultimately became the frozen primary model; subsequent Test outcomes did not cause it to be replaced.

Sources: [Stage 04 participant assignments](output/04_subject_split_assignments.csv), [Stage 05 CV model summary](output/05_training_cv_model_summary.csv), [Stage 05 Validation results](output/05_validation_model_metrics.csv), and the corresponding notebook methods.

## 04. Temporal regression and peak objectives (Stage 06)

Stage 06 compared two compact neural models: a **Temporal CNN** and a residual **Dilated TCN**. Both consume the complete original 101×42 IMU sequence and use an exact 11-position FE index readout. Both used identical predefined training epochs (**35 per fold**), AdamW, training-fold-only standardisation and the same normalised waveform-plus-peak objective (peak weight **0.25**). The architectures do not necessarily have identical parameter counts.

| Stage 06 architecture | Training OOF subject-balanced MAE | Diagnostic Validation subject-balanced MAE |
| --- | ---: | ---: |
| Temporal CNN | 0.0009048 | 0.0007517 |
| **Dilated TCN** | **0.0008513** | **0.0007151** |
| Locked earlier Random Forest reference | **0.0007950** | **0.0006771** |

Random Forest remained lower in overall waveform MAE. Yet on the **four-person Validation** set, Dilated TCN had lower e1/e3 **peak-value** errors than Random Forest: **0.0005855 / 0.0014333** versus **0.0006625 / 0.0016423**. The neural model had *higher* e1/e3 **peak-position** errors: **1.615 / 1.495** versus **1.417 / 1.005** original FE indices.

This establishes a development-stage **trade-off**, not blanket deep-learning superiority. The source target peaks are computed on the eleven original FE coordinates; the integer index difference is not a physical time difference in seconds.

Sources: [Stage 06 Training CV](output/06_training_cv_model_summary.csv), [Stage 06 Validation results](output/06_validation_model_metrics.csv).

## 05. Multi-seed robustness, recurrent models and sensor sensitivity (Stage 07)

Stage 07 introduced **GRU** and **bidirectional GRU (BiGRU)** as genuinely different recurrent families alongside Temporal CNN and Dilated TCN. One additional Dilated TCN configuration removed the peak-loss penalty. Each of the **five configurations × three fixed seeds (2026/2027/2028) × five Training-fold assignments** ran for **35 epochs**, producing **75 actual completed neural training jobs** and **2,625 recorded epoch-loss rows**, all using Training only. The recorded GPU was Tesla T4.

| Stage 07 Training OOF model/configuration | Mean subject-balanced MAE | Research role |
| --- | ---: | --- |
| Dilated TCN **without** peak loss | 0.0008624 | Controlled ablation only; not eligible for primary selection |
| **Dilated TCN** | **0.0008769** | Best mean among four predefined principal neural architectures |
| BiGRU | 0.0008800 | Close recurrent comparison |
| GRU | 0.0008898 | Recurrent comparison |
| Temporal CNN | 0.0009152 | Convolutional comparison |
| Random Forest (Stage 05 locked) | **0.0007950** | Previously recorded primary classical comparator |

These neural values are means over three seed-specific, full Training OOF metrics; the classical reference comes from Stage 05's earlier training run. The comparison is **not** a claim of capacity-matched training or a statistically significant architecture ranking.

**Peak-loss ablation.** Removing the 0.25 peak loss lowered three-seed Dilated TCN mean waveform MAE from **0.0008769** to **0.0008624**. Peak-value and peak-index results did not establish a reliable improvement from *adding* the penalty; per-seed and target effects varied, and the exploratory participant bootstrap for the paired waveform difference included zero. The Stage 06 diagnostic Validation peak advantage therefore cannot be causally attributed to the chosen peak penalty.

**Participant-level sensitivity.** A single prespecified Dilated TCN seed (2026) underwent original six-channel IMU-group mean replacement and fixed normalised Gaussian noise tests on out-of-fold Training trials. Replacing the **right-thigh (RTH)** sensor group produced the largest average error increase, about **+0.0001060**. Fixed Gaussian noise (SD 0.10 in fold-standardised units) produced an approximately **+0.00000171** mean change. These are defined *model-response perturbations*, not physical sensor failures or causal biomechanical importance estimates.

**Selection.** A predeclared Training-only rule compared the principal architectures with the locked Random Forest CV score, excluding the no-peak ablation from primary selection. **Random Forest remained the primary Stage 08 candidate.** No new Stage 07 model ranking was performed using the previously inspected Validation or reserved Test sets.

Sources: [Stage 07 model stability](output/07_training_model_stability_summary.csv), [Training-only candidate decision](output/07_training_candidate_decision.csv), [peak-loss ablation](output/07_peak_loss_ablation.csv), [sensor/noise sensitivity](output/07_sensor_noise_sensitivity.csv).

## 06. Frozen model selection and one-time held-out Test (Stage 08)

**Model selection preceded Test.** The original run used a **Random Forest primary** and a **descriptive Dilated TCN secondary**. Their unchanged recipes are now defined directly in [`final_model_settings()`](code/src/essex_stage08_final.py): Random Forest uses 210 statistical predictors, 200 trees, minimum leaf size 3 and seed 2026; TCN uses native FE indexing, 35 epochs, AdamW, peak weight 0.25 and seed 2128.

**Final fitting.** Both models were fitted on **33 Development participants / 1,546 trials**, combining Training and diagnostic Validation after selection. The recorded final TCN has **51,986 parameters** and used Tesla T4 CUDA. Test labels are excluded from fitting and model selection.

**Held-out Test.** Four unseen participants contributed 183 trials and **8,052 prediction rows** (183 × eleven positions × two targets × two models). The current notebook can be rerun without release tokens; it refits the same models and replaces its outputs. Such reruns are not new independent evaluations.

![One-time original-grid Test waveform MAE: frozen primary versus descriptive secondary](output/08_test_waveform_mae_comparison.png)

| Final same-source held-out Test metric | **Primary Random Forest** | **Secondary Dilated TCN** |
| --- | ---: | ---: |
| **Subject-balanced waveform MAE** | 0.0009384 | **0.0008759** |
| e1 pointwise MAE | 0.0004926 | **0.0004611** |
| e3 pointwise MAE | 0.0013249 | **0.0012482** |
| e1 pointwise RMSE | 0.0007603 | **0.0007132** |
| e3 pointwise RMSE | 0.0020646 | **0.0019296** |
| e1 peak-value MAE | 0.0008059 | **0.0007222** |
| e3 peak-value MAE | 0.0022999 | **0.0020255** |
| e1 peak-index MAE, original FE grid | **0.9672** | 1.3497 |
| e3 peak-index MAE, original FE grid | **0.8415** | 1.1967 |

The predeclared secondary **Dilated TCN** had about **6.7% lower observed participant-balanced waveform MAE** on the final Test, and lower both-target peak-*value* MAE. The frozen primary **Random Forest** more accurately located the source-grid peaks. These are mixed outcomes and do **not** justify retrospective primary-model replacement.

The exploratory paired whole-participant bootstrap for **TCN − RF subject-balanced waveform MAE** was **−0.0000625**, with a 95% percentile interval of **[−0.0001859, +0.0000431]** over only four Test participants (2,000 draws). The interval includes zero. It is not confirmatory evidence of population-level model superiority.

![Final Test peak-value error by source FE target](output/08_test_peak_error_comparison.png)

Other genuine executed figures: [per-person errors](output/08_test_subject_errors.png), [one e1 waveform](output/08_test_waveform_example_e1.png) and [one e3 waveform](output/08_test_waveform_example_e3.png). The last two are **fixed, source-identifiable illustrative trials**, not population summaries.

## 07. Current outputs and reproducibility

The eight notebooks exchange current data tables, not execution-status reports. Notebook 04 supplies trial assignments and paired numerical arrays. Notebooks 05–07 use Training-only comparisons, with diagnostic Validation in 05–06. Notebook 08 uses the existing final model recipes and reports held-out Test errors.

The committed `08_test_trial_predictions.csv` contains **8,052 model/trial/position/target rows**, four Test participants and 183 trials. These saved values support descriptive error analysis without retraining. The simplification does not change their numerical contents. Full-data re-execution of the simplified notebooks is still required to establish runtime behaviour on the original dataset.

## 08. Source provenance, earlier audits and the derivative pilot

This section consolidates non-duplicative findings from the initial **13–14 October 2026** metadata, MATLAB-schema and exploratory derivative audits. These observations explain the source and the earlier derivative pilot; the current execution instructions are in the reproducibility guide.

**Participant workbook and identity governance.** The source `SubCharacteristics.xlsx` workbook contained 40 unique, non-contiguous participant IDs (including `Subj44`), plus summary rows that were **not** participant records. The workbook reported **19 M / 21 F**, whereas the downloaded source README described **20 / 20**; this disagreement was observed, not resolved by inventing records. The workbook's free-text health/pain field contained **35 ellipsis-style placeholders** and five explicit notes. A placeholder is *not* evidence of disease absence or a verified clinical label. Identifying notes were excluded from released research outputs. Source workbook date, limb dominance and metadata were audited as provenance; no clinical or demographic variables were silently introduced as IMU-only predictive inputs.

**Original source structure and exclusion accounting.** The archive inspection recorded **42 ZIP entries**, including **40 original participant MATLAB files**, and approximately **224.79 MiB** uncompressed. First-element schema inspection found 13 fields, including `SubjID`, `Task`, `SubTask`, `TrialNum`, `TrialNumUpdated`, `UpdatedTrialID`, `Cycle`, `IMU`, `ID`, `FP`, `e1e3`, `JCF` and `MuscleF`. SciPy sometimes exposed the MATLAB table/object container through opaque metadata such as `_TypeSystem` or `_ObjectMetadata`; their existence alone did not prove numerical FE recovery. A bounded Python reader pilot recovered **28 numeric IMU records**, but **zero FE tables** met that *pilot's strict* criterion; subsequent full-cohort native decoding established **1,729 numerically valid FE pairs**. These are different verification stages, not contradictory full-cohort results.

The original audit classified **168 source-empty FE records**: 144 across three fully FE-excluded participants (`subj16`, `subj29`, `subj32`; 48 each), plus 24 partial gaps (`subj06`: 9, `subj08`: 9, `subj11`: 6). Five participant files had fewer than 48 `Data` elements (`subj09`: 42; `subj11`: 40; `subj12`: 47; `subj14`: 42; `subj36`: 46). These are observed counts from the original run, not hard-coded requirements for notebook execution. See the retained [versioned data manifest](../docs/DATA_MANIFEST_ESSEX_V13.md) and [source access record](../docs/DATA_ACCESS_ESSEX.md).

**Separate public BoneNet derivative (exploratory, not the final study).** Earlier inspection of the authors' walking/running derivative recovered finite arrays `X=(444,42,101)` and `Y=(444,101,2)`, with 444 label rows from 37 listed subjects (333 Running, 111 Walking). The label columns `SubjID + Task + TrialNum` yielded only **222 distinct combinations**, so the association between tensor rows and labels was not independently verified. This is **not** the full six-activity, original-source, 11-FE-position study described above.

A single exploratory 27-subject/324-record Training and 10-subject/120-record held-out split (assuming the authors' ordering) compared a per-timepoint Training-mean baseline against scaled Ridge using 84 mean/SD features. The pooled derivative-scale results were:

| Historical derivative-only metric | Training mean | Ridge |
| --- | ---: | ---: |
| MAE across two targets and 101 positions | **1325.5737** | 1446.1677 |
| RMSE across two targets and 101 positions | **1851.0981** | 2191.9425 |

These numbers came from a recorded exploratory run, **not** an independently verified participant-mapped benchmark; their target scale was not verified. They must **never** be compared numerically with the source-verified, unitless FE errors of Stages 05–08. The associated [historical derivative workflow](images/bonenet_derivative_pilot.svg) is preserved only for context.

## 09. Research limitations and reproducibility

1. **One source, no external cohort.** The Test participants came from the original Essex collection. Participant independence does not imply independent-site or prospective distributional generalisation.
2. **Only four Test people.** Trial and FE-point counts cannot substitute for the small number of independent Test clusters. The reported bootstrap intervals are exploratory and its difference interval crosses zero.
3. **Model-derived FE ground truth.** The predicted targets are unitless, finite-element model outputs; true in-vivo strain, absolute physical-time peak delays and clinical validity have not been established.
4. **Cross-stage protocols differ.** Stage 05 Random Forest used a classical descriptor pipeline and an earlier fitting setting; Stage 07 neural experiments average three seeds and fixed epochs. The final Stage 08 models are *newly refitted* on 33 Development participants. Do not pool cross-stage metrics into a nominal single matched Test leaderboard.
5. **Peak metric trade-offs.** Waveform accuracy, peak magnitude and native-index peak location are distinct. The prescribed peak-loss penalty did not demonstrate a reliable benefit across Stage 07 Training OOF.
6. **Single-seed final refit.** Stage 07 covered three seeds, whereas Stage 08 each final model has one frozen recipe. Deterministic implementation options do not guarantee bitwise identical GPU results on all systems.
7. **Sensitivity is not causal explanation.** The seven-source sensor substitution and Gaussian perturbation results are artificial input-stress tests, not biomechanical attribution or patient-level sensor recommendations.
8. **Availability boundaries.** Public committed CSV/PNG results and code support retrospective auditing. Original archive, derived full numeric caches, local per-fold checkpoints and trained final weights are not published; independent end-to-end training replication requires original data and suitable dependencies.
9. **No clinical deployment.** There is no independent prospective cohort, measured physiological strain reference, patient-specific finite-element ground-truth assessment or regulated device validation.

**Conclusion:** The project is **complete as a source-verified exploratory IMU-to-FE research benchmark**, with a frozen-before-Test protocol and an authorised one-time participant-held-out assessment. It is **not** a validated clinical biomechanical load-estimation system.

**Related navigation:** [Research README](README.md) · [Reproducibility guide](../docs/BONENET_REPRODUCIBILITY.md) · [Model settings](code/src/essex_stage08_final.py) · [Test metrics](output/08_test_primary_secondary_metrics.csv) · [Study overview](images/study_overview.svg).
