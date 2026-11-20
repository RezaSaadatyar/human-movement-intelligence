# Multimodal Movement Learning | NinaPro DB2

**Research question:** Do paired accelerometer signals add value to surface electromyography (sEMG) for hand-movement recognition in previously unseen participants?

**Research evidence:** Notebooks 01–02 record source and cohort processing; Notebook 03 records the eleven unimodal Development comparisons. **Notebook 04 has now been executed** on the original 35-person Development cohort, with nine fusion methods, additional neural seeds, class-specific diagnostics and a frozen selection of `B1_rf` versus `F1_rf`. Notebook 05 remains an **unexecuted** independent five-person Test evaluation.

**Detailed scientific plan:** [Multimodal Research Methods and Evidence](MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Aspect | Source definition and current study evidence |
| --- | --- |
| Source | [NinaPro Database 2](https://ninapro.hevs.ch/instructions/DB2.html); [original paper](https://doi.org/10.1038/sdata.2014.53); [original Dryad deposition](https://doi.org/10.5061/dryad.1k84r) |
| Published cohort | 40 intact participants; six repetitions in the detailed protocol; 49 hand movements plus rest across three exercises |
| Input modalities | Verified paired E1 arrays: 12 Delsys sEMG channels and 36 acceleration columns (three axes from each of 12 modules) |
| Source timing | Released E1 arrays share a 2,000-Hz row grid; the original acceleration acquisition was approximately 148 Hz, so common stored rows do not imply 2,000 independent accelerometer observations per second |
| Primary task | Verified first non-force exercise (E1: finger movements) with corrected labels 0–17 (Rest plus seventeen movements) |
| Outcome | Released corrected movement class `restimulus` (0–17); `rerepetition` retained as non-predictive segment metadata |
| Independent evaluation unit | Participant; trials and overlapping windows are nested within participants |
| Recorded participant partition | 35 Development and five reserved Test people; Development is partitioned into five person-disjoint folds of seven participants each (see current `02_*_assignments.csv` outputs) |
| Primary measure | Fixed-class macro F1 per Test participant, averaged equally across participants |
| Main comparisons | Prior-class and classical unimodal baselines; deep sEMG/accelerometer models (1D-CNN, TCN and BiGRU); early and late fusion; two-branch CNN/TCN fusion and gated/cross-attention fusion |

**Input boundary:** Use only sEMG and accelerometer measurements for prediction. Keep subject IDs, exercise/repetition IDs and labels for grouping, alignment and reporting, **not as predictive features**. Exclude dataglove, force, displayed cue signals and subject demographics from model inputs.

The official DB2 webpage contains inconsistent wording about repetitions. The inspected S01 E1 source uses corrected repetition codes 0–6, with six non-rest movement segments per class. Notebook 02 inspected the full 40-participant E1 source cohort and found **337,853 eligible windows** before selection; the reproducible experimental subset contains **40,000 paired windows** (1,000 per participant, 300-ms duration, 100-ms step). **Development fusion comparisons are recorded; no reserved Test results have yet been established.**

## 2. Experimental Programme

The five notebooks follow the fixed research sequence. Notebooks 01–04 contain saved source, paired-window, unimodal and fusion Development evidence. Notebook 04's nine fusion arms have complete selected-model outputs and recorded execution. Notebook 05 is prepared for the first independent Test comparison but has not been executed.

| Notebook | Scientific purpose | Main named outputs to produce |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Inspect a real subject ZIP, exercise MATLAB variables, synchronised sEMG/accelerometer shape, corrected labels, repetitions and missing data | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Define eligible non-force classes, reconstruct source-timed paired windows, inspect signal quality and generate participant-preserving feature arrays | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local processed features |
| `03_unimodal_grouped_baselines.ipynb` | Compare the training-prior, Logistic Regression, Random Forest and six unimodal CNN/TCN/BiGRU arms on grouped Development folds; analyse class-specific OOF errors and seed robustness | `03_all_unimodal_metrics.csv`, `03_oof_probabilities.npz`, `03_selected_model_per_class_metrics.csv`, `03_all_models_per_class_metrics.csv`, `03_class_errors.png`, `03_class_errors_normalised.png`, `03_per_class_f1_heatmap.png` |
| `04_fusion_and_modality_ablations.ipynb` | Evaluate separately executable classical early and late fusion, dual-branch CNN/TCN, gated and compact attention models on the same grouped Development folds; compare synthetic acceleration perturbations | `04_dev_fusion_metrics.csv`, `04_dev_deep_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_robustness.csv`, `04_model_selection.csv`, `04_fusion_comparison.png`, `04_subject_paired_differences.png`, `04_modality_sensitivity.png`, `04_all_fusion_per_class_metrics.csv`, `04_selected_fusion_per_class_metrics.csv`, `04_fusion_class_errors_normalised.png`, `04_fusion_per_class_f1_heatmap.png` |
| `05_frozen_person_independent_test.ipynb` | Refit the frozen `B1_rf` sEMG and `F1_rf` early-fusion Random Forest recipes using all 35 Development people; compare five held-out Test people without Test-based model choice | Planned: `05_test_predictions.csv`, `05_test_metrics.csv`, `05_per_class_support.csv` (including F1), `05_confusion_matrix_normalised.png`, `05_subject_differences.png`, `05_cost_accuracy.png` |

All five notebooks use numbered Markdown/code sections, a Colab checkout, defined input/output interfaces and SVG diagrams named after the corresponding notebooks. The [source audit notebook](code/notebooks/01_ninapro_pairing_and_label_audit.ipynb) and [paired-window notebook](code/notebooks/02_signal_processing_and_aligned_windows.ipynb) contain saved source-based displays and refer to their current figures in `images/` and `output/`.

## 3. Main Results

**Recorded source and cohort evidence:** Notebook 01 inspected 1,808,331 common E1 rows for S01. Notebook 02 processed 40 participants and 337,853 eligible paired windows, retaining 40,000 windows (1,000 per participant), with 35 Development and five reserved Test people. The statistical feature representations contain 48 sEMG and 96 acceleration features per window.

**Recorded Notebook 03 Development comparison:** Fixed-class macro F1 was calculated for each of the 35 unseen Development validation participants and averaged equally across people, using five person-disjoint folds and the original 18 corrected E1 classes.

| Unimodal model family | sEMG macro F1 | Acceleration macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.2084 | 0.1034 |
| Random Forest | **0.2348** | **0.1290** |
| 1D-CNN | 0.2182 | 0.0675 |
| TCN | 0.2128 | 0.0709 |
| BiGRU | 0.1874 | 0.0655 |

The no-signal Training-prior baseline achieved **0.0350**. The Development-selected unimodal reference is the sEMG Random Forest (`B1_rf`, macro F1 0.2348). Additional seeds 43/44 gave 0.2135/0.2161 for the selected sEMG CNN and 0.0704/0.0748 for the selected acceleration TCN. These values originate from the saved, executed notebook rather than a new reproduction in this review.

**Interpretation:** sEMG-only models performed better than their acceleration-only counterparts, and the classical sEMG Random Forest exceeded all six temporal candidates on Development people. Rest-class support and the interpolated acceleration grid require careful class-level interpretation. Notebook 03 now includes executed per-class precision, recall, F1, support, a true-class-normalised confusion matrix and an 11-model F1 heatmap. In its selected sEMG Random Forest, Rest (class 0) accounts for 16,329 of 35,000 Development windows (46.7%); pooled recall is 0.9023 for Rest but 0.0221 for class 8. These class-specific findings qualify any interpretation based on pooled accuracy.

**Recorded Notebook 04 fusion evidence:** On the same 35 Development participants, early-fusion Random Forest (`F1_rf`) achieved participant-mean fixed-class macro F1 **0.2630**, versus **0.2348** for the selected sEMG-only comparator `B1_rf` (paired difference **+0.0282**). Fusion improved the score for **25 participants** and reduced it for **10**. Fixed-weight late Random Forest fusion (`F2_rf`) scored **0.2462**; the strongest neural fusion (`F3_dual_tcn`) scored **0.1546**. `F1_rf` was selected using Development evidence only. These results do not establish an independent Test benefit, clinical usefulness or cross-device generalisation.
## 4. Running and Reviewing the Study

**Available evidence:** Notebooks 01–04 have saved executions and matching Development CSV/PNG outputs. The original Notebook 03 `03_oof_probabilities.npz` is now tracked in the study `output/` directory and is validated by Notebook 04 against its keyed predictions. Other large NPZ inputs and temporal signal caches remain untracked. Notebook 05's frozen classical comparators need only the two local Notebook 02 statistical feature arrays; no independent Test results are currently reported.

**Reproducing the workflow:**
1. Use the root `requirements.txt` with a supported Python/Colab environment. Notebook 01 reads one official source ZIP; Notebook 02 incrementally reads E1 recordings across the 40-person source cohort.
2. Original source archives are cached locally under `data/raw/ninapro_db2/`. Paired source arrays and statistical feature matrices are cached locally under `data/processed/ninapro_db2/`. Later notebooks reuse these files when available; a fresh ephemeral runtime may need to reacquire/reconstruct missing numerical inputs.
3. Inspect the source audit and paired-window results in `output/`, especially `01_label_dictionary.csv`, `02_window_index.csv`, `02_participant_split_assignments.csv` and `02_development_fold_assignments.csv`.
4. Notebook 04 reads the tracked `output/03_oof_probabilities.npz` and validates the 11 original OOF model scores; manual restoration of Notebook 03's ZIP is no longer required for that file. For the current **Notebook 05 classical Test**, only `data/processed/ninapro_db2/02_emg_features.npy` and `02_acc_features.npy` are required. If absent locally and the intended Google Drive is already mounted, the notebook can restore these two files from `/content/drive/MyDrive/HMI/ninapro_db2/`; otherwise it raises an explicit missing-file error. Missing features are **not** silently replaced by a different window sample or model.
5. Source ZIPs and dense numerical signals are not interchangeable with output CSVs. The original 40,000-window sequence representation would be several GiB if materialised as dense 32-bit arrays. Keep large caches in a persistent data location if repeating executions across fresh hosted sessions.

**Numerical cache and persistence:** The current canonical locations are `data/raw/ninapro_db2/` (one original ZIP per subject) and `data/processed/ninapro_db2/` (per-subject E1 sEMG/acceleration `.npy` arrays and `02_emg_features.npy` / `02_acc_features.npy`). They are already separated by study, so a second `data/` directory beneath this study would duplicate the established file contract. With 40,000 windows represented as float32, the compact statistical matrices alone are approximately **7.32 MiB** (40,000 × 48 sEMG values) and **14.65 MiB** (40,000 × 96 acceleration values); the source-length S01 arrays are substantially larger, approximately **82.8 MiB sEMG** and **248.3 MiB acceleration**. These are estimated uncompressed numerical-array sizes, not measured archive or compressed-file sizes. They are saved automatically by Notebook 02 and reused when present. In Colab, the working directory may be discarded when the runtime resets; for reuse across sessions, back the same paths with suitable persistent storage (for example, a mounted Drive directory), without adding large source archives or redundant dense tensors to Git. The `output/` folder contains the smaller scientific tables and plots needed to identify source windows and participant roles, **not** the underlying signal tensors.

**Neural experiment budget:** Notebook 03 used seed 42, five participant-disjoint Development folds, up to ten epochs with patience three, and additional seeds 43 and 44 for two selected temporal finalists. Acceleration-only models prepare float32, Training-normalised windows in GPU memory when sufficient VRAM is available. This is a throughput improvement, not a reduced cohort or an altered evaluation protocol.

**Notebook 04 compute and output contracts:** Fixed-class, per-person class-level precision/recall/F1 and true-class-normalised error figures are also computed from matched Development OOF probabilities without retraining. Each of its four neural fusion models uses the same 10-epoch maximum, patience three and batch size 96 with five person-disjoint Development folds. Normalised sEMG and acceleration windows can be staged together in GPU memory when a conservative capacity check permits, with a streamed fallback. Classical and neural fusion arms save model-specific `04_*` OOF probabilities, keyed predictions and participant metrics after each completed fold. The overall fusion selection is only computed when all nine fusion methods have complete outputs. Synthetic sensor replacement is not a measured real-world sensor-failure result.

Loading all 40 source archives simultaneously is not required. The Dryad listing shows that a typical DB2 subject ZIP is hundreds of megabytes; pilot and full-cohort ingestion should process subjects incrementally. The S01 pilot archive and the 40-person E1 cohort have been processed in the executed notebooks. **Notebooks 03 and 04 record completed original-data Development execution; Notebook 05's reserved Test evaluation has not yet been executed.**

## 5. Files and Saved Outputs

| Location | Purpose and current availability |
| --- | --- |
| `README.md` | Study overview with recorded Notebook 03 and Notebook 04 Development comparisons and remaining Test evaluation |
| `MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md` | Research design, executed Development evidence, selected configuration, limitations and Test protocol |
| `code/notebooks/` | Five research notebooks; Notebooks 01–04 retain executed outputs and Notebook 05 remains unexecuted |
| `code/src/` | Source reader, paired data, neural architectures, trained fusion orchestration and evaluation helpers |
| `images/` | Five editable notebook-specific scientific workflow SVG diagrams |
| `output/` | Tracked `01_*`–`04_*` source, Development and fusion CSV/PNG evidence, selection table and `03_oof_probabilities.npz`; Notebook 04 model-specific NPZ files remain in its separate Colab archive |
| `data/raw/ninapro_db2/` | Original participant ZIPs acquired during execution; local runtime cache, not distributed with the repository |
| `data/processed/ninapro_db2/` | Reusable local E1 source arrays and 48/96-feature matrices; reconstructed when absent |
| [Shared core](../python_modules/core/) | Existing candidate splitting, grouped evaluation, plotting and setup utilities; reuse must be checked against DB2 |

The repository retains executed Notebook 01–04 evidence, the Notebook 03 aggregate OOF probability NPZ, Notebook 04's selected Development recipes and scientific diagrams. It does not distribute large source recordings, dense temporal caches or the Notebook 04 model-specific NPZ archive. **No frozen-model Test scores have been measured or published.**

## 6. Interpretation and Scope

This study is designed to measure **whether acceleration adds information to sEMG in a specific person-independent, prompted hand-movement classification task**, using classical models alongside compact deep temporal and multimodal fusion architectures. The enrolled NinaPro DB2 population consists of intact volunteers, so future results would not alone establish amputee performance, prosthesis control, clinical usefulness, online operation or generalisation across acquisition devices.

The released signals were already processed for synchronisation and sEMG interference removal. Any additional filtering must be justified against those original processing steps. The corrected labels may have been aligned from sEMG, so this provenance must be reported when comparing modalities. Modality row correspondence, corrected E1 label codes, source eligibility and participant-disjoint assignments have been inspected and recorded for the present cohort. Equal released sample rows do not independently establish native sensor synchronisation accuracy; the frozen final-comparator scores require the planned independent Notebook 05 Test execution.

**Primary sources:** [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html); Atzori et al., *Scientific Data* (2014), [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53); [Dryad data record](https://doi.org/10.5061/dryad.1k84r). See the scientific document for related fusion literature and planned evaluation.
